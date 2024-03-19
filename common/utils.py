import os
import torch
import numpy as np
from tqdm import trange
from .dataset import *
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


def my_preprocess(image,num_bits=2,training=True):
    # Discretize to the given number of bits
    shape = image.size()
    #image = image*255.
    if num_bits < 8:
        image = torch.floor(image / 2 ** (8 - num_bits))
    num_bins = 2 ** num_bits
    image = image / num_bins - 0.5
    if training:
        image = image + torch.rand(shape,device=image.device)/num_bins
    return image


def my_postprocess(x, num_bits):
    """Map [-0.5, 0.5] quantized images to uint space"""
    num_bins = 2 ** num_bits
    x = torch.floor((x + 0.5) * num_bins)
    x *= 256. / num_bins
    return torch.clip(x, 0, 255).to(torch.uint8)


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return torch.chunk(tensor,2,1)
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def save_model(model, optim, scheduler, dir, iteration):
    path = os.path.join(dir, "checkpoint_{}.pth.tar".format(iteration))
    state = {}
    state["iteration"] = iteration
    state["modelname"] = model.__class__.__name__
    state["model"] = model.state_dict()
    state["optim"] = optim.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    else:
        state["scheduler"] = None

    torch.save(state, path)


def load_state(path, cuda):
    if cuda:
        print ("load to gpu")
        state = torch.load(path)
    else:
        print ("load to cpu")
        state = torch.load(path, map_location=lambda storage, loc: storage)

    return state

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_iou(masks1, masks2):
    # Calculate IoU for each pair of masks in the batch
    ious = []
    for i in range(masks1.shape[0]):
        intersection = (masks1[i] & masks2[i]).sum()
        union = (masks1[i] | masks2[i]).sum()
        iou = (intersection + 1e-10) / (union + 1e-10)
        ious.append(iou.item())
    # Average IoU across all pairs of masks in the batch
    return sum(ious) / len(ious)

class Trainer(object):
    def __init__(self,model,trainloader,valloader,optimizer,args):
        self.model = model
        self.train_loader = trainloader
        self.val_loader = valloader
        self.optim = optimizer
        self.device = args.device
        self.y_bins = args.y_bins

        self.writer = SummaryWriter()
    def train(self,n_epochs):
        for epoch in trange(n_epochs):
            avg_loss = 0.
            n = 0.
            for data in self.train_loader:
                x, y = data['x'],data['y']
                x = x.to(self.device)
                y = y.to(self.device)

                y = my_preprocess(y,self.y_bins,training=True)
                # forward
                z, nll = self.model(x, y)
                # loss
                loss = torch.mean(nll)
                avg_loss = avg_loss + loss.item()*x.size(0)
                # backward
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                n+= x.size(0)
            avg_loss = avg_loss / n

            val_loss = self.validate()
            iou = self.sampled_based_prediction(10)
            self.writer.add_scalar('Loss/train', avg_loss, global_step=epoch)
            self.writer.add_scalar('Loss/val', val_loss, global_step=epoch)
            self.writer.add_scalar('Iou', iou, global_step=epoch)
            if epoch%10==0:
                torch.save(self.model.state_dict(), 'my_model_2.pth')
            #print("train loss := {} , val_loss:= {}".format(avg_loss,val_loss))
        self.writer.close()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        avg_loss = 0.
        n = 0.
        for data in self.val_loader:
                x, y = data['x'],data['y']
                x = x.to(self.device)
                y = y.to(self.device)
                y = my_preprocess(y,self.y_bins,training=False)
                # forward
                z, nll = self.model(x, y)
                # loss
                loss = torch.mean(nll)
                avg_loss = avg_loss + loss.item()*x.size(0)
                n+= x.size(0)
        avg_loss = avg_loss / n
        return avg_loss

    @torch.no_grad()
    def sampled_based_prediction(self, n_samples):
        self.model.eval()
        avg_iou = 0.
        n = 0
        for i_batch, data in enumerate(self.val_loader):     
            x, y = data['x'],data['y']
            x = x.to(self.device) # true img
            y = y.to(self.device) # mask
            sample_list = list()
            nll_list = list()
            for i in range(0, n_samples):
                y_sample,_ = self.model(x, reverse=True)
                sample_list.append(y_sample)
            sample = torch.stack(sample_list)
            sample = torch.mean(sample, dim=0, keepdim=False)
            sample = my_postprocess(sample,self.y_bins)
            iou = compute_iou(y[:,0,:,:], sample[:,0,:,:])
            avg_iou+= iou*x.size(0)
            n+= x.size(0)
        return avg_iou/n

class Inference(object):

    def __init__(self, model, dataloader, args):

        # set path and date
        self.out_root = args.out_root
        if not os.path.exists(self.out_root):
            os.makedirs(self.out_root)
        # model
        self.model = model
        self.y_bins = args.y_bins
        self.num_labels = args.num_labels
        self.device = args.device
        self.dataloader = dataloader
        self.model.eval()

    @torch.no_grad()
    def sampled_based_prediction(self, n_samples):
        metrics = []
        avg_iou = 0.
        n = 0
        for i_batch, data in enumerate(self.dataloader):
            
            x, y = data['x'],data['y']
            x = x.to(self.device) # true img
            y = y.to(self.device) # mask

            sample_list = list()
            nll_list = list()
            for i in range(0, n_samples):
                y_sample,_ = self.model(x, reverse=True)
                sample_list.append(y_sample)

            sample = torch.stack(sample_list)
            sample = torch.mean(sample, dim=0, keepdim=False)
            sample = my_postprocess(sample,self.y_bins)


            iou = compute_iou(y[:,0,:,:], sample[:,0,:,:])
            avg_iou+= iou*x.size(0)
            n+= x.size(0)
            # save trues and preds
            output = None
            for i in range(sample.size(0)):
                true_img = y[i].detach().cpu()
                pred_img = sample[i].detach().cpu()
                rgb_img = x[i].cpu()
                row = torch.cat((rgb_img, true_img/true_img.max(), pred_img/pred_img.max()), dim=1)
                if output is None:
                    output = row
                else:
                    output = torch.cat((output,row), dim=2)
            save_image(output, os.path.join(self.out_root, "trues-{}.png".format(i_batch)))
        return avg_iou/n