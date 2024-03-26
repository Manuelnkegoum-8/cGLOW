import os,glob
import torch
import numpy as np
from PIL import Image
from tqdm import trange
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import skimage.color
import skimage.util
import skimage.io

def my_preprocess(image,training=True):
    image = image - 0.5
    if training:
        image = image + torch.zeros_like(image).uniform_(-0.5,0.5)
    return image

    
def my_postprocess(x):
    """Map [-0.5, 0.5] quantized images to uint space"""
    transform = transforms.ToTensor()
    num_bins = 2 ** 8
    x = torch.floor((x + 0.5) * num_bins)
    x = torch.clip(x, 0, 255)/255.
    return x
    


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    cross for alternative pattern
    split for chunk
    """
    C = tensor.size(1)
    if type == "split":
        return torch.chunk(tensor,2,1)
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def save_model(model, optim, scheduler, dir, iteration):
    path = os.path.join(dir, "cglow.pth.tar")
    state = {}
    state["iteration"] = iteration
    state["modelname"] = model.__class__.__name__
    state["model_state_dict"] = model.state_dict()
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




def sample_based(model,x,n_samples):
    sample_list = list()
    for i in range(0, n_samples):
        y_sample,_ = model(x + torch.zeros_like(x).uniform_(0,1/256.), reverse=True)
        sample_list.append(y_sample)
        sample = torch.stack(sample_list)
        sample = torch.mean(sample, dim=0, keepdim=False)
    return sample

@torch.no_grad()
def prediction(model,device,loader,n_samples):
    all_samples = list()
    all_masks = list()
    all_images = list()
    for i_batch, data in enumerate(loader):
            x, y = data['x'],data['y']
            x = x.to(device) # true img
            y = y.to(device) # mask
            y = my_preprocess(y,False)
            sample = sample_based(model,x,n_samples)

            all_samples.append(sample)
            all_masks.append(y)
            all_images.append(x)

    all_samples = torch.cat(all_samples,dim=0)
    all_masks = torch.cat(all_masks,dim=0)
    all_images = torch.cat(all_images,dim=0)

    all_samples = my_postprocess(all_samples)
    all_masks = my_postprocess(all_masks)
    #iou = compute_iou(pred_seg,true_seg)
    iou = .05
    # save trues and preds
    output = None
    for i in range(len(all_samples)):
        true_img = all_images[i].detach().cpu()
        pred_mask = all_samples[i].detach().cpu()
        true_mask = all_masks[i].detach().cpu()
        row = torch.cat((true_img, true_mask, pred_mask), dim=1)
        if output is None:
            output = row
        else:
            output = torch.cat((output,row), dim=2)
    return output,iou

class Trainer(object):
    def __init__(self,model,train_loader,val_loader,optimizer,scheduler,args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optim = optimizer
        self.device = args.device
        self.y_bits = args.y_bits
        self.out_root = args.out_root
        self.scheduler = scheduler
        self.writer = SummaryWriter()

        if not os.path.exists(self.out_root):
            os.makedirs(self.out_root)

        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

        self.fp_in = os.path.join(self.out_root,"step_*.png")
        self.fp_out = "sample_evolution.gif"
        self.grad_clip = args.grad_clip
        self.grad_norm = args.grad_norm
        self.checkpoint = args.checkpoint

        

    def train(self,n_epochs):
        for epoch in trange(n_epochs,ncols=80):
            avg_loss = 0.
            n = 0.
            self.model.train()
            for data in self.train_loader:
                x, y = data['x'],data['y']
                x = x.to(self.device)
                y = y.to(self.device)
                y = my_preprocess(y)
                # forward
                z, nll = self.model(x + torch.zeros_like(x).uniform_(0,1/256.), y)
                # loss
                loss = torch.mean(nll)
                avg_loss = avg_loss + loss.item()*x.size(0)
                # backward
                self.optim.zero_grad()
                loss.backward()
                if self.grad_clip >0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(),self.grad_clip)
                if self.grad_norm >0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.grad_norm)
                self.optim.step()
                n+= x.size(0)
            avg_loss = avg_loss / n

            val_loss = self.validate()
            self.writer.add_scalar('Loss/train', avg_loss, global_step=epoch)
            self.writer.add_scalar('Loss/val', val_loss, global_step=epoch)

            if self.scheduler is not None:
                scheduler.step()

            if epoch%self.checkpoint==0:
                print("[INFO] Checkpoint")
                save_model(self.model, self.optim, self.scheduler,'checkpoints', epoch)
                output,iou = prediction(self.model,self.device,self.val_loader,10)
                save_image(output, os.path.join(self.out_root, "step_{}.png".format(epoch)))
                self.writer.add_scalar('Iou', iou, global_step=epoch)
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(self.fp_in))]
        img.save(fp=self.fp_out, format='GIF', append_images=imgs,save_all=True, duration=200, loop=0)


    @torch.no_grad()
    def validate(self):
        self.model.eval()
        avg_loss = 0.
        n = 0.
        for data in self.val_loader:
                x, y = data['x'],data['y']
                x = x.to(self.device)
                y = y.to(self.device)
                y = my_preprocess(y)
                # forward
                z, nll = self.model(x + torch.zeros_like(x).uniform_(0,1/256.), y)
                # loss
                loss = torch.mean(nll)
                avg_loss = avg_loss + loss.item()*x.size(0)
                n+= x.size(0)
        avg_loss = avg_loss / n
        return avg_loss

    
        


class Inference(object):

    def __init__(self, model, dataloader, args):

        # set path and date
        self.out_root = args.out_root
        if not os.path.exists(self.out_root):
            os.makedirs(self.out_root)
        # model
        self.model = model
        self.num_labels = args.num_labels
        self.device = args.device
        self.dataloader = dataloader
        self.model.eval()

    @torch.no_grad()
    def sampled_based_prediction(self, n_samples):
        output,iou = prediction(self.model,self.device,self.dataloader,n_samples)
        save_image(output, os.path.join(self.out_root, "trues.png"))
        return iou