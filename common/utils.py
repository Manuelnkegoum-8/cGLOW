import os
import torch
import numpy as np
from tqdm import trange
from .dataset import *
from torchvision.utils import save_image


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
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


class Trainer(object):
    def __init__(self,model,trainloader,valloader,optimizer,args):
        self.model = model
        self.train_loader = trainloader
        self.val_loader = valloader
        self.optim = optimizer
        self.device = args.device
        self.y_bins = args.y_bins


    def train(self,n_epochs):
        for i in trange(n_epochs):
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
            print("train loss := {} , val_loss:= {}".format(avg_loss,val_loss))

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

            # save trues and preds
            output = None
            for i in range(sample.size(0)):
                true_img = y[i].detach().cpu()
                pred_img = sample[i].detach().cpu()
                row = torch.cat((x[i].cpu(), true_img, pred_img), dim=1)
                if output is None:
                    output = row
                else:
                    output = torch.cat((output,row), dim=2)
            save_image(output, os.path.join(self.out_root, "trues-{}.png".format(i_batch)))