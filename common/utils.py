import os
import torch
import numpy as np
from tqdm import trange
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import skimage.color
import skimage.util
import skimage.io

def my_preprocess(image):
    image = image - 0.5
    image = image + torch.zeros_like(image).uniform_(-0.5,0.5)
    return image

def my_postprocess(image):
    image = image + 0.5
    return image

def convert_to_img(y):
    C = y.size(1)
    transform = transforms.ToTensor()
    colors = np.array([[0,0,0],[255,255,255]])
    seg = torch.mean(y, dim=1, keepdim=False).cpu().numpy()
    seg = np.nan_to_num(seg)
    seg = np.clip(np.round(seg),a_min=0, a_max=1)
    B,C,H,W = y.size()
    imgs = list()
    for i in range(B):
        label_i = skimage.color.label2rgb(seg[i], bg_label=0, bg_color=(0, 0, 0), colors=[(1.,1.,1.)])
        label_i = skimage.util.img_as_ubyte(label_i)
        imgs.append(transform(label_i))
    return imgs, seg


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
    path = os.path.join(dir, "checkpoint_{}.pth.tar".format(iteration))
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
            y = my_preprocess(y)
            sample = sample_based(model,x,n_samples)

            all_samples.append(sample)
            all_masks.append(y)
            all_images.append(x)

    all_samples = torch.stack(all_samples,dim=0)
    all_masks = torch.stack(all_masks,dim=0)
    all_images = torch.stack(all_images,dim=0)

    all_samples = my_postprocess(all_samples)
    all_samples,true_seg = convert_to_img(all_samples)
    all_masks = my_postprocess(all_maks)
    all_masks,pred_seg = convert_to_img(all_maks)

    iou = compute_iou(pred_seg,true_seg)
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
        self.fp_in = os.path.join(self.out_root,"step_*.png")
        self.p_out = "sample_evolution.gif"

    def train(self,n_epochs):
        for epoch in trange(n_epochs):
            avg_loss = 0.
            n = 0.
            self.model.train()
            for data in self.train_loader:
                x, y = data['x'],data['y']
                x = x.to(self.device)
                y = y.to(self.device)
                y = my_preprocess(y,self.y_bits,training=True)
                # forward
                z, nll = self.model(x + torch.zeros_like(x).uniform_(0,1/256.), y)
                # loss
                loss = torch.mean(nll)
                avg_loss = avg_loss + loss.item()*x.size(0)
                # backward
                self.optim.zero_grad()
                loss.backward()
                if args.grad_clip >0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(),args.grad_clip)
                if args.grad_norm >0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),args.grad_norm)
                self.optim.step()
                n+= x.size(0)
            avg_loss = avg_loss / n

            val_loss = self.validate()
            self.writer.add_scalar('Loss/train', avg_loss, global_step=epoch)
            self.writer.add_scalar('Loss/val', val_loss, global_step=epoch)

            if self.scheduler is not None:
                scheduler.step()

            if epoch%args.checkpoint==0:
                save_model(self.model, self.optim, self.scheduler,'checkpoints/{}-cglow.pth'.format(epoch), epoch)
                output,iou = prediction(self.model,self.device,self.val_loader,10)
                save_image(output, os.path.join(self.out_root, "step-{}.png".format(epoch)))
                self.writer.add_scalar('Iou', iou, global_step=epoch)
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(self.fp_in))]
        img.save(fp=self.fp_out, format='GIF', append_images=imgs,save_all=True, duration=20, loop=0)


    @torch.no_grad()
    def validate(self):
        self.model.eval()
        avg_loss = 0.
        n = 0.
        for data in self.val_loader:
                x, y = data['x'],data['y']
                x = x.to(self.device)
                y = y.to(self.device)
                y = my_preprocess(y,self.y_bits,training=True)
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