import os
import numpy as np
import torch
import PIL.Image as Image
from torch.utils import data
import torchvision.transforms as transforms
import math

def my_preprocess(image,num_bits=2,training=True):
    # Discretize to the given number of bits
    shape = image.size()
    image = image*255.
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


class CustomDataset(data.Dataset):

    def __init__(self, dir, size, n_classes, portion="train"):
        self.dir = dir
        self.names = self.read_names(dir, portion)
        self.n_c = n_classes
        self.size = size

    def read_names(self, dir, portion):

        path = os.path.join(dir, "{}".format(portion))
        path_im = os.path.join(path, "images")
        path_mask = os.path.join(path, "masks")
        names = list()
        for root, _, files in os.walk(path_im):
            for file in files:
                if file.lower().endswith(('.png', '.jpg')):
                    name = {}
                    name['img'] = os.path.join(root, file)
                    name['mask'] = os.path.join(path_mask,file)
                    names.append(name)
        return names

    def __len__(self):
        return len(self.names)


    def __getitem__(self, index):

        # path
        name = self.names[index]
        img_path = name["img"]
        lbl_path = name["mask"]
        transform = transforms.Compose([transforms.Resize((self.size,self.size)), transforms.ToTensor()])

        # img
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        # lbl
        lbl = Image.open(lbl_path).convert("L")
        lbl = transform(lbl)
        if self.n_c == 2: # binary segmentation
            lbl = lbl.repeat(3,1,1) #make it three channels
        return {"x":img, "y":lbl}



0