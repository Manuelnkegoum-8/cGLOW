import os
import numpy as np
import torch
import PIL.Image as Image
from torch.utils import data
import torchvision.transforms as transforms
import math

class CustomDataset(data.Dataset):

    def __init__(self, dir, size, portion="train"):
        self.dir = dir
        self.names = self.read_names(dir, portion)
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
        lbl = lbl.repeat(3,1,1) #make it three channels
        return {"x":img, "y":lbl}