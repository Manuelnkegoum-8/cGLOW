import torch
import numpy as np
import math
from cGlow.modules import *
from cGlow.model import *
from common.utils import *
from common.dataset import *
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='Training')
# model parameters
parser.add_argument("--x_size", type=int, default=64)
parser.add_argument("--y_size", type=int, default=64)
parser.add_argument("--hidden_channels", type=int, default=64)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("-K", "--flow_depth", type=int, default=8)
parser.add_argument("-L", "--num_levels", type=int, default=3)
parser.add_argument("--learn_top", type=bool, default=True)

# dataset
parser.add_argument("-r", "--dataset_root", type=str, default="Data")
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--y_bits", type=float, default=2.0)

# Optimizer parameters
parser.add_argument("--lr", type=float, default=0.0002)


# Trainer parameters
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_steps", type=int, default=0)

args = parser.parse_args()

class CFG:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
size = args.x_size

config = CFG(
     x_size=(3,size,size),
     y_size=(3,size,size),
     hidden_channels=args.hidden_channels,
     hidden_size=args.hidden_size,
     stride=2,
     learn_top = args.learn_top,
     num_steps = args.num_steps,
     y_bins = args.y_bits,
     batch_size = args.batch_size,
     flow_depth=args.flow_depth,
     num_levels=args.num_levels,
     lr = args.lr,
     betas=(0.9,0.9999),
     out_root = 'results',
     num_labels = args.num_classes,
     device = device
)

if __name__=='__main__':
    print("[INFO] Loading data")
    trainingset = CustomDataset( dir = args.dataset_root, size=size, n_classes = args.num_classes, portion="train")
    testset = CustomDataset( dir = args.dataset_root, size=size, n_classes =  args.num_classes, portion="val")

    n_epochs =  math.ceil((args.num_steps * args.batch_size) / len(trainingset))
    print("[INFO] Initializing cGlow")
    model = cGlowModel(config)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,betas=config.betas)
    train_loader = DataLoader(trainingset,batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(testset,batch_size=args.batch_size,shuffle=False)

    print("[INFO] Training check logs for detailed results")
    trainer = Trainer(model,trainloader=train_loader,
                    valloader=test_loader,
                    optimizer=optimizer,
                    args=config)
                    
    trainer.train(n_epochs)
    print("[INFO] End of trainin,saving model")
    torch.save(model.state_dict(), 'my_model_2.pth')