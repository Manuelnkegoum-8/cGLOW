import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from .modules import *
from common.utils import * 

class cActnorm(nn.Module):
    """
    feed forward network for defining the operations of scaling and translation needed in realnvp
    """
    def __init__(self,in_size=(3,224,224),y_channels = 3,stride=2,hidden_channels=8,hidden_size=32):
        super(cActnorm,self).__init__()

        C,H,W = in_size
        self.convolutions = nn.Sequential(
            Rconv(in_channels=C,out_channels=hidden_channels,stride=stride),
            nn.ReLU(),
            Rconv(in_channels=hidden_channels,out_channels=hidden_channels,stride=stride),
            nn.ReLU(),
            Rconv(in_channels=hidden_channels,out_channels=hidden_channels,stride=stride),
            nn.ReLU(),
            nn.Flatten()
        )
        flat = int(hidden_channels*(H*W)//(stride**6))
        self.fc = nn.Sequential(
            LinearLayer(flat,hidden_size,"zeros"),
            nn.ReLU(),
            LinearLayer(hidden_size,hidden_size,"zeros"),
            nn.ReLU(),
            LinearLayer(hidden_size,y_channels*2,"zeros"),
            nn.Tanh()
        )    

    def forward(self, x,y,logdet=0., reverse = False):
        out = self.fc(self.convolutions(x))
        dimentions = y.size(2) * y.size(3)
        s,b = torch.chunk(out,2,1)
        log_s = s.unsqueeze(-1).unsqueeze(-1)
        b = b.unsqueeze(-1).unsqueeze(-1)
        if reverse:
            # observed to latent v -> u
            v = (y-b)*torch.exp(-log_s)
            dlogdet = - dimentions * torch.sum(log_s, dim=(1,2,3))
        else:
            # latent to observed u -> v
            v = y*torch.exp(log_s) + b
            dlogdet = dimentions * torch.sum(log_s, dim=(1,2,3))

        logdet = logdet + dlogdet
        return v,logdet

class cInvertibleConv(nn.Module):
    """
    feed forward network for defining the operations of scaling and translation needed in realnvp
    """
    def __init__(self,in_size=(3,224,224),y_channels = 3,stride=2,hidden_channels=8,hidden_size=32):
        super(cInvertibleConv,self).__init__()
        C,H,W = in_size
        self.channels = y_channels
        self.convolutions = nn.Sequential(
            Rconv(in_channels=C,out_channels=hidden_channels,stride=stride),
            nn.ReLU(),
            Rconv(in_channels=hidden_channels,out_channels=hidden_channels,stride=stride),
            nn.ReLU(),
            Rconv(in_channels=hidden_channels,out_channels=hidden_channels,stride=stride),
            nn.ReLU(),
            nn.Flatten()
        )
        flat = int(hidden_channels*(H*W)//(stride**6))
        self.fc = nn.Sequential(
            LinearLayer(flat,hidden_size,"zeros"),
            nn.ReLU(),
            LinearLayer(hidden_size,hidden_size,"zeros"),
            nn.ReLU(),
            LinearLayer(hidden_size,y_channels**2,"normal"),
            nn.Tanh(),
        )     


    def forward(self, x,y,logdet = 0.,reverse = False):
        matrix = self.fc(self.convolutions(x)).view(-1,self.channels,self.channels)
        dimensions = y.size(2) * y.size(3)
        dlogdet = torch.slogdet(matrix)[1] * dimensions
        if reverse:
            logdet = logdet - dlogdet
            matrix = torch.inverse(matrix)

        else:
            logdet = logdet + dlogdet
        matrix = matrix.unsqueeze(-1).unsqueeze(-1)
        B,C,H,W = y.size()
        y = y.view(1,B*C,H,W)

        weight = matrix.reshape(B*C,C,1,1)
        out = F.conv2d(y,weight,groups=B)
        out = out.view(B,C,H,W)
        return out,logdet


class cAffine(nn.Module):
    def __init__(self,in_size=(3,224,224),y_size=(3,224,224),hidden_channels=16):
        super(cAffine,self).__init__()

        C,H,W = in_size
        stride = H//y_size[1]
        self.d = y_size[0]
        self.convolutions = nn.Sequential(Conv2d(in_channels=C,out_channels=hidden_channels),
                                        nn.ReLU(),
                                        Rconv(in_channels=16,out_channels=self.d,stride=stride),
                                        nn.ReLU(),
                                        Conv2d(in_channels=self.d,out_channels=self.d),
                                        nn.ReLU()
                                        ) 
        self.conv2 = nn.Sequential(Conv2d(in_channels=2*self.d,out_channels=256),
                                        nn.ReLU(),
                                        Conv2d(in_channels=256,out_channels=256,kernel_size=[1,1]),
                                        nn.ReLU(),
                                        ZeroConv2d(256,2*self.d),
                                        nn.Tanh(),
                                        ) 

    def forward(self, x,y,logdet=0.,reverse = False):
        y_in,y_mod = y.chunk(2,1)
        xr = self.convolutions(x)
        h = torch.cat((xr,y_in),dim=1)
        out = self.conv2(h)
        log_s, b = split_feature(out, "cross")
        sigma = torch.sigmoid(log_s + 2. )
        if reverse:
            tmp = (y_mod-b)/(sigma+1e-8)
            logdet = -torch.sum(log_s, dim=(1, 2, 3)) + logdet
        else:
            tmp = sigma*y_mod + b
            logdet = torch.sum(torch.log(sigma), dim=(1, 2, 3)) + logdet 


        out = torch.cat((y_in,tmp),dim=1)
        return out,logdet

class cGlowStep(nn.Module):
    """
    feed forward network for defining the operations of scaling and translation needed in realnvp
    """
    def __init__(self,in_size=(3,224,224),y_size=(3,224,224),y_channels = 3,stride=2,hidden_channels=8,hidden_size=32):
        super(cGlowStep,self).__init__()
        self.actnorm = cActnorm(in_size,y_channels,stride,hidden_channels,hidden_size)
        self.invconv = cInvertibleConv(in_size,y_channels,stride,hidden_channels,hidden_size)
        self.affine = cAffine(in_size,[y_size[0] // 2, y_size[1], y_size[2]],16)

    def forward(self, x,y,logdet=0., reverse = False):
        if reverse:
            y,logdet = self.affine(x,y,logdet,reverse)
            y,logdet = self.invconv(x,y,logdet,reverse)
            y,logdet = self.actnorm(x,y,logdet,reverse)
        else:    
            y,logdet = self.actnorm(x,y,logdet,reverse)
            y,logdet = self.invconv(x,y,logdet,reverse)
            y,logdet = self.affine(x,y,logdet,reverse)
        return y,logdet


class CondGlow(nn.Module):
    def __init__(self,x_size,y_size,hidden_channels,stride,hidden_size, K, L):
        super(CondGlow,self).__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        self.L = L
        C, H, W = y_size
        for l in range(0, L):
            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            y_size = [C,H,W]
            self.layers.append(SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])
            # 2. K CGlowStep
            for k in range(0, K):
                self.layers.append(cGlowStep(in_size=x_size,y_size=y_size,y_channels =C,stride=stride,
                                            hidden_channels=hidden_channels,hidden_size=hidden_size))
                self.output_shapes.append([-1, C, H, W])
            # 3. Split
            if l < L - 1:
                self.layers.append(Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2
    def forward(self, x, y, logdet=0.0, reverse=False, eps_std=1.0):
        if reverse == False:
            return self.encode(x, y, logdet)
        else:
            return self.decode(x, y, logdet, eps_std)

    def encode(self, x, y, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            if isinstance(layer,Split2d) or isinstance(layer, SqueezeLayer):
                y, logdet = layer(y, logdet, reverse=False)

            else:
                y, logdet = layer(x, y, logdet, reverse=False)
        return y, logdet

    def decode(self, x, y, logdet=0.0, eps_std=1.0):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                y, logdet = layer(y, logdet=logdet, reverse=True, eps_std=eps_std)

            elif isinstance(layer, SqueezeLayer):
                y, logdet = layer(y, logdet=logdet, reverse=True)

            else:
                y, logdet = layer(x, y, logdet=logdet, reverse=True)

        return y, logdet


class cGlowModel(nn.Module):
    def __init__(self, args):
        super(cGlowModel,self).__init__()
        self.flow = CondGlow(x_size=args.x_size,
                            y_size=args.y_size,
                            hidden_channels=args.hidden_channels,
                            hidden_size=args.hidden_size,
                            stride=args.stride,
                            K=args.flow_depth,
                            L=args.num_levels,
                            )

        self.learn_top = args.learn_top
        self.register_parameter("new_mean",
                                nn.Parameter(torch.zeros(
                                    [1,
                                     self.flow.output_shapes[-1][1],
                                     self.flow.output_shapes[-1][2],
                                     self.flow.output_shapes[-1][3]])))


        self.register_parameter("new_logs",
                                nn.Parameter(torch.zeros(
                                    [1,
                                     self.flow.output_shapes[-1][1],
                                     self.flow.output_shapes[-1][2],
                                     self.flow.output_shapes[-1][3]])))
        self.n_bins = 2**args.y_bits

    def prior(self):
        if self.learn_top:
            return self.new_mean, self.new_logs
        else:
            return torch.zeros_like(self.new_mean), torch.zeros_like(self.new_mean)


    def forward(self, x=0.0, y=None, eps_std=1.0, reverse=False):
        if reverse == False:
            dimensions = y.size(1)*y.size(2)*y.size(3)
            logdet = torch.zeros_like(y[:, 0, 0, 0])
            logdet += float(-np.log(self.n_bins) * dimensions)
            z, objective = self.flow(x, y, logdet=logdet, reverse=False)
            mean, logs = self.prior()
            objective += GaussianDiag.logp(mean, logs, z)
            nll = -objective / float(np.log(2.) * dimensions)
            return z, nll

        else:
            with torch.no_grad():
                mean, logs = self.prior()
                if y is None:
                    y = GaussianDiag.batchsample(x.size(0), mean, logs, eps_std)
                y, logdet = self.flow(x, y, eps_std=eps_std, reverse=True)
            return y, logdet