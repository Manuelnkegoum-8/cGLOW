import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from common.utils import*

class Rconv(nn.Conv2d):
    """
    feed forward network for defining the operations of scaling and translation needed in realnvp
    """
    def __init__(self,in_channels,out_channels,stride):
        ks = stride
        super().__init__(in_channels=in_channels,out_channels=out_channels,kernel_size=ks,stride=stride)

class LinearLayer(nn.Linear):
    def __init__(self, in_channels, out_channels, initialization='zeros'):
        super().__init__(in_channels, out_channels)

        if initialization == 'zeros':
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)
        elif initialization == 'normal':
            nn.init.normal_(self.weight, mean=0, std=0.1)
            nn.init.normal_(self.bias, mean=0, std=0.1)
        else:
            raise ValueError("Invalid initialization type. Supported types are 'zeros' and 'normal'.")

class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(out_channel, 1, 1))
        self.newbias = nn.Parameter(torch.zeros(out_channel, 1, 1))

    def forward(self, inputs):
        out = self.conv(inputs)
        out = out + self.newbias
        out = out * torch.exp(self.scale * 3.0)

        return out

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size=[3,3], stride=[1,1]):
        padding = (kernel_size[0] - 1) // 2
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.weight.data.normal_(mean=0.0, std=0.05)
        self.bias.data.zero_()



class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            output = SqueezeLayer.squeeze2d(input, self.factor)
            return output, logdet
        else:
            output = SqueezeLayer.unsqueeze2d(input, self.factor)
            return output, logdet


    @staticmethod
    def squeeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return input
        B, C, H, W = input.size()
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W))
        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * factor * factor, H // factor, W // factor)
        return x


    @staticmethod
    def unsqueeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        factor2 = factor ** 2
        if factor == 1:
            return input
        B, C, H, W = input.size()
        assert C % (factor2) == 0, "{}".format(C)
        x = input.view(B, C // factor2, factor, factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)
        return x


class Split2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.conv =  nn.Sequential(ZeroConv2d(num_channels // 2, num_channels),nn.Tanh())

    def split2d_prior(self, z):
        h = self.conv(z)
        return split_feature(h, "cross")

    def forward(self, input, logdet=0., reverse=False, eps_std=None):
        if not reverse:
            z1, z2 = split_feature(input, "split")
            mean, logs = self.split2d_prior(z1)
            logdet = GaussianDiag.logp(mean, logs, z2) + logdet

            return z1, logdet
        else:
            z1 = input
            mean, logs = self.split2d_prior(z1)
            z2 = GaussianDiag.sample(mean, logs, eps_std)
            z = torch.cat((z1, z2), dim=1)

            return z, logdet


class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        return -0.5 * (logs * 2. + ((x - mean) ** 2.) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return torch.sum(likelihood, dim=(1, 2, 3))

    @staticmethod
    def sample(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        eps = torch.normal(mean=torch.zeros_like(mean),
                           std=torch.ones_like(logs) * eps_std)
        return mean + torch.exp(logs) * eps

    @staticmethod
    def batchsample(batchsize, mean, logs, eps_std=None):
        eps_std = eps_std or 1
        sample = GaussianDiag.sample(mean, logs, eps_std)
        for i in range(1, batchsize):
            s = GaussianDiag.sample(mean, logs, eps_std)
            sample = torch.cat((sample, s), dim=0)
        return sample

