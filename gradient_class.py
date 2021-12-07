import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class Sobel_gray(nn.Module):
    def __init__(self):
        super(Sobel_gray, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3) #注意每一维的意义，以及多个kernel的cat方式。
        self.edge_conv.weight = nn.Parameter(edge_k)  #convert to a kind of tensor that can be used as the parameters of a mudule. have "requires_grad" attribute

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out

class Sobel_RGB(nn.Module):
    def __init__(self):
        super(Sobel_RGB, self).__init__()
        self.edge_conv = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1, groups=3, bias=False)
        edge_kx1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_kx2 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_kx3 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky3 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx1, edge_ky1, edge_kx2, edge_ky2, edge_kx3, edge_ky3))

        edge_k = torch.from_numpy(edge_k).float().view(6, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)  # convert to a kind of tensor that can be used as the parameters of a mudule. have "requires_grad" attribute

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 6, x.size(2), x.size(3))

        return out


class Sobel_RGB1(nn.Module):   #用3维卷积实现彩色图像的梯度计算。
    def __init__(self):
        super(Sobel_RGB1, self).__init__()
        self.edge_conv = nn.Conv3d(1, 2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False)
        edge_kx1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])[np.newaxis]
        edge_ky1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])[np.newaxis]

        edge_k = np.stack((edge_kx1, edge_ky1))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)  # convert to a kind of tensor that can be used as the parameters of a mudule. have "requires_grad" attribute

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(out.size(0), out.size(1)*out.size(2), out.size(3), out.size(4))

        return out
