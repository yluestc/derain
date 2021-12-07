# coding:utf8
"""
code refer to https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/transformer_net.py
"""
import torch as t
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()

        #refine the derain result
        self.pre_refine = nn.Sequential(
            nn.Conv2d(4, 20, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=3,stride=1,padding=1)  # 1mm

        self.refine= nn.Conv2d(20+4, 1, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest
        self.relu=nn.LeakyReLU(0.2, inplace=True)

        self.batch= nn.BatchNorm2d(1)
        self.tanh=nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, derain_inter):

        #refine
        derain = derain_inter
        derain = torch.cat([derain,x],1)

        derain = self.pre_refine(derain)
        shape_out = derain.data.size()
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(derain, 32)
        x102 = F.avg_pool2d(derain, 16)
        x103 = F.avg_pool2d(derain, 8)
        x104 = F.avg_pool2d(derain, 4)

        x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)

        derain = torch.cat((x1010, x1020, x1030, x1040, derain), 1)
        derain_final = self.sigmoid(self.refine(derain))

        return derain_final
