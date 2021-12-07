
import torch as t
from torch import nn
import numpy as np
from models.Dense_block import DenseBlock
from models.Dense_block import Transition
from collections import OrderedDict

class Multi_scale_feature_extraction(nn.Module):
    def __init__(self, num_init_features=32):     #set num_init_features to 16, so that the channels of the input of dense block are 64
        super(Multi_scale_feature_extraction, self).__init__()

        # First convolution
        self.preprocess = nn.Sequential(OrderedDict([
            ('conv0', Reflection_Pad_Conv(3, num_init_features, kernel_size=9, stride=1)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))

        self.res_layers2 = nn.Sequential(
            ResidualBlock(num_init_features),
            ResidualBlock(num_init_features),
            ResidualBlock(num_init_features),
            ResidualBlock(num_init_features),
            ResidualBlock(num_init_features)
        )

        # large-scale preprocessing
        self.large_scale_preprocess = nn.Sequential(OrderedDict([
            ('convL1', Reflection_Pad_Conv(num_init_features, num_init_features*2, kernel_size=3, stride=1)),
            ('normL1', nn.BatchNorm2d(num_init_features*2)),
            ('reluL1', nn.ReLU(inplace=True)),
            ('convL2', Reflection_Pad_Conv(num_init_features*2, num_init_features*4, kernel_size=3, stride=1)),
        ]))

        # middle-scale preprocessing
        self.mid_scale_preprocess = nn.Sequential(OrderedDict([
            ('convM1', Reflection_Pad_Conv(num_init_features, num_init_features*2, kernel_size=3, stride=2)),   #Downsample
            ('normM1', nn.BatchNorm2d(num_init_features*2)),
            ('reluM1', nn.ReLU(inplace=True)),
            ('convM2', Reflection_Pad_Conv(num_init_features*2, num_init_features*4, kernel_size=3, stride=1)),
        ]))

        # low-scale preprocessing
        self.low_scale_preprocess = nn.Sequential(OrderedDict([
            ('convLow1', Reflection_Pad_Conv(num_init_features, num_init_features*2, kernel_size=3, stride=2)),   #Downsample
            ('normLow1', nn.BatchNorm2d(num_init_features*2)),
            ('reluLow1', nn.ReLU(inplace=True)),
            ('convLow2', Reflection_Pad_Conv(num_init_features*2, num_init_features*4, kernel_size=3, stride=2)),
        ]))

        self.res_layers1 = nn.Sequential(
            ResidualBlock(num_init_features*4),
            ResidualBlock(num_init_features*4),
            ResidualBlock(num_init_features*4),
            ResidualBlock(num_init_features*4),
            ResidualBlock(num_init_features*4)
        )

        num_features = num_init_features*4

        self.large_scale_post_process = nn.Sequential(OrderedDict([
            ('convL1post', Reflection_Pad_Conv(num_features, 64, kernel_size=3, stride=1)),
            ('normL1post', nn.BatchNorm2d(64)),
            ('reluL1post', nn.ReLU(inplace=True)),
            ('convL2post', Reflection_Pad_Conv(64, 32, kernel_size=3, stride=1)),
            ('normL2post', nn.BatchNorm2d(32)),
            ('reluL2post', nn.ReLU(inplace=True)),
        ]))

        self.mid_scale_post_process = nn.Sequential(OrderedDict([
            ('convM1post', UpsampleConvLayer(num_features, 64, kernel_size=3, stride=1, upsample=2)),  #upsample
            ('normM1post', nn.BatchNorm2d(64)),
            ('reluM1post', nn.ReLU(inplace=True)),
            ('convM2post', Reflection_Pad_Conv(64, 32, kernel_size=3, stride=1)),
            ('normM2post', nn.BatchNorm2d(32)),
            ('reluM2post', nn.ReLU(inplace=True)),
        ]))

        self.low_scale_post_process = nn.Sequential(OrderedDict([
            ('convLow1post', UpsampleConvLayer(num_features, 64, kernel_size=3, stride=1, upsample=2)),  #upsample
            ('normLow1post', nn.BatchNorm2d(64)),
            ('reluLow1post', nn.ReLU(inplace=True)),
            ('convLow2post', UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)),  #upsample
            ('normLow2post', nn.BatchNorm2d(32)),
            ('reluLow2post', nn.ReLU(inplace=True)),
        ]))

        self.fuse_feature = nn.Sequential(OrderedDict([
            ('conv1Fuse', Reflection_Pad_Conv(32*3, 16*3, kernel_size=3, stride=1)),
            ('norm1Fuse', nn.BatchNorm2d(16*3)),
            ('relu1Fuse', nn.ReLU(inplace=True)),
            ('conv2Fuse', Reflection_Pad_Conv(16*3, 3, kernel_size=3, stride=1)),
            ('norm2Fuse', nn.BatchNorm2d(3)),
            ('relu2Fuse', nn.ReLU(inplace=True)),
        ]))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.preprocess(x)
        x = self.res_layers2(x)

        xL = self.large_scale_preprocess(x)
        xL = self.res_layers1(xL)
        xL = self.large_scale_post_process(xL)


        xM = self.mid_scale_preprocess(x)
        xM = self.res_layers1(xM)
        xM = self.mid_scale_post_process(xM)

        xLow = self.low_scale_preprocess(x)
        xLow = self.res_layers1(xLow)
        xLow = self.low_scale_post_process(xLow)

        xfeature = t.cat([xL, xM, xLow], 1)
        out = self.fuse_feature(xfeature)
#        out = self.sigmoid(out)

        return out

class Reflection_Pad_Conv(nn.Module):
    """
    add ReflectionPad for Conv
    默认的卷积的padding操作是补0，这里使用边界反射填充
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Reflection_Pad_Conv, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    默认的卷积的padding操作是补0，这里使用边界反射填充
    先上采样，然后做一个卷积(Conv2d)，而不是采用ConvTranspose2d，这种效果更好，参见
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Reflection_Pad_Conv(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = Reflection_Pad_Conv(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out
