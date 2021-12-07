import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable
import numpy as np

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)   #the kernel size is 3
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)    #the size of out is the same as x



class BottleneckBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock1, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=5, stride=1,
                               padding=2, bias=False)  #the kernel size is 5
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)   #the size of out is the same as x



class BottleneckBlock2(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock2, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=7, stride=1,
                               padding=3, bias=False)  #the kernel size is 7
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)  #the size of out is the same as x

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)   #upsample to twice



class TransitionBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)  #downsample to half



class TransitionBlock3(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock3, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out   #neither upsampling nor downsampling



class Multi_stream_dense_net(nn.Module):
    def __init__(self):
        super(Multi_stream_dense_net, self).__init__()

        self.conv_refin=nn.Conv2d(9,20,3,1,1)

        self.softmax2d=nn.Softmax2d()

        self.conv610 = nn.Conv2d(20, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv620 = nn.Conv2d(20, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv630 = nn.Conv2d(20, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv640 = nn.Conv2d(20, 1, kernel_size=3,stride=1,padding=1)  # 1mm

        self.refine3= nn.Conv2d(20+4, 2, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest

        self.res_layers = nn.Sequential(
            ResidualBlock(20),
            ResidualBlock(20),
            ResidualBlock(20),
            ResidualBlock(20),
            ResidualBlock(20)
        )

        self.relu=nn.ReLU(inplace=True)

        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)

        self.dense0=Dense_base_down0()
        self.dense1=Dense_base_down1()
        self.dense2=Dense_base_down2()

    def forward(self, x):
        ## 512x512

        x3=self.dense2(x)
        x2=self.dense1(x)
        x1=self.dense0(x)   #the image size no changes

        x4=torch.cat([x1,x,x2,x3],1)  #x1, x2, x3 are 4-D tensor (batchsize, channels, height, width)

        x5=self.relu((self.conv_refin(x4)))  #the size of feature map dose not change

        shape_out = x5.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]  #obtain the height and width of the feature maps

        x61 = F.avg_pool2d(x5, 16)
        x62 = F.avg_pool2d(x5, 8)
        x63 = F.avg_pool2d(x5, 4)
        x64 = F.avg_pool2d(x5, 2)    #pyramid pooling

        x61 = self.res_layers(x61)
        x62 = self.res_layers(x62)
        x63 = self.res_layers(x63)
        x64 = self.res_layers(x64)
        x5 = self.res_layers(x5)

        x610 = self.upsample(self.relu((self.conv610(x61))), size=shape_out)
        x620 = self.upsample(self.relu((self.conv620(x62))), size=shape_out)
        x630 = self.upsample(self.relu((self.conv630(x63))), size=shape_out)
        x640 = self.upsample(self.relu((self.conv640(x64))), size=shape_out) #upsampling to original size

        feature_map = torch.cat((x610, x620, x630, x640, x5), 1)  #concatenating x5 is the operation of 'short path' mentioned in the paper

        feature_map = self.refine3(feature_map)
        rain_map = self.softmax2d(feature_map)  #the input for Softmax2d must have at least two channels.
        rain_map = rain_map[:, 0, :, :]  #by this operation, the dimension will reduce one.
        rain_map = rain_map.unsqueeze(1)  #this operation add the reduced dimension
#        rain_map = torch.cat([rain_map, rain_map, rain_map], 1)
#        rain_map = self.softmax2d(self.refine3(feature_map))

        return rain_map

#Dense_base_down2 is the dense block whose kernel size is 3
class Dense_base_down2(nn.Module):
    def __init__(self):
        super(Dense_base_down2, self).__init__()

        self.dense_block1=BottleneckBlock2(3,13)
        self.trans_block1=TransitionBlock1(16,8)  #downsample to half. the channel number 16 is obtained by 3+16, because of cat operation

        ############# Block2-down   ##############
        self.dense_block2=BottleneckBlock2(8,16)
        self.trans_block2=TransitionBlock1(24,16) #downsample to half

        ############# Block3-down   ##############
        self.dense_block3=BottleneckBlock2(16,16)
        self.trans_block3=TransitionBlock3(32,16) #the size of feature map does not change.

        ############# Block4-up   ##############
        self.dense_block4=BottleneckBlock2(16,16)
        self.trans_block4=TransitionBlock3(32,16) #the size of feature map does not change.

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock2(16,8)
        self.trans_block5=TransitionBlock(24,8)  #upsample to twice

        self.dense_block6=BottleneckBlock2(8,8)
        self.trans_block6=TransitionBlock(16,2)  #upsample to twice

    def forward(self, x):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)

        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)

        # print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        ## Classifier  ##

        x4=(self.dense_block4(x3))
        x4=self.trans_block4(x4)

        x4=x4+x2

        x5=(self.dense_block5(x4))
        x5=self.trans_block5(x5)

        x5=x5+x1

        x6=(self.dense_block6(x5))
        x6=(self.trans_block6(x6))

        return x6


class Dense_base_down1(nn.Module):
    def __init__(self):
        super(Dense_base_down1, self).__init__()

        self.dense_block1=BottleneckBlock1(3,13)
        self.trans_block1=TransitionBlock1(16,8)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock1(8,16)
        self.trans_block2=TransitionBlock3(24,16)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock1(16,16)
        self.trans_block3=TransitionBlock3(32,16)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock1(16,16)
        self.trans_block4=TransitionBlock3(32,16)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock1(16,8)
        self.trans_block5=TransitionBlock3(24,8)

        self.dense_block6=BottleneckBlock1(8,8)
        self.trans_block6=TransitionBlock(16,2)

    def forward(self, x):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)

        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)

        # print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        ## Classifier  ##

        x4=(self.dense_block4(x3))
        x4=self.trans_block4(x4)

        x4=x4+x2

        x5=(self.dense_block5(x4))
        x5=self.trans_block5(x5)

        x5=x5+x1

        x6=(self.dense_block6(x5))
        x6=(self.trans_block6(x6))

        return x6

class Dense_base_down0(nn.Module):
    def __init__(self):
        super(Dense_base_down0, self).__init__()

        self.dense_block1=BottleneckBlock(3,5)
        self.trans_block1=TransitionBlock3(8,4)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock(4,8)
        self.trans_block2=TransitionBlock3(12,12)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock(12,4)
        self.trans_block3=TransitionBlock3(16,12)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(12,4)
        self.trans_block4=TransitionBlock3(16,12)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(12,8)
        self.trans_block5=TransitionBlock3(20,4)

        self.dense_block6=BottleneckBlock(4,8)
        self.trans_block6=TransitionBlock3(12,2)


    def forward(self, x):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)

        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)

        # print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        ## Classifier  ##

        x4=(self.dense_block4(x3))
        x4=self.trans_block4(x4)

        x4=x4+x2

        x5=(self.dense_block5(x4))
        x5=self.trans_block5(x5)

        x5=x5+x1

        x6=(self.dense_block6(x5))
        x6=(self.trans_block6(x6))

        return x6

class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class ConvLayer(nn.Module):
    """
    add ReflectionPad for Conv
    默认的卷积的padding操作是补0，这里使用边界反射填充
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
