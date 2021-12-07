from __future__ import print_function
import torch as t
import torchvision as tv
import torchnet as tnt
from torch.utils import data
import utils
from torch.autograd import Variable
from torch.nn import functional as F
import tqdm
import numpy
import os
import ipdb
import argparse
import sys
import random
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
cudnn.benchmark = True   #it enable cudnn to find the optimal algorithms automatically.
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from misc import *
from torchnet import meter   #value the performance of a method
from models.Refine_Net import RefineNet
from models.Triangle_Net import TriangleNet
from models.Transfer_Net import TransformerNet
from models.Shuffle_Transfer_Net_vapor import ShuffleTransformerNet
from models.Multi_scale_feature_extraction_net import Multi_scale_feature_extraction
from Loca_model.Multi_stream_dense_net_loca import Multi_stream_dense_net
from models.T_vapor import RefineNet
from Dataset_standard.Train_Test_Dataset import Train_Test_Data, Train_Test_Data2, Train_Test_Data1
from Dataset_standard.Train_Test_Dataset import Test_Practical
from math import log10
from init import init_weights
import time
from tensorboardX import SummaryWriter
from PackedVGG import Vgg16
import utils
from gradient_class import Sobel_RGB

parser = argparse.ArgumentParser()
parser.add_argument('--train_root', required=False,
  default='./Dataset_standard/train/', help='path to original training dataset')
parser.add_argument('--val_root', required=False,
  default='./Dataset_standard/val/', help='path to validating dataset')
parser.add_argument('--test_root', required=False,
  default='./test_data/', help='path to test rain dataset')
parser.add_argument('--practical_result_path', required=False,
  default='./results/practical_results/', help='path to loca results')
parser.add_argument('--val_result_path', required=False,
  default='./results/val_results/', help='path to loca results')
parser.add_argument('--test_result_path', required=False,
  default='./results/test_results/', help='path to loca results')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--test_batch_size', type=int, default=2, help='input batch size')
parser.add_argument('--val_batch_size', type=int, default=1, help='val batch size')
parser.add_argument('--max_epoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--manualSeed', type=int, default=100, help='random seed!')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate, default=0.0002')
parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate, default=0.0002')
parser.add_argument('--haze_tune', type=float, default=-0.1, help='handling haze-like effect')
parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--print_freq', type=int, default=20, help='print frequency')
parser.add_argument('--writer_freq', type=int, default=100, help='tensorboard writer frequency')
parser.add_argument('--weight_decay', type=float, default=0e-5, help='weight_decay')
parser.add_argument('--model_save_freq', type=int, default=2, help='model save frequency')
parser.add_argument('--model_path', default='./checkpoints/', help='folder to output model checkpoints')
parser.add_argument('--load_model_path_transmission', default=None, help='load the pretrained Derain_model')
parser.add_argument('--load_model_path_transmission_vapor', default=None, help='load the pretrained Derain_model')
parser.add_argument('--load_model_path_loca', default=None, help='load the pretrained loca_model')
parser.add_argument('--load_model_path_atm_light', default=None, help='load the pretrained atm_light_model')
parser.add_argument('--load_model_path_atm_light_before_tuned', default=None, help='load the pretrained atm_light_model')
parser.add_argument('--debug_file', default='/tmp/debug', help='the path used for debug')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--weather_name', type=str, default='rain', help='input batch size')
parser.add_argument('--train_dataset_name', type=str, default='rain', help='input batch size')
parser.add_argument('--val_dataset_name', type=str, default='rain', help='input batch size')
opt = parser.parse_args()
print(opt)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_if_exist(path):
    if os.path.exists(path):
        os.remove(path)

def test_practical():

    loca_model = Multi_stream_dense_net().eval()
    if opt.load_model_path_loca:
        loca_model.load_state_dict(t.load(opt.load_model_path_loca))
    if opt.cuda: loca_model.cuda()

    atm_light_model = TriangleNet(3, 3, 32).eval()   #刚开始的一个或者几个epoch可以对atm_light_model做微调，之后的epoch中可以设置为eval()模式，这样或许性能会更好。
    if opt.load_model_path_atm_light:
        atm_light_model.load_state_dict(t.load(opt.load_model_path_atm_light))
    if opt.cuda: atm_light_model.cuda()

    transmission_model = ShuffleTransformerNet().eval()
    if opt.load_model_path_transmission:
        transmission_model.load_state_dict(t.load(opt.load_model_path_transmission))
    if opt.cuda: transmission_model.cuda()

    atm_light_model_before_tuned = TriangleNet(3, 3, 32).eval()   #刚开始的一个或者几个epoch可以对atm_light_model做微调，之后的epoch中可以设置为eval()模式，这样或许性能会更好。
    if opt.load_model_path_atm_light_before_tuned:
        atm_light_model_before_tuned.load_state_dict(t.load(opt.load_model_path_atm_light_before_tuned))
    if opt.cuda: atm_light_model_before_tuned.cuda()

    T_vapor_model = RefineNet()
    init_weights(T_vapor_model, 'xavier')
    if opt.load_model_path_transmission_vapor:
        T_vapor_model.load_state_dict(t.load(opt.load_model_path_transmission_vapor))
    if opt.cuda:
        T_vapor_model.cuda()
    T_vapor_model.eval()

    # step2: data
    test_data = Test_Practical(opt.test_root)
    test_dataloader = DataLoader(test_data, opt.batch_size,
                        num_workers=opt.num_workers)

    for ii, (data_light, data) in enumerate(test_dataloader):

    # test model
        input_light = data_light
        input = data
        input = Variable(input)
        input_light = Variable(input_light)
        if opt.cuda:
            input = input.cuda()
            input_light = input_light.cuda()

        input_resize = F.upsample(input, [256, 256])
        with torch.no_grad():

            light = atm_light_model(input_resize)

            transmission, features = transmission_model(input)
            t_vapor = T_vapor_model(transmission, input)

            t_vapor[t_vapor > 0.000001] = 0.000001/2
            t_vapor = t_vapor * 10**5

            derain = (input_light - light) / (transmission - 0.1*t_vapor-0.06 + 10 ** (-10)) + light

    transmission_model.train()
    atm_light_model.train()
    T_vapor_model.train()

if __name__ == '__main__':
    test_practical()
