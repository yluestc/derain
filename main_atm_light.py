from __future__ import print_function
import torch as t
import torchvision as tv
import torchnet as tnt
from torch.utils import data
import utils
from torch.autograd import Variable
from torch.nn import functional as F
import tqdm
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
from models.Triangle_Net import TriangleNet
from Loca_model.Multi_stream_dense_net_loca import Multi_stream_dense_net
from Dataset_standard.Train_Test_Dataset import Train_Test_Data, Train_Test_Data1
from Dataset_standard.Train_Test_Dataset import Test_Practical
import pytorch_ssim
from math import log10
from init import init_weights
import time
from tensorboardX import SummaryWriter
from PackedVGG import Vgg16
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--train_root', required=False,
  default='./Dataset_standard/train/', help='path to original training dataset')
parser.add_argument('--val_root', required=False,
  default='./Dataset_standard/val/', help='path to validating dataset')
parser.add_argument('--test_synthetic_root', required=False,
  default='./test_data/synthetic/', help='path to test rain dataset')
parser.add_argument('--test_practical_root', required=False,
  default='./test_data/practical/', help='path to test rain dataset')
parser.add_argument('--practical_result_path', required=False,
  default='./results/practical_results/', help='path to loca results')
parser.add_argument('--val_result_path', required=False,
  default='./results/val_results/', help='path to loca results')
parser.add_argument('--test_result_path', required=False,
  default='./results/test_results/', help='path to loca results')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--val_batch_size', type=int, default=1, help='val batch size')
parser.add_argument('--max_epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--manualSeed', type=int, default=100, help='random seed!')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate, default=0.0002')
parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate, default=0.0002')
parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--print_freq', type=int, default=1, help='print frequency')
parser.add_argument('--writer_freq', type=int, default=200, help='tensorboard writer frequency')
parser.add_argument('--weight_decay', type=float, default=0e-5, help='weight_decay')
parser.add_argument('--model_save_freq', type=int, default=2, help='model save frequency')
parser.add_argument('--model_path', default='./checkpoints/', help='folder to output model checkpoints')
parser.add_argument('--load_model_path', default=None, help='load the pretrained Derain_model')
parser.add_argument('--load_model_path_loca', default=None, help='load the pretrained loca_model')
parser.add_argument('--debug_file', default='/tmp/debug', help='the path used for debug')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--weather_name', type=str, default='rain', help='input batch size')
parser.add_argument('--train_dataset_name', type=str, default='complex_rain', help='input batch size')
parser.add_argument('--val_dataset_name', type=str, default='rain', help='input batch size')
opt = parser.parse_args()
print(opt)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_if_exist(path):
    if os.path.exists(path):
        os.remove(path)

def train():
    loca_model = Multi_stream_dense_net().eval()
    if opt.load_model_path_loca:
        loca_model.load_state_dict(t.load(opt.load_model_path_loca))
    if opt.cuda: loca_model.cuda()

    atm_light_model = TriangleNet(3, 3, 32)
#    print(Derain_model)
    init_weights(atm_light_model, 'xavier')
    if opt.load_model_path:
        atm_light_model.load_state_dict(t.load(opt.load_model_path))
    if opt.cuda: atm_light_model.cuda()
    # step2: data
    train_data = Train_Test_Data1(opt.train_root + opt.train_dataset_name + '/') #Train_Test_Data1是用于Li ruoteng Heavy rain 这篇文章的数据
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)


    val_data = Train_Test_Data(opt.val_root + opt.val_dataset_name + '/')
    val_dataloader = DataLoader(val_data,opt.val_batch_size,
                        shuffle=False,num_workers=opt.num_workers)

    # step3: criterion and optimizer
    criterion = t.nn.MSELoss()
    lr = opt.lr
    optimizer = t.optim.Adam(atm_light_model.parameters(), lr = lr, weight_decay = opt.weight_decay)

    # step4: meters
    loss_meter = meter.AverageValueMeter()
    previous_loss = 1e5
    subname = opt.weather_name
    logdir_path = './logdir_atm_light/' + subname +'/'
    make_if_not_exist(logdir_path)
    writer = SummaryWriter(logdir_path)  #the root where the data is saved.
    n_iter = 0

    # train
    for epoch in range(opt.max_epoch):
        loss_meter.reset()

        for ii,(data_light, data, label) in enumerate(train_dataloader):

            # 进入debug模式
            if os.path.exists(opt.debug_file):
                import ipdb
                ipdb.set_trace()
            input_light = Variable(data_light)
            input = Variable(data)
            target = Variable(label)
            if opt.cuda:
                input_light = input_light.cuda()
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            if opt.weather_name == 'rain':
                loca = loca_model(input)
                loca = loca - 0.5
                loca = t.sign(loca)
                loca = (loca + 1)/2  #transfer the value in loca to binary value 0 and 1.
                light_target, light_loca = utils.atm_light(input_light, loca)
            elif opt.weather_name == 'haze':
                light_target = utils.atm_light_haze(input_light)

            light_target = Variable(light_target)
            if opt.cuda:
                light_target = light_target.cuda()  #do not forget "=", light_target.cuda() is wrong, should be light_target=light_target.cuda()
            light = atm_light_model(input)
            loss = criterion(light,light_target)

            loss.backward()

            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.data)

            if ii % opt.writer_freq==opt.writer_freq-1:
                n_iter = epoch * len(train_dataloader) + ii
                writer.add_scalar('average_loss', loss_meter.value()[0], n_iter)
                writer.add_scalar('loss', loss, n_iter)

            if ii % opt.print_freq==opt.print_freq-1:
                print (loss_meter.value()[0], loss, lr, ii, epoch)

        val(loca_model, atm_light_model, val_dataloader, opt.weather_name)
        # update learning rate
        if loss_meter.value()[0] > previous_loss - 10**(-6):
            lr = lr * opt.lr_decay
        # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]

        if epoch % opt.model_save_freq==opt.model_save_freq-1:
            prefix = './checkpoints_atm_light/' + opt.weather_name + '/'
            make_if_not_exist(prefix)
            name = prefix + 'atm_light-' + 'epoch' + str(epoch) + '.pth'
            t.save(atm_light_model.state_dict(), name)

def val(loca_model, atm_light_model, dataloader, weather_name):
    """
    计算模型在验证集上的准确率等信息
    """
    loca_model.eval()
    atm_light_model.eval()

    criterion = t.nn.MSELoss()

    for ii, (data_light, data, label) in enumerate(dataloader):
        input_light = Variable(data_light)
        input = Variable(data)
        target = Variable(label)
        if opt.cuda:
            input_light = input_light.cuda()
            input = input.cuda()
            target = target.cuda()
        with torch.no_grad():
            if weather_name == 'rain':
                loca = loca_model(input)
                loca = loca - 0.5
                loca = t.sign(loca)
                loca = (loca + 1)/2  #transfer the value in loca to binary value 0 and 1.
                light_target, light_loca = utils.atm_light(input_light, loca)
            elif weather_name == 'haze':
                light_target = utils.atm_light_haze(input_light)
            light_target = Variable(light_target)
            if opt.cuda:
                light_target = light_target.cuda()
            light = atm_light_model(input)

        loss = criterion(light,light_target)
        print("loss:", loss)

    atm_light_model.train()   #translate the state of model to train() not eval()

def test():
    loca_model = Multi_stream_dense_net().eval()
    if opt.load_model_path_loca:
        loca_model.load_state_dict(t.load(opt.load_model_path_loca))
    if opt.cuda: loca_model.cuda()

    atm_light_model = TriangleNet(3, 3, 32).eval()
#    print(Derain_model)
    init_weights(atm_light_model, 'xavier')
    if opt.load_model_path:
        atm_light_model.load_state_dict(t.load(opt.load_model_path))
    if opt.cuda: atm_light_model.cuda()
    # step2: data
    test_data = Train_Test_Data(opt.test_synthetic_root)
    test_dataloader = DataLoader(test_data, opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers)

    criterion = t.nn.MSELoss()
    loss_meter = meter.AverageValueMeter()

    for ii, (data_light, data, label) in enumerate(test_dataloader):

        # test model
        input_light = Variable(data_light)
        input = Variable(data)
        target = Variable(label)
        if opt.cuda:
            input_light = input_light.cuda()
            input = input.cuda()
            target = target.cuda()

        # 进入debug模式
        if os.path.exists(opt.debug_file):
            import ipdb;
            ipdb.set_trace()
        with torch.no_grad():
            if opt.weather_name == 'rain':
                loca = loca_model(input)
                loca = loca - 0.5
                loca = t.sign(loca)
                loca = (loca + 1)/2  #transfer the value in loca to binary value 0 and 1.
                light_target = utils.atm_light(input_light, loca)
            elif opt.weather_name == 'haze':
                light_target = utils.atm_light_haze(input_light)
            light_target = Variable(light_target)
            if opt.cuda:
                light_target = light_target.cuda()  #do not forget "=", light_target.cuda() is wrong, should be light_target=light_target.cuda()
            light = atm_light_model(input)
            loss = criterion(light,light_target)
            loss_meter.add(loss.data)
            if ii % opt.print_freq==opt.print_freq-1:
                print (loss_meter.value()[0], loss, ii)

    atm_light_model.train()

def test_practical():

    Derain_model = DerainNet()
    Derain_model.eval()
    if opt.load_model_path:
        Derain_model.load_state_dict(t.load(opt.load_model_path))
    if opt.cuda: Derain_model.cuda()
    # step2: data
    test_data = Test_Practical(opt.test_practical_root)
    test_dataloader = DataLoader(test_data, opt.batch_size,
                        num_workers=opt.num_workers)

    for ii, data in enumerate(test_dataloader):
        t0 = time.time()
    # test model
        input = data
        input = Variable(input)
        if opt.cuda:
            input = input.cuda()

        # 进入debug模式
        if os.path.exists(opt.debug_file):
            import ipdb;
            ipdb.set_trace()

        with torch.no_grad():
            derain_final, derain_inter, transmission = Derain_model(input)

        t1 = time.time()

        prefix = opt.practical_result_path
        name_derain1 = prefix + 'final_derain_' + str(ii) + '.jpg'
        name_derain2 = prefix + 'inter_derain_' + str(ii) + '.jpg'
        name_transmission = prefix + 'transmission_' + str(ii) + '.jpg'
        derain_final = derain_final.cpu().data[0]
        derain_inter = derain_inter.cpu().data[0]
        transmission = transmission.cpu().data[0]
        tv.utils.save_image(derain_final, name_derain1)
        tv.utils.save_image(derain_inter, name_derain2)
        tv.utils.save_image(transmission, name_transmission)
        print('Time:', t1-t0)
    Derain_model.train()

if __name__ == '__main__':
    train()
