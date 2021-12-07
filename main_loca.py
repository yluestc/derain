from __future__ import print_function
import torch as t
import string
import torchvision as tv
import torchnet as tnt
from torch.utils import data
from torch.autograd import Variable
from torch.nn import functional as F
import os
import ipdb
import argparse
import sys
import random
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True   #it enable cudnn to find the optimal algorithms automatically.
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from misc import *
from torchnet import meter   #value the performance of a method
from torch.utils.data import DataLoader
from Loca_model.Multi_stream_dense_net_loca import Multi_stream_dense_net
from Dataset_standard.Dataset_loca import Rain_Loca
from Dataset_standard.Dataset_loca import Rain_Test
import init
from tensorboardX import SummaryWriter
from PackedVGG import Vgg16
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_root', required=False,
  default='Dataset_standard/Rain_loca_train/rain_train/', help='path to original training dataset')
parser.add_argument('--train_label_root', required=False,
  default='Dataset_standard/Rain_loca_train/label_train/', help='path to original label dataset')
parser.add_argument('--test_root', required=False,
  default='test_data/loca_data/', help='path to test rain dataset')
parser.add_argument('--result_path', required=False,
  default='./loca_results/', help='path to loca results')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--max_epoch', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate, default=0.0002')
parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay')
parser.add_argument('--weight_decay', type=float, default=0e-5, help='weight_decay')
parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
parser.add_argument('--model_save_freq', type=int, default=10, help='model save frequency')
parser.add_argument('--writer_freq', type=int, default=50, help='writer frequency')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp_loca', default='./checkpoints_loca/', help='folder to output model_loca checkpoints')
parser.add_argument('--load_model_path', default=None, help='load the pretrained loca_model')
parser.add_argument('--debug_file', default='/tmp/debug', help='the path used for debug')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
opt = parser.parse_args()
print(opt)

create_exp_dir(opt.exp_loca)

def train():

    # step1: configure model
    loca_model = Multi_stream_dense_net()
    print(loca_model)
    init.init_weights(loca_model, init_type='kaiming')
    if opt.load_model_path:
        loca_model.load_state_dict(t.load(opt.load_model_path))
    if opt.cuda: loca_model.cuda()

#    vgg = Vgg16().eval()
#    if opt.cuda: vgg.cuda()

    # step2: data
    train_data = Rain_Loca(opt.train_data_root,opt.train_label_root)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)

    # step3: criterion and optimizer
    criterion = t.nn.MSELoss()
    lr = opt.lr
    optimizer = t.optim.Adam(loca_model.parameters(),lr = lr,weight_decay = opt.weight_decay)

    # step4: meters
    loss_meter = meter.AverageValueMeter()

    previous_loss = 1e5
    writer = SummaryWriter('./logdir_loca/')  #the root where the data is saved.
    # train
    n_iter = 0
    for epoch in range(opt.max_epoch):

        loss_meter.reset()

        for ii,(data,label) in enumerate(train_dataloader):

            # 进入debug模式
            if os.path.exists(opt.debug_file):
                import ipdb;
                ipdb.set_trace()

            # train model
            input = Variable(data)
            target = Variable(label)
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            loca = loca_model(input)

            content_loss = criterion(loca,target)

            loss = content_loss

            # meters update and visualize
            loss_meter.add(loss.data)# meters update and visualize

            loss.backward()
            optimizer.step()

            #tensorboard writer
            if ii % opt.writer_freq==opt.writer_freq-1:
                n_iter = epoch * len(train_dataloader) + ii
                writer.add_scalar('loss', loss_meter.value()[0], n_iter)
                writer.add_image('Location', loca, n_iter)

            if ii % opt.print_freq==opt.print_freq-1:
                print (loss_meter.value()[0], content_loss, lr, ii, epoch)

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]

        if epoch % opt.model_save_freq==opt.model_save_freq-1:
            prefix = './checkpoints_loca/'
            name = prefix + 'Loca-' + 'epoch' + str(epoch) + '.pth'
            t.save(loca_model.state_dict(), name)

        # validate and visualize

#        val_cm = val(loca_model,val_dataloader)
#        print val_cm

def test():
    # 进入debug模式
    if os.path.exists(opt.debug_file):
        import ipdb;
        ipdb.set_trace()

    # step1: configure model
    loca_model = Multi_stream_dense_net()
    loca_model.eval()
    if opt.load_model_path:
        loca_model.load_state_dict(t.load(opt.load_model_path))
    if opt.cuda:
        loca_model.cuda()

    # step2: data
    test_data = Rain_Test(opt.test_root)
    test_dataloader = DataLoader(test_data, opt.batch_size,
                        num_workers=opt.num_workers)

    for ii, data in enumerate(test_dataloader):

    # test model
        input = Variable(data)
        if opt.cuda:
            input = input.cuda()

        with torch.no_grad():
            loca = loca_model(input)

        prefix = opt.result_path
        name = prefix + 'loca_' + str(ii+1) + '.jpg'
#        loca = loca.mul(255)
#        loca = tv.transforms.ToPILImage()(loca)
        tv.utils.save_image(loca, name)

    loca_model.train()


def val(model,dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    loss_meter = meter.AverageValueMeter()
    for ii, data in tqdm(enumerate(dataloader)):
        input, label = data
        val_input = Variable(input, volatile=True)
        target = Variable(label)
        if opt.use_gpu:
            val_input = val_input.cuda()
        loca = loca_model(val_input)
        loss = criterion(loca,target)
        loss_meter.add(loss.data)

    model.train()   #translate the state of model to train() not eval()
    cm_value = loss_meter.value()
    return cm_value

if __name__ == '__main__':
    test()
