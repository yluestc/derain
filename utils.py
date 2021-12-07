# coding:utf8
from itertools import chain
import visdom
import torch as t
import time
import torchvision as tv
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def gram_matrix(y):
    """
    输入 b,c,h,w
    输出 b,c,c
    """
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)/(ch*h*w)
    return gram

def atm_light(x, loca):
    """
    the size of x is [b, ch=3, h, w]
    the size of loca is [b, ch=1, h, w]
    """
    x1 = x * loca  #保留loca中非0元素对应位置处的x中的值
    x_mean = t.mean(x1, 1) #the size is [b, h, w]
    [b, h, w] = x_mean.size()
    x_mean = x_mean.view(b, h*w)
    [maxvalue, maxindex] = t.max(x_mean, 1)
    [b1, ch1, h1, w1] = x1.size()
    x2 = x1.view(b1, ch1, h1*w1)
    atm_light_value = t.Tensor(b1, ch1)
    atm_light_loca = t.Tensor(b1, 2)
    for i in range(b1):
        atm_light_value[i, :] = x2[i, :, maxindex[i]]
        index_loca = maxindex[i]
        atm_light_loca[i, 0] = index_loca//w1 + 1
        atm_light_loca[i, 1] = index_loca%w1 + 1 #我们计算得到的位置首行首列都从1开始计数，不是0
    atm_light_value = atm_light_value.unsqueeze(2).unsqueeze(3)
#    atm_light_value = atm_light_value.expand_as(x)

    return atm_light_value, atm_light_loca


def atm_light_haze(x):
    """
    the size of x is [b, ch=3, h, w]
    the size of loca is [b, ch=1, h, w]
    """
    x_mean = t.mean(x, 1) #the size is [b, h, w]
    [b, h, w] = x_mean.size()
    x_mean = x_mean.view(b, h*w)
    [maxvalue, maxindex] = t.max(x_mean, 1)
    [b1, ch1, h1, w1] = x.size()
    x2 = x.view(b1, ch1, h1*w1)
    atm_light_value = t.Tensor(b1, ch1)
    for i in range(b1):
        atm_light_value[i, :] = x2[i, :, maxindex[i]]
    atm_light_value = atm_light_value.unsqueeze(2).unsqueeze(3)
#    atm_light_value = atm_light_value.expand_as(x)

    return atm_light_value



class Visualizer():
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    """

    def __init__(self, env='default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1

    def img(self, name, img_):
        """
        self.img('input_img',t.Tensor(64,64))
        """

        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.image(img_.cpu(),
                       win=name,
                       opts=dict(title=name)
                       )

    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        """
        一个batch的图片转成一个网格图，i.e. input（36，64，64）
        会变成 6*6 的网格图，每个格子大小64*64
        """
        self.img(name, tv.utils.make_grid(
            input_3d.cpu()[0].unsqueeze(1).clamp(max=1, min=0)))

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, name):
        return getattr(self.vis, name)


def get_style_data(path):
    """
    加载风格图片，
    输入： path， 文件路径
    返回： 形状 1*c*h*w， 分布 -2~2
    """
    style_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    style_image = tv.datasets.folder.default_loader(path)
    style_tensor = style_transform(style_image)
    return style_tensor.unsqueeze(0)


def normalize_batch(batch):
    """
    输入: b,ch,h,w  0~255
    输出: b,ch,h,w  -2~2
    """
    mean = batch.data.new(IMAGENET_MEAN).view(1, -1, 1, 1)
    std = batch.data.new(IMAGENET_STD).view(1, -1, 1, 1)
    mean = t.autograd.Variable(mean.expand_as(batch.data))
    std = t.autograd.Variable(std.expand_as(batch.data))
    return (batch / 255.0 - mean) / std

#histogram equalization
def histogram_t(tb):
    totalpixel=0    #用于统计像素总数，即MN
    maptb=[]        #存储映射关系
    count=len(tb)
    for i in range(count):
        totalpixel+=tb[i]
        maptb.append(totalpixel)

    for i in range(count):
        maptb[i]=int(round((maptb[i]*(count-1))/totalpixel))

    def histogram(light):
        return maptb[light]
    return histogram
