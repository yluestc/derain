# coding:utf8
import os
import torch as t
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T


class Rain_Loca(data.Dataset):

    def __init__(self, train_root, label_root, transforms=None):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        train_imgs = [os.path.join(train_root, train_img) for train_img in os.listdir(train_root)]
        label_imgs = [os.path.join(label_root, label_img) for label_img in os.listdir(label_root)]

        train_imgs = sorted(train_imgs, key=lambda x: int(x.split('.')[-2].split('-')[-1]))
        label_imgs = sorted(label_imgs, key=lambda x: int(x.split('.')[-2].split('-')[-1]))

        imgs_num = len(train_imgs)

        self.train_imgs = train_imgs
        self.label_imgs = label_imgs

        if transforms is None:
            normalize_train = T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

            self.transforms_train = T.Compose([
                T.Scale(256),
                T.CenterCrop(256),
                #T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize_train
                ])
            self.transforms_label = T.Compose([
                T.Scale(256),
                T.CenterCrop(256),
                #T.RandomHorizontalFlip(),
                T.ToTensor(),
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        train_img_path = self.train_imgs[index]
        label_img_path = self.label_imgs[index]

        train_data = Image.open(train_img_path)
        label_data = Image.open(label_img_path)

        train_data = self.transforms_train(train_data)
        label_data = self.transforms_label(label_data)

        return train_data, label_data

    def __len__(self):
        return len(self.train_imgs)

class Rain_Test(data.Dataset):

    def __init__(self, rain_root, transforms=None):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        rain_imgs = [os.path.join(rain_root, rain_img) for rain_img in os.listdir(rain_root)]
        rain_imgs = sorted(rain_imgs, key=lambda x: int(x.split('.')[-2].split('-')[-1]))

        imgs_num = len(rain_imgs)

        self.rain_imgs = rain_imgs

        if transforms is None:
            normalize_train = T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

            self.transforms_train = T.Compose([
                T.Scale(512),
                T.CenterCrop(512),
                #T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize_train
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        rain_img_path = self.rain_imgs[index]
        rain_data = Image.open(rain_img_path)
        rain_data = self.transforms_train(rain_data)

        return rain_data

    def __len__(self):
        return len(self.rain_imgs)
