# coding:utf8
import os
import torch as t
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import random
from Dataset_standard.Data_Transforms import RandomCrop1
import torch.nn.functional as F
import math
import torch as t


class Train_Data(data.Dataset):

    def __init__(self, train_root, label_root, crop_size=512, transforms=None):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        train_imgs = [os.path.join(train_root, train_img) for train_img in os.listdir(train_root)]
        label_imgs = [os.path.join(label_root, label_img) for label_img in os.listdir(label_root)]

        train_imgs = sorted(train_imgs, key=lambda x: int(x.split('/')[-1].split('.')[-2]))
        label_imgs = sorted(label_imgs, key=lambda x: int(x.split('/')[-1].split('.')[-2]))

        imgs_num = len(train_imgs)

        self.train_imgs = train_imgs
        self.label_imgs = label_imgs
        self.crop_size = crop_size
        self.fill = 0
        self.padding_mode = 'reflect'
        self.num = imgs_num

        if transforms is None:
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

            self.transforms_input = T.Compose([
                T.Scale(self.crop_size),
                RandomCrop1(self.crop_size, 0),
                #                T.CenterCrop(self.crop_size),
                #                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])

            self.transforms_input_no_norm = T.Compose([
                T.Scale(self.crop_size),
                RandomCrop1(self.crop_size, 0),
                #                T.CenterCrop(self.crop_size),
                #                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])

            self.transforms_label = T.Compose([
                T.Scale(self.crop_size),
                RandomCrop1(self.crop_size, 0),
                #                T.CenterCrop(self.crop_size),
                #                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])

    def __getitem__(self, index):
        train_img_path = self.train_imgs[index % self.num]
        label_img_path = self.label_imgs[index % self.num]

        train_data = Image.open(train_img_path)
        label_data = Image.open(label_img_path)

        w, h = train_data.size  #return the (width, height) of an image

        pad_w = max(0, math.ceil((self.crop_size - w)/2.0))
        pad_h = max(0, math.ceil((self.crop_size - h) / 2.0))

        train_data = np.asarray(train_data)
        label_data = np.asarray(label_data)

        train_data = np.pad(train_data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), self.padding_mode)  #这里的pad操作保证了pad之后样本的大小一定是大于要截取的patch的大小。
        label_data = np.pad(label_data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), self.padding_mode)

        train_data = Image.fromarray(train_data)
        label_data = Image.fromarray(label_data)

        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)  #set the starting point of generaing a random number.
        train = self.transforms_input(train_data)
        random.seed(manualSeed)
        train_no_norm = self.transforms_input_no_norm(train_data)  # 没有规范化
        random.seed(manualSeed)  #set the starting point of generaing a random number.
        label = self.transforms_label(label_data)

        return train_no_norm, train, label

    def __len__(self):
        return 6*len(self.train_imgs)


class Train_Data1(data.Dataset):

    def __init__(self, train_root, label_root, crop_size=512, transforms=None):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        train_imgs = [os.path.join(train_root, train_img) for train_img in os.listdir(train_root)] #snowy data has the same name as its GT, but different directory.
        #label_imgs = [os.path.join(label_root, label_img) for label_img in os.listdir(label_root)]

        #train_imgs = sorted(train_imgs, key=lambda x: int(x.split('.')[-2].split('-')[-1]))
        #label_imgs = sorted(label_imgs, key=lambda x: int(x.split('.')[-2].split('-')[-1]))

        #imgs_num = len(train_imgs)

        self.train_imgs = train_imgs
        #self.label_imgs = label_imgs
        self.label_root = label_root
        self.crop_size = crop_size
        self.padding_mode = 'reflect'

        if transforms is None:
            normalize_train = T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

            self.transforms_input = T.Compose([
                #T.Scale(256),
                #T.CenterCrop(256),
                RandomCrop1(self.crop_size, 0),
                #T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize_train
                ])

            self.transforms_input_no_norm = T.Compose([
                #                T.Scale(self.crop_size),
                RandomCrop1(self.crop_size, 0),
                #                T.CenterCrop(self.crop_size),
                #                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])

            self.transforms_label = T.Compose([
                #T.Scale(256),
                #T.CenterCrop(256),
                RandomCrop1(self.crop_size, 0),
                #T.RandomHorizontalFlip(),
                T.ToTensor(),
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        train_img_path = self.train_imgs[index]
        label_img_path = os.path.join(self.label_root, train_img_path.split('/')[-1])

        input_data = Image.open(train_img_path)
        label_data = Image.open(label_img_path)
        w, h = input_data.size
        w1 = w
        h1 = h
        pad_w = max(0, math.ceil((self.crop_size - w1) / 2.0))
        pad_h = max(0, math.ceil((self.crop_size - h1) / 2.0))

        input_data = np.asarray(input_data)
        label_data = np.asarray(label_data)

        input_data = np.pad(input_data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                            self.padding_mode)  # 这里的pad操作保证了pad之后样本的大小一定是大于要截取的patch的大小。
        label_data = np.pad(label_data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), self.padding_mode)

        input_data = Image.fromarray(input_data)
        label_data = Image.fromarray(label_data)

        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)  # set the starting point of generaing a random number.
        input = self.transforms_input(input_data)
        random.seed(manualSeed)
        input_no_norm = self.transforms_input_no_norm(input_data)  # 没有规范化
        random.seed(manualSeed)  # set the starting point of generaing a random number.
        label = self.transforms_label(label_data)


        return input_no_norm, input, label

    def __len__(self):
        return len(self.train_imgs)


class Test_Data(data.Dataset):

    def __init__(self, train_root, label_root, crop_size=512, transforms=None):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        train_imgs = [os.path.join(train_root, train_img) for train_img in os.listdir(train_root)]
        label_imgs = [os.path.join(label_root, label_img) for label_img in os.listdir(label_root)]

        train_imgs = sorted(train_imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        label_imgs = sorted(label_imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))

        self.train_imgs = train_imgs
        self.label_imgs = label_imgs
        self.crop_size = crop_size
        self.fill = 0
        self.padding_mode = 'reflect'

        if transforms is None:
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

            self.transforms_input = T.Compose([
                #                T.Scale(self.crop_size),
                #RandomCrop1(self.crop_size, 0),
                T.CenterCrop(self.crop_size),
                #                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])

            self.transforms_input_no_norm = T.Compose([
                #                T.Scale(self.crop_size),
                #RandomCrop1(self.crop_size, 0),
                T.CenterCrop(self.crop_size),
                #                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])

            self.transforms_label = T.Compose([
                #                T.Scale(self.crop_size),
                #RandomCrop1(self.crop_size, 0),
                T.CenterCrop(self.crop_size),
                #                T.RandomHorizontalFlip(),
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

        w, h = train_data.size  # return the (width, height) of an image

        pad_w = max(0, math.ceil((self.crop_size - w) / 2.0))
        pad_h = max(0, math.ceil((self.crop_size - h) / 2.0))

        train_data = np.asarray(train_data)
        label_data = np.asarray(label_data)

        train_data = np.pad(train_data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                            self.padding_mode)  # 这里的pad操作保证了pad之后样本的大小一定是大于要截取的patch的大小。
        label_data = np.pad(label_data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), self.padding_mode)

        train_data = Image.fromarray(train_data)
        label_data = Image.fromarray(label_data)

        train = self.transforms_input(train_data)
        train_no_norm = self.transforms_input_no_norm(train_data)  # 没有规范化
        label = self.transforms_label(label_data)

        return train_no_norm, train, label

    def __len__(self):
        return len(self.train_imgs)


class Test_Data1(data.Dataset):

    def __init__(self, train_root, label_root, crop_size, transforms=None):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        train_imgs = [os.path.join(train_root, train_img) for train_img in os.listdir(train_root)]
        label_imgs = [os.path.join(label_root, label_img) for label_img in os.listdir(label_root)]

        train_imgs = sorted(train_imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        label_imgs = sorted(label_imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))

        self.train_imgs = train_imgs
        self.label_imgs = label_imgs
        self.crop_size = crop_size
        self.fill = 0
        self.padding_mode = 'reflect'

        if transforms is None:
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

            self.transforms_input = T.Compose([
                T.Scale(self.crop_size),
                #RandomCrop1(self.crop_size, 0),
                T.CenterCrop(self.crop_size),
                #                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])

            self.transforms_input_no_norm = T.Compose([
                T.Scale(self.crop_size),
                #RandomCrop1(self.crop_size, 0),
                T.CenterCrop(self.crop_size),
                #                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])

            self.transforms_label = T.Compose([
                T.Scale(self.crop_size),
                #RandomCrop1(self.crop_size, 0),
                T.CenterCrop(self.crop_size),
                #                T.RandomHorizontalFlip(),
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

        train = self.transforms_input(train_data)
        train_no_norm = self.transforms_input_no_norm(train_data)  # 没有规范化
        label = self.transforms_label(label_data)

        return train_no_norm, train, label

    def __len__(self):
        return len(self.train_imgs)


class Test_Data3(data.Dataset):

    def __init__(self, train_root, label_root, crop_size=512, transforms=None):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        train_imgs = [os.path.join(train_root, train_img) for train_img in os.listdir(train_root)] #snowy data has the same name as its GT, but different directory.
        #label_imgs = [os.path.join(label_root, label_img) for label_img in os.listdir(label_root)]

        #train_imgs = sorted(train_imgs, key=lambda x: int(x.split('.')[-2].split('-')[-1]))
        #label_imgs = sorted(label_imgs, key=lambda x: int(x.split('.')[-2].split('-')[-1]))

        #imgs_num = len(train_imgs)

        self.train_imgs = train_imgs
        #self.label_imgs = label_imgs
        self.label_root = label_root
        self.crop_size = crop_size
        self.padding_mode = 'reflect'

        if transforms is None:
            normalize_train = T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

            self.transforms_input = T.Compose([
                #T.Scale(256),
                T.CenterCrop(256),
                #RandomCrop1(self.crop_size, 0),
                #T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize_train
                ])

            self.transforms_input_no_norm = T.Compose([
                #                T.Scale(self.crop_size),
                #RandomCrop1(self.crop_size, 0),
                T.CenterCrop(self.crop_size),
                #                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])

            self.transforms_label = T.Compose([
                #T.Scale(256),
                T.CenterCrop(256),
                #RandomCrop1(self.crop_size, 0),
                #T.RandomHorizontalFlip(),
                T.ToTensor(),
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        train_img_path = self.train_imgs[index]
        label_name = train_img_path.split('/')[-1].split('.')[-2] + 'gt.' + train_img_path.split('/')[-1].split('.')[-1]
        label_img_path = os.path.join(self.label_root, label_name)

        input_data = Image.open(train_img_path)
        label_data = Image.open(label_img_path)
        #w, h = input_data.size
        #w1 = w
        #h1 = h
        #pad_w = max(0, math.ceil((self.crop_size - w1) / 2.0))
        #pad_h = max(0, math.ceil((self.crop_size - h1) / 2.0))

        #input_data = np.asarray(input_data)
        #label_data = np.asarray(label_data)

        #input_data = np.pad(input_data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        #                    self.padding_mode)  # 这里的pad操作保证了pad之后样本的大小一定是大于要截取的patch的大小。
        #label_data = np.pad(label_data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), self.padding_mode)

        #input_data = Image.fromarray(input_data)
        #label_data = Image.fromarray(label_data)

        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)  # set the starting point of generaing a random number.
        input = self.transforms_input(input_data)
        random.seed(manualSeed)
        input_no_norm = self.transforms_input_no_norm(input_data)  # 没有规范化
        random.seed(manualSeed)  # set the starting point of generaing a random number.
        label = self.transforms_label(label_data)


        return input_no_norm, input, label

    def __len__(self):
        return len(self.train_imgs)


class Test_Practical(data.Dataset):

    def __init__(self, root, transforms=None):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        pra_imgs = [os.path.join(root, pra_img) for pra_img in os.listdir(root)]
        #pra_imgs = sorted(pra_imgs, key=lambda x: int(x.split('.')[-2].split('-')[-1]))

        self.pra_imgs = pra_imgs

        if transforms is None:
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

            self.transforms = T.Compose([
                #T.Scale(512),
                #T.CenterCrop(512),
                #T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
                ])
            self.transforms_no_norm = T.Compose([
                # T.Scale(512),
                #T.CenterCrop(512),
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.pra_imgs[index]
        pra_data = Image.open(img_path)
        pra_test = self.transforms(pra_data)
        pra_test_no_norm = self.transforms_no_norm(pra_data)

        return pra_test_no_norm, pra_test

    def __len__(self):
        return len(self.pra_imgs)
