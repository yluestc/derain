# coding:utf8
import os
import torch as t
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import random
import math
from Dataset_standard.Data_Transforms import RandomCrop1

IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

#produce the absolute path for open specific images
def make_dataset(dir):
    images = []
    if not os.path.isdir(dir):
        raise Exception('Check dataroot')
    for root, _, fnames in sorted(os.walk(dir)):  #different from os.listdir()
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)   #maybe a problem exists, dir should be substituted with root????
                item = path
                images.append(item) #append the path of every image.
    return images

class Train_Test_Data(data.Dataset):

    def __init__(self, root, transforms=None):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        imgs = make_dataset(root)
        if len(imgs) == 0:
          raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))

        self.imgs = imgs

        if transforms is None:
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

            self.transforms_input = T.Compose([
                T.Scale(256),
#                T.RandomCrop((256, 256)),
                T.CenterCrop(256),
#                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
                ])

            self.transforms_label = T.Compose([
                T.Scale(256),
#                T.RandomCrop((256, 256)),
                T.CenterCrop(256),
#                T.RandomHorizontalFlip(),
                T.ToTensor(),
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]

        img = Image.open(img_path)

        w, h = img.size  #return the (width, height) of an image

#        manualSeed = random.randint(1, 10000)

        input_data = img.crop((0, 0, w/2, h))  #crop(left, up, right, down)
        label_data = img.crop((w/2, 0, w, h))
#        random.seed(manualSeed)  #set the starting point of generaing a random number.
        input_light = self.transforms_label(input_data)
        input = self.transforms_input(input_data)
#        random.seed(manualSeed)  #set the starting point of generaing a random number.
        label = self.transforms_label(label_data)

        return input_light, input, label

    def __len__(self):
        return len(self.imgs)


# crop image patches from original rainy images, no scale operation.
class Train_Test_Data2(data.Dataset):

    def __init__(self, root, crop_size=256, transforms=None):
        imgs = make_dataset(root)
        if len(imgs) == 0:
          raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))

        self.imgs = imgs
        self.crop_size = crop_size
        self.fill = 0
        self.padding_mode = 'reflect'

        if transforms is None:
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

            self.transforms_input = T.Compose([
                #                T.Scale(self.crop_size),
                RandomCrop1(self.crop_size, 0),
                # T.CenterCrop(self.crop_size),
                #                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])

            self.transforms_input_no_norm = T.Compose([
                #                T.Scale(self.crop_size),
                RandomCrop1(self.crop_size, 0),
                # T.CenterCrop(self.crop_size),
                #                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])

            self.transforms_label = T.Compose([
                #                T.Scale(self.crop_size),
                RandomCrop1(self.crop_size, 0),
                # T.CenterCrop(self.crop_size),
                #                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        img = Image.open(img_path)
        w, h = img.size  #return the (width, height) of an image

        input_data = img.crop((0, 0, w/2, h))  #crop(left, up, right, down)
        label_data = img.crop((w/2, 0, w, h))

        w1 = w/2
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
        return len(self.imgs)



#for Li_ruoteng dataset
class Train_Test_Data1(data.Dataset):

    def __init__(self, root, crop_size=256, transforms=None):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.root = root
        #gt_root = root
        #self.gt_root = gt_root.replace("/in", "")  #注意要用双引号，不能用单引号，否则不起作用
        img_root = self.root + "in"
        imgs = make_dataset(img_root)
        if len(imgs) == 0:
          raise(RuntimeError("Found 0 images in subfolders of: " + img_root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        #imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))

        self.imgs = imgs
        self.crop_size = crop_size
        self.fill = 0
        self.padding_mode = 'reflect'

        if transforms is None:
            normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

            self.transforms_input = T.Compose([
#                T.Scale(self.crop_size),
                RandomCrop1(self.crop_size, 0),
                #T.CenterCrop(self.crop_size),
#                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
                ])

            self.transforms_input_no_norm = T.Compose([
                #                T.Scale(self.crop_size),
                RandomCrop1(self.crop_size, 0),
                #T.CenterCrop(self.crop_size),
                #                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])

            self.transforms_label = T.Compose([
#                T.Scale(self.crop_size),
                RandomCrop1(self.crop_size, 0),
                #T.CenterCrop(self.crop_size),
#                T.RandomHorizontalFlip(),
                T.ToTensor(),
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        gt_name = img_path.split('.')[-2].split('_')[-4].split('/')[-1] + '_' + img_path.split('.')[-2].split('_')[-3] + '.' + img_path.split('.')[-1]
        gt_path = self.root + 'gt' + '/' + gt_name

        input_data = Image.open(img_path)
        label_data = Image.open(gt_path)

        w, h = input_data.size  #return the (width, height) of an image

        w1 = w
        h1 = h

        pad_w = max(0, math.ceil((self.crop_size - w1)/2.0))
        pad_h = max(0, math.ceil((self.crop_size - h1) / 2.0))

        input_data = np.asarray(input_data)
        label_data = np.asarray(label_data)

        input_data = np.pad(input_data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), self.padding_mode)  #这里的pad操作保证了pad之后样本的大小一定是大于要截取的patch的大小。
        label_data = np.pad(label_data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), self.padding_mode)

        input_data = Image.fromarray(input_data)
        label_data = Image.fromarray(label_data)

        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)  #set the starting point of generaing a random number.
        input = self.transforms_input(input_data)
        random.seed(manualSeed)
        input_no_norm = self.transforms_input_no_norm(input_data)  # 没有规范化
        random.seed(manualSeed)  #set the starting point of generaing a random number.
        label = self.transforms_label(label_data)

        return input_no_norm, input, label

    def __len__(self):
        return len(self.imgs)


class Test_Practical(data.Dataset):

    def __init__(self, root, transforms=None):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('-')[-1]))

        imgs_num = len(imgs)

        self.imgs = imgs

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
            self.transforms_label = T.Compose([
                #T.Scale(512),
#                T.RandomCrop((256, 256)),
                #T.CenterCrop(512),
#                T.RandomHorizontalFlip(),
                T.ToTensor(),
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        data = Image.open(img_path)
        #w, h = rain_data.size   #original width and height of an image
        #rain_data = rain_data.resize((512, 512))
        input = self.transforms(data)
        input_light = self.transforms_label(data)

        return input_light, input

    def __len__(self):
        return len(self.imgs)
