# -*- coding:utf-8 -*-
import os

import PIL
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms as T


class NormSingleROI(object):
    """
    Normalize the input image (exclude the black region) with 0 mean and 1 std.
    [c,h,w]
    """

    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):

        # if not T.functional._is_tensor_image(tensor):
        #     raise TypeError('tensor is not a torch image.')

        c, h, w = tensor.size()

        if c != 1:
            raise TypeError('only support graysclae image.')

        # print(tensor.size)

        tensor = tensor.view(c, h * w)
        idx = tensor > 0
        t = tensor[idx]

        # print(t)
        m = t.mean()
        s = t.std()
        t = t.sub_(m).div_(s + 1e-6)
        tensor[idx] = t

        tensor = tensor.view(c, h, w)

        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats=self.outchannels, dim=0)

        return tensor


class MyDataset(data.Dataset):
    """
    Load and process the ROI images::

    INPUT::
    txt: a text file containing pathes & labels of the input images \n
    transforms: None
    train: True for a training set, and False for a testing set
    imside: the image size of the output image [imside x imside]
    outchannels: 1 for grayscale image, and 3 for RGB image

    OUTPUT::
    [batch, outchannels, imside, imside]
    """

    def __init__(self, txt, transforms=None, train=True, image_size=128, out_channels=1):
        """
            初始化函数，用于创建Dataset对象

            参数：
            txt：str，文本文件路径
            transforms：torchvision.transforms对象，数据预处理的转换操作，默认为None
            train：bool，是否为训练模式，默认为True
            imside：int，图像的尺寸，默认为128
            out_channels：int，输出图像的通道数，默认为1
        """
        self.train = train

        self.image_size = image_size  # 128, 224
        self.out_channels = out_channels  # 1, 3

        self.text_path = txt

        self.transforms = transforms

        if transforms is None:
            if not train:
                # 如果transforms为空且不是训练集，则使用以下预处理操作
                self.transforms = T.Compose([

                    T.Resize(self.image_size),  # 调整图像大小
                    T.ToTensor(),  # 转换为张量
                    NormSingleROI(outchannels=self.out_channels)  # 归一化单个感兴趣区域

                ])
            else:
                # 如果transforms为空且是训练集，则使用以下预处理操作
                self.transforms = T.Compose([

                    T.Resize(self.image_size),  # 调整图像大小
                    T.RandomChoice(transforms=[
                        # 随机调整图像亮度、对比度、饱和度和色相
                        T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),  # 0.3 0.35
                        # 随机裁剪、缩放和调整宽高比图像范围
                        T.RandomResizedCrop(size=self.image_size, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                        # 随机透视变换 透视变换的概率
                        T.RandomPerspective(distortion_scale=0.15, p=1),  # (0.1, 0.2) (0.05, 0.05)
                        T.RandomChoice(transforms=[  # 随机旋转图像
                            T.RandomRotation(degrees=10, interpolation=PIL.Image.BICUBIC, expand=False,
                                             center=(0.5 * self.image_size, 0.0)),
                            T.RandomRotation(degrees=10, interpolation=PIL.Image.BICUBIC, expand=False,
                                             center=(0.0, 0.5 * self.image_size)),
                        ]),
                    ]),

                    T.ToTensor(),  # 转换为张量
                    NormSingleROI(outchannels=self.out_channels)  # 归一化单个感兴趣区域
                ])

        self._read_txt_file()  # 读取文本文件

    def _read_txt_file(self):
        # 初始化图像路径和标签列表
        self.images_path = []
        self.images_label = []

        # 打开文本文件
        txt_file = self.text_path

        with open(txt_file, 'r') as f:
            # 逐行读取文本文件内容
            lines = f.readlines()
            for line in lines:
                # 去除行首尾的空格，并按空格分割字符串
                item = line.strip().split(' ')
                # 将图像路径添加到图像路径列表中
                self.images_path.append(item[0])
                # 将标签添加到标签列表中
                self.images_label.append(item[1])

    def __getitem__(self, index):
        img_path_anchor = self.images_path[index]  # 获取图像路径 锚样本
        label_anchor = self.images_label[index]  # 获取图像标签
        # print(img_path_anchor)

        # 随机选择一个与当前图像标签相同的索引 正样本
        positive_index = np.random.choice(
            np.arange(len(self.images_label))[np.array(self.images_label) == label_anchor])
        # 随机选择一个与当前图像标签不同的索引 负样本
        negative_index = np.random.choice(
            np.arange(len(self.images_label))[np.array(self.images_label) != label_anchor])

        if self.train == True:  # 如果是训练集
            # 重新随机选择一个与当前图像标签相同的索引 正样本
            while positive_index == index:  # 如果随机选择的索引与当前索引相同
                positive_index = np.random.choice(
                    np.arange(len(self.images_label))[np.array(self.images_label) == label_anchor])

            # 重新随机选择一个与当前图像标签不同的索引 负样本
            while negative_index == index:  # 如果随机选择的索引与当前索引相同
                negative_index = np.random.choice(
                    np.arange(len(self.images_label))[np.array(self.images_label) != label_anchor])

        else:  # 如果是测试集，则将索引设置为当前索引 正样本 负样本
            positive_index = index
            negative_index = index

        img_path_positive = self.images_path[positive_index]  # 获取正样本图像的路径

        anchor = Image.open(img_path_anchor).convert('L')  # 打开并转换锚样本图像为灰度图像
        anchor = self.transforms(anchor)  # 对锚样本图像进行预处理

        positive = Image.open(img_path_positive).convert('L')  # 打开并转换正样本图像为灰度图像
        positive = self.transforms(positive)  # 对正样本图像进行预处理

        negative = Image.open(self.images_path[negative_index]).convert('L')  # 打开并转换负样本图像为灰度图像
        negative = self.transforms(negative)  # 对负样本图像进行预处理

        data = [anchor, positive, negative]  # 将三个图像数据组合成一个列表
        label = [int(label_anchor), int(self.images_label[positive_index]), int(self.images_label[negative_index])]
        # print(data)
        # print(label)

        return data, label  # , img_path_anchor  # 返回图像数据和标签，标签转换为整数类型

    # 用于获取数据集的长度
    def __len__(self):
        return len(self.images_path)
