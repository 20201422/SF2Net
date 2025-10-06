"""
@FileName   gabor
@Author  24
@Date    2024/1/26 22:30
@Version 1.0.0
freedom is the oxygen of the soul.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


# 滤波卷积层 Learnable Gabor Convolution (LGC) Layer
class GaborConv2d(nn.Module):
    """滤波卷积层 Learnable Gabor Convolution (LGC) Layer::
        INPUTS:
            channel_in：输入通道数
            channel_out：输出通道数
            kernel_size：卷积核大小
            stride：步长
            padding：填充大小
            init_ratio：初始参数（感受野）的缩放因子
    """
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super(GaborConv2d, self).__init__()
        self.channel_in = channel_in  # 输入通道数
        self.channel_out = channel_out  # 输出通道数

        self.kernel_size = kernel_size  # 卷积核大小
        self.stride = stride  # 步长
        self.padding = padding  # 填充大小
        # 初始参数（感受野）的缩放因子，用于调整感受野的大小，即影响卷积层对输入数据的感知范围。
        # 通过调整比例因子，可以改变卷积层对输入数据的敏感度和感知范围，从而影响特征提取的效果。
        self.init_ratio = init_ratio    # 初始化比例

        self.kernel = 0  # 卷积核

        self.SIGMA = 9.2 * self.init_ratio  # 高斯函数的尺度
        self.GAMMA = 2.0  # 高斯函数的形状参数
        # 余弦包络的频率，用于调整余弦波的周期，从而影响信号的振荡频率。
        # 在信号处理中，调整余弦包络的频率可以改变信号的周期性和频率特性，对于滤波和调制等操作具有重要作用。
        self.FREQUENCY = 0.057 / self.init_ratio

        # 可训练的高斯函数的形状和尺度
        # 使用torch.FloatTensor()函数创建一个包含单个元素的张量
        # 使用nn.Parameter()函数将该张量转换为可训练的参数，并设置requires_grad=True，表示该参数需要计算梯度
        # 高斯函数的尺度
        self.sigma = nn.Parameter(torch.FloatTensor([self.SIGMA]), requires_grad=True)
        # 高斯函数的形状参数
        self.gamma = nn.Parameter(torch.FloatTensor([self.GAMMA]), requires_grad=True)
        # 余弦包络的相位参数，根据输出通道数生成，初始值为0到channel_out之间的等差数列，每个值乘以π除以channel_out
        self.theta = nn.Parameter(torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
                                  requires_grad=False)
        # 余弦包络的频率
        self.frequency = nn.Parameter(torch.FloatTensor([self.FREQUENCY]), requires_grad=True)
        # 余弦包络的相位偏移
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def forward(self, x):

        self.kernel = self.get_gabor()

        out = F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)

        return out

    def get_gabor(self):
        x_max = self.kernel_size // 2
        y_max = self.kernel_size // 2
        x_min = -x_max
        y_min = -y_max

        k_size = x_max - x_min + 1
        x_0 = torch.arange(x_min, x_max + 1).float()  # 创建一个从x_min到x_max的浮点数序列
        y_0 = torch.arange(y_min, y_max + 1).float()  # 创建一个从y_min到y_max的浮点数序列

        # [channel_out, channel_in, kernel_H, kernel_W]
        # 将x_0的维度变为[len(x_0), 1]，然后将其复制channel_out * channel_in次，最后将维度变为[channel_out, channel_in, len(x_0), ksize]
        x = x_0.view(-1, 1).repeat(self.channel_out, self.channel_in, 1, k_size)
        # 将y_0的维度变为[1, len(y_0)]，然后将其复制channel_out * channel_in次，最后将维度变为[channel_out, channel_in, ksize, len(y_0)]
        y = y_0.view(1, -1).repeat(self.channel_out, self.channel_in, k_size, 1)

        x = x.float().to(self.sigma.device)  # 将x转换为浮点数类型，并将其放置在与sigma相同的设备上
        y = y.float().to(self.sigma.device)  # 将y转换为浮点数类型，并将其放置在与sigma相同的设备上

        x_theta = x * torch.cos(self.theta.view(-1, 1, 1, 1)) + y * torch.sin(
            self.theta.view(-1, 1, 1, 1))  # 计算旋转后的x坐标
        y_theta = -x * torch.sin(self.theta.view(-1, 1, 1, 1)) + y * torch.cos(
            self.theta.view(-1, 1, 1, 1))  # 计算旋转后的y坐标

        # 计算Gabor滤波器响应
        gabor = -torch.exp(
            -0.5 * ((self.gamma * x_theta) ** 2 + y_theta ** 2) / (8 * self.sigma.view(-1, 1, 1, 1) ** 2)) \
                * torch.cos(2 * math.pi * self.frequency.view(-1, 1, 1, 1) * x_theta +
                            self.psi.view(-1, 1, 1, 1))

        # 对gb在第2和第3维度上求均值，并将结果减去gb的均值
        gabor = gabor - gabor.mean(dim=[2, 3], keepdim=True)

        return gabor


'''  
may the force be with you.
@FileName   gabor
Created by 24 on 2024/1/26.
'''
