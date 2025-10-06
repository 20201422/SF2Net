"""
@FileName   feature_extraction
@Author  24
@Date    2024/1/26 22:34
@Version 1.0.0
freedom is the oxygen of the soul.
"""

import torch
from torch import nn

from model.component.gabor import GaborConv2d
from model.component.squeeze_and_excitation import SEModule


# 提取序列特征（SFE）
def get_sequence_feature(feature_tensor, vit_floor_num):
    # 对每个位置的响应值进行排序并获取索引
    # torch.Size([batch_size, 64, 30, 30])  torch.Size([batch_size, 64, 14, 14])
    feature_tensor_for_channel = torch.softmax(feature_tensor, dim=1)   # 按照通道排序

    # 提取前两层和后两层的索引
    # torch.Size([batch_size, vit_floor_num, 30, 30])  torch.Size([batch_size, vit_floor_num, 14, 14])
    # 按照通道
    feature_tensor_for_channel_front = feature_tensor_for_channel[:, :vit_floor_num, :, :]
    feature_tensor_for_channel_back = feature_tensor_for_channel[:, (vit_floor_num * -1):, :, :]

    # torch.Size([batch_size, vit_floor_num * 2, 30, 30])  torch.Size([batch_size, vit_floor_num * 2, 14, 14])
    feature_tensor = torch.cat((feature_tensor_for_channel_front, feature_tensor_for_channel_back), dim=1)

    return feature_tensor


# 局部特征提取
class FeatureExtraction(nn.Module):
    """局部特征提取::
        INPUTS:
            channel_in：输入通道数 初始输入只支持1个输入通道
            filter_num：滤波器数量，即经过Gabor滤波器出来后有filter_num种结果（filter_num也代表通道数）
            kernel_size：卷积核大小
            stride：步长
            padding：填充大小
            init_ratio：初始参数（感受野）的缩放因子
            weight_channel：竞争机制的权重（通道权重）
            weight_space：竞争机制的权重（两个空间竞争特征权重）
            channel_out：conv层（卷积层）的输出通道数
    """

    def __init__(self, channel_in, filter_num, kernel_size, stride, padding, init_ratio, label_num, vit_floor_num,
                 channel_out=36):
        super(FeatureExtraction, self).__init__()

        self.channel_in = channel_in  # 输入通道数 只支持1个输入通道
        self.filter_num = filter_num  # 滤波器数量，即经过Gabor滤波器出来后有filter_num种结果（filter_num也代表通道数,输出通道）
        self.kernel_size = kernel_size  # 卷积核大小
        self.stride = stride  # 步长
        self.padding = padding  # 填充大小
        self.init_ratio = init_ratio  # 初始参数（感受野）的缩放因子 初始化比例
        self.label_num = label_num  # 数据集中标签的个数
        self.vit_floor_num = vit_floor_num  # vit层数
        self.channel_out = channel_out  # conv层的输出通道数

        # 多阶纹理Gabor滤波器卷积层 一阶特征适合于渐变纹理，二阶特征对突变变化纹理表现出令人满意的区分能力
        self.gabor_conv2d_1 = GaborConv2d(channel_in=self.channel_in, channel_out=self.filter_num,
                                          kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                          init_ratio=self.init_ratio)
        self.gabor_conv2d_2 = GaborConv2d(channel_in=self.filter_num, channel_out=self.filter_num,
                                          kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                          init_ratio=self.init_ratio)

        # SE（Squeeze-and-Excitation）模块
        self.squeeze_and_excitation = SEModule(channel=self.filter_num)

        # 卷积层
        # 用于处理加工块儿
        self.conv_0 = nn.Conv2d(in_channels=self.filter_num, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv_1 = nn.Conv2d(in_channels=self.filter_num, out_channels=64, kernel_size=5, stride=1, padding=0)

        # 用于二次卷积
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0)

        # 池化层
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, feature_tensor):  # feature_tensor的torch.Size([batch_size, channel_in, 128, 128])
        # 多阶纹理Gabor滤波器卷积层
        # 一阶特征纹理提取 适合于渐变纹理  torch.Size([batch_size, 36, 64, 64])
        first_order_feature_tensor = self.gabor_conv2d_1(feature_tensor)
        # 二阶特征纹理提取 适合于突变变化纹理    torch.Size([batch_size, 36, 32, 32])
        second_order_feature_tensor = self.gabor_conv2d_2(first_order_feature_tensor)

        # 处理加工块儿
        # torch.Size([batch_size, 64, 30, 30])
        first_order_feature_tensor = self.process_block(first_order_feature_tensor, conv=self.conv_0)
        # torch.Size([batch_size, 64, 14, 14])
        second_order_feature_tensor = self.process_block(second_order_feature_tensor, conv=self.conv_1)

        # 二次卷积或池化
        # torch.Size([batch_size, 32, 14, 14])
        f_order_feature_tensor = self.conv_2(first_order_feature_tensor)
        # torch.Size([batch_size, 32, 6, 6])
        s_order_feature_tensor = self.conv_3(second_order_feature_tensor)

        # torch.Size([batch_size, 32 * 14 * 14 + 32 * 6 * 6 = 7424])
        feature_tensor = torch.cat((f_order_feature_tensor.view(f_order_feature_tensor.shape[0], -1),
                                    s_order_feature_tensor.view(s_order_feature_tensor.shape[0], -1)), dim=1)

        # torch.Size([batch_size, vit_floor_num * 2, 30, 30])
        first_order_feature_tensor = get_sequence_feature(first_order_feature_tensor, self.vit_floor_num)
        # torch.Size([batch_size, vit_floor_num * 2, 14, 14])
        second_order_feature_tensor = get_sequence_feature(second_order_feature_tensor, self.vit_floor_num)

        return feature_tensor, first_order_feature_tensor, second_order_feature_tensor

    # 处理加工块儿
    def process_block(self, feature_tensor, conv):

        # SE模块
        # 一阶torch.Size([batch_size, 36, 64, 64])   二阶torch.Size([batch_size, 36, 32, 32])
        feature_tensor = self.squeeze_and_excitation(feature_tensor)

        # 卷积层
        # 一阶torch.Size([batch_size, 64, 60, 60])  二阶torch.Size([batch_size, 64, 28, 28])
        feature_tensor = conv(feature_tensor)

        # 激活函数
        # 一阶torch.Size([batch_size, 64, 60, 60])  二阶torch.Size([batch_size, 64, 28, 28])
        feature_tensor = torch.relu(feature_tensor)

        # 池化层
        # 一阶torch.Size([batch_size, 64, 30, 30])  二阶torch.Size([batch_size, 64, 14, 14])
        feature_tensor = self.max_pool(feature_tensor)

        return feature_tensor


'''  
may the force be with you.
@FileName   feature_extraction
Created by 24 on 2024/1/26.
'''
