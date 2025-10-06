"""
@FileName   arcface
@Author  24
@Date    2024/1/26 22:35
@Version 1.0.0
freedom is the oxygen of the soul.
"""
import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


# ArcFace（Angular Margin Loss）损失函数，由哈尔滨工业大学的研究人员提出。
# 它是在FaceNet的基础上引入了角度 margin，从而提高了精度和鲁棒性
class ArcMarginProduct(nn.Module):
    """实现大边距弧度距离的类 Implement of large margin arc distance::
        Args:
            in_features: size of each input sample  每个输入样本的大小
            out_features: size of each output sample    每个输出样本的大小
            s: norm of input feature    输入特征的范数
            m: margin   边距

            cos(theta + m)

        From: https://github.com/ronghuaiyang/arcface-pytorch
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数
        self.s = s  # 缩放因子
        self.m = m  # margin参数

        self.weight = Parameter(torch.FloatTensor(self.out_features, in_features))  # 权重参数
        nn.init.xavier_uniform_(self.weight)  # 使用xavier_uniform_方法初始化权重

        self.easy_margin = easy_margin  # 是否使用easy margin
        self.cos_m = math.cos(m)  # margin的余弦值
        self.sin_m = math.sin(m)  # margin的正弦值
        self.threshold = math.cos(math.pi - m)  # margin的阈值
        self.mm = math.sin(math.pi - m) * m  # margin的余弦值与正弦值的乘积

    def forward(self, input, label=None):
        if self.training:
            assert label is not None  # 断言，确保label不为空
            # 使用F.normalize函数对输入向量和权重向量进行归一化，然后使用F.linear函数计算它们之间的余弦相似度
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))  # 计算正弦值

            phi = cosine * self.cos_m - sine * self.sin_m  # 计算phi值

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)  # 根据easy_margin的值选择phi或cosine
            else:
                phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)  # 根据阈值选择phi或cosine

            one_hot = torch.zeros(cosine.size(), device=cosine.device)  # 创建全零的one_hot向量
            # 将label转换为one_hot向量
            # 将标签转换为长整型，并且改变形状为列向量
            # 使用scatter_函数，根据标签的值，在指定的维度上填充值为1。
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # 计算输出值
            output *= self.s  # 缩放输出值
        else:
            # assert label is None
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # 计算余弦相似度
            output = self.s * cosine  # 缩放输出值

        return output  # 返回输出值


'''  
may the force be with you.
@FileName   arcface
Created by 24 on 2024/1/26.
'''
