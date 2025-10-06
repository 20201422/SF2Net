"""
@FileName   squeeze_and_excitation
@Author  24
@Date    2024/1/26 22:32
@Version 1.0.0
freedom is the oxygen of the soul.
"""
from torch import nn


# SE（Squeeze-and-Excitation）模块，用于增强卷积神经网络（CNN）性能的注意力机制。
# 它通过学习输入特征图的通道间关系，动态地调整通道的重要性，从而提高网络对特定特征的关注度。
# SE模块包括squeeze操作（压缩，即全局平均池化）和excitation操作（激发，即全连接层或卷积层），能够有效地提升网络的表征能力和泛化性能。
class SEModule(nn.Module):
    """SE（Squeeze-and-Excitation）模块::
        INPUTS:
            channel：输入特征图的通道数
            reduction：压缩比例
    """
    def __init__(self, channel, reduction=1):
        super(SEModule, self).__init__()
        # 创建一个空的神经网络模型
        self.se = nn.Sequential(
            # 创建一个自适应平均池化层，用于将输入特征图的空间维度降为1，输出大小为1x1
            nn.AdaptiveAvgPool2d(1),
            # 使用1x1的卷积核对输入的特征图进行卷积操作，将通道数从channel减少到channel // reduction
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            # 创建一个ReLU激活函数层
            nn.ReLU(inplace=True),
            # 使用1x1的卷积核对输入的特征图进行卷积操作，将通道数从channel减少到channel // reduction
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            # 创建一个Sigmoid激活函数层
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x)
        return x * se_weight


'''  
may the force be with you.
@FileName   squeeze_and_excitation
Created by 24 on 2024/1/26.
'''
