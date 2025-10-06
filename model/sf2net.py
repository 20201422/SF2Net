"""
@FileName   sf2net
@Author  24
@Date    2024/1/22 23:02
@Version 1.0.0
freedom is the oxygen of the soul.
"""

import torch
import torch.nn.functional as F

from model.vit import ViT
from model.component.arcface import ArcMarginProduct
from model.feature_extraction import FeatureExtraction


# 网络
class SF2Net(torch.nn.Module):
    """网络::
        INPUTS:
            label_num：数据集中标签的个数
            vit_floor_num： vit层数
    """

    def __init__(self, label_num, vit_floor_num, weight=0.8):
        super(SF2Net, self).__init__()

        self.label_num = label_num  # 数据集中标签的个数
        self.vit_floor_num = vit_floor_num  # vit层数

        # 局部特征提取
        self.feature_extraction = FeatureExtraction(channel_in=1, filter_num=36,
                                                    kernel_size=17, stride=2, padding=17 // 2,
                                                    init_ratio=0.5, label_num=self.label_num,
                                                    vit_floor_num=self.vit_floor_num)

        # ViT
        self.vit_0 = ViT(image_size=30, patch_size=5, channels=self.vit_floor_num * 2, num_classes=self.label_num,
                       depth=2, heads=16, dim=128, dim_for_head=64, dim_for_mlp=256, dropout=0.1, emb_dropout=0.1)
        self.vit_1 = ViT(image_size=14, patch_size=2, channels=self.vit_floor_num * 2, num_classes=self.label_num,
                         depth=2, heads=16, dim=128, dim_for_head=64, dim_for_mlp=256, dropout=0.1, emb_dropout=0.1)

        # 全连接层
        self.fully_connection_1 = torch.nn.Linear(7424, 2048)
        self.fully_connection_2 = torch.nn.Linear(2048, 1024)
        self.fully_connection_for_vit_1 = torch.nn.Linear(11136, 4096)
        self.fully_connection_for_vit_2 = torch.nn.Linear(4096, 1024)

        # CNN与ViT加权求和的权重
        self.weight = weight

        # Dropout层
        self.dropout = torch.nn.Dropout(p=0.5)

        # ArcFace（Angular Margin Loss）
        self.arcface = ArcMarginProduct(in_features=1024, out_features=self.label_num)

    def forward(self, feature_tensor, target=None):  # feature_tensor的torch.Size([batch_size, 1, 128, 128])
        # torch.Size([batch_size, 1024])
        feature_tensor = self.processing(feature_tensor)

        # Dropout层  torch.Size([batch_size, 1024])
        feature_tensor = self.dropout(feature_tensor)

        # ArcFace（Angular Margin Loss）  torch.Size([batch_size, label_num])
        feature_tensor = self.arcface(feature_tensor, target)

        return feature_tensor, F.normalize(feature_tensor, dim=-1)

    def get_feature_vector(self, feature_tensor):
        # torch.Size([batch_size, 1024])
        feature_tensor = self.processing(feature_tensor)

        # 返回对输入的特征张量进行L2范数归一化的结果
        return feature_tensor / torch.norm(feature_tensor, p=2, dim=1, keepdim=True)

    def processing(self, feature_tensor):
        # 特征提取
        # torch.Size([batch_size, 7424])
        # torch.Size([batch_size, vit_floor_num*2, 30, 30])
        # torch.Size([batch_size, vit_floor_num*2, 14, 14])
        feature_tensor, first_order_feature_tensor, second_order_feature_tensor \
            = self.feature_extraction(feature_tensor)

        # ViT
        # torch.Size([batch_size, num_patches + 1, dim])
        first_order_feature_tensor = self.vit_0(first_order_feature_tensor)
        # torch.Size([batch_size, num_patches + 1, dim])
        second_order_feature_tensor = self.vit_1(second_order_feature_tensor)

        # torch.Size([batch_size, 2 * (num_patches + 1), dim])
        feature_tensor_for_vit = torch.cat((first_order_feature_tensor, second_order_feature_tensor), dim=1)
        # torch.Size([batch_size, 11136])
        feature_tensor_for_vit = feature_tensor_for_vit.view(feature_tensor_for_vit.shape[0], -1)

        # 全连接层
        feature_tensor = self.fully_connection_1(feature_tensor)  # torch.Size([batch_size, 2048])
        feature_tensor = self.fully_connection_2(feature_tensor)  # torch.Size([batch_size, 1024])
        # torch.Size([batch_size, 2048])
        feature_tensor_for_vit = self.fully_connection_for_vit_1(feature_tensor_for_vit)
        # torch.Size([batch_size, 1024])
        feature_tensor_for_vit = self.fully_connection_for_vit_2(feature_tensor_for_vit)

        # 加权求和
        # torch.Size([batch_size, 1024])
        feature_tensor = feature_tensor * self.weight + feature_tensor_for_vit * (1 - self.weight)

        return feature_tensor


'''  
may the force be with you.
@FileName   sf2net
Created by 24 on 2024/1/22.
'''
