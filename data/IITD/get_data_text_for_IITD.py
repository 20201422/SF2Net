"""
@FileName   get_data_text
@Author  24
@Date    2024/1/26 03:47
@Version 1.0.0
freedom is the oxygen of the soul.
"""
import os

import numpy as np
from sklearn.model_selection import KFold

# 根路径
root = './'

# 德里印度理工学院数据集路径
IITD_path_1 = 'D:/Education/master/Datasets/PalmDatasets/IITD/IITD-Palmprint-V1/Segmented/Left/'
IITD_path_2 = 'D:/Education/master/Datasets/PalmDatasets/IITD/IITD-Palmprint-V1/Segmented/Right/'


# 处理德里印度理工学院掌纹数据集
class GetDataTextForIITD:
    """处理德里印度理工学院掌纹数据集::
        INPUTS:
            train_path：训练数据集路径
            test_path：测试数据集路径
            root_path：根路径
    """
    def __init__(self, path_1, path_2, root_path):
        super(GetDataTextForIITD, self).__init__()
        self.path_1 = path_1    # 数据集路径1
        self.path_2 = path_2      # 数据集路径2

        self.root_path = root_path    # 根路径

        self.save_train_path = self.root_path + 'train_IITD.txt'   # 训练文件路径
        self.save_test_path = self.root_path + 'test_IITD.txt'    # 测试文件路径
        self.save_verify_path = self.root_path + 'verify_IITD.txt'  # 验证文件路径

    # 处理训练和测试文本
    @staticmethod
    def progress_train_and_test(file_path_1, file_path_2, save_path, num_1, num_2, type):
        with open(save_path, 'w') as file:
            # 获取目录下的所有文件列表，并按字母顺序排序
            files_1 = os.listdir(file_path_1)
            files_2 = os.listdir(file_path_2)
            files_1.sort()
            files_2.sort()

            # 遍历目录下的每个文件，拼接出图像的完整路径，从文件名中提取标签
            data = []
            for file_name in files_1:
                if num_1 <= int((file_name.split('_')[1]).split('.')[0]) <= num_2:
                    data.append((os.path.join(file_path_1, file_name), int(file_name.split('_')[0]) - 1))
                    # if type == 'train':
                     #    data.append((os.path.join(file_path_1, file_name), int(file_name.split('_')[0]) - 1))
            for file_name in files_2:
                if num_1 <= int((file_name.split('_')[1]).split('.')[0]) <= num_2:
                    data.append((os.path.join(file_path_2, file_name), int(file_name.split('_')[0]) + 230 - 1))
                    # if type == 'train':
                     #    data.append((os.path.join(file_path_1, file_name), int(file_name.split('_')[0]) - 1))
            # 将图像路径和标签写入文件
            for image_path, label in data:
                file.write('%s %d\n' % (image_path, label))

    # 处理验证文本
    @staticmethod
    def progress_verify(file_path_1, file_path_2, save_verify_path):
        with open(save_verify_path, 'w') as file:
            # 获取目录下的所有文件列表，并按字母顺序排序
            files_1 = os.listdir(file_path_1)
            files_2 = os.listdir(file_path_2)
            files_1.sort()
            files_2.sort()

            # 遍历目录下的每个文件，拼接出图像的完整路径，从文件名中提取标签
            data = []
            for file_name in files_1:
                data.append((os.path.join(file_path_1, file_name), int(file_name.split('_')[0]) - 1))
            for file_name in files_2:
                data.append((os.path.join(file_path_2, file_name), int(file_name.split('_')[0]) + 230 - 1))

            # 将图像路径和标签写入文件
            for image_path, label in data:
                file.write('%s %d\n' % (image_path, label))

    # 5折交叉验证法实现训练集和测试集的储存
    @staticmethod
    def progress_kfold_cross_validation(file_path_1, file_path_2, save_train_path, save_test_path, num_folds=5):
        # 获取file_path_1和file_path_2目录下的所有文件列表，并按字母顺序排序
        files_1 = os.listdir(file_path_1)
        files_2 = os.listdir(file_path_2)
        files_1.sort()
        files_2.sort()

        # 创建一个字典，将文件按类别存储
        data = {}
        for file_name in files_1:
            label = int(file_name.split('_')[0]) - 1
            if label not in data:
                data[label] = []
            data[label].append(os.path.join(file_path_1, file_name))

        for file_name in files_2:
            label = int(file_name.split('_')[0]) + 230 - 1
            if label not in data:
                data[label] = []
            data[label].append(os.path.join(file_path_2, file_name))

        # 对每个类别的数据进行5折交叉验证
        # 使用KFold进行交叉验证，设置分割数量为num_folds，数据洗牌，随机种子为42
        # 42 作为种子值本身并没有特别的意义。它在数据科学和机器学习社区中变得流行，主要是因为它出现在了《银河系漫游指南》这本科幻小说中。
        # 在这本书中，计算机“深思”经过长时间的计算后得出了“42”作为生命、宇宙及一切问题的答案。
        # 虽然这个解释是幽默的，但在实际应用中，42 已成为常见的随机种子值。
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        for fold, (train_indices, test_indices) in enumerate(kf.split(range(5))):    # 遍历每一折
            # 打开训练集和测试集的保存路径，准备写入
            with open(save_train_path, 'w') as train_file, open(save_test_path, 'w') as test_file:
                for label, paths in data.items():    # 遍历数据集中的每个标签和对应的路径
                    # 将路径数组索引为训练集和测试集
                    train_paths = np.array(paths)[train_indices]
                    test_paths = np.array(paths)[test_indices]

                    # 遍历训练集路径，写入训练文件
                    for train_path in train_paths:
                        train_file.write(f'{train_path} {label}\n')

                    # 遍历测试集路径，写入测试文件
                    for test_path in test_paths:
                        test_file.write(f'{test_path} {label}\n')

    # 得到数据集文本
    def get_data_text_for_IITD(self):
        # 由于该数据集整齐处理后每个类别仅有5张ROI图像，所以训练集为前2张，测试集为后4张，其中训练集每张多储存一张
        self.progress_train_and_test(self.path_1, self.path_2, self.save_train_path, 1, 3, 'train')
        self.progress_train_and_test(self.path_1, self.path_2, self.save_test_path, 4, 5, 'test')
        self.progress_verify(self.path_1, self.path_2, self.save_verify_path)
        # self.progress_kfold_cross_validation(self.path_1, self.path_2, self.save_train_path, self.save_test_path)


if __name__ == '__main__':

    # 德里印度理工学院掌纹数据集处理
    (GetDataTextForIITD(path_1=IITD_path_1, path_2=IITD_path_2, root_path=root)
     .get_data_text_for_IITD())


'''  
may the force be with you.
@FileName   get_data_text
Created by 24 on 2024/1/26.
'''
