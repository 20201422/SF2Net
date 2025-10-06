"""
@FileName   get_data_text
@Author  24
@Date    2024/1/26 03:47
@Version 1.0.0
freedom is the oxygen of the soul.
"""
import os
import shutil

# 根路径
root = './'

# 香港理工大学数据集路径
PolyU_train_path = '/public/home/hpc2420083500103/datasets/PolyU/PalmBigDataBase_zq/'
PolyU_test_path = '/public/home/hpc2420083500103/datasets/PolyU/PalmBigDataBase_zq/'


# 处理香港理工大学掌纹数据集
class GetDataTextForPolyU:
    """处理香港理工大学掌纹数据集::
        INPUTS:
            train_path：训练数据集路径
            test_path：测试数据集路径
            root_path：根路径
    """
    def __init__(self, train_path, test_path, root_path):
        super(GetDataTextForPolyU, self).__init__()
        self.train_path = train_path    # 训练数据集路径
        self.test_path = test_path      # 测试数据集路径

        self.root_path = root_path    # 根路径

        self.save_train_path = self.root_path + 'train_PolyU.txt'   # 训练文件路径
        self.save_test_path = self.root_path + 'test_PolyU.txt'    # 测试文件路径
        self.save_verify_path = self.root_path + 'verify_PolyU.txt'  # 验证文件路径

    # 处理训练和测试文本
    @staticmethod
    def progress_train_and_test(file_path, save_path, num_1, num_2):
        with open(save_path, 'w') as file:
            # 获取train_path目录下的所有文件列表，并按字母顺序排序
            files = os.listdir(file_path)
            files.sort()
            # 遍历train_path目录下的每个文件，拼接出图像的完整路径，从文件名中提取标签
            data = []
            for file_name in files:
                if file_name != '.DS_Store' and num_1 <= int((file_name.split('_')[3]).split('.')[0]) <= num_2:
                    data.append((os.path.join(file_path, file_name), int(file_name.split('_')[2]) - 1))
            # 将图像路径和标签写入文件
            for image_path, label in data:
                file.write('%s %d\n' % (image_path, label))

    # 处理验证文本
    @staticmethod
    def progress_verify(file_path, save_path):
        with open(save_path, 'w') as file:
            # 获取train_path目录下的所有文件列表，并按字母顺序排序
            files = os.listdir(file_path)
            files.sort()
            # 遍历train_path目录下的每个文件，拼接出图像的完整路径，从文件名中提取标签
            for file_name in files:
                if file_name != '.DS_Store':
                    # 将图像路径和标签写入文件
                    file.write('%s %d\n' % (os.path.join(file_path, file_name), int(file_name.split('_')[2]) - 1))

    # 得到数据集文本
    def get_data_text_for_PolyU(self):
        self.progress_train_and_test(self.train_path, self.save_train_path, 1, 5)
        self.progress_train_and_test(self.test_path, self.save_test_path, 6, 13)
        self.progress_verify(self.test_path, self.save_verify_path)


if __name__ == '__main__':

    # 香港理工大学掌纹数据集处理
    (GetDataTextForPolyU(train_path=PolyU_train_path, test_path=PolyU_test_path, root_path=root)
     .get_data_text_for_PolyU())

'''  
may the force be with you.
@FileName   get_data_text
Created by 24 on 2024/1/26.
'''
