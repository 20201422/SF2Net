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

# 多光谱（红光）数据集路径
MSRed_train_path = '/public/home/hpc2420083500103/datasets/MSPalm/Red/'
MSRed_test_path = '/public/home/hpc2420083500103/datasets/MSPalm/Red/'


# 处理多光谱（红光）掌纹数据集
class GetDataTextForMSRed:
    """处理多光谱（红光）掌纹数据集::
        INPUTS:
            train_path：训练数据集路径
            test_path：测试数据集路径
            root_path：根路径
    """
    def __init__(self, train_path, test_path, root_path):
        super(GetDataTextForMSRed, self).__init__()
        self.train_path = train_path    # 训练数据集路径
        self.test_path = test_path      # 测试数据集路径

        self.root_path = root_path    # 根路径

        self.save_train_path = self.root_path + 'train_MSRed.txt'   # 训练文件路径
        self.save_test_path = self.root_path + 'test_MSRed.txt'    # 测试文件路径
        self.save_verify_path = self.root_path + 'verify_MSRed.txt'  # 验证文件路径

    # 处理训练和测试文本
    @staticmethod
    def progress_train_and_test(file_path, save_path, num):
        # 通过打开文件并立即关闭来清空文件内容
        with open(save_path, 'w') as file:
            pass
        for i in range(500):
            path = os.path.join(file_path, '{:04d}'.format(i + 1))
            with open(save_path, 'a') as file:
                # 获取train_path目录下的所有文件列表，并按字母顺序排序
                files = os.listdir(path)
                files.sort()
                # 遍历train_path目录下的每个文件，拼接出图像的完整路径，从文件名中提取标签
                data = []
                for file_name in files:
                    if int(file_name.split('_')[0]) == num:
                        data.append((os.path.join(path, file_name), i))
                # 将图像路径和标签写入文件
                for image_path, label in data:
                    file.write('%s %d\n' % (image_path, label))

    # 处理验证文本
    @staticmethod
    def progress_verify(save_train_path, save_test_path, save_verify_path):
        # 打开第一个txt文件并读取内容
        with open(save_train_path, 'r') as file:
            train = file.read()

        # 打开第二个txt文件并读取内容
        with open(save_test_path, 'r') as file:
            test = file.read()

        # 将两个文件的内容合并
        variety = train + test

        # 将合并后的内容写入新的txt文件
        with open(save_verify_path, 'w') as merged_file:
            merged_file.write(variety)

    # 得到数据集文本
    def get_data_text_for_MSRed(self):
        self.progress_train_and_test(self.train_path, self.save_train_path, 1)
        self.progress_train_and_test(self.test_path, self.save_test_path, 2)
        self.progress_verify(self.save_train_path, self.save_test_path, self.save_verify_path)


if __name__ == '__main__':

    # 多光谱（绿光）掌纹数据集处理
    (GetDataTextForMSRed(train_path=MSRed_train_path, test_path=MSRed_test_path, root_path=root)
     .get_data_text_for_MSRed())

'''  
may the force be with you.
@FileName   get_data_text
Created by 24 on 2024/1/26.
'''
