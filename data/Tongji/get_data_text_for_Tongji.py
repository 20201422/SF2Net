"""
@FileName   get_data_text
@Author  24
@Date    2024/1/26 03:47
@Version 1.0.0
freedom is the oxygen of the soul.
"""
import os

# 根路径
root = './'

# 同济大学数据集路径
Tongji_train_path = '/public/home/hpc2420083500103/datasets/Tongji/Tongji_ROI/session1/'
Tongji_test_path = '/public/home/hpc2420083500103/datasets/Tongji/Tongji_ROI/session2/'


# 处理同济大学掌纹数据集
class GetDataTextForTongji:
    """处理同济大学掌纹数据集::
        INPUTS:
            train_path：训练数据集路径
            test_path：测试数据集路径
            root_path：根路径
    """

    def __init__(self, train_path, test_path, root_path):
        super(GetDataTextForTongji, self).__init__()
        self.train_path = train_path  # 训练数据集路径
        self.test_path = test_path  # 测试数据集路径

        self.root_path = root_path  # 根路径

        self.save_train_path = self.root_path + 'train_Tongji.txt'  # 训练文件路径
        self.save_test_path = self.root_path + 'test_Tongji.txt'  # 测试文件路径
        self.save_verify_path = self.root_path + 'verify_Tongji.txt'  # 验证文件路径

    # 处理训练和测试文本
    @staticmethod
    def progress_train_and_test(file_path, save_path):
        with open(save_path, 'w') as file:
            # 获取目录下的所有文件列表，并按字母顺序排序
            files = os.listdir(file_path)
            files.sort()
            # 遍历目录下的每个文件，拼接出图像的完整路径，从文件名中提取标签
            data = [(os.path.join(file_path, file_name), int(int(file_name[:5]) - 1) / 10) for file_name in files]
            # 将图像路径和标签写入文件
            for image_path, label in data:
                file.write('%s %d\n' % (image_path, label))

    @staticmethod
    def progress_train_and_test_together(file_path_1, file_path_2, save_path_1, save_path_2):
        with open(save_path_1, 'w') as file_1, open(save_path_2, 'w') as file_2:
            # 获取目录下的所有文件列表，并按字母顺序排序
            files_1 = os.listdir(file_path_1)
            files_2 = os.listdir(file_path_2)
            files_1.sort()
            files_2.sort()
            # 遍历目录下的每个文件，拼接出图像的完整路径，从文件名中提取标签
            data_1 = []
            data_2 = []
            i = 0
            for file_name in files_1:
                if file_name != '.DS_Store':
                    i += 1
                    if i % 10 <= 5 and i % 10 != 0:
                        data_1.append((os.path.join(file_path_1, file_name), int(int(file_name[:5]) - 1) / 10))
                    else:
                        data_2.append((os.path.join(file_path_1, file_name), int(int(file_name[:5]) - 1) / 10))

            for file_name in files_2:
                if file_name != '.DS_Store':
                    i += 1
                    if i % 10 <= 5 and i % 10 != 0:
                        data_1.append((os.path.join(file_path_2, file_name), int(int(file_name[:5]) - 1) / 10))
                    else:
                        data_2.append((os.path.join(file_path_2, file_name), int(int(file_name[:5]) - 1) / 10))
            # 将图像路径和标签写入文件
            for image_path, label in data_1:
                file_1.write('%s %d\n' % (image_path, label))
            for image_path, label in data_2:
                file_2.write('%s %d\n' % (image_path, label))

    # 处理验证文本
    @staticmethod
    def progress_verify(save_train_path, save_test_path, save_verify_path):
        with open(save_train_path, 'r') as train_file, open(save_test_path, 'r') as test_file, open(save_verify_path,
                                                                                                    'w') as verify_file:
            # 读取训练文件和测试文件的所有行
            train_lines = train_file.readlines()
            test_lines = test_file.readlines()

            # 循环，从训练文件和测试文件中每隔10行取出数据，并写入到验证文件中，以此交替
            for i in range(0, len(train_lines), 10):
                for line in train_lines[i:i + 10]:
                    verify_file.write(line)
                for line in test_lines[i:i + 10]:
                    verify_file.write(line)

    # 得到数据集文本
    def get_data_text_for_Tongji(self):
        # self.progress_train_and_test(self.train_path, self.save_train_path)
        # self.progress_train_and_test(self.test_path, self.save_test_path)
        self.progress_train_and_test_together(self.train_path, self.test_path, self.save_train_path, self.save_test_path)
        self.progress_verify(self.save_train_path, self.save_test_path, self.save_verify_path)


if __name__ == '__main__':
    # 同济大学掌纹数据集处理
    (GetDataTextForTongji(train_path=Tongji_train_path, test_path=Tongji_test_path, root_path=root)
     .get_data_text_for_Tongji())

# # 生成不同比例的训练文件
# # 定义路径和文件名
# data_path = '/home/hipeson/lyl/Program/Palmprint(Heirloom Template)/data/Tongji/'
# train_file = os.path.join(data_path, 'train_Tongji.txt')
#
# # 定义比例
# ratios = [2, 4, 5, 6, 8]
# num_per_class = 10  # 每个类别原始图片数量
#
# # 读取原始训练文件
# with open(train_file, 'r') as f:
#     lines = f.readlines()
#
# # 按类别分组
# class_dict = {}
# for line in lines:
#     img_path, label = line.strip().split()
#     if label not in class_dict:
#         class_dict[label] = []
#     class_dict[label].append(img_path)
#
# # 生成新的训练文件
# for ratio in ratios:
#     new_train_file = os.path.join(data_path, f'train_Tongji_{ratio}.txt')
#     with open(new_train_file, 'w') as f:
#         for label, img_paths in class_dict.items():
#             # 取前 ratio 张图片
#             selected_images = img_paths[:ratio]
#             for img_path in selected_images:
#                 f.write(f'{img_path} {label}\n')
#
# print("生成完毕！")

'''  
may the force be with you.
@FileName   get_data_text
Created by 24 on 2024/1/26.
'''
