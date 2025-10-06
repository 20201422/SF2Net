"""
@FileName   auto_run.py
@Author  24
@Date    2024/2/28 00:31
@Version 1.0.0
freedom is the oxygen of the soul.
"""

from utils import *

save_path = './results/'

print(torch.cuda.is_available())
torch.cuda.empty_cache()

import os


def run_normal():
    # 定义数据集信息
    datasets = {
        'tongji': {
            'id_num': 600,
            'loss_ce': 0.7,
            'loss_tl': 0.3,
            'vit_floor_num': 10,
            'weight': 0.7,
            'train_set_file': './data/Tongji/train_Tongji.txt',
            'test_set_file': './data/Tongji/test_Tongji.txt',
            'des_path': './results/Tongji/checkpoint/',
            'path_rst': './results/Tongji/rst_test/'
        },
        'polyu': {
            'id_num': 386,
            'loss_ce': 0.7,
            'loss_tl': 0.3,
            'vit_floor_num': 10,
            'weight': 0.7,
            'train_set_file': './data/PolyU/train_PolyU.txt',
            'test_set_file': './data/PolyU/test_PolyU.txt',
            'des_path': './results/PolyU/checkpoint/',
            'path_rst': './results/PolyU/rst_test/'
        },
        'iitd': {
            'id_num': 460,
            'loss_ce': 0.7,
            'loss_tl': 0.3,
            'vit_floor_num': 10,
            'weight': 0.7,
            'train_set_file': './data/IITD/train_IITD.txt',
            'test_set_file': './data/IITD/test_IITD.txt',
            'des_path': './results/IITD/checkpoint/',
            'path_rst': './results/IITD/rst_test/'
        },
        'msred': {
            'id_num': 500,
            'loss_ce': 0.7,
            'loss_tl': 0.3,
            'vit_floor_num': 10,
            'weight': 0.7,
            'train_set_file': './data/MSRed/train_MSRed.txt',
            'test_set_file': './data/MSRed/test_MSRed.txt',
            'des_path': './results/MSRed/checkpoint/',
            'path_rst': './results/MSRed/rst_test/'
        },
        'msgreen': {
            'id_num': 500,
            'loss_ce': 0.7,
            'loss_tl': 0.3,
            'vit_floor_num': 10,
            'weight': 0.7,
            'train_set_file': './data/MSGreen/train_MSGreen.txt',
            'test_set_file': './data/MSGreen/test_MSGreen.txt',
            'des_path': './results/MSGreen/checkpoint/',
            'path_rst': './results/MSGreen/rst_test/'
        },
        'msblue': {
            'id_num': 500,
            'loss_ce': 0.7,
            'loss_tl': 0.3,
            'vit_floor_num': 10,
            'weight': 0.7,
            'train_set_file': './data/MSBlue/train_MSBlue.txt',
            'test_set_file': './data/MSBlue/test_MSBlue.txt',
            'des_path': './results/MSBlue/checkpoint/',
            'path_rst': './results/MSBlue/rst_test/'
        },
        'msnir': {
            'id_num': 500,
            'loss_ce': 0.7,
            'loss_tl': 0.3,
            'vit_floor_num': 10,
            'weight': 0.7,
            'train_set_file': './data/MSNIR/train_MSNIR.txt',
            'test_set_file': './data/MSNIR/test_MSNIR.txt',
            'des_path': './results/MSNIR/checkpoint/',
            'path_rst': './results/MSNIR/rst_test/'
        }
    }

    # 定义要运行的命令
    command_template = ("python train.py --id_num {id_num} "
                        "--loss_ce {loss_ce} --loss_tl {loss_tl} --vit_floor_num {vit_floor_num} --weight {weight} "
                        "--train_set_file {train_set_file} --test_set_file {test_set_file} "
                        "--des_path {des_path} --path_rst {path_rst}")

    # 循环执行命令  
    for dataset_name, paths in datasets.items():
        print(dataset_name, paths)
        command = command_template.format(
            id_num=paths['id_num'],
            loss_ce=paths['loss_ce'],
            loss_tl=paths['loss_tl'],
            vit_floor_num=paths['vit_floor_num'],
            weight=paths['weight'],
            train_set_file=paths['train_set_file'],
            test_set_file=paths['test_set_file'],
            des_path=paths['des_path'],
            path_rst=paths['path_rst']
        )
        os.system(command)


# 正常实验
run_normal()


'''  
may the force be with you.
@FileName   auto_run.py
Created by 24 on 2024/2/28.
'''
