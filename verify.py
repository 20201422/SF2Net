import os
# os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
import argparse
import time
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

plt.switch_backend('agg')

from torch.optim import lr_scheduler
# import pickle
import cv2 as cv
from utils.data_set import MyDataset
from model.sf2net import SF2Net
from utils import *

import copy


def test(model):
    print('Start Testing!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    path_hard = os.path.join(path_rst, 'rank1_hard')

    # train_set_file = './data/train_IITD.txt'
    # test_set_file = './data/test_IITD.txt'

    trainset = MyDataset(txt=train_set_file, transforms=None, train=False)
    testset = MyDataset(txt=test_set_file, transforms=None, train=False)

    batch_size = 512  # 128

    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=2)
    data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, num_workers=2)

    fileDB_train = getFileNames(train_set_file)
    fileDB_test = getFileNames(test_set_file)

    # output dir
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    if not os.path.exists(path_hard):
        os.makedirs(path_hard)

    net = model

    net.cuda()
    net.eval()

    # feature extraction:

    featDB_train = []
    iddb_train = []

    for batch_id, (datas, target) in enumerate(data_loader_train):

        data = datas[0]

        data = data.cuda()
        target = target[0].cuda()

        codes = net.get_feature_vector(data)
        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_train = codes
            iddb_train = y
        else:
            featDB_train = np.concatenate((featDB_train, codes), axis=0)
            iddb_train = np.concatenate((iddb_train, y))

    print('completed feature extraction for training set.')
    print('featDB_train.shape: ', featDB_train.shape)

    classNumel = len(set(iddb_train))
    num_training_samples = featDB_train.shape[0]

    trainNum = num_training_samples // classNumel
    print('[classNumel, imgs/class]: ', classNumel, trainNum)
    print('\n')

    featDB_test = []
    iddb_test = []

    print('Start Test Feature Extraction.')
    for batch_id, (datas, target) in enumerate(data_loader_test):

        data = datas[0]
        data = data.cuda()
        target = target[0].cuda()

        codes = net.get_feature_vector(data)

        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_test = codes
            iddb_test = y
        else:
            featDB_test = np.concatenate((featDB_test, codes), axis=0)
            iddb_test = np.concatenate((iddb_test, y))

    if batch_id != 1:
        print('aaaa')

    print('completed feature extraction.')
    print('featDB_test.shape: ', featDB_test.shape)

    print('\nFeature Extraction Done!')

    print('start feature matching ...\n')

    print('Verification EER of the test-test set ...')

    print('Start EER for Test-Test Set! \n')

    # verification EER of the test set
    s = []  # matching score
    l = []  # intra-class or inter-class matching
    ntest = featDB_test.shape[0]
    ntrain = featDB_train.shape[0]

    for i in range(ntest):
        feat1 = featDB_test[i]

        for j in range(ntrain):
            feat2 = featDB_train[j]

            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi

            s.append(dis)

            if iddb_test[i] == iddb_train[j]:  # same palm
                l.append(1)
            else:
                l.append(-1)

    if not os.path.exists(path_rst + 'veriEER'):
        os.makedirs(path_rst + 'veriEER')
    if not os.path.exists(path_rst + 'veriEER/rank1_hard/'):
        os.makedirs(path_rst + 'veriEER/rank1_hard/')

    with open(path_rst + 'veriEER/scores_VeriEER.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')
    os.system('python ./getEER.py' + '  ' + path_rst + 'veriEER/scores_VeriEER.txt scores_VeriEER')

    print('\n------------------')
    print('Rank-1 acc of the test set...')
    # rank-1 acc
    cnt = 0
    corr = 0
    for i in range(ntest):
        probeID = iddb_test[i]

        dis = np.zeros((ntrain, 1))

        for j in range(ntrain):
            dis[j] = s[cnt]
            cnt += 1

        idx = np.argmin(dis[:])

        galleryID = iddb_train[idx]

        if probeID == galleryID:
            corr += 1
        else:
            testname = fileDB_test[i]
            trainname = fileDB_train[idx]
            # store similar inter-class samples
            im_test = cv.imread(testname)
            im_train = cv.imread(trainname)
            img = np.concatenate((im_test, im_train), axis=1)
            cv.imwrite(path_rst + 'veriEER/rank1_hard/%6.4f_%s_%s.png' % (
                np.min(dis[:]), testname[-13:-4], trainname[-13:-4]), img)

    rankacc = corr / ntest * 100
    print('rank-1 acc: %.3f%%' % rankacc)
    print('-----------')

    with open(path_rst + 'veriEER/rank1.txt', 'w') as f:
        f.write('rank-1 acc: %.3f%%' % rankacc)

    print('\n\nReal EER of the test set...')
    # dataset EER of the test set (the gallery set is not used)
    s = []  # matching score
    l = []  # genuine / impostor matching
    n = featDB_test.shape[0]
    for i in range(n - 1):
        feat1 = featDB_test[i]

        for jj in range(n - i - 1):
            j = i + jj + 1
            feat2 = featDB_test[j]

            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi

            s.append(dis)

            if iddb_test[i] == iddb_test[j]:
                l.append(1)
            else:
                l.append(-1)

    print('feature extraction about real EER done!\n')

    with open(path_rst + 'veriEER/scores_EER_test.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + path_rst + 'veriEER/scores_EER_test.txt scores_EER_test')
    os.system('python ./getEER.py' + '  ' + path_rst + 'veriEER/scores_EER_test.txt scores_EER_test')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="HCTNet for Palmprint Recfognition"
    )

    parser.add_argument("--id", type=str, default="IITD")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--id_num", type=int, default=460,
                        help="IITD: 460 KTU: 145 Tongji: 600 REST: 358 XJTU: 200 POLYU 386 Multi-Spec 500 IITD_Right 230 Tongji_LR 300")
    parser.add_argument("--gpu_id", type=str, default='0')

    # Training Path
    parser.add_argument("--train_set_file", type=str, default='./data/IITD/train_IITD.txt')
    parser.add_argument("--test_set_file", type=str, default='./data/IITD/test_IITD.txt')

    # Store Path
    parser.add_argument("--des_path", type=str, default='./results/IITD11/checkpoint/')
    parser.add_argument("--path_rst", type=str, default='./results/IITD11/rst_test/')

    args = parser.parse_args()

    # print(args.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    id = args.id  # id
    batch_size = args.batch_size
    label_num = args.id_num

    model = "./results/" + id + "/checkpoint/net_params_best.pth"  # 源模型路径

    des_path = args.des_path
    path_rst = args.path_rst

    if not os.path.exists(des_path):
        os.makedirs(des_path)
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    # path
    train_set_file = args.train_set_file
    test_set_file = args.test_set_file

    # dataset
    trainset = MyDataset(txt=train_set_file, transforms=None, train=True, imside=128, outchannels=1)
    testset = MyDataset(txt=test_set_file, transforms=None, train=False, imside=128, outchannels=1)

    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=2, shuffle=True)
    data_loader_test = DataLoader(dataset=testset, batch_size=128, num_workers=2, shuffle=True)

    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    print('------Init Model------')
    best_net = SF2Net(label_num=label_num, vit_floor_num=10)
    best_net = type(best_net)(label_num, 10)  # 创建一个新的模型实例
    best_net.load_state_dict(torch.load(model))  # 加载最好的模型状态字典
    best_net.cuda()

    print('------------\n')
    print('Best')
    test(best_net)
