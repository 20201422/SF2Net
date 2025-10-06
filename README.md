# SF2Net: Sequence Feature Fusion Network for Palmprint Verification

This repository is a PyTorch implementation of SF2Net (accepted by IEEE Transactions on Information Forensics and Security). This paper can be downloaded at [this link](https://doi.org/10.1109/TIFS.2025.3611692).

## Abstract
Currently global features are usually extracted directly from local patterns in palmprint verification. Furthermore, sequence features for palmprint verification are only used as local features, but the properties of sequence features are not fully utilized. To solve this issue, this paper introduces Sequence Feature Fusion Network (SF2Net) for palmprint verification. SF2Net proposes a new paradigm: using stable and spatially correlated sequence features as an intermediate bridge to generate robust global representations. SF2Net’s core mechanism is to first extract fine-grained local features that are then converted into sequence features by a Sequence Feature Extractor (SFE). Finally, the sequence features are used as a superior input to capture high-quality global features. By fusing multi-order texture-based local features with globally extracted sequence features, SF2Net achieves superior discrimination. To ensure high accuracy even with limited training data, a hybrid loss function is proposed, which integrate a cross-entropy loss and a triplet loss. Triplet loss effectively optimizes feature separation by explicitly considering negative samples. Extensive experiments on multiple publicly available palmprint datasets demonstrate that SF2Net achieves state-of-the-art (SOTA) performance. Remarkably, even with a small training-to-testing ratio (1:9), SF2Net achieves 100% accuracy, surpassing SOTA methods under several benchmark datasets.
## Citation
If our work is valuable to you, please cite our work:
```
@ARTICLE{yang2023ccnet,
  author={Liu, Yunlong and Leng, Lu and Yang, Ziyuan and Teoh, Andrew Beng Jin and Zhang, Bob},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={SF2Net: Sequence Feature Fusion Network for Palmprint Recognition}, 
  year={2025},
  volume={20},
  pages={9936-9949},
  doi={10.1109/TIFS.2025.3611692}}
```

## Requirements

Our codes were implemented by ```PyTorch 2.4.1``` and ```12.1``` CUDA version. If you wanna try our method, please first install necessary packages as follows:

```
pip install requirements.txt
```

## Data Preprocessing
To help readers to reproduce our method, we also release our training and testing lists (including PolyU, Tongji, IITD, Multi-Spectrum datasets). If you wanna try our method in other datasets, you need to generate training and testing texts as follows:

#### Be careful to modify the data set path in the file!

```
python ./data/PolyU/get_data_text_for_PolyU.py
```

## Quick Start for Training and Testing
If you wanna try our method quickly, you can directly run our code as follows:

#### Check the order in which the data sets are trained in the ```auto_run.py``` file before you begin!

```
python auto_run.py
```

## Training and Testing
If you need to customize the parameters, look like this:

```
python train.py --batch_size 500 --epoch_num 1000 --id_num 600 --one_label_intra_class_num 20 --loss_ce 0.7 --loss_tl 0.3 --vit_floor_num 10 --weight 0.7 --gpu_id 0 --lr 0.001 --redstep 500 --test_interval 1000 --save_interval 500 --train_set_file ./data/Tongji/train_Tongji.txt --test_set_file ./data/Tongji/test_Tongji.txt --des_path ./results/Tongji/checkpoint/ --path_rst ./results/Tongji/rst_test/
```

* batch_size: the size of batch to be used for local training. (default: ```500```)
* epoch_num: the number of total training epoches. (default: ```1000```)
* id_num: the number of ids in the dataset. (default: ```600```)
* one_label_intra_class_num：the number of intra-class samples in one label. (default: ```20```)
* loss_ce: the weight of cross-entropy loss. (default: ```0.7```)
* loss_tl: the weight of triplet loss. (default: ```0.3```)
* vit_floor_num: the values of first-*k* and last-*k* in SFE. (default: ```10```)
* weight: the weights of local and global features. (default: ```0.7```)
* gpu_id: the id of training gpu. (default: ```0```)
* lr: the inital learning rate. (default: ```0.001```)
* redstep: the step size of learning scheduler. (default: ```500```)
* test_interval: the interval of testing. (default: ```1000```)
* save_interval: the interval of saving. (default: ```500```)
* train_set_file: the path of training text file. (default: ```./data/Tongji/train_Tongji.txt```)
* test_set_file: the path of testing text file. (default: ```./data/Tongji/test_Tongji.txt```)
* des_path: the path of saving checkpoints. (default: ```./results/Tongji/checkpoint/```)
* path_rst: the path of saving results. (default: ```./results/Tongji/rst_test/```)


## Acknowledgments
Thanks to my all cooperators, they contributed so much to this work.

## Contact
If you have any question or suggestion to our work, please feel free to contact me. My email is 24.yunongliu@gmail.com.
