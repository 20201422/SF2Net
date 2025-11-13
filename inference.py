import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T

from model.sf2net import SF2Net
from utils.data_set import NormSingleROI


def preprocess_image(image_path, imside=128, outchannels=1):
    """
    加载并预处理单张图像。
    :param image_path: 图像文件路径。
    :param imside: 图像目标尺寸。
    :param outchannels: 输出通道数。
    :return: 预处理后的图像张量。
    """
    transform = T.Compose([
        T.Resize((imside, imside)),
        T.ToTensor(),
        NormSingleROI(outchannels=outchannels)
    ])

    try:
        img = Image.open(image_path).convert('L')
        img_tensor = transform(img)
        return img_tensor.unsqueeze(0)
    except FileNotFoundError:
        print(f"错误：找不到图像文件 {image_path}")
        return None


def inference(model_path, image_path1, image_path2, label_num, vit_floor_num, gpu_id='0'):
    """
    使用预训练模型对两张图像进行推理，并计算相似度得分。
    :param model_path: 预训练模型文件的路径 (.pth)。
    :param image_path1: 第一张图像的路径。
    :param image_path2: 第二张图像的路径。
    :param label_num: 数据集中的类别数。
    :param vit_floor_num: ViT模型的层数。
    :param gpu_id: 要使用的GPU ID。
    """
    # --- 1. 设置设备 ---
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"正在使用设备: {device}")

    # --- 2. 加载模型 ---
    model = SF2Net(label_num=label_num, vit_floor_num=vit_floor_num)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型权重加载成功")
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {model_path}")
        return
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    model.to(device)
    model.eval()

    # --- 3. 图像预处理 ---
    print(f"正在处理图像: {image_path1}")
    img1_tensor = preprocess_image(image_path1)
    print(f"正在处理图像: {image_path2}")
    img2_tensor = preprocess_image(image_path2)

    if img1_tensor is None or img2_tensor is None:
        return

    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)

    # --- 4. 特征提取 ---
    print("正在提取特征向量...")
    with torch.no_grad():
        feat1 = model.get_feature_vector(img1_tensor)
        feat2 = model.get_feature_vector(img2_tensor)

    feat1 = feat1.cpu().numpy()
    feat2 = feat2.cpu().numpy().T

    # --- 5. 计算相似度得分 ---
    cos_sim = np.dot(feat1, feat2)
    dist = np.arccos(np.clip(cos_sim, -1.0, 1.0)) / np.pi

    score = dist[0][0]
    print("\n--- 推理结果 ---")
    print(f"图像1: {os.path.basename(image_path1)}")
    print(f"图像2: {os.path.basename(image_path2)}")
    print(f"计算出的距离分数为: {score:.6f}")
    print("(分数越接近0, 表示两张图像越相似)")


if __name__ == "__main__":
    # --- 6. 设置命令行参数 ---
    parser = argparse.ArgumentParser(description="SF2Net 推理脚本: 比较两张掌纹图像的相似度。")
    parser.add_argument('--model_path', type=str,
                        default='./results/PolyU/checkpoint/net_params_best.pth',
                        help='预训练模型的路径 (.pth文件)')
    parser.add_argument('--image1', type=str,
                        default='./datasets/PolyU/PalmBigDataBase_zq/P_F_102_7.bmp',
                        help='第一张需要推理的图像的路径')
    parser.add_argument('--image2', type=str,
                        default='./datasets/PolyU/PalmBigDataBase_zq/P_F_100_5.bmp',
                        help='第二张需要推理的图像的路径')
    parser.add_argument("--gpu_id", type=str, default='0', help="要使用的GPU的ID")
    parser.add_argument('--label_num', type=int, default=386,
                        help="Tongji: 600 PolyU 386 IITD: 460 Multi-Spec 500")
    parser.add_argument("--vit_floor_num", type=int, default=10, help="ViT模型的层数")

    args = parser.parse_args()

    # 执行推理
    inference(args.model_path, args.image1, args.image2, args.label_num, args.vit_floor_num, args.gpu_id)
