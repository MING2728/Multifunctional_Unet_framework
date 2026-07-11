import os
import time
import random
from datetime import datetime
import glob

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from train import create_model  # 与训练保持一致的模型工厂

def calculate_metrics(pred, target, roi_mask):
    """
    自定义底层打分器：完全基于 TP, TN, FP, FN 的底层张量逻辑。
    绝不调用 sklearn 官方包，完美还原我们之前手撕的混淆矩阵与 Dice 逻辑。
    """
    # 仅在感兴趣区域 (ROI) 内计算指标，排除外部无意义的黑边
    valid_mask = roi_mask > 0
    p = pred[valid_mask]
    t = target[valid_mask]

    # 底层张量统计魔法 (True Positive, True Negative, False Positive, False Negative)
    TP = ((p == 1) & (t == 1)).sum().astype(np.float32)
    TN = ((p == 0) & (t == 0)).sum().astype(np.float32)
    FP = ((p == 1) & (t == 0)).sum().astype(np.float32)
    FN = ((p == 0) & (t == 1)).sum().astype(np.float32)

    # 加上 1e-6 (epsilon) 防止分母为 0 导致报错
    iou = TP / (TP + FP + FN + 1e-6)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    return iou.item(), dice.item(), acc.item(), precision.item(), recall.item()

def generate_error_map(pred, target, roi_mask):
    """
    生成误差图 (Error Map)，用于直观分析模型的漏诊和误诊情况
    """
    h, w = pred.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. 预测正确 (True Positive): 血管用纯白色表示
    error_map[(pred == 1) & (target == 1)] = [255, 255, 255]
    # 2. 预测正确 (True Negative): 背景保持纯黑色
    error_map[(pred == 0) & (target == 0)] = [0, 0, 0]
    # 3. 误诊/假阳性 (False Positive): 本来是背景，模型非说是血管 -> 用红色警告！
    error_map[(pred == 1) & (target == 0)] = [255, 0, 0]
    # 4. 漏诊/假阴性 (False Negative): 本来是血管，模型没找出来 -> 用绿色标出！
    error_map[(pred == 0) & (target == 1)] = [0, 255, 0]
    
    # 视野外部的区域强行抹黑
    error_map[roi_mask == 0] = [0, 0, 0]
    return error_map

def main():
    # 1. 初始化设置
    classes = 1 
    weights_path = "./save_weights/best_model.pth"
    assert os.path.exists(weights_path), f"找不到权重文件: {weights_path}"

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    # 2. 创建目录 (使用当前时间戳，例如: predict20260609-002646)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("predict_result", f"predict{current_time}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"预测结果将保存在: {save_dir}")

    # 3. 搜索并随机抽取 4 张测试图
    img_dir = "./DRIVE/test/images"
    mask_dir = "./DRIVE/test/1st_manual" # DRIVE 数据集的金标准文件夹通常叫这个
    roi_dir = "./DRIVE/test/mask"

    all_img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    assert len(all_img_paths) >= 4, "测试集图片不足 4 张！"
    
    # 随机抽取 4 张图的路径
    sample_img_paths = random.sample(all_img_paths, 4)

    # 4. 加载模型
    model = create_model(num_classes=classes+1)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)
    model.eval()

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 准备画图的画布: 4行4列
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    # 设置列标题
    col_titles = ['Original Image', 'Ground Truth', 'Prediction', 'Error Map (Red:FP, Green:FN)']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=16, fontweight='bold')

    print("开始调用模型进行批量推理...")


    with torch.no_grad():
        for i, img_path in enumerate(sample_img_paths):
            # 获取文件名标识 (如 01_test)
            base_name = os.path.basename(img_path).split('_')[0] 
            
            # 拼接对应的金标准和 ROI 掩码路径
            gt_path = os.path.join(mask_dir, f"{base_name}_manual1.gif")
            roi_path = os.path.join(roi_dir, f"{base_name}_test_mask.gif")

            # 读取原始图像 (用于展示)
            orig_img_pil = Image.open(img_path).convert('RGB')
            # 预处理为模型输入格式
            img_tensor = data_transform(orig_img_pil).unsqueeze(0).to(device)

            # 读取金标准并二值化为 0 和 1
            gt_pil = Image.open(gt_path).convert('L')
            gt_np = np.array(gt_pil)
            gt_np = (gt_np > 127).astype(np.uint8)

            # 读取 ROI 掩码
            roi_pil = Image.open(roi_path).convert('L')
            roi_np = np.array(roi_pil)
            roi_np = (roi_np > 127).astype(np.uint8)

            # --- 模型前向传播 ---
            output = model(img_tensor)
            
            # 获取双通道模型结果: argmax(1)
            pred_tensor = output['out'].argmax(1).squeeze(0)
            pred_np = pred_tensor.cpu().numpy().astype(np.uint8)
            
            # 利用 ROI 清理黑边噪点
            pred_np[roi_np == 0] = 0

            # --- 计算并打印各项指标 ---
            iou, dice, acc, precision, recall = calculate_metrics(pred_np, gt_np, roi_np)
            print(f"图像 [{base_name}] | IoU: {iou:.4f} | Dice/F1: {dice:.4f} | Acc: {acc:.4f} | Pre: {precision:.4f} | Rec: {recall:.4f}")

            # --- 生成误差图 ---
            error_map = generate_error_map(pred_np, gt_np, roi_np)

            # --- 填入 Matplotlib 子图 ---
            axes[i, 0].imshow(orig_img_pil)
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(gt_np, cmap='gray')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_np, cmap='gray')
            axes[i, 2].axis('off')

            axes[i, 3].imshow(error_map)
            axes[i, 3].axis('off')
            
            axes[i, 0].text(-10, 256, f"Img: {base_name}", fontsize=14, rotation=90, va='center', fontweight='bold')

    # 5. 保存最终的全景大图
    plt.tight_layout()
    final_plot_path = os.path.join(save_dir, "batch_evaluation_grid.png")
    plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
    print("="*50)
    print(f"批量推理完成！4x4 评估大图已保存至: {final_plot_path}")

if __name__ == '__main__':
    main()