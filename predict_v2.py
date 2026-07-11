import os
import random
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import DriveDataset_Pro 
from train import create_model
from Tools.preprocess_utils import get_preset_transform 
import Tools.metrics_utils as metrics 


def calculate_metrics(pred, target, predict_classes):
    """
    调用评测工具进行底层打分 (高度封装版)
    严格遵循单一职责原则：只负责算分，不负责解析。
    """
    # 1. 实例化当前图片的打分器
    confmat = metrics.ConfusionMatrix(predict_classes)
    dice = metrics.DiceCoefficient(num_classes=predict_classes, ignore_index=255)
    
    # 2. 喂入数据进行计算
    # 注意：这里的 pred 就是 output [1, 2, H, W]
    confmat.update(target.flatten(), pred.argmax(1).flatten())
    dice.update(pred, target)
    
    # 3. 获取原生指标分数
    acc_global, recall_classes, precision_classes, iou_classes = confmat.compute()
    dice_score = dice.value.item()

    # 我们只关心类 1 (血管)
    vessel_recall = recall_classes[1].item()
    vessel_precision = precision_classes[1].item()
    vessel_iou = iou_classes[1].item()

    return dice_score, vessel_iou, vessel_recall, vessel_precision, acc_global.item()


def generate_error_map(pred, target):
    """
    生成误差图 (Error Map)。
    传入的 target 已过 DriveDataset 处理，里面为 0(背景), 1(血管) 和 255(不感兴趣区域)
    """
    h, w = pred.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. 预测正确 (True Positive): 血管 -> 纯白色
    error_map[(pred == 1) & (target == 1)] = [255, 255, 255]
    # 2. 预测正确 (True Negative): 背景 -> 纯黑色
    error_map[(pred == 0) & (target == 0)] = [0, 0, 0]
    # 3. 误诊/假阳性 (False Positive): 背景，模型预测为血管 -> 红色
    error_map[(pred == 1) & (target == 0)] = [255, 0, 0]
    # 4. 漏诊/假阴性 (False Negative): 血管，模型预测为背景 -> 绿色
    error_map[(pred == 0) & (target == 1)] = [0, 255, 0]
    
    # 将 255 的不感兴趣区域强行改纯黑
    error_map[target == 255] = [0, 0, 0]
    return error_map

def main(args):
    # 环境配置与模型加载
    weights_path = "./save_weights/best_model.pth" #模型权重文件的路径
    assert os.path.exists(weights_path), f"找不到权重文件: {weights_path}"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    # 创建预测结果保存目录
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("predict_result", f"predict{current_time}")
    os.makedirs(save_dir, exist_ok=True)

    # 创建模型，加载权重
    predict_classes=num_classes = args.num_classes + 1
    checkpoint = torch.load(weights_path, map_location='cpu')

    # 自动检测模型架构 (新checkpoint直接读取, 旧checkpoint根据state_dict键推断)
    if hasattr(checkpoint, 'get') and checkpoint.get('args') and hasattr(checkpoint['args'], 'model'):
        model_name = checkpoint['args'].model
    else:
        state_keys = list(checkpoint['model'].keys())
        if any('nested' in k for k in state_keys):
            model_name = 'unet++'
        elif any('downsample' in k for k in state_keys):
            model_name = 'resunet'
        elif any('conv_block' in k or 'blocks' in k for k in state_keys):
            model_name = 'nnunet'
        elif any('att.' in k for k in state_keys):
            model_name = 'attention_unet'
        else:
            model_name = 'unet'

    model = create_model(num_classes=predict_classes, model_name=model_name)
    print(f"检测到模型类型: {model_name}")
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval() # 切换到评测模式，关闭 BatchNorm 和 Dropout

    # 数据加载
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # 实例化数据集 (自动读取图片、拼接黑边 255 掩码)
    test_dataset = DriveDataset_Pro(root=args.data_path, 
                                    train=False, 
                                    transforms=get_preset_transform(train=False, mean=mean, std=std))
    
    # 随机抽取 n 张测试图的索引
    n = args.num_pred
    assert len(test_dataset) >= n, f"测试集图片不足 {n} 张！"
    sample_indices = random.sample(range(len(test_dataset)), n)

    # 准备画布与评测工具
    fig, axes = plt.subplots(n, 4, figsize=(20, 5*n))
    col_titles = ['Original Image', 'Ground Truth', 'Prediction', 'Error Map (Red:FP, Green:FN)']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=16, fontweight='bold')

    print("开始调用模型进行批量推理...")

    # 4. 推理核心循环
    with torch.no_grad():
        for row_idx, data_idx in enumerate(sample_indices):
            # 从 Dataset 获取数据
            img_tensor, target_tensor = test_dataset[data_idx]
            
            # 为了符合模型输入，增加一个 batch 维度
            img_tensor = img_tensor.unsqueeze(0).to(device)
            target_tensor = target_tensor.unsqueeze(0).to(device) # shape: [1, H, W]

            # 获取原始文件名 ，用于打印标识
            img_path = test_dataset.img_list[data_idx]
            base_name = os.path.basename(img_path).split('_')[0]

            # --- 模型前向传播 ---
            output = model(img_tensor)
            output = output['out'] # 拿到带有分类得分的特征图 [1, 2, H, W]
            
            # --- 计算指标 ---
            dice_score, vessel_iou, vessel_recall, vessel_precision, acc_global = calculate_metrics(output, target_tensor, predict_classes)
            print(f"图像 [{base_name}] | Dice: {dice_score:.4f} | IoU(血管): {vessel_iou:.4f} | Recall(血管): {vessel_recall:.4f} | Precision(血管): {vessel_precision:.4f} | 全局准确率: {acc_global:.4f}")

            # --- 图像后处理与可视化 ---
            # 获取单通道预测结果
            pred_np = output.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            target_np = target_tensor.squeeze(0).cpu().numpy().astype(np.uint8)
            
            # 利用 dataset 传来的 target 里的 255 清理预测黑边
            pred_np[target_np == 255] = 0

            # 还原用于展示的原图 (反归一化)
            orig_img = img_tensor.squeeze(0).cpu()
            for t, m, s in zip(orig_img, mean, std):
                t.mul_(s).add_(m)
            orig_img_np = np.transpose(orig_img.numpy(), (1, 2, 0))
            orig_img_np = np.clip(orig_img_np, 0, 1) # 限制到 0-1 以供 imshow 显示

            # 生成误差图
            error_map = generate_error_map(pred_np, target_np)

            # --- 填入 Matplotlib 子图 ---
            axes[row_idx, 0].imshow(orig_img_np)
            axes[row_idx, 0].axis('off')
            
            # 金标准可视化时，把 255 的黑边当作 0 处理，免得画出来是全白的圈
            gt_display = np.copy(target_np)
            gt_display[gt_display == 255] = 0
            axes[row_idx, 1].imshow(gt_display, cmap='gray')
            axes[row_idx, 1].axis('off')

            axes[row_idx, 2].imshow(pred_np, cmap='gray')
            axes[row_idx, 2].axis('off')

            axes[row_idx, 3].imshow(error_map)
            axes[row_idx, 3].axis('off')
            
            axes[row_idx, 0].text(-10, 256, f"Img: {base_name}", fontsize=14, rotation=90, va='center', fontweight='bold')

    # 保存最终全景大图
    plt.tight_layout()
    final_plot_path = os.path.join(save_dir, "batch_evaluation_grid.png")
    plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
    print("="*50)
    print(f"预测完成！{n}x4 评估图已保存至: {final_plot_path}")


def parse_args_predict():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet prediction")

    parser.add_argument("--data-path", default="./", help="DRIVE root")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--num-pred", default=4, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args_predict()
    main(args)