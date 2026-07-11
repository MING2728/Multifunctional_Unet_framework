# 多功能分割优化框架

基于 PyTorch 的视网膜血管语义分割框架，支持 5 种 U-Net 变体自由切换，专为 DRIVE 数据集优化。

## 项目结构

```
├── Model/
│   ├── unet.py                  # 标准 U-Net
│   ├── attention_unet.py        # Attention U-Net（注意力门控）
│   ├── unet++.py                # U-Net++（密集跳跃连接）
│   ├── resunet.py               # ResU-Net（残差编码器）
│   └── nnunet.py                # nnU-Net 风格（自适应配置）
├── Tools/
│   ├── loss_and_dice.py         # 损失函数 (CE + Dice Loss)
│   ├── metrics_utils.py         # 评估指标 (混淆矩阵、Dice、IoU、Recall、Precision)
│   ├── train_utils.py           # 训练/验证循环
│   ├── lr_utils.py              # 学习率调度器 (Warmup + Poly 衰减)
│   ├── transform_utils.py       # 数据增强算子库 (弹性形变、CLAHE、色彩抖动等)
│   ├── preprocess_utils.py      # 训练/验证预处理预设
│   └── monitor_utils.py         # 训练日志监视器
├── dataset.py                   # DRIVE 数据集加载 (自动拼接掩码、动态Padding)
├── train.py                     # 训练入口脚本
├── predict.py                   # 预测脚本 v1 (独立版)
├── predict_v2.py                # 预测脚本 v2 (框架集成版，推荐)
├── mean_std.py                  # 计算数据集 RGB 均值和标准差
├── plot.py                      # 训练曲线可视化仪表盘
└── DRIVE/                       # 数据集目录
    ├── training/
    │   ├── images/              # 训练图像 (.tif)
    │   ├── 1st_manual/          # 金标准标注 (.gif)
    │   └── mask/                # ROI 掩码 (.gif)
    └── test/
        ├── images/              # 测试图像 (.tif)
        ├── 1st_manual/          # 金标准标注 (.gif)
        └── mask/                # ROI 掩码 (.gif)
```

## 环境要求

- Python >= 3.8
- PyTorch >= 1.10
- CUDA (推荐 GPU 训练)
- 详见 `requirements.txt`

## 快速开始

### 1. 数据集准备

将 DRIVE 数据集放入 `DRIVE/` 目录，结构如上所示。

### 2. 计算数据统计量（可选）

```bash
python mean_std.py
```

### 3. 训练

```bash
# 默认配置 (Attention U-Net, base_c=32, 200 epochs)
python train.py

# 选择模型架构 (5 选 1)
python train.py --model unet
python train.py --model attention_unet   # 默认
python train.py --model unet++
python train.py --model resunet
python train.py --model nnunet

# 开启数据增强 + 调整 Dice 权重
python train.py --model attention_unet --use-elastic --use-color --dice-weight 3.0

# 完整参数
python train.py \
    --model attention_unet \
    --data-path ./ \
    --epochs 200 \
    --batch-size 4 \
    --lr 0.01 \
    --dice-weight 1.0 \
    --use-elastic \
    --use-color \
    --use-noise \
    --amp
```

| 关键参数 | 说明 |
|---|---|
| `--model` | 选择模型: `unet` / `attention_unet`(默认) / `unet++` / `resunet` / `nnunet` |
| `--epochs` | 训练轮次 (默认 200) |
| `--lr` | 初始学习率 (默认 0.01) |
| `--dice-weight` | Dice loss 权重系数，loss = CE + dice_weight × Dice (默认 1.0) |
| `--use-elastic` | 开启弹性形变增强 |
| `--use-color` | 开启 CLAHE + 色彩抖动 + Gamma 校正 |
| `--use-noise` | 开启高斯噪声注入 |
| `--amp` | 混合精度训练 |
| `--resume` | 从 checkpoint 恢复训练 |

### 4. 推理

```bash
# 推荐：框架集成版
python predict_v2.py --num-pred 4

# 独立版
python predict.py
```

预测结果保存在 `predict_result/` 目录下，包含原图、金标准、预测图、误差图四列对比。

误差图颜色说明：
- **白色**：预测正确 (True Positive)
- **红色**：误诊 (False Positive，背景错判为血管)
- **绿色**：漏诊 (False Negative，血管未被识别)

### 5. 训练曲线可视化

```bash
python plot.py
```

交互式选择 results 文件，生成 3×3 指标仪表盘：
- Train Loss / Learning Rate / Dice Coefficient
- Global Accuracy / Mean IoU / 各类别 IoU 对比
- Vessel Recall vs Precision / 双轴对比 / Dice 平滑趋势

本地有 GUI 则弹窗显示，服务器无 GUI 环境自动保存为 PNG 文件。

## 模型架构

框架内置 5 种 U-Net 变体，通过 `--model` 参数自由切换。

| 模型 | 特点 |
|---|---|
| `unet` | 标准 U-Net，对称编解码 + 跳跃连接，医学图像分割经典基线 |
| `attention_unet` | 在解码器每层嵌入 Attention Gate，自动学习空间注意力权重，聚焦血管区域、抑制背景噪声（**默认**） |
| `unet++` | 密集跳跃连接 + 深度监督，通过嵌套卷积路径缩小编解码器语义差距 |
| `resunet` | 编码器引入残差块，缓解深层网络退化，加速收敛 |
| `nnunet` | nnU-Net 风格的自适应配置，自动调整归一化与激活策略 |

## 损失函数

```
Loss = CrossEntropy(weight=[1.0, 2.0]) + dice_weight × DiceLoss(仅血管类)
```

- CE loss 带类别权重缓解不平衡（前景:背景 ≈ 1:10）
- Dice loss **仅计算血管通道**，排除背景，与评估指标一致
- `dice_weight` 可调节 CE 与 Dice 的比例

## 数据增强

| 增强 | 说明 | 默认 |
|---|---|---|
| RandomResize | 随机缩放 (0.5x ~ 1.2x) | ON |
| RandomRotation | 随机旋转 ±30° | ON |
| RandomHorizontalFlip | 随机水平翻转 (p=0.5) | ON |
| RandomVerticalFlip | 随机垂直翻转 (p=0.5) | ON |
| RandomCrop | 随机裁剪至 480×480 | ON |
| ElasticTransform | 弹性形变 (α=50, σ=5) | `--use-elastic` |
| CLAHE | 限制对比度直方图均衡化 | `--use-color` |
| ColorJitter | 亮度/对比度/饱和度/色调抖动 | `--use-color` |
| RandomGamma | 随机 Gamma 校正 | `--use-color` |
| RandomGaussianNoise | 随机高斯噪声 | `--use-noise` |
| Normalize | 按 DRIVE 统计值归一化 | ON |

## 评估指标

训练过程中每 epoch 在测试集上输出：
- **Dice Coefficient**（血管，排除背景）
- **Confusion Matrix** → Global Accuracy / Per-class Recall / Per-class Precision / Per-class IoU / Mean IoU

## 训练技巧与优化记录

1. **Dice loss 排除背景** — 训练与评估一致，避免背景高 Dice 稀释血管梯度
2. **Warmup 10 epoch + Poly 衰减** — 给模型充足的初期稳定时间
3. **可调 Dice 权重** — 通过 `--dice-weight` 让模型更关注血管分割质量
4. **Attention Gate** — 自动学习空间注意力，抑制非血管区域
5. **256 作为忽略标签** — ROI 视野外区域不参与损失计算

## 参考结果 (Attention U-Net, base_c=32, 200 epochs)

| 指标 | 数值 |
|---|---|
| Dice (血管) | ~0.805 |
| IoU (血管) | ~67.5% |
| Recall (血管) | ~80.1% |
| Precision (血管) | ~81.1% |
| Global Accuracy | ~95.1% |
