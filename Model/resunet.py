"""
文件名：resunet.py
实现了 ResUNet 架构（基于残差块的 U-Net）。
兼容 train.py 和 predict_v2.py 的调用接口。
"""
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    残差块 (Residual Block)。
    当输入输出通道数不一致时，通过 1x1 卷积调整维度。
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入输出通道数不同，需要调整 x 的维度以进行残差连接
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x  # 保存输入作为残差

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要，调整残差的维度
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接 (Add)
        out = self.relu(out)  # 再次激活
        return out


class Down(nn.Sequential):
    """下采样模块：最大池化 + 残差块"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            ResidualBlock(in_channels, out_channels)
        )


class Up(nn.Module):
    """上采样模块"""

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # bilinear 不改变通道数，拼接后 = in_channels + skip_channels
            self.conv = ResidualBlock(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # 转置卷积输出 in_channels//2，拼接后 = in_channels//2 + skip_channels
            self.conv = ResidualBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # 尺寸对齐
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # 拼接跳跃连接的特征图
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Sequential):
    """输出卷积层"""

    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class ResUNet(nn.Module):
    """
    ResUNet 主干网络。
    使用 ResidualBlock 替代标准卷积块。
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2, bilinear: bool = True, base_c: int = 32):
        super(ResUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 编码器 (下采样路径)
        # 第一层输入通道数可能与 base_c 不同，使用 ResidualBlock 自动处理
        self.in_conv = ResidualBlock(in_channels, base_c)

        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)

        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        # 解码器 (上采样路径)
        # Up(in_channels=深层通道数, skip_channels=对应编码器层真实输出, out_channels=本层目标输出, bilinear)
        # 编码器真实输出(bilinear=True): x1=64, x2=128, x3=256, x4=512, x5=512

        # up1: x5(512) + x4(512) = 1024 → 输出 256
        self.up1 = Up(base_c * 16 // factor, base_c * 8, base_c * 8 // factor, bilinear)
        # up2: up1_out(256) + x3(256) = 512 → 输出 128
        self.up2 = Up(base_c * 8 // factor, base_c * 4, base_c * 4 // factor, bilinear)
        # up3: up2_out(128) + x2(128) = 256 → 输出 64
        self.up3 = Up(base_c * 4 // factor, base_c * 2, base_c * 2 // factor, bilinear)
        # up4: up3_out(64) + x1(64) = 128 → 输出 64
        self.up4 = Up(base_c * 2 // factor, base_c, base_c, bilinear)

        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 编码器路径
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 输出
        logits = self.out_conv(x)

        return {"out": logits}