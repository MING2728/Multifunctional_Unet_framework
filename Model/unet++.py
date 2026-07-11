from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    """基础的双重卷积块，保持特征图尺寸不变"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    """下采样模块：最大池化 + 双重卷积"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    """
    纯上采样与拼接模块。
    注意：U-Net++ 的 Up 模块内部不进行卷积，仅负责尺寸对齐与通道拼接。
    """
    def __init__(self, in_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # 尺寸对齐
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        # 仅拼接，不卷积
        return torch.cat([x2, x1], dim=1)


class NestedDoubleConv(nn.Module):
    """
    嵌套的双重卷积。
    在 U-Net++ 中，每一层的输出是由该层输入和上一层上采样结果拼接后经过此模块处理。
    """

    def __init__(self, in_channels, out_channels):
        super(NestedDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2, bilinear: bool = True, base_c: int = 32):
        super(UNetPlusPlus, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # =====================
        # 编码器 (下采样路径)
        # =====================
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        # =====================
        # 解码器 (嵌套上采样路径)
        # =====================
        # 核心原则：
        # bilinear 上采样不改变通道数；拼接通道数 = 深层特征通道数(上采样后不变) + 浅层特征通道数；深层特征通道数 = 对应 NestedDoubleConv 的 out_channels
        # --- L4 -> L3 ---
        # x4_0(512) upsample + x3_0(256) = 768 -> out: 128
        self.up4_0 = Up(base_c * 16 // factor, bilinear)
        self.nested3_1 = NestedDoubleConv(base_c * 16 // factor + base_c * 8, base_c * 8 // factor)  # in=768, out=128

        # --- L3 -> L2 ---
        # x3_0(256) upsample + x2_0(128) = 384 -> out: 64
        self.up3_0 = Up(base_c * 8, bilinear)
        self.nested2_1 = NestedDoubleConv(base_c * 8 + base_c * 4, base_c * 4 // factor)  # in=384, out=64
        # x3_1(128) upsample + x2_1(64) = 192 -> out: 64
        self.up3_1 = Up(base_c * 8 // factor, bilinear)
        self.nested2_2 = NestedDoubleConv(base_c * 8 // factor + base_c * 4 // factor, base_c * 4 // factor)  # in=192, out=64

        # --- L2 -> L1 ---
        # x2_0(128) upsample + x1_0(64) = 192 -> out: 32
        self.up2_0 = Up(base_c * 4, bilinear)
        self.nested1_1 = NestedDoubleConv(base_c * 4 + base_c * 2, base_c * 2 // factor)  # in=192, out=32
        # x2_1(64) upsample + x1_1(32) = 96 -> out: 32
        self.up2_1 = Up(base_c * 4 // factor, bilinear)
        self.nested1_2 = NestedDoubleConv(base_c * 4 // factor + base_c * 2 // factor, base_c * 2 // factor)  # in=96, out=32
        # x2_2(64) upsample + x1_2(32) = 96 -> out: 32
        self.up2_2 = Up(base_c * 4 // factor, bilinear)
        self.nested1_3 = NestedDoubleConv(base_c * 4 // factor + base_c * 2 // factor, base_c * 2 // factor)  # in=96, out=32

        # --- L1 -> L0 ---
        # x1_0(64) upsample + x0_0(32) = 96 -> out: 32
        self.up1_0 = Up(base_c * 2, bilinear)
        self.nested0_1 = NestedDoubleConv(base_c * 2 + base_c, base_c)  # in=96, out=32
        # x1_1(32) upsample + x0_1(32) = 64 -> out: 32
        self.up1_1 = Up(base_c * 2 // factor, bilinear)
        self.nested0_2 = NestedDoubleConv(base_c + base_c, base_c)  # in=64, out=32
        # x1_2(32) upsample + x0_2(32) = 64 -> out: 32
        self.up1_2 = Up(base_c * 2 // factor, bilinear)
        self.nested0_3 = NestedDoubleConv(base_c + base_c, base_c)  # in=64, out=32
        # x1_3(32) upsample + x0_3(32) = 64 -> out: 32
        self.up1_3 = Up(base_c * 2 // factor, bilinear)
        self.nested0_4 = NestedDoubleConv(base_c + base_c, base_c)  # in=64, out=32

        # 输出层
        self.out_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 编码器
        x0_0 = self.in_conv(x)
        x1_0 = self.down1(x0_0)
        x2_0 = self.down2(x1_0)
        x3_0 = self.down3(x2_0)
        x4_0 = self.down4(x3_0)

        # 解码器 (严格分离上采样拼接与卷积)
        # L3
        x3_1 = self.nested3_1(self.up4_0(x4_0, x3_0))
        # L2
        x2_1 = self.nested2_1(self.up3_0(x3_0, x2_0))
        x2_2 = self.nested2_2(self.up3_1(x3_1, x2_1))
        # L1
        x1_1 = self.nested1_1(self.up2_0(x2_0, x1_0))
        x1_2 = self.nested1_2(self.up2_1(x2_1, x1_1))
        x1_3 = self.nested1_3(self.up2_2(x2_2, x1_2))
        # L0
        x0_1 = self.nested0_1(self.up1_0(x1_0, x0_0))
        x0_2 = self.nested0_2(self.up1_1(x1_1, x0_1))
        x0_3 = self.nested0_3(self.up1_2(x1_2, x0_2))
        x0_4 = self.nested0_4(self.up1_3(x1_3, x0_3))

        logits = self.out_conv(x0_4)
        return {"out": logits}