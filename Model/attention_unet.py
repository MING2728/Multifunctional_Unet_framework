from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Sequential):
    # 同unet的双层卷积块 —— 内含"3x3卷积 + 批量归一化 + ReLU" *2
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
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            # 最大池化层，用于下采样，将特征图尺寸缩小一半
            nn.MaxPool2d(2, stride=2),
            # 使用定义的 DoubleConv 类来构建一个特征提取块
            DoubleConv(in_channels, out_channels)
        )

class Attention_block(nn.Module):
    """
    新增的注意力门控模块 (Attention Gate)
    利用深层特征提供的高阶语义信息指导浅层特征
    过滤浅层跳连接中的无关激活值，抑制黑边、背景等非血管区域的干扰
    """
    def __init__(self, F_g, F_l, F_int):
        #F_g: 深层特征图上采样后的通道数 (gating signal)
        #F_l: 浅层跳连接传过来的特征图通道数 (skip connection)
        #F_int: 内部对齐时使用的中间通道数
        super(Attention_block, self).__init__()
        # 深层特征图的 1x1 线性变换
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        # 浅层特征图的 1x1 线性变换
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        # 融合后的特征变换，映射为单通道，并通过Sigmoid生成空间注意力权重系数
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # 深浅层特征分别进行线性变换并对齐通道
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 空间位置特征互补相加，融合宏观语义与微观结构，激活后映射为0~1的权重系数
        combined_activation = self.relu(g1 + x1)
        # 计算自适应空间注意力系数矩阵 (Spatially Coefficient Matrix)
        attention_weights = self.psi(combined_activation)
        
        # 将空间注意力系数分配给原始浅层特征（即权重矩阵与原始浅层特征点乘），
        # 实现选择性过滤，强行抑制非血管区域的异常激活值
        return x * attention_weights


class Up(nn.Module):
    """
    上采样与特征融合块：融合了 Attention 机制
    在 concat 拼接前，强制跳连接的浅层特征过一遍注意力门
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # 根据输入的参数决定使用双线性插值还是转置卷积
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 引入注意力门：深层上采样后通道为 in_channels // 2，浅层通道亦为 in_channels // 2
            self.att = Attention_block(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # 引入注意力门
            self.att = Attention_block(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1 为深层低分辨率特征图，首先执行上采样（空间分辨率拉伸）
        x1 = self.up(x1)
        
        # 通过两组特征图的几何尺寸差异执行动态对齐填充，确保维度对齐
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        # 新增：在通道维度拼接前，用上采样后的深层高阶特征 x1 为闸门，过滤浅层特征 x2
        x2 = self.att(g=x1, x=x2)
        
        # 将空间上纯净的血管特征与深层特征沿通道维度进行聚合拼装
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x

class OutConv(nn.Sequential):
    # 同unet的输出映射卷积层 —— 用1x1卷积将特征图通道数映射为类别数
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class AttentionUNet(nn.Module):
    """
    Attention U-Net 顶层语义分割网络架构
    """
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(AttentionUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        # 编码器核心组件定义
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        
        # 解码器核心组件定义 (内部已自动集成了 Attention_block)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. 编码器前向提取流
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 2. 解码器门控融合流
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 3. 映射逻辑输出
        logits = self.out_conv(x)
        
        # 严格保持原版的字典格式输出，防止训练流报错
        return {"out": logits}