import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDropoutNormReLU(nn.Module):
    """nnU-Net基础卷积块：Conv3x3 + BatchNorm + LeakyReLU + Dropout"""

    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class StackedConvBlocks(nn.Module):
    """nnU-Net堆叠卷积模块：每个分辨率层级使用两个ConvDropoutNormReLU"""

    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvDropoutNormReLU(in_channels, out_channels, dropout_p),
            ConvDropoutNormReLU(out_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.blocks(x)


class DownsampleBlock(nn.Module):
    """nnU-Net下采样块：使用stride=2的3x3卷积替代MaxPool进行下采样"""

    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        self.conv_block = StackedConvBlocks(out_channels, out_channels, dropout_p)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv_block(x)
        return x


class UpsampleBlock(nn.Module):
    """nnU-Net上采样块：双线性插值上采样 + 拼接 + 堆叠卷积"""

    def __init__(self, in_channels, skip_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.conv_block = StackedConvBlocks(in_channels + skip_channels, out_channels, dropout_p)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # 处理尺寸不匹配
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)
        return x


class NNUNet(nn.Module):
    """
    nnU-Net 2D实现，兼容train.py和predict_v2.py的调用接口。
    核心改进：InstanceNorm、LeakyReLU、strided conv下采样、无bias卷积。
    """

    def __init__(self, in_channels=3, num_classes=2, base_c=32, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_c = base_c

        # 编码器路径
        self.inc = StackedConvBlocks(in_channels, base_c)
        self.down1 = DownsampleBlock(base_c, base_c * 2)
        self.down2 = DownsampleBlock(base_c * 2, base_c * 4)
        self.down3 = DownsampleBlock(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = DownsampleBlock(base_c * 8, base_c * 16 // factor)

        # 解码器路径
        self.up1 = UpsampleBlock(base_c * 16 // factor, base_c * 8, base_c * 8 // factor)
        self.up2 = UpsampleBlock(base_c * 8 // factor, base_c * 4, base_c * 4 // factor)
        self.up3 = UpsampleBlock(base_c * 4 // factor, base_c * 2, base_c * 2 // factor)
        self.up4 = UpsampleBlock(base_c * 2 // factor, base_c, base_c)

        # 输出层：1x1卷积 + bias
        self.outc = nn.Conv2d(base_c, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return {"out": logits} 