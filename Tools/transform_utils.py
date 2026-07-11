import numpy as np
import random
import torch
import numbers
import cv2
from PIL import Image

from torchvision import transforms as T
from torchvision.transforms import functional as F



def pad(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose_Pro(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize_Pro(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip_Pro(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip_Pro(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop_Pro(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad(image, self.size)
        target = pad(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop_Pro(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class RandomRotation_Pro(object):
    def __init__(self, degrees):
        # 接收元组 (min_angle, max_angle) 或一个数值 (表示 [-degrees, degrees])
        if isinstance(degrees, numbers.Number):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees

    def __call__(self, image, target):
        # 生成一个随机旋转角度
        angle = random.uniform(self.degrees[0], self.degrees[1])
        
        # 原图 (默认双线性插值，旋转产生的空区补 0)
        image = F.rotate(image, angle, fill=0)
        # 金标准 (指定NEAREST! 旋转产生的空区补 255)
        target = F.rotate(target, angle, 
                          interpolation=T.InterpolationMode.NEAREST, fill=255)
        
        return image, target


class ToTensor_Pro(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize_Pro(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


# ============================================================
#  新增数据增强类（每个以"Pro"命名：双图片同步变换，底层调torchvision官方API）
# ============================================================

class CLAHE_Pro(object):
    """
    限制对比度自适应直方图均衡化，仅作用于图像。
    RGB -> LAB，在 L 亮度通道上做 CLAHE，再 LAB -> RGB。
    医学图像分割标准做法，色彩无偏、血管对比度显著提升。
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image, target):
        # 将 PIL 图像转换为 OpenCV 兼容的 NumPy BGR/RGB 矩阵
        img_np = np.array(image)
        # RGB -> LAB
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        # 实例化 OpenCV 限制对比度直方图生成器
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        # CLAHE 仅作用于 L 亮度通道
        l_eq = clahe.apply(l)
        # 合并后 LAB -> RGB
        lab_eq = cv2.merge((l_eq, a, b))
        img_np = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        image = Image.fromarray(img_np)
        return image, target


class ColorJitter_Pro(object):
    """
    随机颜色抖动（亮度/对比度/饱和度/色调）, 仅作用于原始图像。
    参数与 torchvision.transforms.ColorJitter 一致：
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_factor(brightness)
        self.contrast = self._check_factor(contrast)
        self.saturation = self._check_factor(saturation)
        self.hue = self._check_hue(hue)

    @staticmethod
    def _check_factor(value):
        if isinstance(value, (tuple, list)):
            return tuple(value)
        if value == 0:
            return None
        return (max(0, 1 - value), 1 + value)

    @staticmethod
    def _check_hue(value):
        if isinstance(value, (tuple, list)):
            return tuple(value)
        if value == 0:
            return None
        return (max(-0.5, -value), min(0.5, value))

    def __call__(self, image, target):
        # 随机打乱四个颜色变换算子的执行顺序 (防止级联偏向性)
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness:
                image = F.adjust_brightness(image, random.uniform(*self.brightness))
            elif fn_id == 1 and self.contrast:
                image = F.adjust_contrast(image, random.uniform(*self.contrast))
            elif fn_id == 2 and self.saturation:
                image = F.adjust_saturation(image, random.uniform(*self.saturation))
            elif fn_id == 3 and self.hue:
                image = F.adjust_hue(image, random.uniform(*self.hue))
        return image, target


class RandomGamma_Pro(object):
    """
    随机Gamma校正，仅作用于图像。
    原理: pixel' = 255 * (pixel/255)^γ。gamma<1变亮，gamma>1变暗。
    """
    def __init__(self, gamma=(0.85, 1.15)):
        self.gamma = gamma

    def __call__(self, image, target):
        # 在安全的医学灰度膨胀区间内随机采样一个 gamma 因子
        gamma_factor = random.uniform(*self.gamma)
        # 调用 torchvision 底层 functional API 执行非线性幂律转换
        image = F.adjust_gamma(image, gamma_factor)
        return image, target


class RandomGaussianNoise_Pro(object):
    """
    随机高斯噪声注入，仅作用于图像(Tensor)。
    应放在ToTensor_Pro之后、Normalize_Pro之前。
    """
    def __init__(self, mean=0.0, sigma=(0.01, 0.05)):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, image, target):
        # 动态采样本次迭代的噪声强度标准差
        current_sigma = random.uniform(*self.sigma)
        image = image + torch.randn(image.shape) * current_sigma + self.mean
        # 将像素值裁剪到 [0, 1] 范围内，避免溢出
        image.clamp_(0.0, 1.0)
        return image, target


class RandomAffine_Pro(object):
    """
    随机仿射变换（旋转+平移+缩放+剪切组合），同时作用于图像和mask。
    参数与 torchvision.transforms.RandomAffine 一致。
    图像双线性插值补0，mask最近邻插值补255（忽略标签）。
    """
    def __init__(self, degrees=0, translate=None, scale=None, shear=None):
        if isinstance(degrees, numbers.Number):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, image, target):
        # 与官方 RandomAffine.forward 一致：img_size = [width, height]
        angle, translations, scale_factor, shear_vals = T.RandomAffine.get_params(
            self.degrees, self.translate, self.scale, self.shear,
            [image.width, image.height])

        image = F.affine(image, angle, translations, scale_factor, shear_vals, fill=0)
        target = F.affine(target, angle, translations, scale_factor, shear_vals,
                          interpolation=T.InterpolationMode.NEAREST, fill=255)
        return image, target


class ElasticTransform_Pro(object):
    """
    弹性形变——UNet论文核心增强，医学图像分割标配。
    同时作用于图像和mask，与 torchvision.transforms.ElasticTransform 实现一致：
        图像用双线性插值补0，mask用最近邻插值补255。
    """
    def __init__(self, alpha=50.0, sigma=5.0):
        # 与官方 ElasticTransform.__init__ 一致：单float展开为 [float, float]
        if isinstance(alpha, numbers.Number):
            alpha = [float(alpha), float(alpha)]
        if isinstance(sigma, numbers.Number):
            sigma = [float(sigma), float(sigma)]
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image, target):
        displacement = T.ElasticTransform.get_params(
            self.alpha, self.sigma, [image.height, image.width])

        # image: PIL → [3, H, W] float32
        img_t = F.to_tensor(image)
        img_t = F.elastic_transform(img_t, displacement,
                                    interpolation=T.InterpolationMode.BILINEAR, fill=[0.0])
        image = F.to_pil_image(img_t.clamp(0, 1))

        # mask: PIL → [1, H, W] float32（F.elastic_transform 要求三维）
        tgt_np = np.array(target)
        # 张量化与高维通道强锁定，强开第一维，升维至安全计算边界 [1, H, W]
        tgt_t = torch.as_tensor(tgt_np, dtype=torch.float32).unsqueeze(0)
        # 高维空间级联形变，强制指定——最近邻插值与 255 越界屏蔽
        tgt_t = F.elastic_transform(tgt_t, displacement,
                                    interpolation=T.InterpolationMode.NEAREST, fill=[255.0])
        #内存就地逆向解构，剥离临时通道轴
        tgt_np_out = tgt_t.squeeze(0).numpy().astype(np.uint8)
        target = Image.fromarray(tgt_np_out)
        return image, target


class RandomApply_Pro(object):
    """按概率 p 随机应用某个变换。p=0 永不应用, p=1 永远应用。"""
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1).item() < self.p:
            return self.transform(image, target)
        return image, target
