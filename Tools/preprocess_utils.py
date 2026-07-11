import Tools.transform_utils as JM

class PresetTrain:
    def __init__(self, base_size, crop_size,
                 hflip_prob=0.5, vflip_prob=0.5,
                 rotation_degrees=30,
                 use_elastic=True, 
                 use_color=False, use_noise=False,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [JM.RandomResize_Pro(min_size, max_size)]
        # ---- 色彩 / 亮度增强  ----
        if use_color:
            # CLAHE 50%概率应用，保留一半图像原始亮度分布
            trans.append(JM.RandomApply_Pro(
                JM.CLAHE_Pro(clip_limit=2.0, tile_grid_size=(8, 8)), p=0.5))
            # 颜色抖动30%概率，去掉色调偏移避免血管颜色失真
            trans.append(JM.RandomApply_Pro(
                JM.ColorJitter_Pro(
                    brightness=0.15, contrast=0.15, saturation=0.1, hue=0), p=0.3))
            # Gamma校正30%概率
            trans.append(JM.RandomApply_Pro(
                JM.RandomGamma_Pro(gamma=(0.85, 1.15)), p=0.3))
        # ---- 几何增强 ----
        if use_elastic: #默认开启
            trans.append(JM.ElasticTransform_Pro(alpha=50.0, sigma=5.0))

        trans.append(JM.RandomRotation_Pro(degrees=rotation_degrees))
        if hflip_prob > 0:
            trans.append(JM.RandomHorizontalFlip_Pro(hflip_prob))
        if vflip_prob > 0:
            trans.append(JM.RandomVerticalFlip_Pro(vflip_prob))
        trans.append(JM.RandomCrop_Pro(crop_size))
        trans.append(JM.ToTensor_Pro())

        if use_noise:
            # 高斯噪声10%概率，多数样本保持无噪声
            trans.append(JM.RandomApply_Pro(
                JM.RandomGaussianNoise_Pro(mean=0.0, sigma=(0.01, 0.05)), p=0.1))
        trans.append(JM.Normalize_Pro(mean=mean, std=std))
        self.transforms = JM.Compose_Pro(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class PresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = JM.Compose_Pro([
            JM.ToTensor_Pro(),
            JM.Normalize_Pro(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_preset_transform(train,
                         mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                         use_elastic=True, use_color=False, use_noise=False):
    base_size = 565
    crop_size = 480

    if train:
        return PresetTrain(base_size, crop_size,
                           mean=mean, std=std,
                           use_elastic=use_elastic,
                           use_color=use_color,
                           use_noise=use_noise)
    else:
        return PresetEval(mean=mean, std=std)