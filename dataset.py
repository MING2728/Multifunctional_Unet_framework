import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

#优化原DriveDataset至DriveDataset_Pro
#增加自动拼接路径、拼接掩码的功能，简化了数据预处理流程
#引入基于 collate_fn 和cat_list的动态边界填充
class DriveDataset_Pro(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset_Pro, self).__init__()
         # 确定是用于训练还是测试
        self.flag = "training" if train else "test"
         # 基于给定的根目录和确定的状态创建数据根路径
        data_root = os.path.join(root, "DRIVE", self.flag)
        # 检查数据根路径是否存在，如果不存在则引发断言错误
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        # 存储提供的数据转换方法
        self.transforms = transforms
         # 获取以 '.tif' 结尾的图像文件名列表，这些文件在 'images' 文件夹中
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
         # 创建图像文件的完整路径列表
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
         # 创建手动标注文件的路径列表，这些文件与每个图像文件对应，位于 '1st_manual' 文件夹中
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]
        # 检查手动标注文件是否存在，如果有任何文件缺失，则引发错误
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")
        # 创建感兴趣区域（ROI）掩码文件的路径列表，这些文件与每个图像文件对应，位于 'mask' 文件夹中
        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
                         for i in img_names]
        # 检查感兴趣区域掩码文件是否存在，如果有任何文件缺失，则引发错误
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        # 打开图像并转换为RGB格式
        img = Image.open(self.img_list[idx]).convert('RGB')
        # 打开手动标注的图像并转换为灰度图，然后转为numpy数组并归一化
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual) / 255
        # 打开感兴趣区域（ROI）掩码图像并转换为灰度图，再转为numpy数组并对颜色取反
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        roi_mask = 255 - np.array(roi_mask)
        # 将手动标注图像和感兴趣区域掩码叠加并进行裁剪，保证像素值在 0 到 255 之间
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 将numpy数组转回PIL格式，因为transforms方法是针对PIL格式数据进行处理
        mask = Image.fromarray(mask)
        # 如果有数据转换方法，则对图像和mask进行转换
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list) # 返回图像列表的长度，即数据集中图像的数量

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) 
                     for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

