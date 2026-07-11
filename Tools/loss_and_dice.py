import torch
from torch import nn

def make_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    # 为dice_loss的计算创建符合“独热编码”的目标张量
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # 计算一整个batch中所有图片的“某个类别（通道）”的dice_coefficient
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)


# 综合交叉熵损失和Dice损失的总损失函数（在train_utils.py中调用）
# 联动dice_loss和make_target函数，实现不感兴趣区域抑制及多类别支持
def loss_fn(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True,
            ignore_index: int = -100, dice_weight: float = 1.0):
    """
    dice_weight: Dice loss 的权重系数，用于调节 CE 和 Dice 的比例。
    loss_weight: CE 损失中各类别的权重系数，用于应对类别不平衡问题。
    """
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        loss = nn.functional.cross_entropy(x, target, 
                                           ignore_index=ignore_index, 
                                           weight=loss_weight)
        if dice is True:
            dice_target = make_target(target, num_classes, ignore_index)
            loss += dice_weight * dice_loss(x[:, 1:], dice_target[:, 1:], 
                                            multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']
