import torch
import Tools.monitor_utils as monitor
import Tools.metrics_utils as metrics
from Tools.loss_and_dice import loss_fn

# 以evaluate替代原有evaluate用于优化性能评估方法（于train.py主循环调用）
def evaluate(model, data_loader, device, num_classes):
    model.eval()
    #新增——metrics工具计算混淆矩阵和Dice系数等指标
    confmat = metrics.ConfusionMatrix(num_classes)
    dice = metrics.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = monitor.MetricLogger(delimiter="  ")
    header = 'Test Result:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item()

# 以train_one_epoch替代train用于训练一个完整轮次（于train.py主循环调用）
def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None, dice_weight=1.0):
    model.train()
    metric_logger = monitor.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', monitor.smoother(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        # 新增——设置cross_entropy中背景和前景的loss权重，以应对类别不平衡问题
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    #新增——加入monitor，监视训练过程中的loss和学习率变化
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.amp.autocast(device_type='cuda',enabled=scaler is not None):
            output = model(image)
            loss = loss_fn(output, target, loss_weight, num_classes=num_classes,
                          ignore_index=255, dice_weight=dice_weight)
        
        #新增——混合精度训练
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


