import os
import time
import datetime
import importlib
import torch

from Model.unet import UNet
from Model.attention_unet import AttentionUNet
from Model.resunet import ResUNet
from Model.nnunet import NNUNet
from dataset import DriveDataset_Pro
from Tools.train_utils import train_one_epoch, evaluate
from Tools.lr_utils import create_lr_scheduler
from Tools.preprocess_utils import get_preset_transform


_MODEL_REGISTRY = {
    'unet':            UNet,
    'attention_unet':  AttentionUNet,
    'resunet':         ResUNet,
    'nnunet':          NNUNet,
}


def create_model(num_classes, model_name='attention_unet'):
    """根据模型名称创建对应的分割模型。"""
    if model_name == 'unet++':
        mod = importlib.import_module('Model.unet++')
        model_cls = mod.UNetPlusPlus
    elif model_name in _MODEL_REGISTRY:
        model_cls = _MODEL_REGISTRY[model_name]
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {', '.join(_MODEL_REGISTRY.keys())}, unet++"
        )
    return model_cls(in_channels=3, num_classes=num_classes, base_c=32)


def main(args):
    # 获取设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 批次大小
    batch_size = args.batch_size
    # 分割类别数（包括背景）
    num_classes = args.num_classes + 1

    # 图像均值和标准差
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # 用于保存训练和验证信息的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 打印增强配置
    print(f"几何增强(elastic): {'ON' if args.use_elastic else 'OFF'}")
    print(f"色彩/亮度增强: {'ON' if args.use_color else 'OFF'}")
    print(f"噪声增强: {'ON' if args.use_noise else 'OFF'}")

    # 创建训练和测试数据集
    train_dataset = DriveDataset_Pro(args.data_path,
                                    train=True,
                                    transforms=get_preset_transform(
                                        train=True, mean=mean, std=std,
                                        use_elastic=args.use_elastic,
                                        use_color=args.use_color,
                                        use_noise=args.use_noise))

    val_dataset = DriveDataset_Pro(args.data_path,
                                    train=False,
                                    transforms=get_preset_transform(train=False, mean=mean, std=std))

    #num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 计算可用的 worker 数量，限制在最小的工作进程数和一些条件下的最小值
    num_workers = 0
    train_loader = torch.utils.data.DataLoader(train_dataset,  # 创建训练数据加载器
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=False,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset, # 创建验证数据加载器
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=False,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes, model_name=args.model)  # 创建模型实例
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad] # 获取需优化的参数
    # 创建优化器
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    # 创建混合精度训练的梯度缩放器（如果开启混合精度训练）
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
        print("有开启混合精度训练") #测试用
    else:
        scaler = None  

    # 创建学习率更新策略，这里是每个step更新一次（不是每个epoch）
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=10)
     # 如果设置了恢复训练
    if args.resume:
        # 加载之前保存的模型状态
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        # 如果开启了混合精度训练，还需恢复梯度缩放器状态
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    # 初始化最佳 Dice 分数和开始时间
    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 训练一个 epoch
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, 
                                        scaler=scaler, dice_weight=args.dice_weight)
        # 在验证集上评估模型性能
        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # 将结果写入到文件中
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")
        # 如果开启了保存最佳模型
        if args.save_best is True:
            # 如果当前 Dice 值优于历史最佳，则更新最佳 Dice 值
            if best_dice < dice:
                best_dice = dice
                print(f"当前dice最佳 →→{best_dice:.3f}，保存最佳模型")
            else:
                print(f"当前dice({dice:.3f})不如最佳 {best_dice:.3f}，继续训练")
                print("\n") #与下一个轮次隔开
                continue
        # 准备要保存的模型状态
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        # 如果开启了混合精度训练，还需保存梯度缩放器的状态
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        # 根据条件选择保存最佳模型或每个 epoch 的模型
        if args.save_best is True:
            torch.save(save_file, "save_weights/best_model.pth")
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))
        print("\n") #与下一个轮次隔开
    
    # 计算总训练时间并打印
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
    with open(results_file, "a") as f:
        f.write("training time: {} \n\n".format(total_time_str))
        if args.amp:
            f.write("有开启混合精度训练")  #写入results文件
        else: 
            f.write("没开启混合精度训练")  #写入results文件


def parse_args_train():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./", help="DRIVE root")
    
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # 模型选择
    parser.add_argument('--model', default='attention_unet', type=str,
                        choices=['unet', 'attention_unet', 'unet++', 'resunet', 'nnunet'],
                        help='model architecture')
    # 混合精度训练参数
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    # 数据增强控制
    parser.add_argument('--use-elastic', default=False, action='store_true',
                        help='enable ElasticTransform')
    parser.add_argument('--use-color', default=False, action='store_true',
                        help='enable CLAHE + ColorJitter + RandomGamma')
    parser.add_argument('--use-noise', default=False, action='store_true',
                        help='enable RandomGaussianNoise')
    # 损失函数控制
    parser.add_argument('--dice-weight', default=1.0, type=float,
                        help='Dice loss 权重系数 (loss = CE + dice_weight * Dice)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args_train()
    # 如果保存模型的文件夹不存在，则创建它
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")
    # 执行主程序入口函数
    main(args)
