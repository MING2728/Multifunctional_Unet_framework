import torch

def create_lr_scheduler(optimizer,           #指定要更新lr的优化器
                        num_step: int,       #每个epoch的步数（迭代次数）
                        epochs: int,         #总共的训练轮次epoch数
                        warmup=True,         #是否在训练初期使用warmup策略
                        warmup_epochs=1,     #warmup策略持续的epoch数
                        warmup_factor=1e-3,  #warmup策略中lr的倍率因子
                        power=0.9):          #预热后学习率衰减的幂指数
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据当前step数返回一个学习率倍率因子
        每批次训练会调用一次lr_scheduler.step()
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / 
                    ((epochs - warmup_epochs) * num_step)) ** power
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
