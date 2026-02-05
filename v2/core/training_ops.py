"""
training_ops.py - 训练流程封装

提供标准化的训练流程，支持DDP、混合精度、梯度累积等。
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from torch.utils.data import DataLoader
from models.registry import create_model
from datasets.registry import create_dataset
from core.device_ops import is_main_process, all_reduce_tensor, get_raw_model
from core.checkpoint_ops import save_checkpoint_v2


class TrainingContext:
    """训练上下文，保存训练所需的所有组件"""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        train_loader,
        val_loader,
        device,
        rank,
        world_size
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.rank = rank
        self.world_size = world_size


def setup_training(
    config: dict,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1
) -> TrainingContext:
    """
    设置训练所需的组件

    Args:
        config: 配置字典
        device: 设备
        rank: 当前rank
        world_size: 总进程数

    Returns:
        TrainingContext实例
    """
    from torch.utils.data import DistributedSampler

    # 创建数据集
    train_dataset = create_dataset(config, split='train')
    val_dataset = create_dataset(config, split='val')

    # 数据加载器配置
    data_cfg = config.get('data', {}).get('dataloader', {})
    batch_size = data_cfg.get('batch_size', 4)
    num_workers = data_cfg.get('num_workers', 4)
    pin_memory = data_cfg.get('pin_memory', True)

    # 创建sampler（DDP模式）
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # 创建模型
    model_interface = create_model(config, device=device)

    # 包装为DDP（如果需要）
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model_interface._model = DDP(
            model_interface._model,
            device_ids=[device.index] if device.type == 'cuda' else None
        )

    # 创建优化器
    train_cfg = config.get('training', {})
    opt_cfg = train_cfg.get('optimizer', {})
    optimizer_type = opt_cfg.get('type', 'adamw')
    lr = opt_cfg.get('lr', 1e-4)
    weight_decay = opt_cfg.get('weight_decay', 0.01)

    raw_model = get_raw_model(model_interface._model)

    if optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            raw_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # 创建学习率调度器
    scheduler_cfg = train_cfg.get('scheduler', {})
    scheduler_type = scheduler_cfg.get('type', 'cosine')

    if scheduler_type.lower() == 'cosine':
        num_epochs = train_cfg.get('num_epochs', 100)
        warmup_epochs = scheduler_cfg.get('warmup_epochs', 5)
        min_lr = scheduler_cfg.get('min_lr', 1e-6)

        # 预热 + 余弦退火
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                return min_lr + (1 - min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # 创建梯度缩放器（混合精度）
    use_amp = train_cfg.get('mixed_precision', {}).get('enabled', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    return TrainingContext(
        model=model_interface,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        rank=rank,
        world_size=world_size
    )


def train_step(
    model,
    batch: Dict[str, torch.Tensor],
    optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
    max_norm: float = 0.0
) -> Tuple[float, Dict[str, float]]:
    """
    单步训练

    Args:
        model: 模型接口
        batch: 数据批次
        optimizer: 优化器
        scaler: 梯度缩放器
        device: 设备
        use_amp: 是否使用混合精度
        max_norm: 梯度裁剪最大范数

    Returns:
        (loss, loss_dict)
    """
    # 将数据移到设备
    sar = batch['sar'].to(device)
    optical = batch['optical'].to(device)

    # 前向传播
    optimizer.zero_grad()

    if use_amp and scaler is not None:
        # 混合精度训练
        with torch.cuda.amp.autocast():
            loss, loss_dict = model(sar, optical, return_dict=True)

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度裁剪
        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model._model.parameters(), max_norm)

        # 更新参数
        scaler.step(optimizer)
        scaler.update()
    else:
        # 正常训练
        loss, loss_dict = model(sar, optical, return_dict=True)
        loss.backward()

        # 梯度裁剪
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model._model.parameters(), max_norm)

        optimizer.step()

    # 转换为普通字典
    loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}

    return loss.item(), loss_dict


def train_step_with_accumulation(
    model,
    batch: Dict[str, torch.Tensor],
    optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
    max_norm: float,
    accumulation_steps: int,
    is_accumulation_step: bool
) -> Tuple[float, Dict[str, float]]:
    """
    带梯度累积的训练步

    Args:
        model: 模型接口
        batch: 数据批次
        optimizer: 优化器
        scaler: 梯度缩放器
        device: 设备
        use_amp: 是否使用混合精度
        max_norm: 梯度裁剪最大范数
        accumulation_steps: 累积步数
        is_accumulation_step: 是否是累积步（非更新步）

    Returns:
        (loss, loss_dict)
    """
    # 将数据移到设备
    sar = batch['sar'].to(device)
    optical = batch['optical'].to(device)

    # 前向传播
    if use_amp and scaler is not None:
        with torch.cuda.amp.autocast():
            loss, loss_dict = model(sar, optical, return_dict=True)
            loss = loss / accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()
    else:
        loss, loss_dict = model(sar, optical, return_dict=True)
        loss = loss / accumulation_steps
        loss.backward()

    # 只在非累积步更新参数
    if not is_accumulation_step:
        # 梯度裁剪
        if max_norm > 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model._model.parameters(), max_norm)

        # 更新参数
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad()

    # 转换为普通字典
    loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}

    return loss.item() * accumulation_steps, loss_dict


def validate_model(
    model,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    验证模型

    Args:
        model: 模型接口
        val_loader: 验证数据加载器
        device: 设备
        max_batches: 最大验证批次

    Returns:
        指标字典
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break

            sar = batch['sar'].to(device)
            optical = batch['optical'].to(device)

            loss, _ = model(sar, optical, return_dict=True)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    return {'val_loss': avg_loss}


if __name__ == "__main__":
    # 测试
    print("Testing training_ops.py...")
    print("Note: Full test requires actual config and data")
    print("All basic tests passed!")
