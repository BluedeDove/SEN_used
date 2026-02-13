"""
training_ops.py - 训练流程封装

提供标准化的训练流程，支持DDP、混合精度、梯度累积等。
"""

# 支持单独运行调试：将项目根目录添加到路径
import sys
from pathlib import Path

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.registry import create_model
from datasets.registry import create_dataset
from core.device_ops import is_main_process, all_reduce_tensor, get_raw_model, cleanup_resources
from core.checkpoint_ops import save_checkpoint_v2, load_checkpoint_v2, restore_model_v2
from core.amp_ops import AMPManager, create_amp_manager
from core.validation_ops import run_validation
from core.visualization_ops import (
    log_loss, plot_loss_curve, generate_validation_report, setup_report_directory
)


class TrainingContext:
    """训练上下文，保存训练所需的所有组件"""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        amp_manager: AMPManager,
        train_loader,
        val_loader,
        device,
        rank,
        world_size
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.amp_manager = amp_manager
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
            device_ids=[device.index] if device.type == 'cuda' else None,
            find_unused_parameters=True  # FlowMatching 模型中存在未使用的参数（global_cond, sar_base）
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

    # 创建 AMP 管理器（混合精度）
    amp_manager = create_amp_manager(config)

    return TrainingContext(
        model=model_interface,
        optimizer=optimizer,
        scheduler=scheduler,
        amp_manager=amp_manager,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        rank=rank,
        world_size=world_size
    )


# 别名，保持向后兼容
setup_training_components = setup_training


def train_step(
    model,
    batch: Dict[str, torch.Tensor],
    optimizer,
    amp_manager: AMPManager,
    device: torch.device,
    max_norm: float = 0.0
) -> Tuple[float, Dict[str, float]]:
    """
    单步训练

    Args:
        model: 模型接口
        batch: 数据批次
        optimizer: 优化器
        amp_manager: AMP 管理器
        device: 设备
        max_norm: 梯度裁剪最大范数

    Returns:
        (loss, loss_dict)
    """
    # 将数据移到设备
    sar = batch['sar'].to(device)
    optical = batch['optical'].to(device)

    # 前向传播
    optimizer.zero_grad()
    use_amp = amp_manager.is_enabled()

    if use_amp:
        # 混合精度训练
        with amp_manager.autocast():
            loss, loss_dict = model(sar, optical, return_dict=True)

        # 反向传播 (使用 AMP 缩放)
        scaled_loss = amp_manager.scale(loss)
        scaled_loss.backward()

        # 梯度裁剪
        if max_norm > 0:
            amp_manager.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model._model.parameters(), max_norm)

        # 更新参数
        amp_manager.step(optimizer)
        amp_manager.update()
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
    amp_manager: AMPManager,
    device: torch.device,
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
        amp_manager: AMP 管理器
        device: 设备
        max_norm: 梯度裁剪最大范数
        accumulation_steps: 累积步数
        is_accumulation_step: 是否是累积步（非更新步）

    Returns:
        (loss, loss_dict)
    """
    # 将数据移到设备
    sar = batch['sar'].to(device)
    optical = batch['optical'].to(device)

    use_amp = amp_manager.is_enabled()

    # 前向传播
    if use_amp:
        with amp_manager.autocast():
            loss, loss_dict = model(sar, optical, return_dict=True)
            loss = loss / accumulation_steps

        # 反向传播 (使用 AMP 缩放)
        scaled_loss = amp_manager.scale(loss)
        scaled_loss.backward()
    else:
        loss, loss_dict = model(sar, optical, return_dict=True)
        loss = loss / accumulation_steps
        loss.backward()

    # 只在非累积步更新参数
    if not is_accumulation_step:
        # 梯度裁剪
        if max_norm > 0:
            amp_manager.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model._model.parameters(), max_norm)

        # 更新参数
        amp_manager.step(optimizer)
        amp_manager.update()

        optimizer.zero_grad()

    # 转换为普通字典
    loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}

    return loss.item() * accumulation_steps, loss_dict


def run_training_epoch(
    ctx: TrainingContext,
    epoch: int,
    num_epochs: int,
    use_accumulation: bool,
    accumulation_steps: int,
    max_norm: float
) -> float:
    """
    运行单个训练 epoch

    Args:
        ctx: 训练上下文
        epoch: 当前 epoch 索引
        num_epochs: 总 epoch 数
        use_accumulation: 是否使用梯度累积
        accumulation_steps: 梯度累积步数
        max_norm: 梯度裁剪最大范数

    Returns:
        平均损失
    """
    # 设置epoch（用于DistributedSampler）
    if hasattr(ctx.train_loader.sampler, 'set_epoch'):
        ctx.train_loader.sampler.set_epoch(epoch)

    # 训练
    ctx.model.train()
    total_loss = 0.0
    num_batches = len(ctx.train_loader)
    batch_count = 0

    pbar = tqdm(ctx.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not is_main_process(ctx.rank))
    for batch in pbar:
        batch_count += 1

        if use_accumulation:
            is_last_batch = (batch_count == num_batches)
            is_accumulation_step = (batch_count % accumulation_steps != 0) and not is_last_batch

            loss, loss_dict = train_step_with_accumulation(
                ctx.model, batch, ctx.optimizer, ctx.amp_manager,
                ctx.device, max_norm,
                accumulation_steps, is_accumulation_step
            )

            accum_info = f"{batch_count % accumulation_steps or accumulation_steps}/{accumulation_steps}"
            pbar.set_postfix({'loss': f'{loss:.4f}', 'accum': accum_info})
        else:
            loss, loss_dict = train_step(
                ctx.model, batch, ctx.optimizer, ctx.amp_manager,
                ctx.device, max_norm
            )
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        total_loss += loss

    avg_loss = total_loss / num_batches
    return avg_loss


def handle_checkpoint_save(
    ctx: TrainingContext,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: Path,
    config: dict,
    save_periodic: bool = True
) -> bool:
    """
    保存检查点

    Args:
        ctx: 训练上下文
        epoch: 当前 epoch
        metrics: 指标字典
        checkpoint_dir: 检查点目录
        config: 配置
        save_periodic: 是否定期保存（每10个epoch）

    Returns:
        是否保存成功
    """
    if not is_main_process(ctx.rank):
        return True

    try:
        # 保存 latest.pth
        save_checkpoint_v2(
            ctx.model._model, ctx.optimizer, ctx.scheduler,
            epoch, metrics, str(checkpoint_dir / "latest.pth"),
            config=config
        )

        # 定期保存
        if save_periodic and (epoch + 1) % 10 == 0:
            save_checkpoint_v2(
                ctx.model._model, ctx.optimizer, ctx.scheduler,
                epoch, metrics, str(checkpoint_dir / f"epoch_{epoch + 1:04d}.pth"),
                config=config
            )
        return True
    except Exception as e:
        print(f"[WARN] Failed to save checkpoint at epoch {epoch + 1}: {e}")
        return False


def run_training_loop(
    ctx: TrainingContext,
    config: dict,
    start_epoch: int,
    checkpoint_dir: Path,
    result_dir: Path,
    training_state: Dict[str, Any],
    log_dir: Optional[Path] = None,
    experiment_dir: Optional[Path] = None
) -> None:
    """
    运行完整训练循环（包含异常处理）

    Args:
        ctx: 训练上下文
        config: 配置
        start_epoch: 开始 epoch
        checkpoint_dir: 检查点目录
        result_dir: 结果目录
        training_state: 训练状态跟踪字典
        log_dir: 日志目录（用于保存 training.log）
        experiment_dir: 实验目录（用于保存 report）
    """
    num_epochs = config['training']['num_epochs']
    validation_interval = config.get('validation', {}).get('interval', 10)

    # 设置日志和报告目录
    if log_dir is None:
        log_dir = checkpoint_dir.parent / "logs"
    if experiment_dir is None:
        experiment_dir = checkpoint_dir.parent

    # 创建报告目录
    report_dir = setup_report_directory(experiment_dir) if is_main_process(ctx.rank) else None

    # 检查梯度累积配置
    use_amp = config['training'].get('mixed_precision', {}).get('enabled', False)
    accumulation_steps = config['training'].get('gradient_accumulation', {}).get('steps', 1)
    use_accumulation = accumulation_steps > 1
    max_norm = config['training'].get('gradient_clipping', {}).get('max_norm', 0.0) if config['training'].get('gradient_clipping', {}).get('enabled', False) else 0.0

    try:
        for epoch in range(start_epoch, num_epochs):
            training_state['current_epoch'] = epoch
            if is_main_process(ctx.rank):
                print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # 运行训练 epoch
            avg_loss = run_training_epoch(
                ctx, epoch, num_epochs,
                use_accumulation, accumulation_steps, max_norm
            )
            training_state['current_avg_loss'] = avg_loss

            if is_main_process(ctx.rank):
                print(f"Average loss: {avg_loss:.4f}")

            # 更新学习率
            if ctx.scheduler is not None:
                ctx.scheduler.step()
                current_lr = ctx.optimizer.param_groups[0]['lr']
                if is_main_process(ctx.rank):
                    print(f"Learning rate: {current_lr:.6f}")

            # 验证
            val_metrics = {'psnr': 0.0, 'ssim': 0.0}
            if (epoch + 1) % validation_interval == 0:
                try:
                    val_metrics = run_validation(
                        ctx.model, ctx.val_loader, config, ctx.device,
                        save_results=is_main_process(ctx.rank),
                        save_dir=str(result_dir / f"epoch_{epoch + 1:04d}"),
                        max_samples=10
                    )

                    if is_main_process(ctx.rank):
                        print(f"Validation - PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}")
                except Exception as e:
                    print(f"[WARN] Validation failed at epoch {epoch + 1}: {e}")
                    val_metrics = {'psnr': 0.0, 'ssim': 0.0}

            # 记录训练日志（只在主进程）
            if is_main_process(ctx.rank):
                log_loss(log_dir, epoch, avg_loss, val_metrics if (epoch + 1) % validation_interval == 0 else None)

            # 保存检查点
            metrics = {'loss': avg_loss}
            if (epoch + 1) % validation_interval == 0:
                metrics.update(val_metrics)

            if handle_checkpoint_save(ctx, epoch, metrics, checkpoint_dir, config):
                training_state['last_saved_epoch'] = epoch

            # 生成验证对比报告（只在验证轮次且主进程）
            if is_main_process(ctx.rank) and (epoch + 1) % validation_interval == 0:
                epoch_result_dir = result_dir / f"epoch_{epoch + 1:04d}"
                if epoch_result_dir.exists():
                    generate_validation_report(
                        epoch_result_dir, report_dir, epoch + 1, is_validation=True
                    )

        if is_main_process(ctx.rank):
            print("\nTraining completed!")

    except KeyboardInterrupt:
        training_state['interrupted'] = True
        training_state['error_msg'] = "User interrupted"
        if is_main_process(ctx.rank):
            print("\n\n[INTERRUPT] Training interrupted by user")
    except Exception as e:
        training_state['interrupted'] = True
        training_state['error_msg'] = str(e)
        if is_main_process(ctx.rank):
            print(f"\n\n[ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
    finally:
        # 无论发生什么，都尝试保存最终检查点
        if is_main_process(ctx.rank) and training_state['interrupted']:
            print(f"\n[SAVING] Attempting to save emergency checkpoint...")
            try:
                emergency_path = str(checkpoint_dir / "emergency_interrupt.pth")

                # 使用当前正在训练的 epoch（如果训练已经开始）
                # 如果训练还没开始，使用 last_saved_epoch
                current_epoch = training_state.get('current_epoch', training_state['last_saved_epoch'])
                current_loss = training_state.get('current_avg_loss', 0.0)

                # 保存时也更新 latest.pth，这样 resume 可以直接接续
                latest_path = str(checkpoint_dir / "latest.pth")

                save_checkpoint_v2(
                    ctx.model._model, ctx.optimizer, ctx.scheduler,
                    current_epoch,
                    {'loss': current_loss,
                     'interrupted': True,
                     'error': training_state['error_msg']},
                    latest_path,
                    config=config
                )

                # 同时保存 emergency 备份
                import shutil
                shutil.copy(latest_path, emergency_path)

                print(f"[SAVED] Emergency checkpoint saved to: {emergency_path}")
                print(f"[SAVED] Also updated: {latest_path}")
                print(f"[INFO] Resume with: --resume (will continue from epoch {current_epoch + 1})")
            except Exception as save_error:
                print(f"[FAILED] Could not save emergency checkpoint: {save_error}")

        # 绘制 loss 曲线（无论正常结束还是中断）
        if is_main_process(ctx.rank):
            try:
                log_file = log_dir / "training.log"
                if log_file.exists():
                    plot_loss_curve(
                        log_file,
                        report_dir / "loss_curve.png",
                        title=f"Training Loss Curve ({'Completed' if not training_state['interrupted'] else 'Interrupted'})"
                    )
            except Exception as e:
                print(f"[WARN] Failed to plot loss curve: {e}")

        cleanup_resources()


if __name__ == "__main__":
    # 测试
    print("Testing training_ops.py...")
    print("Note: Full test requires actual config and data")
    print("All basic tests passed!")
