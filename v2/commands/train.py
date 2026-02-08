"""
train.py - 训练命令

实现模型训练流程。
"""

# 支持单独运行调试：将项目根目录添加到路径
import sys
from pathlib import Path

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

import argparse
import os
import yaml
from tqdm import tqdm
from commands.base import BaseCommand, command
from core.device_ops import (
    setup_device_and_distributed,
    is_main_process,
    set_seed,
    cleanup_resources
)
from core.training_ops import setup_training, train_step, train_step_with_accumulation
from core.checkpoint_ops import save_checkpoint_v2, load_checkpoint_v2, restore_model_v2
from core.validation_ops import run_validation


@command('train')
class TrainCommand(BaseCommand):
    """训练命令"""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--config', type=str, required=True, help='配置文件路径')
        parser.add_argument('--experiment', type=str, default=None, help='实验名称')
        parser.add_argument('--resume', action='store_true', help='续训模式')
        parser.add_argument('--epochs', type=int, default=None, help='覆盖训练轮数')
        parser.add_argument('--batch-size', type=int, default=None, help='覆盖批次大小')
        parser.add_argument('--local_rank', type=int, default=0, help='本地rank（DDP）')

    def execute(self, args: argparse.Namespace):
        """执行训练"""
        # 加载配置
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 覆盖配置
        if args.epochs is not None:
            config['training']['num_epochs'] = args.epochs
        if args.batch_size is not None:
            config['data']['dataloader']['batch_size'] = args.batch_size

        # 设置设备
        device, rank, world_size = setup_device_and_distributed(config)

        # 设置随机种子
        seed = config.get('device', {}).get('random_seed', 42)
        set_seed(seed + rank)  # 每个进程不同的种子

        if is_main_process(rank):
            print(f"=" * 70)
            print(f"SRDM Training")
            print(f"=" * 70)
            print(f"Device: {device}")
            print(f"World size: {world_size}")
            print(f"Config: {args.config}")
            print(f"Epochs: {config['training']['num_epochs']}")
            print(f"=" * 70)

        # 创建实验目录
        experiment_name = args.experiment or "experiment"
        experiment_dir = Path("experiments") / experiment_name
        checkpoint_dir = experiment_dir / "checkpoints"
        log_dir = experiment_dir / "logs"
        result_dir = experiment_dir / "results"

        if is_main_process(rank):
            experiment_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            result_dir.mkdir(parents=True, exist_ok=True)

            # 保存配置
            with open(experiment_dir / "config.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(config, f)

        # 设置训练
        ctx = setup_training(config, device, rank, world_size)

        # 检查梯度累积配置
        use_amp = config['training'].get('mixed_precision', {}).get('enabled', False)
        accumulation_steps = config['training'].get('gradient_accumulation', {}).get('steps', 1)
        use_accumulation = accumulation_steps > 1
        max_norm = config['training'].get('gradient_clipping', {}).get('max_norm', 0.0) if config['training'].get('gradient_clipping', {}).get('enabled', False) else 0.0

        # 恢复训练
        start_epoch = 0
        if args.resume:
            latest_checkpoint = checkpoint_dir / "latest.pth"
            if latest_checkpoint.exists():
                if is_main_process(rank):
                    print(f"Resuming from {latest_checkpoint}")

                checkpoint = load_checkpoint_v2(str(latest_checkpoint), device=device)
                restore_model_v2(ctx.model._model, checkpoint['model_state_dict'])
                ctx.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if ctx.scheduler is not None and checkpoint.get('scheduler_state_dict'):
                    ctx.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1

                if is_main_process(rank):
                    print(f"Resumed from epoch {start_epoch}")
            else:
                if is_main_process(rank):
                    print("No checkpoint found, starting from scratch")

        # 训练循环
        num_epochs = config['training']['num_epochs']
        validation_interval = config.get('validation', {}).get('interval', 10)

        for epoch in range(start_epoch, num_epochs):
            if is_main_process(rank):
                print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # 设置epoch（用于DistributedSampler）
            if hasattr(ctx.train_loader.sampler, 'set_epoch'):
                ctx.train_loader.sampler.set_epoch(epoch)

            # 训练
            ctx.model.train()
            total_loss = 0.0
            num_batches = len(ctx.train_loader)
            batch_count = 0

            pbar = tqdm(ctx.train_loader, desc=f"Train", disable=not is_main_process(rank))
            for batch in pbar:
                batch_count += 1

                if use_accumulation:
                    is_last_batch = (batch_count == num_batches)
                    is_accumulation_step = (batch_count % accumulation_steps != 0) and not is_last_batch

                    loss, loss_dict = train_step_with_accumulation(
                        ctx.model, batch, ctx.optimizer, ctx.scaler,
                        device, use_amp, max_norm,
                        accumulation_steps, is_accumulation_step
                    )

                    accum_info = f"{batch_count % accumulation_steps or accumulation_steps}/{accumulation_steps}"
                    pbar.set_postfix({'loss': f'{loss:.4f}', 'accum': accum_info})
                else:
                    loss, loss_dict = train_step(
                        ctx.model, batch, ctx.optimizer, ctx.scaler,
                        device, use_amp, max_norm
                    )
                    pbar.set_postfix({'loss': f'{loss:.4f}'})

                total_loss += loss

            avg_loss = total_loss / num_batches

            if is_main_process(rank):
                print(f"Average loss: {avg_loss:.4f}")

            # 更新学习率
            if ctx.scheduler is not None:
                ctx.scheduler.step()
                current_lr = ctx.optimizer.param_groups[0]['lr']
                if is_main_process(rank):
                    print(f"Learning rate: {current_lr:.6f}")

            # 验证
            if (epoch + 1) % validation_interval == 0:
                val_metrics = run_validation(
                    ctx.model, ctx.val_loader, config, device,
                    save_results=is_main_process(rank),
                    save_dir=str(result_dir / f"epoch_{epoch + 1:04d}"),
                    max_samples=10
                )

                if is_main_process(rank):
                    print(f"Validation - PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}")

            # 保存检查点
            if is_main_process(rank):
                metrics = {'loss': avg_loss}
                if (epoch + 1) % validation_interval == 0:
                    metrics.update(val_metrics)

                save_checkpoint_v2(
                    ctx.model._model, ctx.optimizer, ctx.scheduler,
                    epoch, metrics, str(checkpoint_dir / "latest.pth")
                )

                # 定期保存
                if (epoch + 1) % 10 == 0:
                    save_checkpoint_v2(
                        ctx.model._model, ctx.optimizer, ctx.scheduler,
                        epoch, metrics, str(checkpoint_dir / f"epoch_{epoch + 1:04d}.pth")
                    )

        if is_main_process(rank):
            print("\nTraining completed!")

        cleanup_resources()


if __name__ == "__main__":
    print("TrainCommand module loaded")
