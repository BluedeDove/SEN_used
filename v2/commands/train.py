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
from commands.base import BaseCommand, command
from core.device_ops import (
    setup_device_and_distributed,
    is_main_process,
    set_seed
)
from core.training_ops import (
    setup_training,
    run_training_loop
)
from core.checkpoint_ops import load_checkpoint_v2, restore_model_v2
from core.config_loader import load_config


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
        # 加载并合并配置
        config = load_config(args.config, verbose=is_main_process(0))

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

        # 恢复训练
        start_epoch = 0
        if args.resume:
            latest_checkpoint = checkpoint_dir / "latest.pth"
            if latest_checkpoint.exists():
                if is_main_process(rank):
                    print(f"Resuming from {latest_checkpoint}")

                checkpoint = load_checkpoint_v2(str(latest_checkpoint), device=str(device))
                
                # 检查模型类型兼容性
                saved_model_type = checkpoint.get('model_type', 'unknown')
                current_model_type = ctx.model._model.__class__.__name__
                if hasattr(ctx.model._model, 'module'):
                    current_model_type = ctx.model._model.module.__class__.__name__
                
                if is_main_process(rank):
                    print(f"  Saved model type: {saved_model_type}")
                    print(f"  Current model type: {current_model_type}")
                
                # 恢复模型权重
                restore_model_v2(ctx.model._model, checkpoint['model_state_dict'])
                
                # 恢复优化器状态（注意：如果修改了学习率，需要重新创建优化器）
                ctx.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # 如果配置文件修改了学习率，覆盖优化器中的学习率
                new_lr = config.get('training', {}).get('optimizer', {}).get('lr')
                if new_lr is not None:
                    for param_group in ctx.optimizer.param_groups:
                        if param_group['lr'] != new_lr:
                            if is_main_process(rank):
                                print(f"  Updating learning rate: {param_group['lr']} -> {new_lr}")
                            param_group['lr'] = new_lr
                
                # 恢复调度器
                if ctx.scheduler is not None and checkpoint.get('scheduler_state_dict'):
                    ctx.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                start_epoch = checkpoint['epoch'] + 1

                if is_main_process(rank):
                    print(f"Resumed from epoch {start_epoch}")
                    print(f"  Current config will be used for: batch_size, lr, num_epochs, etc.")
            else:
                if is_main_process(rank):
                    print("No checkpoint found, starting from scratch")

        # 跟踪训练状态，用于异常时保存
        training_state = {
            'last_saved_epoch': start_epoch - 1,
            'interrupted': False,
            'error_msg': None,
            'current_epoch': start_epoch - 1,
            'current_avg_loss': 0.0
        }

        # 运行训练循环
        run_training_loop(
            ctx=ctx,
            config=config,
            start_epoch=start_epoch,
            checkpoint_dir=checkpoint_dir,
            result_dir=result_dir,
            training_state=training_state,
            log_dir=log_dir,
            experiment_dir=experiment_dir
        )


if __name__ == "__main__":
    print("TrainCommand module loaded")
