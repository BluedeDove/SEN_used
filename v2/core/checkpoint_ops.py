"""
checkpoint_ops.py - 检查点保存和加载

统一的检查点管理接口，支持DDP和非DDP模式。
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from .device_ops import get_raw_model, is_main_process


def save_checkpoint_v2(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    save_path: str
):
    """
    保存检查点

    Args:
        model: 模型（可能是DDP包装）
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        epoch: 当前epoch
        metrics: 指标字典（如 {'psnr': 25.5, 'ssim': 0.85}）
        save_path: 保存路径
    """
    # 只有主进程保存
    if not is_main_process():
        return

    # 获取原始模型状态
    model_state = get_raw_model(model).state_dict()

    # 构建检查点字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }

    # 创建目录
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint_v2(
    checkpoint_path: str,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    加载检查点

    Args:
        checkpoint_path: 检查点路径
        device: 加载设备

    Returns:
        包含state_dict等的字典

    Raises:
        FileNotFoundError: 如果检查点不存在
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def restore_model_v2(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True
):
    """
    恢复模型状态（自动处理DDP）

    Args:
        model: 目标模型
        state_dict: 状态字典
        strict: 是否严格匹配
    """
    # 获取原始模型（处理DDP包装）
    raw_model = get_raw_model(model)

    # 加载状态
    raw_model.load_state_dict(state_dict, strict=strict)


def restore_optimizer_v2(
    optimizer: torch.optim.Optimizer,
    state_dict: Dict[str, Any]
):
    """
    恢复优化器状态

    Args:
        optimizer: 目标优化器
        state_dict: 状态字典
    """
    optimizer.load_state_dict(state_dict)


def restore_scheduler_v2(
    scheduler: Any,
    state_dict: Optional[Dict[str, Any]]
):
    """
    恢复调度器状态

    Args:
        scheduler: 目标调度器
        state_dict: 状态字典（可能为None）
    """
    if scheduler is not None and state_dict is not None:
        scheduler.load_state_dict(state_dict)


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    获取最新的检查点路径

    Args:
        checkpoint_dir: 检查点目录

    Returns:
        最新检查点路径，如果没有则返回None
    """
    checkpoint_path = Path(checkpoint_dir) / 'latest.pth'

    if checkpoint_path.exists():
        return str(checkpoint_path)

    # 如果没有latest.pth，查找所有.pth文件
    pth_files = list(Path(checkpoint_dir).glob('*.pth'))

    if not pth_files:
        return None

    # 按修改时间排序，返回最新的
    pth_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(pth_files[0])


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 3
):
    """
    清理旧的检查点，只保留最新的N个

    Args:
        checkpoint_dir: 检查点目录
        keep_last_n: 保留数量
    """
    if not is_main_process():
        return

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return

    # 获取所有检查点文件（排除latest.pth）
    pth_files = [
        f for f in checkpoint_path.glob('*.pth')
        if f.name != 'latest.pth'
    ]

    if len(pth_files) <= keep_last_n:
        return

    # 按修改时间排序
    pth_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # 删除旧的
    for old_file in pth_files[keep_last_n:]:
        try:
            old_file.unlink()
            print(f"Removed old checkpoint: {old_file}")
        except Exception as e:
            print(f"Failed to remove {old_file}: {e}")


def save_checkpoint_with_backup(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    is_best: bool = False
):
    """
    保存检查点并创建备份

    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 调度器
        epoch: 当前epoch
        metrics: 指标
        checkpoint_dir: 检查点目录
        is_best: 是否是最佳检查点
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 保存最新检查点
    latest_path = checkpoint_dir / 'latest.pth'
    save_checkpoint_v2(
        model, optimizer, scheduler, epoch, metrics,
        str(latest_path)
    )

    # 保存epoch-specific检查点
    epoch_path = checkpoint_dir / f'epoch_{epoch:04d}.pth'
    save_checkpoint_v2(
        model, optimizer, scheduler, epoch, metrics,
        str(epoch_path)
    )

    # 保存最佳检查点
    if is_best:
        best_path = checkpoint_dir / 'best.pth'
        save_checkpoint_v2(
            model, optimizer, scheduler, epoch, metrics,
            str(best_path)
        )


if __name__ == "__main__":
    # 测试
    print("Testing checkpoint_ops...")

    # 创建临时目录
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建简单模型
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 保存检查点
        save_path = os.path.join(tmpdir, 'test.pth')
        save_checkpoint_v2(
            model, optimizer, None, 10,
            {'psnr': 25.5, 'ssim': 0.85},
            save_path
        )

        # 加载检查点
        checkpoint = load_checkpoint_v2(save_path)
        print(f"Loaded checkpoint: epoch={checkpoint['epoch']}")

        # 恢复模型
        new_model = torch.nn.Linear(10, 10)
        restore_model_v2(new_model, checkpoint['model_state_dict'])
        print("✓ Model restored")

        # 恢复优化器
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        restore_optimizer_v2(new_optimizer, checkpoint['optimizer_state_dict'])
        print("✓ Optimizer restored")

    print("All tests passed!")
