"""
visualization_ops.py - 可视化操作模块

提供训练过程可视化和结果报告生成功能。
"""

# 支持单独运行调试：将项目根目录添加到路径
import sys
from pathlib import Path

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import torch


def setup_report_directory(experiment_dir: Path) -> Path:
    """
    创建并返回 report 目录路径

    Args:
        experiment_dir: 实验目录路径

    Returns:
        report 目录路径
    """
    report_dir = experiment_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def log_loss(
    log_dir: Path,
    epoch: int,
    loss: float,
    val_metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    追加写入训练日志

    Args:
        log_dir: 日志目录
        epoch: 当前 epoch
        loss: 平均损失
        val_metrics: 验证指标（可选）
    """
    log_file = log_dir / "training.log"

    # 构建日志行
    log_line = f"{epoch},{loss:.6f}"
    if val_metrics:
        psnr = val_metrics.get('psnr', 0.0)
        ssim = val_metrics.get('ssim', 0.0)
        log_line += f",{psnr:.4f},{ssim:.4f}"
    else:
        log_line += ",,"

    # 追加写入
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_line + '\n')


def load_training_log(log_file: Path) -> Tuple[List[int], List[float], List[float], List[float]]:
    """
    从日志文件加载训练历史

    Args:
        log_file: 日志文件路径

    Returns:
        (epochs, losses, psnrs, ssims)
    """
    epochs = []
    losses = []
    psnrs = []
    ssims = []

    if not log_file.exists():
        return epochs, losses, psnrs, ssims

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                epochs.append(int(parts[0]))
                losses.append(float(parts[1]))
                if len(parts) >= 4 and parts[2]:
                    psnrs.append(float(parts[2]))
                else:
                    psnrs.append(None)
                if len(parts) >= 4 and parts[3]:
                    ssims.append(float(parts[3]))
                else:
                    ssims.append(None)

    return epochs, losses, psnrs, ssims


def plot_loss_curve(
    log_file: Path,
    save_path: Path,
    title: str = "Training Loss Curve"
) -> bool:
    """
    从日志文件读取并绘制 loss 曲线

    Args:
        log_file: 训练日志文件路径
        save_path: 保存图像路径
        title: 图表标题

    Returns:
        是否成功绘制
    """
    epochs, losses, psnrs, ssims = load_training_log(log_file)

    if not epochs:
        print("[WARN] No training log data found")
        return False

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Loss 曲线
    ax = axes[0, 0]
    ax.plot(epochs, losses, 'b-', linewidth=1.5, label='Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # PSNR 曲线（过滤 None 值）
    ax = axes[0, 1]
    valid_psnr = [(e, p) for e, p in zip(epochs, psnrs) if p is not None]
    if valid_psnr:
        psnr_epochs, psnr_values = zip(*valid_psnr)
        ax.plot(psnr_epochs, psnr_values, 'g-', linewidth=1.5, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Validation PSNR')
        ax.grid(True, alpha=0.3)

    # SSIM 曲线（过滤 None 值）
    ax = axes[1, 0]
    valid_ssim = [(e, s) for e, s in zip(epochs, ssims) if s is not None]
    if valid_ssim:
        ssim_epochs, ssim_values = zip(*valid_ssim)
        ax.plot(ssim_epochs, ssim_values, 'r-', linewidth=1.5, marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('SSIM')
        ax.set_title('Validation SSIM')
        ax.grid(True, alpha=0.3)

    # 学习率（如果日志中有）- 这里留空或显示最后状态
    ax = axes[1, 1]
    ax.text(0.5, 0.5, f'Total Epochs: {len(epochs)}\nFinal Loss: {losses[-1]:.6f}',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Training Summary')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[INFO] Loss curve saved to: {save_path}")
    return True


def create_inference_comparison(
    sar: torch.Tensor,
    generated: torch.Tensor,
    optical: torch.Tensor
) -> np.ndarray:
    """
    创建推理对比图（SAR | Generated | Optical）

    Args:
        sar: SAR 输入 [1, C, H, W]
        generated: 生成图像 [1, C, H, W]
        optical: 真值图像 [1, C, H, W]

    Returns:
        uint8 格式的对比图 numpy 数组
    """
    # 转换为 numpy
    def tensor_to_numpy(t):
        t = t[0].cpu().detach().numpy().transpose(1, 2, 0)  # [H, W, C]
        t = np.clip(t, 0, 1)
        return (t * 255).astype(np.uint8)

    sar_np = tensor_to_numpy(sar)
    gen_np = tensor_to_numpy(generated)
    opt_np = tensor_to_numpy(optical)

    # SAR 如果是单通道，转为3通道
    if sar_np.shape[-1] == 1:
        sar_np = np.repeat(sar_np, 3, axis=-1)

    # 拼接
    comparison = np.concatenate([sar_np, gen_np, opt_np], axis=1)
    return comparison


def create_comparison_figure(
    sample_paths: List[Path],
    save_path: Path,
    title: str = "Validation Results",
    samples_per_row: int = 5
) -> bool:
    """
    从已保存的单张图片创建对比报告

    Args:
        sample_paths: 样本图片路径列表（每3张一组：sar, gen, opt）
        save_path: 保存路径
        title: 报告标题
        samples_per_row: 每行显示多少组样本

    Returns:
        是否成功创建
    """
    import cv2

    if not sample_paths:
        return False

    # 读取所有图片
    images = []
    for path in sample_paths:
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)

    if not images:
        print("[WARN] No valid images found for comparison")
        return False

    # 计算布局
    num_samples = len(images)
    num_rows = (num_samples + samples_per_row - 1) // samples_per_row

    # 获取单张图片尺寸
    h, w = images[0].shape[:2]

    # 创建大图
    fig_width = samples_per_row * (w / 100)
    fig_height = num_rows * (h / 100) + 1  # 额外空间给标题

    fig, axes = plt.subplots(num_rows, samples_per_row,
                             figsize=(fig_width, fig_height))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    if num_rows == 1:
        axes = [axes]
    if samples_per_row == 1:
        axes = [[ax] for ax in axes]

    # 填充图片
    for idx, img in enumerate(images):
        row = idx // samples_per_row
        col = idx % samples_per_row

        if row < len(axes) and col < len(axes[row]):
            ax = axes[row][col]
            ax.imshow(img)
            ax.set_title(f"Sample {idx:02d}", fontsize=8)
            ax.axis('off')

    # 隐藏多余的子图
    for idx in range(num_samples, num_rows * samples_per_row):
        row = idx // samples_per_row
        col = idx % samples_per_row
        if row < len(axes) and col < len(axes[row]):
            axes[row][col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[INFO] Comparison report saved to: {save_path}")
    return True


def generate_validation_report(
    result_dir: Path,
    report_dir: Path,
    epoch: int,
    is_validation: bool = True
) -> bool:
    """
    从已保存的验证结果生成对比报告

    Args:
        result_dir: 验证结果保存目录（包含 sample_XXXX.png）
        report_dir: 报告保存目录
        epoch: 当前 epoch
        is_validation: 是否为验证（影响文件名和标题）

    Returns:
        是否成功生成
    """
    if not result_dir.exists():
        return False

    # 查找所有样本图片
    sample_paths = sorted(result_dir.glob("sample_*.png"))

    if not sample_paths:
        print(f"[WARN] No samples found in {result_dir}")
        return False

    # 构建保存路径
    prefix = "val" if is_validation else "train"
    report_path = report_dir / f"{prefix}_samples_epoch_{epoch:04d}.png"

    # 构建标题
    title = f"Validation Results - Epoch {epoch}"

    return create_comparison_figure(sample_paths, report_path, title)


if __name__ == "__main__":
    print("Testing visualization_ops.py...")

    # 测试目录创建
    test_dir = Path("/tmp/test_vis")
    report_dir = setup_report_directory(test_dir)
    print(f"Report dir: {report_dir}")

    # 测试日志记录
    log_dir = test_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_loss(log_dir, 0, 0.5, {'psnr': 25.0, 'ssim': 0.85})
    log_loss(log_dir, 1, 0.4, None)
    log_loss(log_dir, 2, 0.3, {'psnr': 28.0, 'ssim': 0.90})
    print("Log written")

    # 测试曲线绘制
    log_file = log_dir / "training.log"
    save_path = report_dir / "loss_curve.png"
    success = plot_loss_curve(log_file, save_path)
    print(f"Curve plotted: {success}")

    print("\nAll tests passed!")
