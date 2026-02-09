"""
validation_ops.py - 验证流程封装

提供标准化的验证流程，支持指标计算和结果保存。
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
import numpy as np
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader
from utils.image_ops import save_image_v2, create_comparison_figure


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    计算PSNR

    Args:
        img1: 图像1 [B, C, H, W]
        img2: 图像2 [B, C, H, W]
        max_val: 最大像素值

    Returns:
        PSNR值
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """
    计算SSIM

    Args:
        img1: 图像1 [B, C, H, W]
        img2: 图像2 [B, C, H, W]
        window_size: 窗口大小

    Returns:
        SSIM值
    """
    from models.srdm.losses import SSIMLoss

    ssim_loss = SSIMLoss(window_size=window_size)
    ssim_val = ssim_loss(img1, img2)

    return ssim_val.item()


def compute_metrics_batch(
    generated: torch.Tensor,
    optical: torch.Tensor
) -> Dict[str, float]:
    """
    计算批次指标

    Args:
        generated: 生成图像 [B, C, H, W]
        optical: 真值图像 [B, C, H, W]

    Returns:
        指标字典
    """
    psnr_val = compute_psnr(generated, optical, max_val=1.0)
    ssim_val = compute_ssim(generated, optical)

    return {
        'psnr': psnr_val,
        'ssim': ssim_val
    }


def run_validation(
    model,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    save_results: bool = False,
    save_dir: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Dict[str, float]:
    """
    运行验证

    Args:
        model: 模型接口
        val_loader: 验证数据加载器
        config: 配置
        device: 设备
        save_results: 是否保存结果图像
        save_dir: 保存目录
        max_samples: 最大验证样本数

    Returns:
        平均指标字典
    """
    model.eval()

    all_metrics = []
    num_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_samples is not None and num_samples >= max_samples:
                break

            sar = batch['sar'].to(device)
            optical = batch.get('optical')
            if optical is None:
                continue
            optical = optical.to(device)

            batch_size = sar.shape[0]

            # 获取模型输出（模型内部已完成合成）
            model_output = model.get_output(sar, config)
            generated = model_output.generated  # 直接使用模型输出的最终图像

            # 计算指标
            for j in range(batch_size):
                gen_img = generated[j:j+1]
                opt_img = optical[j:j+1]

                metrics = compute_metrics_batch(gen_img, opt_img)
                all_metrics.append(metrics)

                # 保存结果
                if save_results and save_dir is not None:
                    from pathlib import Path
                    import cv2

                    save_path = Path(save_dir)
                    save_path.mkdir(parents=True, exist_ok=True)

                    # 转换为numpy
                    sar_np = sar[j].cpu().numpy().transpose(1, 2, 0)
                    gen_np = gen_img[0].cpu().numpy().transpose(1, 2, 0)
                    opt_np = opt_img[0].cpu().numpy().transpose(1, 2, 0)

                    # 确保是uint8
                    def to_uint8(img):
                        img = np.clip(img, 0, 1)
                        return (img * 255).astype(np.uint8)

                    sar_np = to_uint8(sar_np)
                    if sar_np.shape[-1] == 1:
                        sar_np = np.repeat(sar_np, 3, axis=-1)

                    gen_np = to_uint8(gen_np)
                    opt_np = to_uint8(opt_np)

                    # 创建对比图
                    comparison = np.concatenate([sar_np, gen_np, opt_np], axis=1)

                    # 保存
                    cv2.imwrite(
                        str(save_path / f'sample_{num_samples:04d}.png'),
                        cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
                    )

                num_samples += 1

    # 计算平均指标
    if not all_metrics:
        return {'psnr': 0.0, 'ssim': 0.0}

    avg_metrics = {
        'psnr': np.mean([m['psnr'] for m in all_metrics]),
        'ssim': np.mean([m['ssim'] for m in all_metrics])
    }

    return avg_metrics


if __name__ == "__main__":
    # 测试
    print("Testing validation_ops.py...")

    # 测试指标计算
    img1 = torch.rand(2, 3, 64, 64)
    img2 = img1 + torch.randn(2, 3, 64, 64) * 0.1

    metrics = compute_metrics_batch(img1, img2)
    print(f"PSNR: {metrics['psnr']:.2f}")
    print(f"SSIM: {metrics['ssim']:.4f}")

    print("\nAll tests passed!")
