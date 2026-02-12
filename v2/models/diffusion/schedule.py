"""
schedule.py - 扩散调度器

实现DDPM/DDIM的噪声调度。
"""

import torch
import numpy as np


class Schedule:
    """
    扩散模型噪声调度器

    实现标准的线性beta调度。
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "linear",
        device: str = 'cpu'
    ):
        """
        初始化调度器

        Args:
            num_timesteps: 时间步数T
            beta_start: beta起始值
            beta_end: beta结束值
            schedule_type: 调度类型 ('linear')
            device: 计算设备
        """
        self.T = num_timesteps
        self.device = device

        # 生成beta序列
        if schedule_type == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, device=device
            )
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # 计算alpha
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 预计算常用值
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 后验分布参数（用于DDPM采样）
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device),
            self.alphas_cumprod[:-1]
        ])

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )

        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            torch.sqrt(self.alphas) * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device):
        """移动所有张量到指定设备"""
        self.device = device
        for attr_name in [
            'betas', 'alphas', 'alphas_cumprod',
            'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
            'alphas_cumprod_prev', 'posterior_variance',
            'posterior_log_variance_clipped',
            'posterior_mean_coef1', 'posterior_mean_coef2'
        ]:
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(device))
        return self


if __name__ == "__main__":
    # 测试
    print("Testing schedule.py...")

    schedule = Schedule(num_timesteps=1000)

    print(f"T: {schedule.T}")
    print(f"betas shape: {schedule.betas.shape}")
    print(f"alphas_cumprod range: [{schedule.alphas_cumprod.min():.4f}, {schedule.alphas_cumprod.max():.4f}]")

    print("All tests passed!")
