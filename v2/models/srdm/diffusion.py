"""
diffusion.py - SRDM核心扩散模型

实现SAR-Residual Diffusion Model，预测残差而非完整图像。
"""

# 支持单独运行调试：将项目根目录添加到路径
import sys
from pathlib import Path

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from models.srdm.encoder import SAREncoder
from models.srdm.unet import SRDMUNet
from models.srdm.losses import SRDMLoss


class SRDMDiffusion(nn.Module):
    """
    SRDM: SAR-Residual Diffusion Model

    核心创新: 预测残差 R = Optical - SAR_base 而非完整图像

    数据流:
    1. SAR -> SAREncoder -> SAR_base + SAR_features + global_cond
    2. R0 = Optical - SAR_base (Clean Residual, [-1, 1])
    3. R0 -> 前向扩散 -> Rt (加噪)
    4. Rt + SAR_features -> UNet -> pred_noise
    5. DDPM/DDIM 采样 -> pred_R0
    6. Output = SAR_base + pred_R0 -> clamp to [0, 1]
    """

    def __init__(
        self,
        schedule,
        base_ch: int = 64,
        ch_mults: list = [1, 2, 4, 8],
        num_blocks: int = 2,
        time_emb_dim: int = 256,
        dropout: float = 0.1,
        num_heads: int = 8,
        clamp_output: bool = True,
        loss_config: Optional[dict] = None
    ):
        """
        Args:
            schedule: 扩散调度器 (Schedule对象)
            base_ch: UNet基础通道数
            ch_mults: 通道倍数
            num_blocks: 每层NAFBlock数量
            time_emb_dim: 时间嵌入维度
            dropout: dropout概率
            num_heads: 注意力头数
            clamp_output: 是否裁剪输出到[-1, 1]
            loss_config: 损失配置（可选）
        """
        super().__init__()

        self.schedule = schedule
        self.clamp_output = clamp_output
        self.T = schedule.T

        # SAR编码器 - 输入3通道SAR（SEN1-2格式）
        self.sar_encoder = SAREncoder(
            in_ch=3,
            base_ch=64,
            ch_mults=[1, 2, 4, 8],
            global_dim=time_emb_dim
        )

        # 获取SAR特征通道数
        sar_chs = [64 * m for m in [1, 2, 4, 8]]

        # UNet (预测残差上的噪声)
        self.unet = SRDMUNet(
            in_ch=3,  # 残差是3通道
            base_ch=base_ch,
            ch_mults=ch_mults,
            num_blocks=num_blocks,
            sar_chs=sar_chs,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
            num_heads=num_heads
        )

        # 损失函数（可选）
        self.loss_fn = SRDMLoss(loss_config) if loss_config else None

        # 注册buffer (用于扩散采样)
        self.register_buffer('sqrt_alpha_bars', schedule.sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alpha_bars', schedule.sqrt_one_minus_alphas_cumprod)
        self.register_buffer('posterior_mean_coef1', schedule.posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', schedule.posterior_mean_coef2)
        self.register_buffer('posterior_variance', schedule.posterior_variance)
        self.register_buffer('alphas', schedule.alphas)
        self.register_buffer('alphas_cumprod', schedule.alphas_cumprod)

    def compute_residual(self, optical: torch.Tensor, sar_base: torch.Tensor) -> torch.Tensor:
        """
        计算残差 R = Optical - SAR_base

        Args:
            optical: [B, 3, H, W] @ [0, 1]
            sar_base: [B, 3, H, W] @ [0, 1]

        Returns:
            residual: [B, 3, H, W] @ [-1, 1]
        """
        return optical - sar_base

    def add_noise(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向扩散: q(x_t | x_0)

        Args:
            x0: [B, C, H, W] 干净残差
            t: [B] 时间步
            noise: [B, C, H, W] 可选，默认为标准正态分布

        Returns:
            xt: [B, C, H, W] 加噪后的残差
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]

        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def predict_x0_from_eps(
        self,
        xt: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        从预测的噪声反推 x0

        Args:
            xt: [B, C, H, W] 加噪残差
            eps: [B, C, H, W] 预测噪声
            t: [B] 时间步

        Returns:
            x0_pred: [B, C, H, W] 预测的干净残差
        """
        sqrt_alpha_bar = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]

        return (xt - sqrt_one_minus_alpha_bar * eps) / sqrt_alpha_bar

    def forward(
        self,
        sar: torch.Tensor,
        optical: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ):
        """
        训练前向传播

        Args:
            sar: [B, 1, H, W] @ [0, 1]
            optical: [B, 3, H, W] @ [0, 1] (训练时需要)
            return_dict: 是否返回详细损失字典

        Returns:
            loss or (loss, loss_dict)
        """
        B = sar.shape[0]
        device = sar.device

        # 1. SAR编码
        sar_base, sar_features, global_cond = self.sar_encoder(sar)

        # 如果只提供SAR（推理模式），直接返回SAR_base
        if optical is None:
            return sar_base

        # 2. 计算残差
        residual = self.compute_residual(optical, sar_base)  # [-1, 1]

        # 3. 随机采样时间步
        t = torch.randint(0, self.T, (B,), device=device)

        # 4. 加噪
        noise = torch.randn_like(residual)
        noisy_residual = self.add_noise(residual, t, noise)

        # 5. UNet预测噪声
        pred_noise = self.unet(noisy_residual, t, sar_features)

        # 6. 计算损失
        if self.loss_fn is not None:
            # 计算预测残差 (用于x0_mse, x0_l1等损失)
            with torch.no_grad():
                pred_residual = self.predict_x0_from_eps(noisy_residual, pred_noise, t)
                pred_residual = torch.clamp(pred_residual, -1.0, 1.0)

            # 合成预测光学图像 (用于edge, ssim, perceptual等损失)
            pred_optical = torch.clamp(sar_base + pred_residual, 0.0, 1.0)

            loss, loss_dict = self.loss_fn(
                pred_noise=pred_noise,
                target_noise=noise,
                pred_residual=pred_residual,
                target_residual=residual,
                pred_optical=pred_optical,
                target_optical=optical,
                sar_base=sar_base
            )
        else:
            # 默认MSE损失
            loss = F.mse_loss(pred_noise, noise)
            loss_dict = {'noise_mse': loss, 'total': loss}

        if return_dict:
            return loss, loss_dict
        return loss

    @torch.no_grad()
    def sample(
        self,
        sar: torch.Tensor,
        steps: Optional[int] = None,
        temperature: float = 1.0,
        return_residual_only: bool = False
    ) -> torch.Tensor:
        """
        DDPM采样生成

        Args:
            sar: [B, 1, H, W] @ [0, 1]
            steps: 采样步数 (默认使用T)
            temperature: 采样温度
            return_residual_only: 是否只返回残差

        Returns:
            generated: [B, 3, H, W] @ [0, 1] 生成的光学图像
                或 residual: [B, 3, H, W] @ [-1, 1] 如果只返回残差
        """
        B, _, H, W = sar.shape
        device = sar.device
        steps = steps or self.T

        # SAR编码 (只执行一次)
        sar_base, sar_features, global_cond = self.sar_encoder(sar)

        # 从噪声开始
        xt = torch.randn(B, 3, H, W, device=device)

        # 逆向采样
        for i in reversed(range(steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            xt = self.p_sample(xt, t, sar_features, temperature)

        # 最终残差
        pred_residual = xt

        if return_residual_only:
            return pred_residual

        # 合成最终图像
        output = sar_base + pred_residual
        output = torch.clamp(output, 0.0, 1.0)

        return output

    @torch.no_grad()
    def p_sample(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        sar_features: Dict[str, torch.Tensor],
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        DDPM单步采样

        Args:
            xt: [B, C, H, W] 当前噪声残差
            t: [B] 时间步
            sar_features: dict SAR特征
            temperature: 采样温度

        Returns:
            x_prev: [B, C, H, W] 上一步的残差
        """
        # 预测噪声
        pred_noise = self.unet(xt, t, sar_features)

        # 预测 x0
        x0_pred = self.predict_x0_from_eps(xt, pred_noise, t)

        # 裁剪
        if self.clamp_output:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # 计算后验均值
        coef1 = self.posterior_mean_coef1[t][:, None, None, None]
        coef2 = self.posterior_mean_coef2[t][:, None, None, None]
        mean = coef1 * x0_pred + coef2 * xt

        # 添加噪声 (除了最后一步)
        var = self.posterior_variance[t][:, None, None, None]
        noise = torch.randn_like(xt) * temperature
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)

        return mean + nonzero_mask * torch.sqrt(var) * noise

    @torch.no_grad()
    def ddim_sample(
        self,
        sar: torch.Tensor,
        steps: int = 50,
        eta: float = 0.0,
        return_residual_only: bool = False
    ) -> torch.Tensor:
        """
        DDIM加速采样

        Args:
            sar: [B, 1, H, W] @ [0, 1]
            steps: 采样步数 (默认50)
            eta: DDIM噪声参数 (0为确定性)
            return_residual_only: 是否只返回残差

        Returns:
            generated: [B, 3, H, W] @ [0, 1]
                或 residual: [B, 3, H, W] @ [-1, 1] 如果只返回残差
        """
        B, _, H, W = sar.shape
        device = sar.device

        # SAR编码
        sar_base, sar_features, global_cond = self.sar_encoder(sar)

        # 创建时间步序列
        timesteps = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)

        # 从噪声开始
        xt = torch.randn(B, 3, H, W, device=device)

        # DDIM采样
        for i in range(len(timesteps)):
            t = torch.full((B,), timesteps[i], device=device, dtype=torch.long)

            # 预测噪声
            pred_noise = self.unet(xt, t, sar_features)

            # 预测 x0
            alpha_bar = self.alphas_cumprod[t][:, None, None, None]
            sqrt_alpha_bar = torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

            x0_pred = (xt - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
            if self.clamp_output:
                x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            if i < len(timesteps) - 1:
                # 计算下一个时间步
                t_next = timesteps[i + 1]
                alpha_bar_next = self.alphas_cumprod[t_next]
                sqrt_alpha_bar_next = torch.sqrt(alpha_bar_next)
                sqrt_one_minus_alpha_bar_next = torch.sqrt(1 - alpha_bar_next)

                # DDIM公式
                c1 = sqrt_alpha_bar_next
                c2 = sqrt_one_minus_alpha_bar_next

                if eta > 0:
                    sigma = eta * torch.sqrt(
                        (1 - alpha_bar_next) / (1 - alpha_bar) *
                        (1 - alpha_bar / alpha_bar_next)
                    )
                    noise = torch.randn_like(xt)
                    xt = c1 * x0_pred + torch.sqrt(c2**2 - sigma**2) * pred_noise + sigma * noise
                else:
                    xt = c1 * x0_pred + c2 * pred_noise
            else:
                xt = x0_pred

        # 合成最终图像
        pred_residual = xt

        if return_residual_only:
            return pred_residual

        output = sar_base + pred_residual
        output = torch.clamp(output, 0.0, 1.0)

        return output

    def count_parameters(self) -> Dict[str, int]:
        """计算总参数量"""
        enc_params = sum(p.numel() for p in self.sar_encoder.parameters())
        unet_params = sum(p.numel() for p in self.unet.parameters())
        return {
            'encoder': enc_params,
            'unet': unet_params,
            'total': enc_params + unet_params
        }


if __name__ == "__main__":
    # 测试
    print("Testing diffusion.py...")

    from models.diffusion.schedule import Schedule

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建调度器
    schedule = Schedule(T=1000, schedule='linear')

    # 创建模型
    model = SRDMDiffusion(
        schedule=schedule,
        base_ch=64,
        ch_mults=[1, 2, 4, 8],
        num_blocks=2,
        time_emb_dim=256,
        dropout=0.1,
        num_heads=8
    ).to(device)

    # 参数计数
    params = model.count_parameters()
    print(f"\nParameters:")
    print(f"  Encoder: {params['encoder'] / 1e6:.2f}M")
    print(f"  UNet: {params['unet'] / 1e6:.2f}M")
    print(f"  Total: {params['total'] / 1e6:.2f}M")

    # 测试训练前向
    print("\nTesting training forward...")
    sar = torch.rand(2, 1, 256, 256).to(device)
    optical = torch.rand(2, 3, 256, 256).to(device)

    loss, loss_dict = model(sar, optical, return_dict=True)
    print(f"  Loss: {loss.item():.4f}")

    # 测试DDIM采样
    print("\nTesting DDIM sampling...")
    with torch.no_grad():
        generated = model.ddim_sample(sar, steps=10, return_residual_only=True)
    print(f"  Residual range: [{generated.min():.2f}, {generated.max():.2f}]")

    print("\nAll tests passed!")
