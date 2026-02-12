"""
model_residual.py - 残差版本 Flow Matching 模型

基于 Rectified Flow 预测残差而非完整图像，更容易训练。
类似于 SRDM 的思想，但使用 Flow Matching 代替扩散模型。
"""

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

# 复用 SRDM 的编码器
from models.srdm.encoder import SAREncoder
from models.flowmatching.unet import FlowMatchingUNet
from models.flowmatching.sampler import FlowMatchingSampler


class ResidualFlowMatchingModel(nn.Module):
    """
    残差版本 Flow Matching Model
    
    核心创新：预测残差 R = Optical - SAR_base 的流，而非完整图像
    
    训练:
        1. SAR -> Encoder -> sar_base
        2. R_target = Optical - sar_base (残差目标)
        3. 在残差上应用 Flow Matching:
           - x_t = t * R_target + (1-t) * noise
           - 学习 v_theta(x_t, t, SAR) ≈ R_target - noise
    
    采样:
        1. SAR -> Encoder -> sar_base, sar_features
        2. 从噪声开始，通过 ODE 求解得到 pred_R
        3. Output = sar_base + pred_R -> clamp to [0, 1]
    """
    
    def __init__(
        self,
        base_ch: int = 64,
        ch_mults: list = [1, 2, 4, 8],
        num_blocks: int = 2,
        time_emb_dim: int = 256,
        dropout: float = 0.1,
        num_heads: int = 8,
        sar_encoder_config: Optional[dict] = None,
        clamp_output: bool = True,
        use_normalized_residual: bool = True  # 是否归一化残差到 [-1, 1]
    ):
        """
        Args:
            base_ch: UNet 基础通道数
            ch_mults: 通道倍数
            num_blocks: 每层块数
            time_emb_dim: 时间嵌入维度
            dropout: dropout 概率
            num_heads: 注意力头数
            sar_encoder_config: SAR 编码器配置
            clamp_output: 是否裁剪输出到 [-1, 1]
            use_normalized_residual: 是否将残差归一化到 [-1, 1] 范围
        """
        super().__init__()
        
        self.clamp_output = clamp_output
        self.use_normalized_residual = use_normalized_residual
        
        # 默认编码器配置
        if sar_encoder_config is None:
            sar_encoder_config = {
                'in_ch': 3,
                'base_ch': 64,
                'ch_mults': [1, 2, 4, 8],
                'global_dim': time_emb_dim
            }
        
        # SAR 编码器
        self.sar_encoder = SAREncoder(
            in_ch=sar_encoder_config.get('in_ch', 3),
            base_ch=sar_encoder_config.get('base_ch', 64),
            ch_mults=sar_encoder_config.get('ch_mults', [1, 2, 4, 8]),
            global_dim=sar_encoder_config.get('global_dim', time_emb_dim)
        )
        
        # 获取 SAR 特征通道数
        sar_chs = [
            sar_encoder_config.get('base_ch', 64) * m 
            for m in sar_encoder_config.get('ch_mults', [1, 2, 4, 8])
        ]
        
        # Flow Matching UNet (预测残差上的向量场)
        self.unet = FlowMatchingUNet(
            in_ch=3,  # 输入是残差状态 (3通道)
            out_ch=3,  # 输出是残差向量场 (3通道)
            base_ch=base_ch,
            ch_mults=ch_mults,
            num_blocks=num_blocks,
            sar_chs=sar_chs,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
            num_heads=num_heads
        )
        
    def compute_residual(self, optical: torch.Tensor, sar_base: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        计算残差 R = Optical - SAR_base
        
        Args:
            optical: [B, 3, H, W] @ [0, 1]
            sar_base: [B, 3, H, W] @ [0, 1]
            
        Returns:
            residual: [B, 3, H, W]
            scale: 残差尺度 (用于反归一化)
        """
        residual = optical - sar_base  # 范围约为 [-1, 1]
        
        if self.use_normalized_residual:
            # 动态归一化到 [-1, 1]
            max_val = residual.abs().max() + 1e-8
            residual = residual / max_val
            return residual, max_val
        
        return residual, 1.0
    
    def forward(
        self,
        sar: torch.Tensor,
        optical: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ):
        """
        训练前向传播
        
        Args:
            sar: [B, 1 or 3, H, W] SAR 图像
            optical: [B, 3, H, W] 目标光学图像 (训练时需要)
            return_dict: 是否返回详细损失字典
            
        Returns:
            loss or (loss, loss_dict)
        """
        # 确保 SAR 是 3 通道
        if sar.shape[1] == 1:
            sar = sar.repeat(1, 3, 1, 1)
            
        # 1. SAR 编码
        sar_base, sar_features, global_cond = self.sar_encoder(sar)
        
        # 如果只提供 SAR (推理模式)
        if optical is None:
            return sar_base
        
        B = optical.shape[0]
        device = optical.device
        
        # 2. 计算残差目标
        residual_target, residual_scale = self.compute_residual(optical, sar_base)
        
        # 3. Flow Matching on Residual
        # 采样时间步 t ~ Uniform(0, 1)
        t = torch.rand(B, device=device)
        
        # 采样噪声 x_0 ~ N(0, I)
        x_0 = torch.randn_like(residual_target)
        
        # 计算插值 x_t = t * residual_target + (1-t) * x_0
        t_expanded = t.view(B, 1, 1, 1)
        x_t = t_expanded * residual_target + (1 - t_expanded) * x_0
        
        # 目标向量场 u_t = residual_target - x_0
        u_t = residual_target - x_0
        
        # 4. 模型预测向量场
        v_pred = self.unet(x_t, t, sar_features)
        
        # 5. 计算 Flow Matching 损失
        loss = F.mse_loss(v_pred, u_t)
        
        if return_dict:
            loss_dict = {
                'flow_matching_loss': loss.item(),
                'residual_scale': residual_scale,
                'total': loss.item()
            }
            return loss, loss_dict
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        sar: torch.Tensor,
        steps: int = 20,
        method: str = 'dpmpp',
        return_residual_only: bool = False,
        **sampler_kwargs
    ) -> torch.Tensor:
        """
        采样生成光学图像
        
        Args:
            sar: [B, 1 or 3, H, W] SAR 图像
            steps: 采样步数
            method: 采样方法 ('euler', 'heun', 'dpmpp')
            return_residual_only: 是否只返回残差
            **sampler_kwargs: 传递给采样器的额外参数
            
        Returns:
            generated: [B, 3, H, W] 生成的光学图像 [0, 1]
                或 residual: [B, 3, H, W] 如果只返回残差
        """
        # 确保 SAR 是 3 通道
        if sar.shape[1] == 1:
            sar = sar.repeat(1, 3, 1, 1)
            
        B, _, H, W = sar.shape
        device = sar.device
        
        # 1. SAR 编码 (只执行一次)
        sar_base, sar_features, global_cond = self.sar_encoder(sar)
        
        # 2. 从噪声开始 (在残差空间)
        x_t = torch.randn(B, 3, H, W, device=device)
        
        # 3. 创建采样器并采样
        def model_fn(x, t, sar_feat, global_cond):
            return self.unet(x, t, sar_feat)
        
        sampler = FlowMatchingSampler(
            model_fn=model_fn,
            method=method,
            steps=steps,
            **sampler_kwargs
        )
        
        pred_residual = sampler.sample(x_t, sar_features, global_cond)
        
        if return_residual_only:
            return pred_residual
        
        # 4. 合成最终图像
        if self.clamp_output:
            pred_residual = torch.clamp(pred_residual, -1.0, 1.0)
        
        # 合成: sar_base + residual
        output = sar_base + pred_residual
        
        # 截断负数并归一化到 [0, 1] (SRDM 风格)
        output = torch.clamp(output, min=0.0)
        
        # 归一化到 [0, 1]
        B, C, H, W = output.shape
        output_flat = output.view(B, C, -1)
        o_min = output_flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)
        o_max = output_flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)
        
        if (o_max > o_min).all():
            output = (output - o_min) / (o_max - o_min + 1e-8)
        else:
            output = torch.ones_like(output) * 0.5
        
        return output
    
    def count_parameters(self) -> Dict[str, int]:
        """统计参数量"""
        return {
            'total': sum(p.numel() for p in self.parameters()),
            'sar_encoder': sum(p.numel() for p in self.sar_encoder.parameters()),
            'unet': sum(p.numel() for p in self.unet.parameters()),
        }


if __name__ == "__main__":
    print("Testing Residual FlowMatching Model...")
    
    # 创建模型
    model = ResidualFlowMatchingModel(
        base_ch=64,
        ch_mults=[1, 2, 4, 8],
        num_blocks=2,
        time_emb_dim=256,
        dropout=0.1,
        num_heads=8,
        use_normalized_residual=False
    )
    
    # 测试训练
    B, H, W = 2, 128, 128
    sar = torch.rand(B, 3, H, W)
    optical = torch.rand(B, 3, H, W)
    
    loss = model(sar, optical)
    print(f"Training loss: {loss.item():.4f}")
    
    loss, loss_dict = model(sar, optical, return_dict=True)
    print(f"Loss dict: {loss_dict}")
    
    # 测试采样
    with torch.no_grad():
        generated = model.sample(sar, steps=10, method='dpmpp')
        print(f"Generated shape: {generated.shape}")
        print(f"Generated range: [{generated.min():.3f}, {generated.max():.3f}]")
        
        # 测试只返回残差
        residual = model.sample(sar, steps=10, method='dpmpp', return_residual_only=True)
        print(f"Residual range: [{residual.min():.3f}, {residual.max():.3f}]")
    
    # 参数统计
    params = model.count_parameters()
    print(f"\nParameters:")
    for k, v in params.items():
        print(f"  {k}: {v/1e6:.2f}M")
    
    print("\nAll tests passed!")
