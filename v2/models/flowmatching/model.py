"""
model.py - Flow Matching 核心模型

实现基于 Rectified Flow 的 SAR-to-Optical 转换模型。
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


class FlowMatchingModel(nn.Module):
    """
    Flow Matching Model for SAR-to-Optical Translation
    
    基于 Rectified Flow (直线流) 的图像转换模型。
    使用 DPM-Solver++ 进行高效采样。
    
    训练:
        学习向量场 v_θ(x_t, t, SAR) ≈ x_1 - x_0
        其中 x_t = t * x_1 + (1-t) * x_0
    
    采样:
        从 x_0 ~ N(0, I) 开始，通过 ODE 求解器积分到 t=1
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
        use_sar_base: bool = False
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
            use_sar_base: 是否将输出与 sar_base 合成
        """
        super().__init__()
        
        self.use_sar_base = use_sar_base
        
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
        
        # Flow Matching UNet (预测向量场)
        self.unet = FlowMatchingUNet(
            in_ch=3,  # 输入是 x_t (3通道)
            out_ch=3,  # 输出是向量场 v (3通道)
            base_ch=base_ch,
            ch_mults=ch_mults,
            num_blocks=num_blocks,
            sar_chs=sar_chs,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
            num_heads=num_heads
        )
        
    def get_sar_features(self, sar: torch.Tensor):
        """
        提取 SAR 特征
        
        Args:
            sar: [B, C, H, W] SAR 图像
            
        Returns:
            sar_base: [B, 3, H, W] SAR 基础图像
            sar_features: dict 多尺度特征
            global_cond: [B, D] 全局条件
        """
        return self.sar_encoder(sar)
    
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
        
        # 2. Conditional Flow Matching 训练
        # 采样时间步 t ~ Uniform(0, 1)
        t = torch.rand(B, device=device)
        
        # 采样噪声 x_0 ~ N(0, I)
        x_0 = torch.randn_like(optical)
        
        # 计算插值 x_t = t * optical + (1-t) * x_0
        # 这是标准的 Rectified Flow 概率路径
        t_expanded = t.view(B, 1, 1, 1)
        x_t = t_expanded * optical + (1 - t_expanded) * x_0
        
        # 目标向量场 u_t = dx_t/dt = optical - x_0
        u_t = optical - x_0
        
        # 3. 模型预测条件向量场 v_theta(x_t, t, SAR)
        v_pred = self.unet(x_t, t, sar_features)
        
        # 4. 计算 Flow Matching 损失
        # 这是标准的 Conditional Flow Matching 目标
        loss = F.mse_loss(v_pred, u_t)
        
        # 可选：添加 x_0 预测损失 (对于小 t 更稳定)
        # 从 v_pred 反推预测的 x_0
        # x_t = t * x_1 + (1-t) * x_0
        # v_pred = x_1 - x_0 (近似)
        # 所以 x_0_pred = x_t - t * v_pred
        if self.training:
            with torch.no_grad():
                x_0_pred = x_t - t_expanded * v_pred
                x_0_loss = F.mse_loss(x_0_pred, x_0)
        else:
            x_0_loss = torch.tensor(0.0, device=device)
        
        if return_dict:
            loss_dict = {
                'flow_matching_loss': loss.item(),
                'x_0_pred_loss': x_0_loss.item(),
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
        return_flow_only: bool = False,
        **sampler_kwargs
    ) -> torch.Tensor:
        """
        采样生成光学图像
        
        Args:
            sar: [B, 1 or 3, H, W] SAR 图像
            steps: 采样步数
            method: 采样方法 ('euler', 'heun', 'dpmpp')
            return_flow_only: 是否只返回流的结果 (不合成)
            **sampler_kwargs: 传递给采样器的额外参数
            
        Returns:
            generated: [B, 3, H, W] 生成的光学图像 [0, 1]
        """
        # 确保 SAR 是 3 通道
        if sar.shape[1] == 1:
            sar = sar.repeat(1, 3, 1, 1)
            
        B, _, H, W = sar.shape
        device = sar.device
        
        # 1. SAR 编码 (只执行一次)
        sar_base, sar_features, global_cond = self.sar_encoder(sar)
        
        # 2. 从噪声开始
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
        
        x_1 = sampler.sample(x_t, sar_features, global_cond)
        
        # 4. 后处理
        if self.use_sar_base and not return_flow_only:
            # 与 sar_base 合成
            # 注意：这里假设网络学习的是从噪声到 optical 的流
            # 如果 use_sar_base=True，可能需要调整训练目标
            output = sar_base + x_1
            output = torch.clamp(output, 0.0, 1.0)
        else:
            # 直接输出
            output = torch.clamp(x_1, 0.0, 1.0)
            
        return output
    
    def count_parameters(self) -> Dict[str, int]:
        """统计参数量"""
        return {
            'total': sum(p.numel() for p in self.parameters()),
            'sar_encoder': sum(p.numel() for p in self.sar_encoder.parameters()),
            'unet': sum(p.numel() for p in self.unet.parameters()),
        }


if __name__ == "__main__":
    print("Testing FlowMatching Model...")
    
    # 创建模型
    model = FlowMatchingModel(
        base_ch=64,
        ch_mults=[1, 2, 4, 8],
        num_blocks=2,
        time_emb_dim=256,
        dropout=0.1,
        num_heads=8
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
    
    # 参数统计
    params = model.count_parameters()
    print(f"\nParameters:")
    for k, v in params.items():
        print(f"  {k}: {v/1e6:.2f}M")
    
    print("\nAll tests passed!")
