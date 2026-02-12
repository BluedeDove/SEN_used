"""
unet.py - Flow Matching UNet with SAR-guided Fusion

实现带逐层 SAR 特征注入的 UNet 架构。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TimeEmbedding(nn.Module):
    """正弦时间嵌入"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Args:
            t: [B] 时间步 (0 到 1)
        Returns:
            [B, dim] 时间嵌入
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class SARGuidedBlock(nn.Module):
    """
    SAR 引导的残差块
    在标准卷积块基础上注入 SAR 特征
    """
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        sar_ch: int,
        dropout: float = 0.1,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        # 主卷积路径
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        # 时间嵌入投影
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        
        # SAR 特征融合 (Cross-Attention)
        self.sar_norm = nn.GroupNorm(32, out_ch)
        self.sar_proj = nn.Conv2d(sar_ch, out_ch, 1) if sar_ch != out_ch else nn.Identity()
        
        # 空间注意力融合 SAR
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 2, 2, 1),  # 生成 2 个通道的注意力权重
        )
        
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        
        # 跳跃连接
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
            
    def forward(self, x, t_emb, sar_feature):
        """
        Args:
            x: [B, in_ch, H, W]
            t_emb: [B, time_emb_dim]
            sar_feature: [B, sar_ch, H, W] SAR 特征
        Returns:
            [B, out_ch, H, W]
        """
        # 确保 SAR 特征与输入尺寸匹配
        if sar_feature.shape[2:] != x.shape[2:]:
            sar_feature = F.interpolate(sar_feature, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 注入时间信息
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        # SAR 特征引导 (逐层融合)
        h_norm = self.sar_norm(h)
        sar_proj = self.sar_proj(sar_feature)
        
        # 确保尺寸匹配 (二次检查)
        if sar_proj.shape[2:] != h_norm.shape[2:]:
            sar_proj = F.interpolate(sar_proj, size=h_norm.shape[2:], mode='bilinear', align_corners=False)
        
        # 计算空间注意力权重
        attn_input = torch.cat([h_norm, sar_proj], dim=1)
        attn_weights = self.spatial_attn(attn_input)  # [B, 2, H, W]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 加权融合: w1 * h + w2 * sar
        h = attn_weights[:, 0:1] * h_norm + attn_weights[:, 1:2] * sar_proj
        
        # 第二个卷积
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        # 残差连接
        return h + self.skip(x)


class DownBlock(nn.Module):
    """下采样块"""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        sar_ch: int,
        num_blocks: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        downsample: bool = True
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            SARGuidedBlock(
                in_ch if i == 0 else out_ch,
                out_ch,
                time_emb_dim,
                sar_ch,
                dropout,
                num_heads
            )
            for i in range(num_blocks)
        ])
        
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1) if downsample else nn.Identity()
        
    def forward(self, x, t_emb, sar_feature):
        """
        Args:
            x: [B, in_ch, H, W]
            t_emb: [B, time_emb_dim]
            sar_feature: [B, sar_ch, H, W]
        Returns:
            h: [B, out_ch, H//2, W//2] (if downsample)
            features: list of intermediate features for skip connection
        """
        features = []
        h = x
        
        for block in self.blocks:
            h = block(h, t_emb, sar_feature)
            features.append(h)
            
        h = self.downsample(h)
        return h, features


class UpBlock(nn.Module):
    """上采样块 - 简化版，只使用一个 skip connection"""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        sar_ch: int,
        num_blocks: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        upsample: bool = True
    ):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1) if upsample else nn.Identity()
        
        # 第一个 block 接收上采样后的特征 + skip connection
        self.block1 = SARGuidedBlock(
            in_ch + out_ch,  # 上采样输出 + skip
            out_ch,
            time_emb_dim,
            sar_ch,
            dropout,
            num_heads
        )
        
        # 后续的 blocks
        self.blocks = nn.ModuleList([
            SARGuidedBlock(
                out_ch,
                out_ch,
                time_emb_dim,
                sar_ch,
                dropout,
                num_heads
            )
            for _ in range(num_blocks - 1)
        ])
        
    def forward(self, x, skip_features, t_emb, sar_feature):
        """
        Args:
            x: [B, in_ch, H, W]
            skip_features: list of skip connections from encoder (使用最后一个)
            t_emb: [B, time_emb_dim]
            sar_feature: [B, sar_ch, H, W] - 应与上采样后的尺寸匹配
        Returns:
            [B, out_ch, H*2, W*2]
        """
        # 上采样
        h = self.upsample(x)
        
        # 调整 sar_feature 以匹配 h
        if sar_feature.shape[2:] != h.shape[2:]:
            sar_feature = F.interpolate(sar_feature, size=h.shape[2:], mode='bilinear', align_corners=False)
        
        # 连接 skip feature
        if skip_features:
            skip = skip_features[-1]  # 使用最后一个 skip
            # 确保尺寸匹配
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
        
        # 第一个 block
        h = self.block1(h, t_emb, sar_feature)
        
        # 后续 blocks
        for block in self.blocks:
            h = block(h, t_emb, sar_feature)
                
        return h


class MiddleBlock(nn.Module):
    """中间块 (bottleneck)"""
    
    def __init__(
        self,
        ch: int,
        time_emb_dim: int,
        sar_ch: int,
        dropout: float = 0.1,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.block1 = SARGuidedBlock(ch, ch, time_emb_dim, sar_ch, dropout, num_heads)
        self.block2 = SARGuidedBlock(ch, ch, time_emb_dim, sar_ch, dropout, num_heads)
        
    def forward(self, x, t_emb, sar_feature):
        h = self.block1(x, t_emb, sar_feature)
        h = self.block2(h, t_emb, sar_feature)
        return h


class FlowMatchingUNet(nn.Module):
    """
    Flow Matching UNet
    
    带逐层 SAR 特征融合的 UNet 架构。
    输入: x_t (当前状态), t (时间), SAR 特征
    输出: v (预测的向量场)
    """
    
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        base_ch: int = 64,
        ch_mults: list = [1, 2, 4, 8],
        num_blocks: int = 2,
        sar_chs: list = [64, 128, 256, 512],
        time_emb_dim: int = 256,
        dropout: float = 0.1,
        num_heads: int = 8
    ):
        """
        Args:
            in_ch: 输入通道数 (通常是 3，RGB)
            out_ch: 输出通道数 (向量场，也是 3)
            base_ch: 基础通道数
            ch_mults: 通道倍数 [1, 2, 4, 8]
            num_blocks: 每个分辨率下的块数
            sar_chs: SAR 特征通道数列表 [L1, L2, L3, L4]
            time_emb_dim: 时间嵌入维度
            dropout: dropout 概率
            num_heads: 注意力头数
        """
        super().__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_levels = len(ch_mults)
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # 输入投影
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        
        # 编码器 (下采样路径)
        self.encoder_blocks = nn.ModuleList()
        in_ch_current = base_ch
        
        for i, mult in enumerate(ch_mults):
            out_ch_current = base_ch * mult
            is_last = (i == len(ch_mults) - 1)
            
            self.encoder_blocks.append(
                DownBlock(
                    in_ch_current,
                    out_ch_current,
                    time_emb_dim,
                    sar_chs[i],
                    num_blocks=num_blocks,
                    dropout=dropout,
                    num_heads=num_heads,
                    downsample=not is_last
                )
            )
            in_ch_current = out_ch_current
            
        # 中间块
        mid_ch = base_ch * ch_mults[-1]
        self.middle_block = MiddleBlock(
            mid_ch, time_emb_dim, sar_chs[-1], dropout, num_heads
        )
        
        # 解码器 (上采样路径)
        self.decoder_blocks = nn.ModuleList()
        
        for i in reversed(range(len(ch_mults))):
            out_ch_current = base_ch * ch_mults[i]
            is_first = (i == 0)
            
            self.decoder_blocks.append(
                UpBlock(
                    in_ch_current,
                    out_ch_current,
                    time_emb_dim,
                    sar_chs[i],
                    num_blocks=num_blocks,
                    dropout=dropout,
                    num_heads=num_heads,
                    upsample=not is_first
                )
            )
            in_ch_current = out_ch_current
            
        # 输出投影
        self.norm_out = nn.GroupNorm(32, base_ch)
        self.conv_out = nn.Conv2d(base_ch, out_ch, 3, padding=1)
        
    def forward(self, x, t, sar_features: Dict[str, torch.Tensor]):
        """
        前向传播
        
        Args:
            x: [B, in_ch, H, W] 当前状态 x_t
            t: [B] 时间步 (0 到 1)
            sar_features: dict {'L1': [B, 64, H/2, W/2], 'L2': ..., ...}
            
        Returns:
            [B, out_ch, H, W] 预测的向量场 v
        """
        # 时间嵌入
        t_emb = self.time_embed(t)
        
        # 输入投影
        h = self.conv_in(x)
        
        # 编码器路径 (收集 skip connections)
        skip_connections = []
        
        for i, block in enumerate(self.encoder_blocks):
            level = i + 1
            sar_key = f'L{level}'
            
            # 调整 SAR 特征尺寸以匹配当前特征
            sar_feat = sar_features[sar_key]
            if sar_feat.shape[2:] != h.shape[2:]:
                sar_feat = F.interpolate(sar_feat, size=h.shape[2:], mode='bilinear', align_corners=False)
                
            h, skips = block(h, t_emb, sar_feat)
            skip_connections.append(skips)
            
        # 中间块
        sar_mid = sar_features[f'L{self.num_levels}']
        if sar_mid.shape[2:] != h.shape[2:]:
            sar_mid = F.interpolate(sar_mid, size=h.shape[2:], mode='bilinear', align_corners=False)
        h = self.middle_block(h, t_emb, sar_mid)
        
        # 解码器路径 (使用 skip connections)
        for i, block in enumerate(self.decoder_blocks):
            level = self.num_levels - i
            sar_key = f'L{level}'
            
            # 获取该层对应的 SAR 特征
            sar_feat = sar_features[sar_key]
            
            # 计算上采样后的目标尺寸
            if i < len(self.decoder_blocks) - 1:  # 不是最后一层，需要上采样
                target_size = (h.shape[2] * 2, h.shape[3] * 2)
            else:
                target_size = (h.shape[2], h.shape[3])
            
            # 将 SAR 特征上采样到目标尺寸
            # 注意：SAR 特征是编码器的输出，解码器需要上采样后的尺寸
            if sar_feat.shape[2:] != target_size:
                sar_feat = F.interpolate(sar_feat, size=target_size, mode='bilinear', align_corners=False)
                
            skips = skip_connections[-(i+1)]
            h = block(h, skips, t_emb, sar_feat)
            
        # 输出
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


if __name__ == "__main__":
    import math
    
    print("Testing FlowMatching UNet...")
    
    # 创建模型
    model = FlowMatchingUNet(
        in_ch=3,
        out_ch=3,
        base_ch=64,
        ch_mults=[1, 2, 4, 8],
        num_blocks=2,
        sar_chs=[64, 128, 256, 512],
        time_emb_dim=256,
        dropout=0.1,
        num_heads=8
    )
    
    # 测试输入
    B, H, W = 2, 128, 128
    x = torch.randn(B, 3, H, W)
    t = torch.rand(B)
    
    sar_features = {
        'L1': torch.randn(B, 64, H//2, W//2),
        'L2': torch.randn(B, 128, H//4, W//4),
        'L3': torch.randn(B, 256, H//8, W//8),
        'L4': torch.randn(B, 512, H//16, W//16),
    }
    
    # 前向传播
    output = model(x, t, sar_features)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 检查输出范围 (向量场理论上无界)
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("UNet test passed!")
