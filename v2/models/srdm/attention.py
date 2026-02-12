"""
attention.py - 注意力机制

实现HC-Attention（High-level Context Attention）用于SAR特征注入。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HCAttention(nn.Module):
    """
    High-level Context Attention (Channel-wise Attention)

    使用通道注意力而非空间注意力，避免 HW x HW 的大矩阵。
    """

    def __init__(
        self,
        in_channels: int,
        sar_channels: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        """
        Args:
            in_channels: 输入特征通道数
            sar_channels: SAR特征通道数
            num_heads: 注意力头数
            dropout: dropout概率
        """
        super().__init__()

        self.in_channels = in_channels
        self.sar_channels = sar_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        # 投影：将 SAR 特征投影到与输入特征相同的通道数
        self.sar_proj = nn.Conv2d(sar_channels, in_channels, 1)

        # Q, K, V 投影（使用通道维度）
        self.q_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.k_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.v_proj = nn.Conv2d(in_channels, in_channels, 1)

        # 输出投影
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = self.head_dim ** -0.5

    def forward(self, x, sar_feature):
        """
        前向传播 - 使用通道注意力

        Args:
            x: 输入特征 [B, C, H, W]
            sar_feature: SAR特征 [B, C_sar, H, W]

        Returns:
            输出特征 [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 投影 SAR 特征
        sar_proj = self.sar_proj(sar_feature)  # [B, C, H, W]

        # 平均池化到空间特征向量 [B, C, 1, 1]
        x_pooled = F.adaptive_avg_pool2d(x, 1)
        sar_pooled = F.adaptive_avg_pool2d(sar_proj, 1)

        # 计算 Q, K, V（通道维度上的注意力）
        # Q 来自输入特征，K, V 来自 SAR 特征
        q = self.q_proj(x_pooled).view(B, self.num_heads, self.head_dim)  # [B, heads, head_dim]
        k = self.k_proj(sar_pooled).view(B, self.num_heads, self.head_dim)  # [B, heads, head_dim]
        v = self.v_proj(sar_pooled).view(B, self.num_heads, self.head_dim)  # [B, heads, head_dim]

        # 通道注意力计算
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, head_dim, head_dim]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, heads, head_dim]

        # 重塑回通道形式
        out = out.view(B, C, 1, 1)

        # 上采样回原始尺寸
        out = out.expand(-1, -1, H, W)

        # 输出投影
        out = self.out_proj(out)

        # 残差连接
        return x + out


class CrossAttentionBlock(nn.Module):
    """
    交叉注意力块，结合残差连接
    """

    def __init__(
        self,
        in_channels: int,
        sar_channels: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()

        self.attn = HCAttention(in_channels, sar_channels, num_heads, dropout)
        self.norm = nn.GroupNorm(32, in_channels)

        # FFN
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, 1),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(in_channels * 4, in_channels, 1),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.norm2 = nn.GroupNorm(32, in_channels)

    def forward(self, x, sar_feature):
        """前向传播"""
        # 注意力
        h = self.norm(x)
        h = self.attn(h, sar_feature)
        x = x + h

        # FFN
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h

        return x


if __name__ == "__main__":
    # 测试
    print("Testing attention.py...")

    attn = HCAttention(128, 64, num_heads=8)
    x = torch.rand(2, 128, 32, 32)
    sar = torch.rand(2, 64, 32, 32)
    y = attn(x, sar)
    print(f"HCAttention: x{x.shape}, sar{sar.shape} -> {y.shape}")

    block = CrossAttentionBlock(128, 64, num_heads=8)
    y2 = block(x, sar)
    print(f"CrossAttentionBlock: {x.shape} -> {y2.shape}")

    print("All tests passed!")
