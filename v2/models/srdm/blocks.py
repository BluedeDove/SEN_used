"""
blocks.py - NAFBlock基础块

实现NAFNet风格的简单基线块，用于SRDM的UNet。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NAFBlock(nn.Module):
    """
    NAFNet Simple Baseline Block

    特点:
    - 深度可分离卷积
    - 简化非线性激活（仅使用ReLU）
    - 通道注意力机制
    """

    def __init__(
        self,
        in_channels: int,
        dw_expand_ratio: float = 2.0,
        ffn_expand_ratio: float = 2.0,
        dropout: float = 0.0
    ):
        """
        Args:
            in_channels: 输入通道数
            dw_expand_ratio: 深度卷积扩展比例
            ffn_expand_ratio: FFN扩展比例
            dropout: dropout概率
        """
        super().__init__()

        self.in_channels = in_channels
        self.dw_channels = int(in_channels * dw_expand_ratio)
        self.ffn_channels = int(in_channels * ffn_expand_ratio)

        # 第一层: 1x1 conv + 3x3 depthwise conv
        self.norm1 = nn.LayerNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.dw_channels, 1)
        self.conv2 = nn.Conv2d(
            self.dw_channels, self.dw_channels, 3,
            padding=1, groups=self.dw_channels
        )
        # SimpleGate将通道减半，所以conv3的输入是dw_channels // 2
        self.conv3 = nn.Conv2d(self.dw_channels // 2, in_channels, 1)

        # 通道注意力
        self.se = SimpleGate(self.dw_channels // 2)

        # Dropout
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # FFN
        self.norm2 = nn.LayerNorm(in_channels)
        self.conv4 = nn.Conv2d(in_channels, self.ffn_channels, 1)
        self.conv5 = nn.Conv2d(self.ffn_channels // 2, in_channels, 1)

        # 第二个SimpleGate
        self.se2 = SimpleGate(self.ffn_channels)

        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 缩放因子（用于训练稳定性）
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        """前向传播"""
        inp = x

        # 第一部分
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.gelu(x)

        # 通道注意力
        x = self.se(x)

        x = self.conv3(x)
        x = self.dropout1(x)

        # 残差连接 + 缩放
        y = inp + x * self.beta

        # 第二部分 (FFN)
        x = self.norm2(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv4(x)
        x = F.gelu(x)

        # 第二个SimpleGate
        x = self.se2(x)

        x = self.conv5(x)
        x = self.dropout2(x)

        # 残差连接 + 缩放
        return y + x * self.gamma


class SimpleGate(nn.Module):
    """
    简单门控机制

    将通道分成两部分，一部分作为门控。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        """前向传播"""
        # 分割通道
        x1, x2 = x.chunk(2, dim=1)
        # 门控
        return x1 * x2


class Downsample(nn.Module):
    """下采样模块"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """上采样模块"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)


if __name__ == "__main__":
    # 测试
    print("Testing blocks.py...")

    block = NAFBlock(64, dw_expand_ratio=2.0, ffn_expand_ratio=2.0)
    x = torch.rand(2, 64, 32, 32)
    y = block(x)
    print(f"NAFBlock: {x.shape} -> {y.shape}")

    down = Downsample(64, 128)
    y_down = down(x)
    print(f"Downsample: {x.shape} -> {y_down.shape}")

    up = Upsample(128, 64)
    y_up = up(y_down)
    print(f"Upsample: {y_down.shape} -> {y_up.shape}")

    print("All tests passed!")
