"""
encoder.py - SAR编码器

实现多尺度SAR特征提取器，输出SAR_base、多尺度特征和全局条件。
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


class SAREncoder(nn.Module):
    """
    SAR编码器

    功能:
    1. 提取多尺度SAR特征 (L1-L4)
    2. 生成SAR_base（通过上采样到原分辨率）
    3. 生成全局条件向量
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 64,
        ch_mults: list = [1, 2, 4, 8],
        global_dim: int = 256
    ):
        """
        Args:
            in_ch: 输入通道数（SAR通常是1）
            base_ch: 基础通道数
            ch_mults: 通道倍数列表 [1, 2, 4, 8]
            global_dim: 全局条件维度
        """
        super().__init__()

        self.in_ch = in_ch
        self.base_ch = base_ch
        self.ch_mults = ch_mults
        self.global_dim = global_dim

        # 计算各层通道数
        self.chs = [base_ch * m for m in ch_mults]

        # 初始卷积
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # 下采样编码器
        self.down_blocks = nn.ModuleList()
        in_ch_current = base_ch

        for mult in ch_mults:
            out_ch = base_ch * mult
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch_current, out_ch, 3, stride=2, padding=1),
                    nn.GroupNorm(32, out_ch),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.GroupNorm(32, out_ch),
                    nn.SiLU(),
                )
            )
            in_ch_current = out_ch

        # 全局池化 + MLP 生成全局条件
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_mlp = nn.Sequential(
            nn.Linear(self.chs[-1], global_dim),
            nn.SiLU(),
            nn.Linear(global_dim, global_dim)
        )

        # SAR_base生成器（从L4上采样）
        self.base_generator = nn.Sequential(
            nn.Conv2d(self.chs[-1], self.chs[-1], 3, padding=1),
            nn.GroupNorm(32, self.chs[-1]),
            nn.SiLU(),
        )

        # 上采样到原分辨率
        self.upsample_layers = nn.ModuleList()
        for i in range(len(ch_mults) - 1, -1, -1):
            if i == 0:
                out_ch = 3  # 输出3通道（模拟RGB）
            else:
                out_ch = self.chs[i - 1]

            # 最后一层使用 Sigmoid 确保输出在 [0, 1] 范围
            is_last = (i == 0)
            self.upsample_layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(self.chs[i] if i == len(ch_mults) - 1 else self.chs[i], out_ch, 3, padding=1),
                    nn.GroupNorm(32, out_ch) if not is_last else nn.Identity(),
                    nn.SiLU() if not is_last else nn.Identity(),
                    nn.Sigmoid() if is_last else nn.Identity(),  # 限制输出 [0, 1]
                )
            )

    def forward(self, x):
        """
        前向传播

        Args:
            x: SAR输入 [B, in_ch, H, W]

        Returns:
            sar_base: [B, 3, H, W] - 上采样到RGB空间
            features: dict - 多尺度特征 {'L1': [B, 64, H/2, W/2], ...}
            global_cond: [B, global_dim] - 全局条件
        """
        B, _, H, W = x.shape

        # 初始卷积
        h = self.conv_in(x)  # [B, base_ch, H, W]

        # 下采样提取多尺度特征
        features = {}
        for i, down_block in enumerate(self.down_blocks):
            h = down_block(h)
            features[f'L{i+1}'] = h  # L1, L2, L3, L4

        # 全局条件
        global_feat = self.global_pool(h).view(B, -1)  # [B, chs[-1]]
        global_cond = self.global_mlp(global_feat)  # [B, global_dim]

        # 生成SAR_base
        base_feat = self.base_generator(h)  # [B, chs[-1], H/16, W/16]

        # 上采样到原分辨率
        for upsample_layer in self.upsample_layers:
            base_feat = upsample_layer(base_feat)

        sar_base = base_feat  # [B, 3, H, W]

        return sar_base, features, global_cond


if __name__ == "__main__":
    # 测试
    print("Testing encoder.py...")

    encoder = SAREncoder(
        in_ch=1,
        base_ch=64,
        ch_mults=[1, 2, 4, 8],
        global_dim=256
    )

    x = torch.rand(2, 1, 256, 256)
    sar_base, features, global_cond = encoder(x)

    print(f"SAR base: {sar_base.shape}")
    print(f"Global cond: {global_cond.shape}")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")

    print("All tests passed!")
