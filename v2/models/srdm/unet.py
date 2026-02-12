"""
unet.py - 条件UNet

实现SRDM的条件UNet，用于预测残差上的噪声。
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
import math
from models.srdm.blocks import NAFBlock, Downsample, Upsample
from models.srdm.attention import CrossAttentionBlock


class TimestepEmbedding(nn.Module):
    """时间步嵌入"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        """
        Args:
            timesteps: [B] 整数时间步
        Returns:
            [B, dim] 嵌入向量
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class SRDMUNet(nn.Module):
    """
    SRDM条件UNet

    特点:
    - 输入: 带噪残差 [B, 3, H, W]
    - 条件: SAR多尺度特征
    - 输出: 预测的噪声 [B, 3, H, W]
    """

    def __init__(
        self,
        in_ch: int = 3,
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
            in_ch: 输入通道数（残差3通道）
            base_ch: 基础通道数
            ch_mults: 通道倍数
            num_blocks: 每层NAFBlock数量
            sar_chs: SAR特征通道数列表
            time_emb_dim: 时间嵌入维度
            dropout: dropout概率
            num_heads: 注意力头数
        """
        super().__init__()

        self.in_ch = in_ch
        self.base_ch = base_ch
        self.ch_mults = ch_mults
        self.num_levels = len(ch_mults)

        # 时间嵌入
        self.time_embed = nn.Sequential(
            TimestepEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # 输入投影
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # 编码器
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.encoder_projs = nn.ModuleList()  # 通道投影层

        in_ch_current = base_ch
        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult

            # NAFBlock组
            blocks = []
            for _ in range(num_blocks):
                blocks.append(NAFBlock(in_ch_current, dropout=dropout))
            self.encoder_blocks.append(nn.ModuleList(blocks))

            # 交叉注意力
            self.encoder_attns.append(
                CrossAttentionBlock(in_ch_current, sar_chs[i], num_heads, dropout)
            )

            # 通道投影：将特征投影到 out_ch
            self.encoder_projs.append(
                nn.Conv2d(in_ch_current, out_ch, 1) if in_ch_current != out_ch else nn.Identity()
            )

            # 下采样（最后一层不需要，因为已经有投影到 out_ch）
            if i < len(ch_mults) - 1:
                self.encoder_downs.append(Downsample(out_ch, out_ch))
                in_ch_current = out_ch
            else:
                self.encoder_downs.append(nn.Identity())
                in_ch_current = out_ch

        # 中间层
        self.middle_block1 = NAFBlock(in_ch_current, dropout=dropout)
        self.middle_attn = CrossAttentionBlock(in_ch_current, sar_chs[-1], num_heads, dropout)
        self.middle_block2 = NAFBlock(in_ch_current, dropout=dropout)

        # 时间嵌入投影到各层
        self.time_projs = nn.ModuleList()
        for mult in ch_mults:
            self.time_projs.append(nn.Linear(time_emb_dim, base_ch * mult * 2))

        # 解码器：从最深层(Level 3)到最浅层(Level 0)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()

        # 计算每层的通道数
        level_channels = [base_ch * m for m in ch_mults]  # [64, 128, 256, 512]

        for i in reversed(range(len(ch_mults))):
            out_ch = level_channels[i]  # 当前层输出通道

            # 上采样（除了最深层）
            if i < len(ch_mults) - 1:
                # 从更深层的通道数上采样到当前层
                deeper_ch = level_channels[i + 1]
                self.decoder_ups.insert(0, Upsample(deeper_ch, out_ch))
            else:
                # 最深层不需要上采样
                self.decoder_ups.insert(0, nn.Identity())

            # 交叉注意力：处理 out_ch 通道
            self.decoder_attns.insert(0,
                CrossAttentionBlock(out_ch, sar_chs[i], num_heads, dropout)
            )

            # NAFBlock组：处理 concat 后的通道数
            # h: 上采样后的特征 (out_ch 通道)
            # skip: 编码器跳跃连接 (out_ch 通道)
            # concat 后 = out_ch + out_ch = out_ch * 2
            # 最后一个 NAFBlock 后需要投影回 out_ch 通道
            block_in_ch = out_ch * 2
            
            blocks = []
            for j in range(num_blocks):
                blocks.append(NAFBlock(block_in_ch, dropout=dropout))
            # 添加 1x1 卷积将通道数投影回 out_ch
            blocks.append(nn.Conv2d(block_in_ch, out_ch, 1))
            self.decoder_blocks.insert(0, nn.Sequential(*blocks))

        # 输出投影
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, 3, padding=1)
        )

    def forward(self, x, t, sar_features):
        """
        前向传播

        Args:
            x: 带噪残差 [B, 3, H, W]
            t: 时间步 [B]
            sar_features: dict {'L1': [B, 64, H/2, W/2], ...}

        Returns:
            预测噪声 [B, 3, H, W]
        """
        # 时间嵌入
        t_emb = self.time_embed(t)  # [B, time_emb_dim]

        # 输入投影
        h = self.conv_in(x)  # [B, base_ch, H, W]

        # 编码器
        skips = []
        for i in range(self.num_levels):
            # 应用时间嵌入
            t_proj = self.time_projs[i](t_emb)
            t_proj = t_proj[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])
            # 简单的相加（可以通过更复杂的方式融合）

            # NAFBlock组
            for block in self.encoder_blocks[i]:
                h = block(h)

            # 交叉注意力（SAR条件）
            sar_feat = sar_features[f'L{i+1}']
            # 如果尺寸不匹配，进行插值
            if sar_feat.shape[2:] != h.shape[2:]:
                sar_feat = nn.functional.interpolate(
                    sar_feat, size=h.shape[2:], mode='bilinear', align_corners=False
                )
            h = self.encoder_attns[i](h, sar_feat)

            # 通道投影到目标通道数
            h = self.encoder_projs[i](h)

            skips.append(h)

            # 下采样
            h = self.encoder_downs[i](h)

        # 中间层
        h = self.middle_block1(h)
        sar_feat_mid = sar_features[f'L{self.num_levels}']
        if sar_feat_mid.shape[2:] != h.shape[2:]:
            sar_feat_mid = nn.functional.interpolate(
                sar_feat_mid, size=h.shape[2:], mode='bilinear', align_corners=False
            )
        h = self.middle_attn(h, sar_feat_mid)
        h = self.middle_block2(h)

        # 解码器：从深到浅 (i = 3, 2, 1, 0)
        for i in reversed(range(self.num_levels)):
            # 上采样 (直接使用 i 作为索引，因为 init 时用 insert(0, ...) 已经反转了顺序)
            h = self.decoder_ups[i](h)

            # 跳跃连接
            skip = skips[i]
            if h.shape[2:] != skip.shape[2:]:
                h = nn.functional.interpolate(
                    h, size=skip.shape[2:], mode='bilinear', align_corners=False
                )
            h = torch.cat([h, skip], dim=1)

            # NAFBlock组
            h = self.decoder_blocks[i](h)

            # 交叉注意力
            sar_feat = sar_features[f'L{i+1}']
            if sar_feat.shape[2:] != h.shape[2:]:
                sar_feat = nn.functional.interpolate(
                    sar_feat, size=h.shape[2:], mode='bilinear', align_corners=False
                )
            h = self.decoder_attns[i](h, sar_feat)

        # 输出
        out = self.conv_out(h)

        return out


if __name__ == "__main__":
    # 测试
    print("Testing unet.py...")

    unet = SRDMUNet(
        in_ch=3,
        base_ch=64,
        ch_mults=[1, 2, 4, 8],
        num_blocks=2,
        sar_chs=[64, 128, 256, 512],
        time_emb_dim=256,
        dropout=0.1,
        num_heads=8
    )

    x = torch.rand(2, 3, 256, 256)
    t = torch.randint(0, 1000, (2,))
    sar_features = {
        'L1': torch.rand(2, 64, 128, 128),
        'L2': torch.rand(2, 128, 64, 64),
        'L3': torch.rand(2, 256, 32, 32),
        'L4': torch.rand(2, 512, 16, 16),
    }

    y = unet(x, t, sar_features)
    print(f"UNet output: {y.shape}")

    print("All tests passed!")
