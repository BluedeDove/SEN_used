"""
interface.py - SRDM模型接口实现

实现BaseModelInterface，封装SRDM模型供高层使用。
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
from typing import Tuple, Optional
from models.base import BaseModelInterface, ModelOutput, CompositeMethod
from models.registry import register_model
from models.srdm.diffusion import SRDMDiffusion
from models.diffusion.schedule import Schedule
from core.amp_ops import AMPManager, create_amp_manager


@register_model('srdm')
class SRDMInterface(BaseModelInterface):
    """
    SRDM模型接口实现

    封装SRDMDiffusion，提供标准化接口。
    """

    def __init__(self, config: dict):
        """
        Args:
            config: 配置字典
        """
        super().__init__(config)
        self._output_range = (-1.0, 1.0)  # SRDM输出残差范围
        self._model = None
        self._amp_manager: Optional[AMPManager] = None

    def build_model(self, device: str = 'cpu'):
        """
        构建SRDM模型

        Args:
            device: 目标设备

        Returns:
            SRDMDiffusion实例
        """
        # 从配置读取参数
        srdm_cfg = self.config.get('srdm', {})
        diffusion_cfg = self.config.get('diffusion', {})
        loss_cfg = self.config.get('srdm_loss', None)

        # 创建调度器
        schedule = Schedule(
            num_timesteps=diffusion_cfg.get('num_timesteps', 1000),
            beta_start=diffusion_cfg.get('beta_start', 0.0001),
            beta_end=diffusion_cfg.get('beta_end', 0.02),
            device=device
        )

        # 创建模型
        self._model = SRDMDiffusion(
            schedule=schedule,
            base_ch=srdm_cfg.get('base_ch', 64),
            ch_mults=srdm_cfg.get('ch_mults', [1, 2, 4, 8]),
            num_blocks=srdm_cfg.get('num_blocks', 2),
            time_emb_dim=srdm_cfg.get('time_emb_dim', 256),
            dropout=srdm_cfg.get('dropout', 0.1),
            num_heads=srdm_cfg.get('num_heads', 8),
            clamp_output=True,
            loss_config=loss_cfg
        )

        self._model = self._model.to(device)
        self._device = torch.device(device)

        # 初始化 AMP 管理器
        self._amp_manager = create_amp_manager(self.config.get('amp', {}))

        return self._model

    def get_output(self, sar: torch.Tensor, config: dict) -> ModelOutput:
        """
        获取模型输出 - SRDM推理流程

        SRDM生成残差，然后与sar_base合成为最终图像:
        1. SAR编码得到sar_base
        2. 扩散采样得到残差
        3. 合成: sar_base + residual
        4. 截断负数并归一化到[0, 1]

        Args:
            sar: 输入SAR图像 [B, 1, H, W] @ [0, 1]
            config: 配置字典

        Returns:
            ModelOutput: 包含最终合成图像、中间结果和元数据
        """
        if self._model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        # 确保在正确设备上
        sar = sar.to(self._device)

        # 获取采样配置
        sampling = config.get('diffusion', {}).get('sampling', {})
        use_ddim = sampling.get('use_ddim', True)
        steps = sampling.get('ddim_steps', 50) if use_ddim else sampling.get('steps', 1000)
        eta = sampling.get('ddim_eta', 0.0)

        with torch.no_grad():
            # 使用 AMP 进行推理 (如果启用)
            autocast_context = self._amp_manager.autocast() if self._amp_manager else torch.no_grad()

            with autocast_context:
                # SAR编码
                sar_base, sar_features, _ = self._model.sar_encoder(sar)

                # 采样得到残差
                if use_ddim:
                    residual = self._model.ddim_sample(
                        sar, steps=steps, eta=eta, return_residual_only=True
                    )
                else:
                    residual = self._model.sample(
                        sar, steps=steps, return_residual_only=True
                    )

                # 合成: sar_base + residual
                composite = sar_base + residual

                # 截断负数
                composite = torch.clamp(composite, min=0.0)

                # 归一化到[0, 1]
                c_min = composite.min()
                c_max = composite.max()
                if c_max > c_min:
                    composite = (composite - c_min) / (c_max - c_min)
                else:
                    composite = torch.ones_like(composite) * 0.5

        return ModelOutput(
            generated=composite,  # 最终合成图像 [0, 1]
            output_range=(0.0, 1.0),
            intermediate={
                'residual': residual,      # 保留残差供调试
                'sar_base': sar_base,
                'sar_features': sar_features
            },
            metadata={
                'method': 'ddim' if use_ddim else 'ddpm',
                'steps': steps,
                'eta': eta
            }
        )

    def get_output_range(self) -> Tuple[float, float]:
        """
        返回模型输出的数值范围

        SRDM输出残差，范围[-1, 1]。

        Returns:
            (-1.0, 1.0)
        """
        return self._output_range

    def get_composite_method(self) -> CompositeMethod:
        """
        返回合成方法

        SRDM使用 add_then_clamp 方法:
        1. SAR_base + Residual
        2. 截断负数
        3. 归一化到[0, 1]

        Returns:
            CompositeMethod.ADD_THEN_CLAMP
        """
        return CompositeMethod.ADD_THEN_CLAMP

    def forward(self, sar: torch.Tensor, optical: torch.Tensor = None, return_dict: bool = False):
        """
        训练前向传播（代理到内部模型）

        Args:
            sar: SAR输入 [B, 1, H, W]
            optical: 光学图像 [B, 3, H, W]（训练时需要）
            return_dict: 是否返回详细损失字典

        Returns:
            loss 或 (loss, loss_dict)
        """
        if self._model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        return self._model(sar, optical, return_dict=return_dict)

    def __call__(self, sar: torch.Tensor, optical: torch.Tensor = None, return_dict: bool = False):
        """
        使对象可调用，代理到 forward 方法
        """
        return self.forward(sar, optical, return_dict=return_dict)

    def count_parameters(self):
        """计算模型参数量"""
        if self._model is None:
            return {'total': 0}
        return self._model.count_parameters()

    def debug(self, device, verbose: bool = False):
        """
        运行SRDM特有的调试测试

        Args:
            device: 运行设备（默认为self._device）
            verbose: 是否显示详细信息

        Returns:
            ModelDebugReport: 调试报告
        """
        from models.srdm.debug import create_srdm_debugger

        if device is None:
            device = self._device

        # 确保在评估模式
        self.eval()

        # 创建SRDM专用调试器并运行测试
        debugger = create_srdm_debugger(self)
        return debugger.run_all_tests(verbose=verbose)


if __name__ == "__main__":
    # 测试
    print("Testing interface.py...")

    config = {
        'model': {'type': 'srdm'},
        'srdm': {
            'base_ch': 64,
            'ch_mults': [1, 2, 4, 8],
            'num_blocks': 2,
            'time_emb_dim': 256,
            'dropout': 0.1,
            'num_heads': 8
        },
        'diffusion': {
            'num_timesteps': 1000,
            'sampling': {
                'use_ddim': True,
                'ddim_steps': 50,
                'ddim_eta': 0.0
            }
        }
    }

    # 创建接口
    interface = SRDMInterface(config)
    interface.build_model('cpu')

    # 测试参数计数
    params = interface.count_parameters()
    print(f"Total parameters: {params['total'] / 1e6:.2f}M")

    # 测试推理
    sar = torch.rand(1, 3, 128, 128)
    output = interface.get_output(sar, config)

    print(f"Output shape: {output.generated.shape}")
    print(f"Output range: {output.output_range}")
    print(f"Composite method: {interface.get_composite_method().value}")
    print(f"Metadata: {output.metadata}")

    print("\nAll tests passed!")
