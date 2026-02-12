"""
interface_residual.py - 残差版本 Flow Matching 模型接口

实现 BaseModelInterface，提供标准化的模型接口。
"""

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
from models.flowmatching.model_residual import ResidualFlowMatchingModel
from core.amp_ops import AMPManager, create_amp_manager


@register_model('flowmatching_residual')
class ResidualFlowMatchingInterface(BaseModelInterface):
    """
    残差版本 Flow Matching 模型接口实现
    
    预测残差 R = Optical - SAR_base 的流，更容易训练。
    使用 DPM-Solver++ 高效采样。
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: 配置字典
        """
        super().__init__(config)
        self._output_range = (-1.0, 1.0)  # 残差范围
        self._model: Optional[ResidualFlowMatchingModel] = None
        self._amp_manager: Optional[AMPManager] = None
        
    def build_model(self, device: str = 'cpu'):
        """
        构建残差 Flow Matching 模型
        
        Args:
            device: 目标设备
            
        Returns:
            ResidualFlowMatchingModel 实例
        """
        # 读取配置
        fm_cfg = self.config.get('flowmatching_residual', {})
        
        # SAR 编码器配置
        sar_encoder_cfg = self.config.get('sar_encoder', {
            'in_ch': 3,
            'base_ch': 64,
            'ch_mults': [1, 2, 4, 8],
            'global_dim': fm_cfg.get('time_emb_dim', 256)
        })
        
        # 创建模型
        self._model = ResidualFlowMatchingModel(
            base_ch=fm_cfg.get('base_ch', 64),
            ch_mults=fm_cfg.get('ch_mults', [1, 2, 4, 8]),
            num_blocks=fm_cfg.get('num_blocks', 2),
            time_emb_dim=fm_cfg.get('time_emb_dim', 256),
            dropout=fm_cfg.get('dropout', 0.1),
            num_heads=fm_cfg.get('num_heads', 8),
            sar_encoder_config=sar_encoder_cfg,
            clamp_output=fm_cfg.get('clamp_output', True),
            use_normalized_residual=fm_cfg.get('use_normalized_residual', False)
        )
        
        self._model = self._model.to(device)
        self._device = torch.device(device)
        
        # 初始化 AMP 管理器
        self._amp_manager = create_amp_manager(self.config.get('amp', {}))
        
        return self._model
    
    def get_output(self, sar: torch.Tensor, config: dict) -> ModelOutput:
        """
        获取模型输出 - 残差 Flow Matching 推理流程
        
        1. SAR 编码得到 sar_base
        2. DPM-Solver++ 采样得到残差
        3. 合成并归一化到 [0, 1]
        
        Args:
            sar: 输入 SAR 图像 [B, 1 or 3, H, W] @ [0, 1]
            config: 配置字典
            
        Returns:
            ModelOutput: 包含生成结果和元数据
        """
        if self._model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        # 确保在正确设备上
        sar = sar.to(self._device)
        
        # 获取采样配置
        sampling = config.get('sampling', {})
        method = sampling.get('method', 'dpmpp')
        steps = sampling.get('steps', 20)
        order = sampling.get('order', 3)
        skip_type = sampling.get('skip_type', 'time_uniform')
        
        with torch.no_grad():
            # 使用 AMP 进行推理 (如果启用)
            autocast_context = self._amp_manager.autocast() if self._amp_manager else torch.no_grad()
            
            with autocast_context:
                # 采样生成
                generated = self._model.sample(
                    sar,
                    steps=steps,
                    method=method,
                    order=order,
                    skip_type=skip_type
                )
        
        return ModelOutput(
            generated=generated,  # [B, 3, H, W] @ [0, 1]
            output_range=(0.0, 1.0),
            intermediate={},
            metadata={
                'method': method,
                'steps': steps,
                'order': order if method == 'dpmpp' else None
            }
        )
    
    def get_output_range(self) -> Tuple[float, float]:
        """
        返回模型输出的数值范围
        
        残差 Flow Matching 输出残差，范围 [-1, 1]。
        
        Returns:
            (-1.0, 1.0)
        """
        return self._output_range
    
    def get_composite_method(self) -> CompositeMethod:
        """
        返回合成方法
        
        残差版本使用 add_then_clamp 方法:
        1. SAR_base + Residual
        2. 截断负数
        3. 归一化到 [0, 1]
        
        Returns:
            CompositeMethod.ADD_THEN_CLAMP
        """
        return CompositeMethod.ADD_THEN_CLAMP
    
    def forward(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor = None,
        return_dict: bool = False
    ):
        """
        训练前向传播
        
        Args:
            sar: SAR 输入 [B, 1 or 3, H, W]
            optical: 光学图像 [B, 3, H, W] (训练时需要)
            return_dict: 是否返回详细损失字典
            
        Returns:
            loss 或 (loss, loss_dict)
        """
        if self._model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        return self._model(sar, optical, return_dict=return_dict)
    
    def __call__(self, *args, **kwargs):
        """使对象可调用，代理到 forward 方法"""
        return self.forward(*args, **kwargs)
    
    def count_parameters(self) -> dict:
        """计算模型参数量"""
        if self._model is None:
            return {'total': 0}
        return self._model.count_parameters()
    
    def debug(self, device, verbose: bool = False):
        """
        运行模型调试
        
        Args:
            device: 运行设备
            verbose: 是否显示详细信息
            
        Returns:
            ModelDebugReport: 调试报告
        """
        from models.base import ModelDebugInfo, ModelDebugReport
        
        if device is None:
            device = self._device
            
        self.eval()
        
        tests = []
        
        # 测试 1: 基本推理
        try:
            test_sar = torch.rand(1, 3, 64, 64, device=device)
            config = self.config
            output = self.get_output(test_sar, config)
            tests.append(ModelDebugInfo(
                component_name='inference',
                status='OK',
                message=f'Output shape: {output.generated.shape}, range: {output.output_range}',
                values={'shape': list(output.generated.shape)}
            ))
        except Exception as e:
            tests.append(ModelDebugInfo(
                component_name='inference',
                status='FAIL',
                message=str(e)
            ))
        
        # 测试 2: 训练 forward
        try:
            test_sar = torch.rand(2, 3, 64, 64, device=device)
            test_optical = torch.rand(2, 3, 64, 64, device=device)
            loss = self.forward(test_sar, test_optical)
            tests.append(ModelDebugInfo(
                component_name='training_forward',
                status='OK',
                message=f'Loss: {loss.item():.4f}',
                values={'loss': loss.item()}
            ))
        except Exception as e:
            tests.append(ModelDebugInfo(
                component_name='training_forward',
                status='FAIL',
                message=str(e)
            ))
        
        # 测试 3: 参数统计
        try:
            params = self.count_parameters()
            tests.append(ModelDebugInfo(
                component_name='parameters',
                status='OK',
                message=f'Total: {params["total"]/1e6:.2f}M',
                values=params
            ))
        except Exception as e:
            tests.append(ModelDebugInfo(
                component_name='parameters',
                status='FAIL',
                message=str(e)
            ))
        
        # 确定整体状态
        has_fail = any(t.status == 'FAIL' for t in tests)
        overall_status = 'FAILED' if has_fail else 'PASSED'
        
        return ModelDebugReport(
            model_name='ResidualFlowMatching',
            overall_status=overall_status,
            tests=tests,
            summary=f'Residual Flow Matching model debug: {overall_status}'
        )


if __name__ == "__main__":
    print("Testing Residual FlowMatching Interface...")
    
    config = {
        'model': {'type': 'flowmatching_residual'},
        'flowmatching_residual': {
            'base_ch': 64,
            'ch_mults': [1, 2, 4, 8],
            'num_blocks': 2,
            'time_emb_dim': 256,
            'dropout': 0.1,
            'num_heads': 8,
            'clamp_output': True,
            'use_normalized_residual': False
        },
        'sampling': {
            'method': 'dpmpp',
            'steps': 10,
            'order': 3
        }
    }
    
    # 创建接口
    interface = ResidualFlowMatchingInterface(config)
    interface.build_model('cpu')
    
    # 测试参数计数
    params = interface.count_parameters()
    print(f"Total parameters: {params['total']/1e6:.2f}M")
    
    # 测试推理
    sar = torch.rand(1, 3, 128, 128)
    output = interface.get_output(sar, config)
    
    print(f"Output shape: {output.generated.shape}")
    print(f"Output range: {output.output_range}")
    print(f"Composite method: {interface.get_composite_method().value}")
    print(f"Metadata: {output.metadata}")
    
    # 测试训练 forward
    optical = torch.rand(2, 3, 128, 128)
    loss = interface(sar.repeat(2, 1, 1, 1), optical)
    print(f"Training loss: {loss.item():.4f}")
    
    print("\nAll tests passed!")
