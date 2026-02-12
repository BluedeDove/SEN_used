"""
v2.models.flowmatching - Flow Matching 模型实现

基于 Rectified Flow 的 SAR-to-Optical 图像转换模型。
使用 DPM-Solver++ 进行高效采样。

核心组件:
    - FlowMatchingModel: 核心模型
    - FlowMatchingUNet: 带 SAR 引导的 UNet
    - DPMSolverPlusPlus: 高效采样器
    - FlowMatchingInterface: 模型接口

示例:
    >>> from models.flowmatching import FlowMatchingInterface
    >>> model = FlowMatchingInterface(config)
    >>> model.build_model('cuda')
    >>> output = model.get_output(sar, config)
"""

from .model import FlowMatchingModel
from .model_residual import ResidualFlowMatchingModel
from .unet import FlowMatchingUNet, SARGuidedBlock, TimeEmbedding
from .sampler import (
    FlowMatchingSampler,
    DPMSolverPlusPlus,
    EulerSampler,
    HeunSampler
)
from .interface import FlowMatchingInterface
from .interface_residual import ResidualFlowMatchingInterface

__all__ = [
    # 主要接口
    'FlowMatchingInterface',
    'FlowMatchingModel',
    'ResidualFlowMatchingInterface',
    'ResidualFlowMatchingModel',
    
    # UNet 组件
    'FlowMatchingUNet',
    'SARGuidedBlock',
    'TimeEmbedding',
    
    # 采样器
    'FlowMatchingSampler',
    'DPMSolverPlusPlus',
    'EulerSampler',
    'HeunSampler',
]
