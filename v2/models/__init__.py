"""
v2.models - 模型接口和实现

提供可扩展的模型接口，支持通过配置动态加载不同模型。
"""

from .base import BaseModelInterface, ModelOutput, CompositeMethod, ModelDebugInfo, ModelDebugReport
from .registry import MODEL_REGISTRY, register_model, get_model_class, create_model, list_available_models

# 导入各个模型模块以触发注册
from .srdm.interface import SRDMInterface

__all__ = [
    # base
    'BaseModelInterface',
    'ModelOutput',
    'CompositeMethod',
    'ModelDebugInfo',
    'ModelDebugReport',
    # registry
    'MODEL_REGISTRY',
    'register_model',
    'get_model_class',
    'create_model',
    'list_available_models',
    # models
    'SRDMInterface',
]
