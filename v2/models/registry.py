"""
registry.py - 模型注册表

支持通过配置字符串动态加载模型。
"""

from typing import Type, Dict
from .base import BaseModelInterface

# 模型注册表
MODEL_REGISTRY: Dict[str, Type[BaseModelInterface]] = {}


def register_model(name: str):
    """
    注册模型的装饰器

    Args:
        name: 模型名称

    Example:
        @register_model('srdm')
        class SRDMInterface(BaseModelInterface):
            ...
    """
    def decorator(cls: Type[BaseModelInterface]):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str) -> Type[BaseModelInterface]:
    """
    根据名称获取模型类

    Args:
        name: 模型名称

    Returns:
        模型类

    Raises:
        ValueError: 如果模型不存在
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]


def create_model(config: dict, device: str = 'cpu') -> BaseModelInterface:
    """
    工厂函数：根据配置创建模型

    Args:
        config: 配置字典
        device: 目标设备

    Returns:
        模型接口实例
    """
    # 支持 model.type 和 model.name 两种配置方式
    # 优先使用 type（模型配置中），其次使用 name（主配置中）
    model_cfg = config.get('model', {})
    model_type = model_cfg.get('type') or model_cfg.get('name', 'srdm')
    model_class = get_model_class(model_type)

    # 创建实例
    model_interface = model_class(config)

    # 构建模型
    model_interface.build_model(device)

    return model_interface


def list_available_models():
    """
    列出所有可用的模型

    Returns:
        模型名称列表
    """
    return list(MODEL_REGISTRY.keys())


def import_models():
    """
    自动导入所有模型模块以触发注册

    需要在应用启动时调用一次。
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[DEBUG] import_models() called, current registry: {list(MODEL_REGISTRY.keys())}")
    
    # 延迟导入以避免循环依赖
    try:
        from .srdm import SRDMInterface
        logger.info(f"[DEBUG] SRDMInterface imported successfully, registry now: {list(MODEL_REGISTRY.keys())}")
    except ImportError as e:
        logger.warning(f"[DEBUG] Failed to import SRDMInterface: {e}")
        pass
    
    # 导入 FlowMatching 模型
    try:
        from .flowmatching import FlowMatchingInterface
        logger.info(f"[DEBUG] FlowMatchingInterface imported successfully, registry now: {list(MODEL_REGISTRY.keys())}")
    except ImportError as e:
        logger.warning(f"[DEBUG] Failed to import FlowMatchingInterface: {e}")
        pass
    
    # 导入 Residual FlowMatching 模型
    try:
        from .flowmatching import ResidualFlowMatchingInterface
        logger.info(f"[DEBUG] ResidualFlowMatchingInterface imported successfully, registry now: {list(MODEL_REGISTRY.keys())}")
    except ImportError as e:
        logger.warning(f"[DEBUG] Failed to import ResidualFlowMatchingInterface: {e}")
        pass


if __name__ == "__main__":
    # 测试
    print("Testing registry.py...")

    # 测试注册
    from .base import BaseModelInterface, ModelOutput, CompositeMethod
    import torch

    @register_model('test_model')
    class TestModel(BaseModelInterface):
        def build_model(self, device='cpu'):
            self._model = torch.nn.Linear(10, 10)
            return self._model

        def get_output(self, sar, config):
            return ModelOutput(
                generated=torch.rand(1, 3, 64, 64),
                output_range=(0.0, 1.0)
            )

        def get_composite_method(self):
            return CompositeMethod.DIRECT

    # 测试获取
    model_class = get_model_class('test_model')
    print(f"✓ Got model class: {model_class}")

    # 测试列出
    models = list_available_models()
    print(f"✓ Available models: {models}")

    print("All tests passed!")
