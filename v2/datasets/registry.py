"""
registry.py - 数据集注册表

支持通过配置字符串动态加载数据集。
"""

from typing import Type, Dict
from .base import BaseDataset

# 数据集注册表
DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {}


def register_dataset(name: str):
    """
    注册数据集的装饰器

    Args:
        name: 数据集名称

    Example:
        @register_dataset('sar_optical')
        class SAROpticalDataset(BaseDataset):
            ...
    """
    def decorator(cls: Type[BaseDataset]):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset_class(name: str) -> Type[BaseDataset]:
    """
    根据名称获取数据集类

    Args:
        name: 数据集名称

    Returns:
        数据集类

    Raises:
        ValueError: 如果数据集不存在
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name]


def create_dataset(config: dict, split: str = 'train') -> BaseDataset:
    """
    工厂函数：根据配置创建数据集

    Args:
        config: 配置字典
        split: 'train' 或 'val'

    Returns:
        数据集实例
    """
    dataset_type = config.get('data', {}).get('type', 'sar_optical')
    dataset_class = get_dataset_class(dataset_type)
    return dataset_class(config, split)


def list_available_datasets():
    """
    列出所有可用的数据集

    Returns:
        数据集名称列表
    """
    return list(DATASET_REGISTRY.keys())


def import_datasets():
    """
    自动导入所有数据集模块以触发注册

    需要在应用启动时调用一次。
    """
    # 延迟导入以避免循环依赖
    try:
        from .sar_optical_dataset import WHUDataset
    except ImportError:
        pass

    try:
        from .sen12_dataset import SEN12Dataset
    except ImportError:
        pass


if __name__ == "__main__":
    # 测试
    print("Testing registry.py...")

    # 测试注册
    from .base import BaseDataset
    import torch

    @register_dataset('test_dataset')
    class TestDataset(BaseDataset):
        def __init__(self, config, split='train'):
            super().__init__(config, split)
            self.size = 10

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                'sar': torch.rand(1, 64, 64),
                'optical': torch.rand(3, 64, 64),
                'sar_path': 'test',
                'optical_path': 'test'
            }

    # 测试获取
    dataset_class = get_dataset_class('test_dataset')
    print(f"✓ Got dataset class: {dataset_class}")

    # 测试列出
    datasets = list_available_datasets()
    print(f"✓ Available datasets: {datasets}")

    print("All tests passed!")
