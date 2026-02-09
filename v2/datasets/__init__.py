"""
v2.datasets - 数据集接口和实现

提供可扩展的数据集接口，支持通过配置动态加载不同数据集。
"""

from .base import BaseDataset
from .registry import DATASET_REGISTRY, register_dataset, get_dataset_class, create_dataset, list_available_datasets

# 导入各个数据集模块以触发注册
from .whu.dataset import WHUDataset
from .sen12.dataset import SEN12Dataset

__all__ = [
    'BaseDataset',
    'DATASET_REGISTRY',
    'register_dataset',
    'get_dataset_class',
    'create_dataset',
    'list_available_datasets',
    'WHUDataset',
    'SEN12Dataset',
]
