"""
base.py - 数据集接口基类

定义所有数据集必须实现的接口，确保一致性和可扩展性。
"""

# 支持单独运行调试：将项目根目录添加到路径
import sys
from pathlib import Path

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Tuple


class BaseDataset(Dataset, ABC):
    """
    所有数据集必须实现的接口基类

    子类必须实现:
    - __len__(): 返回数据集大小
    - __getitem__(): 返回样本字典
    - get_data_range(): 返回数据范围
    """

    def __init__(self, config: dict, split: str = 'train'):
        """
        初始化数据集

        Args:
            config: 配置字典
            split: 'train' 或 'val'
        """
        super().__init__()
        self.config = config
        self.split = split
        self.data_range = self._get_data_range()

    @abstractmethod
    def __len__(self) -> int:
        """
        返回数据集大小

        Returns:
            样本数量
        """
        pass

    @abstractmethod
    def __getitem__(self, idx) -> dict:
        """
        获取单个样本

        Returns:
            字典必须包含:
            {
                'sar': torch.Tensor [C, H, W],
                'optical': torch.Tensor [C, H, W] (训练时需要),
                'sar_path': str,
                'optical_path': str
            }
        """
        pass

    @abstractmethod
    def get_data_range(self) -> Tuple[float, float]:
        """
        返回数据集的数值范围

        Returns:
            (min, max) 元组
        """
        return self.data_range

    def _get_data_range(self) -> Tuple[float, float]:
        """
        从配置读取数据范围

        Returns:
            (min, max) 元组
        """
        normalize_cfg = self.config.get('data', {}).get('normalize', {})
        if normalize_cfg.get('enabled', True):
            return tuple(normalize_cfg.get('input_range', [0.0, 1.0]))
        return (0.0, 255.0)  # 未归一化时的默认范围


if __name__ == "__main__":
    # 测试
    print("Testing base.py...")

    # 测试抽象基类
    try:
        dataset = BaseDataset({'data': {'normalize': {'enabled': True, 'input_range': [0.0, 1.0]}}})
    except TypeError as e:
        print(f"✓ Abstract base class works: {e}")

    print("All tests passed!")
