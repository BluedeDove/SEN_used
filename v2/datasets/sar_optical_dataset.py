"""
sar_optical_dataset.py - WHU 数据集加载器

适配 WHU SAR-Optical 数据集结构:
    data/
    ├── train/
    │   ├── sar/           # SAR图像 (1通道 TIFF)
    │   └── opt/           # 光学图像 (3/4通道 TIFF)
    └── test/ (可选)
        ├── sar/
        └── opt/

特点:
- 支持4通道TIFF自动转换为3通道RGB
- 自动归一化到[0, 1]

# 支持单独运行调试：将项目根目录添加到路径
import sys
from pathlib import Path

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))
- 支持训练/验证集按比例划分
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple
from .base import BaseDataset
from .registry import register_dataset

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


@register_dataset('whu')
class WHUDataset(BaseDataset):
    """
    WHU SAR-Optical 数据集加载器

    配置示例:
        data:
          type: "whu"
          train_dir: "data/train"
          train_ratio: 0.9          # 90%训练，10%验证
          channels:
            optical:
              indices: [0, 1, 2]    # 取前3通道
            sar:
              indices: [0]          # 取第1通道

    数据结构:
        train_dir/
        ├── sar/                   # SAR图像 (.tif)
        └── opt/                   # 光学图像 (.tif)
    """

    def __init__(self, config: dict, split: str = 'train'):
        """
        Args:
            config: 配置字典
            split: 'train' 或 'val'
        """
        super().__init__(config, split)

        if not HAS_TIFFFILE:
            raise ImportError("tifffile is required. Install with: pip install tifffile")

        # 从配置读取
        data_cfg = config['data']
        train_dir = Path(data_cfg['train_dir'])

        # 扫描文件
        self.sar_files, self.optical_files = self._scan_files(train_dir)

        if len(self.sar_files) == 0:
            raise ValueError(f"No data found in {train_dir}")

        # 划分训练/验证集
        train_ratio = data_cfg.get('train_ratio', 0.9)
        n_train = int(len(self.sar_files) * train_ratio)

        if split == 'train':
            self.sar_files = self.sar_files[:n_train]
            self.optical_files = self.optical_files[:n_train]
        else:
            self.sar_files = self.sar_files[n_train:]
            self.optical_files = self.optical_files[n_train:]

        # 通道配置
        self.optical_channels = data_cfg.get('channels', {}).get('optical', {}).get('indices', [0, 1, 2])
        self.sar_channels = data_cfg.get('channels', {}).get('sar', {}).get('indices', [0])

        print(f"[{split}] Loaded {len(self.sar_files)} samples from WHU")

    def _scan_files(self, train_dir: Path) -> Tuple[list, list]:
        """
        扫描数据目录

        Args:
            train_dir: 训练数据目录

        Returns:
            (sar_files, optical_files)
        """
        sar_dir = train_dir / 'sar'
        optical_dir = train_dir / 'opt'

        if not sar_dir.exists():
            raise FileNotFoundError(f"SAR directory not found: {sar_dir}")
        if not optical_dir.exists():
            raise FileNotFoundError(f"Optical directory not found: {optical_dir}")

        # 扫描TIFF文件
        sar_files = sorted(sar_dir.glob('*.tif')) + sorted(sar_dir.glob('*.tiff'))
        optical_files = sorted(optical_dir.glob('*.tif')) + sorted(optical_dir.glob('*.tiff'))

        # 确保配对
        if len(sar_files) != len(optical_files):
            print(f"Warning: Number of SAR ({len(sar_files)}) and optical ({len(optical_files)}) files don't match")
            # 尝试按文件名配对
            min_len = min(len(sar_files), len(optical_files))
            sar_files = sar_files[:min_len]
            optical_files = optical_files[:min_len]

        return sar_files, optical_files

    def _load_and_preprocess(self, path: Path, channels: list) -> np.ndarray:
        """
        加载并预处理单张图像

        Args:
            path: 图像路径
            channels: 要使用的通道索引

        Returns:
            预处理后的图像 [H, W, C]
        """
        # 读取TIFF
        img = tifffile.imread(str(path))

        # 确保是3D (H, W, C)
        if img.ndim == 2:
            img = img[..., np.newaxis]

        # 确保通道在最后
        if img.shape[0] < img.shape[-1] and img.ndim == 3:
            # 可能是 (C, H, W)，转换为 (H, W, C)
            img = np.transpose(img, (1, 2, 0))

        # 选择通道
        if img.shape[-1] > len(channels):
            img = img[..., channels]
        elif img.shape[-1] < len(channels):
            # 如果通道不足，重复最后一个通道
            n_repeat = len(channels) - img.shape[-1]
            last_channel = img[..., -1:]
            img = np.concatenate([img] + [last_channel] * n_repeat, axis=-1)

        # 归一化到 [0, 1]
        img = img.astype(np.float32)
        if img.max() > 1.0:
            # 假设是uint8 [0, 255] 或 uint16 [0, 65535]
            if img.max() > 255:
                img = img / 65535.0
            else:
                img = img / 255.0

        return img

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.sar_files)

    def __getitem__(self, idx: int) -> dict:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            样本字典
        """
        # 加载图像
        sar = self._load_and_preprocess(self.sar_files[idx], self.sar_channels)
        optical = self._load_and_preprocess(self.optical_files[idx], self.optical_channels)

        # 转换为tensor [C, H, W]
        sar_tensor = torch.from_numpy(sar).float().permute(2, 0, 1)
        optical_tensor = torch.from_numpy(optical).float().permute(2, 0, 1)

        return {
            'sar': sar_tensor,
            'optical': optical_tensor,
            'sar_path': str(self.sar_files[idx]),
            'optical_path': str(self.optical_files[idx])
        }

    def get_data_range(self) -> Tuple[float, float]:
        """
        返回数据集的数值范围

        Returns:
            (0.0, 1.0)
        """
        return self.data_range


if __name__ == "__main__":
    # 测试
    print("Testing WHUDataset...")

    # 创建临时测试数据
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试目录结构
        sar_dir = Path(tmpdir) / 'sar'
        opt_dir = Path(tmpdir) / 'opt'
        sar_dir.mkdir()
        opt_dir.mkdir()

        # 创建测试图像
        for i in range(5):
            sar_img = np.random.rand(64, 64).astype(np.float32)
            opt_img = np.random.rand(64, 64, 4).astype(np.float32)

            tifffile.imwrite(sar_dir / f'{i:03d}.tif', sar_img)
            tifffile.imwrite(opt_dir / f'{i:03d}.tif', opt_img)

        # 创建数据集
        config = {
            'data': {
                'train_dir': tmpdir,
                'train_ratio': 0.8,
                'normalize': {'enabled': True, 'input_range': [0.0, 1.0]},
                'channels': {
                    'optical': {'indices': [0, 1, 2]},
                    'sar': {'indices': [0]}
                }
            }
        }

        dataset = WHUDataset(config, split='train')

        print(f"Dataset size: {len(dataset)}")
        print(f"Data range: {dataset.get_data_range()}")

        # 获取一个样本
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"SAR shape: {sample['sar'].shape}")
        print(f"Optical shape: {sample['optical'].shape}")
        print(f"SAR range: [{sample['sar'].min():.4f}, {sample['sar'].max():.4f}]")
        print(f"Optical range: [{sample['optical'].min():.4f}, {sample['optical'].max():.4f}]")

    print("\nAll tests passed!")
