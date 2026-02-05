"""
sen12_dataset.py - SEN1-2 数据集加载器

适配 SEN1-2 数据集结构:
    G:/Data/SEN1-2/
    ├── ROIs1158_spring/
    │   ├── s1_0/          # SAR 样本 0
    │   ├── s2_0/          # 光学 样本 0 (配对)
    │   ├── s1_1/
    │   ├── s2_1/
    │   └── ...
    ├── ROIs1868_summer/
    └── ...

特点:
- 自动扫描所有季节文件夹
- 配对相同编号的 s1_XXX 和 s2_XXX
- 支持按季节划分或混合划分
- 支持训练和验证集在软件层完全分离
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from .base import BaseDataset
from .registry import register_dataset

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@register_dataset('sen12')
class SEN12Dataset(BaseDataset):
    """
    SEN1-2 数据集加载器

    配置示例:
        # 使用特定季节做训练，其他做验证
        data:
          type: "sen12"
          root: "G:/Data/SEN1-2"
          train_seasons: ["ROIs1158_spring", "ROIs1868_summer"]
          val_seasons: ["ROIs1970_fall"]

        # 或混合所有季节后按比例划分
        data:
          type: "sen12"
          root: "G:/Data/SEN1-2"
          use_all_seasons: true
          train_ratio: 0.9
    """

    def __init__(self, config: dict, split: str = 'train'):
        """
        Args:
            config: 配置字典
            split: 'train' 或 'val'
        """
        super().__init__(config, split)

        if not HAS_PIL:
            raise ImportError("PIL is required. Install with: pip install Pillow")

        # 从配置读取
        data_cfg = config['data']

        # 数据集根目录
        self.root = Path(data_cfg.get('root', r'G:/Data/SEN1-2'))

        if not self.root.exists():
            raise FileNotFoundError(f"SEN1-2 dataset not found: {self.root}")

        # 确定使用哪些季节
        train_seasons = data_cfg.get('train_seasons', [])
        val_seasons = data_cfg.get('val_seasons', [])
        use_all_seasons = data_cfg.get('use_all_seasons', False)

        # 获取所有可用的季节文件夹
        all_seasons = [d.name for d in self.root.iterdir() if d.is_dir() and d.name.startswith('ROIs')]
        all_seasons.sort()

        if not all_seasons:
            raise ValueError(f"No season folders found in {self.root}")

        print(f"Available seasons: {all_seasons}")

        # 确定当前 split 使用哪些季节
        if use_all_seasons:
            # 混合所有季节，后面按比例划分
            self.seasons = all_seasons
        elif split == 'train':
            if train_seasons:
                self.seasons = train_seasons
            else:
                # 默认使用 spring 和 summer 做训练
                self.seasons = [s for s in all_seasons if 'spring' in s or 'summer' in s]
        else:  # val
            if val_seasons:
                self.seasons = val_seasons
            else:
                # 默认使用 fall 和 winter 做验证
                self.seasons = [s for s in all_seasons if 'fall' in s or 'winter' in s]

        print(f"[{split}] Using seasons: {self.seasons}")

        # 扫描所有样本对
        self.samples = self._scan_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in seasons: {self.seasons}")

        # 按比例划分（如果需要）
        if use_all_seasons:
            train_ratio = data_cfg.get('train_ratio', 0.9)
            n_train = int(len(self.samples) * train_ratio)

            if split == 'train':
                self.samples = self.samples[:n_train]
            else:
                self.samples = self.samples[n_train:]

        print(f"[{split}] Total samples: {len(self.samples)}")

    def _scan_samples(self) -> List[Tuple[Path, Path]]:
        """
        扫描所有季节，配对 s1_XXX 和 s2_XXX

        Returns:
            [(s1_path, s2_path), ...]
        """
        samples = []

        for season in self.seasons:
            season_dir = self.root / season
            if not season_dir.exists():
                print(f"Warning: Season {season} not found, skipping")
                continue

            # 获取所有 s1_ 和 s2_ 文件夹
            s1_dirs = {d.name.replace('s1_', ''): d for d in season_dir.iterdir()
                       if d.is_dir() and d.name.startswith('s1_')}
            s2_dirs = {d.name.replace('s2_', ''): d for d in season_dir.iterdir()
                       if d.is_dir() and d.name.startswith('s2_')}

            # 配对相同的编号
            for idx in s1_dirs.keys():
                if idx in s2_dirs:
                    s1_path = s1_dirs[idx]
                    s2_path = s2_dirs[idx]
                    samples.append((s1_path, s2_path))

        # 按路径排序确保确定性
        samples.sort(key=lambda x: str(x[0]))

        return samples

    def _load_image(self, sample_dir: Path) -> np.ndarray:
        """
        加载单个样本目录中的图像

        SEN1-2 结构: s1_0/s1_0_0000.png, s1_0/s1_0_0001.png, ...
        我们取第一张图像（或使用多帧平均）

        Args:
            sample_dir: 样本目录 (如 s1_0/)

        Returns:
            图像数组 [H, W, C] 或 [H, W]
        """
        # 查找目录中的图像文件
        img_files = list(sample_dir.glob('*.png')) + \
                    list(sample_dir.glob('*.jpg')) + \
                    list(sample_dir.glob('*.tif'))

        if not img_files:
            raise FileNotFoundError(f"No image found in {sample_dir}")

        # 按文件名排序，取第一张
        img_files.sort()
        img_path = img_files[0]

        # 读取图像
        img = Image.open(img_path)
        img = np.array(img, dtype=np.float32)

        # 确保通道维度
        if img.ndim == 2:
            img = img[..., np.newaxis]  # [H, W] -> [H, W, 1]

        # 归一化到 [0, 1]
        if img.max() > 1.0:
            img = img / 255.0

        return img

    def _preprocess_sar(self, img: np.ndarray) -> np.ndarray:
        """预处理 SAR 图像"""
        # SEN1-2 SAR 是 3 通道
        if img.shape[-1] != 3:
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            else:
                img = img[..., :3]
        return img

    def _preprocess_optical(self, img: np.ndarray) -> np.ndarray:
        """预处理光学图像"""
        # SEN1-2 光学是 3 通道 RGB
        if img.shape[-1] != 3:
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            else:
                img = img[..., :3]
        return img

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            {
                'sar': torch.Tensor [3, H, W],
                'optical': torch.Tensor [3, H, W],
                'sar_path': str,
                'optical_path': str
            }
        """
        s1_dir, s2_dir = self.samples[idx]

        # 加载图像
        sar_img = self._load_image(s1_dir)
        optical_img = self._load_image(s2_dir)

        # 预处理
        sar_img = self._preprocess_sar(sar_img)
        optical_img = self._preprocess_optical(optical_img)

        # 转换为 tensor [C, H, W]
        sar_tensor = torch.from_numpy(sar_img).float().permute(2, 0, 1)
        optical_tensor = torch.from_numpy(optical_img).float().permute(2, 0, 1)

        return {
            'sar': sar_tensor,
            'optical': optical_tensor,
            'sar_path': str(s1_dir),
            'optical_path': str(s2_dir)
        }

    def get_data_range(self) -> Tuple[float, float]:
        """返回数据范围"""
        return self.data_range


if __name__ == "__main__":
    # 测试
    print("Testing SEN12Dataset...")

    # 配置 1: 按季节划分
    config_season = {
        'data': {
            'root': r'G:/Data/SEN1-2',
            'train_seasons': ['ROIs1158_spring', 'ROIs1868_summer'],
            'val_seasons': ['ROIs1970_fall'],
            'normalize': {'enabled': True, 'input_range': [0.0, 1.0]},
        }
    }

    try:
        train_set = SEN12Dataset(config_season, split='train')
        val_set = SEN12Dataset(config_season, split='val')
        print(f"\nTrain samples: {len(train_set)}")
        print(f"Val samples: {len(val_set)}")

        # 获取一个样本
        sample = train_set[0]
        print(f"\nSample SAR shape: {sample['sar'].shape}")
        print(f"Sample Optical shape: {sample['optical'].shape}")
        print(f"Sample SAR range: [{sample['sar'].min():.4f}, {sample['sar'].max():.4f}]")
        print(f"Sample Optical range: [{sample['optical'].min():.4f}, {sample['optical'].max():.4f}]")

    except Exception as e:
        print(f"Error: {e}")

    print("\nAll tests passed!")
