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

# 支持单独运行调试：将项目根目录添加到路径
import sys
from pathlib import Path

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))
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

        # 获取所有可用的季节文件夹
        all_seasons = [d.name for d in self.root.iterdir() if d.is_dir() and d.name.startswith('ROIs')]
        all_seasons.sort()

        if not all_seasons:
            raise ValueError(f"No season folders found in {self.root}")

        print(f"Available seasons: {all_seasons}")

        # 读取划分配置
        train_ratio = data_cfg.get('train_ratio', 0.9)
        seed = data_cfg.get('seed', 42)  # 用于可复现的随机打乱

        # 扫描所有季节的样本，并按季节分别划分
        all_train_samples = []
        all_val_samples = []
        
        for season in all_seasons:
            season_samples = self._scan_season_samples(season)
            if not season_samples:
                continue
                
            # 在每个季节内按比例划分
            n_train = int(len(season_samples) * train_ratio)
            
            # 打乱该季节的样本顺序
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(season_samples))
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
            
            season_train = [season_samples[i] for i in train_indices]
            season_val = [season_samples[i] for i in val_indices]
            
            all_train_samples.extend(season_train)
            all_val_samples.extend(season_val)
            
            print(f"[{season}] Total: {len(season_samples)}, Train: {len(season_train)}, Val: {len(season_val)}")

        # 跨季节合并后再次打乱
        rng = np.random.RandomState(seed)
        if split == 'train':
            self.samples = all_train_samples
            indices = rng.permutation(len(self.samples))
            self.samples = [self.samples[i] for i in indices]
        else:  # val
            self.samples = all_val_samples
            indices = rng.permutation(len(self.samples))
            self.samples = [self.samples[i] for i in indices]

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split: {split}")

        print(f"[{split}] Total samples after cross-season merge and shuffle: {len(self.samples)}")

    def _scan_season_samples(self, season: str) -> List[Tuple[Path, Path, str]]:
        """
        扫描单个季节，配对 s1_XXX 和 s2_XXX 的所有 patches
        
        Args:
            season: 季节文件夹名称
            
        Returns:
            [(s1_path, s2_path, patch_name), ...]
        """
        samples = []
        season_dir = self.root / season
        
        if not season_dir.exists():
            print(f"Warning: Season {season} not found, skipping")
            return samples

        # 获取所有 s1_ 和 s2_ 文件夹
        s1_dirs = {d.name.replace('s1_', ''): d for d in season_dir.iterdir()
                   if d.is_dir() and d.name.startswith('s1_')}
        s2_dirs = {d.name.replace('s2_', ''): d for d in season_dir.iterdir()
                   if d.is_dir() and d.name.startswith('s2_')}

        # 配对相同的编号，并收集所有 patches
        for idx in s1_dirs.keys():
            if idx in s2_dirs:
                s1_path = s1_dirs[idx]
                s2_path = s2_dirs[idx]
                
                # 获取 s1 和 s2 目录中的所有图片文件
                s1_files = set(f.name.replace('s1_', '') for f in s1_path.iterdir()
                              if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff'])
                s2_files = set(f.name.replace('s2_', '') for f in s2_path.iterdir()
                              if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff'])
                
                # 取交集：只保留两个文件夹都存在的 patches
                common_patches = sorted(s1_files & s2_files)
                
                # 为每个共有的 patch 创建样本
                for patch_suffix in common_patches:
                    # SAR 文件名用 s1_ 前缀
                    sar_name = f's1_{patch_suffix}'
                    samples.append((s1_path, s2_path, sar_name))

        return samples

    def _scan_samples(self) -> List[Tuple[Path, Path]]:
        """
        扫描所有季节，配对 s1_XXX 和 s2_XXX
        (保留此方法用于向后兼容)
        
        Returns:
            [(s1_path, s2_path), ...]
        """
        samples = []
        
        # 如果存在 self.seasons 则使用，否则扫描所有季节
        seasons = getattr(self, 'seasons', None)
        if seasons is None:
            seasons = [d.name for d in self.root.iterdir()
                      if d.is_dir() and d.name.startswith('ROIs')]

        for season in seasons:
            samples.extend(self._scan_season_samples(season))

        # 按路径排序确保确定性
        samples.sort(key=lambda x: str(x[0]))

        return samples

    def _load_image(self, sample_dir: Path, patch_name: str = None) -> np.ndarray:
        """
        加载单个样本目录中的图像

        SEN1-2 结构: s1_0/s1_0_0000.png, s1_0/s1_0_0001.png, ...
        
        Args:
            sample_dir: 样本目录 (如 s1_0/)
            patch_name: 指定加载的图片文件名，如果为 None 则加载第一张

        Returns:
            图像数组 [H, W, C] 或 [H, W]
        """
        if patch_name is not None:
            # 加载指定的 patch
            img_path = sample_dir / patch_name
            if not img_path.exists():
                raise FileNotFoundError(f"Patch not found: {img_path}")
        else:
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
        s1_dir, s2_dir, sar_name = self.samples[idx]

        # SAR文件名是 s1_XXX.png，光学是 s2_XXX.png
        optical_name = sar_name.replace('s1_', 's2_')

        # 加载图像 (指定 patch)
        sar_img = self._load_image(s1_dir, sar_name)
        optical_img = self._load_image(s2_dir, optical_name)

        # 预处理
        sar_img = self._preprocess_sar(sar_img)
        optical_img = self._preprocess_optical(optical_img)

        # 转换为 tensor [C, H, W]
        sar_tensor = torch.from_numpy(sar_img).float().permute(2, 0, 1)
        optical_tensor = torch.from_numpy(optical_img).float().permute(2, 0, 1)

        return {
            'sar': sar_tensor,
            'optical': optical_tensor,
            'sar_path': str(s1_dir / sar_name),
            'optical_path': str(s2_dir / optical_name)
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
