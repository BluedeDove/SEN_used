"""
dataset.py - SEN1-2 数据集加载器

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

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# 处理相对导入：当作为模块导入时使用相对导入，直接运行时动态导入
try:
    from ..base import BaseDataset
    from ..registry import register_dataset
except ImportError:
    # 直接运行时，通过 sys.path 动态导入
    import sys
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))
    from datasets.base import BaseDataset
    from datasets.registry import register_dataset
    
    # 创建一个空的注册装饰器（仅在直接运行时）
    def register_dataset(name):
        def decorator(cls):
            return cls
        return decorator


@register_dataset('sen12')
class SEN12Dataset(BaseDataset):
    """
    SEN1-2 数据集加载器

    配置示例:
        data:
          type: "sen12"
          root: "G:/Data/SEN1-2"
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
            [(s1_path, s2_path, full_filename), ...]
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

        # DEBUG统计
        total_s1_files = 0
        total_s2_files = 0
        total_paired = 0
        total_unpaired_s1 = 0
        total_unpaired_s2 = 0

        # 配对相同的编号，并收集所有 patches
        for idx in s1_dirs.keys():
            if idx in s2_dirs:
                s1_path = s1_dirs[idx]
                s2_path = s2_dirs[idx]
                
                # 获取 s1 和 s2 目录中的所有图片文件（完整文件名）
                s1_files = {f.name for f in s1_path.iterdir()
                           if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']}
                s2_files = {f.name for f in s2_path.iterdir()
                           if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']}
                
                total_s1_files += len(s1_files)
                total_s2_files += len(s2_files)
                
                # 提取 patch ID（去掉季节名、波段前缀等）
                # 文件名格式: {season}_s1_{idx}_pXXX.png -> 提取为 s1_{idx}_pXXX 作为匹配键
                s1_patches = {}
                s2_patches = {}
                
                for f in s1_files:
                    # 去掉扩展名
                    name = Path(f).stem
                    # 构建匹配键：将 {season}_s1_{idx}_pXXX 转为 s1_{idx}_pXXX
                    # 假设格式为 {season}_s1_{idx}_pXXX 或 s1_{season}_{idx}_pXXX
                    if f'_s1_{idx}_' in name:
                        # 提取 pXXX 部分作为 key
                        parts = name.split(f'_s1_{idx}_')
                        if len(parts) == 2:
                            patch_key = f's1_{idx}_{parts[1]}'
                            s1_patches[patch_key] = f
                
                for f in s2_files:
                    name = Path(f).stem
                    if f'_s2_{idx}_' in name:
                        parts = name.split(f'_s2_{idx}_')
                        if len(parts) == 2:
                            patch_key = f's2_{idx}_{parts[1]}'
                            # 与 s1 对应的 patch key（只改波段名）
                            s1_patch_key = patch_key.replace(f's2_{idx}_', f's1_{idx}_')
                            s2_patches[s1_patch_key] = f
                
                # 统计配对情况
                paired_keys = set(s1_patches.keys()) & set(s2_patches.keys())
                unpaired_s1 = set(s1_patches.keys()) - set(s2_patches.keys())
                unpaired_s2 = set(s2_patches.keys()) - set(s1_patches.keys())
                
                total_paired += len(paired_keys)
                total_unpaired_s1 += len(unpaired_s1)
                total_unpaired_s2 += len(unpaired_s2)
                
                # 取交集：只保留两个文件夹都存在的 patches
                common_keys = sorted(paired_keys)
                
                # 为每个共有的 patch 创建样本，存储完整文件名
                for patch_key in common_keys:
                    s1_filename = s1_patches[patch_key]
                    samples.append((s1_path, s2_path, s1_filename))

        # DEBUG输出
        print(f"  [DEBUG] {season}: S1文件={total_s1_files}, S2文件={total_s2_files}, "
              f"配对成功={total_paired}, S1未配对={total_unpaired_s1}, S2未配对={total_unpaired_s2}")

        return samples

    def _load_image(self, sample_dir: Path, patch_name: str = None) -> Optional[np.ndarray]:
        """
        加载单个样本目录中的图像（带错误处理）

        Args:
            sample_dir: 样本目录 (如 s1_0/)
            patch_name: 指定加载的图片文件名，如果为 None 则加载第一张

        Returns:
            图像数组 [H, W, C] 或 [H, W]，加载失败返回 None
        """
        try:
            if patch_name is not None:
                # 加载指定的 patch
                img_path = sample_dir / patch_name
                if not img_path.exists():
                    return None
            else:
                # 查找目录中的图像文件
                img_files = list(sample_dir.glob('*.png')) + \
                            list(sample_dir.glob('*.jpg')) + \
                            list(sample_dir.glob('*.tif'))

                if not img_files:
                    return None

                # 按文件名排序，取第一张
                img_files.sort()
                img_path = img_files[0]

            # 读取图像
            img = Image.open(img_path)
            
            # 检查图像是否损坏
            img.load()  # 这会强制加载图像数据，如果损坏会抛出异常
            img = np.array(img, dtype=np.float32)

            # 确保通道维度
            if img.ndim == 2:
                img = img[..., np.newaxis]  # [H, W] -> [H, W, 1]

            # 归一化到 [0, 1]
            if img.max() > 1.0:
                img = img / 255.0

            return img
            
        except Exception:
            # 静默跳过损坏文件，不在训练输出中显示警告
            return None

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
        获取单个样本（带损坏文件处理）

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
        # 尝试加载当前样本，如果损坏则尝试下一个
        max_attempts = min(10, len(self.samples))
        for attempt in range(max_attempts):
            try_idx = (idx + attempt) % len(self.samples)
            result = self._try_load_sample(try_idx)
            if result is not None:
                return result
        
        # 如果所有尝试都失败，返回一个默认样本（全零）
        print(f"[ERROR] Failed to load any valid sample after {max_attempts} attempts, returning default")
        return self._get_default_sample()
    
    def _try_load_sample(self, idx: int) -> Optional[dict]:
        """尝试加载单个样本，失败返回 None"""
        s1_dir, s2_dir, sar_name = self.samples[idx]

        # SAR文件名如 ROIs1868_summer_s1_90_p901.png，光学是 ROIs1868_summer_s2_90_p901.png
        optical_name = sar_name.replace('_s1_', '_s2_')

        # 加载图像
        sar_img = self._load_image(s1_dir, sar_name)
        optical_img = self._load_image(s2_dir, optical_name)

        # 检查是否加载成功
        if sar_img is None or optical_img is None:
            return None

        try:
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
        except Exception as e:
            print(f"[WARN] Failed to process sample {idx}: {e}")
            return None
    
    def _get_default_sample(self) -> dict:
        """返回默认样本（全零图像）"""
        # 假设标准尺寸为 256x256
        sar_tensor = torch.zeros(3, 256, 256)
        optical_tensor = torch.zeros(3, 256, 256)
        
        return {
            'sar': sar_tensor,
            'optical': optical_tensor,
            'sar_path': 'default',
            'optical_path': 'default'
        }

    def get_data_range(self) -> Tuple[float, float]:
        """返回数据范围"""
        return self.data_range


if __name__ == "__main__":
    """单独运行测试"""
    import sys

    # 支持单独运行调试：将项目根目录添加到路径
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

    # 重新导入以解决相对导入问题
    from datasets.base import BaseDataset
    from datasets.registry import register_dataset

    print("=" * 60)
    print("SEN1-2 Dataset - Standalone Test with DEBUG Stats")
    print("=" * 60)

    # 配置
    config = {
        'data': {
            'root': r'G:/Data/SEN1-2',
            'train_ratio': 0.9,
            'normalize': {'enabled': True, 'input_range': [0.0, 1.0]},
        }
    }

    print("\n[1/4] Creating SEN12 dataset...")
    try:
        train_set = SEN12Dataset(config, split='train')
        val_set = SEN12Dataset(config, split='val')
        print(f"  [OK] Train samples: {len(train_set)}")
        print(f"  [OK] Val samples: {len(val_set)}")
    except FileNotFoundError as e:
        print(f"  [SKIP] Dataset not found: {e}")
        print("  This is expected if SEN1-2 dataset is not available.")
        sys.exit(0)
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n[2/4] Loading sample...")
    try:
        sample = train_set[0]
        print(f"  [OK] Sample loaded:")
        print(f"       SAR shape: {sample['sar'].shape}")
        print(f"       Optical shape: {sample['optical'].shape}")
        print(f"       SAR range: [{sample['sar'].min():.4f}, {sample['sar'].max():.4f}]")
        print(f"       Optical range: [{sample['optical'].min():.4f}, {sample['optical'].max():.4f}]")
    except Exception as e:
        print(f"  [FAIL] Error loading sample: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n[3/4] Checking data range...")
    print(f"  [OK] Data range: {train_set.get_data_range()}")

    print("\n[4/4] DEBUG: Analyzing ALL samples (full scan)...")
    print("-" * 60)
    
    from collections import Counter
    import time
    
    # 尝试导入tqdm
    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False
        print("  [INFO] tqdm not installed, using simple progress display")
        print("  Install with: pip install tqdm")
    
    # 统计变量
    sar_shapes = []
    optical_shapes = []
    failed_samples = []  # (idx, error_msg, sample_info)
    sar_values = []
    optical_values = []
    
    dataset = train_set
    total = len(dataset)
    
    print(f"  Total samples to check: {total}")
    print(f"  Starting scan...")
    
    start_time = time.time()
    
    # 使用tqdm或简单循环
    if HAS_TQDM:
        iterator = tqdm(range(total), desc="Scanning samples", unit="img")
    else:
        iterator = range(total)
        last_progress = 0
    
    for idx in iterator:
        # 无tqdm时的进度显示（每10%显示一次）
        if not HAS_TQDM:
            progress = int(idx / total * 100)
            if progress >= last_progress + 10:
                elapsed = time.time() - start_time
                eta = (elapsed / (idx + 1)) * (total - idx - 1) if idx > 0 else 0
                print(f"  Progress: {progress}% ({idx}/{total}) - Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
                last_progress = progress
        
        try:
            sample = dataset[idx]
            sar = sample['sar']
            optical = sample['optical']
            
            # 记录尺寸
            sar_shapes.append(tuple(sar.shape))
            optical_shapes.append(tuple(optical.shape))
            
            # 记录数值用于计算均值方差（每100个采样一个）
            if idx % 100 == 0:
                sar_values.append(sar.numpy())
                optical_values.append(optical.numpy())
            
        except Exception as e:
            # 获取样本路径信息
            sample_info = dataset.samples[idx] if idx < len(dataset.samples) else ("N/A", "N/A", "N/A")
            failed_samples.append((idx, str(e), sample_info))
    
    elapsed_total = time.time() - start_time
    print(f"  Scan completed in {elapsed_total:.1f}s")
    
    # 尺寸统计
    sar_shape_counts = Counter(sar_shapes)
    optical_shape_counts = Counter(optical_shapes)
    
    print(f"\n  [DEBUG] Total samples checked: {len(sar_shapes)}")
    print(f"  [DEBUG] Failed samples: {len(failed_samples)} ({len(failed_samples)/total*100:.2f}%)")
    
    # 按季节统计损坏情况
    if failed_samples:
        print("\n  [DEBUG] Failed samples by season:")
        season_failures = Counter()
        for idx, error, (s1_dir, s2_dir, sar_name) in failed_samples:
            # 从路径提取季节名
            season = "Unknown"
            if isinstance(s1_dir, Path):
                parts = s1_dir.parts
                for p in parts:
                    if 'ROIs' in p:
                        season = p
                        break
            season_failures[season] += 1
        
        for season, count in season_failures.most_common():
            print(f"          {season}: {count} failures")
        
        # 显示前20个损坏样本的详细信息
        print("\n  [DEBUG] First 20 failed sample details:")
        for i, (idx, error, (s1_dir, s2_dir, sar_name)) in enumerate(failed_samples[:20]):
            print(f"          [{i+1}] Index: {idx}")
            print(f"              SAR Path: {s1_dir / sar_name}")
            print(f"              Error: {error[:100]}")
    
    print("\n  [DEBUG] SAR Shape Distribution:")
    for shape, count in sar_shape_counts.most_common():
        print(f"          {shape}: {count} samples ({count/len(sar_shapes)*100:.1f}%)")
    
    print("\n  [DEBUG] Optical Shape Distribution:")
    for shape, count in optical_shape_counts.most_common():
        print(f"          {shape}: {count} samples ({count/len(optical_shapes)*100:.1f}%)")
    
    # 计算均值和方差
    if sar_values:
        sar_stack = np.stack([v.reshape(-1) for v in sar_values[:min(1000, len(sar_values))]])
        sar_mean = np.mean(sar_stack)
        sar_std = np.std(sar_stack)
        print(f"\n  [DEBUG] SAR Value Statistics (sampled {len(sar_values)} images):")
        print(f"          Mean: {sar_mean:.4f}, Std: {sar_std:.4f}")
    
    if optical_values:
        optical_stack = np.stack([v.reshape(-1) for v in optical_values[:min(1000, len(optical_values))]])
        optical_mean = np.mean(optical_stack)
        optical_std = np.std(optical_stack)
        print(f"\n  [DEBUG] Optical Value Statistics (sampled {len(optical_values)} images):")
        print(f"          Mean: {optical_mean:.4f}, Std: {optical_std:.4f}")
    
    # 一致性检查
    if len(sar_shape_counts) == 1 and len(optical_shape_counts) == 1:
        print("\n  [OK] All images have consistent shapes!")
    else:
        print("\n  [WARNING] Image shapes are NOT consistent!")
        print(f"          SAR: {len(sar_shape_counts)} unique shapes")
        print(f"          Optical: {len(optical_shape_counts)} unique shapes")
    
    # 损坏率警告
    if len(failed_samples) > 0:
        print(f"\n  [WARNING] Found {len(failed_samples)} corrupted images!")
        print(f"            Failure rate: {len(failed_samples)/total*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
