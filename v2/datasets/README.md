# 数据集开发规范 (Vibe Coding Guide)

> **目标**：让 AI 能快速理解和添加成对图像转换数据集（SAR → Optical）

## 快速开始 (TL;DR)

添加一个新数据集只需 **3 步**：

```python
# 1. 创建数据集文件夹 v2/datasets/your_dataset/
#    ├── dataset.py      # 必须：实现数据集类
#    ├── config.yaml     # 必须：配置模板
#    └── __init__.py     # 可选：导出数据集类

# 2. 继承 BaseDataset 并实现 3 个抽象方法

# 3. 使用 @register_dataset('your_dataset') 装饰器注册
```

---

## 项目架构概览

```
v2/datasets/
├── base.py              # 基类定义 (别碰)
├── registry.py          # 注册机制 (别碰)
├── README.md            # 本文档
├── whu/                 # WHU 数据集示例
│   ├── __init__.py
│   ├── dataset.py       # 数据集实现
│   └── config.yaml      # 配置模板
├── sen12/               # SEN1-2 数据集示例
│   ├── __init__.py
│   ├── dataset.py
│   └── config.yaml
└── your_dataset/        # 你的新数据集在这里
    ├── __init__.py      # 导出数据集类
    ├── dataset.py       # 数据集实现 (必须)
    └── config.yaml      # 默认配置 (必须)
```

---

## 必须实现的接口

### 1. 继承 `BaseDataset`

```python
import torch
import numpy as np
from pathlib import Path
from typing import Tuple
from ..base import BaseDataset
from ..registry import register_dataset


@register_dataset('your_dataset_name')  # 注册名称，用于配置文件中 data.type
class YourDataset(BaseDataset):
    """
    你的数据集加载器
    
    适配数据结构:
        data/
        ├── sar/           # SAR 图像
        └── optical/       # 光学图像 (配对)
    """
    
    def __init__(self, config: dict, split: str = 'train'):
        """
        Args:
            config: 配置字典
            split: 'train' 或 'val'
        """
        super().__init__(config, split)
        
        # 从配置读取参数
        data_cfg = config['data']
        
        # 扫描数据文件
        self.samples = self._scan_files(data_cfg)
        
        # 划分训练/验证集
        if split == 'train':
            self.samples = self.samples[:n_train]
        else:
            self.samples = self.samples[n_train:]
        
        print(f"[{split}] Loaded {len(self.samples)} samples")
```

### 2. 实现 3 个抽象方法

#### `__len__(self) -> int`

返回数据集大小（样本数量）。

```python
def __len__(self) -> int:
    """返回数据集大小"""
    return len(self.samples)
```

---

#### `__getitem__(self, idx) -> dict`

获取单个样本，**必须返回标准格式的字典**。

```python
def __getitem__(self, idx: int) -> dict:
    """
    获取单个样本
    
    Args:
        idx: 样本索引
    
    Returns:
        字典必须包含以下键：
        {
            'sar': torch.Tensor [C, H, W],      # SAR 图像
            'optical': torch.Tensor [C, H, W],  # 光学图像（训练时需要）
            'sar_path': str,                     # SAR 文件路径
            'optical_path': str                  # 光学文件路径
        }
    """
    # 获取样本路径
    sar_path, optical_path = self.samples[idx]
    
    # 加载并预处理图像
    sar_img = self._load_image(sar_path)
    optical_img = self._load_image(optical_path)
    
    # 转换为 tensor [C, H, W]
    sar_tensor = torch.from_numpy(sar_img).float().permute(2, 0, 1)
    optical_tensor = torch.from_numpy(optical_img).float().permute(2, 0, 1)
    
    return {
        'sar': sar_tensor,
        'optical': optical_tensor,
        'sar_path': str(sar_path),
        'optical_path': str(optical_path)
    }
```

**关键要求**：
- `sar`: SAR 图像 tensor，形状 `[C, H, W]`
- `optical`: 光学图像 tensor，形状 `[C, H, W]`
- 训练时 `optical` 必须有值，推理时可为空
- 像素值范围归一化到 `[0, 1]` 或配置指定的范围

---

#### `get_data_range(self) -> Tuple[float, float]`

返回数据的数值范围。

```python
def get_data_range(self) -> Tuple[float, float]:
    """
    返回数据集的数值范围
    
    Returns:
        (min, max) 元组
    """
    return self.data_range  # 基类自动从配置读取
```

**默认行为**：
- 基类 `__init__` 会自动读取 `config['data']['normalize']['input_range']`
- 如果未配置，默认返回 `(0.0, 1.0)`
- 通常不需要重写此方法

---

## 配置文件规范

每个数据集必须提供 `config.yaml` 作为配置模板：

```yaml
# v2/datasets/your_dataset/config.yaml

# ==================== YourDataset 数据集配置 ====================

# 数据集基础配置
data:
  type: "your_dataset_name"   # 必须：对应 @register_dataset 的名称

  # 数据路径
  root: "/path/to/dataset"      # 或 train_dir，根据你的数据集结构调整
  
  train_ratio: 0.9             # 90%训练，10%验证（如果没有预划分）
  
  normalize:
    enabled: true
    input_range: [0.0, 1.0]    # 数据归一化范围

  # 通道配置
  channels:
    optical:
      input: 3                  # 原始通道数
      use: 3                    # 使用通道数
      indices: [0, 1, 2]        # 使用的通道索引
    sar:
      input: 1
      use: 1
      indices: [0]

  # DataLoader 配置
  dataloader:
    batch_size: 4
    num_workers: 4
    pin_memory: true
    prefetch_factor: 2
    persistent_workers: true
```

---

## 完整示例：简单文件夹数据集

假设数据组织为：

```
data/
├── sar/
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── optical/
    ├── 001.png
    ├── 002.png
    └── ...
```

```python
# v2/datasets/simple_folder/dataset.py

import torch
import numpy as np
from pathlib import Path
from typing import Tuple
from PIL import Image
from ..base import BaseDataset
from ..registry import register_dataset


@register_dataset('simple_folder')
class SimpleFolderDataset(BaseDataset):
    """
    简单文件夹数据集
    
    适配结构:
        root/
        ├── sar/
        └── optical/
    
    配置示例:
        data:
          type: "simple_folder"
          root: "data/"
          train_ratio: 0.9
    """
    
    def __init__(self, config: dict, split: str = 'train'):
        super().__init__(config, split)
        
        # 读取配置
        data_cfg = config['data']
        self.root = Path(data_cfg['root'])
        
        # 扫描文件
        self.sar_files, self.optical_files = self._scan_files()
        
        # 划分训练/验证
        train_ratio = data_cfg.get('train_ratio', 0.9)
        n_train = int(len(self.sar_files) * train_ratio)
        
        if split == 'train':
            self.sar_files = self.sar_files[:n_train]
            self.optical_files = self.optical_files[:n_train]
        else:
            self.sar_files = self.sar_files[n_train:]
            self.optical_files = self.optical_files[n_train:]
        
        print(f"[{split}] Loaded {len(self.sar_files)} samples")
    
    def _scan_files(self) -> Tuple[list, list]:
        """扫描并配对 SAR 和光学图像"""
        sar_dir = self.root / 'sar'
        optical_dir = self.root / 'optical'
        
        # 获取所有图像文件
        sar_files = sorted(sar_dir.glob('*.png'))
        optical_files = sorted(optical_dir.glob('*.png'))
        
        # 确保数量匹配
        min_len = min(len(sar_files), len(optical_files))
        return sar_files[:min_len], optical_files[:min_len]
    
    def _load_image(self, path: Path) -> np.ndarray:
        """加载并预处理单张图像"""
        # 读取图像
        img = Image.open(path)
        img = np.array(img, dtype=np.float32)
        
        # 确保通道维度
        if img.ndim == 2:
            img = img[..., np.newaxis]  # [H, W] -> [H, W, C]
        
        # 归一化到 [0, 1]
        if img.max() > 1.0:
            img = img / 255.0
        
        return img
    
    def __len__(self) -> int:
        return len(self.sar_files)
    
    def __getitem__(self, idx: int) -> dict:
        # 加载图像
        sar = self._load_image(self.sar_files[idx])
        optical = self._load_image(self.optical_files[idx])
        
        # 转换为 tensor [C, H, W]
        sar_tensor = torch.from_numpy(sar).float().permute(2, 0, 1)
        optical_tensor = torch.from_numpy(optical).float().permute(2, 0, 1)
        
        return {
            'sar': sar_tensor,
            'optical': optical_tensor,
            'sar_path': str(self.sar_files[idx]),
            'optical_path': str(self.optical_files[idx])
        }
    
    def get_data_range(self) -> Tuple[float, float]:
        return self.data_range
```

---

## 完整示例：多季节数据集

假设数据按季节组织：

```
dataset/
├── spring/
│   ├── s1_000/          # SAR 样本
│   ├── s2_000/          # 光学样本（配对）
│   └── ...
├── summer/
│   └── ...
```

```python
# v2/datasets/seasonal/dataset.py

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List
from PIL import Image
from ..base import BaseDataset
from ..registry import register_dataset


@register_dataset('seasonal')
class SeasonalDataset(BaseDataset):
    """
    多季节数据集
    
    适配结构:
        root/
        ├── spring/
        │   ├── s1_000/ ...
        │   └── s2_000/ ...
        └── summer/
            ├── s1_000/ ...
            └── s2_000/ ...
    """
    
    def __init__(self, config: dict, split: str = 'train'):
        super().__init__(config, split)
        
        data_cfg = config['data']
        self.root = Path(data_cfg['root'])
        
        # 扫描所有季节的样本
        self.samples = self._scan_all_seasons()
        
        # 划分训练/验证
        train_ratio = data_cfg.get('train_ratio', 0.9)
        seed = data_cfg.get('seed', 42)
        
        # 随机打乱并划分
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self.samples))
        
        n_train = int(len(indices) * train_ratio)
        
        if split == 'train':
            self.samples = [self.samples[i] for i in indices[:n_train]]
        else:
            self.samples = [self.samples[i] for i in indices[n_train:]]
        
        print(f"[{split}] Loaded {len(self.samples)} samples")
    
    def _scan_all_seasons(self) -> List[Tuple[Path, Path]]:
        """扫描所有季节的配对样本"""
        all_samples = []
        
        # 遍历所有季节文件夹
        for season_dir in self.root.iterdir():
            if not season_dir.is_dir():
                continue
            
            # 配对 s1_XXX 和 s2_XXX 文件夹
            s1_dirs = {d.name.replace('s1_', ''): d 
                      for d in season_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('s1_')}
            
            s2_dirs = {d.name.replace('s2_', ''): d 
                      for d in season_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('s2_')}
            
            # 收集配对样本
            for idx in s1_dirs:
                if idx in s2_dirs:
                    # 获取文件夹中的图像文件
                    s1_files = list(s1_dirs[idx].glob('*.png'))
                    s2_files = list(s2_dirs[idx].glob('*.png'))
                    
                    # 配对相同名称的文件
                    s1_dict = {f.stem.replace('s1_', ''): f for f in s1_files}
                    s2_dict = {f.stem.replace('s2_', ''): f for f in s2_files}
                    
                    for name in s1_dict:
                        if name in s2_dict:
                            all_samples.append((s1_dict[name], s2_dict[name]))
        
        return all_samples
    
    def _load_image(self, path: Path) -> np.ndarray:
        """加载并预处理图像"""
        img = Image.open(path)
        img = np.array(img, dtype=np.float32)
        
        if img.ndim == 2:
            img = img[..., np.newaxis]
        
        if img.max() > 1.0:
            img = img / 255.0
        
        return img
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sar_path, optical_path = self.samples[idx]
        
        sar = self._load_image(sar_path)
        optical = self._load_image(optical_path)
        
        sar_tensor = torch.from_numpy(sar).float().permute(2, 0, 1)
        optical_tensor = torch.from_numpy(optical).float().permute(2, 0, 1)
        
        return {
            'sar': sar_tensor,
            'optical': optical_tensor,
            'sar_path': str(sar_path),
            'optical_path': str(optical_path)
        }
    
    def get_data_range(self) -> Tuple[float, float]:
        return self.data_range
```

---

## 关键实现细节

### 图像预处理规范

```python
def _load_image(self, path: Path, channels: List[int] = None) -> np.ndarray:
    """
    标准图像加载流程
    
    Returns:
        np.ndarray [H, W, C] 范围 [0, 1]
    """
    # 1. 读取图像
    img = Image.open(path)  # 或 tifffile.imread, cv2.imread 等
    img = np.array(img, dtype=np.float32)
    
    # 2. 确保通道维度
    if img.ndim == 2:
        img = img[..., np.newaxis]
    
    # 3. 处理通道顺序 (如果是 CHW 格式)
    if img.shape[0] < img.shape[-1] and img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    
    # 4. 选择通道
    if channels and img.shape[-1] > len(channels):
        img = img[..., channels]
    
    # 5. 归一化
    if img.max() > 1.0:
        if img.max() > 255:
            img = img / 65535.0  # uint16
        else:
            img = img / 255.0     # uint8
    
    return img
```

### 训练/验证集划分

```python
# 方案 1: 按比例划分
train_ratio = data_cfg.get('train_ratio', 0.9)
n_train = int(len(samples) * train_ratio)

if split == 'train':
    self.samples = samples[:n_train]
else:
    self.samples = samples[n_train:]

# 方案 2: 按预定义列表划分
train_list = data_cfg.get('train_list', [])  # 训练样本ID列表
val_list = data_cfg.get('val_list', [])      # 验证样本ID列表

if split == 'train':
    self.samples = [s for s in all_samples if s['id'] in train_list]
else:
    self.samples = [s for s in all_samples if s['id'] in val_list]
```

---

## 检查清单

添加新数据集后，确认以下事项：

- [ ] 文件放在 `v2/datasets/your_dataset/` 目录下
- [ ] 使用 `@register_dataset('name')` 装饰数据集类
- [ ] 继承 `BaseDataset`
- [ ] 实现 3 个抽象方法：`__len__`, `__getitem__`, `get_data_range`
- [ ] `__getitem__` 返回包含 `'sar'`, `'optical'`, `'sar_path'`, `'optical_path'` 的字典
- [ ] 图像 tensor 形状为 `[C, H, W]`
- [ ] 像素值归一化到 `[0, 1]` 或配置指定范围
- [ ] 提供 `config.yaml` 配置模板
- [ ] 在 `v2/datasets/your_dataset/__init__.py` 中导出数据集类
- [ ] 在 `v2/datasets/registry.py` 的 `import_datasets()` 中导入新数据集

---

## 常见问题

### Q: 数据已经分为 train/val 文件夹？

```python
def __init__(self, config: dict, split: str = 'train'):
    super().__init__(config, split)
    
    data_cfg = config['data']
    
    # 直接根据 split 选择文件夹
    if split == 'train':
        data_dir = Path(data_cfg['train_dir'])
    else:
        data_dir = Path(data_cfg['val_dir'])
    
    self.samples = self._scan_files(data_dir)
```

### Q: SAR 和光学图像文件名不同？

```python
def _scan_files(self, data_dir: Path):
    """按自定义规则配对"""
    sar_dir = data_dir / 'sar'
    optical_dir = data_dir / 'opt'
    
    sar_files = sorted(sar_dir.glob('*.tif'))
    optical_files = sorted(optical_dir.glob('*.tif'))
    
    # 自定义配对逻辑
    pairs = []
    for sar_file in sar_files:
        # 假设 SAR: SAR_001.tif -> Optical: OPT_001.tif
        opt_name = sar_file.name.replace('SAR', 'OPT')
        opt_file = optical_dir / opt_name
        
        if opt_file.exists():
            pairs.append((sar_file, opt_file))
    
    return pairs
```

### Q: 需要数据增强？

在 `__getitem__` 中添加：

```python
def __getitem__(self, idx: int) -> dict:
    # 加载数据
    sar = self._load_image(...)
    optical = self._load_image(...)
    
    # 数据增强 (仅在训练时)
    if self.split == 'train':
        # 随机水平翻转
        if np.random.rand() > 0.5:
            sar = np.fliplr(sar)
            optical = np.fliplr(optical)
        
        # 随机旋转
        k = np.random.randint(0, 4)
        sar = np.rot90(sar, k)
        optical = np.rot90(optical, k)
    
    # 转 tensor
    ...
```

### Q: 需要支持 GeoTIFF？

```python
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

@register_dataset('geotiff_dataset')
class GeoTIFFDataset(BaseDataset):
    def __init__(self, config: dict, split: str = 'train'):
        if not HAS_TIFFFILE:
            raise ImportError("tifffile is required. Install with: pip install tifffile")
        # ...
    
    def _load_image(self, path: Path) -> np.ndarray:
        """使用 tifffile 加载 GeoTIFF"""
        img = tifffile.imread(str(path))
        # 后续处理...
        return img
```

---

## 参考实现

- **WHU**: `v2/datasets/whu/dataset.py` - TIFF 格式，简单文件夹结构
- **SEN1-2**: `v2/datasets/sen12/dataset.py` - 多季节，复杂配对逻辑
