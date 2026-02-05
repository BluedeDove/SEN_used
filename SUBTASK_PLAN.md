# v2项目子任务分配计划

## 任务分配说明

每个子任务独立分配给SubAgent执行，包含完整的上下文和接口规范。

---

## 阶段2: 万年不变的基础元功能

### 子任务2.1: v2/core/numeric_ops.py

**文件路径**: `v2/core/numeric_ops.py`

**功能**: 所有数值范围转换的基础函数，这是整个项目最核心的约定

**必须实现的函数**:

```python
def validate_range(tensor, expected_range, name="tensor"):
    """验证张量是否在期望范围内
    
    Args:
        tensor: 输入张量
        expected_range: (min, max) 期望范围
        name: 张量名称(用于错误信息)
    
    Returns:
        bool: 是否在范围内
    """
    pass

def model_output_to_composite(model_output, base, output_range, clamp_negative=True, normalize=True):
    """将模型输出与base合成为最终图像
    
    SRDM示例: 
        model_output = residual [-1, 1]
        base = sar_base [0, 1]
        composite = sar_base + residual → 范围[-1, 2]
        clamp_negative=True → 截断负数 → [0, 2]
        normalize=True → /2 → [0, 1]
    
    Args:
        model_output: 模型输出张量
        base: 基础图像(如sar_base)
        output_range: 目标输出范围 (min, max)
        clamp_negative: 是否截断负数
        normalize: 是否归一化到output_range
    
    Returns:
        composite: 合成后的图像
    """
    pass

def composite_to_uint8(composite, input_range=(0.0, 1.0)):
    """将合成图像转换为uint8格式用于保存
    
    Args:
        composite: [0, 1]范围的图像张量
        input_range: 输入数值范围
    
    Returns:
        uint8图像 numpy array
    """
    pass

def denormalize(tensor, from_range, to_range=(0.0, 1.0)):
    """将张量从一个范围映射到另一个范围
    
    Args:
        tensor: 输入张量
        from_range: (min, max) 当前范围
        to_range: (min, max) 目标范围
    """
    pass
```

**接口约定**:
- 输入输出必须是torch.Tensor或numpy.ndarray
- 函数必须是纯函数，无副作用
- 必须支持batch维度 [B, C, H, W]

**测试要求**:
- 测试SRDM场景: residual[-1,1] + sar_base[0,1] → 合成[0,1]
- 测试数值范围验证功能
- 测试uint8转换

---

### 子任务2.2: v2/core/device_ops.py

**文件路径**: `v2/core/device_ops.py`

**功能**: 设备管理和分布式训练支持

**必须实现的函数**:

```python
def setup_device_and_distributed(config, rank=None, world_size=None):
    """初始化设备和分布式环境
    
    Args:
        config: 配置字典
        rank: 当前进程rank(可选，从环境变量读取)
        world_size: 总进程数(可选，从环境变量读取)
    
    Returns:
        device: torch.device
        rank: 当前rank
        world_size: 总进程数
    """
    pass

def get_raw_model(model):
    """获取原始模型(自动处理DDP包装)
    
    Args:
        model: 可能是DDP包装的模型
    
    Returns:
        原始模型
    """
    pass

def is_main_process(rank=None):
    """检查是否是主进程
    
    Args:
        rank: 当前rank(可选)
    
    Returns:
        bool: 是否是主进程(rank 0)
    """
    pass

def cleanup_resources():
    """清理资源(垃圾回收、显存清空等)"""
    pass

def synchronize():
    """分布式同步屏障"""
    pass

def set_seed(seed):
    """设置随机种子"""
    pass
```

**依赖**:
- 原项目的 `core/distributed.py` 作为参考
- 保持与原项目兼容的DDP设置

---

### 子任务2.3: v2/core/checkpoint_ops.py

**文件路径**: `v2/core/checkpoint_ops.py`

**功能**: 检查点保存和加载

**必须实现的函数**:

```python
def save_checkpoint_v2(model, optimizer, scheduler, epoch, metrics, save_path):
    """保存检查点
    
    Args:
        model: 模型(可能是DDP包装)
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        metrics: 指标字典
        save_path: 保存路径
    """
    pass

def load_checkpoint_v2(checkpoint_path, device='cpu'):
    """加载检查点
    
    Args:
        checkpoint_path: 检查点路径
        device: 加载设备
    
    Returns:
        checkpoint: 包含state_dict等的字典
    """
    pass

def restore_model_v2(model, state_dict, strict=True):
    """恢复模型状态(自动处理DDP)
    
    Args:
        model: 目标模型
        state_dict: 状态字典
        strict: 是否严格匹配
    """
    pass

def restore_optimizer_v2(optimizer, state_dict):
    """恢复优化器状态"""
    pass
```

**接口约定**:
- 使用 `get_raw_model()` 获取原始模型状态
- 支持DDP和非DDP两种模式

---

### 子任务2.4: v2/utils/image_ops.py

**文件路径**: `v2/utils/image_ops.py`

**功能**: 基础图像操作

**必须实现的函数**:

```python
def tensor_to_numpy(tensor, channel_order='hwc'):
    """将tensor转换为numpy array
    
    Args:
        tensor: [B, C, H, W] 或 [C, H, W]
        channel_order: 'hwc' 或 'chw'
    
    Returns:
        numpy array
    """
    pass

def numpy_to_tensor(array, channel_order='chw'):
    """将numpy array转换为tensor
    
    Args:
        array: numpy array
        channel_order: 目标channel顺序
    
    Returns:
        torch.Tensor
    """
    pass

def create_comparison_figure(images_dict, titles=None, layout='horizontal'):
    """创建对比图
    
    Args:
        images_dict: {'SAR': sar_img, 'Generated': gen_img, 'GT': gt_img}
        titles: 标题列表
        layout: 'horizontal' 或 'vertical' 或 'grid'
    
    Returns:
        合成后的numpy图像
    """
    pass

def save_image_v2(image_array, save_path, create_dir=True):
    """保存图像
    
    Args:
        image_array: numpy array (H, W, C) 或 (H, W)
        save_path: 保存路径
        create_dir: 是否自动创建目录
    """
    pass
```

**依赖**:
- OpenCV (cv2)
- numpy

---

## 阶段3: 可扩展接口层

### 子任务3.1: v2/models/base.py

**文件路径**: `v2/models/base.py`

**功能**: 定义所有模型必须实现的接口

**必须实现的类**:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import torch

@dataclass
class ModelOutput:
    """模型输出的标准格式"""
    generated: torch.Tensor  # 生成的图像 [B, C, H, W]
    output_range: Tuple[float, float]  # 实际输出范围
    intermediate: Optional[Dict[str, torch.Tensor]] = None  # 中间结果
    metadata: Optional[Dict[str, Any]] = None  # 额外元数据


class BaseModelInterface(ABC):
    """所有模型必须实现的接口基类"""
    
    def __init__(self, config: dict):
        self.config = config
        self._model = None
        self._output_range = self._get_default_output_range()
    
    @abstractmethod
    def build_model(self, device='cpu') -> torch.nn.Module:
        """构建并返回实际的PyTorch模型"""
        pass
    
    @abstractmethod
    def get_output(self, sar: torch.Tensor, config: dict) -> ModelOutput:
        """
        获取模型输出 - 这是核心接口
        
        Args:
            sar: 输入SAR图像 [B, C, H, W]
            config: 配置字典
        
        Returns:
            ModelOutput: 标准化的模型输出
        """
        pass
    
    @abstractmethod
    def get_output_range(self) -> Tuple[float, float]:
        """返回模型输出的数值范围"""
        return self._output_range
    
    @abstractmethod
    def get_composite_method(self) -> str:
        """
        返回合成方法名称
        
        Returns:
            str: 如 "add", "add_then_clamp", "direct" 等
        """
        pass
    
    def _get_default_output_range(self) -> Tuple[float, float]:
        """获取默认输出范围(从配置)"""
        return tuple(self.config.get('model', {}).get('output_range', [-1.0, 1.0]))
```

**设计说明**:
- 使用dataclass定义标准输出格式
- 抽象方法强制子类实现核心逻辑
- 提供默认配置读取方法

---

### 子任务3.2: v2/models/srdm_interface.py

**文件路径**: `v2/models/srdm_interface.py`

**功能**: SRDM模型的接口实现

**参考代码** (原项目):
- `models/srdm/srdm_diffusion.py`
- `models/builder.py`

**必须实现的类**:

```python
from .base import BaseModelInterface, ModelOutput

class SRDMInterface(BaseModelInterface):
    """SRDM模型接口实现"""
    
    def build_model(self, device='cpu'):
        """构建SRDM模型"""
        # 从配置读取参数
        srdm_cfg = self.config.get('srdm', {})
        # 构建SRDMDiffusion模型
        # 返回模型
        pass
    
    def get_output(self, sar, config):
        """
        SRDM推理流程:
        1. SAR编码 → sar_base + sar_features
        2. 扩散采样 → pred_residual [-1, 1]
        3. 合成 → sar_base + pred_residual
        
        注意: 这里返回的是合成前的residual，
              真正的合成由numeric_ops处理
        """
        model = self._model
        
        # 获取采样配置
        sampling = config.get('diffusion', {}).get('sampling', {})
        method = sampling.get('method', 'ddim')
        steps = sampling.get('steps', 50)
        
        with torch.no_grad():
            # SAR编码
            sar_base, sar_features, _ = model.sar_encoder(sar)
            
            # 采样得到残差
            if method == 'ddim':
                residual = model.ddim_sample(sar, steps=steps, return_residual_only=True)
            else:
                residual = model.sample(sar, steps=steps, return_residual_only=True)
            
            return ModelOutput(
                generated=residual,  # 这是residual不是最终图像
                output_range=self.get_output_range(),
                intermediate={
                    'sar_base': sar_base,
                    'sar_features': sar_features
                },
                metadata={'method': method, 'steps': steps}
            )
    
    def get_composite_method(self):
        """SRDM使用 add_then_clamp 方法"""
        return "add_then_clamp"
```

**注意**:
- 需要修改原SRDMDiffusion模型，支持 `return_residual_only` 参数
- 或者在这里手动计算residual

---

### 子任务3.3: v2/datasets/base.py

**文件路径**: `v2/datasets/base.py`

**功能**: 数据集接口基类

**必须实现的类**:

```python
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Tuple

class BaseDataset(Dataset, ABC):
    """所有数据集必须实现的接口基类"""
    
    def __init__(self, config: dict, split: str = 'train'):
        """
        Args:
            config: 配置字典
            split: 'train' 或 'val'
        """
        self.config = config
        self.split = split
        self.data_range = self._get_data_range()
    
    @abstractmethod
    def __len__(self) -> int:
        """返回数据集大小"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx) -> dict:
        """
        返回字典必须包含:
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
        """返回数据集的数值范围"""
        return self.data_range
    
    def _get_data_range(self) -> Tuple[float, float]:
        """从配置读取数据范围"""
        normalize_cfg = self.config.get('data', {}).get('normalize', {})
        if normalize_cfg.get('enabled', True):
            return tuple(normalize_cfg.get('input_range', [0.0, 1.0]))
        return (0.0, 255.0)  # 未归一化时的默认范围
```

---

### 子任务3.4: v2/datasets/sar_optical_dataset.py

**文件路径**: `v2/datasets/sar_optical_dataset.py`

**功能**: SAR-Optical数据集实现

**参考代码**: `data_split/dataset.py`, `dataloader/SAR_Optical_dataset.py`

**必须实现的类**:

```python
from .base import BaseDataset
import tifffile
import numpy as np
import torch
from pathlib import Path

class SAROpticalDataset(BaseDataset):
    """SAR到光学图像翻译数据集"""
    
    def __init__(self, config, split='train'):
        super().__init__(config, split)
        
        # 从配置读取
        data_cfg = config['data']
        train_dir = data_cfg['train_dir']
        
        # 扫描文件
        sar_files, optical_files = self._scan_files(train_dir)
        
        # 划分训练/验证集
        if split == 'train':
            self.sar_files = sar_files[:int(len(sar_files) * data_cfg['train_ratio'])]
            self.optical_files = optical_files[:int(len(optical_files) * data_cfg['train_ratio'])]
        else:
            self.sar_files = sar_files[int(len(sar_files) * data_cfg['train_ratio']):]
            self.optical_files = optical_files[int(len(optical_files) * data_cfg['train_ratio']):]
        
        # 通道配置
        self.optical_channels = data_cfg.get('channels', {}).get('optical', {}).get('indices', [0, 1, 2])
        self.sar_channels = data_cfg.get('channels', {}).get('sar', {}).get('indices', [0])
    
    def _scan_files(self, train_dir):
        """扫描数据目录"""
        sar_dir = Path(train_dir) / 'sar'
        optical_dir = Path(train_dir) / 'opt'
        
        sar_files = sorted(sar_dir.glob('*.tif'))
        optical_files = sorted(optical_dir.glob('*.tif'))
        
        return sar_files, optical_files
    
    def _load_and_preprocess(self, path, channels):
        """加载并预处理单张图像"""
        img = tifffile.imread(path)
        
        # 确保是3D (H, W, C)
        if img.ndim == 2:
            img = img[..., np.newaxis]
        
        # 选择通道
        if img.shape[-1] > len(channels):
            img = img[..., channels]
        
        # 归一化到 [0, 1]
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0  # 如果是uint8输入
        
        return img
    
    def __len__(self):
        return len(self.sar_files)
    
    def __getitem__(self, idx):
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
```

---

### 子任务3.5: v2/datasets/registry.py

**文件路径**: `v2/datasets/registry.py`

**功能**: 数据集注册表，支持通过配置字符串动态加载

```python
from typing import Type, Dict
from .base import BaseDataset
from .sar_optical_dataset import SAROpticalDataset

# 数据集注册表
DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {
    'sar_optical': SAROpticalDataset,
    # 未来可以在这里添加更多数据集
    # 'my_custom_dataset': MyCustomDataset,
}

def register_dataset(name: str, dataset_class: Type[BaseDataset]):
    """注册新的数据集"""
    DATASET_REGISTRY[name] = dataset_class

def get_dataset_class(name: str) -> Type[BaseDataset]:
    """根据名称获取数据集类"""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]

def create_dataset(config: dict, split: str = 'train') -> BaseDataset:
    """工厂函数：根据配置创建数据集"""
    dataset_type = config.get('data', {}).get('type', 'sar_optical')
    dataset_class = get_dataset_class(dataset_type)
    return dataset_class(config, split)

def list_available_datasets():
    """列出所有可用的数据集"""
    return list(DATASET_REGISTRY.keys())
```

---

## 阶段4: 高阶功能封装

### 子任务4.1: v2/core/inference_ops.py

**文件路径**: `v2/core/inference_ops.py`

**功能**: 推理流程封装

```python
def run_inference(config, checkpoint_path=None, max_samples=None):
    """
    运行推理并返回生成器
    
    Args:
        config: 配置字典
        checkpoint_path: 检查点路径(可选，如果为None使用随机初始化)
        max_samples: 最大样本数(可选)
    
    Yields:
        InferenceOutput: 包含generated, sar, optical(真值)等
    """
    pass

class InferenceOutput:
    """推理输出结构"""
    def __init__(self, generated, sar, optical=None, metadata=None):
        self.generated = generated  # [0, 1]范围的生成图像
        self.sar = sar
        self.optical = optical  # 真值(如果有)
        self.metadata = metadata
```

**流程**:
1. 加载数据集
2. 加载模型
3. 遍历数据集
4. 对每个样本:
   - 调用model_interface.get_output()
   - 调用numeric_ops.model_output_to_composite()
   - 返回InferenceOutput

---

### 子任务4.2: v2/core/training_ops.py

**文件路径**: `v2/core/training_ops.py`

**功能**: 训练流程封装

```python
def setup_training(config, device):
    """设置训练所需的组件"""
    # 创建数据集
    # 创建模型
    # 创建优化器、调度器
    # 返回training_context
    pass

def run_training_epoch(model, dataloader, optimizer, scheduler, scaler, config, epoch):
    """运行一个训练epoch"""
    pass

def handle_gradient_accumulation(loss, optimizer, scaler, accumulation_steps, is_accumulation_step):
    """处理梯度累积"""
    pass
```

---

### 子任务4.3: v2/core/validation_ops.py

**文件路径**: `v2/core/validation_ops.py`

**功能**: 验证流程封装

```python
def run_validation(model, dataloader, config, device, save_results=False):
    """运行验证"""
    pass

def compute_metrics_batch(generated, optical):
    """计算批次指标(PSNR, SSIM)"""
    pass
```

---

## 执行顺序和依赖关系

```
阶段2 (基础元功能) - 可并行:
  ├─ 2.1 numeric_ops (最优先，被其他模块依赖)
  ├─ 2.2 device_ops
  ├─ 2.3 checkpoint_ops
  └─ 2.4 image_ops

阶段3 (接口层) - 依赖阶段2:
  ├─ 3.1 models/base
  ├─ 3.2 models/srdm_interface (依赖3.1)
  ├─ 3.3 datasets/base
  ├─ 3.4 datasets/sar_optical_dataset (依赖3.3)
  └─ 3.5 datasets/registry (依赖3.3, 3.4)

阶段4 (高阶功能) - 依赖阶段2,3:
  ├─ 4.1 inference_ops (依赖3.1, 3.2, 3.5)
  ├─ 4.2 training_ops
  └─ 4.3 validation_ops
```

---

## 全局注意事项

1. **所有模块必须导入v2的其他模块时使用相对导入**
2. **所有数值范围转换必须通过numeric_ops**
3. **所有DDP处理必须通过device_ops**
4. **所有检查点操作必须通过checkpoint_ops**
5. **禁止在命令层直接操作模型内部**
