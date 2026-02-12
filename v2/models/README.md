# 模型开发规范 (Vibe Coding Guide)

> **目标**：让 AI 能快速理解和添加与 DDPM 结构相似的图像转换模型

## 快速开始 (TL;DR)

添加一个新模型只需 **3 步**：

```python
# 1. 创建模型文件夹 v2/models/your_model/
#    ├── interface.py    # 必须：实现接口
#    ├── config.yaml     # 必须：配置模板
#    └── ...             # 可选：模型实现文件

# 2. 继承 BaseModelInterface 并实现 4 个抽象方法

# 3. 使用 @register_model('your_model') 装饰器注册
```

---

## 项目架构概览

```
v2/models/
├── base.py              # 基类定义 (别碰)
├── registry.py          # 注册机制 (别碰)
├── README.md            # 本文档
├── diffusion/           # 共享的扩散组件
│   ├── schedule.py      # 噪声调度器
│   └── ...
└── your_model/          # 你的新模型在这里
    ├── __init__.py      # 导出接口类
    ├── interface.py     # 模型接口 (必须)
    ├── config.yaml      # 默认配置 (必须)
    └── model.py         # 模型实现 (按需)
```

---

## 必须实现的接口

### 1. 继承 `BaseModelInterface`

```python
from models.base import BaseModelInterface, ModelOutput, CompositeMethod
from models.registry import register_model

@register_model('your_model_name')  # 注册名称，用于配置文件中 model.type
class YourModelInterface(BaseModelInterface):
    """你的模型接口"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self._output_range = (-1.0, 1.0)  # 设置你的输出范围
        self._model = None
```

### 2. 实现 4 个抽象方法

#### `build_model(self, device='cpu') -> torch.nn.Module`

构建并返回实际的 PyTorch 模型。

```python
def build_model(self, device='cpu'):
    """构建模型"""
    # 从配置读取参数
    model_cfg = self.config.get('your_model', {})
    
    # 创建模型实例
    self._model = YourActualModel(
        base_ch=model_cfg.get('base_ch', 64),
        # ... 其他参数
    )
    
    self._model = self._model.to(device)
    self._device = torch.device(device)
    
    return self._model
```

**关键要求**：
- 必须将模型移到指定 `device`
- 必须保存到 `self._model`
- 返回模型实例

---

#### `get_output(self, sar: torch.Tensor, config: dict) -> ModelOutput`

**核心接口**：执行推理并返回标准化输出。

```python
def get_output(self, sar: torch.Tensor, config: dict) -> ModelOutput:
    """
    推理输出
    
    Args:
        sar: 输入 SAR 图像 [B, C, H, W]
        config: 完整配置字典
    
    Returns:
        ModelOutput: 包含生成结果和元数据
    """
    if self._model is None:
        raise RuntimeError("Model not built. Call build_model() first.")
    
    # 确保设备一致
    sar = sar.to(self._device)
    
    with torch.no_grad():
        # 你的推理逻辑
        generated = self._model(sar)
        
    return ModelOutput(
        generated=generated,           # [B, C, H, W] 生成的图像
        output_range=(0.0, 1.0),       # 实际输出数值范围
        intermediate={                 # 可选：中间结果供调试
            'features': some_features,
        },
        metadata={                     # 可选：推理参数等元数据
            'method': 'ddim',
            'steps': 50,
        }
    )
```

**关键要求**：
- 返回 `ModelOutput` 对象
- `generated` 必须是 `[B, C, H, W]` 的 tensor
- 支持批量推理

---

#### `get_output_range(self) -> Tuple[float, float]`

返回模型输出的数值范围。

```python
def get_output_range(self) -> Tuple[float, float]:
    """返回 (min, max) 元组"""
    return self._output_range  # 如 (-1.0, 1.0) 或 (0.0, 1.0)
```

**常见范围**：
- DDPM 预测噪声/残差：`(-1.0, 1.0)`
- 直接预测图像：`[0.0, 1.0]`
- 其他范围需在 `get_output` 中处理归一化

---

#### `get_composite_method(self) -> CompositeMethod`

返回合成方法（如何将模型输出合成为最终图像）。

```python
def get_composite_method(self) -> CompositeMethod:
    """返回合成方法枚举"""
    return CompositeMethod.ADD_THEN_CLAMP
```

**可选值**：

| 枚举值 | 含义 | 适用场景 |
|--------|------|----------|
| `CompositeMethod.DIRECT` | 直接输出，无需合成 | GAN、直接预测图像的模型 |
| `CompositeMethod.ADD` | 直接相加：`output + sar` | 预测残差后直接相加 |
| `CompositeMethod.ADD_THEN_CLAMP` | 相加、截断负数、归一化 | SRDM 风格：预测残差后合成并归一化 |
| `CompositeMethod.MULTIPLY` | 相乘：`output * sar` | 特殊情况 |

---

## 可选实现的方法

### 训练相关

```python
def forward(self, sar: torch.Tensor, optical: torch.Tensor = None, 
            return_dict: bool = False):
    """
    训练前向传播
    
    Args:
        sar: SAR 输入 [B, C, H, W]
        optical: 目标光学图像 [B, C, H, W]（训练时需要）
        return_dict: 是否返回详细损失字典
    
    Returns:
        loss 或 (loss, loss_dict)
    """
    return self._model(sar, optical, return_dict=return_dict)

def __call__(self, *args, **kwargs):
    """使对象可调用，代理到 forward"""
    return self.forward(*args, **kwargs)
```

### 调试支持

```python
def debug(self, device, verbose=False):
    """
    运行模型调试
    
    Args:
        device: 运行设备
        verbose: 是否显示详细信息
    
    Returns:
        ModelDebugReport: 调试报告
    """
    # 调用基类默认实现
    return super().debug(device, verbose)
```

### 自定义参数统计

```python
def count_parameters(self) -> Dict[str, int]:
    """统计参数量（可选按组件分类）"""
    if self._model is None:
        return {'total': 0}
    
    return {
        'total': sum(p.numel() for p in self._model.parameters()),
        'encoder': sum(p.numel() for p in self._model.encoder.parameters()),
        'decoder': sum(p.numel() for p in self._model.decoder.parameters()),
    }
```

---

## 配置文件规范

每个模型必须提供 `config.yaml` 作为配置模板：

```yaml
# v2/models/your_model/config.yaml

# ==================== YourModel 模型配置 ====================

# 模型基础配置
model:
  type: "your_model_name"    # 必须：对应 @register_model 的名称
  name: "YourModel"
  output_range: [-1.0, 1.0]  # 输出范围

# 模型特定配置
your_model:
  base_ch: 64
  ch_mults: [1, 2, 4, 8]
  num_blocks: 2
  time_emb_dim: 256
  dropout: 0.1

# 如果用到扩散
diffusion:
  num_timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  
  sampling:
    use_ddim: true
    ddim_steps: 50
```

---

## 完整示例：最小化模型

```python
# v2/models/simple_gan/interface.py

import torch
import torch.nn as nn
from typing import Tuple
from models.base import BaseModelInterface, ModelOutput, CompositeMethod
from models.registry import register_model


class SimpleGenerator(nn.Module):
    """简单的生成器示例"""
    
    def __init__(self, in_ch=1, out_ch=3, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_ch, 3, padding=1),
            nn.Sigmoid()  # 输出 [0, 1]
        )
    
    def forward(self, x):
        return self.net(x)


@register_model('simple_gan')
class SimpleGANInterface(BaseModelInterface):
    """简单 GAN 模型接口"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self._output_range = (0.0, 1.0)  # Sigmoid 输出
        self._model = None
    
    def build_model(self, device='cpu'):
        """构建模型"""
        cfg = self.config.get('simple_gan', {})
        
        self._model = SimpleGenerator(
            in_ch=cfg.get('in_ch', 1),
            out_ch=cfg.get('out_ch', 3),
            base_ch=cfg.get('base_ch', 64)
        ).to(device)
        
        self._device = torch.device(device)
        return self._model
    
    def get_output(self, sar: torch.Tensor, config: dict) -> ModelOutput:
        """推理"""
        if self._model is None:
            raise RuntimeError("Model not built")
        
        sar = sar.to(self._device)
        
        with torch.no_grad():
            generated = self._model(sar)
        
        return ModelOutput(
            generated=generated,
            output_range=(0.0, 1.0)
        )
    
    def get_output_range(self) -> Tuple[float, float]:
        return self._output_range
    
    def get_composite_method(self) -> CompositeMethod:
        # GAN 直接输出图像，无需合成
        return CompositeMethod.DIRECT
```

---

## 完整示例：DDPM 风格模型

```python
# v2/models/my_diffusion/interface.py

import torch
from typing import Tuple
from models.base import BaseModelInterface, ModelOutput, CompositeMethod
from models.registry import register_model
from models.diffusion.schedule import Schedule  # 可复用现有组件


@register_model('my_diffusion')
class MyDiffusionInterface(BaseModelInterface):
    """
    自定义扩散模型接口
    预测残差并合成
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self._output_range = (-1.0, 1.0)  # 残差范围
        self._model = None
    
    def build_model(self, device='cpu'):
        """构建模型"""
        # 读取配置
        model_cfg = self.config.get('my_diffusion', {})
        diff_cfg = self.config.get('diffusion', {})
        
        # 创建调度器
        schedule = Schedule(
            num_timesteps=diff_cfg.get('num_timesteps', 1000),
            beta_start=diff_cfg.get('beta_start', 0.0001),
            beta_end=diff_cfg.get('beta_end', 0.02),
            device=device
        )
        
        # 创建模型
        self._model = MyDiffusionModel(
            schedule=schedule,
            base_ch=model_cfg.get('base_ch', 64),
            # ...
        ).to(device)
        
        self._device = torch.device(device)
        return self._model
    
    def get_output(self, sar: torch.Tensor, config: dict) -> ModelOutput:
        """
        推理流程：
        1. 条件编码得到基础特征
        2. 扩散采样预测残差
        3. 合成最终图像
        """
        if self._model is None:
            raise RuntimeError("Model not built")
        
        sar = sar.to(self._device)
        
        with torch.no_grad():
            # 条件编码
            condition = self._model.encode_condition(sar)
            
            # 扩散采样
            sampling_cfg = config.get('diffusion', {}).get('sampling', {})
            residual = self._model.sample(
                condition=condition,
                steps=sampling_cfg.get('steps', 1000),
                return_residual_only=True
            )
            
            # 合成
            base = condition['base']
            generated = base + residual
            generated = torch.clamp(generated, min=0.0)
            # 归一化到 [0, 1]
            generated = (generated - generated.min()) / (generated.max() - generated.min() + 1e-8)
        
        return ModelOutput(
            generated=generated,
            output_range=(0.0, 1.0),
            intermediate={'residual': residual, 'base': base},
            metadata={'steps': sampling_cfg.get('steps', 1000)}
        )
    
    def get_output_range(self) -> Tuple[float, float]:
        return self._output_range
    
    def get_composite_method(self) -> CompositeMethod:
        # 虽然我们在 get_output 中处理了合成，但这里仍标记方法类型
        return CompositeMethod.ADD_THEN_CLAMP
```

---

## 检查清单

添加新模型后，确认以下事项：

- [ ] 文件放在 `v2/models/your_model/` 目录下
- [ ] 使用 `@register_model('name')` 装饰接口类
- [ ] 继承 `BaseModelInterface`
- [ ] 实现 4 个抽象方法
- [ ] 提供 `config.yaml` 配置模板
- [ ] 在 `v2/models/your_model/__init__.py` 中导出接口类
- [ ] 模型能在 `main.py` 中被正确加载

---

## 常见问题

### Q: 模型需要额外的损失函数？

在模型内部处理。`BaseModelInterface` 不限制损失实现方式，只需确保 `forward()` 返回 loss。

### Q: 需要特殊的优化器配置？

在训练命令中处理，模型接口只负责模型本身。

### Q: 模型需要多阶段训练？

在 `build_model()` 中初始化所有组件，在 `forward()` 中根据配置决定训练哪个部分。

### Q: 需要加载预训练权重？

在 `build_model()` 最后添加权重加载逻辑：

```python
def build_model(self, device='cpu'):
    # ... 创建模型
    
    # 加载预训练权重（如果配置中有）
    pretrained_path = self.config.get('model', {}).get('pretrained')
    if pretrained_path and Path(pretrained_path).exists():
        state_dict = torch.load(pretrained_path, map_location=device)
        self._model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    return self._model
```

---

## 参考实现

- **SRDM**: `v2/models/srdm/interface.py` - 完整的残差扩散模型示例
