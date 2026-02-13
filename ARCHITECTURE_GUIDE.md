# SAR图像翻译项目架构指导文档

> 本文档记录项目的架构设计原则和特殊约定，用于指导后续开发和维护。

## 1. 项目核心理念

### 1.1 分层设计原则

```
┌─────────────────────────────────────────────────────────────┐
│  命令层 (commands/)                                          │
│  - 唯一职责: 解析参数、调用高层API、输出结果                    │
│  - 禁止包含业务逻辑                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  流程层 (core/inference_ops.py, training_ops.py,             │
│           core/validation_ops.py, core/visualization_ops.py) │
│  - 职责: 编排训练/推理/验证/可视化流程                         │
│  - 通过接口调用模型和数据集                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  接口层 (models/base.py, datasets/base.py)                   │
│  - 定义模型和数据集必须实现的接口                              │
│  - 实现通过配置文件动态加载                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  实现层 (models/srdm_interface.py, datasets/sar_optical_*)   │
│  - 具体模型和数据集实现                                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  元功能层 (core/numeric_ops.py, device_ops.py, ...)          │
│  - 万年不变的基础功能                                          │
│  - 被所有上层依赖                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 配置驱动设计

- **config_srdm.yaml 是唯一配置入口**
- 所有命令必须接收 `--config` 参数
- DEBUG模式强制进行数值检验，验证配置指定的数据范围和模型输出范围

### 1.3 可扩展性设计

**模型扩展**: 新增模型只需:
1. 继承 `BaseModelInterface`
2. 实现 `get_output()`, `get_output_range()` 方法
3. 在配置中指定 `model.name`

**数据集扩展**: 新增数据集只需:
1. 继承 `BaseDataset`
2. 实现 `__getitem__()`, `get_data_range()` 方法
3. 注册到 `DatasetRegistry`

---

## 2. 数值范围约定 (非常重要!)

### 2.1 标准数据流

```
输入数据 (SAR/Optical)
    ↓ [0, 1] 范围 (由Dataset保证)
Dataset输出
    ↓
模型输入 (SRDM: SAR [0,1], Residual训练目标 [-1,1])
    ↓
模型输出 (SRDM: 预测的Residual [-1,1])
    ↓
合成: SAR_base + Residual → 范围可能在 [-1, 2]
    ↓
负数截断: clamp(min=0) → 范围 [0, 2]
    ↓
归一化: / max_val → 范围 [0, 1]
    ↓
转uint8: * 255 → 范围 [0, 255]
    ↓
保存图像
```

### 2.2 数值转换函数使用规范

```python
from v2.core.numeric_ops import input_to_model, model_output_to_composite, composite_to_uint8

# 1. Dataset输出已经是[0,1], 无需转换

# 2. 模型输出处理 (SRDM为例)
residual_pred = model.get_residual(sar)  # [-1, 1]
sar_base = model.get_sar_base(sar)       # [0, 1]

# 3. 合成处理
composite = model_output_to_composite(
    model_output=residual_pred,
    base=sar_base,
    output_range=[0, 1]  # 目标输出范围
)  # 自动处理: base + output → 截断负数 → 归一化

# 4. 保存图像
image_uint8 = composite_to_uint8(composite, input_range=[0, 1])
```

### 2.3 配置中的数值范围声明

```yaml
# config_srdm.yaml
data:
  normalize:
    input_range: [0.0, 1.0]  # Dataset输出范围
    
model:
  output_range: [-1.0, 1.0]  # 模型输出范围 (SRDM预测残差)
  
output:
  composite_range: [0.0, 1.0]  # 合成后范围 (截断后归一化)
  save_format: "uint8"  # 最终保存格式
```

---

## 3. 关键接口定义

### 3.1 模型接口 (v2/models/base.py)

```python
class BaseModelInterface(ABC):
    """所有模型必须实现的接口"""
    
    @abstractmethod
    def get_output(self, sar: torch.Tensor, config: dict) -> ModelOutput:
        """
        获取模型输出
        
        Args:
            sar: [B, C, H, W] 输入SAR图像
            config: 配置字典
            
        Returns:
            ModelOutput: 包含以下字段:
                - generated: 生成的图像 [B, 3, H, W]
                - intermediate: 中间结果(可选)
                - output_range: 实际输出范围
        """
        pass
    
    @abstractmethod
    def get_output_range(self) -> Tuple[float, float]:
        """返回模型输出的数值范围"""
        pass
```

### 3.2 数据集接口 (v2/datasets/base.py)

```python
class BaseDataset(Dataset, ABC):
    """所有数据集必须实现的接口"""
    
    @abstractmethod
    def get_data_range(self) -> Tuple[float, float]:
        """返回Dataset输出的数值范围"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx) -> dict:
        """
        返回字典必须包含:
        - 'sar': SAR图像 tensor
        - 'optical': 光学图像 tensor (训练时)
        - 'sar_path': SAR文件路径
        - 'optical_path': 光学文件路径
        """
        pass
```

---

## 4. DEBUG模式数值检验

DEBUG命令必须强制进行数值检验:

```python
# v2/commands/debug.py
def execute(self, args):
    config = load_config(args.config)
    
    # 1. 检验Dataset输出范围
    dataset = load_dataset(config)
    validate_dataset_range(dataset, expected=config['data']['normalize']['input_range'])
    
    # 2. 检验模型输出范围
    model = load_model(config)
    validate_model_output_range(model, expected=config['model']['output_range'])
    
    # 3. 检验合成流程
    validate_composite_flow(model, dataset, config)
```

---

## 5. 新增功能开发指南

### 5.1 新增模型

```python
# v2/models/my_new_model.py
from .base import BaseModelInterface

class MyNewModelInterface(BaseModelInterface):
    def __init__(self, config):
        self.model = load_actual_model(config)
        self._output_range = (-1.0, 1.0)  # 声明输出范围
    
    def get_output(self, sar, config):
        # 实现推理逻辑
        generated = self.model(sar)
        return ModelOutput(
            generated=generated,
            output_range=self._output_range
        )
    
    def get_output_range(self):
        return self._output_range

# 在配置中使用
# model:
#   name: "my_new_model"
```

### 5.2 新增数据集

```python
# v2/datasets/my_new_dataset.py
from .base import BaseDataset

class MyNewDataset(BaseDataset):
    def __init__(self, config):
        self.data_range = (0.0, 1.0)
        # 加载数据
    
    def get_data_range(self):
        return self.data_range
    
    def __getitem__(self, idx):
        return {
            'sar': sar_tensor,
            'optical': optical_tensor,
            'sar_path': path,
            'optical_path': path
        }

# 注册到registry
# v2/datasets/registry.py
from .my_new_dataset import MyNewDataset
DATASET_REGISTRY['my_new_dataset'] = MyNewDataset
```

### 5.3 新增推理脚本

```python
# v2/commands/my_infer.py
from .base import BaseCommand
from v2.core.inference_ops import run_inference
from v2.core.numeric_ops import composite_to_uint8
from v2.utils.image_ops import save_image

class MyInferCommand(BaseCommand):
    def execute(self, args):
        config = load_config(args.config)
        
        # 使用封装好的推理流程
        for batch_output in run_inference(config):
            # batch_output.generated 已经在 [0, 1] 范围
            
            # 转换为uint8并保存
            for img in batch_output.generated:
                image_uint8 = composite_to_uint8(img, input_range=[0, 1])
                save_image(image_uint8, output_path)
```

---

## 6. 禁止事项

1. **禁止在命令层直接操作tensor的数值范围** - 必须使用 `numeric_ops` 中的函数
2. **禁止硬编码模型类型判断** - 使用 `model.name` 配置 + 接口多态
3. **禁止在Dataset外部再次归一化** - Dataset必须输出配置声明的范围
4. **禁止在命令层重复实现加载逻辑** - 使用 `checkpoint_ops`, `device_ops` 中的函数
5. **禁止直接访问model.module** - 使用 `get_raw_model()` 函数

---

## 7. 核心模块封装函数参考

### 7.1 training_ops.py - 训练流程封装

```python
# 初始化训练组件
setup_training(config, device, rank, world_size) -> TrainingContext

# 单 epoch 训练
run_training_epoch(ctx, epoch, num_epochs, use_accumulation, accumulation_steps, max_norm) -> float

# 保存检查点
handle_checkpoint_save(ctx, epoch, metrics, checkpoint_dir, config, save_periodic) -> bool

# 完整训练循环（含异常处理、可视化）
run_training_loop(ctx, config, start_epoch, checkpoint_dir, result_dir, training_state, log_dir, experiment_dir)
```

### 7.2 validation_ops.py - 验证与DDP测试封装

```python
# 运行验证并计算指标
run_validation(model, val_loader, config, device, save_results, save_dir, max_samples) -> Dict[str, float]

# 保存验证样本（已提取为独立函数）
save_validation_samples(sar, generated, optical, save_dir, sample_idx, batch_idx)

# DDP 组件测试（已合并原 ddp_validation.py）
run_all_ddp_validations(verbose=False) -> Tuple[bool, List[str]]
```

### 7.3 visualization_ops.py - 可视化封装

```python
# 创建报告目录
setup_report_directory(experiment_dir) -> Path

# 记录训练日志（实时追加）
log_loss(log_dir, epoch, loss, val_metrics=None)

# 绘制训练曲线（从日志读取）
plot_loss_curve(log_file, save_path, title) -> bool

# 创建推理对比图（SAR | Generated | Optical）
create_inference_comparison(sar, generated, optical) -> np.ndarray

# 生成对比报告（从已保存图片）
create_comparison_figure(sample_paths, save_path, title, samples_per_row=5) -> bool

# 生成验证报告（整合函数）
generate_validation_report(result_dir, report_dir, epoch, is_validation=True) -> bool
```

**可视化输出目录结构**:
```
experiments/{name}/
├── logs/training.log           # 实时记录 epoch, loss, psnr, ssim
└── report/
    ├── loss_curve.png          # 训练结束/中断时生成（4子图：loss/psnr/ssim/摘要）
    └── val_samples_epoch_XXXX.png  # 验证时生成（汇总报告）
```

---

## 8. 文件组织约定

```
v2/
├── ARCHITECTURE_GUIDE.md      # 本文件 - 架构指导
├── API_REFERENCE.md           # 详细API文档
├── config_srdm.yaml           # 唯一配置入口
├── core/                      # 元功能和流程封装
│   ├── __init__.py
│   ├── numeric_ops.py         # 数值转换(万年不变)
│   ├── device_ops.py          # 设备管理(万年不变)
│   ├── checkpoint_ops.py      # 检查点管理(万年不变)
│   ├── inference_ops.py       # 推理流程封装
│   ├── training_ops.py        # 训练流程封装
│   ├── validation_ops.py      # 验证流程封装
│   └── visualization_ops.py   # 可视化封装（新增）
├── models/                    # 模型接口和实现
│   ├── __init__.py
│   ├── base.py                # 模型接口基类
│   ├── registry.py            # 模型注册表
│   └── srdm_interface.py      # SRDM实现
├── datasets/                  # 数据集接口和实现
│   ├── __init__.py
│   ├── base.py                # 数据集接口基类
│   ├── registry.py            # 数据集注册表
│   └── sar_optical_dataset.py # SAR光学数据集实现
├── utils/                     # 工具函数
│   ├── __init__.py
│   └── image_ops.py           # 基础图像操作
├── commands/                  # 命令入口
│   ├── __init__.py
│   ├── base.py                # 命令基类
│   ├── train.py
│   ├── infer.py
│   ├── debug.py               # 强制数值检验
│   └── exp.py                 # 实验脚本命令（新增）
├── exp_script/                # 实验脚本系统（新增）
│   ├── __init__.py
│   ├── context.py             # ExpContext 沙盒上下文
│   ├── runner.py              # 脚本加载与执行器
│   ├── errors.py              # 错误定义
│   ├── README.md              # 使用文档
│   ├── example_analysis.py    # 示例脚本
│   └── logs/                  # 错误日志目录
├── tests/                     # 测试
│   └── test_numeric_ops.py
└── main.py                    # 入口
```

---

## 9. 实验脚本系统 (Experiment Script System)

### 9.1 系统定位

实验脚本系统提供了一个**沙盒化**的实验环境：
- **高自由度**：支持动态加载任意 `.py` 实验脚本
- **高容错**：单点错误不导致程序崩溃，完善的错误捕获和日志记录
- **沙盒隔离**：脚本通过 `ExpContext` 访问项目资源，禁止直接导入底层模块

### 9.2 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│  ExpCommand (commands/exp.py)                                │
│  - 解析参数、调用 ExperimentRunner                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  ExperimentRunner (exp_script/runner.py)                     │
│  - 发现脚本 → 验证规范 → 加载模块 → 安全执行                 │
│  - 捕获异常 → 记录日志 → 输出报告                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  ExpContext (exp_script/context.py)                          │
│  - 配置管理 API                                              │
│  - 设备管理 API                                              │
│  - 模型/数据集 API                                           │
│  - 数值转换 API                                              │
│  - 推理/训练/验证 API                                        │
│  - 可视化 API                                                │
│  - 安全执行 API (safe_run, safe_call)                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  核心模块 (core/, models/, datasets/, utils/)               │
│  - 被 ExpContext 封装和代理                                  │
└─────────────────────────────────────────────────────────────┘
```

### 9.3 脚本编写规范

**入口函数**：
```python
def run_experiment(ctx, **kwargs):
    """
    实验入口函数
    
    Args:
        ctx: ExpContext 实例，提供所有API访问
        **kwargs: 额外参数
        
    Returns:
        任意类型
    """
    pass
```

**禁止事项**：
1. 禁止直接导入底层模块（`from v2.core.numeric_ops import ...`）
2. 禁止直接访问 `model.module`
3. 禁止手动处理数值范围转换
4. 必须使用 `ctx.safe_run()` 包装高风险操作

### 9.4 ExpContext API 注册规范

**新增功能必须在 ExpContext 注册**：

如果在核心模块（如 `numeric_ops.py`、`device_ops.py` 等）中添加了新的封装函数，**必须**在 `ExpContext` 中注册对应的包装方法。

**注册步骤**：
1. 在 `v2/exp_script/context.py` 顶部导入新函数
2. 在 `ExpContext` 类中添加对应的包装方法
3. 按照功能分类放置（数值转换、设备管理等）
4. 编写 docstring 说明参数和返回值

**示例**：
```python
# 1. 导入新函数
from core.numeric_ops import new_normalize_function

class ExpContext:
    # ... 其他方法 ...
    
    # 2. 添加包装方法（放在数值转换 API 区域）
    def new_normalize(self, tensor, param):
        """
        新的归一化函数
        
        Args:
            tensor: 输入张量
            param: 参数
            
        Returns:
            归一化后的张量
        """
        return new_normalize_function(tensor, param)
```

**注册 checklist**：
- [ ] 新函数已添加到对应的 `core/` 模块
- [ ] 新函数已导入到 `exp_script/context.py`
- [ ] `ExpContext` 中已添加对应的包装方法
- [ ] 方法已按功能分类放置
- [ ] 方法包含完整的 docstring
- [ ] 方法已在 `README.md` 中记录
- [ ] 已通过 `python main.py exp --name example_analysis` 测试

---

## 10. 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v2.0 | 2026-02-05 | 初始架构设计 |
| v2.1 | 2026-02-13 | 新增可视化模块（visualization_ops.py），重构训练/验证流程，集成loss曲线和对比报告生成 |
| v2.2 | 2026-02-13 | 新增实验脚本系统（exp_script/），支持动态加载和高容错执行 |
