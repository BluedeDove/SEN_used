# CLAUDE.md - 项目执行规则

> 本文档由 Claude Code 维护，用于指导后续开发任务的执行。

## 项目背景

本项目是 SAR（合成孔径雷达）到光学图像翻译的深度学习项目v2版本。
- **精神源头**: `E:\Coding\SRDM\workspace_backup` - 一个完整的 SRDM（SAR-Residual Diffusion Model）实现
- **升级目标**: 基于分层架构设计，实现可扩展、可维护的现代化代码库
- **核心技术**: PyTorch, DDIM/DDPM扩散模型, 残差学习

## 架构核心原则

### 1. 分层设计（严格分层）

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
│  - visualization_ops: loss曲线、对比报告生成（训练结束/中断时） │
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

### 2. 数值范围约定（极其重要）

```
输入数据 → [0, 1] 范围 (Dataset保证)
    ↓
模型输入 → SRDM: SAR [0,1], Residual训练目标 [-1,1]
    ↓
模型输出 → SRDM: 预测Residual [-1,1]
    ↓
合成处理 → SAR_base + Residual → 截断负数 → 归一化 → [0, 1]
    ↓
保存图像 → uint8 [0, 255]
```

**数值转换函数必须通过 `core/numeric_ops.py`**

### 3. 配置驱动设计

- `config_srdm.yaml` 是唯一配置入口
- 所有命令必须接收 `--config` 参数
- DEBUG模式强制进行数值检验

### DDP代码验证（单卡可测）

在单卡环境下使用 MockDDP 验证 DDP 相关逻辑：

```bash
# 只测试DDP代码逻辑（无需多卡，无需配置文件）
cd v2 && python -m commands.debug --test-ddp --verbose

# 测试内容包括：
# 1. get_raw_model - 正确解包DDP获取原始模型
# 2. checkpoint_save_load - DDP模型检查点保存/加载正确
# 3. is_main_process - 主进程检测逻辑正确
# 4. all_reduce - 分布式聚合操作（单进程下返回原值）
# 5. ddp_integration - 完整训练-保存-恢复流程
```

**关键验证点**：
- 检查点保存的状态字典不包含 `module.` 前缀
- DDP包装和原始模型保存的状态一致
- 恢复后的模型可以正常继续训练

实现位置：`core/validation_ops.py`（已合并原 ddp_validation.py 内容）

### 可视化功能封装（新增）

训练/推理的可视化功能已封装到 `core/visualization_ops.py`：

```python
# 训练过程可视化
log_loss(log_dir, epoch, loss, val_metrics)          # 实时记录训练日志
plot_loss_curve(log_file, save_path, title)          # 绘制loss/psnr/ssim曲线
generate_validation_report(result_dir, report_dir, epoch, is_validation)  # 生成验证报告

# 推理可视化
create_inference_comparison(sar, generated, optical)   # 创建单张对比图（SAR|Gen|Opt）
create_comparison_figure(sample_paths, save_path, title, samples_per_row)  # 汇总报告
```

**使用位置**：
- `training_ops.run_training_loop()` - 训练结束/中断时自动调用 `plot_loss_curve()`
- `training_ops.run_training_loop()` - 验证后自动调用 `generate_validation_report()`
- `commands.infer.py` - 推理后生成对比报告

**禁止在命令层直接操作 matplotlib，必须通过 visualization_ops 封装函数**

## 开发规则

### 代码编写规则

1. **导入规则**
   - 导入v2的其他模块时使用相对导入
   - 外部库导入在前，项目内部导入在后

2. **数值处理规则**
   - 禁止在命令层直接操作tensor的数值范围
   - 必须使用 `numeric_ops` 中的函数
   - Dataset必须输出配置声明的范围

3. **模型处理规则**
   - 禁止直接访问 `model.module`
   - 使用 `device_ops.get_raw_model()` 函数
   - 禁止硬编码模型类型判断

4. **DDP处理规则**
   - 所有DDP处理必须通过 `device_ops`
   - 检查点保存/加载使用 `checkpoint_ops`

5. **接口实现规则**
   - 新增模型必须继承 `BaseModelInterface`
   - 新增数据集必须继承 `BaseDataset`
   - 必须实现所有抽象方法

### 文件组织规则

```
v2/
├── CLAUDE.md                  # 本文件 - 执行规则
├── ARCHITECTURE_GUIDE.md      # 架构指导文档
├── SUBTASK_PLAN.md           # 子任务计划
├── config_srdm.yaml          # 唯一配置入口
├── core/                     # 元功能和流程封装
│   ├── __init__.py
│   ├── numeric_ops.py        # 数值转换(万年不变)
│   ├── device_ops.py         # 设备管理(万年不变)
│   ├── checkpoint_ops.py     # 检查点管理(万年不变)
│   ├── ddp_validation.py     # DDP代码验证(无需多卡)
│   ├── inference_ops.py      # 推理流程封装
│   ├── training_ops.py       # 训练流程封装（含训练循环、epoch训练、检查点保存）
│   ├── validation_ops.py     # 验证流程封装（含DDP测试、指标计算）
│   └── visualization_ops.py  # 可视化封装（loss曲线、对比报告）
├── models/                   # 模型接口和实现
│   ├── __init__.py
│   ├── base.py               # 模型接口基类
│   ├── registry.py           # 模型注册表
│   └── srdm/                 # SRDM实现目录
│       ├── __init__.py
│       ├── interface.py      # SRDM接口实现
│       ├── diffusion.py      # SRDM核心模型
│       ├── encoder.py        # SAR编码器
│       ├── unet.py           # 条件UNet
│       ├── blocks.py         # NAFBlock基础块
│       ├── attention.py      # HC-Attention
│       └── losses.py         # 多损失组合
├── datasets/                 # 数据集接口和实现
│   ├── __init__.py
│   ├── base.py               # 数据集接口基类
│   ├── registry.py           # 数据集注册表
│   ├── sar_optical_dataset.py # WHU SAR-光学数据集
│   └── sen12_dataset.py      # SEN1-2 数据集
├── utils/                    # 工具函数
│   ├── __init__.py
│   └── image_ops.py          # 基础图像操作
├── commands/                 # 命令入口
│   ├── __init__.py
│   ├── base.py               # 命令基类
│   ├── train.py
│   ├── infer.py
│   └── debug.py              # 强制数值检验
├── engine/                   # 训练引擎
│   ├── __init__.py
│   ├── trainer.py            # 训练循环
│   ├── step.py               # 单步训练
│   └── validator.py          # 验证逻辑
└── main.py                   # 入口
```

### 命名规范

1. **文件命名**: 小写下划线，如 `numeric_ops.py`
2. **类命名**: 驼峰式，如 `BaseModelInterface`
3. **函数命名**: 小写下划线，如 `get_output_range()`
4. **常量命名**: 大写下划线，如 `DEFAULT_OUTPUT_RANGE`

## 实施顺序

### 阶段2: 基础元功能（可并行）

1. `core/numeric_ops.py` - 数值范围转换函数
2. `core/device_ops.py` - 设备管理和分布式训练
3. `core/checkpoint_ops.py` - 检查点保存和加载
4. `utils/image_ops.py` - 基础图像操作

### 阶段3: 可扩展接口层（依赖阶段2）

1. `models/base.py` - 模型接口基类
2. `models/srdm/` - SRDM模型实现
3. `datasets/base.py` - 数据集接口基类
4. `datasets/sar_optical_dataset.py` - SAR光学数据集
5. `datasets/registry.py` - 数据集注册表

### 阶段4: 高阶功能封装（依赖阶段2,3）

1. `core/inference_ops.py` - 推理流程封装
2. `core/training_ops.py` - 训练流程封装
3. `core/validation_ops.py` - 验证流程封装

### 阶段5: 命令层和引擎（依赖阶段2,3,4）

1. `commands/base.py` - 命令基类
2. `commands/train.py` - 训练命令
3. `commands/infer.py` - 推理命令
4. `commands/debug.py` - 调试命令
5. `engine/trainer.py` - 训练引擎
6. `main.py` - 入口文件

## 关键参考文件

### SRDM 核心实现（workspace_backup）

- `models/srdm/srdm_diffusion.py` - SRDM主类
- `models/srdm/encoder.py` - SAR编码器
- `models/srdm/unet.py` - 条件UNet
- `models/srdm/blocks.py` - NAFBlock
- `models/srdm/attention.py` - HC-Attention
- `models/srdm/losses.py` - 损失函数

### 训练系统（workspace_backup）

- `engine/trainer.py` - 训练循环
- `engine/step.py` - 单步训练
- `engine/validator.py` - 验证逻辑

### 数值调试（workspace_backup）

- `utils/numerical_debug.py` - 数值调试工具

## 质量检查清单

实现每个模块后，检查：

- [ ] 是否符合接口规范
- [ ] 是否正确处理数值范围
- [ ] 是否支持DDP模式
- [ ] 是否有适当的错误处理
- [ ] 是否有文档字符串
- [ ] 是否符合命名规范

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v2.0 | 2026-02-05 | 初始架构设计 |
| v2.1 | 2026-02-13 | 新增可视化模块 visualization_ops.py，集成 loss 曲线和对比报告生成，重构训练流程封装 |

---

## 今后新增功能规范

### 新增功能前必读

1. **理解分层架构**：
   - 命令层只做参数解析，业务逻辑下放到流程层
   - 模型/数据集通过接口抽象，支持配置驱动
   - 数值转换必须通过 `numeric_ops`

2. **DEBUG模式优先**：
   - 任何改动后必须先通过 `python main.py debug --verbose`
   - 确保数值范围、数据流、合成流程正确
   - DEBUG通过是提交代码的必要条件

3. **向后兼容性**：
   - 配置文件格式变更需更新 `config_srdm.yaml` 模板
   - 接口变更需在 CLAUDE.md 中记录
   - 破坏性变更需升级版本号并写迁移指南

### 新增功能审查清单

#### 代码层面

- [ ] 符合分层架构设计
- [ ] 使用正确的导入方式
- [ ] 数值范围处理正确
- [ ] 支持 DDP 模式
- [ ] 有适当的类型注解
- [ ] 有完整的 docstring
- [ ] 通过 DEBUG 模式验证

#### 配置层面

- [ ] 新增配置项有默认值
- [ ] 配置项有注释说明
- [ ] 更新了配置模板
- [ ] 文档中有配置示例

#### 测试层面

- [ ] 单元测试覆盖
- [ ] 集成测试通过
- [ ] DEBUG 模式通过
- [ ] DDP 模式验证（如相关）

#### 文档层面

- [ ] 更新 CLAUDE.md（如影响架构）
- [ ] 更新 README.md
- [ ] 代码中有必要的注释
- [ ] 提交信息清晰说明改动

### 常见错误避免

**禁止事项**：
1. 禁止绕过 `numeric_ops` 直接操作 tensor 范围
2. 禁止在命令层硬编码模型/数据集类型
3. 禁止直接使用 `model.module`（用 `get_raw_model()`）
4. 禁止在 Dataset 外部重复归一化
5. 禁止在命令层重复实现加载逻辑
6. 禁止在命令层直接调用 matplotlib（用 `visualization_ops` 封装）

**必须事项**：
1. 必须使用 `validate_range()` 验证数值范围
2. 必须使用 `composite_to_uint8()` 转换图像格式
3. 必须检查 DDP 下主进程判断（`is_main_process()`）
4. 必须使用注册表注册新模型/数据集
5. 必须实现 `get_output_range()` 和 `get_composite_method()`

### 性能优化指导

1. **数据加载优化**：
   - 使用 `pin_memory=True`
   - 调整 `num_workers` 避免 CPU 瓶颈
   - 大数据集考虑内存预加载

2. **训练优化**：
   - 使用混合精度（`mixed_precision.enabled`）
   - 使用梯度累积增大等效 batch size
   - 使用 DDIM 加速采样验证

3. **内存优化**：
   - 验证时关闭梯度计算（`torch.no_grad()`）
   - 定期清理 CUDA 缓存
   - 使用 `max_samples` 限制验证样本数

### 扩展指南

#### 新增模型

```python
# 1. 在 models/your_model/ 下实现
from models.base import BaseModelInterface, CompositeMethod

@register_model('your_model')
class YourModelInterface(BaseModelInterface):
    def build_model(self, device='cpu'):
        # 构建实际模型
        pass

    def get_output(self, sar, config):
        # 返回 ModelOutput
        pass

    def get_composite_method(self):
        # 返回 CompositeMethod
        return CompositeMethod.DIRECT

# 2. 在 models/__init__.py 中导入
from .your_model import YourModelInterface

# 3. 在 config 中使用
model:
  type: "your_model"
```

#### 新增数据集

```python
# 1. 在 datasets/your_dataset.py 下实现
from datasets.base import BaseDataset

@register_dataset('your_dataset')
class YourDataset(BaseDataset):
    def __getitem__(self, idx):
        return {
            'sar': sar_tensor,      # [C, H, W]
            'optical': opt_tensor,  # [C, H, W]
            'sar_path': path,
            'optical_path': path
        }

# 2. 在 datasets/__init__.py 中导入
from .your_dataset import YourDataset

# 3. 在 config 中使用
data:
  type: "your_dataset"
```

#### 新增命令

```python
# 1. 在 commands/your_command.py 下实现
from commands.base import BaseCommand, command

@command('your_cmd')
class YourCommand(BaseCommand):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--your-arg')

    def execute(self, args):
        # 调用流程层 API
        pass

# 2. 在 main.py 中导入
from commands.your_command import YourCommand
```

### 提交规范

**Commit Message 格式**：
```
类型: 简要描述

详细说明（可选）

- 改动1
- 改动2
```

**类型**：
- `feat`: 新功能
- `fix`: Bug 修复
- `refactor`: 重构
- `perf`: 性能优化
- `docs`: 文档更新
- `test`: 测试相关

**示例**：
```
feat: 新增边缘损失函数 edge_roberts

- 在 models/srdm/losses.py 中实现 Roberts 边缘检测损失
- 支持配置中启用 edge_roberts 选项
- 添加对应单元测试
- 更新 config_srdm.yaml 模板
```
