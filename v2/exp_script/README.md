# 实验脚本系统 (Experiment Script System)

高自由度、高容错的实验脚本系统，支持动态加载和沙盒化执行。

## 目录结构

```
v2/exp_script/
├── user/                   # 用户自定义脚本目录（你的实验脚本放在这里）
│   └── my_experiment.py    # 你的实验脚本
├── examples/               # 示例脚本目录（参考用）
│   └── example_analysis.py # 批量推理分析示例
├── logs/                   # 错误日志目录（自动生成）
├── __init__.py             # 包导出
├── context.py              # ExpContext 沙盒上下文
├── runner.py               # 脚本加载与执行器
├── errors.py               # 错误定义
└── README.md               # 本文档
```

**注意**：请将你的实验脚本放在 `v2/exp_script/user/` 目录下，系统会自动搜索该目录。

## 快速开始

### 1. 运行实验脚本

```bash
# 列出所有可用脚本
python main.py exp --list

# 运行指定脚本
python main.py exp --name my_experiment

# 使用自定义配置
python main.py exp --name my_experiment --config custom_config.yaml

# 显示详细输出
python main.py exp --name my_experiment --verbose
```

### 2. 创建实验脚本

在 `v2/exp_script/user/` 目录下创建 `.py` 文件：

```python
# v2/exp_script/user/my_experiment.py

def run_experiment(ctx):
    """
    实验入口函数
    
    Args:
        ctx: ExpContext 实例，提供所有API访问
        
    Returns:
        可选，执行结果字典
    """
    # 加载配置
    config = ctx.load_config()
    
    # 获取数据集
    dataset = ctx.create_dataset(split='val')
    
    # 创建模型
    device = ctx.get_device()
    model = ctx.create_model(device=device)
    
    # 你的实验逻辑...
    result = {"samples": len(dataset)}
    
    return result
```

## 脚本编写规则

### 入口函数规范

每个实验脚本必须定义一个入口函数：

```python
def run_experiment(ctx, **kwargs):
    """
    实验入口函数
    
    Args:
        ctx: ExpContext 实例，提供所有项目能力
        **kwargs: 命令行传递的额外参数
        
    Returns:
        任意类型，将作为执行结果返回
    """
    pass
```

**要求：**
- 函数名必须是 `run_experiment`
- 第一个参数必须是 `ctx`（ExpContext 实例）
- 返回值可选，可以是任意类型

### 禁止事项

#### 1. 禁止直接导入底层模块

```python
# ❌ 错误 - 破坏了沙盒隔离
from v2.core.numeric_ops import composite_to_uint8
from v2.models.registry import create_model

# ✅ 正确 - 通过 ctx 访问
ctx.composite_to_uint8(image)
ctx.create_model()
```

#### 2. 禁止直接访问 `model.module`

```python
# ❌ 错误
raw_model = model.module

# ✅ 正确
raw_model = ctx.get_raw_model(model)
```

#### 3. 禁止手动处理数值范围

```python
# ❌ 错误 - 绕过 numeric_ops
output = (output - output.min()) / (output.max() - output.min())

# ✅ 正确
output = ctx.clamp_and_normalize(output)
```

#### 4. 禁止直接访问文件系统（用于实验数据）

```python
# ❌ 错误
with open('my_result.txt', 'w') as f:
    f.write(result)

# ✅ 正确 - 使用 ctx 提供的工具
ctx.save_image(image, 'result.png')
```

### 错误处理最佳实践

#### 使用 `safe_run` 包装高风险操作

```python
def run_experiment(ctx):
    results = []
    
    for i in range(100):
        # 使用 safe_run 确保单点错误不影响整体
        result = ctx.safe_run(
            lambda idx=i: process_sample(ctx, idx),
            default=None,
            error_msg=f"Failed to process sample {i}"
        )
        
        if result is not None:
            results.append(result)
    
    return {"processed": len(results)}

def process_sample(ctx, idx):
    # 可能出错的操作
    sample = dataset[idx]
    return ctx.compute_metrics(sample['pred'], sample['gt'])
```

#### 优雅处理缺失资源

```python
def run_experiment(ctx):
    # 尝试加载检查点，失败时使用随机初始化
    checkpoint = ctx.safe_run(
        lambda: ctx.load_checkpoint('checkpoint.pth'),
        default=None,
        error_msg="Checkpoint not found, using random init"
    )
    
    if checkpoint:
        ctx.restore_model(model, checkpoint['model_state_dict'])
    
    # 继续执行...
```

## ExpContext API 列表

### 配置管理

| 方法 | 说明 | 来源模块 |
|------|------|----------|
| `load_config(config_path)` | 加载并合并配置 | `config_loader` |
| `get_config()` | 获取当前配置 | - |
| `validate_config(config)` | 验证配置 | `config_loader` |
| `get_config_summary(config)` | 获取配置摘要 | `config_loader` |

### 设备管理

| 方法 | 说明 | 来源模块 |
|------|------|----------|
| `setup_device_and_distributed(config)` | 初始化设备和DDP | `device_ops` |
| `get_device()` | 获取当前设备 | - |
| `get_raw_model(model)` | 获取原始模型（解包DDP） | `device_ops` |
| `is_main_process(rank)` | 是否主进程 | `device_ops` |
| `cleanup_resources()` | 清理资源 | `device_ops` |
| `synchronize()` | 分布式同步 | `device_ops` |
| `set_seed(seed)` | 设置随机种子 | `device_ops` |
| `get_device_info()` | 获取设备信息 | `device_ops` |
| `wrap_model_ddp(model, find_unused)` | 包装为DDP | `device_ops` |
| `all_reduce_tensor(tensor, op)` | All reduce | `device_ops` |

### 模型和数据集

| 方法 | 说明 | 来源模块 |
|------|------|----------|
| `create_model(config, device)` | 创建模型 | `models/registry` |
| `create_dataset(config, split)` | 创建数据集 | `datasets/registry` |
| `list_available_models()` | 列出可用模型 | `models/registry` |
| `list_available_datasets()` | 列出可用数据集 | `datasets/registry` |

### 数值转换

| 方法 | 说明 | 来源模块 |
|------|------|----------|
| `validate_range(tensor, range, name)` | 验证数值范围 | `numeric_ops` |
| `normalize_to_range(tensor, from, to)` | 范围映射 | `numeric_ops` |
| `clamp_and_normalize(tensor, ...)` | 截断并归一化 | `numeric_ops` |
| `model_output_to_composite(output, base, ...)` | 模型输出合成 | `numeric_ops` |
| `composite_to_uint8(composite, range)` | 转 uint8 | `numeric_ops` |
| `tensor_info(tensor, name)` | 获取张量信息 | `numeric_ops` |

### 推理

| 方法 | 说明 | 来源模块 |
|------|------|----------|
| `run_inference(config, checkpoint, max_samples, device)` | 运行推理 | `inference_ops` |
| `inference_batch(model, dataloader, config, device)` | 批量推理 | `inference_ops` |

### 训练

| 方法 | 说明 | 来源模块 |
|------|------|----------|
| `setup_training(config, device)` | 设置训练组件 | `training_ops` |
| `get_training_context()` | 获取训练上下文 | - |
| `train_step(model, batch, optimizer, amp, max_norm)` | 单步训练 | `training_ops` |
| `run_training_epoch(ctx, epoch, num_epochs, ...)` | 训练 epoch | `training_ops` |
| `run_training_loop(ctx, config, ...)` | 训练循环 | `training_ops` |

### 验证

| 方法 | 说明 | 来源模块 |
|------|------|----------|
| `compute_psnr(img1, img2, max_val)` | 计算 PSNR | `validation_ops` |
| `compute_ssim(img1, img2, window)` | 计算 SSIM | `validation_ops` |
| `compute_metrics_batch(gen, opt)` | 计算批次指标 | `validation_ops` |
| `run_validation(model, val_loader, config, ...)` | 运行验证 | `validation_ops` |

### 检查点

| 方法 | 说明 | 来源模块 |
|------|------|----------|
| `save_checkpoint(model, opt, scheduler, epoch, metrics, path, config)` | 保存 | `checkpoint_ops` |
| `load_checkpoint(path, device)` | 加载 | `checkpoint_ops` |
| `restore_model(model, state_dict, strict)` | 恢复模型 | `checkpoint_ops` |
| `get_latest_checkpoint(dir)` | 获取最新检查点 | `checkpoint_ops` |

### 可视化

| 方法 | 说明 | 来源模块 |
|------|------|----------|
| `setup_report_directory(exp_dir)` | 创建报告目录 | `visualization_ops` |
| `log_training_loss(log_dir, epoch, loss, val_metrics)` | 记录训练日志 | `visualization_ops` |
| `plot_loss_curve(log_file, save_path, title)` | 绘制 loss 曲线 | `visualization_ops` |
| `create_inference_comparison(sar, gen, opt)` | 创建对比图 | `visualization_ops` |
| `create_comparison_figure(paths, save_path, title)` | 创建对比报告 | `visualization_ops` |
| `generate_validation_report(result_dir, report_dir, epoch)` | 生成验证报告 | `visualization_ops` |

### 图像操作

| 方法 | 说明 | 来源模块 |
|------|------|----------|
| `tensor_to_numpy(tensor, channel_order)` | Tensor 转 numpy | `image_ops` |
| `numpy_to_tensor(array, channel_order, device)` | numpy 转 Tensor | `image_ops` |
| `save_image(image, path, create_dir)` | 保存图像 | `image_ops` |
| `load_image(path, to_rgb)` | 加载图像 | `image_ops` |
| `create_grid(images, n_cols, padding)` | 创建图像网格 | `image_ops` |

### 安全执行

| 方法 | 说明 |
|------|------|
| `safe_run(func, *args, default=None, error_msg=None, **kwargs)` | 安全执行函数 |
| `safe_call(obj, method_name, *args, default=None, error_msg=None, **kwargs)` | 安全调用方法 |

### 日志

| 方法 | 说明 |
|------|------|
| `log_info(message)` | 信息日志 |
| `log_warning(message)` | 警告日志 |
| `log_error(message)` | 错误日志 |
| `log_debug(message)` | 调试日志 |

## 示例脚本

### 示例 1：批量推理分析

```python
# v2/exp_script/batch_inference.py
"""
批量推理并分析指标分布
"""

def run_experiment(ctx):
    import numpy as np
    
    # 1. 加载配置和模型
    config = ctx.load_config()
    device = ctx.get_device()
    model = ctx.create_model(device=device)
    
    # 2. 加载检查点
    checkpoint_path = config.get('checkpoint', 'checkpoints/latest.pth')
    checkpoint = ctx.safe_run(
        lambda: ctx.load_checkpoint(checkpoint_path, device),
        default=None,
        error_msg="Failed to load checkpoint"
    )
    if checkpoint:
        ctx.restore_model(model, checkpoint['model_state_dict'])
    
    # 3. 获取数据集
    dataset = ctx.create_dataset(split='val')
    
    # 4. 批量推理
    all_psnr = []
    all_ssim = []
    
    for idx in range(min(100, len(dataset))):
        result = ctx.safe_run(
            lambda i=idx: process_single(ctx, model, dataset, i, device),
            default=None,
            error_msg=f"Failed to process sample {idx}"
        )
        
        if result:
            all_psnr.append(result['psnr'])
            all_ssim.append(result['ssim'])
    
    # 5. 分析结果
    report = {
        'total_samples': len(all_psnr),
        'psnr_mean': float(np.mean(all_psnr)) if all_psnr else 0,
        'psnr_std': float(np.std(all_psnr)) if all_psnr else 0,
        'ssim_mean': float(np.mean(all_ssim)) if all_ssim else 0,
        'ssim_std': float(np.std(all_ssim)) if all_ssim else 0,
    }
    
    # 6. 保存报告
    import json
    with open('analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


def process_single(ctx, model, dataset, idx, device):
    """处理单个样本"""
    import torch
    
    sample = dataset[idx]
    sar = sample['sar'].unsqueeze(0).to(device)
    optical = sample['optical'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.get_output(sar, {})
        generated = output.generated
    
    psnr = ctx.compute_psnr(generated, optical)
    ssim = ctx.compute_ssim(generated, optical)
    
    return {'psnr': psnr, 'ssim': ssim}
```

### 示例 2：对比多个检查点

```python
# v2/exp_script/compare_checkpoints.py
"""
对比多个检查点的性能
"""

def run_experiment(ctx):
    import torch
    
    config = ctx.load_config()
    device = ctx.get_device()
    dataset = ctx.create_dataset(split='val')
    
    # 定义要对比的检查点
    checkpoints = [
        ('Epoch 10', 'checkpoints/epoch_0010.pth'),
        ('Epoch 50', 'checkpoints/epoch_0050.pth'),
        ('Latest', 'checkpoints/latest.pth'),
    ]
    
    results = []
    
    for name, path in checkpoints:
        result = ctx.safe_run(
            lambda n=name, p=path: evaluate_checkpoint(ctx, config, dataset, device, n, p),
            default=None,
            error_msg=f"Failed to evaluate {name}"
        )
        
        if result:
            results.append(result)
    
    # 打印对比结果
    print("\n" + "=" * 60)
    print("Checkpoint Comparison")
    print("=" * 60)
    for r in results:
        print(f"{r['name']:15s} PSNR: {r['psnr']:.2f}  SSIM: {r['ssim']:.4f}")
    print("=" * 60)
    
    return results


def evaluate_checkpoint(ctx, config, dataset, device, name, path):
    """评估单个检查点"""
    import torch
    
    model = ctx.create_model(config, device)
    checkpoint = ctx.load_checkpoint(path, device)
    ctx.restore_model(model, checkpoint['model_state_dict'])
    model.eval()
    
    all_psnr = []
    all_ssim = []
    
    with torch.no_grad():
        for i in range(min(50, len(dataset))):
            sample = dataset[i]
            sar = sample['sar'].unsqueeze(0).to(device)
            optical = sample['optical'].unsqueeze(0).to(device)
            
            output = model.get_output(sar, config)
            generated = output.generated
            
            all_psnr.append(ctx.compute_psnr(generated, optical))
            all_ssim.append(ctx.compute_ssim(generated, optical))
    
    import numpy as np
    return {
        'name': name,
        'path': path,
        'psnr': float(np.mean(all_psnr)),
        'ssim': float(np.mean(all_ssim)),
    }
```

## 错误处理机制

### 日志文件

错误日志保存在 `v2/exp_script/logs/` 目录下：

- `{script_name}_error.log` - 错误详情
- `{script_name}_summary.log` - 执行摘要

### 错误类型

| 错误类型 | 说明 |
|----------|------|
| `ExperimentNotFoundError` | 脚本文件不存在 |
| `ExperimentValidationError` | 脚本缺少入口函数或格式错误 |
| `ExperimentConfigError` | 配置文件无效 |
| `ExperimentExecutionError` | 脚本执行时发生异常 |
| `ExperimentAPITimeoutError` | API 调用超时 |

### 容错策略

1. **单点错误不中断**：使用 `safe_run` 包装的操作出错时返回默认值，继续执行
2. **详细日志记录**：所有错误记录到文件，包含完整 traceback
3. **控制台简明输出**：只显示关键错误信息，详细信息查看日志文件
4. **执行摘要**：记录成功/失败状态和耗时

## 开发规范

### 新增 ExpContext API

如果在核心模块（如 `numeric_ops.py`、`device_ops.py` 等）中添加了新的封装函数，需要在 `ExpContext` 中注册对应的包装方法：

1. **导入函数**：在 `v2/exp_script/context.py` 顶部导入新函数
2. **添加方法**：在 `ExpContext` 类中添加对应的包装方法
3. **分类放置**：按照功能分类（数值转换、设备管理等）放置在对应区域
4. **编写文档**：为方法编写 docstring，说明参数和返回值

示例：

```python
# v2/exp_script/context.py

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

### 提交 checklist

- [ ] 脚本包含 `run_experiment(ctx)` 入口函数
- [ ] 通过 `ctx` 访问所有项目功能，无直接导入
- [ ] 高风险操作使用 `safe_run` 包装
- [ ] 包含 docstring 说明脚本功能
- [ ] 测试通过：`python main.py exp --name your_script --verbose`
