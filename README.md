# SRDM v2 - SAR到光学图像翻译框架

> 支持 SRDM（残差扩散模型）和 Flow Matching 的现代化 SAR 到光学图像翻译框架

## 项目简介

SRDM v2 是一个支持多种生成模型的 SAR（合成孔径雷达）到光学图像翻译框架：

- **SRDM**: 预测残差 `R = Optical - SAR_base`，大幅降低学习难度
- **Flow Matching**: 基于 Rectified Flow 的直接图像生成

### 核心特性

- **多模型支持**: SRDM（扩散）和 Flow Matching（流匹配）
- **多数据集支持**: SEN1-2、WHU 等 SAR-光学配对数据集
- **分层架构**: 命令层/流程层/接口层/实现层分离
- **配置驱动**: YAML 配置 + 子配置组合，灵活可控
- **健壮性设计**: 损坏文件自动跳过、训练中断自动保存
- **DDP支持**: 原生多卡分布式训练，支持混合精度
- **续训灵活**: 中断后可修改配置（lr、batch_size等）再续训

---

## 快速开始

### 安装依赖

```bash
# 创建环境
conda create -n srdm python=3.10
conda activate srdm

# 安装 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install pyyaml tqdm pillow numpy
```

### 项目结构

```
SEN_used/
├── config.yaml              # 主配置文件
├── main.py                  # 入口脚本
├── experiments/             # 实验输出目录
│   └── my_exp/
│       ├── checkpoints/     # 模型检查点
│       │   ├── latest.pth
│       │   ├── epoch_0010.pth
│       │   └── emergency_interrupt.pth  # 紧急保存
│       ├── logs/            # 训练日志
│       └── results/         # 验证结果
└── v2/
    ├── commands/            # 命令入口
    ├── core/                # 核心流程
    ├── models/              # 模型实现
    │   ├── srdm/            # SRDM 模型
    │   └── flowmatching/    # Flow Matching 模型
    └── datasets/            # 数据集实现
        ├── sen12/           # SEN1-2 配置
        └── whu/             # WHU 配置
```

### 数据准备

#### SEN1-2 数据集

```
/root/SEN1-2/                    # 数据集根目录
├── ROIs1158_spring/
│   ├── s1_0/                    # SAR 样本目录
│   │   └── ROIs1158_spring_s1_0_p1.png
│   └── s2_0/                    # 光学样本目录
│       └── ROIs1158_spring_s2_0_p1.png
├── ROIs1868_summer/
├── ROIs1970_fall/
└── ROIs2017_winter/
```

**自动配对**: 根据文件名 `_s1_` 和 `_s2_` 自动配对 SAR 和光学图像。

---

## 命令行用法

### 1. 训练

#### 单卡训练

```bash
python main.py train --config config.yaml --experiment my_first_exp
```

#### DDP 分布式训练（推荐）

```bash
# 双卡训练
torchrun --nproc_per_node=2 main.py train --config config.yaml --experiment my_exp

# 四卡训练
torchrun --nproc_per_node=4 main.py train --config config.yaml --experiment my_exp
```

**训练输出**:
```
experiments/my_exp/
├── checkpoints/
│   ├── latest.pth              # 最新检查点（每 epoch 更新）
│   ├── epoch_0010.pth          # 每 10 epoch 保存
│   └── emergency_interrupt.pth # 中断/崩溃时保存
├── logs/                       # TensorBoard 日志
└── results/                    # 验证可视化结果
    └── epoch_0010/
        ├── sample_000.png
        └── ...
```

### 2. 续训（Resume）

#### 基本续训

```bash
# 从 latest.pth 续训
torchrun --nproc_per_node=2 main.py train --config config.yaml --experiment my_exp --resume
```

#### 修改配置后续训

**支持修改的配置**（修改 config.yaml 后 --resume 自动生效）：

```yaml
# config.yaml - 修改这些参数后 resume 会生效
training:
  num_epochs: 200          # 增加训练轮数
  optimizer:
    lr: 5e-5               # 降低学习率
  gradient_accumulation:
    steps: 4               # 调整梯度累积

data:
  dataloader:
    batch_size: 45         # 修改 batch size
    num_workers: 8         # 修改数据加载线程
```

**输出示例**:
```
Resuming from experiments/my_exp/checkpoints/latest.pth
  Saved model type: FlowMatchingModel
  Current model type: FlowMatchingModel
  Updating learning rate: 0.0001 -> 5e-05
Resumed from epoch 51
  Current config will be used for: batch_size, lr, num_epochs, etc.
Epoch 52/200:  45%|████▌| 2822/6283 [23:12<28:18, 2.04it/s, loss=0.0892]
```

### 3. 推理

```bash
python main.py infer \
    --config config.yaml \
    --checkpoint experiments/my_exp/checkpoints/latest.pth \
    --output results/
```

### 4. DEBUG 模式

```bash
# 基本调试
python main.py debug --config config.yaml

# 详细输出
python main.py debug --config config.yaml --verbose

# 只测试数据集
python main.py debug --config config.yaml --test-dataset

# 只测试模型
python main.py debug --config config.yaml --test-model
```

### 5. 查看帮助

```bash
python main.py --help
python main.py train --help
python main.py infer --help
```

---

## 配置文件详解

### 配置结构

项目使用**主配置 + 子配置**的组合方式：

```yaml
# config.yaml (主配置)
data:
  type: "sen12"                    # 使用 v2/datasets/sen12/config.yaml

model:
  name: "flowmatching"             # 使用 v2/models/flowmatching/config.yaml

training:
  num_epochs: 100
  optimizer:
    type: "adamw"
    lr: 1e-4
```

### 完整主配置示例

```yaml
# ==================== 数据配置 ====================
data:
  type: "sen12"                    # 数据集类型: sen12, whu

# ==================== 模型配置 ====================
model:
  name: "flowmatching"             # 模型: flowmatching, srdm

# ==================== 训练配置 ====================
training:
  num_epochs: 100
  
  optimizer:
    type: "adamw"
    lr: 1e-4
    weight_decay: 0.01
    
  scheduler:
    type: "cosine"
    warmup_epochs: 5
    min_lr: 1e-6
    
  gradient_clipping:
    enabled: true
    max_norm: 1.0
    
  gradient_accumulation:
    steps: 2                       # 梯度累积步数
    
  mixed_precision:
    enabled: true                  # 混合精度训练

# ==================== 验证配置 ====================
validation:
  interval: 10                     # 每 10 epoch 验证一次
  
# ==================== 设备配置 ====================
device:
  use_cuda: true
  random_seed: 42

# ==================== 其他配置 ====================
output:
  save_interval: 10                # 每 10 epoch 保存检查点
```

### 数据集子配置

位置: `v2/datasets/sen12/config.yaml`

```yaml
data:
  type: "sen12"
  root: "/root/SEN1-2"
  train_ratio: 0.9

  normalize:
    enabled: true
    input_range: [0.0, 1.0]

  channels:
    optical:
      input: 3
      use: 3
    sar:
      input: 3
      use: 3

  dataloader:
    batch_size: 45
    num_workers: 8
    pin_memory: true
    prefetch_factor: 4
    persistent_workers: true
```

### 模型子配置

位置: `v2/models/flowmatching/config.yaml`

```yaml
model:
  name: "flowmatching"
  output_range: [-1.0, 1.0]

flowmatching:
  base_ch: 64
  ch_mults: [1, 2, 4, 8]
  num_blocks: 2
  time_emb_dim: 256
  dropout: 0.1
  num_heads: 8
  use_sar_base: false

sar_encoder:
  in_ch: 3
  base_ch: 64
  ch_mults: [1, 2, 4, 8]
```

---

## 健壮性设计

### 1. 损坏文件自动跳过

数据集中有约 0.4% 的文件可能损坏，训练时会：
- 自动跳过损坏文件
- 尝试加载其他样本（最多10次）
- 极端情况下返回零张量，保证训练不中断

### 2. 训练中断自动保存

**Ctrl+C 中断**:
```
[INTERRUPT] Training interrupted by user
[SAVING] Attempting to save emergency checkpoint...
[SAVED] Emergency checkpoint saved to: experiments/my_exp/checkpoints/emergency_interrupt.pth
[SAVED] Also updated: experiments/my_exp/checkpoints/latest.pth
[INFO] Resume with: --resume (will continue from epoch 52)
```

**意外错误**:
- 自动保存 `emergency_interrupt.pth`
- 同时更新 `latest.pth`
- 使用 `--resume` 可从中断点恢复

### 3. 配置修改续训

支持在续训时修改：
- `batch_size`, `num_workers`（数据加载）
- `lr`, `weight_decay`（优化器参数）
- `num_epochs`, `validation_interval`（训练流程）
- `gradient_clipping`, `mixed_precision`（训练设置）

---

## 模型架构

### Flow Matching 模型

基于 Rectified Flow 的 SAR-to-Optical 翻译：

```
SAR Input [B, 3, H, W]
    ↓
SAR Encoder (U-Net 风格下采样)
    ↓
Multi-scale Features + Global Condition
    ↓
Flow Matching UNet (预测向量场)
    ↓
DPM-Solver++ 采样
    ↓
Optical Output [B, 3, H, W]
```

### SRDM 模型

基于扩散的残差学习：

```
SAR Input
    ↓
SAR Encoder → SAR_base + Features
    ↓
Residual Diffusion (预测残差)
    ↓
SAR_base + Residual = Optical
```

---

## 扩展开发

### 添加新模型

1. 在 `v2/models/` 创建新目录
2. 实现 `interface.py` 继承 `BaseModelInterface`
3. 注册模型：`@register_model('my_model')`
4. 创建 `config.yaml`

### 添加新数据集

1. 在 `v2/datasets/` 创建新目录
2. 实现 `dataset.py` 继承 `BaseDataset`
3. 注册数据集：`DATASET_REGISTRY['my_dataset'] = MyDataset`
4. 创建 `config.yaml`

---

## 常见问题

### Q: 训练时显示损坏文件警告
A: 这是正常的（约0.4%损坏率），系统会自动跳过。如需完全清理：
```bash
cd /root/SEN1-2 && find . -name "*.png" -size 0 -delete
```

### Q: 如何降低显存占用？
A: 修改 `config.yaml`:
```yaml
training:
  gradient_accumulation:
    steps: 4        # 增大累积步数，降低 batch_size
data:
  dataloader:
    batch_size: 20  # 降低 batch size
```

### Q: Resume 时学习率没有更新？
A: 确保修改的是主 `config.yaml` 中的 `training.optimizer.lr`，而非子配置。

### Q: DDP 训练卡住？
A: 检查：
1. 所有进程的 batch_size 相同
2. 没有损坏文件导致某些进程卡住
3. 使用 `drop_last=True` 避免最后批次大小不一致

---

## 许可证

MIT License
