# SRDM v2 - SAR图像翻译扩散模型

> 基于 SRDM（SAR-Residual Diffusion Model）的 SAR 到光学图像翻译框架

## 项目简介

SRDM v2 是一个现代化的 SAR（合成孔径雷达）到光学图像翻译框架，核心创新是**预测残差** `R = Optical - SAR_base` 而非完整图像，大幅降低学习难度。

### 核心特性

- **残差学习**：预测残差分布比完整图像更简单高效
- **分层架构**：清晰的命令层/流程层/接口层/实现层分离
- **配置驱动**：单一 YAML 配置控制所有行为
- **DEBUG模式**：强制数值检验，确保数据流正确性
- **DDP支持**：原生支持多卡分布式训练
- **可扩展接口**：通过注册表轻松扩展新模型/数据集

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
pip install pyyaml tqdm tifffile opencv-python numpy
```

### 数据准备

```
data/
├── train/
│   ├── sar/           # SAR图像 (1通道 TIFF)
│   │   ├── 001.tif
│   │   └── ...
│   └── opt/           # 光学图像 (3/4通道 TIFF)
│       ├── 001.tif
│       └── ...
└── test/              # 测试数据 (可选)
    ├── sar/
    └── opt/
```

**数据说明**：
- SAR图像：单通道灰度，支持 8/16 位 TIFF
- 光学图像：支持 3/4 通道，自动取前 3 通道转换为 RGB

---

## 命令行用法

### 1. DEBUG 模式（首次运行必做）

在训练前，运行 DEBUG 模式验证数值范围和数据流：

```bash
# 基本 DEBUG 测试
python main.py debug --config config_srdm.yaml

# 详细输出
python main.py debug --config config_srdm.yaml --verbose

# 只测试数据集
python main.py debug --config config_srdm.yaml --test-dataset

# 只测试模型
python main.py debug --config config_srdm.yaml --test-model
```

**DEBUG 模式检验内容**：
- 数据集输出范围是否在 [0, 1]
- 模型输出范围是否在配置指定范围
- 合成流程是否正确（SAR_base + Residual → [0, 1]）
- uint8 转换是否正常

### 2. 训练

#### 单卡训练

```bash
# 创建新实验
python main.py train --config config_srdm.yaml --experiment my_first_exp

# 指定训练轮数
python main.py train --config config_srdm.yaml --experiment my_exp --epochs 200

# 覆盖批次大小
python main.py train --config config_srdm.yaml --experiment my_exp --batch-size 8
```

#### 续训

```bash
python main.py train --config config_srdm.yaml --experiment my_exp --resume
```

#### DDP 分布式训练（多卡）

```bash
# 双卡训练
torchrun --nproc_per_node=2 main.py train --config config_srdm.yaml --experiment ddp_exp

# 四卡训练
torchrun --nproc_per_node=4 main.py train --config config_srdm.yaml --experiment ddp_exp
```

**DDP 配置建议**：
```yaml
data:
  dataloader:
    batch_size: 23        # 每张卡的 batch_size
    num_workers: 4        # 每张卡的 worker 数
    pin_memory: true
```

### 3. 推理

```bash
# 基础推理
python main.py infer \
    --config config_srdm.yaml \
    --checkpoint experiments/my_exp/checkpoints/best.pth \
    --output results/

# 限制样本数
python main.py infer \
    --config config_srdm.yaml \
    --checkpoint experiments/my_exp/checkpoints/best.pth \
    --output results/ \
    --max-samples 50
```

### 4. 查看帮助

```bash
# 主帮助
python main.py --help

# 各命令帮助
python main.py train --help
python main.py infer --help
python main.py debug --help
```

---

## 配置文件详解

### 完整配置示例

```yaml
# ==================== 数据配置 ====================
data:
  type: "sar_optical"           # 数据集类型
  train_dir: "data/train"
  test_dir: "data/test"
  train_ratio: 0.9              # 训练集比例

  normalize:
    enabled: true
    input_range: [0.0, 1.0]     # Dataset 输出范围

  channels:
    optical:
      input: 4                  # 输入通道数
      use: 3                    # 使用通道数
      indices: [0,1,2]          # 通道索引
    sar:
      input: 1
      use: 1
      indices: [0]

  dataloader:
    batch_size: 4
    num_workers: 4
    pin_memory: true
    prefetch_factor: 2
    persistent_workers: true

# ==================== 模型配置 ====================
model:
  type: "srdm"                  # 模型类型
  name: "SRDM"
  output_range: [-1.0, 1.0]     # 模型输出范围

# SRDM 专用配置
srdm:
  base_ch: 64
  ch_mults: [1, 2, 4, 8]
  num_blocks: 2
  time_emb_dim: 256
  dropout: 0.1
  num_heads: 8

# ==================== 损失配置 ====================
srdm_loss:
  enabled:
    - noise_mse        # 噪声预测 MSE（推荐）
    - x0_mse           # 残差 MSE
    - x0_l1            # 残差 L1
    # - edge_sobel     # 边缘损失
    # - ssim           # SSIM 损失
    # - perceptual     # 感知损失

  weights:
    noise_mse: 1.0
    x0_mse: 0.5
    x0_l1: 0.3
    edge_sobel: 0.1
    ssim: 0.1
    perceptual: 0.2

# ==================== 扩散配置 ====================
diffusion:
  num_timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02

  sampling:
    enabled: true
    use_ddim: true            # 使用 DDIM 加速
    ddim_steps: 50            # DDIM 采样步数
    ddim_eta: 0.0             # 0=确定性，1=随机

# ==================== 训练配置 ====================
training:
  num_epochs: 100

  optimizer:
    type: "adamw"
    lr: 0.0002
    weight_decay: 0.01
    betas: [0.9, 0.999]

  scheduler:
    type: "cosine"
    warmup_epochs: 5
    min_lr: 0.000001

  mixed_precision:
    enabled: true             # 混合精度训练

  gradient_clipping:
    enabled: true
    max_norm: 1.0

  gradient_accumulation:
    enabled: true
    steps: 2                  # 累积步数，等效 batch_size * 2

# ==================== 验证配置 ====================
validation:
  enabled: true
  interval: 10                # 每 N epoch 验证一次
  batch_size: 4
  max_samples: 50             # 最大验证样本数

  save_results:
    enabled: true
    save_dir: "results"

# ==================== 设备配置 ====================
device:
  use_cuda: true
  random_seed: 42
  deterministic: false
```

### 关键配置说明

| 配置项 | 说明 | 建议值 |
|--------|------|--------|
| `srdm.ch_mults` | UNet 通道倍数 | `[1, 2, 4, 8]` |
| `srdm.num_blocks` | 每层 NAFBlock 数 | `2` |
| `training.mixed_precision.enabled` | 混合精度 | `true` (节省显存) |
| `training.gradient_accumulation.steps` | 梯度累积 | `2-4` (等效增大 batch) |
| `diffusion.sampling.ddim_steps` | 采样步数 | `50` (质量/速度平衡) |
| `validation.interval` | 验证间隔 | `5-10` epoch |

---

## 训练输出结构

```
experiments/
└── {experiment_name}/
    ├── checkpoints/              # 模型检查点
    │   ├── latest.pth            # 最新检查点
    │   ├── best.pth              # 最佳检查点
    │   └── epoch_0010.pth        # 定期保存的检查点
    ├── logs/                     # 训练日志
    │   └── train.log
    ├── results/                  # 验证结果
    │   └── epoch_0010/           # 每个 epoch 的验证图像
    │       ├── sample_0000.png   # SAR | Generated | GT
    │       └── ...
    └── config.yaml               # 保存的配置副本
```

**验证结果图像布局**：
```
┌─────────┬────────────┬─────────────┐
│  SAR    │  Generated │ Ground Truth│
├─────────┼────────────┼─────────────┤
│SAR Base │  Residual  │ Target Res. │
└─────────┴────────────┴─────────────┘
```

---

## 性能优化指南

### 显存优化

```yaml
# 减小 batch_size
data:
  dataloader:
    batch_size: 2

# 启用混合精度
training:
  mixed_precision:
    enabled: true

# 使用梯度累积
training:
  gradient_accumulation:
    enabled: true
    steps: 4          # 等效 batch_size = 2 * 4 = 8

# 减少 UNet 通道数
srdm:
  base_ch: 32         # 默认 64
  ch_mults: [1, 2, 4, 8]  # 或使用 [1, 2, 2, 4]
```

### 训练速度优化

```yaml
# 增加 worker 数
data:
  dataloader:
    num_workers: 8
    prefetch_factor: 4
    persistent_workers: true

# 使用 DDIM 加速验证
diffusion:
  sampling:
    use_ddim: true
    ddim_steps: 50     # 对比 1000 步快 20 倍

# 减少验证频率
validation:
  interval: 20         # 每 20 epoch 验证一次
  max_samples: 20      # 减少验证样本数
```

---

## 故障排除

### 问题：DEBUG 模式报错

```
ValueError: SAR sample 0 数值范围错误
```

**解决**：
- 检查数据归一化配置
- 确认 TIFF 文件格式正确
- 运行 `python main.py debug --verbose` 查看详细信息

### 问题：显存不足 (OOM)

```
RuntimeError: CUDA out of memory
```

**解决**：
```yaml
# 减小 batch_size
data:
  dataloader:
    batch_size: 2

# 启用混合精度
training:
  mixed_precision:
    enabled: true

# 使用梯度累积
training:
  gradient_accumulation:
    enabled: true
    steps: 4
```

### 问题：DDP 训练卡住

```
[WARNING] NCCL operation failed
```

**解决**：
- 检查 NCCL 环境变量已在 `main.py` 中设置
- 确保所有节点网络互通
- 使用 `torchrun` 而非 `python` 启动

### 问题：Loss 不下降

**排查步骤**：
1. 运行 `python main.py debug --verbose` 检查数值范围
2. 降低学习率：`lr: 0.0001`
3. 检查数据对齐：SAR 和光学图像是否对应
4. 减少验证频率，增加训练轮数

---

## 扩展开发

### 新增损失函数

```python
# v2/models/srdm/losses.py
class SRDMLoss(nn.Module):
    def forward(self, ...):
        # 新增损失项
        if 'your_loss' in self.enabled:
            w = self.weights.get('your_loss', 0.1)
            l = your_loss_function(pred_optical, target_optical)
            loss_dict['your_loss'] = l
            total_loss += w * l
```

### 新增模型架构

参考 `v2/models/srdm/interface.py` 实现：

```python
from models.base import BaseModelInterface, CompositeMethod

@register_model('your_model')
class YourModelInterface(BaseModelInterface):
    def build_model(self, device='cpu'):
        pass

    def get_output(self, sar, config):
        return ModelOutput(
            generated=output,
            output_range=self.get_output_range(),
            intermediate={...},
            metadata={...}
        )
```

详细扩展指南请参考 `CLAUDE.md`。

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v2.0 | 2026-02-05 | 初始架构设计，实现 SRDM 核心功能 |

---

## 许可证

MIT License

## 致谢

- 扩散模型实现参考：DDPM、DDIM 论文
- UNet 架构参考：NAFNet、Attention UNet
