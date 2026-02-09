# 配置文件重构设计

## 当前问题
- 配置文件混杂了模型、数据集、训练参数
- 数据集脚本都放在一个文件夹，没有隔离
- 添加新模型/数据集时需要修改多处

## 目标架构

### 1. 模型配置 (v2/models/{model_name}/config.yaml)
每个模型有自己的配置文件夹：
```
v2/models/srdm/
├── config.yaml      # SRDM特有配置
├── interface.py
├── diffusion.py
└── ...

v2/models/ddpm/      # 未来添加的标准DDPM
├── config.yaml
├── interface.py
└── ...
```

### 2. 数据集配置 (v2/datasets/{dataset_name}/)
每个数据集有自己的配置文件夹：
```
v2/datasets/whu/
├── config.yaml
├── dataset.py
└── ...

v2/datasets/sen12/
├── config.yaml
├── dataset.py
└── ...
```

### 3. 主配置 (根目录 config.yaml)
只保留训练相关配置，引用模型和数据集：
```yaml
# 使用的模型（会自动加载 v2/models/{model}/config.yaml）
model:
  name: "srdm"

# 使用的数据集（会自动加载 v2/datasets/{dataset}/config.yaml）
data:
  name: "sen12"

# 训练配置
training:
  num_epochs: 100
  ...

# 设备配置
device:
  use_cuda: true
  ...

# 日志配置
logging:
  level: "INFO"
  ...
```

## 配置合并逻辑

```python
def load_config(main_config_path):
    # 1. 加载主配置
    main_config = yaml.load(main_config_path)
    
    # 2. 加载模型配置
    model_name = main_config['model']['name']
    model_config = yaml.load(f"v2/models/{model_name}/config.yaml")
    
    # 3. 加载数据集配置
    dataset_name = main_config['data']['name']
    dataset_config = yaml.load(f"v2/datasets/{dataset_name}/config.yaml")
    
    # 4. 合并配置（主配置优先级最高）
    final_config = merge_configs(model_config, dataset_config, main_config)
    
    return final_config
```

## 文件变更清单

### 新建文件
1. `v2/models/srdm/config.yaml` - SRDM模型配置
2. `v2/datasets/whu/config.yaml` - WHU数据集配置
3. `v2/datasets/sen12/config.yaml` - SEN12数据集配置
4. `config.yaml` - 新的主配置
5. `v2/core/config_loader.py` - 配置加载器

### 移动文件
1. `v2/datasets/sar_optical_dataset.py` -> `v2/datasets/whu/dataset.py`
2. `v2/datasets/sen12_dataset.py` -> `v2/datasets/sen12/dataset.py`

### 删除文件
1. `config_srdm.yaml` (被拆分)
2. `config_sen12.yaml` (被拆分)
3. `config_whu.yaml` (被拆分)

## 向后兼容
保留旧的配置文件一段时间，添加 deprecation 警告。
