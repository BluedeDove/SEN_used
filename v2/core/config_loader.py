"""
config_loader.py - 配置加载器

支持分层配置系统：
1. 主配置 (config.yaml) - 训练流程参数
2. 模型配置 (v2/models/{model}/config.yaml) - 模型特有参数
3. 数据集配置 (v2/datasets/{dataset}/config.yaml) - 数据集特有参数

配置优先级: 主配置 > 数据集配置 > 模型配置
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    深度合并两个字典

    Args:
        base: 基础字典
        override: 覆盖字典（优先级更高）

    Returns:
        合并后的字典
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并子字典
            result[key] = deep_merge(result[key], value)
        else:
            # 覆盖或添加
            result[key] = value

    return result


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 文件不存在
        yaml.YAMLError: YAML解析错误
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config if config else {}


def get_model_config_path(model_name: str) -> Path:
    """
    获取模型配置文件路径

    Args:
        model_name: 模型名称

    Returns:
        配置文件路径
    """
    # 从项目根目录开始找
    base_dir = Path(__file__).parent.parent.parent
    return base_dir / 'v2' / 'models' / model_name / 'config.yaml'


def get_dataset_config_path(dataset_name: str) -> Path:
    """
    获取数据集配置文件路径

    Args:
        dataset_name: 数据集名称

    Returns:
        配置文件路径
    """
    base_dir = Path(__file__).parent.parent.parent
    return base_dir / 'v2' / 'datasets' / dataset_name / 'config.yaml'


def load_config(
    main_config_path: str = 'config.yaml',
    verbose: bool = False
) -> Dict[str, Any]:
    """
    加载并合并配置

    加载顺序（后加载的优先级更高）：
    1. 模型配置 (v2/models/{model}/config.yaml)
    2. 数据集配置 (v2/datasets/{dataset}/config.yaml)
    3. 主配置 (config.yaml)

    Args:
        main_config_path: 主配置文件路径
        verbose: 是否显示加载详情

    Returns:
        合并后的完整配置

    Raises:
        ValueError: 配置无效
    """
    # 加载主配置
    main_config = load_yaml_config(main_config_path)

    if verbose:
        print(f"[ConfigLoader] Loaded main config from: {main_config_path}")

    # 获取模型和数据集名称
    model_name = main_config.get('model', {}).get('name', 'srdm')
    dataset_name = main_config.get('data', {}).get('name', 'sen12')

    # 加载模型配置
    model_config_path = get_model_config_path(model_name)
    if model_config_path.exists():
        model_config = load_yaml_config(str(model_config_path))
        if verbose:
            print(f"[ConfigLoader] Loaded model config from: {model_config_path}")
    else:
        model_config = {}
        if verbose:
            print(f"[ConfigLoader] Warning: Model config not found: {model_config_path}")

    # 加载数据集配置
    dataset_config_path = get_dataset_config_path(dataset_name)
    if dataset_config_path.exists():
        dataset_config = load_yaml_config(str(dataset_config_path))
        if verbose:
            print(f"[ConfigLoader] Loaded dataset config from: {dataset_config_path}")
    else:
        dataset_config = {}
        if verbose:
            print(f"[ConfigLoader] Warning: Dataset config not found: {dataset_config_path}")

    # 合并配置（优先级：主配置 > 数据集配置 > 模型配置）
    # 先合并模型和数据集
    merged = deep_merge(model_config, dataset_config)
    # 再用主配置覆盖
    final_config = deep_merge(merged, main_config)

    # 添加元信息
    final_config['_meta'] = {
        'main_config': str(main_config_path),
        'model_config': str(model_config_path) if model_config_path.exists() else None,
        'dataset_config': str(dataset_config_path) if dataset_config_path.exists() else None,
        'model_name': model_name,
        'dataset_name': dataset_name,
    }

    if verbose:
        print(f"[ConfigLoader] Config merged successfully")
        print(f"[ConfigLoader]   Model: {model_name}")
        print(f"[ConfigLoader]   Dataset: {dataset_name}")

    return final_config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置是否有效

    Args:
        config: 配置字典

    Returns:
        是否有效

    Raises:
        ValueError: 配置无效时抛出详细错误
    """
    # 检查必需的顶级键
    required_keys = ['model', 'data', 'training']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")

    # 检查模型配置
    if 'name' not in config.get('model', {}):
        raise ValueError("Missing 'model.name' in config")

    # 检查数据集配置
    if 'name' not in config.get('data', {}):
        raise ValueError("Missing 'data.name' in config")

    # 检查训练配置
    training = config.get('training', {})
    if 'num_epochs' not in training:
        raise ValueError("Missing 'training.num_epochs' in config")

    return True


def get_config_summary(config: Dict[str, Any]) -> str:
    """
    获取配置摘要

    Args:
        config: 配置字典

    Returns:
        摘要字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Configuration Summary")
    lines.append("=" * 60)

    # 模型信息
    model_name = config.get('model', {}).get('name', 'unknown')
    lines.append(f"Model: {model_name}")

    # 数据集信息
    dataset_name = config.get('data', {}).get('name', 'unknown')
    lines.append(f"Dataset: {dataset_name}")

    # 训练信息
    training = config.get('training', {})
    lines.append(f"Epochs: {training.get('num_epochs', 'N/A')}")

    # 优化器
    optimizer = training.get('optimizer', {})
    lines.append(f"Optimizer: {optimizer.get('type', 'N/A')} (lr={optimizer.get('lr', 'N/A')})")

    # 数据来源
    meta = config.get('_meta', {})
    lines.append("-" * 60)
    lines.append("Config files:")
    if meta.get('model_config'):
        lines.append(f"  Model: {meta['model_config']}")
    if meta.get('dataset_config'):
        lines.append(f"  Dataset: {meta['dataset_config']}")
    lines.append(f"  Main: {meta.get('main_config', 'N/A')}")
    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    """测试配置加载器"""
    import sys

    print("Testing ConfigLoader...")
    print()

    # 测试1: 加载配置
    print("[Test 1] Load and merge configs")
    try:
        config = load_config(verbose=True)
        print("  [OK] Config loaded and merged")
    except Exception as e:
        print(f"  [FAIL] Failed to load config: {e}")
        sys.exit(1)

    # 测试2: 验证配置
    print()
    print("[Test 2] Validate config")
    try:
        validate_config(config)
        print("  [OK] Config is valid")
    except ValueError as e:
        print(f"  [FAIL] Config validation failed: {e}")
        sys.exit(1)

    # 测试3: 打印摘要
    print()
    print("[Test 3] Config summary")
    print(get_config_summary(config))

    # 测试4: 深度合并
    print()
    print("[Test 4] Deep merge")
    base = {'a': 1, 'b': {'c': 2, 'd': 3}}
    override = {'b': {'c': 4, 'e': 5}}
    result = deep_merge(base, override)
    expected = {'a': 1, 'b': {'c': 4, 'd': 3, 'e': 5}}
    if result == expected:
        print("  [OK] Deep merge works correctly")
    else:
        print(f"  [FAIL] Expected {expected}, got {result}")
        sys.exit(1)

    print()
    print("All tests passed!")
