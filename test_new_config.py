"""测试新的配置系统"""
import sys
from pathlib import Path

# 添加项目路径
v2_dir = Path(__file__).parent / 'v2'
if str(v2_dir) not in sys.path:
    sys.path.insert(0, str(v2_dir))

# 先导入模型和数据集模块以触发注册
import models
import datasets

print("=" * 70)
print("Testing New Configuration System")
print("=" * 70)

# 测试1: 配置加载
print("\n[TEST 1] Loading merged configuration...")
from core.config_loader import load_config, get_config_summary

try:
    config = load_config(verbose=True)
    print("[OK] Configuration loaded successfully")
except Exception as e:
    print(f"[FAIL] Failed to load config: {e}")
    sys.exit(1)

# 打印配置摘要
print("\n" + get_config_summary(config))

# 测试2: 数据集创建
print("\n[TEST 2] Creating dataset from config...")
from datasets.registry import create_dataset

try:
    # 注意：这需要实际数据集存在
    dataset = create_dataset(config, split='train')
    print(f"[OK] Dataset created with {len(dataset)} samples")
    print(f"     Data range: {dataset.get_data_range()}")
except FileNotFoundError as e:
    print(f"[SKIP] Dataset not found (expected if data not available): {e}")
except Exception as e:
    print(f"[FAIL] Dataset creation failed: {e}")
    import traceback
    traceback.print_exc()

# 测试3: 模型创建
print("\n[TEST 3] Creating model from config...")
from models.registry import create_model

try:
    model = create_model(config, device='cpu')
    print(f"[OK] Model created with {model.count_parameters()['total']:,} parameters")
    print(f"     Model type: {config['model']['name']}")
    print(f"     Output range: {model.get_output_range()}")
except Exception as e:
    print(f"[FAIL] Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 模型调试
print("\n[TEST 4] Running model debug...")
try:
    import torch
    report = model.debug(torch.device('cpu'), verbose=False)
    print(f"[OK] Debug completed: {report.overall_status}")
    for test in report.tests:
        print(f"     [{test.status}] {test.component_name}")
except Exception as e:
    print(f"[FAIL] Debug failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
