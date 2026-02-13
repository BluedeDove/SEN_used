"""
debug.py - 调试命令

实现DEBUG模式，强制进行数值检验，确保数据流正确性。

设计原则：
- 通用调试逻辑在此文件中实现
- 模型特有调试逻辑在各自模型的 debug.py 中实现
- 通过 model.debug() 接口调用模型特有测试
"""

# 支持单独运行调试：将项目根目录添加到路径
import sys
from pathlib import Path

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

import argparse
import yaml
import torch
from commands.base import BaseCommand, command
from core.device_ops import setup_device_and_distributed, is_main_process
from core.numeric_ops import validate_range, composite_to_uint8
from core.validation_ops import run_all_ddp_validations
from datasets.registry import create_dataset
from models.registry import create_model


@command('debug')
class DebugCommand(BaseCommand):
    """
    调试命令

    DEBUG模式强制进行数值检验，验证以下内容：
    1. 数据集输出范围
    2. 模型输出范围
    3. 模型特有逻辑（通过model.debug()）
    4. 数值转换一致性
    """

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--config', type=str, default='config_srdm.yaml',
                            help='配置文件路径')
        parser.add_argument('--verbose', action='store_true',
                            help='显示详细日志')
        parser.add_argument('--test-dataset', action='store_true',
                            help='只测试数据集')
        parser.add_argument('--test-model', action='store_true',
                            help='只测试模型')
        parser.add_argument('--test-ddp', action='store_true',
                            help='测试DDP代码逻辑（无需多卡）')

    def execute(self, args: argparse.Namespace):
        """执行调试"""
        print("=" * 70)
        print("DEBUG MODE")
        print("=" * 70)
        print("This mode performs strict numerical validation to ensure")
        print("that the data flow is correct before training.")
        print("=" * 70)

        # 如果只测试DDP，可以跳过配置加载
        if args.test_ddp and not (args.test_dataset or args.test_model):
            return self._run_ddp_test(args.verbose)

        # 加载配置
        config = self._load_config(args.config)
        if config is None:
            return 1

        # 设置设备
        device, rank, world_size = setup_device_and_distributed(config)

        all_tests_passed = True

        # 测试 DDP 代码逻辑（与 config 相关时）
        if args.test_ddp:
            ddp_passed = self._run_ddp_test(args.verbose)
            all_tests_passed = all_tests_passed and ddp_passed

        # 测试1: 数据集数值范围
        if not args.test_model and not args.test_ddp:
            dataset_passed = self._run_dataset_test(config, device, args.verbose)
            all_tests_passed = all_tests_passed and dataset_passed

        # 测试2: 模型特有测试（通过通用接口调用）
        if not args.test_dataset and not args.test_ddp:
            model_passed = self._run_model_test(config, device, args.verbose)
            all_tests_passed = all_tests_passed and model_passed

        # 测试3: 数值转换一致性
        if not args.test_dataset and not args.test_ddp:
            numeric_passed = self._run_numeric_test(config, device, args.verbose)
            all_tests_passed = all_tests_passed and numeric_passed

        # 总结
        print("\n" + "=" * 70)
        if all_tests_passed:
            print("ALL TESTS PASSED - Ready for training!")
        else:
            print("SOME TESTS FAILED - Please fix before training!")
        print("=" * 70)

        return 0 if all_tests_passed else 1

    def _load_config(self, config_path: str):
            """
            加载并合并配置文件
    
            使用 config_loader 加载并合并:
            1. 模型配置 (v2/models/{model}/config.yaml)
            2. 数据集配置 (v2/datasets/{dataset}/config.yaml)
            3. 主配置 (config.yaml)
            优先级: 主配置 > 数据集配置 > 模型配置
            """
            # 导入 config_loader
            try:
                from core.config_loader import load_config as load_merged_config
                return load_merged_config(config_path, verbose=False)
            except ImportError as e:
                print(f"Warning: Failed to import config_loader: {e}")
                print("Falling back to basic YAML loading...")
    
            # 回退到基础 YAML 加载
            if not Path(config_path).exists():
                # 尝试不同路径
                alt_path = Path(__file__).parent.parent / config_path
                if alt_path.exists():
                    config_path = str(alt_path)
    
            if not Path(config_path).exists():
                print(f"Error: Config file not found: {config_path}")
                return None
    
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)

    def _run_ddp_test(self, verbose: bool) -> bool:
        """运行DDP代码测试"""
        print("\n[TEST] DDP Code Validation")
        print("-" * 70)
        print("Note: Testing DDP logic with MockDDP (no multi-GPU required)")

        ddp_passed, ddp_messages = run_all_ddp_validations(verbose=verbose)
        for msg in ddp_messages:
            status_symbol = "[OK]" if "PASSED" in msg else "[FAIL]"
            print(f"  {status_symbol} {msg}")

        if ddp_passed:
            print("[OK] DDP code test PASSED")
        else:
            print("[FAIL] DDP code test FAILED")

        return ddp_passed

    def _run_dataset_test(self, config: dict, device: torch.device, verbose: bool) -> bool:
        """运行数据集测试"""
        print("\n[TEST] Dataset Numerical Range Validation")
        print("-" * 70)

        try:
            # 创建数据集
            dataset = create_dataset(config, split='train')

            # 验证数据范围
            expected_range = dataset.get_data_range()
            print(f"Expected data range: {expected_range}")

            # 测试多个样本
            for i in range(min(5, len(dataset))):
                sample = dataset[i]
                sar = sample['sar']
                optical = sample['optical']

                if verbose:
                    print(f"  Sample {i}:")
                    print(f"    SAR: shape={list(sar.shape)}, range=[{sar.min():.4f}, {sar.max():.4f}]")
                    print(f"    Optical: shape={list(optical.shape)}, range=[{optical.min():.4f}, {optical.max():.4f}]")

                # 验证范围
                validate_range(sar, expected_range, f"SAR sample {i}")
                validate_range(optical, expected_range, f"Optical sample {i}")

            print(f"[OK] Dataset test PASSED (validated {min(5, len(dataset))} samples)")
            return True
        except Exception as e:
            print(f"[FAIL] Dataset test FAILED: {e}")
            return False

    def _run_model_test(self, config: dict, device: torch.device, verbose: bool) -> bool:
        """
        运行模型特有测试

        通过 model.debug() 接口调用模型特有的调试逻辑。
        这使得每个模型可以实现自己的测试，而不需要修改此文件。
        """
        print("\n[TEST] Model-Specific Validation")
        print("-" * 70)

        try:
            # 创建模型
            model_interface = create_model(config, device=str(device))

            # 调用模型的debug方法（通用接口）
            report = model_interface.debug(device, verbose=verbose)

            # 打印报告
            print(f"Model: {report.model_name}")
            for test in report.tests:
                status_symbol = "[OK]" if test.status == "OK" else ("[WARN]" if test.status == "WARN" else "[FAIL]")
                print(f"  {status_symbol} {test.component_name}: {test.message}")

            # 打印摘要
            print(f"\nSummary: {report.summary}")

            # 根据整体状态返回结果
            if report.overall_status == "PASSED":
                print("[OK] Model test PASSED")
                return True
            elif report.overall_status == "WARNING":
                print("[WARN] Model test completed with warnings")
                return True
            else:
                print("[FAIL] Model test FAILED")
                return False

        except AttributeError as e:
            # 如果模型没有实现debug方法，回退到基本测试
            if "'debug'" in str(e):
                print(f"Model does not implement debug() method, running basic tests...")
                return self._run_basic_model_test(config, device, verbose)
            raise
        except Exception as e:
            print(f"[FAIL] Model test FAILED: {e}")
            return False

    def _run_basic_model_test(self, config: dict, device: torch.device, verbose: bool) -> bool:
        """
        基本模型测试（当模型没有实现debug方法时的回退方案）
        """
        try:
            model_interface = create_model(config, device=str(device))
            model_interface.eval()

            # 基本推理测试
            test_sar = torch.rand(2, 1, 128, 128).to(device)
            with torch.no_grad():
                output = model_interface.get_output(test_sar, config)

            print(f"  Model output shape: {list(output.generated.shape)}")
            print(f"  Model output range: [{output.generated.min():.4f}, {output.generated.max():.4f}]")

            print("[OK] Basic model test PASSED")
            return True
        except Exception as e:
            print(f"[FAIL] Basic model test FAILED: {e}")
            return False

    def _run_numeric_test(self, config: dict, device: torch.device, verbose: bool) -> bool:
        """运行数值转换测试"""
        print("\n[TEST] Numeric Conversion Validation")
        print("-" * 70)

        try:
            # 创建模型
            model_interface = create_model(config, device=str(device))
            model_interface.eval()

            # 从配置获取SAR通道数
            sar_channels = config.get('data', {}).get('channels', {}).get('sar', {}).get('use', 3)

            # 创建测试输入
            sar = torch.rand(2, sar_channels, 128, 128).to(device)

            # 获取模型输出
            with torch.no_grad():
                output = model_interface.get_output(sar, config)
                generated = output.generated

            # 验证输出范围
            validate_range(generated, (0.0, 1.0), "Model output")

            # 测试uint8转换
            uint8_img = composite_to_uint8(generated, input_range=(0.0, 1.0))
            assert uint8_img.dtype == 'uint8', "uint8 conversion failed"
            assert uint8_img.max() <= 255, "uint8 max exceeded"
            assert uint8_img.min() >= 0, "uint8 min exceeded"

            if verbose:
                print(f"  Generated range: [{generated.min():.4f}, {generated.max():.4f}]")
                print(f"  uint8 image shape: {uint8_img.shape}")
                print(f"  uint8 range: [{uint8_img.min()}, {uint8_img.max()}]")

            print("[OK] Numeric conversion test PASSED")
            return True
        except Exception as e:
            print(f"[FAIL] Numeric conversion test FAILED: {e}")
            return False


if __name__ == "__main__":
    print("DebugCommand module loaded")
