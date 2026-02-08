"""
debug.py - 调试命令

实现DEBUG模式，强制进行数值检验，确保数据流正确性。
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
from core.numeric_ops import validate_range, model_output_to_composite, composite_to_uint8
from core.ddp_validation import run_all_validations
from datasets.registry import create_dataset
from models.registry import create_model


@command('debug')
class DebugCommand(BaseCommand):
    """
    调试命令

    DEBUG模式强制进行数值检验，验证以下内容：
    1. 数据集输出范围
    2. 模型输出范围
    3. 合成流程
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
        print("SRDM DEBUG MODE")
        print("=" * 70)
        print("This mode performs strict numerical validation to ensure")
        print("that the data flow is correct before training.")
        print("=" * 70)

        # 如果只测试DDP，可以跳过配置加载
        if args.test_ddp and not (args.test_dataset or args.test_model):
            print("\n[TEST 0] DDP Code Validation")
            print("-" * 70)
            print("Note: Testing DDP logic with MockDDP (no multi-GPU required)")
            from core.ddp_validation import run_all_validations
            ddp_passed, ddp_messages = run_all_validations(verbose=args.verbose)
            for msg in ddp_messages:
                status_symbol = "[OK]" if "PASSED" in msg else "[FAIL]"
                print(f"  {status_symbol} {msg}")
            if ddp_passed:
                print("[OK] DDP code test PASSED")
                return 0
            else:
                print("[FAIL] DDP code test FAILED")
                return 1

        # 加载配置
        config_path = args.config
        if not Path(config_path).exists():
            # 尝试不同路径
            alt_path = Path(__file__).parent.parent / config_path
            if alt_path.exists():
                config_path = str(alt_path)

        if not Path(config_path).exists():
            print(f"Error: Config file not found: {args.config}")
            return 1

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 设置设备
        device, rank, world_size = setup_device_and_distributed(config)

        all_tests_passed = True

        # 测试 DDP 代码逻辑（与 config 相关时）
        if args.test_ddp:
            print("\n[TEST 0] DDP Code Validation")
            print("-" * 70)
            print("Note: Testing DDP logic with MockDDP (no multi-GPU required)")
            ddp_passed, ddp_messages = run_all_validations(verbose=args.verbose)
            for msg in ddp_messages:
                status_symbol = "[OK]" if "PASSED" in msg else "[FAIL]"
                print(f"  {status_symbol} {msg}")
            if ddp_passed:
                print("[OK] DDP code test PASSED")
            else:
                print("[FAIL] DDP code test FAILED")
                all_tests_passed = False

        # 测试1: 数据集数值范围
        if not args.test_model and not args.test_ddp:
            print("\n[TEST 1] Dataset Numerical Range Validation")
            print("-" * 70)
            try:
                self._test_dataset(config, device, args.verbose)
                print("[OK] Dataset test PASSED")
            except Exception as e:
                print(f"[FAIL] Dataset test FAILED: {e}")
                all_tests_passed = False

        # 测试2: 模型输出范围
        if not args.test_dataset and not args.test_ddp:
            print("\n[TEST 2] Model Output Range Validation")
            print("-" * 70)
            try:
                self._test_model_output(config, device, args.verbose)
                print("[OK] Model output test PASSED")
            except Exception as e:
                print(f"[FAIL] Model output test FAILED: {e}")
                all_tests_passed = False

        # 测试3: 合成流程
        if not args.test_dataset and not args.test_ddp:
            print("\n[TEST 3] Composite Flow Validation")
            print("-" * 70)
            try:
                self._test_composite_flow(config, device, args.verbose)
                print("[OK] Composite flow test PASSED")
            except Exception as e:
                print(f"[FAIL] Composite flow test FAILED: {e}")
                all_tests_passed = False

        # 总结
        print("\n" + "=" * 70)
        if all_tests_passed:
            print("ALL TESTS PASSED - Ready for training!")
        else:
            print("SOME TESTS FAILED - Please fix before training!")
        print("=" * 70)

        return 0 if all_tests_passed else 1

    def _test_dataset(self, config: dict, device: torch.device, verbose: bool):
        """测试数据集"""
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

        print(f"  Validated {min(5, len(dataset))} samples")

    def _test_model_output(self, config: dict, device: torch.device, verbose: bool):
        """测试模型输出"""
        # 创建模型
        model_interface = create_model(config, device=device)
        model_interface.eval()

        # 获取预期输出范围
        expected_range = model_interface.get_output_range()
        print(f"Expected model output range: {expected_range}")

        # ========== 测试1: 推理模式 (get_output) ==========
        print("\n  [Test 1] Inference mode (get_output)")
        # 创建测试输入 - 3通道SAR (根据SEN12数据集)
        test_sar = torch.rand(2, 3, 128, 128).to(device)

        with torch.no_grad():
            output = model_interface.get_output(test_sar, config)
            residual = output.generated

        if verbose:
            print(f"    Model output shape: {list(residual.shape)}")
            print(f"    Model output range: [{residual.min():.4f}, {residual.max():.4f}]")

        # 验证范围
        validate_range(residual, expected_range, "Model output (residual)")

        # 验证中间结果
        if 'sar_base' in output.intermediate:
            sar_base = output.intermediate['sar_base']
            if verbose:
                print(f"    SAR base range: [{sar_base.min():.4f}, {sar_base.max():.4f}]")
            validate_range(sar_base, (0.0, 1.0), "SAR base")

        # ========== 测试2: 训练模式前向传播 ==========
        print("\n  [Test 2] Training mode forward pass")
        model_interface.train()

        # 创建训练输入
        train_sar = torch.rand(2, 3, 256, 256).to(device)
        train_optical = torch.rand(2, 3, 256, 256).to(device)

        try:
            loss, loss_dict = model_interface(train_sar, train_optical, return_dict=True)
            print(f"    Training forward pass: OK")
            print(f"    Loss: {loss.item():.4f}")
            if verbose and loss_dict:
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        print(f"      {k}: {v.item():.4f}")
                    else:
                        print(f"      {k}: {v}")
        except Exception as e:
            print(f"    [FAIL] Training forward pass failed: {e}")
            raise

        print("\n  [OK] Model tests passed")

    def _test_composite_flow(self, config: dict, device: torch.device, verbose: bool):
        """测试合成流程"""
        # 创建测试数据 - 3通道SAR (SEN12格式)
        sar = torch.rand(2, 3, 128, 128).to(device)

        # 创建模型
        model_interface = create_model(config, device=device)
        model_interface.eval()

        # 获取模型输出
        with torch.no_grad():
            output = model_interface.get_output(sar, config)
            residual = output.generated
            sar_base = output.intermediate['sar_base']

        # 合成
        composite = model_output_to_composite(
            model_output=residual,
            base=sar_base,
            output_range=(0.0, 1.0),
            clamp_negative=True,
            normalize=True
        )

        if verbose:
            print(f"  Residual range: [{residual.min():.4f}, {residual.max():.4f}]")
            print(f"  SAR base range: [{sar_base.min():.4f}, {sar_base.max():.4f}]")
            print(f"  Composite range: [{composite.min():.4f}, {composite.max():.4f}]")

        # 验证合成结果范围
        validate_range(composite, (0.0, 1.0), "Composite output")

        # 测试转换为uint8
        uint8_img = composite_to_uint8(composite, input_range=(0.0, 1.0))
        assert uint8_img.dtype == 'uint8', "uint8 conversion failed"
        assert uint8_img.max() <= 255, "uint8 max exceeded"
        assert uint8_img.min() >= 0, "uint8 min exceeded"

        if verbose:
            print(f"  uint8 image shape: {uint8_img.shape}, dtype: {uint8_img.dtype}")
            print(f"  uint8 range: [{uint8_img.min()}, {uint8_img.max()}]")


if __name__ == "__main__":
    print("DebugCommand module loaded")
