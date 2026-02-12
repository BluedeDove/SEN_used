"""
debug.py - SRDM模型特有的调试逻辑

包含SRDM特定的测试：
- 扩散过程验证
- 残差生成验证
- SAR编码器验证
- 合成流程验证
- 训练/推理一致性
"""

import torch
from typing import List
from models.base import ModelDebugInfo, ModelDebugReport
from core.numeric_ops import validate_range


class SRDMDebugger:
    """
    SRDM模型调试器

    封装SRDM特有的调试逻辑，从通用调试命令中分离。
    """

    def __init__(self, model_interface):
        """
        初始化调试器

        Args:
            model_interface: SRDMInterface实例
        """
        self.interface = model_interface
        self.model = model_interface._model
        self.device = model_interface._device
        self.config = model_interface.config

    def _get_sar_channels(self) -> int:
        """
        从配置获取SAR通道数

        Returns:
            SAR通道数（默认为3，适配SEN1-2数据集）
        """
        return self.config.get('data', {}).get('channels', {}).get('sar', {}).get('use', 3)

    def run_all_tests(self, verbose: bool = False) -> ModelDebugReport:
        """
        运行所有SRDM特有测试

        Args:
            verbose: 是否显示详细信息

        Returns:
            ModelDebugReport: 调试报告
        """
        tests = []

        # 测试1: SAR编码器
        tests.append(self._test_sar_encoder(verbose))

        # 测试2: 扩散采样（DDIM）
        tests.append(self._test_diffusion_sampling(verbose))

        # 测试3: 残差生成
        tests.append(self._test_residual_generation(verbose))

        # 测试4: 合成流程
        tests.append(self._test_composite_flow(verbose))

        # 测试5: 训练模式
        tests.append(self._test_training_mode(verbose))

        # 确定整体状态
        has_fail = any(t.status == "FAIL" for t in tests)
        has_warn = any(t.status == "WARN" for t in tests)
        overall = "FAILED" if has_fail else ("WARNING" if has_warn else "PASSED")

        # 生成摘要
        passed = sum(1 for t in tests if t.status == "OK")
        failed = sum(1 for t in tests if t.status == "FAIL")
        warnings = sum(1 for t in tests if t.status == "WARN")

        summary = f"SRDM tests: {passed} passed, {failed} failed, {warnings} warnings"

        return ModelDebugReport(
            model_name="SRDM",
            overall_status=overall,
            tests=tests,
            summary=summary
        )

    def _test_sar_encoder(self, verbose: bool) -> ModelDebugInfo:
        """测试SAR编码器"""
        try:
            # 从配置获取SAR通道数
            sar_channels = self._get_sar_channels()
            sar = torch.rand(2, sar_channels, 128, 128).to(self.device)

            with torch.no_grad():
                sar_base, sar_features, _ = self.model.sar_encoder(sar)

            # 验证输出
            validate_range(sar_base, (0.0, 1.0), "SAR base")

            if verbose:
                msg = (f"SAR base shape: {list(sar_base.shape)}, "
                       f"range: [{sar_base.min():.4f}, {sar_base.max():.4f}]")
            else:
                msg = "SAR encoder output valid"

            return ModelDebugInfo(
                component_name="SAR Encoder",
                status="OK",
                message=msg,
                values={
                    "base_shape": list(sar_base.shape),
                    "base_range": [float(sar_base.min()), float(sar_base.max())],
                    "features_count": len(sar_features) if isinstance(sar_features, (list, tuple)) else 1
                }
            )
        except Exception as e:
            return ModelDebugInfo(
                component_name="SAR Encoder",
                status="FAIL",
                message=f"SAR encoder test failed: {str(e)}"
            )

    def _test_diffusion_sampling(self, verbose: bool) -> ModelDebugInfo:
        """测试扩散采样"""
        try:
            # 从配置获取SAR通道数
            sar_channels = self._get_sar_channels()
            sar = torch.rand(2, sar_channels, 64, 64).to(self.device)

            with torch.no_grad():
                # 测试DDIM采样
                residual_ddim = self.model.ddim_sample(
                    sar, steps=10, eta=0.0, return_residual_only=True
                )

                # 测试DDPM采样
                residual_ddpm = self.model.sample(
                    sar, steps=10, return_residual_only=True
                )

            if verbose:
                msg = (f"DDIM residual range: [{residual_ddim.min():.4f}, {residual_ddim.max():.4f}], "
                       f"DDPM residual range: [{residual_ddpm.min():.4f}, {residual_ddpm.max():.4f}]")
            else:
                msg = "Diffusion sampling valid"

            return ModelDebugInfo(
                component_name="Diffusion Sampling",
                status="OK",
                message=msg,
                values={
                    "ddim_range": [float(residual_ddim.min()), float(residual_ddim.max())],
                    "ddpm_range": [float(residual_ddpm.min()), float(residual_ddpm.max())],
                    "output_shape": list(residual_ddim.shape)
                }
            )
        except Exception as e:
            return ModelDebugInfo(
                component_name="Diffusion Sampling",
                status="FAIL",
                message=f"Diffusion sampling test failed: {str(e)}"
            )

    def _test_residual_generation(self, verbose: bool) -> ModelDebugInfo:
        """测试残差生成"""
        try:
            # 从配置获取SAR通道数
            sar_channels = self._get_sar_channels()
            sar = torch.rand(2, sar_channels, 64, 64).to(self.device)

            with torch.no_grad():
                output = self.interface.get_output(sar, self.config)
                residual = output.intermediate.get('residual')

            if residual is None:
                return ModelDebugInfo(
                    component_name="Residual Generation",
                    status="WARN",
                    message="Residual not found in intermediate outputs"
                )

            # 验证残差范围（通常在[-1, 1]附近）
            res_min, res_max = float(residual.min()), float(residual.max())

            status = "OK"
            message = f"Residual range: [{res_min:.4f}, {res_max:.4f}]"

            # 如果残差范围异常，发出警告
            if abs(res_min) > 5 or abs(res_max) > 5:
                status = "WARN"
                message += " (unusually large range)"

            if verbose:
                message += f", shape: {list(residual.shape)}"

            return ModelDebugInfo(
                component_name="Residual Generation",
                status=status,
                message=message,
                values={
                    "residual_range": [res_min, res_max],
                    "residual_shape": list(residual.shape)
                }
            )
        except Exception as e:
            return ModelDebugInfo(
                component_name="Residual Generation",
                status="FAIL",
                message=f"Residual generation test failed: {str(e)}"
            )

    def _test_composite_flow(self, verbose: bool) -> ModelDebugInfo:
        """测试合成流程"""
        try:
            # 从配置获取SAR通道数
            sar_channels = self._get_sar_channels()
            sar = torch.rand(2, sar_channels, 64, 64).to(self.device)

            with torch.no_grad():
                output = self.interface.get_output(sar, self.config)
                generated = output.generated

            # 验证最终输出范围
            validate_range(generated, (0.0, 1.0), "Generated output")

            gen_min, gen_max = float(generated.min()), float(generated.max())

            if verbose:
                msg = f"Generated image range: [{gen_min:.4f}, {gen_max:.4f}], shape: {list(generated.shape)}"
            else:
                msg = f"Composite output valid (range: [{gen_min:.4f}, {gen_max:.4f}])"

            return ModelDebugInfo(
                component_name="Composite Flow",
                status="OK",
                message=msg,
                values={
                    "output_range": [gen_min, gen_max],
                    "output_shape": list(generated.shape)
                }
            )
        except Exception as e:
            return ModelDebugInfo(
                component_name="Composite Flow",
                status="FAIL",
                message=f"Composite flow test failed: {str(e)}"
            )

    def _test_training_mode(self, verbose: bool) -> ModelDebugInfo:
        """测试训练模式"""
        try:
            # 从配置获取通道数
            sar_channels = self._get_sar_channels()
            opt_channels = self.config.get('data', {}).get('channels', {}).get('optical', {}).get('use', 3)

            train_sar = torch.rand(2, sar_channels, 64, 64).to(self.device)
            train_optical = torch.rand(2, opt_channels, 64, 64).to(self.device)

            self.interface.train()

            loss, loss_dict = self.interface(train_sar, train_optical, return_dict=True)

            loss_value = float(loss.item())

            if verbose:
                msg = f"Training loss: {loss_value:.4f}"
                if loss_dict:
                    details = ", ".join([f"{k}: {v:.4f}" if isinstance(v, torch.Tensor) else f"{k}: {v}"
                                         for k, v in list(loss_dict.items())[:3]])
                    msg += f" ({details})"
            else:
                msg = f"Training mode valid (loss: {loss_value:.4f})"

            return ModelDebugInfo(
                component_name="Training Mode",
                status="OK",
                message=msg,
                values={
                    "loss": loss_value,
                    "loss_components": list(loss_dict.keys()) if loss_dict else []
                }
            )
        except Exception as e:
            return ModelDebugInfo(
                component_name="Training Mode",
                status="FAIL",
                message=f"Training mode test failed: {str(e)}"
            )


def create_srdm_debugger(model_interface) -> SRDMDebugger:
    """
    工厂函数：创建SRDM调试器

    Args:
        model_interface: SRDMInterface实例

    Returns:
        SRDMDebugger: 调试器实例
    """
    return SRDMDebugger(model_interface)


if __name__ == "__main__":
    """单独运行测试"""
    import sys
    from pathlib import Path

    # 添加项目根目录到路径
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

    print("=" * 70)
    print("SRDM Debug Module - Standalone Test")
    print("=" * 70)

    from models.registry import create_model

    # 测试配置
    config = {
        'model': {'type': 'srdm'},
        'srdm': {
            'base_ch': 32,
            'ch_mults': [1, 2, 4],
            'num_blocks': 1,
            'time_emb_dim': 128,
            'dropout': 0.1,
            'num_heads': 4
        },
        'diffusion': {
            'num_timesteps': 1000,
            'sampling': {
                'use_ddim': True,
                'ddim_steps': 10,
                'ddim_eta': 0.0
            }
        }
    }

    print("\n[1/3] Creating SRDM model...")
    try:
        model = create_model(config, device='cpu')
        print(f"  [OK] Model created with {model.count_parameters()['total']:,} parameters")
    except Exception as e:
        print(f"  [FAIL] Failed to create model: {e}")
        sys.exit(1)

    print("\n[2/3] Creating SRDM debugger...")
    try:
        debugger = create_srdm_debugger(model)
        print("  [OK] Debugger created")
    except Exception as e:
        print(f"  [FAIL] Failed to create debugger: {e}")
        sys.exit(1)

    print("\n[3/3] Running all debug tests (verbose mode)...")
    print("-" * 70)
    try:
        report = debugger.run_all_tests(verbose=True)

        print("\n" + "=" * 70)
        print(f"Test Report: {report.model_name}")
        print(f"Overall Status: {report.overall_status}")
        print("=" * 70)

        for test in report.tests:
            status_symbol = "[OK]" if test.status == "OK" else ("[WARN]" if test.status == "WARN" else "[FAIL]")
            print(f"{status_symbol} {test.component_name}")
            print(f"      {test.message}")
            if test.values:
                values_str = ", ".join([f"{k}={v}" for k, v in list(test.values.items())[:3]])
                print(f"      Values: {values_str}")

        print("\n" + "-" * 70)
        print(f"Summary: {report.summary}")

        if report.overall_status == "PASSED":
            print("\n[OK] All tests passed!")
            sys.exit(0)
        elif report.overall_status == "WARNING":
            print("\n[WARN] Tests completed with warnings")
            sys.exit(0)
        else:
            print("\n[FAIL] Some tests failed!")
            sys.exit(1)

    except Exception as e:
        print(f"  [FAIL] Debug tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
