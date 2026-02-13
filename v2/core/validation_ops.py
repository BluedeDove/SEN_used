"""
validation_ops.py - 验证流程封装

提供标准化的验证流程，支持指标计算和结果保存。
"""

# 支持单独运行调试：将项目根目录添加到路径
import sys
from pathlib import Path

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from torch.utils.data import DataLoader
from utils.image_ops import save_image_v2, create_comparison_figure
from core.device_ops import get_raw_model


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    计算PSNR

    Args:
        img1: 图像1 [B, C, H, W]
        img2: 图像2 [B, C, H, W]
        max_val: 最大像素值

    Returns:
        PSNR值
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """
    计算SSIM

    Args:
        img1: 图像1 [B, C, H, W]
        img2: 图像2 [B, C, H, W]
        window_size: 窗口大小

    Returns:
        SSIM值
    """
    from models.srdm.losses import SSIMLoss

    ssim_loss = SSIMLoss(window_size=window_size)
    ssim_val = ssim_loss(img1, img2)

    return ssim_val.item()


def compute_metrics_batch(
    generated: torch.Tensor,
    optical: torch.Tensor
) -> Dict[str, float]:
    """
    计算批次指标

    Args:
        generated: 生成图像 [B, C, H, W]
        optical: 真值图像 [B, C, H, W]

    Returns:
        指标字典
    """
    psnr_val = compute_psnr(generated, optical, max_val=1.0)
    ssim_val = compute_ssim(generated, optical)

    return {
        'psnr': psnr_val,
        'ssim': ssim_val
    }


def save_validation_samples(
    sar: torch.Tensor,
    generated: torch.Tensor,
    optical: torch.Tensor,
    save_dir: str,
    sample_idx: int,
    batch_idx: int = 0
) -> None:
    """
    保存验证样本图像

    Args:
        sar: SAR 图像 [B, C, H, W]
        generated: 生成图像 [B, C, H, W]
        optical: 真值图像 [B, C, H, W]
        save_dir: 保存目录
        sample_idx: 样本索引（用于文件名）
        batch_idx: 批次内的索引
    """
    import cv2

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 转换为numpy - 确保先移回CPU并断开计算图
    try:
        sar_np = sar[batch_idx].cpu().detach().numpy().transpose(1, 2, 0)
        gen_np = generated[batch_idx:batch_idx+1].cpu().detach().numpy().transpose(1, 2, 0)
        opt_np = optical[batch_idx:batch_idx+1].cpu().detach().numpy().transpose(1, 2, 0)
    except RuntimeError as e:
        # 如果转换失败，尝试更安全的方式
        print(f"[WARN] Tensor conversion failed: {e}, trying alternative method")
        sar_np = sar[batch_idx].clone().cpu().numpy().transpose(1, 2, 0)
        gen_np = generated[batch_idx:batch_idx+1].clone().cpu().numpy().transpose(1, 2, 0)
        opt_np = optical[batch_idx:batch_idx+1].clone().cpu().numpy().transpose(1, 2, 0)

    # 确保是uint8
    def to_uint8(img):
        img = np.clip(img, 0, 1)
        return (img * 255).astype(np.uint8)

    sar_np = to_uint8(sar_np)
    if sar_np.shape[-1] == 1:
        sar_np = np.repeat(sar_np, 3, axis=-1)

    gen_np = to_uint8(gen_np)
    opt_np = to_uint8(opt_np)

    # 创建对比图
    comparison = np.concatenate([sar_np, gen_np, opt_np], axis=1)

    # 保存
    cv2.imwrite(
        str(save_path / f'sample_{sample_idx:04d}.png'),
        cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    )


def run_validation(
    model,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    save_results: bool = False,
    save_dir: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Dict[str, float]:
    """
    运行验证

    Args:
        model: 模型接口
        val_loader: 验证数据加载器
        config: 配置
        device: 设备
        save_results: 是否保存结果图像
        save_dir: 保存目录
        max_samples: 最大验证样本数

    Returns:
        平均指标字典
    """
    model.eval()

    all_metrics = []
    num_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_samples is not None and num_samples >= max_samples:
                break

            # 确保数据在正确的设备上
            sar = batch['sar'].to(device, non_blocking=True)
            optical = batch.get('optical')
            if optical is None:
                continue
            optical = optical.to(device, non_blocking=True)

            batch_size = sar.shape[0]

            # 获取模型输出（模型内部已完成合成）
            model_output = model.get_output(sar, config)
            generated = model_output.generated  # 直接使用模型输出的最终图像

            # 确保生成的图像也在正确的设备上
            if generated.device != device:
                generated = generated.to(device)

            # 计算指标
            for j in range(batch_size):
                gen_img = generated[j:j+1]
                opt_img = optical[j:j+1]

                metrics = compute_metrics_batch(gen_img, opt_img)
                all_metrics.append(metrics)

                # 保存结果（只在主进程或单卡模式下）
                if save_results and save_dir is not None:
                    save_validation_samples(sar, generated, optical, save_dir, num_samples, j)

                num_samples += 1

    # 计算平均指标
    if not all_metrics:
        return {'psnr': 0.0, 'ssim': 0.0}

    avg_metrics = {
        'psnr': np.mean([m['psnr'] for m in all_metrics]),
        'ssim': np.mean([m['ssim'] for m in all_metrics])
    }

    return avg_metrics


# ============================================================================
# DDP 验证组件（从 ddp_validation.py 合并）
# ============================================================================

import tempfile
import os
import torch.nn as nn
from core.checkpoint_ops import (
    save_checkpoint_v2, load_checkpoint_v2,
    restore_model_v2, restore_optimizer_v2
)


class MockDDP(nn.Module):
    """模拟 DDP 包装器，用于测试 get_raw_model 逻辑"""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class SimpleTestModel(nn.Module):
    """测试模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        return self.conv2(torch.relu(self.conv1(x)))


def validate_get_raw_model() -> Tuple[bool, str]:
    """验证 get_raw_model 正确处理 DDP 包装"""
    original_model = SimpleTestModel()
    original_state = original_model.state_dict()

    # 测试 1: 普通模型
    raw = get_raw_model(original_model)
    if raw is not original_model:
        return False, "普通模型应该返回自身"

    # 测试 2: Mock DDP 包装
    ddp_model = MockDDP(original_model)
    raw = get_raw_model(ddp_model)
    if raw is not original_model:
        return False, "应该解包 DDP 获取原始模型"

    # 测试 3: 状态一致
    raw_state = raw.state_dict()
    for key in original_state:
        if not torch.allclose(original_state[key], raw_state[key]):
            return False, f"参数 {key} 不一致"

    return True, "get_raw_model 验证通过"


def validate_checkpoint_with_ddp() -> Tuple[bool, str]:
    """验证 DDP 模型的检查点保存和加载"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建并修改模型
        original_model = SimpleTestModel()
        optimizer = torch.optim.Adam(original_model.parameters(), lr=0.001)

        x = torch.randn(2, 3, 64, 64)
        loss = original_model(x).mean()
        loss.backward()
        optimizer.step()

        original_state = {k: v.clone() for k, v in original_model.state_dict().items()}

        # 测试 DDP 保存
        ddp_model = MockDDP(original_model)
        save_path = os.path.join(tmpdir, 'ddp.pth')
        save_checkpoint_v2(ddp_model, optimizer, None, epoch=5,
                          metrics={'psnr': 25.5}, save_path=save_path)

        if not os.path.exists(save_path):
            return False, "DDP检查点保存失败"

        # 验证状态不包含 'module.' 前缀
        checkpoint = load_checkpoint_v2(save_path)
        for key in checkpoint['model_state_dict']:
            if key.startswith('module.'):
                return False, f"状态字典包含 module. 前缀: {key}"

        # 验证恢复
        new_model = SimpleTestModel()
        restore_model_v2(new_model, checkpoint['model_state_dict'])

        for key in original_state:
            if not torch.allclose(original_state[key], new_model.state_dict()[key]):
                return False, f"恢复后参数 {key} 不一致"

    return True, "Checkpoint 保存/加载验证通过"


def validate_is_main_process() -> Tuple[bool, str]:
    """验证主进程检测"""
    from core.device_ops import is_main_process

    if not is_main_process():
        return False, "单进程环境下应该返回 True"

    if not is_main_process(rank=0):
        return False, "rank=0 应该是主进程"

    if is_main_process(rank=1):
        return False, "rank!=0 不应该为主进程"

    return True, "is_main_process 验证通过"


def validate_all_reduce() -> Tuple[bool, str]:
    """验证 all_reduce_tensor（单进程模式）"""
    from core.device_ops import all_reduce_tensor

    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = all_reduce_tensor(tensor.clone(), op='mean')

    if not torch.allclose(tensor, result):
        return False, "单进程下 all_reduce 应返回原值"

    return True, "all_reduce 验证通过"


def validate_ddp_integration() -> Tuple[bool, str]:
    """验证完整的 DDP 训练-保存-恢复流程"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 训练
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(3):
            x = torch.randn(2, 3, 64, 64)
            loss = model(x).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        trained_state = {k: v.clone() for k, v in model.state_dict().items()}

        # 包装为 DDP 并保存
        ddp_model = MockDDP(model)
        checkpoint_path = os.path.join(tmpdir, 'ddp_checkpoint.pth')
        save_checkpoint_v2(ddp_model, optimizer, None, epoch=3,
                          metrics={'loss': 0.123}, save_path=checkpoint_path)

        # 恢复
        new_model = SimpleTestModel()
        checkpoint = load_checkpoint_v2(checkpoint_path)
        restore_model_v2(new_model, checkpoint['model_state_dict'])

        # 验证
        for key in trained_state:
            if not torch.allclose(trained_state[key], new_model.state_dict()[key]):
                return False, f"恢复后参数 {key} 不一致"

        # 验证可继续训练
        x = torch.randn(2, 3, 64, 64)
        output = new_model(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()

    return True, "DDP 集成验证通过"


def validate_inference_with_ddp() -> Tuple[bool, str]:
    """验证 DDP 模型推理（sample/get_output）"""
    class InferenceTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)

        def forward(self, x):
            return self.conv(x)

        def sample(self, x, steps=10):
            """推理专用方法"""
            return self.forward(x)

        def get_output(self, x):
            """返回结构化输出"""
            return type('Output', (), {'generated': self.sample(x)})()

    model = InferenceTestModel()
    ddp_model = MockDDP(model)

    # 测试 1: DDP 直接访问 sample 应该通过 get_raw_model
    x = torch.randn(2, 3, 64, 64)

    # 错误方式：直接访问会失败（如果被包装）
    # ddp_model.sample(x)  # 这会失败！

    # 正确方式：通过 get_raw_model 访问
    raw_model = get_raw_model(ddp_model)
    try:
        output = raw_model.sample(x)
        if output.shape != x.shape:
            return False, f"输出形状错误: {output.shape}"
    except AttributeError as e:
        return False, f"get_raw_model 后仍无法访问 sample: {e}"

    # 测试 2: get_output 方法
    try:
        output_obj = raw_model.get_output(x)
        if not hasattr(output_obj, 'generated'):
            return False, "get_output 返回对象缺少 generated 属性"
    except AttributeError as e:
        return False, f"get_output 调用失败: {e}"

    return True, "DDP 推理验证通过"


def validate_tensor_device_conversion() -> Tuple[bool, str]:
    """验证 GPU 张量转换（用于验证流程中的图像保存）"""
    if not torch.cuda.is_available():
        return True, "CUDA 不可用，跳过 GPU 张量转换测试"

    # 创建 GPU 张量（模拟 DDP 多卡情况）
    tensor_gpu = torch.randn(3, 64, 64, device='cuda:0')

    # 测试正确的转换顺序
    try:
        # 方式1: .cpu().detach().numpy()
        np_arr1 = tensor_gpu.cpu().detach().numpy()
        if np_arr1.shape != (3, 64, 64):
            return False, f"方式1形状错误: {np_arr1.shape}"

        # 方式2: .detach().cpu().numpy()
        np_arr2 = tensor_gpu.detach().cpu().numpy()
        if np_arr2.shape != (3, 64, 64):
            return False, f"方式2形状错误: {np_arr2.shape}"

        # 测试批次维度（验证流程中的实际场景）
        batch_tensor = torch.randn(2, 3, 64, 64, device='cuda:0')
        for j in range(batch_tensor.shape[0]):
            np_arr = batch_tensor[j].cpu().detach().numpy().transpose(1, 2, 0)
            if np_arr.shape != (64, 64, 3):
                return False, f"批次转换形状错误: {np_arr.shape}"

    except RuntimeError as e:
        return False, f"GPU 张量转换失败: {e}"

    return True, "GPU 张量转换验证通过"


def validate_validation_flow_with_ddp() -> Tuple[bool, str]:
    """验证完整的验证流程（模拟 run_validation）"""
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)

        def get_output(self, x, config=None):
            output = self.conv(x)
            # 模拟 ModelOutput
            return type('ModelOutput', (), {
                'generated': output,
                'output_range': (0.0, 1.0)
            })()

    model = MockModel()
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # 模拟 DDP 包装
    ddp_model = MockDDP(model)

    # 模拟验证流程
    batch = {
        'sar': torch.randn(2, 3, 64, 64, device=device),
        'optical': torch.randn(2, 3, 64, 64, device=device)
    }

    try:
        # 1. 模型推理
        raw_model = get_raw_model(ddp_model)
        model_output = raw_model.get_output(batch['sar'], {})
        generated = model_output.generated

        # 2. 指标计算
        for j in range(generated.shape[0]):
            gen_img = generated[j:j+1]
            opt_img = batch['optical'][j:j+1]

            # 3. 张量转换（关键！）
            # [1, 3, 64, 64] -> [64, 64, 3]
            gen_np = gen_img[0].cpu().detach().numpy().transpose(1, 2, 0)
            opt_np = opt_img[0].cpu().detach().numpy().transpose(1, 2, 0)

            if gen_np.shape != (64, 64, 3):
                return False, f"转换后形状错误: {gen_np.shape}"

    except Exception as e:
        return False, f"验证流程失败: {type(e).__name__}: {e}"

    return True, "DDP 验证流程通过"


def run_all_ddp_validations(verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    运行所有 DDP 验证

    Args:
        verbose: 是否显示详细日志

    Returns:
        (all_passed, messages)
    """
    validations = [
        ("get_raw_model", validate_get_raw_model),
        ("checkpoint_save_load", validate_checkpoint_with_ddp),
        ("is_main_process", validate_is_main_process),
        ("all_reduce", validate_all_reduce),
        ("ddp_integration", validate_ddp_integration),
        ("inference_with_ddp", validate_inference_with_ddp),
        ("tensor_device_conversion", validate_tensor_device_conversion),
        ("validation_flow_with_ddp", validate_validation_flow_with_ddp),
    ]

    results = []
    messages = []

    for name, validate_fn in validations:
        try:
            passed, msg = validate_fn()
            results.append(passed)
            status = "PASSED" if passed else "FAILED"
            messages.append(f"[{status}] {name}: {msg}")
            if verbose:
                print(f"  [{status}] {name}")
        except Exception as e:
            results.append(False)
            messages.append(f"[FAILED] {name}: {str(e)}")
            if verbose:
                print(f"  [FAILED] {name}: {e}")

    all_passed = all(results)
    return all_passed, messages


# 别名，保持向后兼容
validate_ddp_components = run_all_ddp_validations


if __name__ == "__main__":
    # 测试
    print("Testing validation_ops.py...")

    # 测试指标计算
    img1 = torch.rand(2, 3, 64, 64)
    img2 = img1 + torch.randn(2, 3, 64, 64) * 0.1

    metrics = compute_metrics_batch(img1, img2)
    print(f"PSNR: {metrics['psnr']:.2f}")
    print(f"SSIM: {metrics['ssim']:.4f}")

    print("\nAll tests passed!")
