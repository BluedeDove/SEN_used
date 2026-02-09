"""
numeric_ops.py - 数值范围转换的基础函数

这是整个项目最核心的约定。所有数值范围转换必须通过本模块。
数据流约定:
    输入数据 -> [0, 1] 范围 (Dataset保证)
    模型输入 -> SRDM: SAR [0,1], Residual训练目标 [-1,1]
    模型输出 -> SRDM: 预测Residual [-1,1]
    合成处理 -> SAR_base + Residual -> 截断负数 -> 归一化 -> [0, 1]
    保存图像 -> uint8 [0, 255]
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
from typing import Tuple, Union, Optional


def validate_range(
    tensor: Union[torch.Tensor, np.ndarray],
    expected_range: Tuple[float, float],
    name: str = "tensor",
    tolerance: float = 1e-5
) -> bool:
    """
    验证张量是否在期望范围内

    Args:
        tensor: 输入张量
        expected_range: (min, max) 期望范围
        name: 张量名称(用于错误信息)
        tolerance: 容差范围

    Returns:
        bool: 是否在范围内

    Raises:
        ValueError: 如果超出范围
    """
    min_val, max_val = expected_range

    if isinstance(tensor, torch.Tensor):
        actual_min = tensor.min().item()
        actual_max = tensor.max().item()
    else:
        actual_min = tensor.min()
        actual_max = tensor.max()

    if actual_min < min_val - tolerance or actual_max > max_val + tolerance:
        raise ValueError(
            f"{name} 数值范围错误: "
            f"期望 [{min_val}, {max_val}], "
            f"实际 [{actual_min:.6f}, {actual_max:.6f}]"
        )

    return True


def normalize_to_range(
    tensor: torch.Tensor,
    from_range: Tuple[float, float],
    to_range: Tuple[float, float] = (0.0, 1.0)
) -> torch.Tensor:
    """
    将张量从一个范围映射到另一个范围

    Args:
        tensor: 输入张量
        from_range: (min, max) 当前范围
        to_range: (min, max) 目标范围

    Returns:
        映射后的张量
    """
    from_min, from_max = from_range
    to_min, to_max = to_range

    # 先归一化到 [0, 1]
    normalized = (tensor - from_min) / (from_max - from_min)
    # 再映射到目标范围
    return normalized * (to_max - to_min) + to_min


def denormalize(
    tensor: torch.Tensor,
    from_range: Tuple[float, float] = (0.0, 1.0),
    to_range: Tuple[float, float] = (0.0, 255.0)
) -> torch.Tensor:
    """
    将张量从归一化范围还原到原始范围

    Args:
        tensor: 输入张量
        from_range: (min, max) 当前范围
        to_range: (min, max) 目标范围

    Returns:
        还原后的张量
    """
    return normalize_to_range(tensor, from_range, to_range)


def clamp_and_normalize(
    tensor: torch.Tensor,
    clamp_min: float = 0.0,
    clamp_max: Optional[float] = None,
    target_range: Tuple[float, float] = (0.0, 1.0)
) -> torch.Tensor:
    """
    先截断再归一化到目标范围

    Args:
        tensor: 输入张量
        clamp_min: 截断下限
        clamp_max: 截断上限（如果为None则不限制上限）
        target_range: 目标范围

    Returns:
        处理后的张量
    """
    # 截断
    if clamp_max is not None:
        tensor = torch.clamp(tensor, clamp_min, clamp_max)
    else:
        tensor = torch.clamp(tensor, min=clamp_min)

    # 归一化
    t_min = tensor.min()
    t_max = tensor.max()

    if t_max > t_min:
        tensor = (tensor - t_min) / (t_max - t_min)
    else:
        # 如果所有值相同，返回中值
        tensor = torch.ones_like(tensor) * (target_range[0] + target_range[1]) / 2

    # 映射到目标范围
    return normalize_to_range(tensor, (0.0, 1.0), target_range)


def model_output_to_composite(
    model_output: torch.Tensor,
    base: torch.Tensor,
    output_range: Tuple[float, float] = (0.0, 1.0),
    clamp_negative: bool = True,
    normalize: bool = True
) -> torch.Tensor:
    """
    将模型输出与base合成为最终图像

    通用示例:
        model_output: 模型预测的变化量/残差，范围可能为[-1, 1]
        base: 基础图像（如输入图像），范围通常为[0, 1]
        composite = base + model_output -> 可能超出有效范围
        clamp_negative=True -> 截断负数部分
        normalize=True -> 归一化到目标范围

    适用场景:
        - 残差学习模型: 输出残差与基础图像相加
        - 变化量预测: 预测变化量叠加到基础图像

    Args:
        model_output: 模型输出张量（如残差、变化量）
        base: 基础图像张量
        output_range: 目标输出范围 (min, max)
        clamp_negative: 是否截断负数
        normalize: 是否归一化到output_range

    Returns:
        composite: 合成后的图像
    """
    # 合成
    composite = base + model_output

    # 截断负数（如果需要）
    if clamp_negative:
        composite = torch.clamp(composite, min=0.0)

    # 归一化（如果需要）
    if normalize:
        composite = clamp_and_normalize(
            composite,
            clamp_min=output_range[0],
            clamp_max=None,  # 不限制上限，因为已经截断了负数
            target_range=output_range
        )

    return composite


def composite_to_uint8(
    composite: Union[torch.Tensor, np.ndarray],
    input_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    将合成图像转换为uint8格式用于保存

    Args:
        composite: [0, 1]范围的图像张量或numpy数组
        input_range: 输入数值范围

    Returns:
        uint8图像 numpy array [0, 255]
    """
    # 转换为numpy
    if isinstance(composite, torch.Tensor):
        # 确保在CPU上
        composite = composite.detach().cpu().numpy()

    # 确保是 (H, W, C) 或 (H, W) 格式
    if composite.ndim == 4:
        # [B, C, H, W] -> 取第一个batch
        composite = composite[0]

    if composite.ndim == 3:
        # [C, H, W] -> [H, W, C]
        composite = np.transpose(composite, (1, 2, 0))

    # 归一化到 [0, 1]（如果需要）
    if input_range != (0.0, 1.0):
        from_min, from_max = input_range
        composite = (composite - from_min) / (from_max - from_min)

    # 截断到 [0, 1]
    composite = np.clip(composite, 0.0, 1.0)

    # 转换为 uint8 [0, 255]
    uint8_img = (composite * 255.0).astype(np.uint8)

    return uint8_img


def tensor_info(tensor: torch.Tensor, name: str = "tensor") -> str:
    """
    获取张量的详细信息字符串

    Args:
        tensor: 输入张量
        name: 张量名称

    Returns:
        信息字符串
    """
    return (
        f"{name}: shape={list(tensor.shape)}, "
        f"dtype={tensor.dtype}, "
        f"device={tensor.device}, "
        f"range=[{tensor.min().item():.4f}, {tensor.max().item():.4f}], "
        f"mean={tensor.mean().item():.4f}"
    )


if __name__ == "__main__":
    """
    测试数值操作模块

    测试场景1: 残差合成（适用于残差学习模型如SRDM）
    测试场景2: 直接输出（适用于端到端生成模型如标准DDPM）
    """
    print("=" * 60)
    print("Testing numeric_ops module")
    print("=" * 60)

    # ========== 测试1: validate_range ==========
    print("\n[TEST 1] validate_range")
    x = torch.rand(2, 3, 64, 64)
    try:
        validate_range(x, (0.0, 1.0), "test_tensor")
        print("  [OK] validate_range passed")
    except ValueError as e:
        print(f"  [FAIL] validate_range failed: {e}")

    # ========== 测试2: 残差合成场景 ==========
    print("\n[TEST 2] Residual composition (for residual-learning models)")
    # 场景：模型输出残差，与基础图像相加
    delta = torch.rand(2, 3, 64, 64) * 2 - 1  # 变化量 [-1, 1]
    base_image = torch.rand(2, 3, 64, 64)      # 基础图像 [0, 1]

    composite_residual = model_output_to_composite(
        delta, base_image, output_range=(0.0, 1.0),
        clamp_negative=True, normalize=True
    )
    print(f"  [OK] Residual composition: delta [{delta.min():.2f}, {delta.max():.2f}] + base [{base_image.min():.2f}, {base_image.max():.2f}]")
    print(f"       -> composite [{composite_residual.min():.4f}, {composite_residual.max():.4f}]")

    # ========== 测试3: 端到端生成场景 ==========
    print("\n[TEST 3] End-to-end generation (for direct generation models)")
    # 场景：模型直接输出图像，无需合成
    generated = torch.rand(2, 3, 64, 64) * 0.8 + 0.1  # 生成图像 [0.1, 0.9]

    # 对于这种模型，可以直接使用 clamp_and_normalize
    normalized = clamp_and_normalize(
        generated,
        clamp_min=0.0,
        clamp_max=1.0,
        target_range=(0.0, 1.0)
    )
    print(f"  [OK] Direct generation: [{generated.min():.2f}, {generated.max():.2f}] -> normalized [{normalized.min():.4f}, {normalized.max():.4f}]")

    # ========== 测试4: composite_to_uint8 ==========
    print("\n[TEST 4] composite_to_uint8")
    uint8_img = composite_to_uint8(composite_residual, input_range=(0.0, 1.0))
    print(f"  [OK] uint8 image: shape={uint8_img.shape}, dtype={uint8_img.dtype}, range=[{uint8_img.min()}, {uint8_img.max()}]")

    print("\n" + "=" * 60)
    print("All numeric_ops tests passed!")
    print("=" * 60)
