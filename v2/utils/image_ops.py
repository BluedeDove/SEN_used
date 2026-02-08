"""
image_ops.py - 基础图像操作

提供图像转换、保存、可视化等功能。
"""

# 支持单独运行调试：将项目根目录添加到路径
import sys
from pathlib import Path

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

import numpy as np
import torch
from typing import Union, Optional, Dict, List, Tuple


def tensor_to_numpy(
    tensor: torch.Tensor,
    channel_order: str = 'hwc'
) -> np.ndarray:
    """
    将tensor转换为numpy array

    Args:
        tensor: [B, C, H, W] 或 [C, H, W] 或 [H, W]
        channel_order: 'hwc' 或 'chw'

    Returns:
        numpy array
    """
    # 确保在CPU上
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # 转换为numpy
    array = tensor.detach().numpy()

    # 处理batch维度
    if array.ndim == 4:
        # [B, C, H, W] -> 取第一个batch
        array = array[0]

    # 调整通道顺序
    if channel_order == 'hwc':
        if array.ndim == 3:
            # [C, H, W] -> [H, W, C]
            array = np.transpose(array, (1, 2, 0))
    # 否则保持 'chw'

    return array


def numpy_to_tensor(
    array: np.ndarray,
    channel_order: str = 'chw',
    device: str = 'cpu'
) -> torch.Tensor:
    """
    将numpy array转换为tensor

    Args:
        array: numpy array [H, W, C] 或 [H, W]
        channel_order: 目标channel顺序 'chw' 或 'hw'
        device: 目标设备

    Returns:
        torch.Tensor
    """
    # 添加batch维度（如果需要）
    if array.ndim == 2:
        # [H, W] -> [1, 1, H, W]
        array = array[np.newaxis, np.newaxis, ...]
    elif array.ndim == 3:
        if channel_order == 'chw':
            # [H, W, C] -> [C, H, W]
            array = np.transpose(array, (2, 0, 1))
        # [C, H, W] -> [1, C, H, W]
        array = array[np.newaxis, ...]

    # 转换为tensor
    tensor = torch.from_numpy(array).float().to(device)

    return tensor


def normalize_for_display(
    image: np.ndarray,
    input_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    将图像归一化到 [0, 255] 用于显示

    Args:
        image: 输入图像
        input_range: 输入数值范围（如果为None则自动检测）

    Returns:
        uint8图像 [0, 255]
    """
    # 自动检测范围
    if input_range is None:
        input_range = (image.min(), image.max())

    min_val, max_val = input_range

    # 归一化到 [0, 1]
    if max_val > min_val:
        normalized = (image - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(image)

    # 转换到 [0, 255]
    display_img = (normalized * 255).astype(np.uint8)

    return display_img


def create_comparison_figure(
    images_dict: Dict[str, np.ndarray],
    titles: Optional[Dict[str, str]] = None,
    layout: str = 'horizontal',
    figsize: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    创建对比图

    Args:
        images_dict: {'SAR': sar_img, 'Generated': gen_img, 'GT': gt_img}
                     每个图像应该是 numpy array (H, W, C) 或 (H, W)
        titles: 标题字典（如果为None使用images_dict的key）
        layout: 'horizontal' 或 'vertical' 或 'grid'
        figsize: 输出图像尺寸 (H, W, C)

    Returns:
        合成后的numpy图像
    """
    import cv2

    # 获取图像列表
    images = list(images_dict.values())
    n_images = len(images)

    if n_images == 0:
        raise ValueError("No images provided")

    # 标准化图像尺寸和通道
    processed_images = []
    for img in images:
        # 确保是 numpy
        if isinstance(img, torch.Tensor):
            img = tensor_to_numpy(img, channel_order='hwc')

        # 确保是 uint8
        if img.dtype != np.uint8:
            img = normalize_for_display(img)

        # 确保有3个通道
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        processed_images.append(img)

    # 获取最大尺寸
    max_h = max(img.shape[0] for img in processed_images)
    max_w = max(img.shape[1] for img in processed_images)

    # 统一尺寸
    standardized_images = []
    for img in processed_images:
        if img.shape[0] != max_h or img.shape[1] != max_w:
            img = cv2.resize(img, (max_w, max_h))
        standardized_images.append(img)

    # 创建布局
    if layout == 'horizontal':
        # 水平排列
        result = np.concatenate(standardized_images, axis=1)
    elif layout == 'vertical':
        # 垂直排列
        result = np.concatenate(standardized_images, axis=0)
    elif layout == 'grid':
        # 网格排列
        n_cols = int(np.ceil(np.sqrt(n_images)))
        n_rows = int(np.ceil(n_images / n_cols))

        rows = []
        for i in range(n_rows):
            row_images = []
            for j in range(n_cols):
                idx = i * n_cols + j
                if idx < n_images:
                    row_images.append(standardized_images[idx])
                else:
                    # 填充空白
                    row_images.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))
            rows.append(np.concatenate(row_images, axis=1))
        result = np.concatenate(rows, axis=0)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # 添加标题（可选）
    if titles:
        title_list = [titles.get(k, k) for k in images_dict.keys()]
        # 这里可以添加标题绘制逻辑

    return result


def create_grid(
    images: List[np.ndarray],
    n_cols: int = 4,
    padding: int = 2
) -> np.ndarray:
    """
    创建图像网格

    Args:
        images: 图像列表
        n_cols: 列数
        padding: 间距像素

    Returns:
        网格图像
    """
    import cv2

    if not images:
        raise ValueError("No images provided")

    n_images = len(images)
    n_rows = int(np.ceil(n_images / n_cols))

    # 标准化图像
    processed_images = []
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    for img in images:
        if isinstance(img, torch.Tensor):
            img = tensor_to_numpy(img, channel_order='hwc')

        if img.dtype != np.uint8:
            img = normalize_for_display(img)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        if img.shape[0] != max_h or img.shape[1] != max_w:
            img = cv2.resize(img, (max_w, max_h))

        processed_images.append(img)

    # 创建网格
    grid_h = n_rows * max_h + (n_rows + 1) * padding
    grid_w = n_cols * max_w + (n_cols + 1) * padding

    # 白色背景
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    # 填充图像
    for idx, img in enumerate(processed_images):
        row = idx // n_cols
        col = idx % n_cols

        y = row * max_h + (row + 1) * padding
        x = col * max_w + (col + 1) * padding

        grid[y:y+max_h, x:x+max_w] = img

    return grid


def save_image_v2(
    image_array: np.ndarray,
    save_path: str,
    create_dir: bool = True
):
    """
    保存图像

    Args:
        image_array: numpy array (H, W, C) 或 (H, W)
        save_path: 保存路径
        create_dir: 是否自动创建目录
    """
    import cv2

    save_path = Path(save_path)

    if create_dir:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # 确保是 numpy
    if isinstance(image_array, torch.Tensor):
        image_array = tensor_to_numpy(image_array, channel_order='hwc')

    # 确保是 uint8
    if image_array.dtype != np.uint8:
        image_array = normalize_for_display(image_array)

    # 处理通道
    if image_array.ndim == 3:
        if image_array.shape[-1] == 3:
            # RGB -> BGR for OpenCV
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        elif image_array.shape[-1] == 1:
            image_array = image_array.squeeze(-1)

    # 保存
    cv2.imwrite(str(save_path), image_array)


def load_image_v2(
    image_path: str,
    to_rgb: bool = True
) -> np.ndarray:
    """
    加载图像

    Args:
        image_path: 图像路径
        to_rgb: 是否转换为RGB

    Returns:
        numpy array (H, W, C)
    """
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # BGR -> RGB
    if to_rgb and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


if __name__ == "__main__":
    # 测试
    print("Testing image_ops...")

    # 创建测试图像
    test_img = np.random.rand(64, 64, 3).astype(np.float32)

    # 测试 tensor_to_numpy
    tensor = numpy_to_tensor(test_img, channel_order='chw')
    print(f"Tensor shape: {tensor.shape}")

    numpy_back = tensor_to_numpy(tensor, channel_order='hwc')
    print(f"Numpy shape: {numpy_back.shape}")

    # 测试 normalize_for_display
    display_img = normalize_for_display(test_img)
    print(f"Display image dtype: {display_img.dtype}")

    # 测试 create_comparison_figure
    images = {
        'Image 1': np.random.rand(64, 64, 3),
        'Image 2': np.random.rand(64, 64, 3),
        'Image 3': np.random.rand(64, 64, 3)
    }
    comparison = create_comparison_figure(images, layout='horizontal')
    print(f"Comparison shape: {comparison.shape}")

    print("All tests passed!")
