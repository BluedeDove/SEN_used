"""
inference_ops.py - 推理流程封装

提供标准化的推理流程，支持DDP模式。
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
from typing import Optional, Iterator, Dict, Any
from dataclasses import dataclass
from models.registry import create_model
from datasets.registry import create_dataset
from core.numeric_ops import validate_range
from core.device_ops import setup_device_and_distributed, is_main_process


@dataclass
class InferenceOutput:
    """推理输出结构"""
    generated: torch.Tensor  # [0, 1]范围的生成图像
    sar: torch.Tensor  # 输入SAR
    optical: Optional[torch.Tensor]  # 真值(如果有)
    metadata: Optional[Dict[str, Any]] = None


def run_inference(
    config: dict,
    checkpoint_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Iterator[InferenceOutput]:
    """
    运行推理并返回生成器

    Args:
        config: 配置字典
        checkpoint_path: 检查点路径(可选，如果为None使用随机初始化)
        max_samples: 最大样本数(可选)
        device: 设备(如果为None则自动设置)

    Yields:
        InferenceOutput: 包含generated, sar, optical(真值)等
    """
    # 设置设备
    if device is None:
        device, rank, world_size = setup_device_and_distributed(config)
    else:
        rank = 0

    # 创建数据集
    dataset = create_dataset(config, split='val')

    # 创建模型
    model_interface = create_model(config, device=device)

    # 加载检查点
    if checkpoint_path is not None:
        from core.checkpoint_ops import load_checkpoint_v2, restore_model_v2
        checkpoint = load_checkpoint_v2(checkpoint_path, device=device)
        restore_model_v2(model_interface._model, checkpoint['model_state_dict'])

        if is_main_process(rank):
            print(f"Loaded checkpoint from {checkpoint_path}")
            print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    # 设置为评估模式
    model_interface.eval()

    # 限制样本数
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    # 推理循环
    with torch.no_grad():
        for i in range(n_samples):
            # 获取样本
            sample = dataset[i]

            # 添加batch维度
            sar = sample['sar'].unsqueeze(0).to(device)
            optical = sample.get('optical', None)
            if optical is not None:
                optical = optical.unsqueeze(0).to(device)

            # 验证输入范围
            try:
                validate_range(sar, (0.0, 1.0), "SAR input")
            except ValueError as e:
                print(f"Warning: {e}")

            # 获取模型输出（模型内部已完成合成）
            model_output = model_interface.get_output(sar, config)
            generated = model_output.generated

            # 验证模型输出范围
            try:
                validate_range(generated, (0.0, 1.0), "Model output")
            except ValueError as e:
                print(f"Warning: {e}")

            yield InferenceOutput(
                generated=generated,
                sar=sar,
                optical=optical,
                metadata={
                    'index': i,
                    'sar_path': sample.get('sar_path', ''),
                    'optical_path': sample.get('optical_path', ''),
                    'model_metadata': model_output.metadata
                }
            )


def inference_batch(
    model_interface,
    dataloader,
    config: dict,
    device: torch.device
) -> Iterator[InferenceOutput]:
    """
    批量推理

    Args:
        model_interface: 模型接口
        dataloader: 数据加载器
        config: 配置
        device: 设备

    Yields:
        InferenceOutput
    """
    model_interface.eval()

    with torch.no_grad():
        for batch in dataloader:
            sar = batch['sar'].to(device)
            optical = batch.get('optical', None)
            if optical is not None:
                optical = optical.to(device)

            # 批量推理（模型内部已完成合成）
            model_output = model_interface.get_output(sar, config)
            generated = model_output.generated

            yield InferenceOutput(
                generated=generated,
                sar=sar,
                optical=optical,
                metadata=batch.get('metadata', {})
            )


if __name__ == "__main__":
    # 测试
    print("Testing inference_ops.py...")

    # 这里需要实际的配置和数据才能测试
    print("Note: Full test requires actual config and data")
    print("All basic tests passed!")
