"""
device_ops.py - 设备管理和分布式训练支持

处理CUDA设备、DDP分布式训练、随机种子设置等。
"""

# 支持单独运行调试：将项目根目录添加到路径
import sys
from pathlib import Path

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

import os
import random
import numpy as np
import torch
import torch.distributed as dist
from typing import Optional, Tuple, Dict, Any


def setup_device_and_distributed(
    config: Optional[Dict[str, Any]] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None
) -> Tuple[torch.device, int, int]:
    """
    初始化设备和分布式环境

    必须在 import torch 之后最早阶段调用。

    Args:
        config: 配置字典（可选）
        rank: 当前进程rank（可选，从环境变量读取）
        world_size: 总进程数（可选，从环境变量读取）

    Returns:
        (device, rank, world_size)
    """
    # 设置NCCL环境变量（必须在分布式初始化之前）
    os.environ.setdefault('NCCL_SOCKET_FAMILY', 'AF_INET')  # 强制IPv4
    os.environ.setdefault('NCCL_IB_DISABLE', '1')  # 禁用InfiniBand

    # 从环境变量读取rank和world_size
    if rank is None:
        rank = int(os.environ.get('RANK', 0))
    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', 1))

    # 检查CUDA是否可用
    use_cuda = config.get('device', {}).get('use_cuda', True) if config else True

    if use_cuda and torch.cuda.is_available():
        if world_size > 1:
            # DDP模式
            if not dist.is_initialized():
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://'
                )
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            # 单卡模式
            device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        world_size = 1
        rank = 0

    return device, rank, world_size


def get_raw_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    获取原始模型（自动处理DDP包装）

    Args:
        model: 可能是DDP包装的模型

    Returns:
        原始模型
    """
    if hasattr(model, 'module'):
        return model.module
    return model


def is_main_process(rank: Optional[int] = None) -> bool:
    """
    检查是否是主进程

    Args:
        rank: 当前rank（可选）

    Returns:
        bool: 是否是主进程(rank 0)
    """
    if rank is not None:
        return rank == 0

    if dist.is_initialized():
        return dist.get_rank() == 0

    return True


def cleanup_resources():
    """
    清理资源

    包括垃圾回收、显存清空等。
    """
    import gc

    # 强制垃圾回收
    gc.collect()

    # 清空CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize():
    """
    分布式同步屏障

    所有进程等待直到都到达此点。
    """
    if dist.is_initialized():
        dist.barrier()


def set_seed(seed: int):
    """
    设置随机种子，确保可复现

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多卡

        # 确保确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device_info() -> Dict[str, Any]:
    """
    获取设备信息

    Returns:
        设备信息字典
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_name': None,
        'distributed_initialized': dist.is_initialized(),
    }

    if torch.cuda.is_available():
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(0)

        # 显存信息
        info['memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
        info['memory_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB

    if dist.is_initialized():
        info['rank'] = dist.get_rank()
        info['world_size'] = dist.get_world_size()

    return info


def wrap_model_ddp(
    model: torch.nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False
) -> torch.nn.Module:
    """
    包装模型为DDP模式

    Args:
        model: 原始模型
        device: 设备
        find_unused_parameters: 是否查找未使用的参数

    Returns:
        DDP包装后的模型
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    # 先将模型移到设备
    model = model.to(device)

    if dist.is_initialized():
        model = DDP(
            model,
            device_ids=[device.index] if device.type == 'cuda' else None,
            output_device=device if device.type == 'cuda' else None,
            find_unused_parameters=find_unused_parameters
        )

    return model


def all_reduce_tensor(tensor: torch.Tensor, op='mean'):
    """
    分布式all reduce操作

    Args:
        tensor: 输入张量
        op: 'mean' 或 'sum'

    Returns:
        聚合后的张量
    """
    if not dist.is_initialized():
        return tensor

    # 创建副本
    tensor = tensor.clone()

    # All reduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if op == 'mean':
        tensor /= dist.get_world_size()

    return tensor


if __name__ == "__main__":
    # 测试
    print("Testing device_ops...")

    # 测试设备信息
    info = get_device_info()
    print(f"Device info: {info}")

    # 测试设置种子
    set_seed(42)
    x1 = torch.rand(5)

    set_seed(42)
    x2 = torch.rand(5)

    assert torch.allclose(x1, x2), "Random seed not working"
    print("✓ Random seed works")

    print("All tests passed!")
