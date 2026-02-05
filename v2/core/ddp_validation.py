"""
ddp_validation.py - DDP 验证模块

在单卡环境下模拟和验证 DDP 相关逻辑的正确性。
独立于训练流程，可单独运行验证。
"""

import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path
from typing import List, Tuple, Callable

from .device_ops import get_raw_model, is_main_process, all_reduce_tensor
from .checkpoint_ops import (
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
    if not is_main_process():
        return False, "单进程环境下应该返回 True"

    if not is_main_process(rank=0):
        return False, "rank=0 应该是主进程"

    if is_main_process(rank=1):
        return False, "rank!=0 不应该为主进程"

    return True, "is_main_process 验证通过"


def validate_all_reduce() -> Tuple[bool, str]:
    """验证 all_reduce_tensor（单进程模式）"""
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


def run_all_validations(verbose: bool = False) -> Tuple[bool, List[str]]:
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


if __name__ == "__main__":
    print("Running DDP validations...")
    passed, messages = run_all_validations(verbose=True)
    print("\nResults:")
    for msg in messages:
        print(f"  {msg}")
    print(f"\nOverall: {'PASSED' if passed else 'FAILED'}")
