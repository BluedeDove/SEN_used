"""
validator.py - 验证逻辑

实现验证流程。
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
from typing import Dict, Any
from core.validation_ops import run_validation


class Validator:
    """验证器"""

    def __init__(
        self,
        model,
        val_loader,
        config: dict,
        device: torch.device
    ):
        self.model = model
        self.val_loader = val_loader
        self.config = config
        self.device = device

    def validate(self, epoch: int, save_results: bool = False, save_dir: str = None) -> Dict[str, float]:
        """
        运行验证

        Args:
            epoch: 当前epoch
            save_results: 是否保存结果
            save_dir: 保存目录

        Returns:
            验证指标字典
        """
        return run_validation(
            self.model,
            self.val_loader,
            self.config,
            self.device,
            save_results=save_results,
            save_dir=save_dir
        )


if __name__ == "__main__":
    print("Validator module loaded")
