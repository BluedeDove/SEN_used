"""
v2.engine - 训练引擎

提供训练循环、验证逻辑等核心训练功能。
"""

from .trainer import Trainer
from .validator import Validator

__all__ = ['Trainer', 'Validator']
