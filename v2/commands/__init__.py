"""
v2.commands - 命令入口模块

提供命令行命令的实现。
"""

from .base import BaseCommand, CommandRegistry
from .train import TrainCommand
from .infer import InferCommand
from .debug import DebugCommand

__all__ = [
    'BaseCommand',
    'CommandRegistry',
    'TrainCommand',
    'InferCommand',
    'DebugCommand',
]
