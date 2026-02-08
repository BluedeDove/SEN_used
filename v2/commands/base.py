"""
base.py - 命令基类

定义所有命令必须实现的接口。
"""

# 支持单独运行调试：将项目根目录添加到路径
import sys
from pathlib import Path

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

from abc import ABC, abstractmethod
from typing import Dict, Type
import argparse


class BaseCommand(ABC):
    """
    所有命令的基类

    子类必须实现:
    - add_arguments(): 添加命令行参数
    - execute(): 执行命令
    """

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """
        添加命令行参数

        Args:
            parser: 参数解析器
        """
        pass

    @abstractmethod
    def execute(self, args: argparse.Namespace):
        """
        执行命令

        Args:
            args: 解析后的参数
        """
        pass


class CommandRegistry:
    """命令注册表"""

    _commands: Dict[str, Type[BaseCommand]] = {}

    @classmethod
    def register(cls, name: str, command_class: Type[BaseCommand]):
        """注册命令"""
        cls._commands[name] = command_class

    @classmethod
    def get(cls, name: str) -> Type[BaseCommand]:
        """获取命令类"""
        if name not in cls._commands:
            raise ValueError(f"Unknown command: {name}")
        return cls._commands[name]

    @classmethod
    def list_commands(cls):
        """列出所有命令"""
        return list(cls._commands.keys())


def command(name: str):
    """命令注册装饰器"""
    def decorator(cls: Type[BaseCommand]):
        CommandRegistry.register(name, cls)
        return cls
    return decorator


if __name__ == "__main__":
    # 测试
    print("Testing base.py...")

    @command('test')
    class TestCommand(BaseCommand):
        @classmethod
        def add_arguments(cls, parser):
            parser.add_argument('--test-arg', type=str, default='test')

        def execute(self, args):
            print(f"Test arg: {args.test_arg}")

    # 测试获取
    cmd_class = CommandRegistry.get('test')
    print(f"✓ Got command: {cmd_class}")

    print("All tests passed!")
