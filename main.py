#!/usr/bin/env python3
"""
SRDM v2 - SAR图像翻译扩散模型

主入口文件，负责解析命令行参数并执行相应命令。

Usage:
    python main.py train --config config_srdm.yaml
    python main.py infer --config config_srdm.yaml --checkpoint path/to/checkpoint.pth
    python main.py debug --config config_srdm.yaml
"""

import sys
import os

# 设置NCCL环境变量（必须在import torch.distributed之前）
os.environ.setdefault('NCCL_SOCKET_FAMILY', 'AF_INET')  # 强制IPv4
os.environ.setdefault('NCCL_IB_DISABLE', '1')  # 禁用InfiniBand

import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'v2'))

# 导入命令
from commands.base import CommandRegistry
from commands.train import TrainCommand
from commands.infer import InferCommand
from commands.debug import DebugCommand


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='SRDM v2 - SAR Image Translation Diffusion Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train a new model
  python main.py train --config config_srdm.yaml --experiment my_exp

  # Resume training
  python main.py train --config config_srdm.yaml --experiment my_exp --resume

  # Run inference
  python main.py infer --config config_srdm.yaml --checkpoint experiments/my_exp/checkpoints/best.pth

  # Debug mode (validate numerical ranges)
  python main.py debug --config config_srdm.yaml --verbose

  # DDP training (multi-GPU)
  torchrun --nproc_per_node=2 main.py train --config config_srdm.yaml --experiment my_exp
''')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # 注册所有命令
    for cmd_name in CommandRegistry.list_commands():
        cmd_class = CommandRegistry.get(cmd_name)
        cmd_parser = subparsers.add_parser(cmd_name, help=f'{cmd_name} command')
        cmd_class.add_arguments(cmd_parser)

    # 解析参数
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # 执行命令
    cmd_class = CommandRegistry.get(args.command)
    cmd = cmd_class()
    exit_code = cmd.execute(args)

    sys.exit(exit_code if exit_code is not None else 0)


if __name__ == '__main__':
    main()
