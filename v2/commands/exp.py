"""
exp.py - 实验脚本命令

运行 v2/exp_script/ 目录下的实验脚本。
提供高自由度、高容错的实验环境。

Usage:
    python main.py exp --name <script_name> [--config <config_path>]
    
Examples:
    python main.py exp --name example_analysis
    python main.py exp --name my_experiment --config custom_config.yaml
    python main.py exp --list  # 列出所有可用脚本
"""

import sys
from pathlib import Path

# 支持单独运行调试
if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

import argparse
from typing import Optional

from .base import BaseCommand, command


@command('exp')
class ExpCommand(BaseCommand):
    """
    实验脚本命令
    
    运行 v2/exp_script/ 目录下的实验脚本。
    脚本通过 ExpContext 访问项目资源，禁止直接导入底层模块。
    """
    
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """添加命令行参数"""
        parser.add_argument(
            '--name',
            type=str,
            help='实验脚本名称（不需要.py后缀）'
        )
        parser.add_argument(
            '--config',
            type=str,
            default='config.yaml',
            help='配置文件路径（默认: config.yaml）'
        )
        parser.add_argument(
            '--list',
            action='store_true',
            help='列出所有可用的实验脚本'
        )
        parser.add_argument(
            '--verbose',
            '-v',
            action='store_true',
            help='显示详细输出'
        )
        parser.add_argument(
            '--script-dir',
            type=str,
            default=None,
            help='脚本目录路径（默认: v2/exp_script/user 和 v2/exp_script/examples）'
        )
    
    def execute(self, args: argparse.Namespace):
        """
        执行命令
        
        Args:
            args: 解析后的参数
        """
        # 延迟导入，避免循环依赖
        from exp_script.runner import ExperimentRunner
        
        # 创建 runner
        if args.script_dir:
            runner = ExperimentRunner(
                script_dirs=[args.script_dir],
                verbose=args.verbose
            )
        else:
            runner = ExperimentRunner(verbose=args.verbose)
        
        # 列出脚本
        if args.list:
            scripts = runner.list_scripts()
            
            print("=" * 60)
            print("Available Experiment Scripts")
            print("=" * 60)
            
            if not scripts:
                print(f"No scripts found in: {runner.script_dir}")
                print("\nTo create a new script:")
                print("  1. Create a .py file in v2/exp_script/")
                print("  2. Define run_experiment(ctx) function")
                print("  3. See v2/exp_script/README.md for details")
            else:
                for i, script_name in enumerate(scripts, 1):
                    info = runner.get_script_info(script_name)
                    desc = info.get('description', '')
                    if desc:
                        # 只显示第一行描述
                        desc = desc.split('\n')[0][:50]
                        print(f"{i}. {script_name:20s} - {desc}")
                    else:
                        print(f"{i}. {script_name}")
                
                print(f"\nTotal: {len(scripts)} scripts")
                print(f"Location: {runner.script_dir}")
            
            print("=" * 60)
            return
        
        # 检查脚本名称
        if not args.name:
            print("Error: --name is required (or use --list to see available scripts)")
            print("\nUsage:")
            print("  python main.py exp --name <script_name>")
            print("  python main.py exp --list")
            return
        
        # 运行脚本
        print(f"Running experiment script: {args.name}")
        print(f"Config: {args.config}")
        print("-" * 60)
        
        result = runner.run(
            script_name=args.name,
            config_path=args.config
        )
        
        print("-" * 60)
        
        # 输出结果
        if result.success:
            print(f"✓ Experiment completed successfully")
            print(f"  Execution time: {result.execution_time:.2f}s")
            
            if result.result is not None:
                print(f"  Result: {result.result}")
            
            if result.log_file:
                print(f"  Log: {result.log_file}")
        else:
            print(f"✗ Experiment failed")
            print(f"  Execution time: {result.execution_time:.2f}s")
            
            if result.error:
                print(f"  Error: {result.error.message}")
                
                if args.verbose and result.error.traceback_str:
                    print("\n  Traceback:")
                    for line in result.error.traceback_str.split('\n'):
                        print(f"    {line}")
            
            if result.log_file:
                print(f"  Full log: {result.log_file}")
            
            # 返回非零退出码
            sys.exit(1)


if __name__ == "__main__":
    # 测试
    print("Testing ExpCommand...")
    
    # 创建参数
    parser = argparse.ArgumentParser()
    ExpCommand.add_arguments(parser)
    
    # 测试帮助
    args = parser.parse_args(['--list'])
    
    cmd = ExpCommand()
    cmd.execute(args)
    
    print("\nExpCommand tests completed!")