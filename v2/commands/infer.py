"""
infer.py - 推理命令

实现模型推理流程。
"""

import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from commands.base import BaseCommand, command
from core.device_ops import setup_device_and_distributed, is_main_process
from core.inference_ops import run_inference
from utils.image_ops import save_image_v2
from core.numeric_ops import composite_to_uint8


@command('infer')
class InferCommand(BaseCommand):
    """推理命令"""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--config', type=str, required=True, help='配置文件路径')
        parser.add_argument('--checkpoint', type=str, required=True, help='检查点路径')
        parser.add_argument('--output', type=str, default='infer_results', help='输出目录')
        parser.add_argument('--max-samples', type=int, default=None, help='最大样本数')

    def execute(self, args: argparse.Namespace):
        """执行推理"""
        # 加载配置
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 设置设备
        device, rank, world_size = setup_device_and_distributed(config)

        if is_main_process(rank):
            print(f"=" * 70)
            print(f"SRDM Inference")
            print(f"=" * 70)
            print(f"Device: {device}")
            print(f"Checkpoint: {args.checkpoint}")
            print(f"Output: {args.output}")
            print(f"=" * 70)

        # 创建输出目录
        output_dir = Path(args.output)
        if is_main_process(rank):
            output_dir.mkdir(parents=True, exist_ok=True)

        # 运行推理
        inference_outputs = run_inference(
            config,
            checkpoint_path=args.checkpoint,
            max_samples=args.max_samples,
            device=device
        )

        # 保存结果
        if is_main_process(rank):
            for i, output in enumerate(tqdm(inference_outputs, desc="Saving")):
                # 转换并保存
                img_uint8 = composite_to_uint8(output.generated, input_range=(0.0, 1.0))
                save_image_v2(img_uint8, str(output_dir / f"result_{i:04d}.png"))

            print(f"\nInference completed! Results saved to {output_dir}")


if __name__ == "__main__":
    print("InferCommand module loaded")
