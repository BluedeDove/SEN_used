"""
infer.py - 推理命令

实现模型推理流程。
"""

# 支持单独运行调试：将项目根目录添加到路径
import sys
from pathlib import Path

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

import argparse
from tqdm import tqdm
from commands.base import BaseCommand, command
from core.device_ops import setup_device_and_distributed, is_main_process
from core.inference_ops import run_inference
from utils.image_ops import save_image_v2
from core.numeric_ops import composite_to_uint8
from core.config_loader import load_config
from core.visualization_ops import create_inference_comparison


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
        # 加载并合并配置
        config = load_config(args.config, verbose=False)

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
        samples_dir = output_dir / "samples"
        report_dir = output_dir / "report"
        if is_main_process(rank):
            output_dir.mkdir(parents=True, exist_ok=True)
            samples_dir.mkdir(parents=True, exist_ok=True)
            report_dir.mkdir(parents=True, exist_ok=True)

        # 运行推理
        inference_outputs = run_inference(
            config,
            checkpoint_path=args.checkpoint,
            max_samples=args.max_samples,
            device=device
        )

        # 保存结果
        if is_main_process(rank):
            sample_paths = []
            for i, output in enumerate(tqdm(inference_outputs, desc="Saving")):
                # 保存生成的结果
                img_uint8 = composite_to_uint8(output.generated, input_range=(0.0, 1.0))
                save_image_v2(img_uint8, str(samples_dir / f"result_{i:04d}.png"))

                # 创建对比图（SAR | Generated | Optical）
                if output.optical is not None:
                    comparison = create_inference_comparison(
                        output.sar, output.generated, output.optical
                    )
                    comparison_path = samples_dir / f"comparison_{i:04d}.png"
                    save_image_v2(comparison, str(comparison_path))
                    sample_paths.append(comparison_path)

            # 生成汇总报告（前10张）
            if sample_paths:
                from core.visualization_ops import create_comparison_figure
                report_path = report_dir / "inference_report.png"
                create_comparison_figure(
                    sample_paths[:10],
                    report_path,
                    title=f"Inference Results - {args.checkpoint}",
                    samples_per_row=5
                )

            print(f"\nInference completed!")
            print(f"  Samples: {samples_dir}")
            print(f"  Report: {report_dir / 'inference_report.png'}")


if __name__ == "__main__":
    print("InferCommand module loaded")
