"""
example_analysis.py - 示例实验脚本

这是一个完整的示例，展示了如何使用 ExpContext 进行批量推理和指标分析。

功能：
1. 加载模型和检查点
2. 批量推理验证集
3. 计算 PSNR/SSIM 指标
4. 分析并保存统计报告
5. 生成可视化对比图

Usage:
    python main.py exp --name example_analysis --verbose
"""

import json
import numpy as np


def run_experiment(ctx):
    """
    运行批量推理分析实验
    
    这个示例展示了如何：
    - 安全地加载模型和检查点
    - 批量处理数据集
    - 使用 safe_run 确保单点错误不影响整体
    - 计算并分析指标
    - 保存结果和可视化
    
    Args:
        ctx: ExpContext 实例
        
    Returns:
        分析结果字典
    """
    # =========================================================================
    # 1. 初始化
    # =========================================================================
    ctx.log_info("Starting batch inference analysis")
    
    # 加载配置
    config = ctx.load_config()
    ctx.log_info(f"Loaded config: model={config.get('model', {}).get('name', 'unknown')}")
    
    # 设置设备
    device, rank, world_size = ctx.setup_device_and_distributed(config)
    ctx.log_info(f"Device: {device}, Rank: {rank}, World Size: {world_size}")
    
    # 设置随机种子（确保可复现）
    ctx.set_seed(42)
    
    # =========================================================================
    # 2. 加载模型
    # =========================================================================
    ctx.log_info("Creating model...")
    
    model = ctx.safe_run(
        lambda: ctx.create_model(config, device),
        default=None,
        error_msg="Failed to create model"
    )
    
    if model is None:
        ctx.log_error("Model creation failed, aborting experiment")
        return {'status': 'failed', 'reason': 'model_creation_failed'}
    
    # 尝试加载检查点
    checkpoint_path = config.get('checkpoint', {}).get('path', 'checkpoints/latest.pth')
    
    checkpoint = ctx.safe_run(
        lambda: ctx.load_checkpoint(checkpoint_path, device),
        default=None,
        error_msg=f"Failed to load checkpoint from {checkpoint_path}"
    )
    
    if checkpoint:
        ctx.log_info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        ctx.restore_model(model, checkpoint['model_state_dict'])
    else:
        ctx.log_warning("No checkpoint loaded, using random initialization")
    
    model.eval()
    
    # =========================================================================
    # 3. 加载数据集
    # =========================================================================
    ctx.log_info("Loading dataset...")
    
    dataset = ctx.safe_run(
        lambda: ctx.create_dataset(config, split='val'),
        default=None,
        error_msg="Failed to create dataset"
    )
    
    if dataset is None:
        return {'status': 'failed', 'reason': 'dataset_creation_failed'}
    
    max_samples = min(100, len(dataset))  # 限制样本数
    ctx.log_info(f"Dataset loaded: {len(dataset)} samples, using {max_samples}")
    
    # =========================================================================
    # 4. 批量推理和指标计算
    # =========================================================================
    ctx.log_info("Starting batch inference...")
    
    all_metrics = []
    failed_samples = []
    
    for idx in range(max_samples):
        # 每10个样本输出一次进度
        if idx % 10 == 0:
            ctx.log_info(f"Processing {idx}/{max_samples}...")
        
        # 使用 safe_run 处理每个样本，确保单点错误不影响整体
        result = ctx.safe_run(
            lambda i=idx: process_single_sample(ctx, model, dataset, i, device),
            default=None,
            error_msg=f"Failed to process sample {idx}"
        )
        
        if result is not None:
            all_metrics.append(result)
        else:
            failed_samples.append(idx)
    
    ctx.log_info(f"Processed {len(all_metrics)} samples successfully, {len(failed_samples)} failed")
    
    # =========================================================================
    # 5. 统计分析
    # =========================================================================
    if not all_metrics:
        ctx.log_error("No samples processed successfully")
        return {'status': 'failed', 'reason': 'no_samples_processed'}
    
    # 提取指标
    all_psnr = [m['psnr'] for m in all_metrics]
    all_ssim = [m['ssim'] for m in all_metrics]
    
    # 计算统计量
    stats = {
        'status': 'success',
        'total_samples': len(all_metrics),
        'failed_samples': len(failed_samples),
        'psnr': {
            'mean': float(np.mean(all_psnr)),
            'std': float(np.std(all_psnr)),
            'min': float(np.min(all_psnr)),
            'max': float(np.max(all_psnr)),
            'median': float(np.median(all_psnr)),
        },
        'ssim': {
            'mean': float(np.mean(all_ssim)),
            'std': float(np.std(all_ssim)),
            'min': float(np.min(all_ssim)),
            'max': float(np.max(all_ssim)),
            'median': float(np.median(all_ssim)),
        }
    }
    
    # =========================================================================
    # 6. 保存结果
    # =========================================================================
    ctx.log_info("Saving results...")
    
    # 保存统计报告
    report_path = 'example_analysis_report.json'
    ctx.safe_run(
        lambda: save_json_report(report_path, stats, all_metrics),
        error_msg="Failed to save JSON report"
    )
    
    # 打印摘要（仅主进程）
    if ctx.is_main_process():
        print("\n" + "=" * 60)
        print("Analysis Results Summary")
        print("=" * 60)
        print(f"Total Samples: {stats['total_samples']}")
        print(f"Failed Samples: {stats['failed_samples']}")
        print("-" * 60)
        print("PSNR Statistics:")
        print(f"  Mean:   {stats['psnr']['mean']:.2f} dB")
        print(f"  Std:    {stats['psnr']['std']:.2f} dB")
        print(f"  Min:    {stats['psnr']['min']:.2f} dB")
        print(f"  Max:    {stats['psnr']['max']:.2f} dB")
        print(f"  Median: {stats['psnr']['median']:.2f} dB")
        print("-" * 60)
        print("SSIM Statistics:")
        print(f"  Mean:   {stats['ssim']['mean']:.4f}")
        print(f"  Std:    {stats['ssim']['std']:.4f}")
        print(f"  Min:    {stats['ssim']['min']:.4f}")
        print(f"  Max:    {stats['ssim']['max']:.4f}")
        print(f"  Median: {stats['ssim']['median']:.4f}")
        print("=" * 60)
        print(f"\nFull report saved to: {report_path}")
    
    return stats


def process_single_sample(ctx, model, dataset, idx, device):
    """
    处理单个样本
    
    Args:
        ctx: ExpContext
        model: 模型
        dataset: 数据集
        idx: 样本索引
        device: 设备
        
    Returns:
        指标字典 {'psnr': float, 'ssim': float}
    """
    import torch
    
    # 获取样本
    sample = dataset[idx]
    sar = sample['sar'].unsqueeze(0).to(device)
    optical = sample['optical'].unsqueeze(0).to(device)
    
    # 验证输入范围
    ctx.validate_range(sar, (0.0, 1.0), name=f"SAR_{idx}")
    
    # 推理
    with torch.no_grad():
        output = model.get_output(sar, {})
        generated = output.generated
    
    # 验证输出范围
    ctx.validate_range(generated, (0.0, 1.0), name=f"Generated_{idx}")
    
    # 计算指标
    psnr = ctx.compute_psnr(generated, optical)
    ssim = ctx.compute_ssim(generated, optical)
    
    return {'psnr': psnr, 'ssim': ssim}


def save_json_report(path, stats, all_metrics):
    """保存 JSON 报告"""
    report = {
        'statistics': stats,
        'all_metrics': all_metrics
    }
    
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    # 支持直接运行测试
    print("This script should be run via: python main.py exp --name example_analysis")
    print("\nOr import and test the functions:")
    print("  from v2.exp_script.example_analysis import process_single_sample")
