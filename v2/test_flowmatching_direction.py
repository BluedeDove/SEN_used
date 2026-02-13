"""
test_flowmatching_direction.py - Flow Matching 采样方向验证脚本

验证 DPM-Solver++ 采样方向是否正确。

问题：
- DPM-Solver++ 使用 timesteps 从 1→0 递减
- 但 EulerSampler 使用 timesteps 从 0→1 递增
- 这导致 DPM-Solver++ 采样方向反了，生成噪声而不是图像

验证方法：
1. 用随机输入测试，观察输出范围
2. 如果方向正确，输出应该在 [0, 1] 范围内
3. 如果方向反了，输出会是随机噪声（值范围很大）

使用方法：
    cd v2 && python test_flowmatching_direction.py --config models/flowmatching/config.yaml --checkpoint <path_to_checkpoint.pth>
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
v2_dir = Path(__file__).parent
if str(v2_dir) not in sys.path:
    sys.path.insert(0, str(v2_dir))

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from models.flowmatching.model import FlowMatchingModel
from core.config_loader import load_config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Flow Matching 采样方向验证')
    parser.add_argument('--config', type=str, default='models/flowmatching/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径')
    parser.add_argument('--output-dir', type=str, default='test_outputs',
                        help='输出目录')
    parser.add_argument('--test-mode', type=str, default='both',
                        choices=['original', 'fixed', 'both'],
                        help='测试模式: original=原始方向(可能有bug), fixed=修复方向, both=两者都测试')
    return parser.parse_args()


def create_model_from_config(config):
    """从配置创建模型"""
    fm_config = config.get('flowmatching', {})
    sar_enc_config = config.get('sar_encoder', {})
    
    model = FlowMatchingModel(
        base_ch=fm_config.get('base_ch', 64),
        ch_mults=fm_config.get('ch_mults', [1, 2, 4, 8]),
        num_blocks=fm_config.get('num_blocks', 2),
        time_emb_dim=fm_config.get('time_emb_dim', 256),
        dropout=fm_config.get('dropout', 0.1),
        num_heads=fm_config.get('num_heads', 8),
        sar_encoder_config=sar_enc_config,
        use_sar_base=fm_config.get('use_sar_base', False)
    )
    return model


def load_checkpoint(model, checkpoint_path):
    """加载检查点"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 处理不同的检查点格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    print("Checkpoint loaded successfully")
    return model


def fixed_dpmpp_sampling(model, sar, steps=20, order=3):
    """
    修复方向的 DPM-Solver++ 采样 (t 从 0→1)
    
    原始代码问题：
    - timesteps = torch.linspace(1.0, 0.0, ...)  # 1→0
    - dt = t_next - t_cur  # 负值
    - 导致采样往噪声方向走
    
    修复：
    - timesteps = torch.linspace(0.0, 1.0, ...)  # 0→1
    - dt = t_next - t_cur  # 正值
    """
    B, _, H, W = sar.shape
    device = sar.device
    
    # SAR 编码
    sar_base, sar_features, global_cond = model.sar_encoder(sar)
    
    # 从噪声开始
    x = torch.randn(B, 3, H, W, device=device)
    
    # ✅ 修复：t 从 0→1 递增（而不是 1→0 递减）
    timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)
    
    buffer_model = []
    
    for i in range(len(timesteps) - 1):
        t_cur = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_cur  # ✅ 正值
        
        # 当前时间步 (batch)
        t_batch = torch.full((B,), t_cur, device=device)
        
        # 计算向量场
        v_cur = model.unet(x, t_batch, sar_features)
        
        if order == 1 or i == 0:
            # 一阶 Euler
            x = x + v_cur * dt
        elif order == 2:
            if len(buffer_model) == 0:
                x = x + v_cur * dt
            else:
                v_prev = buffer_model[-1]
                x = x + (1.5 * v_cur - 0.5 * v_prev) * dt
        elif order == 3:
            if len(buffer_model) < 2:
                x = x + v_cur * dt
            else:
                v_prev1 = buffer_model[-1]
                v_prev2 = buffer_model[-2]
                x = x + (23/12 * v_cur - 16/12 * v_prev1 + 5/12 * v_prev2) * dt
        
        buffer_model.append(v_cur)
        if len(buffer_model) > order:
            buffer_model.pop(0)
    
    return torch.clamp(x, 0.0, 1.0)


def original_dpmpp_sampling(model, sar, steps=20, order=3):
    """
    原始方向的 DPM-Solver++ 采样 (t 从 1→0) - 可能有 Bug
    
    这是复制原始 sampler.py 中的逻辑
    """
    B, _, H, W = sar.shape
    device = sar.device
    
    # SAR 编码
    sar_base, sar_features, global_cond = model.sar_encoder(sar)
    
    # 从噪声开始
    x = torch.randn(B, 3, H, W, device=device)
    
    # ❌ 原始：t 从 1→0 递减
    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)
    
    buffer_model = []
    
    for i in range(len(timesteps) - 1):
        t_cur = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_cur  # ❌ 负值
        
        t_batch = torch.full((B,), t_cur, device=device)
        v_cur = model.unet(x, t_batch, sar_features)
        
        if order == 1 or i == 0:
            x = x + v_cur * dt
        elif order == 2:
            if len(buffer_model) == 0:
                x = x + v_cur * dt
            else:
                v_prev = buffer_model[-1]
                x = x + (1.5 * v_cur - 0.5 * v_prev) * dt
        elif order == 3:
            if len(buffer_model) < 2:
                x = x + v_cur * dt
            else:
                v_prev1 = buffer_model[-1]
                v_prev2 = buffer_model[-2]
                x = x + (23/12 * v_cur - 16/12 * v_prev1 + 5/12 * v_prev2) * dt
        
        buffer_model.append(v_cur)
        if len(buffer_model) > order:
            buffer_model.pop(0)
    
    return torch.clamp(x, 0.0, 1.0)


def euler_sampling(model, sar, steps=50):
    """
    使用 EulerSampler (已经正确)
    """
    B, _, H, W = sar.shape
    device = sar.device
    
    # SAR 编码
    sar_base, sar_features, global_cond = model.sar_encoder(sar)
    
    # 从噪声开始
    x = torch.randn(B, 3, H, W, device=device)
    
    dt = 1.0 / steps
    
    for i in range(steps):
        # ✅ 已经正确：t 从 0→1
        t = torch.full((B,), i * dt, device=device)
        v = model.unet(x, t, sar_features)
        x = x + v * dt
    
    return torch.clamp(x, 0.0, 1.0)


def test_sampling_direction(model, sar, mode='both', output_dir='test_outputs'):
    """测试采样方向"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    sar = sar.to(device)
    
    results = {}
    
    with torch.no_grad():
        # 1. 测试 EulerSampler (作为基准)
        print("\n[1/3] Testing EulerSampler (baseline)...")
        result_euler = euler_sampling(model, sar, steps=50)
        results['euler'] = result_euler
        print(f"  Euler range: [{result_euler.min():.4f}, {result_euler.max():.4f}]")
        print(f"  Euler mean: {result_euler.mean():.4f}, std: {result_euler.std():.4f}")
        save_image(result_euler, output_dir / 'test_euler.png')
        
        if mode in ['original', 'both']:
            # 2. 测试原始 DPM-Solver++ 方向
            print("\n[2/3] Testing Original DPM-Solver++ (1→0 direction)...")
            result_orig = original_dpmpp_sampling(model, sar, steps=20, order=3)
            results['original'] = result_orig
            print(f"  Original range: [{result_orig.min():.4f}, {result_orig.max():.4f}]")
            print(f"  Original mean: {result_orig.mean():.4f}, std: {result_orig.std():.4f}")
            save_image(result_orig, output_dir / 'test_original_dpmpp.png')
            
            # 判断是否有问题
            if result_orig.std() > 0.3 or result_orig.mean() < 0.2 or result_orig.mean() > 0.8:
                print("  ⚠️  WARNING: Original direction produces noise-like output!")
        
        if mode in ['fixed', 'both']:
            # 3. 测试修复后的 DPM-Solver++ 方向
            print("\n[3/3] Testing Fixed DPM-Solver++ (0→1 direction)...")
            result_fixed = fixed_dpmpp_sampling(model, sar, steps=20, order=3)
            results['fixed'] = result_fixed
            print(f"  Fixed range: [{result_fixed.min():.4f}, {result_fixed.max():.4f}]")
            print(f"  Fixed mean: {result_fixed.mean():.4f}, std: {result_fixed.std():.4f}")
            save_image(result_fixed, output_dir / 'test_fixed_dpmpp.png')
    
    # 比较结果
    if mode == 'both':
        print("\n" + "="*60)
        print("COMPARISON:")
        print("="*60)
        print(f"{'Method':<20} {'Range':<20} {'Mean':<10} {'Std':<10}")
        print("-"*60)
        for name, result in results.items():
            r_min = result.min().item()
            r_max = result.max().item()
            r_mean = result.mean().item()
            r_std = result.std().item()
            print(f"{name:<20} [{r_min:.4f}, {r_max:.4f}]   {r_mean:.4f}     {r_std:.4f}")
        
        print("\n" + "="*60)
        print("DIAGNOSIS:")
        print("="*60)
        
        euler_std = results['euler'].std().item()
        
        if 'original' in results:
            orig_std = results['original'].std().item()
            if orig_std > euler_std * 1.5:
                print("❌ ORIGINAL DPM-Solver++ has BUG: std too high (noise)")
            else:
                print("✅ ORIGINAL DPM-Solver++ seems OK")
        
        if 'fixed' in results:
            fixed_std = results['fixed'].std().item()
            if abs(fixed_std - euler_std) < euler_std * 0.3:
                print("✅ FIXED DPM-Solver++ matches EulerSampler (correct!)")
            else:
                print("⚠️ FIXED DPM-Solver++ deviates from EulerSampler")
    
    print(f"\nOutputs saved to: {output_dir}")
    return results


def main():
    args = parse_args()
    
    # 加载配置
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # 创建模型
    print("Creating model...")
    model = create_model_from_config(config)
    
    # 加载检查点（如果有）
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint)
    else:
        print("WARNING: No checkpoint provided, using random weights!")
    
    model.eval()
    
    # 移动到 GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # 创建测试输入（随机 SAR）
    print("\nCreating test input (random SAR)...")
    sar = torch.rand(2, 3, 128, 128)  # 2张测试图
    
    # 运行测试
    print("\n" + "="*60)
    print(f"Running Flow Matching Direction Test (mode: {args.test_mode})")
    print("="*60)
    
    test_sampling_direction(model, sar, mode=args.test_mode, output_dir=args.output_dir)
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()
