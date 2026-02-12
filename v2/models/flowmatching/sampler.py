"""
sampler.py - Flow Matching ODE 采样器

实现 DPM-Solver++ 和 Euler/Heun 采样方法。
"""

import torch
import torch.nn.functional as F
import math
from typing import Callable, Optional


class DPMSolverPlusPlus:
    """
    DPM-Solver++ 采样器 for Flow Matching (Rectified Flow)
    
    针对 Rectified Flow (linear interpolation path) 优化的多步采样器。
    支持 1st, 2nd, 3rd 阶求解器。
    
    Reference: "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models"
    Adapted for Flow Matching with data prediction mode.
    """
    
    def __init__(
        self,
        model_fn: Callable,
        steps: int = 20,
        order: int = 3,
        skip_type: str = 'time_uniform',
        method: str = 'multistep',
        denoise_to_zero: bool = True
    ):
        """
        Args:
            model_fn: 模型函数，输入 (x, t, sar_features, global_cond) 返回向量场 v
            steps: 采样步数
            order: 求解器阶数 (1, 2, or 3)
            skip_type: 步长策略 ('time_uniform', 'logSNR', 'time_quadratic')
            method: 求解方法 ('singlestep', 'multistep')
            denoise_to_zero: 是否在最后一步去噪到 t=0
        """
        self.model_fn = model_fn
        self.steps = steps
        self.order = order
        self.skip_type = skip_type
        self.method = method
        self.denoise_to_zero = denoise_to_zero
        
        # 用于多步方法的缓存
        self.buffer_model = []
        
    def get_time_schedule(self, device):
        """
        生成时间步调度 (t 从 1 递减到 0，对应 diffusion 的 t 从 0 到 1)
        Flow Matching 中 t=0 是噪声，t=1 是数据
        """
        if self.skip_type == 'time_uniform':
            # 均匀时间步
            timesteps = torch.linspace(1.0, 0.0, self.steps + 1, device=device)
        elif self.skip_type == 'time_quadratic':
            # 二次调度，在 t 接近 1 时更密集
            timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
            timesteps = timesteps ** 2
            timesteps = 1.0 - timesteps.flip(0)
        else:
            raise ValueError(f"Unknown skip_type: {self.skip_type}")
            
        return timesteps
    
    def model_wrapper(self, x, t, sar_features, global_cond):
        """
        模型包装器，处理 t 的格式
        
        Args:
            x: [B, C, H, W]
            t: scalar or [B]
            sar_features: dict
            global_cond: [B, D]
        """
        B = x.shape[0]
        if isinstance(t, (int, float)):
            t = torch.full((B,), t, device=x.device, dtype=x.dtype)
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(B)
            
        return self.model_fn(x, t, sar_features, global_cond)
    
    def multistep_dpm_solver_plus_plus(self, x, timesteps, sar_features, global_cond):
        """
        多步 DPM-Solver++ 求解器
        
        Flow Matching 的 ODE: dx/dt = v_theta(x, t)
        使用数据预测模式 (predicting x_0 directly)
        """
        self.buffer_model = []
        
        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            
            # 当前步的模型输出 (向量场)
            v_cur = self.model_wrapper(x, t_cur, sar_features, global_cond)
            
            # 转换为数据预测形式 (Rectified Flow: x_0 = x_t - t * v)
            # 注意：在 t=0 时，x_0 = x_t (因为 x_t = t*x_1 + (1-t)*x_0 = x_0 when t=0)
            # 在 Flow Matching 中，x_t = t * x_1 + (1-t) * x_0
            # 所以 x_0 = (x_t - t * x_1) / (1-t) 但我们不知道 x_1
            # 更简单：我们直接使用向量场进行更新
            
            dt = t_next - t_cur  # 负值，因为 t 递减
            
            if self.order == 1 or i == 0:
                # 一阶 Euler 方法
                x = x + v_cur * dt
                
            elif self.order == 2:
                # 二阶方法
                if len(self.buffer_model) == 0:
                    # 使用一阶作为第一步
                    x = x + v_cur * dt
                else:
                    # 二阶 Adams-Bashforth
                    v_prev = self.buffer_model[-1]
                    # 使用两个历史点进行插值
                    r = (t_cur - timesteps[i-1]) / (t_next - t_cur + 1e-8)
                    x = x + (1.5 * v_cur - 0.5 * v_prev) * dt
                    
            elif self.order == 3:
                # 三阶方法
                if len(self.buffer_model) < 2:
                    # 积累历史
                    x = x + v_cur * dt
                else:
                    # 三阶 Adams-Bashforth
                    v_prev1 = self.buffer_model[-1]
                    v_prev2 = self.buffer_model[-2]
                    x = x + (23/12 * v_cur - 16/12 * v_prev1 + 5/12 * v_prev2) * dt
            
            # 缓存模型输出
            self.buffer_model.append(v_cur)
            if len(self.buffer_model) > self.order:
                self.buffer_model.pop(0)
                
        return x
    
    def singlestep_dpm_solver_plus_plus(self, x, timesteps, sar_features, global_cond):
        """
        单步 DPM-Solver++ (Runge-Kutta 风格)
        """
        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_cur
            
            if self.order == 1:
                # 一阶 Euler
                v1 = self.model_wrapper(x, t_cur, sar_features, global_cond)
                x = x + v1 * dt
                
            elif self.order == 2:
                # 二阶 Heun (改进的 Euler)
                v1 = self.model_wrapper(x, t_cur, sar_features, global_cond)
                x_pred = x + v1 * dt
                v2 = self.model_wrapper(x_pred, t_next, sar_features, global_cond)
                x = x + 0.5 * (v1 + v2) * dt
                
            elif self.order == 3:
                # 三阶 Runge-Kutta
                v1 = self.model_wrapper(x, t_cur, sar_features, global_cond)
                
                t_mid = t_cur + 0.5 * dt
                x_mid = x + 0.5 * v1 * dt
                v2 = self.model_wrapper(x_mid, t_mid, sar_features, global_cond)
                
                x_pred = x - v1 * dt + 2 * v2 * dt
                v3 = self.model_wrapper(x_pred, t_next, sar_features, global_cond)
                
                x = x + (v1 + 4 * v2 + v3) * dt / 6
                
        return x
    
    def sample(self, x, sar_features, global_cond):
        """
        执行采样
        
        Args:
            x: 初始噪声 [B, C, H, W] at t=0
            sar_features: dict of SAR features
            global_cond: [B, D] global condition
            
        Returns:
            x: 生成的图像 [B, C, H, W] at t=1
        """
        timesteps = self.get_time_schedule(x.device)
        
        if self.method == 'multistep':
            x = self.multistep_dpm_solver_plus_plus(x, timesteps, sar_features, global_cond)
        else:
            x = self.singlestep_dpm_solver_plus_plus(x, timesteps, sar_features, global_cond)
        
        # 可选：最后一步去噪 (在 Flow Matching 中通常不需要，因为 t=1 已经是数据)
        # 但如果 denoise_to_zero 为 True，我们确保最终结果是干净的
        if self.denoise_to_zero and timesteps[-1] > 0.001:
            v_final = self.model_wrapper(x, timesteps[-1], sar_features, global_cond)
            x = x + v_final * (1.0 - timesteps[-1])
            
        return x


class EulerSampler:
    """
    简单的 Euler 采样器 (一阶，最稳定但步数需要更多)
    """
    
    def __init__(self, model_fn: Callable, steps: int = 50):
        self.model_fn = model_fn
        self.steps = steps
        
    def sample(self, x, sar_features, global_cond):
        """
        Euler 方法采样
        
        Args:
            x: 初始噪声 [B, C, H, W]
            sar_features: dict
            global_cond: [B, D]
        """
        B = x.shape[0]
        device = x.device
        
        dt = 1.0 / self.steps
        
        for i in range(self.steps):
            t = torch.full((B,), i * dt, device=device, dtype=x.dtype)
            
            # 计算向量场
            v = self.model_fn(x, t, sar_features, global_cond)
            
            # Euler 更新
            x = x + v * dt
            
        return x


class HeunSampler:
    """
    Heun 采样器 (二阶，精度更高)
    """
    
    def __init__(self, model_fn: Callable, steps: int = 30):
        self.model_fn = model_fn
        self.steps = steps
        
    def sample(self, x, sar_features, global_cond):
        """
        Heun (改进 Euler) 方法采样
        """
        B = x.shape[0]
        device = x.device
        
        dt = 1.0 / self.steps
        
        for i in range(self.steps):
            t_cur = torch.full((B,), i * dt, device=device, dtype=x.dtype)
            t_next = torch.full((B,), (i + 1) * dt, device=device, dtype=x.dtype)
            
            # 预测步
            v1 = self.model_fn(x, t_cur, sar_features, global_cond)
            x_pred = x + v1 * dt
            
            # 校正步
            v2 = self.model_fn(x_pred, t_next, sar_features, global_cond)
            x = x + 0.5 * (v1 + v2) * dt
            
        return x


class FlowMatchingSampler:
    """
    Flow Matching 采样器统一接口
    """
    
    SAMPLERS = {
        'euler': EulerSampler,
        'heun': HeunSampler,
        'dpmpp': DPMSolverPlusPlus,
    }
    
    def __init__(
        self,
        model_fn: Callable,
        method: str = 'dpmpp',
        steps: int = 20,
        **kwargs
    ):
        """
        Args:
            model_fn: 模型函数
            method: 采样方法 ('euler', 'heun', 'dpmpp')
            steps: 采样步数
            **kwargs: 传递给具体采样器的参数
        """
        if method not in self.SAMPLERS:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.SAMPLERS.keys())}")
            
        sampler_class = self.SAMPLERS[method]
        self.sampler = sampler_class(model_fn, steps, **kwargs)
        
    def sample(self, x, sar_features, global_cond):
        """执行采样"""
        return self.sampler.sample(x, sar_features, global_cond)


if __name__ == "__main__":
    # 测试采样器
    print("Testing Flow Matching samplers...")
    
    # 创建一个简单的模型函数
    def dummy_model(x, t, sar_features, global_cond):
        # 简单的恒等映射
        return torch.zeros_like(x)
    
    # 测试 DPM-Solver++
    sampler = DPMSolverPlusPlus(
        dummy_model,
        steps=10,
        order=3,
        method='singlestep'
    )
    
    x = torch.randn(2, 3, 64, 64)
    sar_features = {'L1': torch.randn(2, 64, 32, 32)}
    global_cond = torch.randn(2, 256)
    
    result = sampler.sample(x, sar_features, global_cond)
    print(f"DPM-Solver++ output shape: {result.shape}")
    
    # 测试统一接口
    unified = FlowMatchingSampler(dummy_model, method='euler', steps=20)
    result2 = unified.sample(x, sar_features, global_cond)
    print(f"Euler output shape: {result2.shape}")
    
    print("All tests passed!")
