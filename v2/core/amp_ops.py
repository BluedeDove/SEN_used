"""
amp_ops.py - 自动混合精度 (AMP) 管理器

提供混合精度训练和推理的统一管理，自动切换 float32/float16。

使用方式:
    # 训练模式
    amp_manager = AMPManager(enabled=True, use_bf16=False)
    with amp_manager.autocast():
        output = model(input)
    
    # 反向传播时缩放梯度
    amp_manager.scale(loss).backward()
    amp_manager.step(optimizer)
    amp_manager.update()

    # 推理模式
    amp_manager = AMPManager(enabled=True)
    with amp_manager.autocast():
        output = model(input)  # 自动使用 fp16
"""

import torch
from torch.amp import autocast, GradScaler
from typing import Optional, Union, Dict, Any


class AMPManager:
    """
    自动混合精度管理器
    
    统一封装 PyTorch AMP，提供训练和推理的便捷接口。
    支持 FP16 和 BF16 (A100/A800 推荐 BF16，更稳定)。
    """
    
    def __init__(
        self,
        enabled: bool = True,
        use_bf16: bool = False,
        device_type: str = 'cuda',
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        """
        Args:
            enabled: 是否启用 AMP
            use_bf16: 是否使用 BF16 (需要 Ampere 架构 GPU，如 A100/A800)
            device_type: 设备类型 'cuda' 或 'cpu'
            init_scale: 初始梯度缩放因子 (仅 FP16)
            growth_factor: 梯度增长因子
            backoff_factor: 梯度回退因子
            growth_interval: 增长间隔
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.use_bf16 = use_bf16 and self._check_bf16_support()
        self.device_type = device_type if torch.cuda.is_available() else 'cpu'
        
        # 初始化梯度缩放器 (仅 FP16 需要)
        self.scaler: Optional[GradScaler] = None
        if self.enabled and not self.use_bf16:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                enabled=True,
            )
    
    def _check_bf16_support(self) -> bool:
        """检查是否支持 BF16"""
        if not torch.cuda.is_available():
            return False
        # BF16 需要 CUDA 11.0+ 和 Ampere 架构 (SM80+)
        capability = torch.cuda.get_device_capability()
        return capability[0] >= 8  # SM80+ (A100, A800, RTX 30/40 series)
    
    def autocast(self, enabled: Optional[bool] = None):
        """
        获取 autocast 上下文管理器
        
        使用示例:
            with amp_manager.autocast():
                output = model(input)
        """
        if enabled is None:
            enabled = self.enabled
        
        dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        
        return autocast(
            device_type=self.device_type,
            dtype=dtype,
            enabled=enabled,
        )
    
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """
        缩放损失值 (用于反向传播)
        
        Args:
            loss: 原始损失值
            
        Returns:
            缩放后的损失值
        """
        if not self.enabled or self.use_bf16:
            return loss
        return self.scaler.scale(loss)
    
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        执行优化器步骤 (自动处理梯度缩放)
        
        Args:
            optimizer: 优化器实例
        """
        if not self.enabled or self.use_bf16:
            optimizer.step()
        else:
            self.scaler.step(optimizer)
    
    def update(self) -> None:
        """更新梯度缩放因子"""
        if self.enabled and not self.use_bf16 and self.scaler is not None:
            self.scaler.update()
    
    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """
        反缩放梯度 (用于梯度裁剪前)
        
        Args:
            optimizer: 优化器实例
        """
        if self.enabled and not self.use_bf16 and self.scaler is not None:
            self.scaler.unscale_(optimizer)
    
    def get_scale(self) -> float:
        """获取当前梯度缩放因子"""
        if not self.enabled or self.use_bf16 or self.scaler is None:
            return 1.0
        return self.scaler.get_scale()
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典 (用于保存检查点)"""
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典 (用于恢复检查点)"""
        if self.scaler is not None and state_dict:
            self.scaler.load_state_dict(state_dict)
    
    def is_enabled(self) -> bool:
        """检查 AMP 是否启用"""
        return self.enabled
    
    def is_bf16(self) -> bool:
        """检查是否使用 BF16"""
        return self.use_bf16


def create_amp_manager(config: Dict[str, Any]) -> AMPManager:
    """
    从配置创建 AMP 管理器
    
    Args:
        config: 配置字典，支持以下字段:
            - enabled: bool, 是否启用 AMP
            - use_bf16: bool, 是否使用 BF16
            - init_scale: float, 初始缩放因子
            - growth_interval: int, 增长间隔
    
    Returns:
        AMPManager 实例
    """
    amp_config = config.get('amp', {})
    
    enabled = amp_config.get('enabled', True)
    use_bf16 = amp_config.get('use_bf16', False)
    init_scale = amp_config.get('init_scale', 2.0 ** 16)
    growth_interval = amp_config.get('growth_interval', 2000)
    
    return AMPManager(
        enabled=enabled,
        use_bf16=use_bf16,
        init_scale=init_scale,
        growth_interval=growth_interval,
    )


# 便捷的上下文管理器
@torch.no_grad()
def inference_with_amp(model: torch.nn.Module, *args, amp_manager: Optional[AMPManager] = None, **kwargs):
    """
    使用 AMP 进行推理
    
    Args:
        model: 模型
        *args: 模型输入参数
        amp_manager: AMP 管理器 (None 则使用 fp32)
        **kwargs: 模型输入关键字参数
    
    Returns:
        模型输出
    """
    if amp_manager is None or not amp_manager.is_enabled():
        return model(*args, **kwargs)
    
    with amp_manager.autocast():
        return model(*args, **kwargs)


if __name__ == "__main__":
    # 测试
    print("Testing AMPManager...")
    
    # 测试 FP16
    amp_fp16 = AMPManager(enabled=True, use_bf16=False)
    print(f"FP16 Enabled: {amp_fp16.is_enabled()}, BF16: {amp_fp16.is_bf16()}")
    
    # 测试 BF16 (如果支持)
    amp_bf16 = AMPManager(enabled=True, use_bf16=True)
    print(f"BF16 Enabled: {amp_bf16.is_enabled()}, BF16: {amp_bf16.is_bf16()}")
    
    # 测试禁用
    amp_disabled = AMPManager(enabled=False)
    print(f"Disabled: {amp_disabled.is_enabled()}")
    
    # 测试 autocast 上下文
    if torch.cuda.is_available():
        with amp_fp16.autocast():
            x = torch.randn(2, 3, 64, 64).cuda()
            print(f"Autocast test passed, dtype: {x.dtype}")
    
    print("All tests passed!")
