"""
context.py - 实验脚本上下文

ExpContext 作为实验脚本的"工具箱"，注入项目核心能力。
脚本不直接依赖项目内部模块，而是通过 ExpContext 对象获取资源。
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Callable, Union, List
from dataclasses import dataclass, field
import logging

# 支持单独运行调试：将项目根目录添加到路径
if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

import torch
import numpy as np
from torch.utils.data import DataLoader

# 导入项目核心模块
from core.numeric_ops import (
    validate_range,
    normalize_to_range,
    clamp_and_normalize,
    model_output_to_composite,
    composite_to_uint8,
    tensor_info,
)
from core.device_ops import (
    setup_device_and_distributed,
    get_raw_model,
    is_main_process,
    cleanup_resources,
    synchronize,
    set_seed,
    get_device_info,
    wrap_model_ddp,
    all_reduce_tensor,
)
from core.inference_ops import (
    run_inference,
    inference_batch,
    InferenceOutput,
)
from core.training_ops import (
    setup_training,
    train_step,
    train_step_with_accumulation,
    run_training_epoch,
    handle_checkpoint_save,
    run_training_loop,
    TrainingContext,
)
from core.validation_ops import (
    compute_psnr,
    compute_ssim,
    compute_metrics_batch,
    save_validation_samples,
    run_validation,
)
from core.checkpoint_ops import (
    save_checkpoint_v2,
    load_checkpoint_v2,
    restore_model_v2,
    restore_optimizer_v2,
    restore_scheduler_v2,
    get_latest_checkpoint,
    cleanup_old_checkpoints,
    save_checkpoint_with_backup,
)
from core.visualization_ops import (
    setup_report_directory,
    log_loss,
    load_training_log,
    plot_loss_curve,
    create_inference_comparison,
    create_comparison_figure,
    generate_validation_report,
)
from utils.image_ops import (
    tensor_to_numpy,
    numpy_to_tensor,
    normalize_for_display,
    create_comparison_figure as create_img_comparison_figure,
    create_grid,
    save_image_v2,
    load_image_v2,
)
from datasets.registry import create_dataset, list_available_datasets
from models.registry import create_model, list_available_models
from core.config_loader import load_config, validate_config, get_config_summary

from exp_script.errors import (
    ExperimentError,
    ExperimentExecutionError,
    ErrorLogger,
    get_error_logger,
)


@dataclass
class ExpContextConfig:
    """
    ExpContext 配置
    
    Attributes:
        config_path: 主配置文件路径
        device: 指定设备（如果为 None 则自动选择）
        verbose: 是否输出详细信息
        safe_run_default_return: safe_run 默认返回值
        log_errors: 是否自动记录错误
    """
    config_path: str = "config.yaml"
    device: Optional[torch.device] = None
    verbose: bool = False
    safe_run_default_return: Any = None
    log_errors: bool = True


class ExpContext:
    """
    实验脚本上下文 - 脚本的"工具箱"
    
    ExpContext 封装了项目的所有核心能力，实验脚本通过此类的
    方法访问模型、数据、数值转换等功能，禁止直接导入底层模块。
    
    核心特性：
    1. **沙盒隔离**：脚本通过 ctx 访问资源，无法直接操作底层
    2. **安全执行**：提供 safe_run() 方法包装高风险操作
    3. **自动日志**：错误自动记录到日志文件
    4. **状态追踪**：维护设备、配置等运行时状态
    
    Example:
        >>> ctx = ExpContext(config_path="config.yaml")
        >>> model = ctx.create_model()
        >>> dataset = ctx.create_dataset(split='val')
        >>> 
        >>> # 使用 safe_run 处理可能出错的操作
        >>> result = ctx.safe_run(
        ...     lambda: model.inference(dataset[0]),
        ...     default=None,
        ...     error_msg="Inference failed"
        ... )
    
    Attributes:
        config: 加载的配置字典
        device: 当前使用的设备
        rank: 当前进程 rank（DDP）
        world_size: 总进程数（DDP）
        logger: 错误日志记录器
    """
    
    def __init__(self, config_path: str = "config.yaml", verbose: bool = False):
        """
        初始化实验上下文
        
        Args:
            config_path: 主配置文件路径
            verbose: 是否输出详细信息
        """
        self._config_path = config_path
        self._verbose = verbose
        self._config: Optional[Dict[str, Any]] = None
        self._device: Optional[torch.device] = None
        self._rank: int = 0
        self._world_size: int = 1
        self._logger = get_error_logger()
        self._training_ctx: Optional[TrainingContext] = None
        
        # 初始化日志
        self._setup_logging()
        
        if self._verbose:
            self.log_info(f"ExpContext initialized with config: {config_path}")
    
    def _setup_logging(self) -> None:
        """设置日志系统"""
        self._py_logger = logging.getLogger(f"ExpContext.{id(self)}")
        if not self._py_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(name)s] %(levelname)s: %(message)s'
            )
            handler.setFormatter(formatter)
            self._py_logger.addHandler(handler)
            self._py_logger.setLevel(logging.INFO if self._verbose else logging.WARNING)
    
    # =========================================================================
    # 配置管理 API
    # =========================================================================
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载并合并配置
        
        加载顺序（后加载的优先级更高）：
        1. 模型配置 (v2/models/{model}/config.yaml)
        2. 数据集配置 (v2/datasets/{dataset}/config.yaml)
        3. 主配置 (config.yaml)
        
        Args:
            config_path: 配置文件路径，如果为 None 使用初始化时的路径
            
        Returns:
            合并后的完整配置字典
        """
        if config_path is None:
            config_path = self._config_path
        
        self._config = load_config(config_path, verbose=self._verbose)
        return self._config
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取当前配置
        
        Returns:
            当前加载的配置字典，如果未加载则先加载
        """
        if self._config is None:
            self.load_config()
        assert self._config is not None
        return self._config
    
    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        验证配置是否有效
        
        Args:
            config: 要验证的配置，如果为 None 使用当前配置
            
        Returns:
            是否有效
            
        Raises:
            ValueError: 配置无效时抛出详细错误
        """
        if config is None:
            config = self.get_config()
        return validate_config(config)
    
    def get_config_summary(self, config: Optional[Dict[str, Any]] = None) -> str:
        """
        获取配置摘要
        
        Args:
            config: 配置字典，如果为 None 使用当前配置
            
        Returns:
            格式化的配置摘要字符串
        """
        if config is None:
            config = self.get_config()
        return get_config_summary(config)
    
    # =========================================================================
    # 设备管理 API
    # =========================================================================
    
    def setup_device_and_distributed(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.device, int, int]:
        """
        初始化设备和分布式环境
        
        Args:
            config: 配置字典，如果为 None 使用当前配置
            
        Returns:
            (device, rank, world_size)
        """
        if config is None:
            config = self.get_config()
        
        self._device, self._rank, self._world_size = setup_device_and_distributed(
            config, self._rank, self._world_size
        )
        
        if self._verbose:
            self.log_info(f"Device: {self._device}, Rank: {self._rank}, World Size: {self._world_size}")
        
        return self._device, self._rank, self._world_size
    
    def get_device(self) -> torch.device:
        """
        获取当前设备
        
        Returns:
            当前使用的 torch.device
        """
        if self._device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device
    
    def get_raw_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        获取原始模型（自动处理DDP包装）
        
        Args:
            model: 可能是DDP包装的模型
            
        Returns:
            原始模型
        """
        return get_raw_model(model)
    
    def is_main_process(self, rank: Optional[int] = None) -> bool:
        """
        检查是否是主进程
        
        Args:
            rank: 当前rank（可选）
            
        Returns:
            是否是主进程(rank 0)
        """
        if rank is None:
            rank = self._rank
        return is_main_process(rank)
    
    def cleanup_resources(self) -> None:
        """清理资源，包括垃圾回收和显存清空"""
        cleanup_resources()
    
    def synchronize(self) -> None:
        """分布式同步屏障，所有进程等待直到都到达此点"""
        synchronize()
    
    def set_seed(self, seed: int) -> None:
        """
        设置随机种子，确保可复现
        
        Args:
            seed: 随机种子
        """
        set_seed(seed)
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        获取设备信息
        
        Returns:
            设备信息字典，包含 CUDA 可用性、显存信息等
        """
        return get_device_info()
    
    def wrap_model_ddp(
        self,
        model: torch.nn.Module,
        find_unused_parameters: bool = False
    ) -> torch.nn.Module:
        """
        包装模型为DDP模式
        
        Args:
            model: 原始模型
            find_unused_parameters: 是否查找未使用的参数
            
        Returns:
            DDP包装后的模型
        """
        return wrap_model_ddp(model, self.get_device(), find_unused_parameters)
    
    def all_reduce_tensor(self, tensor: torch.Tensor, op: str = 'mean') -> torch.Tensor:
        """
        分布式all reduce操作
        
        Args:
            tensor: 输入张量
            op: 'mean' 或 'sum'
            
        Returns:
            聚合后的张量
        """
        return all_reduce_tensor(tensor, op)
    
    # =========================================================================
    # 模型和数据集 API
    # =========================================================================
    
    def create_model(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ) -> Any:
        """
        根据配置创建模型
        
        Args:
            config: 配置字典，如果为 None 使用当前配置
            device: 目标设备，如果为 None 使用当前设备
            
        Returns:
            模型接口实例
        """
        if config is None:
            config = self.get_config()
        if device is None:
            device = self.get_device()
        
        return create_model(config, str(device))
    
    def create_dataset(
        self,
        config: Optional[Dict[str, Any]] = None,
        split: str = 'train'
    ) -> Any:
        """
        根据配置创建数据集
        
        Args:
            config: 配置字典，如果为 None 使用当前配置
            split: 数据分割，'train' 或 'val'
            
        Returns:
            数据集实例
        """
        if config is None:
            config = self.get_config()
        
        return create_dataset(config, split)
    
    def list_available_models(self) -> List[str]:
        """
        列出所有可用的模型
        
        Returns:
            模型名称列表
        """
        return list_available_models()
    
    def list_available_datasets(self) -> List[str]:
        """
        列出所有可用的数据集
        
        Returns:
            数据集名称列表
        """
        return list_available_datasets()
    
    # =========================================================================
    # 数值转换 API
    # =========================================================================
    
    def validate_range(
        self,
        tensor: Union[torch.Tensor, np.ndarray],
        expected_range: Tuple[float, float],
        name: str = "tensor",
        tolerance: float = 1e-5
    ) -> bool:
        """
        验证张量是否在期望范围内
        
        Args:
            tensor: 输入张量
            expected_range: (min, max) 期望范围
            name: 张量名称(用于错误信息)
            tolerance: 容差范围
            
        Returns:
            是否在范围内
            
        Raises:
            ValueError: 如果超出范围
        """
        return validate_range(tensor, expected_range, name, tolerance)
    
    def normalize_to_range(
        self,
        tensor: torch.Tensor,
        from_range: Tuple[float, float],
        to_range: Tuple[float, float] = (0.0, 1.0)
    ) -> torch.Tensor:
        """
        将张量从一个范围映射到另一个范围
        
        Args:
            tensor: 输入张量
            from_range: (min, max) 当前范围
            to_range: (min, max) 目标范围
            
        Returns:
            映射后的张量
        """
        return normalize_to_range(tensor, from_range, to_range)
    
    def clamp_and_normalize(
        self,
        tensor: torch.Tensor,
        clamp_min: float = 0.0,
        clamp_max: Optional[float] = None,
        target_range: Tuple[float, float] = (0.0, 1.0)
    ) -> torch.Tensor:
        """
        先截断再归一化到目标范围
        
        Args:
            tensor: 输入张量
            clamp_min: 截断下限
            clamp_max: 截断上限（如果为None则不限制上限）
            target_range: 目标范围
            
        Returns:
            处理后的张量
        """
        return clamp_and_normalize(tensor, clamp_min, clamp_max, target_range)
    
    def model_output_to_composite(
        self,
        model_output: torch.Tensor,
        base: torch.Tensor,
        output_range: Tuple[float, float] = (0.0, 1.0),
        clamp_negative: bool = True,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        将模型输出与base合成为最终图像
        
        Args:
            model_output: 模型输出张量（如残差、变化量）
            base: 基础图像张量
            output_range: 目标输出范围 (min, max)
            clamp_negative: 是否截断负数
            normalize: 是否归一化到output_range
            
        Returns:
            composite: 合成后的图像
        """
        return model_output_to_composite(
            model_output, base, output_range, clamp_negative, normalize
        )
    
    def composite_to_uint8(
        self,
        composite: Union[torch.Tensor, np.ndarray],
        input_range: Tuple[float, float] = (0.0, 1.0)
    ) -> np.ndarray:
        """
        将合成图像转换为uint8格式用于保存
        
        Args:
            composite: [0, 1]范围的图像张量或numpy数组
            input_range: 输入数值范围
            
        Returns:
            uint8图像 numpy array [0, 255]
        """
        return composite_to_uint8(composite, input_range)
    
    def tensor_info(self, tensor: torch.Tensor, name: str = "tensor") -> str:
        """
        获取张量的详细信息字符串
        
        Args:
            tensor: 输入张量
            name: 张量名称
            
        Returns:
            信息字符串
        """
        return tensor_info(tensor, name)
    
    # =========================================================================
    # 推理 API
    # =========================================================================
    
    def run_inference(
        self,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        """
        运行推理并返回生成器
        
        Args:
            config: 配置字典，如果为 None 使用当前配置
            checkpoint_path: 检查点路径(可选，如果为None使用随机初始化)
            max_samples: 最大样本数(可选)
            device: 设备(如果为None则自动设置)
            
        Yields:
            InferenceOutput: 包含generated, sar, optical(真值)等
        """
        if config is None:
            config = self.get_config()
        if device is None:
            device = self.get_device()
        
        yield from run_inference(config, checkpoint_path, max_samples, device)
    
    def inference_batch(
        self,
        model_interface,
        dataloader: DataLoader,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        """
        批量推理
        
        Args:
            model_interface: 模型接口
            dataloader: 数据加载器
            config: 配置，如果为 None 使用当前配置
            device: 设备，如果为 None 使用当前设备
            
        Yields:
            InferenceOutput
        """
        if config is None:
            config = self.get_config()
        if device is None:
            device = self.get_device()
        
        yield from inference_batch(model_interface, dataloader, config, device)
    
    # =========================================================================
    # 训练 API
    # =========================================================================
    
    def setup_training(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ) -> TrainingContext:
        """
        设置训练所需的组件
        
        Args:
            config: 配置字典，如果为 None 使用当前配置
            device: 设备，如果为 None 使用当前设备
            
        Returns:
            TrainingContext实例
        """
        if config is None:
            config = self.get_config()
        if device is None:
            device = self.get_device()
        
        self._training_ctx = setup_training(config, device, self._rank, self._world_size)
        return self._training_ctx
    
    def get_training_context(self) -> Optional[TrainingContext]:
        """
        获取当前训练上下文
        
        Returns:
            TrainingContext 实例，如果未设置则返回 None
        """
        return self._training_ctx
    
    def train_step(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        optimizer,
        amp_manager,
        max_norm: float = 0.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        单步训练
        
        Args:
            model: 模型接口
            batch: 数据批次
            optimizer: 优化器
            amp_manager: AMP 管理器
            max_norm: 梯度裁剪最大范数
            
        Returns:
            (loss, loss_dict)
        """
        return train_step(model, batch, optimizer, amp_manager, self.get_device(), max_norm)
    
    def run_training_epoch(
        self,
        ctx: TrainingContext,
        epoch: int,
        num_epochs: int,
        use_accumulation: bool,
        accumulation_steps: int,
        max_norm: float
    ) -> float:
        """
        运行单个训练 epoch
        
        Args:
            ctx: 训练上下文
            epoch: 当前 epoch 索引
            num_epochs: 总 epoch 数
            use_accumulation: 是否使用梯度累积
            accumulation_steps: 梯度累积步数
            max_norm: 梯度裁剪最大范数
            
        Returns:
            平均损失
        """
        return run_training_epoch(ctx, epoch, num_epochs, use_accumulation, accumulation_steps, max_norm)
    
    def run_training_loop(
        self,
        ctx: TrainingContext,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        运行完整训练循环（包含异常处理）
        
        Args:
            ctx: 训练上下文
            config: 配置，如果为 None 使用当前配置
            **kwargs: 传递给 run_training_loop 的额外参数
        """
        if config is None:
            config = self.get_config()
        
        run_training_loop(ctx, config, **kwargs)
    
    # =========================================================================
    # 验证 API
    # =========================================================================
    
    def compute_psnr(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        max_val: float = 1.0
    ) -> float:
        """
        计算PSNR
        
        Args:
            img1: 图像1 [B, C, H, W]
            img2: 图像2 [B, C, H, W]
            max_val: 最大像素值
            
        Returns:
            PSNR值
        """
        return compute_psnr(img1, img2, max_val)
    
    def compute_ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11
    ) -> float:
        """
        计算SSIM
        
        Args:
            img1: 图像1 [B, C, H, W]
            img2: 图像2 [B, C, H, W]
            window_size: 窗口大小
            
        Returns:
            SSIM值
        """
        return compute_ssim(img1, img2, window_size)
    
    def compute_metrics_batch(
        self,
        generated: torch.Tensor,
        optical: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算批次指标
        
        Args:
            generated: 生成图像 [B, C, H, W]
            optical: 真值图像 [B, C, H, W]
            
        Returns:
            指标字典 {'psnr': float, 'ssim': float}
        """
        return compute_metrics_batch(generated, optical)
    
    def run_validation(
        self,
        model,
        val_loader: DataLoader,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        save_results: bool = False,
        save_dir: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        运行验证
        
        Args:
            model: 模型接口
            val_loader: 验证数据加载器
            config: 配置，如果为 None 使用当前配置
            device: 设备，如果为 None 使用当前设备
            save_results: 是否保存结果图像
            save_dir: 保存目录
            max_samples: 最大验证样本数
            
        Returns:
            平均指标字典
        """
        if config is None:
            config = self.get_config()
        if device is None:
            device = self.get_device()
        
        return run_validation(model, val_loader, config, device, save_results, save_dir, max_samples)
    
    # =========================================================================
    # 检查点 API
    # =========================================================================
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
        save_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        保存检查点
        
        Args:
            model: 模型（可能是DDP包装）
            optimizer: 优化器
            scheduler: 学习率调度器（可选）
            epoch: 当前epoch
            metrics: 指标字典
            save_path: 保存路径
            config: 配置字典（可选）
        """
        if config is None:
            config = self.get_config()
        
        save_checkpoint_v2(model, optimizer, scheduler, epoch, metrics, save_path, config)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
            device: 加载设备
            
        Returns:
            包含state_dict等的字典
        """
        if device is None:
            device = self.get_device()
        
        return load_checkpoint_v2(checkpoint_path, str(device))
    
    def restore_model(
        self,
        model: torch.nn.Module,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True
    ) -> None:
        """
        恢复模型状态（自动处理DDP）
        
        Args:
            model: 目标模型
            state_dict: 状态字典
            strict: 是否严格匹配
        """
        restore_model_v2(model, state_dict, strict)
    
    def get_latest_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """
        获取最新的检查点路径
        
        Args:
            checkpoint_dir: 检查点目录
            
        Returns:
            最新检查点路径，如果没有则返回None
        """
        return get_latest_checkpoint(checkpoint_dir)
    
    # =========================================================================
    # 可视化 API
    # =========================================================================
    
    def setup_report_directory(self, experiment_dir) -> Path:
        """
        创建并返回 report 目录路径
        
        Args:
            experiment_dir: 实验目录路径
            
        Returns:
            report 目录路径
        """
        return setup_report_directory(Path(experiment_dir))
    
    def log_training_loss(
        self,
        log_dir,
        epoch: int,
        loss: float,
        val_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        追加写入训练日志
        
        Args:
            log_dir: 日志目录
            epoch: 当前 epoch
            loss: 平均损失
            val_metrics: 验证指标（可选）
        """
        log_loss(Path(log_dir), epoch, loss, val_metrics)
    
    def plot_loss_curve(
        self,
        log_file,
        save_path,
        title: str = "Training Loss Curve"
    ) -> bool:
        """
        从日志文件读取并绘制 loss 曲线
        
        Args:
            log_file: 训练日志文件路径
            save_path: 保存图像路径
            title: 图表标题
            
        Returns:
            是否成功绘制
        """
        return plot_loss_curve(Path(log_file), Path(save_path), title)
    
    def create_inference_comparison(
        self,
        sar: torch.Tensor,
        generated: torch.Tensor,
        optical: torch.Tensor
    ) -> np.ndarray:
        """
        创建推理对比图（SAR | Generated | Optical）
        
        Args:
            sar: SAR 输入 [1, C, H, W]
            generated: 生成图像 [1, C, H, W]
            optical: 真值图像 [1, C, H, W]
            
        Returns:
            uint8 格式的对比图 numpy 数组
        """
        return create_inference_comparison(sar, generated, optical)
    
    def create_comparison_figure(
        self,
        sample_paths: List[Path],
        save_path: Path,
        title: str = "Validation Results",
        samples_per_row: int = 5
    ) -> bool:
        """
        从已保存的单张图片创建对比报告
        
        Args:
            sample_paths: 样本图片路径列表
            save_path: 保存路径
            title: 报告标题
            samples_per_row: 每行显示多少组样本
            
        Returns:
            是否成功创建
        """
        return create_comparison_figure(sample_paths, Path(save_path), title, samples_per_row)
    
    def generate_validation_report(
        self,
        result_dir,
        report_dir,
        epoch: int,
        is_validation: bool = True
    ) -> bool:
        """
        从已保存的验证结果生成对比报告
        
        Args:
            result_dir: 验证结果保存目录
            report_dir: 报告保存目录
            epoch: 当前 epoch
            is_validation: 是否为验证
            
        Returns:
            是否成功生成
        """
        return generate_validation_report(
            Path(result_dir), Path(report_dir), epoch, is_validation
        )
    
    # =========================================================================
    # 图像操作 API
    # =========================================================================
    
    def tensor_to_numpy(
        self,
        tensor: torch.Tensor,
        channel_order: str = 'hwc'
    ) -> np.ndarray:
        """
        将tensor转换为numpy array
        
        Args:
            tensor: [B, C, H, W] 或 [C, H, W] 或 [H, W]
            channel_order: 'hwc' 或 'chw'
            
        Returns:
            numpy array
        """
        return tensor_to_numpy(tensor, channel_order)
    
    def numpy_to_tensor(
        self,
        array: np.ndarray,
        channel_order: str = 'chw',
        device: Optional[str] = None
    ) -> torch.Tensor:
        """
        将numpy array转换为tensor
        
        Args:
            array: numpy array [H, W, C] 或 [H, W]
            channel_order: 目标channel顺序
            device: 目标设备
            
        Returns:
            torch.Tensor
        """
        if device is None:
            device = str(self.get_device())
        return numpy_to_tensor(array, channel_order, device)
    
    def save_image(
        self,
        image_array: np.ndarray,
        save_path: str,
        create_dir: bool = True
    ) -> None:
        """
        保存图像
        
        Args:
            image_array: numpy array (H, W, C) 或 (H, W)
            save_path: 保存路径
            create_dir: 是否自动创建目录
        """
        save_image_v2(image_array, save_path, create_dir)
    
    def load_image(self, image_path: str, to_rgb: bool = True) -> np.ndarray:
        """
        加载图像
        
        Args:
            image_path: 图像路径
            to_rgb: 是否转换为RGB
            
        Returns:
            numpy array (H, W, C)
        """
        return load_image_v2(image_path, to_rgb)
    
    def create_grid(
        self,
        images: List[np.ndarray],
        n_cols: int = 4,
        padding: int = 2
    ) -> np.ndarray:
        """
        创建图像网格
        
        Args:
            images: 图像列表
            n_cols: 列数
            padding: 间距像素
            
        Returns:
            网格图像
        """
        return create_grid(images, n_cols, padding)
    
    # =========================================================================
    # 安全执行 API
    # =========================================================================
    
    def safe_run(
        self,
        func: Callable,
        *args,
        default: Any = None,
        error_msg: Optional[str] = None,
        log_error: bool = True,
        **kwargs
    ) -> Any:
        """
        安全执行函数，捕获异常并记录日志
        
        这是 ExpContext 的核心安全特性。包装高风险操作，
        捕获异常并记录日志，不中断主流程。
        
        Args:
            func: 要执行的函数或可调用对象
            *args: 函数的位置参数
            default: 出错时返回的默认值
            error_msg: 自定义错误信息
            log_error: 是否记录错误到日志文件
            **kwargs: 函数的关键字参数
            
        Returns:
            func 的返回值，如果出错则返回 default
            
        Example:
            >>> result = ctx.safe_run(
            ...     lambda: risky_operation(data),
            ...     default=None,
            ...     error_msg="Risky operation failed"
            ... )
            >>> if result is None:
            ...     print("Operation failed but program continues")
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = error_msg or f"Error in safe_run: {str(e)}"
            
            if self._verbose:
                self.log_warning(f"{msg} - Exception: {e}")
            
            if log_error:
                error = ExperimentExecutionError(
                    message=msg,
                    details={'original_error': str(e), 'function': func.__name__ if hasattr(func, '__name__') else 'unknown'}
                ).with_traceback_info()
                
                self._logger.log_error(error, console_output=False)
            
            return default
    
    def safe_call(
        self,
        obj: Any,
        method_name: str,
        *args,
        default: Any = None,
        error_msg: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        安全调用对象方法
        
        类似 safe_run，但是用于调用对象的方法。
        
        Args:
            obj: 目标对象
            method_name: 方法名称
            *args: 方法的位置参数
            default: 出错时返回的默认值
            error_msg: 自定义错误信息
            **kwargs: 方法的关键字参数
            
        Returns:
            方法的返回值，如果出错则返回 default
            
        Example:
            >>> result = ctx.safe_call(
            ...     model,
            ...     'get_output',
            ...     sar_input,
            ...     default=None,
            ...     error_msg="Model inference failed"
            ... )
        """
        try:
            method = getattr(obj, method_name)
            return method(*args, **kwargs)
        except Exception as e:
            msg = error_msg or f"Error calling {method_name}: {str(e)}"
            
            if self._verbose:
                self.log_warning(f"{msg} - Exception: {e}")
            
            return default
    
    # =========================================================================
    # 日志 API
    # =========================================================================
    
    def log_info(self, message: str) -> None:
        """输出信息日志"""
        print(f"[ExpContext] {message}")
        self._py_logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """输出警告日志"""
        print(f"[ExpContext] WARNING: {message}")
        self._py_logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """输出错误日志"""
        print(f"[ExpContext] ERROR: {message}", file=sys.stderr)
        self._py_logger.error(message)
    
    def log_debug(self, message: str) -> None:
        """输出调试日志"""
        self._py_logger.debug(message)


if __name__ == "__main__":
    # 测试 ExpContext
    print("Testing ExpContext...")
    
    # 创建上下文
    ctx = ExpContext(verbose=True)
    
    # 测试配置加载
    try:
        config = ctx.load_config()
        print(f"✓ Config loaded: {config.get('model', {}).get('name', 'unknown')}")
    except Exception as e:
        print(f"✗ Config load failed: {e}")
    
    # 测试设备信息
    device_info = ctx.get_device_info()
    print(f"✓ Device info: {device_info}")
    
    # 测试 safe_run
    def risky_function():
        raise ValueError("Test error")
    
    result = ctx.safe_run(
        risky_function,
        default="fallback",
        error_msg="Risky function failed (expected)"
    )
    print(f"✓ Safe run result: {result}")
    
    print("\nAll ExpContext tests completed!")