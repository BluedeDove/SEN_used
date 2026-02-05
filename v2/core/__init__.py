"""
v2.core - 元功能和流程封装模块

包含万年不变的基础功能，被所有上层模块依赖。
"""

from .numeric_ops import (
    validate_range,
    model_output_to_composite,
    composite_to_uint8,
    denormalize,
    normalize_to_range,
    clamp_and_normalize
)

from .device_ops import (
    setup_device_and_distributed,
    get_raw_model,
    is_main_process,
    cleanup_resources,
    synchronize,
    set_seed,
    get_device_info
)

from .checkpoint_ops import (
    save_checkpoint_v2,
    load_checkpoint_v2,
    restore_model_v2,
    restore_optimizer_v2
)

__all__ = [
    # numeric_ops
    'validate_range',
    'model_output_to_composite',
    'composite_to_uint8',
    'denormalize',
    'normalize_to_range',
    'clamp_and_normalize',
    # device_ops
    'setup_device_and_distributed',
    'get_raw_model',
    'is_main_process',
    'cleanup_resources',
    'synchronize',
    'set_seed',
    'get_device_info',
    # checkpoint_ops
    'save_checkpoint_v2',
    'load_checkpoint_v2',
    'restore_model_v2',
    'restore_optimizer_v2',
]
