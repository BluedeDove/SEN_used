"""
exp_script - 高自由度实验脚本系统

提供沙盒化的实验环境，支持动态加载脚本并安全执行。
所有脚本通过 ExpContext 访问项目资源，禁止直接导入底层模块。

Example:
    >>> from exp_script import ExpContext, ExperimentRunner
    >>> runner = ExperimentRunner()
    >>> result = runner.run("my_experiment")
"""

from exp_script.context import ExpContext
from exp_script.runner import ExperimentRunner
from exp_script.errors import (
    ExperimentError,
    ExperimentExecutionError,
    ExperimentConfigError,
    ExperimentAPITimeoutError,
    ExperimentNotFoundError,
    ExperimentValidationError,
    ErrorLogger,
    get_error_logger,
    set_error_logger,
)

__all__ = [
    'ExpContext',
    'ExperimentRunner',
    'ExperimentError',
    'ExperimentExecutionError',
    'ExperimentConfigError',
    'ExperimentAPITimeoutError',
    'ExperimentNotFoundError',
    'ExperimentValidationError',
    'ErrorLogger',
    'get_error_logger',
    'set_error_logger',
]

__version__ = '1.0.0'