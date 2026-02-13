"""
errors.py - 实验脚本系统的错误定义

提供完整的错误层次结构，确保单点错误不会导致整个程序崩溃。
所有错误会被记录到日志文件，同时在控制台输出简明信息。
"""

import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import sys


class ExperimentError(Exception):
    """
    实验脚本系统的基类异常
    
    所有实验相关的异常都继承自此类。
    提供统一的错误信息格式和日志记录支持。
    
    Attributes:
        message: 错误信息
        script_name: 发生错误的脚本名称
        details: 额外的错误详情字典
        timestamp: 错误发生时间
        traceback_str: 堆栈跟踪字符串
    """
    
    def __init__(
        self, 
        message: str, 
        script_name: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.script_name = script_name
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        self.traceback_str: Optional[str] = None
    
    def with_traceback_info(self, tb: Optional[str] = None) -> 'ExperimentError':
        """
        附加 traceback 信息
        
        Args:
            tb: traceback 字符串，如果为 None 则使用当前异常信息
            
        Returns:
            self，支持链式调用
        """
        if tb is None:
            tb = traceback.format_exc()
        self.traceback_str = tb
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式，便于日志记录和序列化
        
        Returns:
            包含错误信息的字典
        """
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'script_name': self.script_name,
            'timestamp': self.timestamp,
            'details': self.details,
            'traceback': self.traceback_str,
        }
    
    def __str__(self) -> str:
        """格式化错误信息"""
        parts = [f"[{self.__class__.__name__}] {self.message}"]
        if self.script_name:
            parts.append(f"  Script: {self.script_name}")
        if self.details:
            parts.append(f"  Details: {self.details}")
        return "\n".join(parts)


class ExperimentExecutionError(ExperimentError):
    """
    脚本执行错误
    
    当实验脚本运行过程中发生异常时抛出。
    包含完整的 traceback 和上下文信息，包括发生错误的行号。
    """
    
    def __init__(
        self, 
        message: str, 
        script_name: Optional[str] = None, 
        line_number: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, script_name, details)
        self.line_number = line_number
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，包含行号信息"""
        result = super().to_dict()
        result['line_number'] = self.line_number
        return result
    
    def __str__(self) -> str:
        """格式化错误信息，包含行号"""
        base = super().__str__()
        if self.line_number:
            return f"{base}\n  Line: {self.line_number}"
        return base


class ExperimentConfigError(ExperimentError):
    """
    实验配置错误
    
    当脚本配置无效或缺少必需配置时抛出。
    例如：配置文件格式错误、缺少必需的配置项等。
    """
    pass


class ExperimentAPITimeoutError(ExperimentError):
    """
    API 调用超时错误
    
    当 ExpContext 的方法调用超时时抛出。
    用于保护脚本免受长时间运行的操作阻塞。
    """
    
    def __init__(
        self, 
        message: str, 
        script_name: Optional[str] = None, 
        api_name: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        super().__init__(message, script_name)
        self.api_name = api_name
        self.timeout = timeout
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，包含API名称和超时时间"""
        result = super().to_dict()
        result['api_name'] = self.api_name
        result['timeout'] = self.timeout
        return result


class ExperimentNotFoundError(ExperimentError):
    """
    实验脚本未找到错误
    
    当指定的实验脚本不存在或无法访问时抛出。
    """
    pass


class ExperimentValidationError(ExperimentError):
    """
    实验脚本验证错误
    
    当脚本不符合编写规范时抛出。
    例如：缺少必需的入口函数 run_experiment。
    """
    pass


class ErrorLogger:
    """
    错误日志记录器
    
    负责将错误信息记录到文件和控制台输出。
    支持同时记录多个脚本的错误，每个脚本有独立的日志文件。
    
    Attributes:
        log_dir: 日志文件保存目录
    """
    
    def __init__(self, log_dir: str = "v2/exp_script/logs"):
        """
        初始化错误日志记录器
        
        Args:
            log_dir: 日志文件保存目录，会自动创建
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log_error(
        self, 
        error: ExperimentError, 
        console_output: bool = True
    ) -> Path:
        """
        记录错误到日志文件
        
        错误信息会被追加写入到对应的日志文件中，
        同时在控制台输出简明的错误信息。
        
        Args:
            error: 错误对象
            console_output: 是否同时输出到控制台
            
        Returns:
            日志文件路径
        """
        script_name = error.script_name or "unknown"
        log_file = self.log_dir / f"{script_name}_error.log"
        
        # 构建日志内容
        lines: List[str] = [
            "=" * 70,
            f"Experiment Error Log",
            f"Timestamp: {error.timestamp}",
            f"Script: {script_name}",
            f"Error Type: {error.__class__.__name__}",
            "-" * 70,
            f"Message: {error.message}",
        ]
        
        if error.details:
            lines.append("-" * 70)
            lines.append("Details:")
            for key, value in error.details.items():
                lines.append(f"  {key}: {value}")
        
        if hasattr(error, 'line_number') and error.line_number:
            lines.append(f"  Line Number: {error.line_number}")
        
        if hasattr(error, 'api_name') and error.api_name:
            lines.append(f"  API Name: {error.api_name}")
        
        if hasattr(error, 'timeout') and error.timeout:
            lines.append(f"  Timeout: {error.timeout}s")
        
        if error.traceback_str:
            lines.append("-" * 70)
            lines.append("Traceback:")
            lines.append(error.traceback_str)
        
        lines.append("=" * 70)
        lines.append("")  # 空行分隔
        
        # 追加写入日志文件
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("\n".join(lines) + "\n")
        except IOError as e:
            print(f"[ErrorLogger] Failed to write to log file: {e}", file=sys.stderr)
        
        # 控制台输出（简明信息）
        if console_output:
            print(f"\n[Experiment Error] {error.message}", file=sys.stderr)
            print(f"  Script: {script_name}", file=sys.stderr)
            print(f"  Full log: {log_file}", file=sys.stderr)
        
        return log_file
    
    def log_summary(
        self, 
        script_name: str, 
        results: Dict[str, Any]
    ) -> Path:
        """
        记录实验执行摘要
        
        用于记录脚本的成功执行结果和统计信息。
        
        Args:
            script_name: 脚本名称
            results: 执行结果字典
            
        Returns:
            日志文件路径
        """
        log_file = self.log_dir / f"{script_name}_summary.log"
        timestamp = datetime.now().isoformat()
        
        lines: List[str] = [
            "=" * 70,
            f"Experiment Summary - {script_name}",
            f"Timestamp: {timestamp}",
            "-" * 70,
        ]
        
        for key, value in results.items():
            lines.append(f"{key}: {value}")
        
        lines.append("=" * 70)
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines) + "\n")
        except IOError as e:
            print(f"[ErrorLogger] Failed to write summary: {e}", file=sys.stderr)
        
        return log_file
    
    def clear_logs(self, script_name: Optional[str] = None) -> int:
        """
        清理日志文件
        
        Args:
            script_name: 如果指定，只清理该脚本的日志；
                        否则清理所有日志
                        
        Returns:
            删除的文件数量
        """
        count = 0
        pattern = f"{script_name}_*.log" if script_name else "*.log"
        
        for log_file in self.log_dir.glob(pattern):
            try:
                log_file.unlink()
                count += 1
            except IOError:
                pass
        
        return count


# 全局错误日志记录器实例
_default_error_logger: Optional[ErrorLogger] = None


def get_error_logger() -> ErrorLogger:
    """
    获取全局错误日志记录器实例
    
    如果实例不存在，会自动创建一个默认实例。
    
    Returns:
        ErrorLogger 实例
    """
    global _default_error_logger
    if _default_error_logger is None:
        _default_error_logger = ErrorLogger()
    return _default_error_logger


def set_error_logger(logger: ErrorLogger) -> None:
    """
    设置全局错误日志记录器
    
    用于替换默认的日志记录器，例如使用自定义配置。
    
    Args:
        logger: 新的 ErrorLogger 实例
    """
    global _default_error_logger
    _default_error_logger = logger