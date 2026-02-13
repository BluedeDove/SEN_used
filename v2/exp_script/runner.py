"""
runner.py - 实验脚本运行器

负责动态加载 v2/exp_script/ 下的 .py 文件并安全执行。
提供脚本验证、错误捕获、日志记录等功能。
"""

import sys
import os
import importlib.util
import inspect
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Tuple
from dataclasses import dataclass, field
import traceback

# 支持单独运行调试：将项目根目录添加到路径
if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    v2_dir = current_file.parent.parent
    if str(v2_dir) not in sys.path:
        sys.path.insert(0, str(v2_dir))

from exp_script.context import ExpContext
from exp_script.errors import (
    ExperimentError,
    ExperimentExecutionError,
    ExperimentConfigError,
    ExperimentNotFoundError,
    ExperimentValidationError,
    ErrorLogger,
    get_error_logger,
)


@dataclass
class ExecutionResult:
    """
    实验执行结果
    
    Attributes:
        success: 是否成功
        script_name: 脚本名称
        result: 返回值（如果成功）
        error: 错误对象（如果失败）
        execution_time: 执行时间（秒）
        log_file: 日志文件路径
    """
    success: bool
    script_name: str
    result: Any = None
    error: Optional[ExperimentError] = None
    execution_time: float = 0.0
    log_file: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'script_name': self.script_name,
            'result': self.result,
            'error': str(self.error) if self.error else None,
            'execution_time': self.execution_time,
            'log_file': str(self.log_file) if self.log_file else None,
        }


@dataclass
class ExperimentScript:
    """
    实验脚本信息
    
    Attributes:
        name: 脚本名称
        path: 脚本文件路径
        module: 加载的模块对象
        run_function: run_experiment 函数
        description: 脚本描述
    """
    name: str
    path: Path
    module: Any = None
    run_function: Optional[Callable] = None
    description: str = ""


class ExperimentRunner:
    """
    实验脚本运行器
    
    负责发现、验证和执行 v2/exp_script/ 目录下的实验脚本。
    提供完整的错误处理和日志记录机制。
    
    使用流程：
    1. 创建 runner: runner = ExperimentRunner()
    2. 列出脚本: scripts = runner.list_scripts()
    3. 运行脚本: result = runner.run("my_script")
    
    Attributes:
        script_dirs: 脚本目录路径列表
        logger: 错误日志记录器
        verbose: 是否输出详细信息
    """
    
    # 脚本入口函数名
    ENTRY_FUNCTION = "run_experiment"
    
    def __init__(
        self,
        script_dirs: List[str] = None,
        verbose: bool = False
    ):
        """
        初始化实验运行器
        
        Args:
            script_dirs: 脚本目录路径列表，默认搜索 v2/exp_script/user/ 和 v2/exp_script/examples/
            verbose: 是否输出详细信息
        """
        if script_dirs is None:
            # 默认搜索用户脚本目录和示例目录
            script_dirs = ["v2/exp_script/user", "v2/exp_script/examples"]
        
        self.script_dirs = [Path(d) for d in script_dirs]
        self.verbose = verbose
        self.logger = get_error_logger()
        self._loaded_scripts: Dict[str, ExperimentScript] = {}
        
        # 确保脚本目录存在（自动创建 user 目录）
        for script_dir in self.script_dirs:
            script_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"[ExperimentRunner] Initialized with script_dirs: {self.script_dirs}")
    
    def list_scripts(self) -> List[str]:
        """
        列出所有可用的实验脚本
        
        Returns:
            脚本名称列表（不含 .py 后缀）
        """
        scripts = []
        seen = set()
        
        for script_dir in self.script_dirs:
            if not script_dir.exists():
                continue
            
            for file_path in script_dir.glob("*.py"):
                # 跳过私有文件和特殊文件
                if file_path.name.startswith("_"):
                    continue
                
                script_name = file_path.stem
                if script_name not in seen:
                    seen.add(script_name)
                    scripts.append(script_name)
        
        return sorted(scripts)
    
    def find_script_path(self, script_name: str) -> Optional[Path]:
        """
        在所有脚本目录中查找脚本
        
        Args:
            script_name: 脚本名称
            
        Returns:
            找到的脚本路径，如果未找到则返回 None
        """
        # 移除 .py 后缀（如果有）
        if script_name.endswith(".py"):
            script_name = script_name[:-3]
        
        for script_dir in self.script_dirs:
            script_path = script_dir / f"{script_name}.py"
            if script_path.exists():
                return script_path
        
        return None
    
    def get_script_path(self, script_name: str) -> Path:
        """
        获取脚本文件路径
        
        Args:
            script_name: 脚本名称（可以包含或不包含 .py 后缀）
            
        Returns:
            脚本文件路径（优先返回第一个搜索路径）
        """
        path = self.find_script_path(script_name)
        if path is not None:
            return path
        
        # 如果未找到，返回默认路径（用于错误提示）
        if script_name.endswith(".py"):
            script_name = script_name[:-3]
        
        return self.script_dirs[0] / f"{script_name}.py"
    
    def validate_script(self, script_name: str) -> Tuple[bool, Optional[str]]:
        """
        验证脚本是否符合规范
        
        检查项：
        1. 脚本文件是否存在
        2. 脚本是否包含 run_experiment 函数
        3. run_experiment 是否为可调用对象
        
        Args:
            script_name: 脚本名称
            
        Returns:
            (是否有效, 错误信息)
        """
        script_path = self.get_script_path(script_name)
        
        # 检查文件是否存在
        if not script_path.exists():
            return False, f"Script not found: {script_path}"
        
        # 尝试加载模块（不执行）
        try:
            spec = importlib.util.spec_from_file_location(
                f"exp_script_{script_name}",
                script_path
            )
            if spec is None or spec.loader is None:
                return False, f"Cannot load module spec from: {script_path}"
            
            module = importlib.util.module_from_spec(spec)
            
            # 先不执行模块，只检查源代码
            with open(script_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # 检查是否包含入口函数
            if f"def {self.ENTRY_FUNCTION}(" not in source:
                return False, f"Script missing required function: {self.ENTRY_FUNCTION}(ctx)"
            
        except Exception as e:
            return False, f"Error validating script: {e}"
        
        return True, None
    
    def load_script(self, script_name: str) -> ExperimentScript:
        """
        加载实验脚本
        
        Args:
            script_name: 脚本名称
            
        Returns:
            ExperimentScript 对象
            
        Raises:
            ExperimentNotFoundError: 脚本不存在
            ExperimentValidationError: 脚本验证失败
        """
        # 检查是否已缓存
        if script_name in self._loaded_scripts:
            return self._loaded_scripts[script_name]
        
        script_path = self.get_script_path(script_name)
        
        # 检查文件是否存在
        if not script_path.exists():
            available = self.list_scripts()
            raise ExperimentNotFoundError(
                f"Script '{script_name}' not found",
                script_name=script_name,
                details={'available_scripts': available}
            )
        
        # 验证脚本
        is_valid, error_msg = self.validate_script(script_name)
        if not is_valid:
            raise ExperimentValidationError(
                error_msg or "Script validation failed",
                script_name=script_name
            )
        
        # 加载模块
        try:
            spec = importlib.util.spec_from_file_location(
                f"exp_script_{script_name}",
                script_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot create module spec")
            
            module = importlib.util.module_from_spec(spec)
            
            # 添加脚本目录到路径（支持脚本内相对导入）
            script_parent = str(self.script_dir.parent)
            if script_parent not in sys.path:
                sys.path.insert(0, script_parent)
            
            # 执行模块
            spec.loader.exec_module(module)
            
            # 获取入口函数
            run_function = getattr(module, self.ENTRY_FUNCTION, None)
            if run_function is None or not callable(run_function):
                raise ExperimentValidationError(
                    f"Script has no callable {self.ENTRY_FUNCTION} function",
                    script_name=script_name
                )
            
            # 提取描述
            description = inspect.getdoc(run_function) or ""
            
            script_info = ExperimentScript(
                name=script_name,
                path=script_path,
                module=module,
                run_function=run_function,
                description=description
            )
            
            # 缓存
            self._loaded_scripts[script_name] = script_info
            
            if self.verbose:
                print(f"[ExperimentRunner] Loaded script: {script_name}")
            
            return script_info
            
        except Exception as e:
            if isinstance(e, ExperimentError):
                raise
            
            raise ExperimentExecutionError(
                f"Failed to load script: {e}",
                script_name=script_name
            ).with_traceback_info()
    
    def run(
        self,
        script_name: str,
        config_path: str = "config.yaml",
        **kwargs
    ) -> ExecutionResult:
        """
        运行实验脚本
        
        完整的执行流程：
        1. 加载并验证脚本
        2. 初始化 ExpContext
        3. 执行 run_experiment(ctx)
        4. 捕获所有异常
        5. 记录执行结果
        
        Args:
            script_name: 脚本名称
            config_path: 配置文件路径
            **kwargs: 传递给脚本的其他参数
            
        Returns:
            ExecutionResult 执行结果
        """
        import time
        
        start_time = time.time()
        
        # 加载脚本
        try:
            script = self.load_script(script_name)
        except ExperimentError as e:
            self.logger.log_error(e)
            return ExecutionResult(
                success=False,
                script_name=script_name,
                error=e,
                execution_time=time.time() - start_time
            )
        
        # 初始化上下文
        try:
            ctx = ExpContext(config_path=config_path, verbose=self.verbose)
        except Exception as e:
            error = ExperimentConfigError(
                f"Failed to initialize ExpContext: {e}",
                script_name=script_name
            ).with_traceback_info()
            self.logger.log_error(error)
            return ExecutionResult(
                success=False,
                script_name=script_name,
                error=error,
                execution_time=time.time() - start_time
            )
        
        # 执行脚本
        result = None
        error = None
        
        try:
            if self.verbose:
                print(f"[ExperimentRunner] Executing: {script_name}")
            
            # 调用入口函数
            if script.run_function is not None:
                result = script.run_function(ctx, **kwargs)
            else:
                raise ExperimentExecutionError(
                    "Run function is None",
                    script_name=script_name
                )
            
            if self.verbose:
                print(f"[ExperimentRunner] Completed: {script_name}")
                
        except Exception as e:
            # 捕获所有异常
            if isinstance(e, ExperimentError):
                error = e
            else:
                # 获取行号
                tb = traceback.extract_tb(sys.exc_info()[2])
                line_number = tb[-1].lineno if tb else None
                
                error = ExperimentExecutionError(
                    message=str(e),
                    script_name=script_name,
                    line_number=line_number,
                    details={'exception_type': type(e).__name__}
                ).with_traceback_info()
            
            # 记录错误
            log_file = self.logger.log_error(error)
            
            if self.verbose:
                print(f"[ExperimentRunner] Failed: {script_name} - {e}")
        
        execution_time = time.time() - start_time
        
        # 记录摘要
        summary = {
            'script_name': script_name,
            'success': error is None,
            'execution_time': f"{execution_time:.2f}s",
            'result_type': type(result).__name__ if result else None,
        }
        
        if error:
            summary['error'] = error.message
        
        log_file = self.logger.log_summary(script_name, summary)
        
        return ExecutionResult(
            success=error is None,
            script_name=script_name,
            result=result,
            error=error,
            execution_time=execution_time,
            log_file=log_file
        )
    
    def run_multiple(
        self,
        script_names: List[str],
        config_path: str = "config.yaml",
        stop_on_error: bool = False,
        **kwargs
    ) -> List[ExecutionResult]:
        """
        运行多个实验脚本
        
        Args:
            script_names: 脚本名称列表
            config_path: 配置文件路径
            stop_on_error: 出错时是否停止
            **kwargs: 传递给脚本的参数
            
        Returns:
            ExecutionResult 列表
        """
        results = []
        
        for script_name in script_names:
            result = self.run(script_name, config_path, **kwargs)
            results.append(result)
            
            if not result.success and stop_on_error:
                print(f"[ExperimentRunner] Stopping due to error in {script_name}")
                break
        
        # 打印汇总
        self._print_summary(results)
        
        return results
    
    def run_all(
        self,
        config_path: str = "config.yaml",
        stop_on_error: bool = False,
        **kwargs
    ) -> List[ExecutionResult]:
        """
        运行所有可用的实验脚本
        
        Args:
            config_path: 配置文件路径
            stop_on_error: 出错时是否停止
            **kwargs: 传递给脚本的参数
            
        Returns:
            ExecutionResult 列表
        """
        scripts = self.list_scripts()
        
        if not scripts:
            print("[ExperimentRunner] No scripts found in", self.script_dir)
            return []
        
        print(f"[ExperimentRunner] Running {len(scripts)} scripts: {scripts}")
        
        return self.run_multiple(scripts, config_path, stop_on_error, **kwargs)
    
    def _print_summary(self, results: List[ExecutionResult]) -> None:
        """打印执行汇总"""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        
        print("\n" + "=" * 70)
        print("Experiment Execution Summary")
        print("-" * 70)
        print(f"Total: {total} | Success: {successful} | Failed: {failed}")
        print("-" * 70)
        
        for result in results:
            status = "✓" if result.success else "✗"
            print(f"{status} {result.script_name} ({result.execution_time:.2f}s)")
            if not result.success and result.error:
                print(f"   Error: {result.error.message}")
        
        print("=" * 70)
    
    def get_script_info(self, script_name: str) -> Dict[str, Any]:
        """
        获取脚本信息
        
        Args:
            script_name: 脚本名称
            
        Returns:
            脚本信息字典
        """
        try:
            script = self.load_script(script_name)
            return {
                'name': script.name,
                'path': str(script.path),
                'description': script.description,
                'has_run_function': script.run_function is not None,
            }
        except ExperimentError as e:
            return {
                'name': script_name,
                'error': str(e),
            }


def run_experiment_script(
    script_name: str,
    config_path: str = "config.yaml",
    script_dir: str = "v2/exp_script",
    verbose: bool = False,
    **kwargs
) -> ExecutionResult:
    """
    便捷函数：运行单个实验脚本
    
    Args:
        script_name: 脚本名称
        config_path: 配置文件路径
        script_dir: 脚本目录
        verbose: 是否输出详细信息
        **kwargs: 传递给脚本的参数
        
    Returns:
        ExecutionResult
        
    Example:
        >>> result = run_experiment_script("my_analysis", verbose=True)
        >>> if result.success:
        ...     print(f"Result: {result.result}")
        ... else:
        ...     print(f"Failed: {result.error}")
    """
    runner = ExperimentRunner(script_dir=script_dir, verbose=verbose)
    return runner.run(script_name, config_path, **kwargs)


if __name__ == "__main__":
    # 测试 ExperimentRunner
    print("Testing ExperimentRunner...")
    
    runner = ExperimentRunner(verbose=True)
    
    # 列出脚本
    scripts = runner.list_scripts()
    print(f"✓ Found scripts: {scripts}")
    
    # 测试验证（假设有一个测试脚本）
    if scripts:
        script_name = scripts[0]
        is_valid, error = runner.validate_script(script_name)
        print(f"✓ Validation of '{script_name}': {is_valid}")
        if error:
            print(f"  Error: {error}")
        
        # 获取路径信息（不加载模块，避免独立运行时的导入问题）
        script_path = runner.find_script_path(script_name)
        print(f"✓ Script found at: {script_path}")
    
    print("\nExperimentRunner tests completed!")