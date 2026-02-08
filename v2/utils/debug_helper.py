"""
debug_helper.py - 调试辅助工具

为支持单独运行调试的模块提供统一的路径设置工具。
使用方式：在需要支持单独运行的模块开头导入并调用 setup_debug_path()

Example:
    from utils.debug_helper import setup_debug_path
    setup_debug_path(__file__)
    
    # 然后正常导入项目模块
    from models.base import ...
"""

import sys
from pathlib import Path


def setup_debug_path(current_file: str, target_dir_name: str = "v2") -> bool:
    """
    设置调试路径，将目标目录添加到 sys.path
    
    当直接运行文件时（__name__ == "__main__"），自动添加项目路径。
    
    Args:
        current_file: 当前文件的 __file__ 变量
        target_dir_name: 目标目录名，默认为 "v2"
        
    Returns:
        bool: 是否添加了路径
    """
    # 只有直接运行时才设置路径（避免重复设置）
    if sys.path[0].endswith(target_dir_name):
        return False
        
    current = Path(current_file).resolve()
    
    # 向上查找目标目录
    for parent in [current.parent] + list(current.parents):
        if parent.name == target_dir_name:
            target_dir = parent
            break
    else:
        # 如果没找到 v2，尝试从项目根目录找
        for parent in [current.parent] + list(current.parents):
            potential = parent / target_dir_name
            if potential.exists() and potential.is_dir():
                target_dir = potential
                break
        else:
            return False
    
    # 添加到路径开头
    target_str = str(target_dir)
    if target_str not in sys.path:
        sys.path.insert(0, target_str)
        return True
    
    return False


def setup_debug_path_if_main(current_file: str, target_dir_name: str = "v2") -> bool:
    """
    仅在 __name__ == "__main__" 时设置调试路径
    
    Args:
        current_file: 当前文件的 __file__ 变量
        target_dir_name: 目标目录名，默认为 "v2"
        
    Returns:
        bool: 是否添加了路径
    """
    if __name__ != "__main__":
        return False
    return setup_debug_path(current_file, target_dir_name)
