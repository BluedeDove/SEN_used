"""
base.py - 模型接口基类

定义所有模型必须实现的接口，确保一致性和可扩展性。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
from enum import Enum
import torch


@dataclass
class ModelDebugInfo:
    """
    模型调试信息

    Attributes:
        component_name: 组件名称
        status: 状态 ('OK', 'WARN', 'FAIL')
        message: 详细信息
        values: 相关数值（可选）
    """
    component_name: str
    status: str  # 'OK', 'WARN', 'FAIL'
    message: str
    values: Optional[Dict[str, Any]] = None


@dataclass
class ModelDebugReport:
    """
    模型调试报告

    Attributes:
        model_name: 模型名称
        overall_status: 整体状态
        tests: 各测试项结果列表
        summary: 摘要信息
    """
    model_name: str
    overall_status: str  # 'PASSED', 'FAILED', 'WARNING'
    tests: List[ModelDebugInfo]
    summary: str


class CompositeMethod(Enum):
    """合成方法枚举"""
    DIRECT = "direct"           # 直接输出，不需要合成
    ADD = "add"                 # 直接相加
    ADD_THEN_CLAMP = "add_then_clamp"  # 相加后截断并归一化
    MULTIPLY = "multiply"       # 相乘


@dataclass
class ModelOutput:
    """
    模型输出的标准格式

    Attributes:
        generated: 生成的图像 [B, C, H, W]
        output_range: 实际输出范围
        intermediate: 中间结果(可选)
        metadata: 额外元数据(可选)
    """
    generated: torch.Tensor  # 生成的图像 [B, C, H, W]
    output_range: Tuple[float, float]  # 实际输出范围
    intermediate: Optional[Dict[str, torch.Tensor]] = None  # 中间结果
    metadata: Optional[Dict[str, Any]] = None  # 额外元数据


class BaseModelInterface(ABC):
    """
    所有模型必须实现的接口基类

    子类必须实现:
    - build_model(): 构建实际模型
    - get_output(): 推理输出
    - get_output_range(): 输出范围
    - get_composite_method(): 合成方法
    """

    def __init__(self, config: dict):
        """
        初始化模型接口

        Args:
            config: 配置字典
        """
        self.config = config
        self._model: Optional[torch.nn.Module] = None
        self._output_range = self._get_default_output_range()
        self._device = torch.device('cpu')

    @abstractmethod
    def build_model(self, device: str = 'cpu') -> torch.nn.Module:
        """
        构建并返回实际的PyTorch模型

        Args:
            device: 目标设备

        Returns:
            PyTorch模型实例
        """
        pass

    @abstractmethod
    def get_output(self, sar: torch.Tensor, config: dict) -> ModelOutput:
        """
        获取模型输出 - 这是核心接口

        Args:
            sar: 输入SAR图像 [B, C, H, W]
            config: 配置字典

        Returns:
            ModelOutput: 标准化的模型输出
        """
        pass

    @abstractmethod
    def get_output_range(self) -> Tuple[float, float]:
        """
        返回模型输出的数值范围

        Returns:
            (min, max) 元组
        """
        return self._output_range

    @abstractmethod
    def get_composite_method(self) -> CompositeMethod:
        """
        返回合成方法

        Returns:
            CompositeMethod枚举值
        """
        pass

    def _get_default_output_range(self) -> Tuple[float, float]:
        """
        获取默认输出范围（从配置）

        Returns:
            (min, max) 元组
        """
        return tuple(self.config.get('model', {}).get('output_range', [-1.0, 1.0]))

    def to(self, device: torch.device):
        """
        将模型移动到指定设备

        Args:
            device: 目标设备
        """
        self._device = device
        if self._model is not None:
            self._model = self._model.to(device)

    def train(self, mode: bool = True):
        """
        设置训练/评估模式

        Args:
            mode: True为训练模式，False为评估模式
        """
        if self._model is not None:
            self._model.train(mode)

    def eval(self):
        """设置为评估模式"""
        self.train(False)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        获取模型状态字典

        Returns:
            状态字典
        """
        if self._model is not None:
            return self._model.state_dict()
        return {}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """
        加载模型状态

        Args:
            state_dict: 状态字典
            strict: 是否严格匹配
        """
        if self._model is not None:
            self._model.load_state_dict(state_dict, strict=strict)

    def count_parameters(self) -> Dict[str, int]:
        """
        计算模型参数量

        Returns:
            参数字典，包含 'total' 和按组件分类的数量
        """
        if self._model is None:
            return {'total': 0}

        total = sum(p.numel() for p in self._model.parameters())
        return {'total': total}

    def debug(self, device: torch.device, verbose: bool = False) -> ModelDebugReport:
        """
        运行模型调试测试

        子类应重写此方法以实现特定的调试逻辑。
        基类提供默认实现，只测试基本功能。

        Args:
            device: 运行设备
            verbose: 是否显示详细信息

        Returns:
            ModelDebugReport: 调试报告
        """
        tests = []

        # 测试1: 模型是否已构建
        if self._model is None:
            tests.append(ModelDebugInfo(
                component_name="Model Build",
                status="FAIL",
                message="Model not built. Call build_model() first."
            ))
            return ModelDebugReport(
                model_name=self.__class__.__name__,
                overall_status="FAILED",
                tests=tests,
                summary="Model not built"
            )

        tests.append(ModelDebugInfo(
            component_name="Model Build",
            status="OK",
            message=f"Model built with {self.count_parameters()['total']:,} parameters"
        ))

        # 测试2: 基本推理
        try:
            test_input = torch.rand(1, 1, 64, 64).to(device)
            with torch.no_grad():
                output = self.get_output(test_input, self.config)
            tests.append(ModelDebugInfo(
                component_name="Basic Inference",
                status="OK",
                message=f"Output shape: {list(output.generated.shape)}",
                values={"shape": list(output.generated.shape)}
            ))
        except Exception as e:
            tests.append(ModelDebugInfo(
                component_name="Basic Inference",
                status="FAIL",
                message=f"Inference failed: {str(e)}"
            ))

        # 确定整体状态
        has_fail = any(t.status == "FAIL" for t in tests)
        has_warn = any(t.status == "WARN" for t in tests)
        overall = "FAILED" if has_fail else ("WARNING" if has_warn else "PASSED")

        return ModelDebugReport(
            model_name=self.__class__.__name__,
            overall_status=overall,
            tests=tests,
            summary=f"Basic tests completed: {len(tests)} test(s)"
        )


if __name__ == "__main__":
    # 测试
    print("Testing base.py...")

    # 测试 ModelOutput dataclass
    output = ModelOutput(
        generated=torch.rand(2, 3, 64, 64),
        output_range=(-1.0, 1.0),
        intermediate={'feature': torch.rand(2, 64, 32, 32)}
    )
    print(f"ModelOutput created: {output.output_range}")

    # 测试 CompositeMethod
    print(f"Available methods: {[m.value for m in CompositeMethod]}")

    print("All tests passed!")
