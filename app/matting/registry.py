from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .base import MattingEngineSpec, SegmentJobRequest
from .legacy_rvm_torch import LegacyRvmTorchWorker
from .model_paths import get_rvm_model_path, get_u2net_model_path
from .windows_onnx_common import describe_windows_onnx_unavailable_reason
from .windows_rvm_onnx import WindowsRvmOnnxWorker
from .windows_u2net_onnx import WindowsU2netOnnxWorker


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _has_torch_runtime() -> bool:
    return _has_module("torch") and _has_module("torchvision")


def _missing_model_reason(model_path: Path) -> str | None:
    if model_path.is_file():
        return None
    return f"缺少模型文件：{model_path}"


def get_engines_for_current_platform(
    project_root: Path | None = None,
) -> list[MattingEngineSpec]:
    if _is_windows():
        onnx_unavailable_reason = describe_windows_onnx_unavailable_reason()
        rvm_model_path = (
            get_rvm_model_path(project_root)
            if project_root is not None
            else Path("models/rvm/rvm_mobilenetv3_fp32.onnx")
        )
        u2net_model_path = (
            get_u2net_model_path(project_root)
            if project_root is not None
            else Path("models/u2net/u2net_human_seg.onnx")
        )
        rvm_reason = onnx_unavailable_reason or (
            _missing_model_reason(rvm_model_path) if project_root is not None else None
        )
        u2net_reason = onnx_unavailable_reason or (
            _missing_model_reason(u2net_model_path) if project_root is not None else None
        )
        legacy_available = _has_torch_runtime()
        return [
            MattingEngineSpec(
                engine_id="rvm_onnx_directml",
                label="RVM (ONNX Runtime DirectML)",
                description=(
                    "Windows 新链路。"
                    f"\n模型路径：{rvm_model_path}"
                ),
                available=rvm_reason is None,
                unavailable_reason=rvm_reason,
            ),
            MattingEngineSpec(
                engine_id="u2net_onnx_directml",
                label="U2Net Human Seg (ONNX Runtime DirectML)",
                description=(
                    "Windows 新链路，适合人物分割。"
                    f"\n模型路径：{u2net_model_path}"
                ),
                available=u2net_reason is None,
                unavailable_reason=u2net_reason,
            ),
            MattingEngineSpec(
                engine_id="rvm_torch_legacy",
                label="RVM (PyTorch, legacy)",
                description="当前可工作的旧实现，保留作为回退选项。",
                available=legacy_available,
                unavailable_reason=(
                    None if legacy_available else "未安装 torch/torchvision。"
                ),
            ),
        ]

    if _is_linux():
        return [
            MattingEngineSpec(
                engine_id="rvm_torch_linux",
                label="RVM (Current)",
                description="Linux 继续沿用当前抠图实现，保持现有行为不变。",
            ),
        ]

    return [
        MattingEngineSpec(
            engine_id="unsupported_platform",
            label="Unsupported Platform",
            description="当前平台暂未接入抠图引擎。",
            available=False,
            unavailable_reason="当前平台暂未支持抠图。",
        )
    ]


def get_engine_spec(
    engine_id: str,
    project_root: Path | None = None,
) -> MattingEngineSpec | None:
    for spec in get_engines_for_current_platform(project_root):
        if spec.engine_id == engine_id:
            return spec
    return None


def create_segment_worker(engine_id: str, request: SegmentJobRequest):
    spec = get_engine_spec(engine_id, request.project_root)
    if spec is None:
        raise ValueError(f"Unknown matting engine: {engine_id}")
    if not spec.available:
        reason = spec.unavailable_reason or f"Matting engine '{engine_id}' is unavailable."
        raise RuntimeError(reason)

    if engine_id in {"rvm_torch_legacy", "rvm_torch_linux"}:
        return LegacyRvmTorchWorker(request)
    if engine_id == "rvm_onnx_directml":
        return WindowsRvmOnnxWorker(request)
    if engine_id == "u2net_onnx_directml":
        return WindowsU2netOnnxWorker(request)

    raise ValueError(f"Unsupported matting engine: {engine_id}")
