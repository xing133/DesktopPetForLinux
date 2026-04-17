from __future__ import annotations

import importlib.util
from pathlib import Path


def has_onnxruntime_module() -> bool:
    return importlib.util.find_spec("onnxruntime") is not None


def has_directml_provider() -> bool:
    if not has_onnxruntime_module():
        return False

    try:
        import onnxruntime as ort
    except Exception:
        return False

    try:
        return "DmlExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def require_directml_session(model_path: Path):
    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    if "DmlExecutionProvider" not in available_providers:
        raise RuntimeError(
            "当前环境未提供 DirectML Execution Provider。"
            "请安装 onnxruntime-directml，并确认是 Windows 10/11 图形环境。"
        )

    return ort.InferenceSession(
        str(model_path),
        providers=["DmlExecutionProvider"],
    )


def describe_windows_onnx_unavailable_reason() -> str | None:
    if not has_onnxruntime_module():
        return "未安装 onnxruntime-directml。"
    if not has_directml_provider():
        return "当前 onnxruntime 不包含 DirectML Execution Provider。"
    return None
