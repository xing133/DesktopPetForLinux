from __future__ import annotations

from pathlib import Path


def get_rvm_model_path(project_root: Path) -> Path:
    rvm_dir = project_root / "models" / "rvm"
    fp16_path = rvm_dir / "rvm_mobilenetv3_fp16.onnx"
    if fp16_path.is_file():
        return fp16_path
    return rvm_dir / "rvm_mobilenetv3_fp32.onnx"


def get_u2net_model_path(project_root: Path) -> Path:
    return project_root / "models" / "u2net" / "u2net_human_seg.onnx"
