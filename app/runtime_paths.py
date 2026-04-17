from __future__ import annotations

import shutil
import sys
from pathlib import Path


def get_runtime_root() -> Path:
    try:
        containing_dir = __compiled__.containing_dir
    except NameError:
        containing_dir = None

    if containing_dir:
        return Path(containing_dir).resolve()
    if "__compiled__" in globals():
        return Path(sys.argv[0]).resolve().parent
    return Path(__file__).resolve().parents[1]


def get_models_root() -> Path:
    return get_runtime_root() / "models"


def get_tools_root() -> Path:
    return get_runtime_root() / "tools"


def find_tool_binary(name: str) -> str | None:
    tools_root = get_tools_root()
    candidates = [name]
    if sys.platform.startswith("win") and not name.lower().endswith(".exe"):
        candidates.insert(0, f"{name}.exe")

    for candidate in candidates:
        path = tools_root / candidate
        if path.is_file():
            return str(path)

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    return None
