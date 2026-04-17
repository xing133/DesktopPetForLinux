from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SegmentJobRequest:
    project_root: Path
    video_path: Path
    dancer_dir: Path
    display_height: int = 450


@dataclass(frozen=True)
class MattingEngineSpec:
    engine_id: str
    label: str
    description: str
    available: bool = True
    unavailable_reason: str | None = None
