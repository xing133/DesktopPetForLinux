from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal

from .base import SegmentJobRequest
from .model_paths import get_rvm_model_path
from .windows_onnx_common import require_directml_session


class WindowsRvmOnnxWorker(QThread):
    progress = Signal(int, int)
    stage_changed = Signal(str)
    finished_ok = Signal()
    finished_err = Signal(str)

    def __init__(
        self,
        request: SegmentJobRequest,
        *,
        variant: str = "mobilenetv3",
        model_filename: str = "rvm_mobilenetv3_fp32.onnx",
    ) -> None:
        super().__init__()
        self._request = request
        self._variant = variant
        self._model_filename = model_filename
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        from src.rvm_matting import MattingCancelled

        try:
            self._do_segment()
        except MattingCancelled:
            self._cleanup_output_dir()
            self.finished_err.emit("已取消")
        except Exception as exc:
            self._cleanup_output_dir()
            self.finished_err.emit(str(exc))

    def _cleanup_output_dir(self) -> None:
        if self._request.dancer_dir.exists():
            shutil.rmtree(str(self._request.dancer_dir), ignore_errors=True)

    def _model_path(self) -> Path:
        return get_rvm_model_path(self._request.project_root)

    def _do_segment(self) -> None:
        from src.rvm_matting import (
            LANCZOS,
            MattingCancelled,
            auto_downsample_ratio,
            compute_output_size,
            ensure_clean_output_dir,
            iter_raw_frames,
            probe_video,
            require_binary,
        )

        def is_cancelled() -> bool:
            return self._cancelled

        if is_cancelled():
            raise MattingCancelled("Cancelled before start.")

        model_path = self._model_path()
        if not model_path.is_file():
            raise RuntimeError(
                "未找到 RVM ONNX 模型文件："
                f"{model_path}\n"
                "请将官方 ONNX 模型放到 models/rvm/ 目录。"
            )

        ffmpeg_bin = require_binary("ffmpeg")
        ffprobe_bin = require_binary("ffprobe")
        ensure_clean_output_dir(self._request.dancer_dir, overwrite=True)

        self.stage_changed.emit("正在读取视频信息…")
        fps, frame_count, source_width, source_height = probe_video(
            ffprobe_bin, self._request.video_path
        )
        output_width, output_height = compute_output_size(
            source_width,
            source_height,
            self._request.display_height,
        )
        downsample_ratio = auto_downsample_ratio(output_height, output_width)

        if is_cancelled():
            raise MattingCancelled("Cancelled before model load.")

        self.stage_changed.emit(f"正在加载 RVM ONNX 模型 {self._variant}（DirectML）…")
        session = require_directml_session(model_path)

        if is_cancelled():
            raise MattingCancelled("Cancelled after model load.")

        self.stage_changed.emit(f"正在使用 DirectML 处理 {frame_count} 帧…")

        rec = [np.zeros((1, 1, 1, 1), dtype=np.float32) for _ in range(4)]
        dsr = np.asarray([downsample_ratio], dtype=np.float32)
        processed = 0

        for index, frame_np in enumerate(
            iter_raw_frames(
                ffmpeg_bin,
                self._request.video_path,
                source_width,
                source_height,
            ),
            start=1,
        ):
            if is_cancelled():
                raise MattingCancelled("Cancelled while processing frames.")

            image = Image.fromarray(frame_np)
            if (output_width, output_height) != (source_width, source_height):
                image = image.resize((output_width, output_height), LANCZOS)

            src = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
            fgr, pha, *rec = session.run(
                [],
                {
                    "src": src.astype(np.float32, copy=False),
                    "r1i": rec[0],
                    "r2i": rec[1],
                    "r3i": rec[2],
                    "r4i": rec[3],
                    "downsample_ratio": dsr,
                },
            )

            foreground = np.transpose(fgr[0], (1, 2, 0))
            alpha = pha[0, 0]
            foreground = foreground * (alpha[..., None] > 0)
            rgba = np.concatenate([foreground, alpha[..., None]], axis=2)
            rgba = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(rgba, "RGBA").save(
                self._request.dancer_dir / f"frame_{index:04d}.png",
                "PNG",
            )

            processed = index
            self.progress.emit(processed, frame_count)

        if processed == 0:
            raise RuntimeError("未读取到任何视频帧，请检查视频文件。")

        metadata = {
            "fps": fps,
            "frame_count": processed,
            "width": output_width,
            "height": output_height,
            "source_video": str(self._request.video_path),
            "matting_backend": "RobustVideoMattingONNXDirectML",
            "rvm_variant": self._variant,
            "onnx_model": str(model_path),
            "execution_provider": "DmlExecutionProvider",
            "downsample_ratio": downsample_ratio,
            "output_type": "png_sequence",
        }
        metadata_path = self._request.dancer_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.finished_ok.emit()
