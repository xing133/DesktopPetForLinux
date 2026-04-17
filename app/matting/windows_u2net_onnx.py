from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal

from .base import SegmentJobRequest
from .model_paths import get_u2net_model_path
from .windows_onnx_common import require_directml_session


class WindowsU2netOnnxWorker(QThread):
    progress = Signal(int, int)
    stage_changed = Signal(str)
    finished_ok = Signal()
    finished_err = Signal(str)

    def __init__(
        self,
        request: SegmentJobRequest,
        *,
        model_name: str = "u2net_human_seg",
        model_filename: str = "u2net_human_seg.onnx",
    ) -> None:
        super().__init__()
        self._request = request
        self._model_name = model_name
        self._model_filename = model_filename
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            self._do_segment()
        except Exception as exc:
            self._cleanup_output_dir()
            self.finished_err.emit(str(exc))

    def _cleanup_output_dir(self) -> None:
        if self._request.dancer_dir.exists():
            shutil.rmtree(str(self._request.dancer_dir), ignore_errors=True)

    def _model_path(self) -> Path:
        return get_u2net_model_path(self._request.project_root)

    def _predict_mask(self, session, image: Image.Image) -> Image.Image:
        resized = image.convert("RGB").resize((320, 320), Image.Resampling.LANCZOS)
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
        normalized = np.empty_like(image_array, dtype=np.float32)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        for channel in range(3):
            normalized[:, :, channel] = (image_array[:, :, channel] - mean[channel]) / std[channel]

        src = normalized.transpose((2, 0, 1))[None].astype(np.float32, copy=False)
        input_name = session.get_inputs()[0].name
        ort_outs = session.run(None, {input_name: src})
        pred = ort_outs[0][:, 0, :, :]
        max_value = np.max(pred)
        min_value = np.min(pred)
        if max_value - min_value < 1e-6:
            normalized_pred = np.zeros_like(pred)
        else:
            normalized_pred = (pred - min_value) / (max_value - min_value)
        mask = np.squeeze(normalized_pred)
        result = Image.fromarray((mask.clip(0, 1) * 255).astype("uint8"), mode="L")
        return result.resize(image.size, Image.Resampling.LANCZOS)

    def _do_segment(self) -> None:
        from src.rvm_matting import (
            MattingCancelled,
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
                "未找到 U2Net ONNX 模型文件："
                f"{model_path}\n"
                "请将 u2net_human_seg.onnx 放到 models/u2net/ 目录。"
            )

        ffmpeg_bin = require_binary("ffmpeg")
        ffprobe_bin = require_binary("ffprobe")
        ensure_clean_output_dir(self._request.dancer_dir, overwrite=True)

        self.stage_changed.emit("正在读取视频信息…")
        fps, frame_count, source_width, source_height = probe_video(
            ffprobe_bin,
            self._request.video_path,
        )
        output_width, output_height = compute_output_size(
            source_width,
            source_height,
            self._request.display_height,
        )

        if is_cancelled():
            raise MattingCancelled("Cancelled before model load.")

        self.stage_changed.emit("正在加载 U2Net ONNX 模型（DirectML）…")
        session = require_directml_session(model_path)

        if is_cancelled():
            raise MattingCancelled("Cancelled after model load.")

        self.stage_changed.emit(f"正在使用 DirectML 处理 {frame_count} 帧…")

        processed = 0
        for index, frame_np in enumerate(
            iter_raw_frames(
                ffmpeg_bin,
                self._request.video_path,
                source_width,
                source_height,
                output_width=output_width,
                output_height=output_height,
            ),
            start=1,
        ):
            if is_cancelled():
                raise MattingCancelled("Cancelled while processing frames.")

            rgb_image = Image.fromarray(frame_np, "RGB")
            alpha_mask = self._predict_mask(session, rgb_image)
            rgba_image = rgb_image.copy()
            rgba_image.putalpha(alpha_mask)
            rgba_image.save(
                self._request.dancer_dir / f"frame_{index:04d}.png",
                "PNG",
                compress_level=1,
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
            "matting_backend": "U2NetONNXDirectML",
            "u2net_model": self._model_name,
            "onnx_model": str(model_path),
            "execution_provider": "DmlExecutionProvider",
            "output_type": "png_sequence",
        }
        metadata_path = self._request.dancer_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.finished_ok.emit()
