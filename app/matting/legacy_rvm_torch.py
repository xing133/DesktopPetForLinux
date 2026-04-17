from __future__ import annotations

import shutil

from PySide6.QtCore import QThread, Signal

from .base import SegmentJobRequest


class LegacyRvmTorchWorker(QThread):
    progress = Signal(int, int)
    stage_changed = Signal(str)
    finished_ok = Signal()
    finished_err = Signal(str)

    def __init__(
        self,
        request: SegmentJobRequest,
        *,
        rvm_variant: str = "mobilenetv3",
        device: str = "auto",
    ) -> None:
        super().__init__()
        self._request = request
        self._rvm_variant = rvm_variant
        self._device = device
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

    def _do_segment(self) -> None:
        from src.rvm_matting import run_matting

        run_matting(
            input_path=self._request.video_path,
            frames_dir=self._request.dancer_dir,
            variant=self._rvm_variant,
            device=self._device,
            display_height=self._request.display_height,
            overwrite=True,
            progress_callback=self.progress.emit,
            stage_callback=self.stage_changed.emit,
            cancel_requested=lambda: self._cancelled,
        )
        self.finished_ok.emit()
