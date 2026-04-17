from __future__ import annotations

import re
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QProcess, Qt, QThread, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.matting import (
    SegmentJobRequest,
    create_segment_worker,
    get_engine_spec,
    get_engines_for_current_platform,
)
from app.runtime_paths import find_tool_binary, get_runtime_root


class ImportState(str, Enum):
    IMPORT_SOURCE = "import_source"
    URL_INPUT = "url_input"
    DOWNLOADING = "downloading"
    TRANSCODING = "transcoding"
    PREVIEW_READY = "preview_ready"
    NAMING = "naming"
    SEGMENTING = "segmenting"
    DONE = "done"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# 分割 Worker（在独立线程中跑 AI 抠图）
# ---------------------------------------------------------------------------

class _SegmentWorker(QThread):
    progress = Signal(int, int)   # (current_frame, total_frames)
    stage_changed = Signal(str)   # 阶段描述文字
    finished_ok = Signal()
    finished_err = Signal(str)    # 错误信息

    def __init__(
        self,
        video_path: Path,
        dancer_dir: Path,
        display_height: int = 450,
        rvm_variant: str = "mobilenetv3",
        device: str = "auto",
    ):
        super().__init__()
        self._video_path = video_path
        self._dancer_dir = dancer_dir
        self._display_height = display_height
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
        except Exception as e:
            self._cleanup_output_dir()
            self.finished_err.emit(str(e))

    def _cleanup_output_dir(self) -> None:
        if self._dancer_dir.exists():
            shutil.rmtree(str(self._dancer_dir), ignore_errors=True)

    def _do_segment(self) -> None:
        from src.rvm_matting import run_matting

        run_matting(
            input_path=self._video_path,
            frames_dir=self._dancer_dir,
            variant=self._rvm_variant,
            device=self._device,
            display_height=self._display_height,
            overwrite=True,
            progress_callback=self.progress.emit,
            stage_callback=self.stage_changed.emit,
            cancel_requested=lambda: self._cancelled,
        )

        self.finished_ok.emit()


# ---------------------------------------------------------------------------
# 主向导窗口
# ---------------------------------------------------------------------------

class AddWifeWizard(QWidget):
    def __init__(
        self,
        on_dancer_ready: Callable[[str], None] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("添加一个老婆")
        self.setMinimumSize(780, 520)

        self._on_dancer_ready = on_dancer_ready

        self._project_root = get_runtime_root()
        self._workspace_dir = self._project_root / "workspace"
        self._raw_dir = self._workspace_dir / "raw"
        self._mp4_dir = self._workspace_dir / "mp4"
        self._dancer_root = self._project_root / "dancer"
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._mp4_dir.mkdir(parents=True, exist_ok=True)

        self._is_windows = sys.platform.startswith("win")
        self._matting_engines = get_engines_for_current_platform(self._project_root)
        self._selected_matting_engine_id = next(
            (
                spec.engine_id
                for spec in self._matting_engines
                if spec.available
            ),
            self._matting_engines[0].engine_id if self._matting_engines else "",
        )

        self._state = ImportState.IMPORT_SOURCE
        self._job_id: str | None = None
        self._pending_input_video: Path | None = None
        self._preview_video: Path | None = None
        self._dancer_name: str = ""
        self._segment_worker = None

        self._download_proc: QProcess | None = None
        self._transcode_proc: QProcess | None = None
        self._transcode_duration_sec: float = 0.0

        self._status_label = QLabel()
        self._status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self._stack = QStackedWidget()

        self._page_source = self._build_source_page()
        self._page_url = self._build_url_page()
        self._page_progress = self._build_progress_page()
        self._page_preview = self._build_preview_page()
        self._page_naming = self._build_naming_page()
        self._page_done = self._build_done_page()

        self._stack.addWidget(self._page_source)
        self._stack.addWidget(self._page_url)
        self._stack.addWidget(self._page_progress)
        self._stack.addWidget(self._page_preview)
        self._stack.addWidget(self._page_naming)
        self._stack.addWidget(self._page_done)

        root = QVBoxLayout(self)
        root.addWidget(self._status_label)
        root.addWidget(self._stack)

        self._set_state(ImportState.IMPORT_SOURCE)

    # ------------------------------------------------------------------
    # 页面构建
    # ------------------------------------------------------------------

    def _build_source_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("请选择导入方式")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        hint = QLabel("你可以从本地选择视频，或者输入网络链接下载。")

        local_btn = QPushButton("本地视频")
        local_btn.setMinimumHeight(40)
        local_btn.clicked.connect(self._on_choose_local_video)

        net_btn = QPushButton("网络下载")
        net_btn.setMinimumHeight(40)
        net_btn.clicked.connect(lambda: self._set_state(ImportState.URL_INPUT))

        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addSpacing(12)
        layout.addWidget(local_btn)
        layout.addWidget(net_btn)
        layout.addStretch(1)
        return page

    def _build_url_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("输入视频链接")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        hint = QLabel("支持 bilibili / douyin / youtube")

        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText("https://...")

        btn_row = QHBoxLayout()
        back_btn = QPushButton("返回")
        back_btn.clicked.connect(lambda: self._set_state(ImportState.IMPORT_SOURCE))

        next_btn = QPushButton("开始下载")
        next_btn.clicked.connect(self._on_url_next)

        btn_row.addWidget(back_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(next_btn)

        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addSpacing(10)
        layout.addWidget(self._url_edit)
        layout.addLayout(btn_row)
        layout.addStretch(1)
        return page

    def _build_progress_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self._progress_title = QLabel("处理中")
        self._progress_title.setStyleSheet("font-size: 18px; font-weight: 600;")
        self._progress_hint = QLabel("请稍候...")

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)

        self._progress_log = QLabel("")
        self._progress_log.setWordWrap(True)

        btn_row = QHBoxLayout()
        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.clicked.connect(self._cancel_current_job)

        self._back_source_btn = QPushButton("返回导入方式")
        self._back_source_btn.clicked.connect(lambda: self._set_state(ImportState.IMPORT_SOURCE))
        self._back_source_btn.setVisible(False)

        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._back_source_btn)

        layout.addWidget(self._progress_title)
        layout.addWidget(self._progress_hint)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._progress_log)
        layout.addLayout(btn_row)
        layout.addStretch(1)
        return page

    def _build_preview_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("视频预览")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        hint = QLabel('请确认视频内容，然后点击"下一步"为你的老婆命名。')

        self._video_widget = QVideoWidget()
        self._video_widget.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self._video_widget.setMinimumHeight(320)

        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._audio.setVolume(0.8)
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video_widget)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)

        btn_row = QHBoxLayout()
        self._play_pause_btn = QPushButton("播放")
        self._play_pause_btn.clicked.connect(self._toggle_play_pause)

        next_btn = QPushButton("下一步")
        next_btn.clicked.connect(self._on_preview_next)

        back_btn = QPushButton("返回导入方式")
        back_btn.clicked.connect(self._back_to_source_from_preview)

        btn_row.addWidget(self._play_pause_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(next_btn)
        btn_row.addWidget(back_btn)

        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(self._video_widget)
        layout.addLayout(btn_row)
        return page

    def _build_naming_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("给你的老婆起个名字")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        hint = QLabel("名字将作为角色目录名，确认后开始 AI 抠图。")

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("例如：yuna")
        self._name_edit.setMinimumHeight(36)
        self._name_edit.textChanged.connect(self._on_naming_input_changed)

        self._engine_combo = QComboBox()
        self._engine_combo.setMinimumHeight(36)
        for spec in self._matting_engines:
            self._engine_combo.addItem(spec.label, spec.engine_id)
        selected_index = self._engine_combo.findData(self._selected_matting_engine_id)
        if selected_index >= 0:
            self._engine_combo.setCurrentIndex(selected_index)
        self._engine_combo.currentIndexChanged.connect(self._on_engine_changed)

        self._engine_hint_label = QLabel("")
        self._engine_hint_label.setWordWrap(True)

        self._name_error_label = QLabel("")
        self._name_error_label.setStyleSheet("color: red;")

        self._start_segment_btn = QPushButton("开始 3D 打印你的老婆")
        self._start_segment_btn.setMinimumHeight(40)
        self._start_segment_btn.setEnabled(False)
        self._start_segment_btn.clicked.connect(self._on_start_segment)

        back_btn = QPushButton("返回预览")
        back_btn.clicked.connect(lambda: self._set_state(ImportState.PREVIEW_READY))

        btn_row = QHBoxLayout()
        btn_row.addWidget(back_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._start_segment_btn)

        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addSpacing(12)
        layout.addWidget(self._name_edit)
        if self._is_windows:
            layout.addWidget(QLabel("抠图引擎"))
            layout.addWidget(self._engine_combo)
            layout.addWidget(self._engine_hint_label)
        layout.addWidget(self._name_error_label)
        layout.addSpacing(8)
        layout.addLayout(btn_row)
        layout.addStretch(1)
        self._update_engine_hint()
        return page

    def _build_done_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self._done_title = QLabel("恭喜！你的老婆 3D 打印完成！")
        self._done_title.setStyleSheet("font-size: 20px; font-weight: 700;")
        self._done_hint = QLabel("")

        next_wife_btn = QPushButton("再加一个")
        next_wife_btn.setMinimumHeight(40)
        next_wife_btn.clicked.connect(self._on_next_wife)

        self._come_to_me_btn = QPushButton("爱妃，朕来了！！")
        self._come_to_me_btn.setMinimumHeight(40)
        self._come_to_me_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; font-weight: 600; }"
            "QPushButton:hover { background-color: #c0392b; }"
        )
        self._come_to_me_btn.clicked.connect(self._on_come_to_me)

        btn_row = QHBoxLayout()
        btn_row.addWidget(next_wife_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._come_to_me_btn)

        layout.addStretch(1)
        layout.addWidget(self._done_title)
        layout.addWidget(self._done_hint)
        layout.addSpacing(20)
        layout.addLayout(btn_row)
        layout.addStretch(2)
        return page

    # ------------------------------------------------------------------
    # 状态机
    # ------------------------------------------------------------------

    def _set_state(self, state: ImportState) -> None:
        self._state = state
        self._status_label.setText(f"当前状态：{state.value}")

        if state == ImportState.IMPORT_SOURCE:
            self._stack.setCurrentWidget(self._page_source)
        elif state == ImportState.URL_INPUT:
            self._stack.setCurrentWidget(self._page_url)
        elif state == ImportState.PREVIEW_READY:
            self._stack.setCurrentWidget(self._page_preview)
        elif state == ImportState.NAMING:
            self._stack.setCurrentWidget(self._page_naming)
        elif state == ImportState.DONE:
            self._stack.setCurrentWidget(self._page_done)
        else:
            # DOWNLOADING, TRANSCODING, SEGMENTING, FAILED
            self._stack.setCurrentWidget(self._page_progress)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _next_job_id(self) -> str:
        max_id = 0
        for d in (self._raw_dir, self._mp4_dir):
            if not d.exists():
                continue
            for p in d.iterdir():
                stem = p.stem
                if stem.isdigit():
                    max_id = max(max_id, int(stem))
        return f"{max_id + 1:06d}"

    def _validate_dancer_name(self, name: str) -> str | None:
        """返回错误信息，None 表示合法。"""
        if not name:
            return "名字不能为空"
        illegal = set(r'/\:*?"<>|')
        if any(c in illegal for c in name):
            return "名字包含非法字符（/ \\ : * ? \" < > |）"
        if self._dancer_root.exists() and (self._dancer_root / name).exists():
            return f'已存在名为"{name}"的角色'
        return None

    def _current_engine_id(self) -> str:
        if self._is_windows:
            return str(self._engine_combo.currentData() or self._selected_matting_engine_id)
        return self._selected_matting_engine_id

    def _update_engine_hint(self) -> None:
        if not self._is_windows:
            return

        spec = get_engine_spec(self._current_engine_id(), self._project_root)
        if spec is None:
            self._engine_hint_label.setText("未找到当前抠图引擎。")
            return

        message = spec.description
        if not spec.available and spec.unavailable_reason:
            message = f"{message}\n不可用：{spec.unavailable_reason}"
        self._engine_hint_label.setText(message)

    def _on_engine_changed(self, _index: int) -> None:
        self._selected_matting_engine_id = self._current_engine_id()
        self._update_engine_hint()

    # ------------------------------------------------------------------
    # 导入来源
    # ------------------------------------------------------------------

    def _on_choose_local_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择本地视频",
            "",
            "Video Files (*.mp4 *.mov *.mkv *.webm *.avi);;All Files (*)",
        )
        if not file_path:
            return

        self._job_id = self._next_job_id()
        self._pending_input_video = Path(file_path)

        # 本地已是 mp4 则跳过 ffmpeg，直接预览
        if self._pending_input_video.suffix.lower() == ".mp4":
            self._show_preview(self._pending_input_video)
            return

        self._start_transcode(self._pending_input_video, self._job_id)

    def _on_url_next(self) -> None:
        url = self._url_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "提示", "URL 不能为空")
            return

        self._job_id = self._next_job_id()
        self._start_download(url, self._job_id)

    # ------------------------------------------------------------------
    # 下载
    # ------------------------------------------------------------------

    def _start_download(self, url: str, job_id: str) -> None:
        self._set_state(ImportState.DOWNLOADING)
        self._progress_title.setText("正在下载视频")
        self._progress_hint.setText("正在调用 yt-dlp 下载，请稍候...")
        self._progress_log.setText(f"URL: {url}")
        self._progress_bar.setValue(0)
        self._cancel_btn.setVisible(True)
        self._back_source_btn.setVisible(False)

        out_tpl = str(self._raw_dir / f"{job_id}.%(ext)s")
        args = [
            "--newline",
            "--progress",
            "--no-playlist",
            "-o",
            out_tpl,
            url,
        ]

        self._download_proc = QProcess(self)
        yt_dlp_bin = find_tool_binary("yt-dlp")
        if yt_dlp_bin is None:
            self._mark_failed("未找到 yt-dlp，可执行文件请放到 tools/ 目录或加入 PATH")
            return
        self._download_proc.setProgram(yt_dlp_bin)
        self._download_proc.setArguments(args)
        self._download_proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._download_proc.readyReadStandardOutput.connect(self._on_download_output)
        self._download_proc.finished.connect(self._on_download_finished)
        self._download_proc.start()

    def _on_download_output(self) -> None:
        if not self._download_proc:
            return
        text = bytes(self._download_proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        for line in text.splitlines():
            m = re.search(r"(\d+(?:\.\d+)?)%", line)
            if m:
                self._progress_bar.setValue(min(100, int(float(m.group(1)))))
            if line.strip():
                self._progress_log.setText(line.strip())

    def _on_download_finished(self, exit_code: int, _status) -> None:
        if not self._job_id:
            self._mark_failed("下载状态异常：缺少任务 ID")
            return

        if exit_code != 0:
            self._mark_failed("下载失败，请检查链接或网络环境")
            return

        candidates = [
            p for p in self._raw_dir.glob(f"{self._job_id}.*")
            if p.suffix not in {".part", ".ytdl"} and p.is_file()
        ]
        if not candidates:
            self._mark_failed("下载完成但未找到输出文件")
            return

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        self._pending_input_video = candidates[0]
        self._start_transcode(self._pending_input_video, self._job_id)

    # ------------------------------------------------------------------
    # 转码
    # ------------------------------------------------------------------

    def _probe_duration_sec(self, video_path: Path) -> float:
        ffprobe_bin = find_tool_binary("ffprobe")
        if ffprobe_bin is None:
            raise RuntimeError("未找到 ffprobe，可执行文件请放到 tools/ 目录或加入 PATH")
        cmd = [
            ffprobe_bin,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        return float(out)

    def _start_transcode(self, input_path: Path, job_id: str) -> None:
        self._set_state(ImportState.TRANSCODING)
        self._progress_title.setText("正在转换视频")
        self._progress_hint.setText("正在调用 ffmpeg 转 mp4，请稍候...")
        self._progress_log.setText(f"输入: {input_path.name}")
        self._progress_bar.setValue(0)
        self._cancel_btn.setVisible(True)
        self._back_source_btn.setVisible(False)

        try:
            self._transcode_duration_sec = self._probe_duration_sec(input_path)
        except Exception:
            self._transcode_duration_sec = 0.0

        out_path = self._mp4_dir / f"{job_id}.mp4"
        args = [
            "-y",
            "-i", str(input_path),
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-progress", "pipe:1",
            "-nostats",
            str(out_path),
        ]

        self._transcode_proc = QProcess(self)
        ffmpeg_bin = find_tool_binary("ffmpeg")
        if ffmpeg_bin is None:
            self._mark_failed("未找到 ffmpeg，可执行文件请放到 tools/ 目录或加入 PATH")
            return
        self._transcode_proc.setProgram(ffmpeg_bin)
        self._transcode_proc.setArguments(args)
        self._transcode_proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._transcode_proc.readyReadStandardOutput.connect(self._on_transcode_output)
        self._transcode_proc.finished.connect(
            lambda code, status: self._on_transcode_finished(code, status, out_path)
        )
        self._transcode_proc.start()

    def _on_transcode_output(self) -> None:
        if not self._transcode_proc:
            return

        text = bytes(self._transcode_proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith("out_time_ms=") and self._transcode_duration_sec > 0:
                try:
                    ms = int(line.split("=", 1)[1])
                    sec = ms / 1_000_000.0
                    pct = int(min(100, sec / self._transcode_duration_sec * 100))
                    self._progress_bar.setValue(pct)
                except Exception:
                    pass

            if line.startswith("progress="):
                self._progress_log.setText(f"ffmpeg: {line}")
            elif "time=" in line:
                self._progress_log.setText(line)

    def _on_transcode_finished(self, exit_code: int, _status, out_path: Path) -> None:
        if exit_code != 0:
            self._mark_failed("转码失败，请确认 ffmpeg 可用")
            return

        if not out_path.exists():
            self._mark_failed("转码完成但未找到 mp4 输出")
            return

        self._show_preview(out_path)

    # ------------------------------------------------------------------
    # 预览
    # ------------------------------------------------------------------

    def _show_preview(self, video_path: Path) -> None:
        self._preview_video = video_path
        self._set_state(ImportState.PREVIEW_READY)
        self._player.setSource(QUrl.fromLocalFile(str(video_path)))
        self._player.pause()
        self._play_pause_btn.setText("播放")

    def _toggle_play_pause(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_playback_state_changed(self, state) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._play_pause_btn.setText("暂停")
        else:
            self._play_pause_btn.setText("播放")

    def _on_preview_next(self) -> None:
        self._player.pause()
        # 清空上次命名状态
        self._name_edit.clear()
        self._name_error_label.setText("")
        self._start_segment_btn.setEnabled(False)
        self._set_state(ImportState.NAMING)

    def _back_to_source_from_preview(self) -> None:
        self._player.stop()
        self._set_state(ImportState.IMPORT_SOURCE)

    # ------------------------------------------------------------------
    # 命名页
    # ------------------------------------------------------------------

    def _on_naming_input_changed(self, text: str) -> None:
        name = text.strip()
        err = self._validate_dancer_name(name)
        if err:
            self._name_error_label.setText(err)
            self._start_segment_btn.setEnabled(False)
        else:
            self._name_error_label.setText("")
            self._start_segment_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # 分割
    # ------------------------------------------------------------------

    def _on_start_segment(self) -> None:
        name = self._name_edit.text().strip()
        err = self._validate_dancer_name(name)
        if err:
            QMessageBox.warning(self, "名字无效", err)
            return

        self._dancer_name = name
        dancer_dir = self._dancer_root / name

        video_path = self._preview_video
        if not video_path or not video_path.exists():
            QMessageBox.critical(self, "错误", "找不到视频文件，请重新导入")
            return

        self._set_state(ImportState.SEGMENTING)
        self._progress_title.setText("正在 3D 打印你的老婆")
        self._progress_hint.setText("AI 抠图中，请稍候（可能需要数分钟）…")
        self._progress_log.setText("")
        self._progress_bar.setValue(0)
        self._cancel_btn.setVisible(True)
        self._back_source_btn.setVisible(False)

        engine_id = self._current_engine_id()
        request = SegmentJobRequest(
            project_root=self._project_root,
            video_path=video_path,
            dancer_dir=dancer_dir,
            display_height=450,
        )
        try:
            self._segment_worker = create_segment_worker(engine_id, request)
        except Exception as exc:
            self._mark_failed(f"抠图引擎启动失败：{exc}")
            return

        self._segment_worker.progress.connect(self._on_segment_progress)
        self._segment_worker.stage_changed.connect(self._on_segment_stage)
        self._segment_worker.finished_ok.connect(self._on_segment_done)
        self._segment_worker.finished_err.connect(self._on_segment_failed)
        self._segment_worker.start()

    def _on_segment_progress(self, current: int, total: int) -> None:
        pct = int(current / total * 100) if total > 0 else 0
        self._progress_bar.setValue(pct)
        self._progress_log.setText(f"处理帧 {current} / {total}")

    def _on_segment_stage(self, msg: str) -> None:
        self._progress_hint.setText(msg)

    def _on_segment_done(self) -> None:
        self._done_title.setText("恭喜！老婆 3D 打印完成！")
        self._done_hint.setText(f'角色"{self._dancer_name}"已就绪，随时可以出场。')
        self._come_to_me_btn.setEnabled(True)
        self._set_state(ImportState.DONE)

    def _on_segment_failed(self, msg: str) -> None:
        if msg == "已取消":
            self._mark_failed("操作已取消，你可以返回重新选择。")
        else:
            self._mark_failed(f"3D 打印失败：{msg}")

    # ------------------------------------------------------------------
    # 完成页
    # ------------------------------------------------------------------

    def _on_next_wife(self) -> None:
        """重置向导，准备导入下一个角色。"""
        self._preview_video = None
        self._pending_input_video = None
        self._job_id = None
        self._dancer_name = ""
        self._segment_worker = None
        self._set_state(ImportState.IMPORT_SOURCE)

    def _on_come_to_me(self) -> None:
        """切换到刚完成的角色。"""
        if self._on_dancer_ready:
            self._on_dancer_ready(self._dancer_name)
        self.close()

    # ------------------------------------------------------------------
    # 取消 / 失败
    # ------------------------------------------------------------------

    def _cancel_current_job(self) -> None:
        if self._download_proc and self._download_proc.state() != QProcess.ProcessState.NotRunning:
            self._download_proc.kill()
        if self._transcode_proc and self._transcode_proc.state() != QProcess.ProcessState.NotRunning:
            self._transcode_proc.kill()
        if self._segment_worker and self._segment_worker.isRunning():
            self._segment_worker.cancel()
            # 不等待线程结束，让它自己退出；失败信号会触发 _on_segment_failed
            return

        self._progress_title.setText("任务已取消")
        self._progress_hint.setText("你可以返回导入方式，重新选择。")
        self._progress_log.setText("")
        self._progress_bar.setValue(0)
        self._cancel_btn.setVisible(False)
        self._back_source_btn.setVisible(True)
        self._set_state(ImportState.FAILED)

    def _mark_failed(self, msg: str) -> None:
        self._progress_title.setText("处理失败")
        self._progress_hint.setText(msg)
        self._progress_log.setText("")
        self._cancel_btn.setVisible(False)
        self._back_source_btn.setVisible(True)
        self._set_state(ImportState.FAILED)

    def closeEvent(self, event):
        self._cancel_current_job()
        self._player.stop()
        super().closeEvent(event)
