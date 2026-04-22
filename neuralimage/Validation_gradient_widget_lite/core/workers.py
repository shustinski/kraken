"""Background Qt workers for the extended validation gradient widget."""
from __future__ import annotations

from time import monotonic
from threading import Event

from PyQt6.QtCore import QObject, pyqtSignal

from .domain import BuildOptions, BuildResult, FolderSpec, FrameRecord, ModelSpec
from .repository import collect_frame_records, compute_build_result_analytics, load_frame_detail_base, load_frame_detail_model_confidence


class WorkerBase(QObject):
    """Provide cancellation and signal plumbing for background workers."""

    PROGRESS_MIN_INTERVAL_SECONDS = 0.10
    FRAME_STATE_MIN_INTERVAL_SECONDS = 0.05

    progress = pyqtSignal(int, int, str)
    frameStateChanged = pyqtSignal(str, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._cancel_requested = Event()
        self._last_progress_emit_at = 0.0
        self._last_frame_state_emit_at = 0.0
        self._emitted_running_keys: set[str] = set()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def _is_cancelled(self) -> bool:
        return self._cancel_requested.is_set()

    def _emit_progress(self, current: int, total: int, key: str, *, force: bool = False) -> None:
        current_i = int(current)
        total_i = int(total)
        now = monotonic()
        if (
            force
            or current_i <= 0
            or (total_i > 0 and current_i >= total_i)
            or now - self._last_progress_emit_at >= self.PROGRESS_MIN_INTERVAL_SECONDS
        ):
            self._last_progress_emit_at = now
            self.progress.emit(current_i, total_i, str(key or ""))

    def _emit_frame_state(self, key: str, status: str) -> None:
        normalized_key = str(key or "")
        if not normalized_key:
            return
        normalized_status = str(status or "running")
        now = monotonic()
        if normalized_status == "running":
            if now - self._last_frame_state_emit_at < self.FRAME_STATE_MIN_INTERVAL_SECONDS:
                return
            self._last_frame_state_emit_at = now
            self._emitted_running_keys.add(normalized_key)
            self.frameStateChanged.emit(normalized_key, normalized_status)
            return
        if normalized_key in self._emitted_running_keys:
            self._emitted_running_keys.discard(normalized_key)
            self.frameStateChanged.emit(normalized_key, normalized_status)


class FrameIndexWorker(WorkerBase):
    """Index all shared frames across the selected model folders."""

    def __init__(self, model_specs: tuple[ModelSpec, ...], options: BuildOptions, original_folder: FolderSpec | None, gt_folder: FolderSpec | None) -> None:
        super().__init__()
        self._model_specs = model_specs
        self._options = options
        self._original_folder = original_folder
        self._gt_folder = gt_folder

    def run(self) -> None:
        try:
            self._emit_progress(0, 0, "", force=True)
            result = collect_frame_records(
                self._model_specs,
                self._options,
                original_folder=self._original_folder,
                gt_folder=self._gt_folder,
                cancel_check=self._is_cancelled,
                progress_callback=self._emit_progress,
            )
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished.emit(result)


class AnalyticsWorker(WorkerBase):
    """Compute frame-level analytics for an indexed build result."""

    def __init__(self, build_result: BuildResult, metric_key: str) -> None:
        super().__init__()
        self._build_result = build_result
        self._metric_key = metric_key

    def run(self) -> None:
        try:
            self._emit_progress(0, 0, "", force=True)
            result = compute_build_result_analytics(
                self._build_result,
                metric_key=self._metric_key,
                progress_callback=self._emit_progress,
                state_callback=self._emit_frame_state,
                cancel_check=self._is_cancelled,
            )
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished.emit(result)


class DetailPayloadWorker(WorkerBase):
    """Load the base detail payload without blocking the UI thread."""

    def __init__(self, record: FrameRecord, build_result: BuildResult, model_id: str | None, max_side: int | None) -> None:
        super().__init__()
        self._record = record
        self._build_result = build_result
        self._model_id = model_id
        self._max_side = max_side

    def run(self) -> None:
        try:
            self._emit_progress(0, 0, "", force=True)
            payload = load_frame_detail_base(
                self._record,
                self._build_result,
                model_id=self._model_id,
                max_side=self._max_side,
            )
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished.emit(payload)


class DetailConfidenceWorker(WorkerBase):
    """Compute heavy confidence/debug payload for one selected model in background."""

    def __init__(self, record: FrameRecord, build_result: BuildResult, model_id: str | None, max_side: int | None, detail_payload: dict[str, object]) -> None:
        super().__init__()
        self._record = record
        self._build_result = build_result
        self._model_id = model_id
        self._max_side = max_side
        self._detail_payload = dict(detail_payload)
        self._detail_payload["model_confidence"] = dict((detail_payload.get("model_confidence") or {}))

    def run(self) -> None:
        try:
            self._emit_progress(0, 0, "", force=True)
            payload = load_frame_detail_model_confidence(
                self._record,
                self._build_result,
                model_id=self._model_id,
                max_side=self._max_side,
                detail_payload=self._detail_payload,
            )
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished.emit(payload)


# Preferred alias for the analytics worker used by the widget.
MetricsWorker = AnalyticsWorker

# Backward-compatible alias for legacy lite imports.
MismatchWorker = AnalyticsWorker
