"""Background Qt workers for the extended validation gradient widget."""
from __future__ import annotations

from threading import Event

from PyQt6.QtCore import QObject, pyqtSignal

from .domain import BuildOptions, BuildResult, FolderSpec, FrameRecord, ModelSpec
from .repository import collect_frame_records, compute_build_result_analytics, load_frame_detail_base, load_frame_detail_model_confidence


class WorkerBase(QObject):
    """Provide cancellation and signal plumbing for background workers."""

    progress = pyqtSignal(int, int, str)
    frameStateChanged = pyqtSignal(str, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._cancel_requested = Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def _is_cancelled(self) -> bool:
        return self._cancel_requested.is_set()


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
            self.progress.emit(0, 0, "")
            result = collect_frame_records(
                self._model_specs,
                self._options,
                original_folder=self._original_folder,
                gt_folder=self._gt_folder,
                cancel_check=self._is_cancelled,
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
            self.progress.emit(0, 0, "")
            result = compute_build_result_analytics(
                self._build_result,
                metric_key=self._metric_key,
                progress_callback=lambda current, total, key: self.progress.emit(current, total, key),
                state_callback=lambda key, status: self.frameStateChanged.emit(str(key), str(status)),
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
            self.progress.emit(0, 0, "")
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
            self.progress.emit(0, 0, "")
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
