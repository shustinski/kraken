"""Run long-running indexing and mismatch computation tasks in background Qt workers for the lite widget."""
from __future__ import annotations

from threading import Event

from PyQt6.QtCore import QObject, pyqtSignal

from .domain import BuildOptions, BuildResult, ComparisonMode, FolderSpec
from .repository import collect_frame_records, compute_build_result_mismatches


class WorkerBase(QObject):
    """Provide cancellation and signal plumbing for background Qt workers."""

    progress = pyqtSignal(int, int, str)
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
    """Build the frame matrix in a worker thread without computing mismatches."""

    def __init__(self, first_folder: FolderSpec, second_folder: FolderSpec, options: BuildOptions, base_folder: FolderSpec | None) -> None:
        super().__init__()
        self._first_folder = first_folder
        self._second_folder = second_folder
        self._options = options
        self._base_folder = base_folder

    def run(self) -> None:
        try:
            self.progress.emit(0, 0, '')
            result = collect_frame_records(
                self._first_folder,
                self._second_folder,
                self._options,
                base_folder=self._base_folder,
                cancel_check=self._is_cancelled,
            )
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished.emit(result)


class MismatchWorker(WorkerBase):
    """Compute per-frame mismatches for an existing matrix in a worker thread."""

    activeKeysChanged = pyqtSignal(object)

    def __init__(self, build_result: BuildResult, comparison_mode: ComparisonMode) -> None:
        super().__init__()
        self._build_result = build_result
        self._comparison_mode = comparison_mode

    def run(self) -> None:
        try:
            self.progress.emit(0, 0, '')
            result = compute_build_result_mismatches(
                self._build_result,
                comparison_mode=self._comparison_mode,
                display_metric='relative',
                progress_callback=lambda current, total, key: self.progress.emit(current, total, key),
                active_keys_callback=lambda keys: self.activeKeysChanged.emit(tuple(keys)),
                cancel_check=self._is_cancelled,
            )
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished.emit(result)
