from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, QThreadPool, pyqtSignal

from ...adapters.qt.directory_scan import ScanInputDirectoryRunnable, ScanInputDirectorySignals


class DirectoryScanController(QObject):
    started = pyqtSignal(str)
    idle = pyqtSignal()
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._busy = False
        self._pending_directory: str | None = None
        self._generation = 0
        self._signals = ScanInputDirectorySignals(self)
        self._signals.finished.connect(self._on_finished)
        self._signals.failed.connect(self._on_failed)
        self._thread_pool = QThreadPool(self)
        self._thread_pool.setMaxThreadCount(1)
        self._thread_pool.setExpiryTimeout(-1)

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self, directory: str | Path) -> None:
        normalized = str(Path(directory))
        if self._busy:
            self._pending_directory = normalized
            return
        self._run_now(normalized)

    def invalidate_pending_results(self) -> None:
        self._generation += 1

    def _run_now(self, directory: str) -> None:
        self._busy = True
        self.started.emit(directory)
        self._thread_pool.start(
            ScanInputDirectoryRunnable(
                directory=directory,
                signals=self._signals,
                run_generation=self._generation,
            )
        )

    def _complete_turn(self) -> str | None:
        self._busy = False
        self.idle.emit()
        pending = self._pending_directory
        self._pending_directory = None
        return pending

    def _start_pending_if_needed(self, pending_directory: str | None) -> None:
        if pending_directory:
            self._run_now(pending_directory)

    def _on_finished(self, paths: list[str], run_generation: int) -> None:
        pending = self._complete_turn()
        if run_generation == self._generation:
            self.finished.emit(paths)
        self._start_pending_if_needed(pending)

    def _on_failed(self, message: str, run_generation: int) -> None:
        pending = self._complete_turn()
        if run_generation == self._generation:
            self.failed.emit(message)
        self._start_pending_if_needed(pending)


__all__ = ["DirectoryScanController"]
