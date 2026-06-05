from __future__ import annotations

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from ...utils import scan_image_files


class ScanInputDirectorySignals(QObject):
    finished = pyqtSignal(list, int)
    failed = pyqtSignal(str, int)


class ScanInputDirectoryRunnable(QRunnable):
    def __init__(self, *, directory: str, signals: ScanInputDirectorySignals, run_generation: int) -> None:
        super().__init__()
        self._directory = directory
        self._signals = signals
        self._run_generation = run_generation

    def run(self) -> None:
        try:
            self._signals.finished.emit(scan_image_files(self._directory), self._run_generation)
        except Exception as exc:  # noqa: BLE001 — surface to UI
            self._signals.failed.emit(str(exc), self._run_generation)
