from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from ...utils import scan_image_files

_LOGGER = logging.getLogger(__name__)


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
        except Exception as exc:
            _LOGGER.exception("Input directory scan failed for %s", self._directory)
            self._signals.failed.emit(str(exc), self._run_generation)
