from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from ...application.use_cases import index_cif_directory

_LOGGER = logging.getLogger(__name__)


class VectorIndexSignals(QObject):
    finished = pyqtSignal(object, int)
    failed = pyqtSignal(str, int)


class VectorIndexRunnable(QRunnable):
    def __init__(self, *, directory: str, signals: VectorIndexSignals, run_generation: int) -> None:
        super().__init__()
        self._directory = directory
        self._signals = signals
        self._run_generation = run_generation

    def run(self) -> None:
        try:
            self._signals.finished.emit(index_cif_directory(self._directory), self._run_generation)
        except Exception as exc:
            _LOGGER.exception("CIF index creation failed for %s", self._directory)
            self._signals.failed.emit(str(exc), self._run_generation)
