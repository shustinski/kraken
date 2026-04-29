from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from kategb.domain.models import CopyPlan
from kategb.infrastructure.file_copy import FrameFileCopier


class CopyWorker(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, plan: CopyPlan, copier: FrameFileCopier | None = None) -> None:
        super().__init__()
        self._plan = plan
        self._copier = copier or FrameFileCopier()

    @pyqtSlot()
    def run(self) -> None:
        try:
            report = self._copier.copy(self._plan, self._emit_progress)
        except Exception as exc:  # pragma: no cover - Qt boundary
            self.failed.emit(str(exc))
            return
        self.finished.emit(report)

    def _emit_progress(self, done: int, total: int, path: Path) -> None:
        self.progress.emit(done, total, str(path))
