from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from csliser.application.use_cases import ExecuteTransferPlan
from csliser.domain.models import OperationPlan


class TransferWorker(QObject):
    progress_changed = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)

    def __init__(self, plan: OperationPlan, executor: ExecuteTransferPlan | None = None) -> None:
        super().__init__()
        self._plan = plan
        self._executor = executor or ExecuteTransferPlan()
        self._cancelled = False

    @pyqtSlot()
    def run(self) -> None:
        result = self._executor.execute(
            self._plan,
            progress=lambda current, total, path: self.progress_changed.emit(current, total, path),
            cancelled=lambda: self._cancelled,
        )
        self.finished.emit(result)

    def cancel(self) -> None:
        self._cancelled = True
