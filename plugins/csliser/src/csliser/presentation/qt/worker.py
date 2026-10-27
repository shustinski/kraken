from __future__ import annotations

from pathlib import Path
from threading import Event

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from csliser.application.use_cases import ExecuteTransferPlan
from csliser.domain.models import OperationError, OperationPlan, OperationResult


class TransferWorker(QObject):
    progress_changed = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)

    def __init__(self, plan: OperationPlan, executor: ExecuteTransferPlan | None = None) -> None:
        super().__init__()
        self._plan = plan
        self._executor = executor or ExecuteTransferPlan()
        self._cancelled = Event()

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = self._executor.execute(
                self._plan,
                progress=lambda current, total, path: self.progress_changed.emit(current, total, path),
                cancelled=self._cancelled.is_set,
            )
            if not isinstance(result, OperationResult):
                raise TypeError("Transfer executor returned no valid result")
        except Exception as exc:
            result = OperationResult(requested=len(self._plan.operations), completed=0, skipped=0, outcome_unknown=True,
                errors=(OperationError(Path(), None, f"Completion is unknown; inspect source and destination files: {exc}"),))
        self.finished.emit(result)

    def cancel(self) -> None:
        self._cancelled.set()
