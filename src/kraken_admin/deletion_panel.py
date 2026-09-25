"""Review deletion requests and run disk cleanup outside the Qt thread."""

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QInputDialog, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from kraken_server.configuration import ServerConfig
from kraken_server.project_deletion import DeletionRequests

from .project_deletion import ProjectDeletion


class _DeletionWorker(QThread):
    result = pyqtSignal(str)

    def __init__(self, operation, request_id, name, plan, parent) -> None:
        super().__init__(parent)
        self.operation, self.request_id, self.name, self.plan = operation, request_id, name, plan

    def run(self) -> None:
        try:
            self.operation.execute(self.request_id, self.name, self.plan)
        except Exception as exc:
            self.result.emit(str(exc))
        else:
            self.result.emit("")


class DeletionPanel(QWidget):
    def __init__(self, admin) -> None:
        super().__init__(admin.window)
        self.admin = admin
        self.rows = []
        self.worker = None
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 5)
        self.table.setObjectName("adminDeletionRequests")
        self.table.setHorizontalHeaderLabels(["Проект", "Инициатор", "Время", "Состояние", "Ошибка"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)
        actions = QHBoxLayout()
        self.approve = QPushButton("Подтвердить / продолжить удаление…")
        self.reject = QPushButton("Отклонить")
        self.approve.clicked.connect(self.execute)
        self.reject.clicked.connect(self.decline)
        actions.addWidget(self.approve)
        actions.addWidget(self.reject)
        layout.addLayout(actions)

    def reload(self) -> None:
        if self.admin._services is None or not hasattr(self.admin._services, "engine"):
            return
        self.rows = DeletionRequests(self.admin._services.engine).list()
        self.table.setRowCount(len(self.rows))
        labels = {"pending": "Ожидает подтверждения", "deleting": "Удаляется", "failed": "Ошибка — можно продолжить",
                  "rejected": "Отклонена", "deleted": "Удалён"}
        for index, row in enumerate(self.rows):
            for column, value in enumerate((row["project_name"], row["requested_by"], row["requested_at"],
                                            labels.get(row["state"], row["state"]), row["error"])):
                self.table.setItem(index, column, QTableWidgetItem(str(value)))

    def operation(self):
        return ProjectDeletion(self.admin._services, self.admin._accounts,
                               ServerConfig.load(self.admin._config_path))

    def decline(self) -> None:
        index = self.table.currentRow()
        if index < 0:
            return
        try:
            self.operation().reject(self.rows[index]["request_id"])
            self.admin.reload()
        except Exception as exc:
            self.admin._report("Не удалось отклонить заявку", exc)

    def execute(self) -> None:
        index = self.table.currentRow()
        if index < 0 or self.admin._deletion_busy:
            return
        try:
            operation = self.operation()
            request_id = self.rows[index]["request_id"]
            plan = operation.preview(request_id)
            name, accepted = QInputDialog.getText(self, "Полное удаление проекта",
                "Будут удалены метаданные, история, задания, кэш и принадлежащие проекту файлы:\n"
                + "\n".join(entry["path"] for entry in plan["paths"])
                + f"\n\nВведите имя проекта: {plan['name']}")
            if not accepted:
                return
            if name != plan["name"]:
                raise ValueError("Имя проекта не совпадает")
            self.admin._deletion_busy = True
            self.approve.setEnabled(False)
            self.reject.setEnabled(False)
            self.worker = _DeletionWorker(operation, request_id, name, plan, self)
            self.worker.result.connect(self._result)
            self.worker.finished.connect(self._finished)
            self.worker.start()
        except Exception as exc:
            self.admin._report("Не удалось начать удаление", exc)

    def _result(self, error: str) -> None:
        if error:
            self.admin._report("Удаление не завершено", error)

    def _finished(self) -> None:
        self.admin._deletion_busy = False
        self.approve.setEnabled(True)
        self.reject.setEnabled(True)
        self.worker.deleteLater()
        self.worker = None
        self.admin.reload()
