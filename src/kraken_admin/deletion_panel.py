"""Review deletion requests and run disk cleanup outside the Qt thread."""

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kraken_server.configuration import ServerConfig
from kraken_server.project_deletion import DeletionRequests

from .deletion_confirmation import DeletionPolicy, ensure_deletion_schema, short_database_error
from .project_deletion import ProjectDeletion


class _DeletionWorker(QThread):
    result = pyqtSignal(str)

    def __init__(self, operation, request_id, name, plan, auto_confirm, reason, parent) -> None:
        super().__init__(parent)
        self.operation = operation
        self.request_id = request_id
        self.name = name
        self.plan = plan
        self.auto_confirm = auto_confirm
        self.reason = reason

    def run(self) -> None:
        try:
            self.operation.execute(
                self.request_id,
                self.name,
                self.plan,
                auto_confirm=self.auto_confirm,
                reason=self.reason,
            )
        except Exception as exc:  # noqa: BLE001 - worker reports every cleanup failure
            self.result.emit(str(exc))
        else:
            self.result.emit("")


class _DeletionConfirmationDialog(QDialog):
    def __init__(self, parent, *, project_name: str, paths: list[str], reason: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Полное удаление проекта")
        layout = QVBoxLayout(self)
        summary = QLabel(
            "Будут удалены метаданные, история, задания, кэш и принадлежащие проекту файлы:\n"
            + "\n".join(paths)
            + f"\n\nДля подтверждения введите имя проекта: {project_name}"
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        form = QFormLayout()
        self.reason = QLineEdit(reason)
        self.reason.setObjectName("adminDeletionReason")
        self.reason.setPlaceholderText("Необязательно")
        self.name = QLineEdit()
        self.name.setObjectName("adminDeletionProjectName")
        form.addRow("Причина удаления", self.reason)
        form.addRow("Имя проекта", self.name)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Удалить")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, str]:
        return self.name.text(), self.reason.text()


class DeletionPanel(QWidget):
    def __init__(self, admin) -> None:
        super().__init__(admin.window)
        self.admin = admin
        self.rows = []
        self.worker = None
        self._auto_failed: set[str] = set()
        self._auto_request_id = ""
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 6)
        self.table.setObjectName("adminDeletionRequests")
        self.table.setHorizontalHeaderLabels(
            ["Проект", "Инициатор", "Причина", "Время", "Состояние", "Ошибка"]
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self.warning = QLabel(
            "Небезопасно: автоподтверждение выполняет любую заявку, которая удаляет файлы, "
            "без ввода имени проекта. Новые заявки запускаются при открытии и обновлении списка."
        )
        self.warning.setObjectName("adminDeletionAutoConfirmWarning")
        self.warning.setWordWrap(True)
        layout.addWidget(self.warning)
        self.auto_confirm = QCheckBox("Автоподтверждение всех удалений")
        self.auto_confirm.setObjectName("adminDeletionAutoConfirm")
        self.auto_confirm.toggled.connect(self._set_auto_confirm)
        layout.addWidget(self.auto_confirm)
        actions = QHBoxLayout()
        self.approve = QPushButton("Подтвердить / продолжить удаление…")
        self.reject = QPushButton("Отклонить")
        self.approve.clicked.connect(lambda: self.execute())
        self.reject.clicked.connect(self.decline)
        actions.addWidget(self.approve)
        actions.addWidget(self.reject)
        layout.addLayout(actions)

    def reload(self) -> None:
        if self.admin._services is None or not hasattr(self.admin._services, "engine"):
            return
        try:
            ensure_deletion_schema(self.admin._services.engine)
            self.rows = DeletionRequests(self.admin._services.engine).list()
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self.rows = []
            self.table.setRowCount(0)
            self._set_checked(False)
            self.admin._report(
                "Заявки на удаление недоступны",
                "Не удалось подготовить таблицу заявок на удаление.\n\n" + short_database_error(exc),
            )
            return
        self._show_rows()
        self._set_checked(DeletionPolicy(self.admin._services.engine).enabled())
        self._start_auto()

    def _show_rows(self) -> None:
        labels = {"pending": "Ожидает подтверждения", "deleting": "Удаляется", "failed": "Ошибка — можно продолжить",
                  "rejected": "Отклонена", "deleted": "Удалён"}
        self.table.setRowCount(len(self.rows))
        for index, row in enumerate(self.rows):
            for column, value in enumerate((
                row["project_name"], row["requested_by"], row.get("reason", ""), row["requested_at"],
                labels.get(row["state"], row["state"]), row["error"],
            )):
                self.table.setItem(index, column, QTableWidgetItem(str(value)))

    def operation(self):
        return ProjectDeletion(self.admin._services, self.admin._accounts,
                               ServerConfig.load(self.admin._config_path))

    def delete_project(self, project_id: str, project_name: str) -> None:
        if self.admin._accounts is None or self.admin._services is None or self.admin._deletion_busy:
            return
        try:
            from .project_roles import administrator

            ensure_deletion_schema(self.admin._services.engine)
            actor = administrator(self.admin._accounts)
            created = DeletionRequests(self.admin._services.engine).request(
                project_id, project_name, str(actor.id),
            )
            self.rows = DeletionRequests(self.admin._services.engine).list()
            self._show_rows()
            index = next(
                (position for position, row in enumerate(self.rows) if row["request_id"] == created["request_id"]),
                -1,
            )
            if index < 0:
                raise ValueError("Заявка на удаление не создана")
            self.table.selectRow(index)
            self.execute()
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self.admin._report("Не удалось начать удаление проекта", exc)

    def decline(self) -> None:
        index = self.table.currentRow()
        if index < 0:
            return
        try:
            self.operation().reject(self.rows[index]["request_id"])
            self.admin.reload()
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self.admin._report("Не удалось отклонить заявку", exc)

    def execute(self, *, auto: bool = False) -> None:
        index = self.table.currentRow()
        if index < 0 or self.admin._deletion_busy:
            return
        auto = auto or self.auto_confirm.isChecked()
        try:
            operation = self.operation()
            request_id = self.rows[index]["request_id"]
            plan = operation.preview(request_id)
            reason = None
            if auto:
                name = plan["name"]
            else:
                dialog = _DeletionConfirmationDialog(
                    self.admin.window,
                    project_name=plan["name"],
                    paths=[entry["path"] for entry in plan["paths"]],
                    reason=str(self.rows[index].get("reason") or ""),
                )
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                name, reason = dialog.values()
                if name != plan["name"]:
                    raise ValueError("Имя проекта не совпадает")
            self.admin._deletion_busy = True
            self.approve.setEnabled(False)
            self.reject.setEnabled(False)
            self._auto_request_id = request_id if auto else ""
            self.worker = _DeletionWorker(operation, request_id, name, plan, auto, reason, self)
            self.worker.result.connect(self._result)
            self.worker.finished.connect(self._finished)
            self.worker.start()
        except Exception as exc:  # noqa: BLE001 - UI boundary
            if auto:
                self._auto_failed.add(self.rows[index]["request_id"])
            self.admin._deletion_busy = False
            self.approve.setEnabled(True)
            self.reject.setEnabled(True)
            self._auto_request_id = ""
            self.admin._report("Не удалось начать удаление", exc)

    def _result(self, error: str) -> None:
        if error:
            if self._auto_request_id:
                self._auto_failed.add(self._auto_request_id)
            self.admin._report("Удаление не завершено", error)

    def _finished(self) -> None:
        self.admin._deletion_busy = False
        self.approve.setEnabled(True)
        self.reject.setEnabled(True)
        self._auto_request_id = ""
        self.worker.deleteLater()
        self.worker = None
        self.admin.reload()

    def _set_checked(self, enabled: bool) -> None:
        self.auto_confirm.blockSignals(True)
        self.auto_confirm.setChecked(enabled)
        self.auto_confirm.blockSignals(False)

    def _set_auto_confirm(self, enabled: bool) -> None:
        if self.admin._services is None or not hasattr(self.admin._services, "engine"):
            self._set_checked(False)
            self.admin._report("Kraken Admin", "Сначала откройте server.toml.")
            return
        if enabled:
            answer = QMessageBox.warning(
                self.admin.window,
                "Небезопасное автоподтверждение",
                "Все операции, которые удаляют файлы, будут подтверждаться автоматически, "
                "без ввода имени проекта.\n\n"
                "Это небезопасно: ошибочная заявка удалит данные сразу.\n\n"
                "Включить автоподтверждение?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self._set_checked(False)
                return
        try:
            DeletionPolicy(self.admin._services.engine).set_enabled(enabled, self.admin._accounts)
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._set_checked(not enabled)
            self.admin._report("Не удалось изменить автоподтверждение", exc)
            return
        if enabled:
            self._auto_failed.clear()
            self._start_auto()

    def _start_auto(self) -> None:
        if not self.auto_confirm.isChecked() or self.admin._deletion_busy:
            return
        for index, row in enumerate(self.rows):
            if row["state"] == "pending" and row["request_id"] not in self._auto_failed:
                self.table.selectRow(index)
                self.execute(auto=True)
                return
