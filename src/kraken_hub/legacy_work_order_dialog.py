"""Review and capture legacy work without blocking Qt or destroying live tasks."""
from pathlib import Path
from uuid import uuid4

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QFileDialog, QLabel, QPlainTextEdit, QVBoxLayout

from kraken_hub.manager_app_parts.layer_create_thread import _BackgroundCallThread
from kraken_manager.domain.work_orders import WorkKind

_RUNNING = set()


def _wait_for_migrations():
    for worker in tuple(_RUNNING):
        worker.wait()


def _release(worker):
    _RUNNING.discard(worker)


class LegacyWorkOrderMigrationDialog(QDialog):
    migrated = pyqtSignal(object)

    def __init__(self, service, principal, project_id, parent=None):
        super().__init__(parent)
        self.service, self.principal, self.project_id = service, principal, str(project_id)
        self._worker, self._plan = None, None
        self._phase, self._key = "", str(uuid4())
        self.setWindowTitle("Миграция старых поручений")
        self.resize(850, 620)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        layout = QVBoxLayout(self)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setObjectName("legacyWorkOrderPreview")
        layout.addWidget(self.preview)
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Reset
            | QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)
        self.apply_button = self.buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.setText("Сохранить поручения в журнале")
        self.apply_button.setAutoDefault(False)
        self.refresh_button = self.buttons.button(QDialogButtonBox.StandardButton.Reset)
        self.refresh_button.setText("Обновить план")
        self.save_button = self.buttons.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setText("Сохранить полный план…")
        self.buttons.button(QDialogButtonBox.StandardButton.Close).setText("Закрыть")
        layout.addWidget(self.buttons)
        self.apply_button.clicked.connect(self.apply_migration)
        self.refresh_button.clicked.connect(self.refresh)
        self.save_button.clicked.connect(self.save_plan)
        self.buttons.rejected.connect(self.reject)
        app = QApplication.instance()
        if not app.property("krakenLegacyMigrationWaitConnected"):
            app.aboutToQuit.connect(_wait_for_migrations)
            app.setProperty("krakenLegacyMigrationWaitConnected", True)
        self.refresh()

    def _start(self, phase, call):
        if self._worker is not None:
            return
        self._phase = phase
        # A parent can be destroyed without closeEvent. Keep the thread outside
        # its QObject tree; bound Qt slots disconnect when the dialog dies.
        worker = _BackgroundCallThread(call)
        self._worker = worker
        _RUNNING.add(worker)
        worker.succeeded.connect(self._received)
        worker.failed.connect(self._failed)
        worker.finished.connect(self._finished)
        worker.finished.connect(lambda retained=worker: _release(retained))
        worker.finished.connect(worker.deleteLater)
        self.buttons.setEnabled(False)
        self.status.setText("Операция выполняется…")
        worker.start()

    def refresh(self):
        service, principal, project_id = self.service, self.principal, self.project_id
        self._start("plan", lambda: service.plan_legacy_work_order_migration(principal=principal, project_id=project_id))

    def apply_migration(self):
        if self._worker is not None or not self._plan or not self._plan["legacy"]:
            return
        service, principal, project_id = self.service, self.principal, self.project_id
        digest, key = self._plan["digest"], self._key
        self._start("apply", lambda: service.migrate_legacy_work_orders(principal=principal, project_id=project_id,
            expected_plan_digest=digest, idempotency_key=key))

    def save_plan(self):
        if self._worker is not None or self._plan is None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить полный план", "legacy-work-orders-plan.json", "JSON (*.json)")
        if not filename:
            return
        plan = self._plan
        from kraken_manager.infrastructure.filesystem._atomic import atomic_write_json

        def write():
            atomic_write_json(Path(filename), plan, overwrite=False)
            return filename
        self._start("save", write)

    @pyqtSlot(object)
    def _received(self, value):
        try:
            if self._phase == "plan":
                if not isinstance(value, dict) or not isinstance(value.get("legacy"), (list, tuple)):
                    raise ValueError("Сервис вернул неверный план миграции")
                if not self._plan or value["digest"] != self._plan["digest"]:
                    self._key = str(uuid4())
                self._plan = value
                rows = value["legacy"]
                lines = [f"Проект: {self.project_id}", f"Поручений без событий: {len(rows)}",
                    "Сохраняется известное состояние и наблюдения из старого индекса.",
                    "История операций до переноса остаётся неизвестной.", "Пакеты поручений остаются на прежних местах.", ""]
                for row in rows[:100]:
                    snapshot = row["snapshot"]
                    state = {"issued": "Выдано", "running": "В работе", "completed": "Завершено", "cancelled": "Отменено"}.get(row["state"], row["state"])
                    lines += [f"{WorkKind(row['kind']).title} — {state} — {row['id']}",
                        f"Исполнитель: {snapshot['assignee_id']}", f"Назначение: {row['destination']}",
                        f"Наблюдений в старом индексе: {len(row['observations'])}",
                        f"Инструкции: {str(snapshot['instructions'])[:2000]}", ""]
                if len(rows) > 100:
                    lines.append("Показаны первые 100 поручений. Полный план можно сохранить отдельным файлом.")
                self.preview.setPlainText("\n".join(lines))
                self.status.setText("Проверьте план перед сохранением." if rows else "Все поручения уже представлены в журнале.")
            elif self._phase == "apply":
                self._plan = None
                self.status.setText(f"Сохранено поручений: {value['migrated']}.")
                self.migrated.emit(value)
            else:
                self.status.setText(f"Полный план сохранён: {value}")
        except Exception as error:
            if self._phase == "plan":
                self._plan = None
            self.status.setText(f"Ошибка обработки результата: {error}")

    @pyqtSlot(str)
    def _failed(self, message):
        self.status.setText(f"Ошибка: {message}")

    @pyqtSlot()
    def _finished(self):
        self._worker = None
        self.buttons.setEnabled(True)
        self.apply_button.setEnabled(bool(self._plan and self._plan["legacy"]))
        self.save_button.setEnabled(self._plan is not None)

    def reject(self):
        if self._worker is None:
            super().reject()

    def closeEvent(self, event):
        if self._worker is not None:
            event.ignore()
        else:
            super().closeEvent(event)
