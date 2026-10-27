"""Project quality desk using real artifact versions and asynchronous service calls."""

from functools import partial
from uuid import uuid4

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QScrollArea)

from kraken_hub.manager_app_parts.layer_create_thread import _BackgroundCallThread
from kraken_hub.quality_analysis_dialog import QualityAnalysisControls

_ACTIVE_WORKERS = set()


class QualityDialog(QualityAnalysisControls, QDialog):
    def __init__(self, service, principal, project_id, parent=None, *, background_workers=None):
        super().__init__(parent)
        self.service, self.principal, self.project_id = service, principal, str(project_id)
        self._background_workers = background_workers
        self._worker = None
        self._callback = None
        self._deferred_details = False
        self.issue = None
        self.project = None
        self.setWindowTitle("Контроль качества")
        self.resize(1100, 760)
        root = QVBoxLayout(self)
        self.controls = QWidget(self)
        body = QVBoxLayout(self.controls)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.controls)
        root.addWidget(scroll)
        row = QHBoxLayout()
        button = QPushButton("Обновить")
        button.clicked.connect(self.refresh)
        row.addWidget(button)
        self.status = QLabel()
        row.addWidget(self.status, 1)
        body.addLayout(row)
        self.table = QTableWidget(0, 4)
        self.table.setObjectName("qualityIssues")
        self.table.setHorizontalHeaderLabels(["Дефект", "Критичность", "Состояние", "Версия"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._select_issue)
        body.addWidget(self.table, 2)
        form = QFormLayout()
        self.title = QLineEdit()
        self.description = QLineEdit()
        self.severity = QComboBox()
        for label, value in (("Блокирующий", "blocker"), ("Критический", "critical"), ("Существенный", "major"),
                             ("Незначительный", "minor"), ("Улучшение", "improvement")):
            self.severity.addItem(label, value)
        self.severity.setCurrentIndex(2)
        self.layers = QComboBox()
        self.performers = QComboBox()
        self.x, self.y = QSpinBox(), QSpinBox()
        self.x.setRange(1, 1_000_000_000)
        self.y.setRange(1, 1_000_000_000)
        position = QHBoxLayout()
        position.addWidget(QLabel("X"))
        position.addWidget(self.x)
        position.addWidget(QLabel("Y"))
        position.addWidget(self.y)
        load = QPushButton("Версии выбранного кадра")
        load.clicked.connect(self.load_frame_versions)
        position.addWidget(load)
        for label, widget in (("Название нового дефекта", self.title), ("Описание", self.description),
                              ("Критичность", self.severity), ("Слой", self.layers), ("Исполнитель", self.performers)):
            form.addRow(label, widget)
        form.addRow("Кадр для нового дефекта", position)
        body.addLayout(form)
        self._build_analysis_controls(body)
        split = QSplitter()
        self.versions = QListWidget()
        self.versions.setObjectName("qualityVersions")
        self.versions.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.versions.setToolTip("Выберите неизменяемые версии для дефекта, исправления или повторной проверки")
        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        split.addWidget(self.versions)
        split.addWidget(self.history)
        body.addWidget(split, 2)
        self.note = QLineEdit()
        self.note.setPlaceholderText("Обоснование действия / результат проверки")
        body.addWidget(self.note)
        row = QHBoxLayout()
        commands = (("Создать дефект", self.create_issue), ("Назначить", partial(self.command, "assign")),
                    ("Исправлено", partial(self.command, "correct")), ("Проверка пройдена", partial(self.command, "recheck", True)),
                    ("Проверка не пройдена", partial(self.command, "recheck", False)), ("Принять", partial(self.command, "accept")),
                    ("Отклонить", partial(self.command, "reject")), ("Исключение", partial(self.command, "exception")))
        for label, callback in commands:
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        body.addLayout(row)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)
        self.refresh()

    def _run(self, call, callback):
        if self._worker is not None:
            return
        self.controls.setEnabled(False)
        self._callback = callback
        self._worker = worker = _BackgroundCallThread(call)
        # The thread has no QWidget parent: explicit parent deletion must not
        # destroy a running QThread. Bound slots disconnect with the dialog.
        workers = self._background_workers if self._background_workers is not None else _ACTIVE_WORKERS
        workers.add(worker)
        worker.finished.connect(lambda: workers.discard(worker))
        worker.succeeded.connect(self._succeeded)
        worker.failed.connect(self._failed)
        worker.finished.connect(self._finished)
        worker.finished.connect(worker.deleteLater)
        self.status.setText("Загрузка / сохранение…")
        worker.start()

    def _succeeded(self, result):
        try:
            self._callback(result)
            self.status.setText("Сохранено / загружено")
        except Exception as exc:
            self.status.setText(str(exc))

    def _failed(self, message):
        self.status.setText("Ошибка: " + message)

    def _finished(self):
        self._worker = None
        self._callback = None
        self.controls.setEnabled(True)
        if self._deferred_details:
            self._deferred_details = False
            QTimer.singleShot(0, self._select_issue)

    def reject(self):
        if self._worker is None:
            super().reject()

    def closeEvent(self, event):
        if self._worker is not None:
            event.ignore()
        else:
            super().closeEvent(event)

    def refresh(self):
        service, project_id = self.service, self.project_id
        def load():
            performers = (service.list_project_performers(project_id) if hasattr(service, "list_project_performers")
                          else service.list_performers())
            return service.get_project(project_id), service.list_layers(project_id), performers, service.list_quality_issues(project_id)
        self._run(load, self._loaded)

    def _loaded(self, values):
        self.project, layers, performers, issues = values
        previous = self.issue.id if self.issue else None
        self.layers.clear()
        for layer in layers:
            self.layers.addItem(layer.name, str(layer.id))
        self.performers.clear()
        for performer in performers:
            if performer.active:
                self.performers.addItem(performer.name, str(performer.id))
        self.x.setMaximum(self.project.width)
        self.y.setMaximum(self.project.height)
        self._show_issues(issues, previous)

    def _show_issues(self, issues, selected=None):
        self.table.blockSignals(True)
        self.table.setRowCount(len(issues))
        for row, issue in enumerate(issues):
            for column, value in enumerate((issue.title, issue.severity, issue.state.value, str(issue.revision))):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, issue)
                self.table.setItem(row, column, item)
        self.table.blockSignals(False)
        index = next((row for row, issue in enumerate(issues) if issue.id == selected), -1)
        if index >= 0:
            self.issue = issues[index]
            self.table.selectRow(index)
            self._deferred_details = True

    def _select_issue(self):
        row = self.table.currentRow()
        if row < 0 or self._worker is not None:
            return
        self.issue = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        service, project_id, issue = self.service, self.project_id, self.issue
        def load():
            versions = {}
            for identifier in issue.source_version_ids:
                version = service.artifact_version(project_id, identifier)
                for candidate in service.artifact_versions(project_id, version.series_id):
                    versions[str(candidate.id)] = candidate
            return tuple(versions.values()), service.quality_issue_history(project_id, issue.id)
        self._run(load, self._show_details)

    def _show_details(self, result):
        versions, history = result
        self._show_versions(versions)
        policy = ""
        if self.issue is not None and self.issue.analysis_rule is not None:
            rule = self.issue.analysis_rule
            policy = (f"Правило дефекта: {rule.metric_key} / {rule.value_field} {rule.comparison} {rule.threshold:g}; "
                      f"источник {rule.target_binding}. Повторная проверка использует это сохранённое правило.\n\n")
        self.history.setPlainText(policy + "\n\n".join(
            f"{event['at']} — {event['operation']}\n" +
            (event["issue"]["reviews"][-1]["note"] if event["issue"]["reviews"] else event["issue"]["description"])
            for event in history))

    def _show_versions(self, versions):
        self.versions.clear()
        for version in versions:
            item = QListWidgetItem(f"{version.filename} — {version.created_at:%Y-%m-%d %H:%M:%S} — {version.sha256[:10]}")
            item.setData(Qt.ItemDataRole.UserRole, str(version.id))
            self.versions.addItem(item)

    def load_frame_versions(self):
        if self.project is None or not self.layers.currentData():
            return
        service, project_id, layer_id = self.service, self.project_id, self.layers.currentData()
        self._run(partial(service.quality_frame_versions, project_id, layer_id, self.x.value(), self.y.value()),
                  self._show_versions)

    def selected_versions(self):
        return [item.data(Qt.ItemDataRole.UserRole) for item in self.versions.selectedItems()]

    def create_issue(self):
        if not self.layers.currentData() or not self.selected_versions():
            self.status.setText("Выберите слой, кадр и исходные версии")
            return
        args = {"title": self.title.text(), "description": self.description.text(), "severity": self.severity.currentData(),
                "scope": {"layer_ids": [self.layers.currentData()], "frame_coordinates": [[self.x.value(), self.y.value()]]},
                "source_version_ids": self.selected_versions()}
        self._run(partial(self.service.create_quality_issue, project_id=self.project_id, principal=self.principal,
                          arguments=args, idempotency_key=str(uuid4())), self._saved)

    def command(self, operation, passed=None):
        if self.issue is None:
            self.status.setText("Выберите дефект")
            return
        args = {"note": self.note.text()}
        if operation == "assign":
            args["assignee_id"] = self.performers.currentData()
        if operation in {"correct", "recheck"}:
            args["evidence_version_ids"] = self.selected_versions()
        if operation == "recheck":
            args["passed"] = bool(passed)
        self._run(partial(self.service.mutate_quality_issue, project_id=self.project_id, principal=self.principal,
            issue_id=self.issue.id, expected_revision=self.issue.revision, operation=operation, arguments=args,
            idempotency_key=str(uuid4())), self._saved)

    def _saved(self, issue):
        self.issue = issue
        self.history.setPlainText("\n\n".join(f"{r.at:%Y-%m-%d %H:%M:%S} — {r.action}\n{r.note}" for r in issue.reviews))
        # Refresh runs after the current worker's finished signal, via the user
        # refresh action, keeping one service call active per dialog.
        self._show_issues((issue,), issue.id)
