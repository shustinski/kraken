"""Named immutable-version selection on demand through existing service APIs."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from kraken_hub.manager_app_parts.layer_create_thread import _BackgroundCallThread


class PlanningEvidencePicker(QDialog):
    def __init__(self, service, project_id, project, layers, scope, selected, parent=None, *, language="Русский", background_workers=None):
        super().__init__(parent)
        self.service, self.project_id, self.project, self.scope = service, project_id, project, scope
        self._worker = None
        self._background_workers = background_workers
        self._selected = dict(selected)
        self._english = language == "English"
        self.setWindowTitle(self._t("Доказательства приёмки", "Acceptance evidence"))
        self.resize(760, 480)
        root = QVBoxLayout(self)
        self.controls = QWidget()
        body = QVBoxLayout(self.controls)
        root.addWidget(self.controls)
        row = QHBoxLayout()
        self.layer = QComboBox()
        for item in layers:
            if not scope.layer_ids or str(item.id) in scope.layer_ids:
                self.layer.addItem(item.name, str(item.id))
        row.addWidget(self.layer, 1)
        self.kind = QComboBox()
        self.kind.addItem(self._t("Кадр", "Frame"), "frame")
        if not scope.frame_ids:
            self.kind.addItem(self._t("Результаты слоя", "Layer attachments"), "layer_attachment")
            if not scope.layer_ids:
                self.kind.addItem(self._t("Результаты проекта", "Project attachments"), "project_attachment")
        row.addWidget(self.kind)
        self.x, self.y = QSpinBox(), QSpinBox()
        self.x.setRange(1, project.width)
        self.y.setRange(1, project.height)
        row.addWidget(QLabel("X"))
        row.addWidget(self.x)
        row.addWidget(QLabel("Y"))
        row.addWidget(self.y)
        self.load_button = QPushButton(self._t("Загрузить версии", "Load versions"))
        self.load_button.clicked.connect(self.load)
        row.addWidget(self.load_button)
        body.addLayout(row)
        self.series = QComboBox()
        self.series.setVisible(False)
        self.series.currentIndexChanged.connect(self.load_series)
        body.addWidget(self.series)
        self.versions = QListWidget()
        self.versions.setObjectName("planningEvidenceVersions")
        self.versions.itemChanged.connect(self._checked)
        body.addWidget(self.versions, 1)
        selected_row = QHBoxLayout()
        selected_row.addWidget(QLabel(self._t("Выбранные доказательства", "Selected evidence")), 1)
        self.remove_button = QPushButton(self._t("Убрать выбранное", "Remove selected"))
        self.remove_button.clicked.connect(self._remove_selected)
        selected_row.addWidget(self.remove_button)
        body.addLayout(selected_row)
        self.chosen = QListWidget()
        self.chosen.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.chosen.setMaximumHeight(100)
        self.chosen.itemSelectionChanged.connect(lambda: self.remove_button.setEnabled(bool(self.chosen.selectedItems())))
        body.addWidget(self.chosen)
        self.status = QLabel()
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.layer.currentIndexChanged.connect(self._context_changed)
        self.x.valueChanged.connect(self._context_changed)
        self.y.valueChanged.connect(self._context_changed)
        self._kind_changed()
        self._refresh_chosen()
        if selected and hasattr(service, "artifact_version"):
            identifiers = tuple(selected)
            def restored(values):
                for identifier, value in zip(identifiers, values, strict=True):
                    self._selected[identifier] = self._label(value) if value else self._t("Сохранённая версия недоступна", "Recorded version unavailable")
                self._refresh_chosen()
                self.status.setText(self._t("Сохранённые доказательства загружены", "Recorded evidence loaded"))
            self._run(lambda: [service.artifact_version(project_id, identifier) for identifier in identifiers], restored)

    def _t(self, ru, en):
        return en if self._english else ru

    @property
    def selected_versions(self):
        return dict(self._selected)

    def _kind_changed(self):
        frame = self.kind.currentData() == "frame"
        self.x.setEnabled(frame)
        self.y.setEnabled(frame)
        self.layer.setEnabled(self.kind.currentData() != "project_attachment")
        self.load_button.setEnabled(self.kind.currentData() == "project_attachment" or self.layer.count() > 0)
        self._context_changed()

    def _context_changed(self):
        self.series.setVisible(False)
        self.versions.clear()

    def _run(self, call, callback):
        if self._worker is not None:
            return
        self.controls.setEnabled(False)
        self.buttons.setEnabled(False)
        self.status.setText(self._t("Загрузка…", "Loading…"))
        worker = _BackgroundCallThread(call, parent=self)
        self._worker = worker
        if self._background_workers is not None:
            self._background_workers.add(worker)
        def succeeded(result):
            try:
                callback(result)
            except Exception as error:
                self.status.setText(str(error))
        worker.succeeded.connect(succeeded)
        worker.failed.connect(lambda error: self.status.setText(error))

        def finished():
            self._worker = None
            if self._background_workers is not None:
                self._background_workers.discard(worker)
            self.controls.setEnabled(True)
            self.buttons.setEnabled(True)
        worker.finished.connect(finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def load(self):
        kind, layer = self.kind.currentData(), self.layer.currentData()
        if kind == "frame":
            if layer is None:
                return
            x, y = self.x.value(), self.y.value()
            if self.scope.frame_ids and str(self.project.frame_id_at(x, y)) not in self.scope.frame_ids:
                self.status.setText(self._t("Кадр не входит в выбранную работу", "Frame is outside the selected work"))
                return
            self._run(lambda: self.service.quality_frame_versions(self.project_id, layer, x, y), self._show_versions)
        else:
            def read():
                return [s for s in self.service.list_artifact_series(self.project_id, layer_id=layer if kind == "layer_attachment" else None)
                    if str(s.scope) == kind and not s.archived]
            self._run(read, self._show_series)

    def _show_series(self, values):
        self.series.blockSignals(True)
        self.series.clear()
        self.series.addItem(self._t("Выберите результат", "Choose a result"), None)
        for item in values:
            self.series.addItem(item.name, str(item.id))
        self.series.blockSignals(False)
        self.series.setVisible(True)
        self.versions.clear()
        self.status.setText(self._t("Выберите результат для загрузки его истории версий", "Choose a result to load its version history"))

    def load_series(self):
        identifier = self.series.currentData()
        if identifier is not None:
            self._run(lambda: self.service.artifact_versions(self.project_id, identifier), self._show_versions)

    def _show_versions(self, values):
        self.versions.blockSignals(True)
        self.versions.clear()
        for version in sorted(values, key=lambda v: (v.created_at, str(v.id)), reverse=True):
            label = self._label(version)
            if str(version.id) in self._selected:
                self._selected[str(version.id)] = label
            item = QListWidgetItem(label)
            item.setToolTip(label)
            item.setData(Qt.ItemDataRole.UserRole, str(version.id))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if str(version.id) in self._selected else Qt.CheckState.Unchecked)
            self.versions.addItem(item)
        self.versions.blockSignals(False)
        self._refresh_chosen()
        self.status.setText(self._t("Выберите версии флажками; выбор сохраняется между загрузками", "Check versions; selections are retained across loads")
            if self.versions.count() else self._t("Здесь нет версий; предыдущий выбор сохранён", "No versions here; previous selections retained"))

    def _checked(self, item):
        identifier = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            self._selected[identifier] = item.text()
        else:
            self._selected.pop(identifier, None)
        self._refresh_chosen()

    @staticmethod
    def _label(version):
        return f"{version.filename} · {version.created_at:%Y-%m-%d %H:%M:%S.%f} · SHA {version.sha256[:12]}"

    def _refresh_chosen(self):
        self.chosen.clear()
        for identifier, label in self._selected.items():
            item = QListWidgetItem(label)
            item.setToolTip(label)
            item.setData(Qt.ItemDataRole.UserRole, identifier)
            self.chosen.addItem(item)
        self.remove_button.setEnabled(False)

    def _remove_selected(self):
        identifiers = [item.data(Qt.ItemDataRole.UserRole) for item in self.chosen.selectedItems()]
        for identifier in identifiers:
            self._selected.pop(identifier, None)
        self.versions.blockSignals(True)
        for i in range(self.versions.count()):
            item = self.versions.item(i)
            if item.data(Qt.ItemDataRole.UserRole) in identifiers:
                item.setCheckState(Qt.CheckState.Unchecked)
        self.versions.blockSignals(False)
        self._refresh_chosen()

    def reject(self):
        if self._worker is None:
            super().reject()

    def accept(self):
        if self._worker is None:
            super().accept()

    def closeEvent(self, event):
        if self._worker is not None:
            event.ignore()
        else:
            super().closeEvent(event)
