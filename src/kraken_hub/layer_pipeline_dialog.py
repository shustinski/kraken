"""Typed layer workflow editor and execution monitor over the existing service."""

from __future__ import annotations

import json
from collections.abc import Mapping
from uuid import uuid4

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from kraken_hub.manager_app_parts.layer_create_thread import _BackgroundCallThread
from kraken_manager.application.errors import NotFoundError
from kraken_manager.domain.pipeline import CAPABILITY_PORTS, LayerPipelineDefinition
from kraken_manager.domain.project import RepresentationKind, RepresentationPurpose


class LayerPipelineDialog(QDialog):
    def __init__(
        self,
        service,
        principal,
        project_id,
        layer_id,
        parent=None,
        *,
        language="Русский",
        plugin_versions=None,
        coordinates=(),
        background_workers=None,
        state_changed=None,
    ):
        super().__init__(parent)
        self.service, self.principal = service, principal
        self.project_id, self.layer_id = str(project_id), str(layer_id)
        self.language = language
        self._versions = dict(plugin_versions or {})
        self._initial_coordinates = tuple(coordinates)
        self._background_workers, self._state_changed = background_workers, state_changed
        self._worker = None
        self._state = None
        self._operations = []
        self._representations = ()
        self._published_models = ()
        self._history = ()
        self._project = None
        self._revision = 0
        self._dirty = False
        self._loading = False
        self._pending = None
        self.setWindowTitle(self._t("Технологический процесс слоя", "Layer process workflow"))
        self.resize(1150, 850)
        root = QVBoxLayout(self)
        self.controls = QWidget(self)
        layout = QVBoxLayout(self.controls)
        root.addWidget(self.controls)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        row = QHBoxLayout()
        self.refresh_button = self._button(row, "Обновить", "Refresh", self.refresh)
        self.start_button = self._button(row, "Запустить", "Start", lambda: self.command("start"))
        self.pause_button = self._button(row, "Пауза", "Pause", lambda: self.command("pause"))
        self.resume_button = self._button(row, "Продолжить", "Resume", lambda: self.command("resume"))
        self.cancel_button = self._button(row, "Отменить", "Cancel", lambda: self.command("cancel"))
        self.approve_button = self._button(
            row, "Разрешить выбранный этап", "Approve selected step", lambda: self.command("approve")
        )
        self.retry_button = self._button(
            row, "Повторить выбранную ветвь", "Retry selected branch", lambda: self.command("retry")
        )
        layout.addLayout(row)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        execution = QWidget()
        execution_layout = QVBoxLayout(execution)
        self.operations = self._table(
            [
                self._t("Этап", "Step"),
                self._t("Операция", "Operation"),
                self._t("Источник", "Source"),
                self._t("Результат", "Output"),
                self._t("Состояние", "State"),
                self._t("Попытки", "Attempts"),
            ]
        )
        execution_layout.addWidget(self.operations)
        self.attempts = self._table(
            [
                self._t("Этап", "Step"),
                self._t("Попытка", "Attempt"),
                self._t("Состояние задания", "Job state"),
                self._t("Ошибка", "Error"),
                self._t("Закреплённые исходные версии", "Pinned input versions"),
                self._t("Версия модели", "Model version"),
            ]
        )
        execution_layout.addWidget(self.attempts)
        self.tabs.addTab(execution, self._t("Выполнение", "Execution"))
        editor = QWidget()
        editor_layout = QVBoxLayout(editor)
        form = QFormLayout()
        self.operation_title = QLineEdit()
        self.capability = QComboBox()
        titles = {
            "frames.vectorize.v1": ("Векторизация Contour", "Contour vectorization"),
            "frames.binary-segment.v1": ("Сегментация NeuralImage v1", "NeuralImage segmentation v1"),
            "frames.binary-segment.v2": ("Сегментация NeuralImage v2", "NeuralImage segmentation v2"),
            "frames.confidence.v1": ("Карта уверенности", "Confidence map"),
            "frames.dataset.prepare.v1": ("Подготовка выборки", "Prepare dataset"),
            "dataset.model.train.v1": ("Обучение модели", "Train model"),
        }
        for code in CAPABILITY_PORTS:
            self.capability.addItem(self._t(*titles[code]), code)
        self.algorithm_version = QLineEdit()
        self.algorithm_version.setPlaceholderText(self._t("Версия установленного плагина", "Installed plugin version"))
        self.input_kind = QComboBox()
        self.output_kind = QLabel()
        self.source = QComboBox()
        self.target = QComboBox()
        self.producer = QComboBox()
        self.model_input = QComboBox()
        self.model_input.setObjectName("pipelineModelInput")
        self.model_details = QLabel()
        self.model_details.setWordWrap(True)
        self.model_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.model_input.currentIndexChanged.connect(self._model_changed)
        self.dependencies = QListWidget()
        self.dependencies.setMaximumHeight(95)
        self.approval = QCheckBox(
            self._t("Требуется подтверждение инженера перед запуском", "Engineer approval required before dispatch")
        )
        self.cpu = self._spin(1, 1024, 1)
        self.gpu = self._spin(0, 1024, 0)
        self.memory = self._spin(1, 1_000_000, 256)
        resource_row = QWidget()
        resource_layout = QHBoxLayout(resource_row)
        resource_layout.setContentsMargins(0, 0, 0, 0)
        for label, widget in (("CPU", self.cpu), ("GPU", self.gpu), (self._t("Память, МБ", "Memory, MB"), self.memory)):
            resource_layout.addWidget(QLabel(label))
            resource_layout.addWidget(widget)
        for label, widget in (
            (self._t("Название этапа", "Step title"), self.operation_title),
            (self._t("Возможность плагина", "Plugin capability"), self.capability),
            (self._t("Ожидаемая версия плагина", "Expected plugin version"), self.algorithm_version),
            (self._t("Тип входа", "Input type"), self.input_kind),
            (self._t("Тип результата", "Output type"), self.output_kind),
            (self._t("Исходное представление", "Source representation"), self.source),
            (self._t("Целевое представление", "Target representation"), self.target),
            (self._t("Вход из предыдущего этапа", "Input producer"), self.producer),
            (self._t("Модель: опубликованная версия или этап обучения", "Model: published version or training step"), self.model_input),
            (self._t("Закреплённая версия модели", "Pinned model version"), self.model_details),
            (self._t("Зависит от этапов", "Dependencies"), self.dependencies),
            (self._t("Ресурсы этапа", "Step resources"), resource_row),
        ):
            form.addRow(label, widget)
        form.addRow(self.approval)
        self.quality_gate = QCheckBox(
            self._t(
                "Блокировать запуск при открытых существенных дефектах",
                "Block dispatch on unresolved major quality issues",
            )
        )
        self.quality_gate.setChecked(True)
        form.addRow(self.quality_gate)
        editor_layout.addLayout(form)
        coordinate_row = QHBoxLayout()
        self.x = self._spin(1, 1_000_000, 1)
        self.y = self._spin(1, 1_000_000, 1)
        coordinate_row.addWidget(QLabel(self._t("Кадр X, Y", "Frame X, Y")))
        coordinate_row.addWidget(self.x)
        coordinate_row.addWidget(self.y)
        self.add_coordinate_button = self._button(coordinate_row, "Добавить кадр", "Add frame", self.add_coordinate)
        self.remove_coordinate_button = self._button(
            coordinate_row, "Убрать кадр", "Remove frame", self.remove_coordinate
        )
        editor_layout.addLayout(coordinate_row)
        self.coordinates = self._table(["X", "Y"])
        self.coordinates.setMaximumHeight(100)
        editor_layout.addWidget(self.coordinates)
        self.parameters = QPlainTextEdit("{}")
        self.parameters.setMaximumHeight(80)
        self.parameters.setPlaceholderText(
            self._t("Параметры алгоритма в формате JSON", "Algorithm parameters as JSON")
        )
        editor_layout.addWidget(QLabel(self._t("Параметры выбранного алгоритма", "Selected algorithm parameters")))
        editor_layout.addWidget(self.parameters)
        row = QHBoxLayout()
        self.add_button = self._button(row, "Добавить этап", "Add step", self.add_operation)
        self.add_stitch_button = self._button(row, "Добавить сшивку…", "Add stitch…", self.add_stitch_operation)
        self.update_button = self._button(row, "Изменить выбранный этап", "Update selected step", self.update_operation)
        self.remove_button = self._button(row, "Удалить выбранный этап", "Remove selected step", self.remove_operation)
        editor_layout.addLayout(row)
        self.tabs.addTab(editor, self._t("Определение этапа", "Step definition"))
        policy = QWidget()
        policy_layout = QFormLayout(policy)
        self.parallel = self._spin(1, 1024, 1)
        self.capacity_cpu = self._spin(1, 1024, 1)
        self.capacity_gpu = self._spin(0, 1024, 0)
        self.capacity_memory = self._spin(1, 1_000_000, 256)
        self.max_attempts = self._spin(1, 1000, 1)
        self.retry_delay = QDoubleSpinBox()
        self.retry_delay.setRange(0, 86400)
        self.retry_delay.setValue(30)
        self.recompute = QCheckBox(
            self._t("Автоматически пересчитывать устаревшие результаты", "Automatically recompute stale results")
        )
        for ru, en, widget in (
            ("Одновременные операции", "Parallel operations", self.parallel),
            ("Доступные CPU", "CPU capacity", self.capacity_cpu),
            ("Доступные GPU", "GPU capacity", self.capacity_gpu),
            ("Общая память, МБ", "Memory capacity, MB", self.capacity_memory),
            ("Попыток в группе", "Attempts per retry group", self.max_attempts),
            ("Пауза между попытками, с", "Retry delay, seconds", self.retry_delay),
        ):
            policy_layout.addRow(self._t(ru, en), widget)
            widget.valueChanged.connect(self._mark_dirty)
        self.recompute.toggled.connect(self._mark_dirty)
        policy_layout.addRow(self.recompute)
        self.save_button = QPushButton(self._t("Сохранить новую версию процесса", "Save new workflow version"))
        self.save_button.clicked.connect(self.save_definition)
        policy_layout.addRow(self.save_button)
        self.tabs.addTab(policy, self._t("Политика выполнения", "Execution policy"))
        history_page = QWidget()
        history_layout = QVBoxLayout(history_page)
        self.history = self._table(
            [
                self._t("Время", "Time"),
                self._t("Действие", "Action"),
                self._t("Состояние", "State"),
                self._t("Версия", "Revision"),
            ]
        )
        history_layout.addWidget(self.history)
        self.tabs.addTab(history_page, self._t("История", "History"))
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)
        self.capability.currentIndexChanged.connect(self._capability_changed)
        self.input_kind.currentIndexChanged.connect(self._port_changed)
        self.producer.currentIndexChanged.connect(self._producer_changed)
        self.operations.itemSelectionChanged.connect(self._operation_selected)
        self._capability_changed()
        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def _t(self, ru, en):
        return en if self.language == "English" else ru

    def _button(self, layout, ru, en, call):
        button = QPushButton(self._t(ru, en))
        button.clicked.connect(lambda checked=False: call())
        layout.addWidget(button)
        return button

    @staticmethod
    def _spin(low, high, value):
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        return spin

    @staticmethod
    def _table(labels):
        table = QTableWidget(0, len(labels))
        table.setHorizontalHeaderLabels(labels)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _mark_dirty(self, *args):
        if not self._loading:
            self._dirty = True

    @staticmethod
    def _event(event, name, default=None):
        return event.get(name, default) if isinstance(event, Mapping) else getattr(event, name, default)

    def _load(self):
        from kraken_hub.published_model_dialog import published_models
        project = self.service.get_project(self.project_id)
        representations = self.service.list_representations(self.project_id, self.layer_id)
        history = self.service.history(self.project_id)
        try:
            state = self.service.get_layer_pipeline(self.project_id, self.layer_id)
        except Exception as exc:
            if isinstance(exc, NotFoundError) or getattr(exc, "status", None) == 404:
                state = None
            else:
                raise
        models = published_models(self.service, self.project_id, self.layer_id)
        return project, representations, history, state, models

    def _run(self, call, succeeded):
        if self._worker is not None:
            return
        self.controls.setEnabled(False)
        self.buttons.setEnabled(False)
        worker = _BackgroundCallThread(call, parent=self)
        self._worker = worker
        if self._background_workers is not None:
            self._background_workers.add(worker)

        def loaded(result):
            try:
                succeeded(result)
            except Exception as exc:
                self.status.setText(self._t("Ошибка: ", "Error: ") + str(exc))

        worker.succeeded.connect(loaded)
        worker.failed.connect(lambda message: self.status.setText(self._t("Ошибка: ", "Error: ") + message))

        def finished():
            if self._background_workers is not None:
                self._background_workers.discard(worker)
            self._worker = None
            self.controls.setEnabled(True)
            self.buttons.setEnabled(True)

        worker.finished.connect(finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def refresh(self):
        if self.isVisible() or self._project is None:
            self._run(self._load, self._loaded)

    def _loaded(self, result):
        project, representations, history, state, self._published_models = result
        self._project, self._representations, self._history = project, tuple(representations), tuple(history)
        self._state = state
        self._loading = True
        self.x.setMaximum(project.width)
        self.y.setMaximum(project.height)
        relevant = [e for e in history if self._event(e, "stream_id") == f"layer-pipeline:{self.layer_id}"]
        self._revision = (
            state["revision"] if state else max((self._event(e, "revision", 0) for e in relevant), default=0)
        )
        if not self._dirty:
            self._operations = list(state["definition"]["operations"]) if state else []
            policy = state["definition"]["policy"] if state else {}
            capacity = policy.get("capacity", {})
            for widget, value in (
                (self.parallel, policy.get("max_parallel", 1)),
                (self.max_attempts, policy.get("max_attempts", 1)),
                (self.retry_delay, policy.get("retry_delay_seconds", 30)),
                (self.capacity_cpu, capacity.get("cpu", 1)),
                (self.capacity_gpu, capacity.get("gpu", 0)),
                (self.capacity_memory, capacity.get("memory_mb", 256)),
            ):
                widget.setValue(value)
            self.recompute.setChecked(policy.get("recompute_stale", False))
        self._port_changed()
        if not self.coordinates.rowCount() and self._initial_coordinates:
            self._set_coordinates(self._initial_coordinates)
        self._redraw()
        workflow_events = [e for e in relevant if self._event(e, "event_type") == "LayerPipelineWorkflowV1"]
        self.history.setRowCount(len(workflow_events))
        for row, event in enumerate(workflow_events):
            payload = self._event(event, "payload", {})
            values = (
                str(self._event(event, "recorded_at", "")),
                str(payload.get("command", {}).get("operation", "update")),
                self._state_label(payload.get("workflow", {}).get("status", "")),
                str(self._event(event, "revision", "")),
            )
            for col, value in enumerate(values):
                self.history.setItem(row, col, QTableWidgetItem(value))
        self._loading = False
        if self._state_changed is not None and state is not None:
            self._state_changed(state)

    def _state_label(self, value):
        ru = {
            "draft": "Черновик",
            "running": "Выполняется",
            "paused": "Приостановлен",
            "cancelling": "Отменяется",
            "cancelled": "Отменён",
            "completed": "Завершён",
            "failed": "Ошибка",
            "stale": "Устарел",
            "pending": "Ожидает",
            "dispatching": "Передаётся Agent",
            "blocked": "Заблокирован",
            "succeeded": "Успешно",
        }
        return self._t(ru.get(value, value), value.replace("_", " "))

    def _kind_label(self, kind):
        labels = {
            "image": ("Изображения", "Images"),
            "binary_mask": ("Бинарная маска", "Binary mask"),
            "vector": ("Векторы", "Vectors"),
            "confidence": ("Уверенность", "Confidence"),
            "dataset": ("Выборка", "Dataset"),
            "model": ("Модель", "Model"),
        }
        return self._t(*labels[str(kind)])

    def _capability_changed(self):
        capability = self.capability.currentData()
        self.input_kind.blockSignals(True)
        self.input_kind.clear()
        for kind in CAPABILITY_PORTS[capability][0]:
            self.input_kind.addItem(self._kind_label(kind), str(kind))
        self.input_kind.blockSignals(False)
        self.output_kind.setText(self._kind_label(CAPABILITY_PORTS[capability][1]))
        self.algorithm_version.setText(str(self._versions.get(capability, "")))
        self._port_changed()

    def _representation_matches(self, representation, kind):
        if kind in {"dataset", "model"}:
            return True
        if kind == "vector":
            return representation.kind is RepresentationKind.VECTOR
        if representation.kind is not RepresentationKind.IMAGE:
            return False
        if kind == "binary_mask":
            return representation.purpose is RepresentationPurpose.BINARY
        if kind == "confidence":
            return representation.purpose is RepresentationPurpose.CONFIDENCE
        return True

    def _port_changed(self):
        capability, input_kind = self.capability.currentData(), self.input_kind.currentData()
        if not capability or not input_kind:
            return
        self.model_input.setEnabled(capability in {"frames.binary-segment.v2", "frames.confidence.v1"})
        for combo, kind in ((self.source, input_kind), (self.target, str(CAPABILITY_PORTS[capability][1]))):
            selected = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for representation in self._representations:
                if self._representation_matches(representation, kind):
                    combo.addItem(representation.name, str(representation.id))
            index = combo.findData(selected)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

    def _selected_id(self):
        row = self.operations.currentRow()
        return (
            self.operations.item(row, 0).data(Qt.ItemDataRole.UserRole)
            if row >= 0 and self.operations.item(row, 0)
            else None
        )

    def _redraw(self):
        selected = self._selected_id()
        self.operations.blockSignals(True)
        self.operations.setRowCount(len(self._operations))
        names = {str(r.id): r.name for r in self._representations}
        nodes = self._state.get("nodes", {}) if self._state else {}
        attempt_rows = []
        for row, operation in enumerate(self._operations):
            node = nodes.get(operation["id"], {})
            values = (
                operation["title"],
                self._t("Сшивка слоя", "Layer stitching") if operation.get("kind") == "layer" else self.capability.itemText(self.capability.findData(operation["capability"])),
                (f"Geometry r{operation['stitch_revision']}" if operation.get("kind") == "layer" else names.get(operation["source_representation_id"], self._t("Недоступен", "Unavailable"))),
                self._t("Мозаика слоя", "Layer mosaic") if operation.get("kind") == "layer" else names.get(operation["target_representation_id"], self._t("Недоступен", "Unavailable")),
                self._state_label(node.get("status", "draft")),
                str(len(node.get("attempts", ()))),
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, operation["id"])
                self.operations.setItem(row, col, item)
            if operation["id"] == selected:
                self.operations.selectRow(row)
            for number, attempt in enumerate(node.get("attempts", ()), 1):
                attempt_rows.append(
                    (
                        operation["title"],
                        str(number),
                        self._state_label(attempt.get("job_state", node.get("status", ""))),
                        str(attempt.get("error") or ""),
                        "\n".join(f"{series}: {version}" for series, version in attempt.get("request", {}).get("input_heads", {}).items()),
                        str(attempt.get("request", {}).get("parameters", {}).get("model_version_id") or "—"),
                    )
                )
        self.operations.blockSignals(False)
        self.attempts.setRowCount(len(attempt_rows))
        for row, values in enumerate(attempt_rows):
            for col, value in enumerate(values):
                self.attempts.setItem(row, col, QTableWidgetItem(value))
        status = self._state["status"] if self._state else "draft"
        self.summary.setText(
            self._t("Состояние: ", "State: ")
            + self._state_label(status)
            + self._t(" · Ревизия ", " · Revision ")
            + str(self._revision)
        )
        active = status in {"running", "paused", "cancelling"}
        for button in (self.add_button, self.add_stitch_button, self.update_button, self.remove_button, self.save_button):
            button.setEnabled(not active)
        self.start_button.setEnabled(self._state is not None and not active and not self._dirty)
        self.pause_button.setEnabled(status == "running")
        self.resume_button.setEnabled(status == "paused")
        self.cancel_button.setEnabled(status in {"running", "paused", "failed"})
        self._refresh_dependencies()
        self._update_node_commands()

    def _refresh_dependencies(self):
        selected = self._selected_id()
        checked = {
            self.dependencies.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.dependencies.count())
            if self.dependencies.item(i).checkState() == Qt.CheckState.Checked
        }
        producer = self.producer.currentData()
        self.dependencies.clear()
        self.producer.blockSignals(True)
        self.producer.clear()
        self.producer.addItem(self._t("Опубликованные исходные данные", "Published input data"), None)
        for operation in self._operations:
            if operation["id"] == selected:
                continue
            item = QListWidgetItem(operation["title"])
            item.setData(Qt.ItemDataRole.UserRole, operation["id"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if operation["id"] in checked else Qt.CheckState.Unchecked)
            self.dependencies.addItem(item)
            if operation.get("kind") != "layer":
                self.producer.addItem(operation["title"], operation["id"])
        index = self.producer.findData(producer)
        self.producer.setCurrentIndex(index if index >= 0 else 0)
        self.producer.blockSignals(False)
        selected_model = self.model_input.currentData()
        self.model_input.blockSignals(True)
        self.model_input.clear()
        self.model_input.addItem(self._t("Существующие параметры / без модели", "Existing parameters / no model"), None)
        for name, version in self._published_models:
            self.model_input.addItem(f"{name} — {version.filename} — {version.created_at.isoformat()} — {str(version.id)[:8]}", f"version:{version.id}")
        for operation in self._operations:
            if operation["id"] != selected and operation["output_kind"] == "model":
                self.model_input.addItem(self._t("Результат этапа: ", "Step output: ") + operation["title"], f"producer:{operation['id']}")
        if selected_model and self.model_input.findData(selected_model) < 0:
            self.model_input.addItem(self._t("Недоступная закреплённая версия: ", "Unavailable pinned version: ") + selected_model, selected_model)
        self.model_input.setCurrentIndex(max(0, self.model_input.findData(selected_model)))
        self.model_input.blockSignals(False)
        self._model_changed()

    def _model_changed(self):
        selected = self.model_input.currentData() or ""
        version = next((version for _, version in self._published_models if selected == f"version:{version.id}"), None)
        self.model_details.setText(
            f"{version.id}\nSHA-256: {version.sha256}" if version is not None else
            self._t("Версия будет закреплена после публикации результата обучения.", "Version is pinned after the training result is published.")
            if selected.startswith("producer:") else "—"
        )

    def _operation_selected(self):
        operation = next((op for op in self._operations if op["id"] == self._selected_id()), None)
        if operation is None:
            self._update_node_commands()
            return
        if operation.get("kind") == "layer":
            self.operation_title.setText(operation["title"])
            self.algorithm_version.setText(operation["algorithm_version"])
            self.parameters.setPlainText(json.dumps(operation["parameters"], ensure_ascii=False, indent=2))
            self._update_node_commands()
            return
        self.operation_title.setText(operation["title"])
        self.capability.setCurrentIndex(self.capability.findData(operation["capability"]))
        self.input_kind.setCurrentIndex(self.input_kind.findData(str(operation["input_kind"])))
        self.algorithm_version.setText(operation["algorithm_version"])
        self.source.setCurrentIndex(self.source.findData(operation["source_representation_id"]))
        self.target.setCurrentIndex(self.target.findData(operation["target_representation_id"]))
        self._refresh_dependencies()
        self.producer.setCurrentIndex(self.producer.findData(operation.get("input_operation_id")))
        model = operation.get("model_input", {})
        selection = f"version:{model['version_id']}" if model.get("version_id") else f"producer:{model['producer_id']}" if model.get("producer_id") else None
        index = self.model_input.findData(selection)
        if index < 0 and selection:
            self.model_input.addItem(self._t("Недоступная закреплённая версия: ", "Unavailable pinned version: ") + selection, selection)
            index = self.model_input.count() - 1
        self.model_input.setCurrentIndex(max(0, index))
        for index in range(self.dependencies.count()):
            item = self.dependencies.item(index)
            item.setCheckState(
                Qt.CheckState.Checked
                if item.data(Qt.ItemDataRole.UserRole) in operation["dependencies"]
                else Qt.CheckState.Unchecked
            )
        self._set_coordinates(operation["coordinates"])
        self.cpu.setValue(operation["resources"]["cpu"])
        self.gpu.setValue(operation["resources"]["gpu"])
        self.memory.setValue(operation["resources"]["memory_mb"])
        self.approval.setChecked(operation["approval_required"])
        self.quality_gate.setChecked(operation.get("quality_gate", True))
        self.parameters.setPlainText(json.dumps(operation["parameters"], ensure_ascii=False, indent=2))
        self._update_node_commands()

    def _update_node_commands(self):
        node = self._state.get("nodes", {}).get(self._selected_id(), {}) if self._state else {}
        self.approve_button.setEnabled(bool(node and node.get("status") == "pending" and not node.get("approved")))
        self.retry_button.setEnabled(node.get("status") in {"failed", "stale", "blocked"})

    def _producer_changed(self):
        operation = next((op for op in self._operations if op["id"] == self.producer.currentData()), None)
        if operation is not None:
            self.input_kind.setCurrentIndex(self.input_kind.findData(str(operation["output_kind"])))
            self.source.setCurrentIndex(self.source.findData(operation["target_representation_id"]))

    def _set_coordinates(self, values):
        self.coordinates.setRowCount(len(values))
        for row, pair in enumerate(values):
            for col, value in enumerate(pair):
                self.coordinates.setItem(row, col, QTableWidgetItem(str(value)))

    def add_coordinate(self):
        values = self._coordinates()
        pair = (self.x.value(), self.y.value())
        if pair not in values:
            self._set_coordinates([*values, pair])

    def remove_coordinate(self):
        row = self.coordinates.currentRow()
        if row >= 0:
            self.coordinates.removeRow(row)

    def _coordinates(self):
        return tuple(
            (int(self.coordinates.item(r, 0).text()), int(self.coordinates.item(r, 1).text()))
            for r in range(self.coordinates.rowCount())
        )

    def _operation_payload(self, identifier):
        parameters = json.loads(self.parameters.toPlainText())
        if not isinstance(parameters, dict):
            raise ValueError(self._t("Параметры должны быть объектом", "Parameters must be an object"))
        deps = [
            self.dependencies.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.dependencies.count())
            if self.dependencies.item(i).checkState() == Qt.CheckState.Checked
        ]
        producer = self.producer.currentData()
        if producer and producer not in deps:
            deps.append(producer)
        model = self.model_input.currentData() if self.model_input.isEnabled() else None
        model_binding = None
        if model:
            kind, identifier_value = model.split(":", 1)
            model_binding = {"version_id" if kind == "version" else "producer_id": identifier_value}
            if kind == "producer" and identifier_value not in deps:
                deps.append(identifier_value)
        return dict(
            id=identifier,
            title=self.operation_title.text().strip(),
            capability=self.capability.currentData(),
            algorithm_version=self.algorithm_version.text().strip(),
            source_representation_id=self.source.currentData(),
            target_representation_id=self.target.currentData(),
            input_kind=self.input_kind.currentData(),
            output_kind=str(CAPABILITY_PORTS[self.capability.currentData()][1]),
            coordinates=self._coordinates(),
            dependencies=deps,
            input_operation_id=producer,
            parameters=parameters,
            resources=dict(cpu=self.cpu.value(), gpu=self.gpu.value(), memory_mb=self.memory.value()),
            approval_required=self.approval.isChecked(),
            quality_gate=self.quality_gate.isChecked(),
            **({"model_input": model_binding} if model_binding else {}),
        )

    def _definition(self, operations=None):
        return dict(
            project_id=self.project_id,
            layer_id=self.layer_id,
            operations=self._operations if operations is None else operations,
            version=(self._state["definition"]["version"] + 1) if self._state else 1,
            policy=dict(
                max_parallel=self.parallel.value(),
                capacity=dict(
                    cpu=self.capacity_cpu.value(), gpu=self.capacity_gpu.value(), memory_mb=self.capacity_memory.value()
                ),
                max_attempts=self.max_attempts.value(),
                retry_delay_seconds=self.retry_delay.value(),
                recompute_stale=self.recompute.isChecked(),
            ),
        )

    def _edit(self, update=False):
        selected = self._selected_id()
        if update and selected is None:
            return
        try:
            operation = self._operation_payload(selected if update else str(uuid4()))
            operations = (
                [operation if old["id"] == selected else old for old in self._operations]
                if update
                else [*self._operations, operation]
            )
            LayerPipelineDefinition.from_dict(self._definition(operations))
        except (ValueError, TypeError, KeyError) as exc:
            self.status.setText(str(exc))
            return
        self._operations = operations
        self._dirty = True
        self._redraw()
        self.status.setText(self._t("Изменения ещё не сохранены", "Changes are not saved yet"))

    def add_operation(self):
        self._edit()

    def update_operation(self):
        operation = next((op for op in self._operations if op["id"] == self._selected_id()), None)
        if operation is not None and operation.get("kind") == "layer":
            self.add_stitch_operation(operation)
            return
        self._edit(True)

    def add_stitch_operation(self, existing=None):
        from kraken_hub.pipeline_stitch_editor import edit_stitch_step
        service, project_id = self.service, self.project_id
        self._run(lambda: service.get_multilayer_model(project_id), lambda snapshot: edit_stitch_step(self, snapshot, existing))

    def remove_operation(self):
        selected = self._selected_id()
        if selected is None:
            return
        if any(selected in op["dependencies"] for op in self._operations if op["id"] != selected):
            self.status.setText(self._t("Сначала измените зависимые этапы", "Update dependent steps first"))
            return
        self._operations = [op for op in self._operations if op["id"] != selected]
        self._dirty = True
        self._redraw()

    def _key(self, command):
        signature = json.dumps(command, sort_keys=True)
        if self._pending is None or self._pending[0] != signature:
            self._pending = (signature, str(uuid4()))
        return self._pending[1]

    def _saved(self, state):
        self._pending = None
        self._dirty = False
        self._state = state
        self._revision = state["revision"]
        self._operations = list(state["definition"]["operations"])
        self._redraw()
        self.status.setText(self._t("Сохранено", "Saved"))
        if self._state_changed is not None:
            self._state_changed(state)

    def save_definition(self):
        try:
            payload = self._definition()
            LayerPipelineDefinition.from_dict(payload)
        except (ValueError, TypeError, KeyError) as exc:
            self.status.setText(str(exc))
            return
        revision = self._revision
        key = self._key({"operation": "define", "definition": payload})
        self._run(
            lambda: self.service.define_layer_pipeline(
                project_id=self.project_id,
                layer_id=self.layer_id,
                definition=payload,
                principal=self.principal,
                expected_revision=revision,
                idempotency_key=key,
            ),
            self._saved,
        )

    def command(self, operation):
        if self._state is None:
            return
        selected = self._selected_id() if operation in {"approve", "retry"} else None
        revision = self._revision
        key = self._key({"operation": operation, "node_id": selected})
        self._run(
            lambda: self.service.command_layer_pipeline(
                project_id=self.project_id,
                layer_id=self.layer_id,
                operation=operation,
                node_id=selected,
                principal=self.principal,
                expected_revision=revision,
                idempotency_key=key,
            ),
            self._saved,
        )

    def reject(self):
        if self._worker is None:
            self._timer.stop()
            super().reject()

    def closeEvent(self, event):
        if self._worker is not None:
            event.ignore()
        else:
            self._timer.stop()
            super().closeEvent(event)
