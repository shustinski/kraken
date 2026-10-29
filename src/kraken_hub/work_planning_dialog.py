"""Engineering work planning through the Hub service boundary."""

from __future__ import annotations

import json
from uuid import uuid4

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
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
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from kraken_hub.manager_app_parts.layer_create_thread import _BackgroundCallThread
from kraken_manager.domain.planning import (
    WORK_ITEM_TRANSITIONS,
    WorkItemState,
    WorkPlan,
    WorkPlanState,
)


class WorkPlanningDialog(QDialog):
    def __init__(
        self, service, principal, project_id: str, parent=None, *, language="Русский", background_workers=None
    ):
        super().__init__(parent)
        self.service, self.principal, self.project_id = service, principal, str(project_id)
        self.language = language
        self.plan: WorkPlan | None = None
        self._worker = None
        self._background_workers = background_workers
        self._pending_command = None
        self._performer_names, self._layer_names = {}, {}
        self._available_layers = []
        self._project = None
        self._evidence_versions = {}
        self._evidence_context = None
        self._evidence_picker = None
        self.setWindowTitle(self._t("Планы восстановления", "Recovery plans"))
        self.resize(1050, 750)
        root = QVBoxLayout(self)
        self.controls = QWidget(self)
        layout = QVBoxLayout(self.controls)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.controls)
        root.addWidget(scroll)
        row = QHBoxLayout()
        self.plans = QComboBox()
        self.plans.setObjectName("workPlans")
        row.addWidget(self.plans, 1)
        self.plan_title = QLineEdit()
        self.plan_title.setPlaceholderText(self._t("Название нового плана", "New plan title"))
        row.addWidget(self.plan_title, 1)
        self.create_button = self._button(row, "Создать план", "Create plan", self.create_plan)
        self.refresh_button = self._button(row, "Обновить", "Refresh", self.refresh)
        self.board_button = self._button(row, "Доска и аналитика", "Board and analytics", self.open_board)
        self.board_button.setEnabled(hasattr(service, "work_plan_analytics"))
        layout.addLayout(row)
        row = QHBoxLayout()
        self.templates = QComboBox()
        self.templates.setObjectName("workTemplates")
        self.templates.setPlaceholderText(self._t("Выберите шаблон", "Choose a template"))
        row.addWidget(self.templates, 1)
        self.new_template_button = self._button(
            row, "Создать шаблон", "Create template", lambda: self.edit_template(False)
        )
        self.edit_template_button = self._button(
            row, "Новая версия шаблона", "New template version", lambda: self.edit_template(True)
        )
        self.apply_template_button = self._button(
            row, "Сформировать работы", "Generate work items", self.apply_template
        )
        supported = hasattr(service, "list_work_templates") and hasattr(service, "save_work_template")
        self.new_template_button.setEnabled(supported)
        self.edit_template_button.setEnabled(supported)
        self.apply_template_button.setEnabled(supported)
        layout.addLayout(row)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        row = QHBoxLayout()
        self.activate_button = self._button(row, "Активировать", "Activate", lambda: self.transition("active"))
        self.complete_button = self._button(
            row, "Завершить план", "Complete plan", lambda: self.transition("completed")
        )
        self.cancel_button = self._button(row, "Отменить план", "Cancel plan", lambda: self.transition("cancelled"))
        self.baseline_button = self._button(
            row, "Зафиксировать базовый план", "Capture baseline", self.capture_baseline
        )
        layout.addLayout(row)
        self.items = QTableWidget(0, 6)
        self.items.setObjectName("workItems")
        self.items.setHorizontalHeaderLabels(
            [
                self._t("Работа", "Work"),
                self._t("Состояние", "State"),
                self._t("Исполнитель", "Performer"),
                self._t("Прогресс, %", "Progress, %"),
                self._t("Трудозатраты, ч", "Effort, hours"),
                self._t("Слои", "Layers"),
            ]
        )
        self.items.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.items.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.items.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.items.horizontalHeader().setStretchLastSection(False)
        self.items.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column, width in ((1, 100), (2, 150), (3, 90), (4, 110), (5, 160)):
            self.items.setColumnWidth(column, width)
        layout.addWidget(self.items, 1)
        form = QFormLayout()
        self.item_title = QLineEdit()
        self.performer = QComboBox()
        self.performer.setObjectName("planningPerformer")
        self.layer_ids = QListWidget()
        self.layer_ids.setObjectName("planningLayers")
        self.layer_ids.setMaximumHeight(70)
        self.layer_ids.setToolTip(self._t("Без выбранных слоёв — весь проект", "No layers selected means the whole project"))
        self.criterion_description = QLineEdit()
        self.template_frames = QLineEdit()
        self.template_frames.setPlaceholderText("1,1; 2,1")
        form.addRow(self._t("Новая работа", "New work"), self.item_title)
        form.addRow(self._t("Исполнитель", "Performer"), self.performer)
        form.addRow(self._t("Слои (без выбора — весь проект)", "Layers (none means whole project)"), self.layer_ids)
        form.addRow(
            self._t("Обязательный критерий приёмки", "Required acceptance criterion"), self.criterion_description
        )
        form.addRow(self._t("Кадры для шаблона (X,Y; X,Y)", "Template frames (X,Y; X,Y)"), self.template_frames)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.add_button = self._button(row, "Добавить работу", "Add work", self.add_item)
        self.assign_button = self._button(row, "Назначить выбранную работу", "Assign selected work", self.assign)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.target_state = QComboBox()
        row.addWidget(self.target_state)
        self.reason = QLineEdit()
        self.reason.setPlaceholderText(self._t("Причина блокировки", "Blocking reason"))
        row.addWidget(self.reason, 1)
        self.move_button = self._button(row, "Изменить состояние", "Change state", self.move_item)
        self.percent = QDoubleSpinBox()
        self.percent.setRange(0, 100)
        self.percent.setSuffix(" %")
        row.addWidget(self.percent)
        self.effort = QDoubleSpinBox()
        self.effort.setRange(0, 1_000_000)
        self.effort.setSuffix(self._t(" ч", " h"))
        row.addWidget(self.effort)
        self.progress_button = self._button(row, "Сохранить прогресс", "Save progress", self.record_progress)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.criteria = QComboBox()
        row.addWidget(self.criteria, 1)
        self.evidence = QLineEdit()
        self.evidence.setReadOnly(True)
        self.evidence.setPlaceholderText(self._t("Версии ещё не выбраны", "No versions selected"))
        row.addWidget(self.evidence, 1)
        self.evidence_button = self._button(row, "Выбрать версии…", "Choose versions…", self.choose_evidence)
        self.clear_evidence_button = self._button(row, "Очистить", "Clear", self.clear_evidence)
        self.satisfy_button = self._button(row, "Подтвердить критерий", "Satisfy criterion", self.satisfy_criterion)
        layout.addLayout(row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)
        self.plans.currentIndexChanged.connect(self.load_selected_plan)
        self.items.itemSelectionChanged.connect(self._selection_changed)
        self.criteria.currentIndexChanged.connect(self._criterion_selected)
        self.performer.currentIndexChanged.connect(self._assignment_changed)
        self.refresh()

    def _t(self, ru, en):
        return en if self.language == "English" else ru

    def _button(self, row, ru, en, callback):
        button = QPushButton(self._t(ru, en))
        button.clicked.connect(lambda _checked=False: callback())
        row.addWidget(button)
        return button

    def _run(self, call, callback):
        if self._worker is not None:
            return
        self.controls.setEnabled(False)
        self.buttons.setEnabled(False)
        self.status.setText(self._t("Выполнение…", "Working…"))
        worker = _BackgroundCallThread(call, parent=self)
        self._worker = worker
        if self._background_workers is not None:
            self._background_workers.add(worker)

        def succeeded(result):
            try:
                callback(result)
                self.status.setText(self._t("Сохранено / загружено", "Saved / loaded"))
            except Exception as exc:
                self.status.setText(str(exc))

        def finished():
            if self._background_workers is not None:
                self._background_workers.discard(worker)
            self._worker = None
            self.controls.setEnabled(True)
            self.buttons.setEnabled(True)

        worker.succeeded.connect(succeeded)
        worker.failed.connect(lambda message: self.status.setText(self._t("Ошибка: ", "Error: ") + message))
        worker.finished.connect(finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def reject(self):
        if self._worker is None and not (self._evidence_picker and self._evidence_picker._worker):
            super().reject()

    def closeEvent(self, event):
        if self._worker is not None or (self._evidence_picker and self._evidence_picker._worker):
            event.ignore()
        else:
            super().closeEvent(event)

    def refresh(self):
        selected = self.plan.id if self.plan else None

        def loaded(result):
            plans, templates, performers, layers, project = result
            self._show_catalog(performers, layers, project)
            self._show_templates(templates)
            self.plans.blockSignals(True)
            self.plans.clear()
            for plan in plans:
                self.plans.addItem(plan.title, plan)
            index = next((i for i in range(self.plans.count()) if self.plans.itemData(i).id == selected), 0)
            self.plans.setCurrentIndex(index if self.plans.count() else -1)
            self.plans.blockSignals(False)
            self._show_plan(self.plans.currentData())

        self._run(
            lambda: (
                self.service.list_work_plans(self.project_id),
                self.service.list_work_templates(self.project_id)
                if hasattr(self.service, "list_work_templates")
                else (),
                self.service.list_project_performers(self.project_id, include_archived=True)
                if hasattr(self.service, "list_project_performers") else
                self.service.list_performers(include_archived=True) if hasattr(self.service, "list_performers") else (),
                self.service.list_layers(self.project_id, include_archived=True) if hasattr(self.service, "list_layers") else (),
                self.service.get_project(self.project_id) if hasattr(self.service, "get_project") else None,
            ),
            loaded,
        )

    def _show_catalog(self, performers, layers, project):
        selected = self.performer.currentData()
        chosen_layers = set(self._selected_layers())
        self._project = project
        self._performer_names = {str(p.id): p.name for p in performers}
        self._layer_names = {str(layer.id): layer.name for layer in layers}
        self.performer.clear()
        self.performer.addItem(self._t("Не назначен", "Unassigned"), None)
        for performer in performers:
            if performer.active:
                self.performer.addItem(performer.name, str(performer.id))
        index = self.performer.findData(selected)
        self.performer.setCurrentIndex(max(0, index))
        self.layer_ids.clear()
        self._available_layers = [layer for layer in layers if str(layer.state) != "archived"]
        for layer in self._available_layers:
            item = QListWidgetItem(layer.name)
            item.setData(Qt.ItemDataRole.UserRole, str(layer.id))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if str(layer.id) in chosen_layers else Qt.CheckState.Unchecked)
            self.layer_ids.addItem(item)

    def _selected_layers(self):
        return [self.layer_ids.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.layer_ids.count())
            if self.layer_ids.item(i).checkState() == Qt.CheckState.Checked]

    def load_selected_plan(self):
        plan = self.plans.currentData()
        if plan:
            self._run(lambda: self.service.get_work_plan(self.project_id, plan.id), self._show_plan)

    def open_board(self):
        from kraken_hub.work_plan_board import WorkPlanBoardDialog

        current = getattr(self, "_board_dialog", None)
        if current is not None and current.isVisible():
            current.raise_()
            return
        self._board_dialog = WorkPlanBoardDialog(
            self.service, self.principal, self.project_id, self, language=self.language,
            background_workers=self._background_workers,
        )
        self._board_dialog.board.workItemActivated.connect(self.open_work_item)
        self._board_dialog.show()

    def open_work_item(self, plan_id, item_id):
        def loaded(plan):
            self._show_plan(plan)
            for row in range(self.items.rowCount()):
                if self.items.item(row, 0).data(Qt.ItemDataRole.UserRole) == item_id:
                    self.items.selectRow(row)
                    break
            self.show()
            self.raise_()
        self._run(lambda: self.service.get_work_plan(self.project_id, plan_id), loaded)

    def _show_plan(self, plan):
        selected = self.selected_item()
        selected_id = selected.id if selected else None
        self.plan = plan
        self.items.blockSignals(True)
        self.items.setRowCount(0)
        if plan:
            index = next((i for i in range(self.plans.count()) if self.plans.itemData(i).id == plan.id), -1)
            self.plans.blockSignals(True)
            if index < 0:
                self.plans.addItem(plan.title, plan)
                index = self.plans.count() - 1
            self.plans.setItemData(index, plan)
            self.plans.setCurrentIndex(index)
            self.plans.blockSignals(False)
            self.summary.setText(
                f"{plan.title} · {self._state_label(plan.state)} · "
                + self._t("Ревизия", "Revision")
                + f" {plan.revision} · "
                + self._t("Базовых планов", "Baselines")
                + f": {len(plan.baselines)}"
            )
            for row, item in enumerate(plan.items):
                self.items.insertRow(row)
                values = [
                    item.title,
                    self._state_label(item.state),
                    self._performer_names.get(str(item.assignment.performer_id), self._t("Недоступный исполнитель", "Unavailable performer")) if item.assignment else "—",
                    f"{item.progress.fraction * 100:g}",
                    f"{item.progress.actual_effort_hours:g}",
                    ", ".join(self._layer_names.get(str(identifier), self._t("Недоступный слой", "Unavailable layer")) for identifier in item.scope.layer_ids)
                        or self._t("Весь проект", "Whole project"),
                ]
                for col, value in enumerate(values):
                    cell = QTableWidgetItem(value)
                    cell.setToolTip(value)
                    cell.setData(Qt.ItemDataRole.UserRole, item.id)
                    self.items.setItem(row, col, cell)
                if item.id == selected_id:
                    self.items.selectRow(row)
            if self.items.rowCount() and not self.items.selectedItems():
                self.items.selectRow(0)
        else:
            self.summary.setText(self._t("Создайте план восстановления", "Create a recovery plan"))
        self.items.blockSignals(False)
        terminal = not plan or plan.state in {WorkPlanState.COMPLETED, WorkPlanState.CANCELLED}
        for button in (self.add_button, self.assign_button, self.baseline_button, self.cancel_button):
            button.setEnabled(not terminal)
        self.apply_template_button.setEnabled(not terminal and self.templates.currentData() is not None)
        self.activate_button.setEnabled(bool(plan and plan.state is WorkPlanState.DRAFT and plan.items))
        self.complete_button.setEnabled(bool(plan and plan.state is WorkPlanState.ACTIVE))
        self._selection_changed()

    def selected_item(self):
        row = self.items.currentRow()
        if self.plan and row >= 0 and self.items.item(row, 0):
            key = self.items.item(row, 0).data(Qt.ItemDataRole.UserRole)
            return next((item for item in self.plan.items if item.id == key), None)
        return None

    def _state_label(self, state):
        ru = {
            "draft": "Черновик",
            "active": "Активен",
            "completed": "Завершён",
            "planned": "Запланирована",
            "running": "Выполняется",
            "blocked": "Заблокирована",
            "in_review": "На проверке",
            "accepted": "Принята",
            "cancelled": "Отменена",
        }
        return self._t(ru.get(str(state), str(state)), str(state).replace("_", " "))

    def _selection_changed(self):
        item = self.selected_item()
        previous_criterion = self.criteria.currentData()
        self.target_state.clear()
        self.criteria.blockSignals(True)
        self.criteria.clear()
        active = bool(self.plan and self.plan.state is WorkPlanState.ACTIVE)
        if item:
            for state in sorted(WORK_ITEM_TRANSITIONS[item.state]):
                self.target_state.addItem(self._state_label(state), str(state))
            self.percent.setValue(item.progress.fraction * 100)
            self.effort.setValue(item.progress.actual_effort_hours)
            for criterion in item.criteria:
                self.criteria.addItem((self._t("Подтверждён: ", "Satisfied: ") if criterion.satisfied else "") + criterion.description, criterion.id)
            index = self.criteria.findData(previous_criterion)
            if index >= 0:
                self.criteria.setCurrentIndex(index)
        self.criteria.blockSignals(False)
        self._criterion_selected()
        self.move_button.setEnabled(active and bool(item) and self.target_state.count() > 0)
        self.progress_button.setEnabled(active and bool(item and item.state is WorkItemState.RUNNING))
        self.satisfy_button.setEnabled(
            active and bool(item and item.state is WorkItemState.IN_REVIEW) and self.criteria.count() > 0
        )
        self.evidence_button.setEnabled(bool(item and self._project is not None))
        self.clear_evidence_button.setEnabled(bool(self._evidence_versions))
        self._assignment_changed()

    def _assignment_changed(self):
        self.assign_button.setEnabled(bool(self.selected_item() and self.plan.state not in
            {WorkPlanState.COMPLETED, WorkPlanState.CANCELLED} and self.performer.currentData()))

    def _key(self, operation, arguments):
        signature = json.dumps([self.plan.id if self.plan else None, operation, arguments], sort_keys=True)
        if self._pending_command is None or self._pending_command[0] != signature:
            self._pending_command = (signature, str(uuid4()))
        return self._pending_command[1]

    def _saved(self, plan):
        self._pending_command = None
        self._show_plan(plan)

    def create_plan(self):
        title = self.plan_title.text().strip()
        key = self._key("create", {"title": title})
        self._run(
            lambda: self.service.create_work_plan(
                project_id=self.project_id,
                title=title,
                principal=self.principal,
                idempotency_key=key,
            ),
            self._saved,
        )

    def _mutate(self, operation, arguments):
        if self.plan is None:
            return
        plan_id, revision = self.plan.id, self.plan.revision
        key = self._key(operation, arguments)
        self._run(
            lambda: self.service.mutate_work_plan(
                project_id=self.project_id,
                plan_id=plan_id,
                principal=self.principal,
                expected_revision=revision,
                operation=operation,
                arguments=arguments,
                idempotency_key=key,
            ),
            self._saved,
        )

    def clear_evidence(self):
        self._evidence_versions = {}
        self.evidence.clear()
        self.evidence.setToolTip("")
        self.clear_evidence_button.setEnabled(False)

    def _criterion_selected(self):
        item = self.selected_item()
        criterion_id = self.criteria.currentData()
        context = (item.id if item else None, criterion_id, item.scope if item else None)
        if context != self._evidence_context:
            self._evidence_context = context
            self.clear_evidence()
            criterion = next((c for c in item.criteria if c.id == criterion_id), None) if item else None
            if criterion:
                self._evidence_versions = {str(identifier): self._t("Сохранённая версия", "Recorded version") for identifier in criterion.evidence_version_ids}
            self._show_evidence_summary()

    def _show_evidence_summary(self):
        self.evidence.setText(self._t("Выбрано версий: ", "Selected versions: ") + str(len(self._evidence_versions)) if self._evidence_versions else "")
        self.evidence.setToolTip("\n".join(self._evidence_versions.values()))
        self.clear_evidence_button.setEnabled(bool(self._evidence_versions))

    def choose_evidence(self):
        from kraken_hub.planning_evidence_picker import PlanningEvidencePicker
        item = self.selected_item()
        if item is None or self._project is None:
            return
        picker = PlanningEvidencePicker(self.service, self.project_id, self._project, self._available_layers,
            item.scope, self._evidence_versions, self, language=self.language, background_workers=self._background_workers)
        self._evidence_picker = picker
        try:
            if picker.exec() == QDialog.DialogCode.Accepted:
                self._evidence_versions = picker.selected_versions
                self._show_evidence_summary()
        finally:
            self._evidence_picker = None
            picker.deleteLater()

    def add_item(self):
        item = {"title": self.item_title.text().strip(), "scope": {"layer_ids": self._selected_layers()}}
        if self.performer.currentData():
            item["assignment"] = {"performer_id": self.performer.currentData()}
        if self.criterion_description.text().strip():
            item["criteria"] = [{"description": self.criterion_description.text().strip(), "required": True}]
        self._mutate("add_item", {"item": item})

    def assign(self):
        item = self.selected_item()
        if item and self.performer.currentData():
            self._mutate("assign", {"item_id": item.id, "assignment": {"performer_id": self.performer.currentData()}})

    def transition(self, state):
        self._mutate("transition", {"state": state})

    def capture_baseline(self):
        self._mutate("capture_baseline", {})

    def move_item(self):
        item = self.selected_item()
        if item and self.target_state.currentData():
            arguments = {"item_id": item.id, "state": self.target_state.currentData()}
            if arguments["state"] == "blocked":
                arguments["reason"] = self.reason.text().strip()
            self._mutate("move_item", arguments)

    def record_progress(self):
        item = self.selected_item()
        if item:
            self._mutate(
                "record_progress",
                {
                    "item_id": item.id,
                    "fraction": self.percent.value() / 100,
                    "actual_effort_hours": self.effort.value(),
                },
            )

    def satisfy_criterion(self):
        item = self.selected_item()
        if item and self.criteria.currentData():
            self._mutate(
                "satisfy_criterion",
                {
                    "item_id": item.id,
                    "criterion_id": self.criteria.currentData(),
                    "evidence_version_ids": list(self._evidence_versions),
                },
            )

    def _show_templates(self, templates):
        selected = self.templates.currentData()
        selected_id = selected.id if selected else None
        self.templates.clear()
        for template in templates:
            self.templates.addItem(f"{template.title} (v{template.version})", template)
        index = next((i for i in range(self.templates.count()) if self.templates.itemData(i).id == selected_id), 0)
        self.templates.setCurrentIndex(index if self.templates.count() else -1)
        self.edit_template_button.setEnabled(bool(templates))
        self.apply_template_button.setEnabled(
            bool(templates and self.plan and self.plan.state in {WorkPlanState.DRAFT, WorkPlanState.ACTIVE})
        )

    def edit_template(self, existing):
        from kraken_hub.work_template_editor import WorkTemplateEditor

        template = self.templates.currentData() if existing else None
        if existing and template is None:
            return
        editor = WorkTemplateEditor(self, template=template, language=self.language)
        if editor.exec() != QDialog.DialogCode.Accepted:
            return
        definition = editor.definition
        identity = {
            "template_id": template.id if template else None,
            "expected_version": template.version if template else 0,
            "definition": definition,
        }
        key = self._key("save_template", identity)

        def saved(result):
            self._pending_command = None
            templates = [self.templates.itemData(i) for i in range(self.templates.count())]
            self._show_templates([t for t in templates if t.id != result.id] + [result])
            self.templates.setCurrentIndex(
                next(i for i in range(self.templates.count()) if self.templates.itemData(i).id == result.id)
            )

        self._run(
            lambda: self.service.save_work_template(
                project_id=self.project_id,
                principal=self.principal,
                idempotency_key=key,
                **identity,
            ),
            saved,
        )

    def apply_template(self):
        template = self.templates.currentData()
        if template is None or self.plan is None:
            return
        try:
            coordinates = []
            for pair in self.template_frames.text().split(";"):
                if pair.strip():
                    values = [int(value.strip()) for value in pair.split(",")]
                    if len(values) != 2:
                        raise ValueError(self._t("Укажите координаты X,Y", "Enter X,Y coordinates"))
                    coordinates.append(values)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        scope = {"layer_ids": self._selected_layers()}
        if coordinates:
            scope["frame_coordinates"] = coordinates
        assignments = {}
        if self.performer.currentData():
            assignments = {task.key: {"performer_id": self.performer.currentData()} for task in template.tasks}
        self._mutate(
            "apply_template",
            {"template_id": template.id, "version": template.version, "scope": scope, "assignments": assignments},
        )
