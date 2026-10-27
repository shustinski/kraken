"""Engineer-facing editor for process template steps, without runtime state."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)
from kraken_manager.infrastructure.planning_templates import decode_template_definition
from kraken_manager.domain.common import new_uuid


class WorkTemplateEditor(QDialog):
    def __init__(self, parent=None, *, template=None, language="English"):
        super().__init__(parent)
        self.language = language
        self._tasks = []
        self._dependencies = []
        self.setWindowTitle(self._t("Шаблон технологического процесса", "Process template"))
        self.resize(800, 620)
        root = QVBoxLayout(self)
        form = QFormLayout()
        self.title = QLineEdit(template.title if template else "")
        self.description = QLineEdit(template.description if template else "")
        form.addRow(self._t("Название шаблона", "Template title"), self.title)
        form.addRow(self._t("Описание", "Description"), self.description)
        root.addLayout(form)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            [
                self._t("Этап", "Step"),
                self._t("Область", "Scope"),
                self._t("Трудоёмкость, ч", "Effort, hours"),
                self._t("Приёмка", "Acceptance"),
            ]
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table)
        form = QFormLayout()
        self.step_title = QLineEdit()
        self.multiplicity = QComboBox()
        for ru, en, mode in (
            ("Весь выбранный объём", "Whole selected scope", "project"),
            ("Каждый выбранный слой", "Each selected layer", "layer"),
            ("Каждый выбранный кадр слоя", "Each selected frame per layer", "frame"),
        ):
            self.multiplicity.addItem(self._t(ru, en), mode)
        self.multiplicity.setCurrentIndex(1)
        self.effort = QDoubleSpinBox()
        self.effort.setRange(0, 1_000_000)
        self.criterion = QLineEdit()
        self.predecessor = QComboBox()
        self.parent_step = QComboBox()
        form.addRow(self._t("Название этапа", "Step title"), self.step_title)
        form.addRow(self._t("Создать работу для", "Create work for"), self.multiplicity)
        form.addRow(self._t("Плановые часы на работу", "Planned hours per work item"), self.effort)
        form.addRow(self._t("Обязательный критерий приёмки", "Required acceptance criterion"), self.criterion)
        form.addRow(self._t("После приёмки этапа", "After acceptance of step"), self.predecessor)
        form.addRow(self._t("Родительский этап", "Parent step"), self.parent_step)
        root.addLayout(form)
        row = QHBoxLayout()
        self.add_button = QPushButton(self._t("Добавить этап", "Add step"))
        self.add_button.clicked.connect(self.add_step)
        row.addWidget(self.add_button)
        self.remove_button = QPushButton(self._t("Удалить последний этап", "Remove last step"))
        self.remove_button.clicked.connect(self.remove_last)
        row.addWidget(self.remove_button)
        root.addLayout(row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        if template:
            data = template.to_dict()["template"]
            self._tasks = [dict(t) for t in data["tasks"]]
            self._dependencies = [dict(d) for d in data["dependencies"]]
        self._redraw()

    def _t(self, ru, en):
        return en if self.language == "English" else ru

    @property
    def definition(self):
        return {
            "title": self.title.text().strip(),
            "description": self.description.text().strip(),
            "tasks": self._tasks,
            "dependencies": self._dependencies,
        }

    def _validate(self, definition):
        decode_template_definition(definition, template_id=new_uuid(), project_id=new_uuid(), version=1)

    def add_step(self):
        task = {
            "key": f"step_{len(self._tasks) + 1}",
            "title": self.step_title.text().strip(),
            "multiplicity": self.multiplicity.currentData(),
            "effort_hours": self.effort.value(),
            "criteria": [],
        }
        # Existing version keys may use different identifiers.
        while any(old["key"] == task["key"] for old in self._tasks):
            task["key"] += "_new"
        if self.criterion.text().strip():
            task["criteria"] = [{"description": self.criterion.text().strip(), "required": True}]
        if self.parent_step.currentData():
            task["parent_key"] = self.parent_step.currentData()
        dependencies = list(self._dependencies)
        if self.predecessor.currentData():
            dependencies.append({"predecessor_key": self.predecessor.currentData(), "successor_key": task["key"]})
        candidate = self.definition | {"tasks": [*self._tasks, task], "dependencies": dependencies}
        try:
            self._validate(candidate | {"title": candidate["title"] or self._t("Новый шаблон", "New template")})
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
            return
        self._tasks.append(task)
        self._dependencies = dependencies
        self.step_title.clear()
        self.criterion.clear()
        self.status.clear()
        self._redraw()

    def remove_last(self):
        if self._tasks:
            removed = self._tasks.pop()["key"]
            self._dependencies = [
                d for d in self._dependencies if removed not in (d["predecessor_key"], d["successor_key"])
            ]
            self._redraw()

    def _redraw(self):
        self.table.setRowCount(len(self._tasks))
        for combo in (self.predecessor, self.parent_step):
            combo.clear()
            combo.addItem(self._t("Нет", "None"), None)
        for row, task in enumerate(self._tasks):
            mode_index = self.multiplicity.findData(str(task["multiplicity"]))
            values = (
                task["title"],
                self.multiplicity.itemText(mode_index),
                str(task.get("effort_hours", 0)),
                "; ".join(c["description"] for c in task.get("criteria", ())),
            )
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))
            for combo in (self.predecessor, self.parent_step):
                combo.addItem(task["title"], task["key"])
        self.remove_button.setEnabled(bool(self._tasks))

    def accept(self):
        try:
            self._validate(self.definition)
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
            return
        super().accept()
