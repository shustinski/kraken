"""Project work board, measured analytics and event-level task drill-down."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from PyQt6.QtCore import QDateTime, Qt, pyqtSignal
from kraken_core.settings import QSettings
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from html import escape
from math import isfinite

from kraken_hub.presentation.qt.parts.layer_create_thread import _BackgroundCallThread
from kraken_manager.application.errors import NotFoundError
from kraken_manager.infrastructure.reports import ReportService
from kraken_manager.infrastructure.reports.work_plans import WORK_PLAN_METRICS

STATES = ("planned", "running", "blocked", "in_review", "accepted", "cancelled")


class WorkPlanBoardWidget(QWidget):
    workItemActivated = pyqtSignal(str, str)

    def __init__(
        self,
        service,
        principal,
        project_id=None,
        parent=None,
        *,
        language="Русский",
        background_workers=None,
        settings=None,
    ):
        super().__init__(parent)
        self.service, self.principal = service, principal
        self.language, self._fixed_project = language, str(project_id) if project_id else None
        self._background_workers, self._worker = background_workers, None
        self._settings = settings or QSettings("Kraken", "KrakenHub")
        self._state_key = f"planning/board/{self._fixed_project or 'catalog'}/{principal.id}"
        try:
            restored = json.loads(str(self._settings.value(self._state_key, "{}")))
            self._initial_filters = restored if isinstance(restored, dict) else {}
        except ValueError, TypeError:
            self._initial_filters = {}
        self.report = None
        self._end_follows_now = True
        self._capacities = {}
        self._drilldown = None
        self._performer_names, self._layer_names = {}, {}
        root = QVBoxLayout(self)
        self.controls = QWidget()
        layout = QVBoxLayout(self.controls)
        root.addWidget(self.controls)
        row = QHBoxLayout()
        self.project = QComboBox()
        self.project.setObjectName("workBoardProject")
        self.plans = QComboBox()
        self.layers = QComboBox()
        self.performers = QComboBox()
        self.states = QComboBox()
        for combo, ru, en in (
            (self.project, "Проект", "Project"),
            (self.plans, "План", "Plan"),
            (self.layers, "Слой", "Layer"),
            (self.performers, "Исполнитель", "Performer"),
            (self.states, "Состояние", "State"),
        ):
            row.addWidget(QLabel(self._t(ru, en)))
            row.addWidget(combo)
        self.plans.addItem(self._t("Все планы", "All plans"), None)
        self.layers.addItem(self._t("Все слои", "All layers"), None)
        self.performers.addItem(self._t("Все исполнители", "All performers"), None)
        self.states.addItem(self._t("Все состояния", "All states"), None)
        for state in STATES:
            self.states.addItem(self._state(state), state)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.start = QDateTimeEdit(QDateTime.currentDateTime().addDays(-30))
        self.end = QDateTimeEdit(QDateTime.currentDateTime())
        self.as_of = QDateTimeEdit(QDateTime.currentDateTime())
        self.live_snapshot = QCheckBox(self._t("Текущий срез", "Current snapshot"))
        self.live_snapshot.setChecked(True)
        for widget, ru, en in (
            (self.start, "Период с", "Period from"),
            (self.end, "До", "To"),
            (self.as_of, "Срез на", "As of"),
        ):
            widget.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            widget.setCalendarPopup(True)
            row.addWidget(QLabel(self._t(ru, en)))
            row.addWidget(widget)
        row.addWidget(self.live_snapshot)
        self.end.dateTimeChanged.connect(lambda: setattr(self, "_end_follows_now", False))
        self.as_of.dateTimeChanged.connect(lambda: self.live_snapshot.setChecked(False))
        self.refresh_button = QPushButton(self._t("Рассчитать", "Calculate"))
        self.refresh_button.clicked.connect(self.refresh)
        row.addWidget(self.refresh_button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.capacity_performer = QComboBox()
        self.capacity = QDoubleSpinBox()
        self.capacity.setRange(0, 24)
        self.capacity.setDecimals(2)
        self.capacity.setSpecialValueText(self._t("Не задана", "Unknown"))
        self.set_capacity = QPushButton(self._t("Задать мощность", "Set capacity"))
        self.set_capacity.clicked.connect(self._set_capacity)
        row.addWidget(QLabel(self._t("Явные часы на календарный день", "Declared hours per calendar day")))
        row.addWidget(self.capacity_performer)
        row.addWidget(self.capacity)
        row.addWidget(self.set_capacity)
        self.csv_button = QPushButton("CSV")
        self.xlsx_button = QPushButton("XLSX")
        self.csv_button.clicked.connect(lambda: self.export("csv"))
        self.xlsx_button.clicked.connect(lambda: self.export("xlsx"))
        row.addWidget(self.csv_button)
        row.addWidget(self.xlsx_button)
        layout.addLayout(row)
        self.search = QLineEdit()
        self.search.setPlaceholderText(
            self._t("Поиск карточек: название, причина блокировки", "Find cards: title, blocking reason")
        )
        self.search.textChanged.connect(self._render_board)
        layout.addWidget(self.search)
        self.metric_drilldown_button = QPushButton(
            self._t("Показать работы выбранного показателя", "Show work behind selected metric")
        )
        self.metric_drilldown_button.setEnabled(False)
        self.metric_drilldown_button.clicked.connect(lambda: self._metric_selected(self.metrics.currentRow(), 0))
        layout.addWidget(self.metric_drilldown_button)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        board = QWidget()
        board_layout = QHBoxLayout(board)
        self._column_labels = {}
        self.columns = {}
        for state in STATES:
            column = QWidget()
            column_layout = QVBoxLayout(column)
            label = QLabel(self._state(state))
            self._column_labels[state] = label
            column_layout.addWidget(label)
            cards = QListWidget()
            cards.setObjectName(f"workBoard_{state}")
            cards.itemClicked.connect(self.show_task)
            cards.itemDoubleClicked.connect(self._open_task)
            column_layout.addWidget(cards)
            board_layout.addWidget(column)
            self.columns[state] = cards
        self.tabs.addTab(board, self._t("Доска работ", "Work board"))
        self.hierarchy = QTreeWidget()
        self.hierarchy.setHeaderLabels(
            [
                self._t("Этап / работа", "Stage / work"),
                self._t("Состояние", "State"),
                self._t("Плановое окончание", "Planned finish"),
            ]
        )
        self.hierarchy.itemClicked.connect(lambda item, column: self._show_row(item.data(0, Qt.ItemDataRole.UserRole)))
        self.hierarchy.itemDoubleClicked.connect(
            lambda item, column: self._activate(item.data(0, Qt.ItemDataRole.UserRole))
        )
        self.tabs.addTab(self.hierarchy, self._t("Иерархия", "Hierarchy"))
        self.metrics = self._table(
            [self._t("Показатель", "Metric"), self._t("Значение", "Value"), self._t("Единицы", "Unit")]
        )
        self.metrics.cellDoubleClicked.connect(self._metric_selected)
        self.metrics.itemSelectionChanged.connect(
            lambda: self.metric_drilldown_button.setEnabled(self.metrics.currentRow() >= 0)
        )
        self.tabs.addTab(self.metrics, self._t("Аналитика", "Analytics"))
        self.workload = self._table(
            [
                self._t("Исполнитель", "Performer"),
                self._t("Открыто работ", "Open work"),
                self._t("Зарегистрировано, ч", "Reported hours"),
                self._t("Осталось по оценке, ч", "Estimated hours left"),
                self._t("Часов/календарный день", "Hours/calendar day"),
            ]
        )
        self.workload.cellDoubleClicked.connect(self._workload_selected)
        self.tabs.addTab(self.workload, self._t("Загрузка исполнителей", "Performer workload"))
        self.baselines = self._table(
            [
                self._t("Работа", "Work"),
                self._t("Изменение объёма", "Scope change"),
                self._t("Разница оценки, ч", "Estimate delta, h"),
                self._t("Сдвиг плана, ч", "Plan shift, h"),
                self._t("Отклонение приёмки, ч", "Acceptance variance, h"),
            ]
        )
        self.baselines.cellDoubleClicked.connect(self._baseline_selected)
        self.tabs.addTab(self.baselines, self._t("Базовые планы", "Baselines"))
        self.compute = self._table(
            [
                self._t("Слой", "Layer"),
                self._t("Активные операции", "Active operations"),
                self._t("Резерв CPU / ёмкость", "CPU reserved / capacity"),
                self._t("Резерв GPU / ёмкость", "GPU reserved / capacity"),
                self._t("Резерв памяти МБ / ёмкость", "Memory MB reserved / capacity"),
                self._t("Измеренная загрузка", "Measured utilization"),
            ]
        )
        self.tabs.addTab(self.compute, self._t("Резервы вычислений", "Compute reservations"))
        self.host_resources = QTextBrowser()
        self.tabs.addTab(self.host_resources, self._t("Ресурсы сейчас", "Current host resources"))
        details = QWidget()
        details_layout = QVBoxLayout(details)
        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(False)
        details_layout.addWidget(self.details)
        self.changes = self._table(
            [
                self._t("Время", "Time"),
                self._t("Состояние", "State"),
                self._t("Ревизия плана", "Plan revision"),
                self._t("Событие", "Event"),
                self._t("Кто", "Who"),
                self._t("Действие", "Action"),
            ]
        )
        details_layout.addWidget(self.changes)
        self.open_button = QPushButton(self._t("Открыть выбранную работу", "Open selected work"))
        self.open_button.clicked.connect(lambda: self._activate(self._selected_task))
        details_layout.addWidget(self.open_button)
        self._selected_task = None
        splitter.addWidget(details)
        splitter.setSizes([450, 230])
        layout.addWidget(splitter, 1)
        self.forecast = QLabel()
        self.forecast.setWordWrap(True)
        layout.addWidget(self.forecast)
        self.status = QLabel()
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.clear_drilldown = QPushButton(self._t("Показать все карточки", "Show all cards"))
        self.clear_drilldown.clicked.connect(self._clear_drilldown)
        layout.addWidget(self.clear_drilldown)
        self.states.setCurrentIndex(max(0, self.states.findData(self._initial_filters.get("states"))))
        self.search.setText(str(self._initial_filters.get("search", "")))
        saved_tab = self._initial_filters.get("tab", 0)
        self.tabs.setCurrentIndex(saved_tab if isinstance(saved_tab, int) and 0 <= saved_tab < self.tabs.count() else 0)
        self.tabs.currentChanged.connect(self._save_preferences)
        for combo in (self.plans, self.layers, self.performers, self.states):
            combo.currentIndexChanged.connect(self.refresh)
        self.project.currentIndexChanged.connect(self._project_changed)
        self.capacity_performer.currentIndexChanged.connect(self._capacity_selected)
        if self._fixed_project:
            try:
                self._capacities = self._capacity_preferences()
            except ValueError, TypeError:
                self._capacities = {}
        self.refresh()

    def _t(self, ru, en):
        return en if self.language == "English" else ru

    def _state(self, value):
        ru = {
            "planned": "Запланировано",
            "running": "В работе",
            "blocked": "Заблокировано",
            "in_review": "На приёмке",
            "accepted": "Принято",
            "cancelled": "Отменено",
        }
        return self._t(ru.get(value, value), value.replace("_", " "))

    @staticmethod
    def _table(headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _format(self, value):
        if value is None:
            return self._t("Нет данных", "Unknown")
        return f"{value:.3g}" if isinstance(value, float) else str(value)

    def _request(self):
        as_of = (
            datetime.now(UTC)
            if self.live_snapshot.isChecked()
            else datetime.fromtimestamp(self.as_of.dateTime().toMSecsSinceEpoch() / 1000, UTC)
        )
        request = {
            "as_of": as_of,
            "start": datetime.fromtimestamp(self.start.dateTime().toSecsSinceEpoch(), UTC),
            "end": as_of
            if self._end_follows_now
            else datetime.fromtimestamp(self.end.dateTime().toMSecsSinceEpoch() / 1000, UTC),
            "capacity_by_performer": dict(self._capacities),
        }
        for combo, key in (
            (self.plans, "plan_ids"),
            (self.layers, "layer_ids"),
            (self.performers, "performer_ids"),
            (self.states, "states"),
        ):
            selected = combo.currentData() or self._initial_filters.get(key)
            if selected:
                request[key] = [selected]
        return request

    def _run(self, call, callback):
        if self._worker is not None:
            return
        self.controls.setEnabled(False)
        worker = _BackgroundCallThread(call, parent=self)
        self._worker = worker
        if self._background_workers is not None:
            self._background_workers.add(worker)

        def succeeded(value):
            try:
                callback(value)
            except Exception as exc:
                self.status.setText(str(exc))

        worker.succeeded.connect(succeeded)
        worker.failed.connect(lambda message: self.status.setText(self._t("Ошибка: ", "Error: ") + message))

        def finished():
            if self._background_workers is not None:
                self._background_workers.discard(worker)
            self._worker = None
            self.controls.setEnabled(True)

        worker.finished.connect(finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def refresh(self, *args):
        request = self._request()
        project_id = self._fixed_project or self.project.currentData() or self._initial_filters.get("project_id")

        def load():
            projects = tuple(self.service.list_projects(include_archived=True))
            known_projects = {str(p.id) for p in projects}
            if self._fixed_project and self._fixed_project not in known_projects:
                raise NotFoundError("The selected project is no longer available")
            chosen = project_id if project_id in known_projects else (str(projects[0].id) if projects else None)
            if not chosen:
                return projects, (), (), None, None
            plans = self.service.list_work_plans(chosen)
            performers = self.service.list_performers(include_archived=True)
            layers = self.service.list_layers(chosen, include_archived=True)
            for key, catalog in (("plan_ids", plans), ("performer_ids", performers), ("layer_ids", layers)):
                if key in request and not set(request[key]) <= {str(item.id) for item in catalog}:
                    request.pop(key)
            report = self.service.work_plan_analytics(chosen, principal=self.principal, **request)
            return projects, plans, (performers, layers), chosen, report

        self._run(load, self._loaded)

    @staticmethod
    def _fill(combo, choices, first=None):
        selected = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        if first is not None:
            combo.addItem(first, None)
        for label, value in choices:
            combo.addItem(label, value)
        found = combo.findData(selected)
        combo.setCurrentIndex(found if found >= 0 else 0)
        combo.blockSignals(False)

    def _loaded(self, result):
        projects, plans, catalogs, chosen, report = result
        self._fill(self.project, [(p.name, str(p.id)) for p in projects])
        self.project.blockSignals(True)
        self.project.setCurrentIndex(self.project.findData(chosen))
        self.project.blockSignals(False)
        self.project.setEnabled(self._fixed_project is None)
        self.report = report
        if report is None:
            self.status.setText(self._t("Нет проектов", "No projects"))
            return
        self._selected_task = None
        self.details.clear()
        self.changes.setRowCount(0)
        for widget, value in ((self.as_of, report["as_of"]), (self.end, report["end"])):
            widget.blockSignals(True)
            widget.setDateTime(QDateTime.fromMSecsSinceEpoch(int(datetime.fromisoformat(value).timestamp() * 1000)))
            widget.blockSignals(False)
        performers, layers = catalogs
        self._performer_names = {str(p.id): p.name for p in performers}
        self._layer_names = {str(layer.id): layer.name for layer in layers}
        self._fill(self.plans, [(p.title, p.id) for p in plans], self._t("Все планы", "All plans"))
        self._fill(
            self.performers, [(p.name, str(p.id)) for p in performers], self._t("Все исполнители", "All performers")
        )
        self._fill(self.capacity_performer, [(p.name, str(p.id)) for p in performers])
        self._fill(self.layers, [(layer.name, str(layer.id)) for layer in layers], self._t("Все слои", "All layers"))
        for combo, key in ((self.plans, "plan_ids"), (self.performers, "performer_ids"), (self.layers, "layer_ids")):
            if self._initial_filters.get(key):
                combo.blockSignals(True)
                combo.setCurrentIndex(max(0, combo.findData(self._initial_filters[key])))
                combo.blockSignals(False)
        self._initial_filters = {}
        self._capacity_selected()
        self._render_board()
        self.hierarchy.clear()
        tree_rows = report["items"][:1000]
        tree_items = {}
        for row in tree_rows:
            item = QTreeWidgetItem([row["title"], self._state(row["state"]), self._format(row["planned_finish"])])
            item.setData(0, Qt.ItemDataRole.UserRole, row)
            tree_items[(row["plan_id"], row["item_id"])] = item
        for row in tree_rows:
            item = tree_items[(row["plan_id"], row["item_id"])]
            parent = tree_items.get((row["plan_id"], row["parent_id"]))
            if parent is not None:
                parent.addChild(item)
            else:
                self.hierarchy.addTopLevelItem(item)
        self.hierarchy.setToolTip(
            self._t("Показано работ: ", "Work items shown: ") + f"{len(tree_rows)}/{len(report['items'])}"
        )
        self.tabs.setTabText(
            1,
            self._t("Иерархия", "Hierarchy")
            + (f" ({len(tree_rows)}/{len(report['items'])})" if len(tree_rows) < len(report["items"]) else ""),
        )
        self.metrics.setRowCount(len(WORK_PLAN_METRICS))
        for row, (key, ru, en, unit, formula) in enumerate(WORK_PLAN_METRICS):
            for col, value in enumerate((self._t(ru, en), self._format(report["summary"].get(key)), unit)):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, key)
                item.setToolTip(formula)
                self.metrics.setItem(row, col, item)
        self.workload.setRowCount(len(report["workload"]))
        for row, item in enumerate(report["workload"]):
            values = (
                self._performer_names.get(item["performer_id"], self._t("Не назначен", "Unassigned")),
                item["open_items"],
                item["reported_effort_hours"],
                item["remaining_estimated_hours"],
                item["capacity_hours_per_calendar_day"],
            )
            for col, value in enumerate(values):
                cell = QTableWidgetItem(self._format(value))
                cell.setData(Qt.ItemDataRole.UserRole, item["performer_id"])
                self.workload.setItem(row, col, cell)
        self.baselines.setRowCount(len(report["baseline_variance"]))
        for row, item in enumerate(report["baseline_variance"]):
            for col, value in enumerate(
                (
                    item["title"],
                    item["scope_change"],
                    item["planned_effort_delta_hours"],
                    item["planned_finish_delta_hours"],
                    item["acceptance_finish_variance_hours"],
                )
            ):
                cell = QTableWidgetItem(self._format(value))
                cell.setData(Qt.ItemDataRole.UserRole, (item["plan_id"], item["item_id"]))
                self.baselines.setItem(row, col, cell)
        forecast = report["forecast"]
        import json
        resources = report.get("current_compute_resources", {"admission": {"state": "unknown"}})
        self.host_resources.setPlainText(
            self._t(
                "Текущие наблюдения всего узла, включая сторонние процессы. Это не история и не учёт по задачам. "
                "unenforced означает, что глобальный бюджет не включён. null означает неизвестное значение.\n\n",
                "Current whole-host observations include unrelated processes; these are not historical or per-job accounting. "
                "unenforced means the global budget is disabled. null means unknown.\n\n",
            ) + json.dumps(resources, ensure_ascii=False, indent=2)
        )
        self.compute.setRowCount(len(report.get("compute_workload", ())))
        for row, item in enumerate(report.get("compute_workload", ())):
            values = (
                self._layer_names.get(item["layer_id"], item["layer_id"]),
                item["active_operations"],
                *(f"{item['reserved'][key]}/{item['declared_capacity'][key]}" for key in ("cpu", "gpu", "memory_mb")),
                self._format(item["measured_utilization"]),
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                cell.setToolTip(
                    self._t(
                        "Объявленные резервы процесса, а не измеренная загрузка оборудования.",
                        "Declared workflow reservations, not measured hardware utilization.",
                    )
                )
                self.compute.setItem(row, column, cell)
        self.forecast.setText(
            self._t(
                "Сценарий окончания конечных работ по явным оценкам: ",
                "Leaf execution scenario from declared estimates: ",
            )
            + self._format(forecast["execution_finish_at"])
            + self._t(". Прогноз формальной приёмки: нет данных.", ". Formal acceptance forecast: unknown.")
            + f" ({forecast['reason']})"
        )
        self.forecast.setToolTip("\n".join(forecast.get("assumptions", ())))
        self.status.setText(
            self._t("Срез: ", "Snapshot: ")
            + report["as_of"]
            + self._t(" · Ревизии планов: ", " · Plan revisions: ")
            + ", ".join(str(r) for r in report["source_plan_revisions"].values())
        )
        self._save_preferences()

    def _render_board(self, *args):
        if self.report is None:
            return
        search = self.search.text().strip().casefold()
        for column in self.columns.values():
            column.clear()
        counts = {state: 0 for state in STATES}
        for row in self.report["items"]:
            key = (row["plan_id"], row["item_id"])
            if self._drilldown is not None and key not in self._drilldown:
                continue
            if (
                search
                and search
                not in (
                    row["title"] + " " + (row["blocking_reason"] or "") + " " + " / ".join(row["hierarchy"])
                ).casefold()
            ):
                continue
            counts[row["state"]] += 1
            if self.columns[row["state"]].count() >= 250:
                continue
            text = row["title"] + "\n" + row["plan_title"]
            if row["hierarchy"]:
                text += "\n" + " › ".join(row["hierarchy"])
            if row["blocking_reason"]:
                text += "\n" + row["blocking_reason"]
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, row)
            self.columns[row["state"]].addItem(item)
        for state, count in counts.items():
            shown = min(count, 250)
            self._column_labels[state].setText(
                self._state(state) + f" ({shown}/{count})" if shown < count else self._state(state) + f" ({count})"
            )
            self.columns[state].setToolTip(
                self._t("Показано ", "Showing ")
                + f"{min(count, 250)}/{count}"
                + self._t(". Уточните фильтры для большого объёма.", ". Refine filters for large scopes.")
            )
        self._save_preferences()

    def _save_preferences(self, *args):
        if self.report is None or self._initial_filters:
            return
        value = {
            "project_id": self.project.currentData(),
            "plan_ids": self.plans.currentData(),
            "layer_ids": self.layers.currentData(),
            "performer_ids": self.performers.currentData(),
            "states": self.states.currentData(),
            "search": self.search.text(),
            "tab": self.tabs.currentIndex(),
        }
        self._settings.setValue(self._state_key, json.dumps(value))

    def show_task(self, item):
        row = item.data(Qt.ItemDataRole.UserRole)
        self._show_row(row)

    def _show_row(self, row):
        self._selected_task = row
        performer = self._performer_names.get(row["performer_id"], self._t("Не назначен", "Unassigned"))
        values = [
            (self._t("План", "Plan"), row["plan_title"]),
            (self._t("Иерархия", "Hierarchy"), " › ".join(row["hierarchy"])),
            (self._t("Состояние", "State"), self._state(row["state"])),
            (self._t("Исполнитель", "Performer"), performer),
            (self._t("Слои", "Layers"), ", ".join(self._layer_names.get(i, i) for i in row["layer_ids"])),
            (self._t("Кадров в области", "Frames in scope"), len(row["frame_ids"])),
            (self._t("Плановые часы", "Planned hours"), row["planned_effort_hours"]),
            (self._t("Зарегистрированные часы", "Reported hours"), row["reported_effort_hours"]),
            (self._t("Причина блокировки", "Blocking reason"), row["blocking_reason"]),
            (self._t("Время до приёмки, ч", "Lead time, hours"), row["lead_hours"]),
            (self._t("Цикл, ч", "Cycle time, hours"), row["cycle_hours"]),
        ]
        html = f"<h3>{escape(row['title'])}</h3>" + "".join(
            f"<p><b>{escape(label)}:</b> {escape(self._format(value))}</p>" for label, value in values
        )
        for criterion in row["criteria"]:
            html += f"<p>{'✓' if criterion['satisfied'] else '○'} {escape(criterion['description'])}<br>{escape(', '.join(criterion['evidence_version_ids']))}</p>"
        self.details.setHtml(html)
        self.changes.setRowCount(len(row["changes"]))
        for r, change in enumerate(row["changes"]):
            for col, value in enumerate(
                (
                    change["at"],
                    self._state(change["state"]),
                    str(change["plan_revision"]),
                    change["event_id"],
                    change.get("actor_name") or self._format(None),
                    change.get("operation") or self._format(None),
                )
            ):
                self.changes.setItem(r, col, QTableWidgetItem(value))

    def _activate(self, row):
        if row:
            self.workItemActivated.emit(row["plan_id"], row["item_id"])

    def _open_task(self, item):
        self._activate(item.data(Qt.ItemDataRole.UserRole))

    def _metric_selected(self, row, column):
        if self.report is None or row < 0:
            return
        key = self.metrics.item(row, 0).data(Qt.ItemDataRole.UserRole)

        def matches(item):
            if key == "total_items":
                return True
            if key == "reported_effort_hours":
                return item["reported_effort_hours"] > 0
            if key == "completed_plans":
                return False
            if not item["is_leaf"]:
                return False
            if key in {"accepted_items", "readiness_fraction", "accepted_frame_scope_count"}:
                return item["state"] == "accepted"
            if key == "mean_lead_hours":
                return item["lead_hours"] is not None
            if key == "mean_cycle_hours":
                return item["cycle_hours"] is not None
            if key.startswith("rework"):
                return item["rework_attempts"] > 0
            if key.startswith("acceptance"):
                return item["acceptances_in_period"] is not None and item["acceptances_in_period"] > 0
            if key == "overdue_items":
                return item["overdue"]
            if key == "reported_effort_hours":
                return item["reported_effort_hours"] > 0
            if key == "measured_compute_utilization":
                return False
            return True

        self._drilldown = {(item["plan_id"], item["item_id"]) for item in self.report["items"] if matches(item)}
        self._render_board()
        self.tabs.setCurrentIndex(0)

    def _clear_drilldown(self):
        self._drilldown = None
        self._render_board()

    def _workload_selected(self, row, column):
        identifier = self.workload.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if identifier is None:
            self._drilldown = {(i["plan_id"], i["item_id"]) for i in self.report["items"] if i["performer_id"] is None}
            self._render_board()
            self.tabs.setCurrentIndex(0)
            return
        index = self.performers.findData(identifier)
        if index >= 0:
            self.performers.setCurrentIndex(index)

    def _baseline_selected(self, row, column):
        key = self.baselines.item(row, 0).data(Qt.ItemDataRole.UserRole)
        item = next((r for r in self.report["items"] if (r["plan_id"], r["item_id"]) == key), None)
        if item:
            card = QListWidgetItem()
            card.setData(Qt.ItemDataRole.UserRole, item)
            self.show_task(card)

    def _capacity_selected(self):
        self.capacity.setValue(self._capacities.get(self.capacity_performer.currentData(), 0))

    def _set_capacity(self):
        identifier = self.capacity_performer.currentData()
        if not identifier:
            return
        if self.capacity.value():
            self._capacities[identifier] = self.capacity.value()
        else:
            self._capacities.pop(identifier, None)
        self._settings.setValue(self._preference_key(), json.dumps(self._capacities))
        self.refresh()

    def _preference_key(self):
        return f"planning/capacity/{self._fixed_project or self.project.currentData()}/{self.principal.id}"

    def _capacity_preferences(self):
        data = json.loads(str(self._settings.value(self._preference_key(), "{}")))
        if not isinstance(data, dict):
            return {}
        return {
            key: value
            for key, value in data.items()
            if isinstance(key, str)
            and isinstance(value, (float, int))
            and not isinstance(value, bool)
            and isfinite(value)
            and value > 0
        }

    def _project_changed(self):
        if self._worker is not None:
            return
        for combo in (self.plans, self.layers, self.performers):
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        try:
            self._capacities = self._capacity_preferences()
        except ValueError, TypeError:
            self._capacities = {}
        self._drilldown = None
        self.refresh()

    def export(self, extension):
        if self.report is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            self._t("Экспорт аналитики работ", "Export work analytics"),
            f"work-plans.{extension}",
            f"{extension.upper()} (*.{extension})",
        )
        if path:
            report = self.report
            method = ReportService().write_work_plan_csv if extension == "csv" else ReportService().write_work_plan_xlsx
            self._run(
                lambda: method(path, report),
                lambda result: self.status.setText(self._t("Сохранено: ", "Saved: ") + str(result)),
            )


class WorkPlanBoardDialog(QDialog):
    def __init__(self, service, principal, project_id, parent=None, *, language="Русский", background_workers=None):
        super().__init__(parent)
        self.setWindowTitle("Work board" if language == "English" else "Доска инженерных работ")
        self.resize(1300, 900)
        layout = QVBoxLayout(self)
        self.board = WorkPlanBoardWidget(
            service, principal, project_id, self, language=language, background_workers=background_workers
        )
        layout.addWidget(self.board)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def reject(self):
        if self.board._worker is None:
            super().reject()

    def closeEvent(self, event):
        if self.board._worker is not None:
            event.ignore()
        else:
            super().closeEvent(event)
