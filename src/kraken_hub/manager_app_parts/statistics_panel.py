"""manager_app: statistics panel."""

from __future__ import annotations
from kraken_hub.composition import EmbeddedProjectService


def _statistics_panel(service: EmbeddedProjectService, principal=None, *, background_workers=None, async_reads=False):
    from datetime import UTC, datetime

    from PyQt6.QtCore import QDateTime, Qt
    from PyQt6.QtWidgets import (
        QComboBox,
        QDateTimeEdit,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    from kraken_manager.infrastructure.reports import (
        METRIC_DEFINITIONS,
        ReportFilters,
        ReportGranularity,
        ReportService,
        present_metrics,
    )

    from kraken_hub.statistics_widgets import MetricChartWidget

    host = QWidget()
    layout = QVBoxLayout(host)
    controls = QHBoxLayout()
    project = QComboBox()
    project.setObjectName("statisticsProjectFilter")
    project.addItem("Все проекты", None)
    initial_projects = () if async_reads else service.list_projects(include_archived=True)
    for item in sorted(
        initial_projects,
        key=lambda value: (value.name.casefold(), str(value.id)),
    ):
        archived = item.state.value == "archived"
        project.addItem(
            f"{item.name} (архивный)" if archived else item.name,
            str(item.id),
        )
    start = QDateTimeEdit(QDateTime.currentDateTime().addDays(-30))
    end = QDateTimeEdit(QDateTime.currentDateTime())
    start.setObjectName("statisticsStart")
    end.setObjectName("statisticsEnd")
    start.setDisplayFormat("dd.MM.yyyy HH:mm")
    end.setDisplayFormat("dd.MM.yyyy HH:mm")
    start.setCalendarPopup(True)
    end.setCalendarPopup(True)
    refresh_button = QPushButton("Рассчитать")
    csv_button = QPushButton("Экспорт CSV")
    xlsx_button = QPushButton("Экспорт XLSX")
    controls.addWidget(QLabel("Проект"))
    controls.addWidget(project)
    controls.addWidget(QLabel("С"))
    controls.addWidget(start)
    controls.addWidget(QLabel("По"))
    controls.addWidget(end)
    controls.addWidget(refresh_button)
    controls.addWidget(csv_button)
    controls.addWidget(xlsx_button)
    controls.addStretch(1)
    layout.addLayout(controls)

    tabs = QTabWidget()
    tabs.setObjectName("statisticsTabs")
    table = QTableWidget(0, 2)
    table.setObjectName("statisticsSummary")
    table.setHorizontalHeaderLabels(("Показатель", "Значение"))
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
    tabs.addTab(table, "Сводка")

    granularities = (
        (ReportGranularity.DAY, "По дням"),
        (ReportGranularity.WEEK, "По неделям"),
        (ReportGranularity.MONTH, "По месяцам"),
        (ReportGranularity.YEAR, "По годам"),
    )
    charts: dict[ReportGranularity, dict[str, MetricChartWidget]] = {}
    for granularity, title in granularities:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        chart_host = QWidget()
        chart_layout = QGridLayout(chart_host)
        chart_layout.setContentsMargins(8, 8, 8, 8)
        chart_layout.setSpacing(10)
        granularity_charts: dict[str, MetricChartWidget] = {}
        for index, definition in enumerate(METRIC_DEFINITIONS):
            chart = MetricChartWidget(definition)
            chart.setObjectName(f"statisticsChart_{granularity.value}_{definition.key}")
            chart_layout.addWidget(chart, index // 2, index % 2)
            granularity_charts[definition.key] = chart
        chart_layout.setColumnStretch(0, 1)
        chart_layout.setColumnStretch(1, 1)
        scroll.setWidget(chart_host)
        tabs.addTab(scroll, title)
        charts[granularity] = granularity_charts
    layout.addWidget(tabs)
    if principal is not None and hasattr(service, "work_plan_analytics"):
        from kraken_hub.work_plan_board import WorkPlanBoardWidget

        work_board = WorkPlanBoardWidget(service, principal, parent=host, background_workers=background_workers)
        tabs.addTab(work_board, "Работы и аналитика")
    reports = ReportService()
    local_timezone = datetime.now().astimezone().tzinfo or UTC

    def filters() -> ReportFilters:
        start_at = datetime.fromtimestamp(start.dateTime().toSecsSinceEpoch(), UTC).replace(
            second=0,
            microsecond=0,
        )
        end_at = datetime.fromtimestamp(end.dateTime().toSecsSinceEpoch(), UTC).replace(
            second=59,
            microsecond=999999,
        )
        selected_project = project.currentData()
        project_ids = frozenset((str(selected_project),)) if selected_project else frozenset()
        return ReportFilters(start_at, end_at, project_ids=project_ids)

    host._report_worker = None
    host._project_catalog_loaded = not async_reads

    def run(call, success, title):
        if not async_reads:
            try:
                success(call())
            except Exception as exc:
                QMessageBox.warning(host, title, str(exc))
            return
        if host._report_worker is not None:
            return
        from kraken_hub.manager_app_parts.layer_create_thread import _BackgroundCallThread

        worker = _BackgroundCallThread(call, parent=host)
        host._report_worker = worker
        for control in (refresh_button, csv_button, xlsx_button, project, start, end):
            control.setEnabled(False)
        if background_workers is not None:
            background_workers.add(worker)

        def loaded(value):
            try:
                success(value)
            except Exception as exc:
                QMessageBox.warning(host, title, str(exc))

        worker.succeeded.connect(loaded)
        worker.failed.connect(lambda message: QMessageBox.warning(host, title, message))

        def finished():
            if background_workers is not None:
                background_workers.discard(worker)
            host._report_worker = None
            for control in (refresh_button, csv_button, xlsx_button, project, start, end):
                control.setEnabled(True)

        worker.finished.connect(finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def refresh() -> None:
        try:
            selected_filters = filters()
            selected_project = project.currentData()
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось рассчитать статистику", str(exc))
            return

        def calculate():
            projects = tuple(service.list_projects(include_archived=True)) if not host._project_catalog_loaded else None
            server_statistics = getattr(service, "statistics", None)
            if selected_project and callable(server_statistics) and service.is_remote_project(selected_project):
                metrics, remote_series = server_statistics(
                    selected_project,
                    start=selected_filters.start,
                    end=selected_filters.end,
                    timezone=local_timezone,
                )
                series_by_granularity = {
                    granularity: remote_series[granularity.value] for granularity, _title in granularities
                }
            else:
                records = service.activity_records()
                metrics = reports.aggregate(records, selected_filters)
                series_by_granularity = {
                    granularity: reports.aggregate_series(
                        records,
                        selected_filters,
                        granularity,
                        timezone=local_timezone,
                    )
                    for granularity, _title in granularities
                }
            return metrics, series_by_granularity, projects

        run(calculate, show_metrics, "Не удалось рассчитать статистику")

    def show_metrics(result):
        metrics, series_by_granularity, projects = result
        if projects is not None:
            for item in sorted(projects, key=lambda value: (value.name.casefold(), str(value.id))):
                project.addItem(
                    f"{item.name} (архивный)" if item.state.value == "archived" else item.name, str(item.id)
                )
            host._project_catalog_loaded = True
        presented = present_metrics(metrics.values)
        table.setRowCount(len(presented))
        for row, metric in enumerate(presented):
            name_item = QTableWidgetItem(metric.definition.label)
            name_item.setToolTip(metric.definition.description)
            value_item = QTableWidgetItem(metric.formatted_value)
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, 0, name_item)
            table.setItem(row, 1, value_item)
        for granularity, series in series_by_granularity.items():
            for definition in METRIC_DEFINITIONS:
                charts[granularity][definition.key].set_series(
                    series.buckets,
                    metrics.values.get(definition.key, 0),
                )

    def export(extension: str) -> None:
        destination, _ = QFileDialog.getSaveFileName(
            host,
            "Экспорт статистики",
            f"kraken-report.{extension}",
            f"{extension.upper()} (*.{extension})",
        )
        if not destination:
            return
        try:
            selected_filters = filters()
        except Exception as exc:
            QMessageBox.warning(host, "Не удалось экспортировать отчёт", str(exc))
            return

        def write():
            records = service.activity_records()
            if extension == "csv":
                return reports.write_csv(destination, records, selected_filters, assume_sorted=True)
            else:
                return reports.write_xlsx(
                    destination,
                    records,
                    selected_filters,
                    assume_sorted=True,
                    include_series=True,
                    timezone=local_timezone,
                )

        run(write, lambda result: None, "Не удалось экспортировать отчёт")

    refresh_button.clicked.connect(refresh)
    csv_button.clicked.connect(lambda: export("csv"))
    xlsx_button.clicked.connect(lambda: export("xlsx"))
    refresh()
    return host
