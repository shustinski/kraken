"""test_hub_manager_app: test statistics panel has."""
from __future__ import annotations
from datetime import UTC, datetime
from types import SimpleNamespace
from PyQt6.QtWidgets import (
    QComboBox,
    QPushButton,
    QTableWidget,
    QTabWidget,
)
from kraken_hub.manager_app import (
    _statistics_panel,
)
from kraken_hub.statistics_widgets import MetricChartWidget
from kraken_manager.infrastructure.reports import ActivityRecord

def test_statistics_panel_has_human_readable_project_filtered_charts(qapp) -> None:
    now = datetime.now(UTC)
    projects = (
        SimpleNamespace(
            id="project-active",
            name="Active project",
            state=SimpleNamespace(value="active"),
        ),
        SimpleNamespace(
            id="project-archived",
            name="Old project",
            state=SimpleNamespace(value="archived"),
        ),
    )
    records = (
        ActivityRecord("one", now, "artifact.imported", "project-active", bytes_count=1024),
        ActivityRecord("two", now, "artifact.imported", "project-archived", bytes_count=2048),
    )

    class Service:
        @staticmethod
        def list_projects(*, include_archived=False):
            assert include_archived
            return projects

        @staticmethod
        def activity_records():
            return records

    panel = _statistics_panel(Service())
    tabs = panel.findChild(QTabWidget, "statisticsTabs")
    table = panel.findChild(QTableWidget, "statisticsSummary")
    project = panel.findChild(QComboBox, "statisticsProjectFilter")
    assert tabs is not None
    assert table is not None
    assert project is not None
    assert tuple(tabs.tabText(index) for index in range(tabs.count())) == (
        "Сводка",
        "По дням",
        "По неделям",
        "По месяцам",
        "По годам",
    )
    assert table.rowCount() == 18
    assert table.item(0, 0).text() == "Импортировано файлов"
    assert table.item(0, 1).text() == "2"
    assert project.itemText(0) == "Все проекты"
    assert "архивный" in project.itemText(2)

    charts = panel.findChildren(MetricChartWidget)
    assert len(charts) == 72
    daily_imports = panel.findChild(MetricChartWidget, "statisticsChart_day_imported_files")
    assert daily_imports is not None
    assert daily_imports.total_text == "2"
    assert daily_imports.point_count >= 30

    project.setCurrentIndex(1)
    calculate = next(button for button in panel.findChildren(QPushButton) if button.text() == "Рассчитать")
    calculate.click()
    assert table.item(0, 1).text() == "1"
    assert daily_imports.total_text == "1"

    panel.show()
    tabs.setCurrentIndex(1)
    qapp.processEvents()
    assert not daily_imports.grab().isNull()
    panel.close()
