"""test_hub_manager_app: test my work exposes."""
from __future__ import annotations
from datetime import UTC, datetime
from types import SimpleNamespace
from PyQt6.QtWidgets import (
    QPushButton,
    QTableWidget,
)
from kraken_hub.manager_app import (
    _my_work_panel,
)
from kraken_manager.domain.identity import Permission

def test_my_work_exposes_review_and_agent_recovery_actions(qapp) -> None:
    now = datetime.now(UTC)
    batch = SimpleNamespace(
        id="batch-1",
        project_id="project-1",
        layer_id="layer-1",
        assignee_id="performer-1",
        state=SimpleNamespace(value="awaiting_acceptance"),
        due_at=None,
        items=(object(),),
    )
    recovery_job = SimpleNamespace(
        id="job-1",
        project_id="project-1",
        layer_id="layer-1",
        state=SimpleNamespace(value="recovery_required"),
        updated_at=now,
        progress=0.5,
        error="agent stopped",
    )
    partial_job = SimpleNamespace(
        id="job-2",
        project_id="project-1",
        layer_id="layer-1",
        state=SimpleNamespace(value="partial"),
        updated_at=now,
        progress=0.75,
        error="",
    )

    class Service:
        @staticmethod
        def list_projects(*, include_archived):
            assert include_archived
            return (SimpleNamespace(id="project-1", name="Project"),)

        @staticmethod
        def project_permissions(_project_id, _principal):
            return frozenset(
                {
                    Permission.ASSIGN_WORK,
                    Permission.ACCEPT_REVIEW,
                    Permission.MANAGE_REVIEW,
                    Permission.RUN_PLUGIN,
                }
            )

        @staticmethod
        def list_layers(_project_id, *, include_archived):
            assert include_archived
            return (SimpleNamespace(id="layer-1", name="Layer"),)

        @staticmethod
        def list_performers(*, include_archived):
            assert include_archived
            return (SimpleNamespace(id="performer-1", name="Worker"),)

        @staticmethod
        def review_batches():
            return (batch,)

        @staticmethod
        def plugin_jobs():
            return (recovery_job, partial_job)

    controller = SimpleNamespace(
        cancel_agent_job=lambda _job: None,
        retry_agent_job=lambda _job: None,
        import_partial_agent_job=lambda _job: None,
        my_work_refresh=None,
    )
    panel = _my_work_panel(
        Service(),
        SimpleNamespace(principal=SimpleNamespace(id="owner")),
        controller,
    )
    table = panel.findChild(QTableWidget)
    assert table is not None
    assert table.rowCount() == 3
    assert table.item(0, 4).text() == "Ожидает принятия"
    assert table.item(1, 4).text() == "Требуется восстановление"
    assert table.item(2, 4).text() == "Частичный результат"
    assert {
        button.text()
        for button in table.cellWidget(0, 7).findChildren(QPushButton)
    } == {"Повторно выгрузить", "Принять", "На доработку", "Отменить"}
    assert {
        button.text()
        for button in table.cellWidget(1, 7).findChildren(QPushButton)
    } == {"Повторить", "Отменить", "Удалить"}
    assert {
        button.text()
        for button in table.cellWidget(2, 7).findChildren(QPushButton)
    } == {"Импортировать", "Отменить", "Удалить"}
    assert controller.my_work_refresh is not None
