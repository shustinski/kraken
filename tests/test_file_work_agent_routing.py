"""File work packages use their work-order lifecycle instead of Agent controls."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QTableWidget

from kraken_hub.presentation.qt.parts.gateway_job import GatewayJobMixin
from kraken_hub.presentation.qt.parts.my_work_panel import _my_work_panel
from kraken_hub.presentation.qt.parts.poll_agent_jobs import PollAgentJobsMixin


def _order():
    return SimpleNamespace(
        id="work",
        related_job_id="work",
        transport="files",
        project_id="project",
        layer_id="layer",
        assignee_id="performer",
        kind=SimpleNamespace(title="Vectorization", value="vectorize"),
        state=SimpleNamespace(value="issued"),
        due_at=None,
        frame_count=2,
    )


def _job(state="queued"):
    return SimpleNamespace(
        id="work",
        project_id="project",
        layer_id="layer",
        state=SimpleNamespace(value=state),
        progress=0,
        updated_at=datetime.now(UTC),
        error=None,
    )


def test_file_work_is_not_resumed_by_agent():
    calls = []
    controller = SimpleNamespace(
        service=SimpleNamespace(
            plugin_jobs=lambda: (_job(),), work_orders=lambda: (_order(),), is_remote_project=lambda _id: False
        ),
        _gateway_for_job=lambda job: calls.append(job),
        _poll_agent_jobs=lambda: calls.append("poll"),
        shell=SimpleNamespace(statusBar=lambda: SimpleNamespace(showMessage=lambda *args: calls.append(args))),
    )
    PollAgentJobsMixin._resume_agent_jobs(controller)
    assert calls == []


def test_poll_does_not_fetch_results_for_file_work():
    controller = SimpleNamespace(
        service=SimpleNamespace(
            synchronize_plugin_jobs=lambda **kwargs: (_job("importing"),), work_orders=lambda: (_order(),)
        ),
        session=SimpleNamespace(principal=None),
        _agent_gateway=SimpleNamespace(get_publications=lambda _id: pytest.fail("File work queried through Agent")),
        my_work_refresh=None,
        _agent_poll_error="",
        shell=SimpleNamespace(statusBar=lambda: SimpleNamespace(showMessage=lambda *args: pytest.fail(str(args)))),
    )
    PollAgentJobsMixin._poll_agent_jobs(controller)


def test_file_work_cannot_open_agent_gateway():
    controller = SimpleNamespace(service=SimpleNamespace(work_orders=lambda: (_order(),)))
    with pytest.raises(ValueError, match="выданной работы"):
        GatewayJobMixin._gateway_for_job(controller, _job())


@pytest.mark.parametrize("operation", ["cancel_agent_job", "retry_agent_job", "import_partial_agent_job"])
@pytest.mark.parametrize("remote", [False, True])
def test_file_work_cannot_use_agent_lifecycle(operation, remote):
    controller = SimpleNamespace(
        service=SimpleNamespace(work_orders=lambda: (_order(),), is_remote_project=lambda _id: remote),
    )
    controller._gateway_for_job = lambda job: GatewayJobMixin._gateway_for_job(controller, job)
    with pytest.raises(ValueError, match="выданной работы"):
        getattr(GatewayJobMixin, operation)(controller, _job())


def test_file_work_is_listed_once_with_work_order_controls(qtbot):
    service = SimpleNamespace(
        list_projects=lambda **kwargs: (),
        list_performers=lambda **kwargs: (),
        review_batches=lambda: (),
        plugin_jobs=lambda: (_job(),),
        work_orders=lambda: (_order(),),
    )
    panel = _my_work_panel(service, SimpleNamespace(principal=None), SimpleNamespace())
    qtbot.addWidget(panel)
    table = panel.findChild(QTableWidget)
    assert table.rowCount() == 1
    assert table.item(0, 0).text() == "Vectorization"
