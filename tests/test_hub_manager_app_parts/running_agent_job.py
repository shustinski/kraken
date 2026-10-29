"""test_hub_manager_app: test running agent job."""
from __future__ import annotations
from types import SimpleNamespace
import pytest
from PyQt6.QtWidgets import (
    QMessageBox,
    QWidget,
)
from kraken_hub.presentation.qt.manager_app import (
    DesktopController,
)

@pytest.mark.parametrize(
    "state",
    ("recovery_required", "partial", "awaiting_authorization", "importing"),
)
def test_non_running_agent_jobs_do_not_block_close(qapp, state) -> None:
    detached = []
    controller = object.__new__(DesktopController)
    controller.shell = QWidget()
    controller.service = SimpleNamespace(
        plugin_jobs=lambda: (SimpleNamespace(state=SimpleNamespace(value=state)),)
    )
    controller.agent_runtime = SimpleNamespace(detach=lambda: detached.append(True))
    controller._background_workers = set()

    assert controller._allow_close() is True
    assert detached == [True]

@pytest.mark.parametrize("state", ("queued", "staging", "running", "waiting_for_user"))
def test_running_agent_job_does_not_block_close(qapp, monkeypatch, state) -> None:
    detached = []
    controller = object.__new__(DesktopController)
    controller.shell = QWidget()
    controller.service = SimpleNamespace(
        plugin_jobs=lambda: (SimpleNamespace(state=SimpleNamespace(value=state)),)
    )
    controller.agent_runtime = SimpleNamespace(detach=lambda: detached.append(True))
    controller._background_workers = set()
    monkeypatch.setattr(
        QMessageBox, "information", lambda *_args: pytest.fail("no close warning expected")
    )

    assert controller._allow_close() is True
    assert detached == [True]
