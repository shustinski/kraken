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
    shutdown = []
    controller = object.__new__(DesktopController)
    controller.shell = QWidget()
    controller.service = SimpleNamespace(
        plugin_jobs=lambda: (SimpleNamespace(state=SimpleNamespace(value=state)),)
    )
    controller.agent_runtime = SimpleNamespace(shutdown=lambda: shutdown.append(True))
    controller._background_workers = set()

    assert controller._allow_close() is True
    assert shutdown == [True]

def test_running_agent_job_still_blocks_close(qapp, monkeypatch) -> None:
    messages = []
    controller = object.__new__(DesktopController)
    controller.shell = QWidget()
    controller.service = SimpleNamespace(
        plugin_jobs=lambda: (
            SimpleNamespace(state=SimpleNamespace(value="waiting_for_user")),
        )
    )
    controller.agent_runtime = SimpleNamespace(
        shutdown=lambda: pytest.fail("active Agent must not be stopped")
    )
    controller._background_workers = set()
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, title, message: messages.append((title, message)),
    )

    assert controller._allow_close() is False
    assert messages[0][0] == "Есть активные задания"
