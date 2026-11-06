"""My Work survives failed remote reads and supports a later refresh."""

from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QLabel, QPushButton, QTableWidget

from kraken_hub.application.remote_error import RemoteServerError
from kraken_hub.presentation.qt.parts.my_work_panel import _my_work_panel


def test_my_work_survives_remote_failure_and_preserves_rows_on_retry(qtbot):
    class Service:
        failure = RemoteServerError("GET /api/v1/projects/project/jobs returned HTTP 500", status=500)

        def list_projects(self, *, include_archived):
            return ()

        def list_performers(self, *, include_archived):
            return ()

        def review_batches(self):
            return ()

        def plugin_jobs(self):
            if self.failure is not None:
                raise self.failure
            return ()

    service = Service()
    controller = SimpleNamespace()
    panel = _my_work_panel(service, SimpleNamespace(principal=None), controller)
    qtbot.addWidget(panel)
    table = panel.findChild(QTableWidget)
    error = next(label for label in panel.findChildren(QLabel) if "HTTP 500" in label.text())
    launch = next(button for button in panel.findChildren(QPushButton) if button.text() == "Запустить плагин")
    refresh = next(button for button in panel.findChildren(QPushButton) if button.text() == "Обновить")
    assert not error.isHidden()
    assert not launch.isEnabled()

    service.failure = None
    refresh.click()
    assert error.isHidden()
    assert launch.isEnabled()
    assert table.rowCount() == 0

    # A failed refresh must leave the last displayed data intact.
    table.setRowCount(1)
    service.failure = RemoteServerError("Server unavailable", status=503)
    controller.my_work_refresh()
    assert table.rowCount() == 1
    assert "Server unavailable" in error.text()
    assert not error.isHidden()
    assert not launch.isEnabled()

    # Programming errors must remain visible to the caller.
    service.failure = ValueError("invalid job")
    with pytest.raises(ValueError, match="invalid job"):
        controller.my_work_refresh()
