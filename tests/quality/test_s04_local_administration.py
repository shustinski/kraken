"""S04: local integrity scan and administration catalog refresh."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QPushButton, QTableWidget

from kraken_hub.presentation.qt.parts.administration_panel import _administration_panel
from tests.test_remaining_qt_acceptance import message_pump
from tests.test_work_orders import _project_with_frames, _service


def _button(panel, label):
    return next(button for button in panel.findChildren(QPushButton) if button.text() == label)


def test_s04_scan_integrity_reports_clean_and_errors(qtbot, tmp_path):
    service = _service(tmp_path)
    owner, _, project, layer, images, _ = _project_with_frames(service, tmp_path / "source", with_vectors=False)
    panel = _administration_panel(service, owner)
    qtbot.addWidget(panel)
    panel.show()
    timer, messages = message_pump()
    try:
        qtbot.waitUntil(lambda: not panel._admin_tasks.busy)
        scan = _button(panel, "Проверить целостность")
        QTest.mouseClick(scan, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: len(messages) == 1 and not panel._admin_tasks.busy)
        assert messages[0][0] == "Проверка завершена"
        assert "Ошибок нет" in messages[0][1]

        digest = service.frame_cells(project.id, layer.id, images.id)[0].sha256
        payloads = list(service.catalog_root.rglob(digest))
        assert len(payloads) == 1
        payloads[0].write_bytes(b"corrupted test payload")
        QTest.mouseClick(scan, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: len(messages) == 2 and not panel._admin_tasks.busy)
        assert messages[1][0] == "Обнаружены ошибки"
        assert "Ошибок нет" not in messages[1][1]
    finally:
        timer.stop()
        panel.close()


def test_s04_admin_refresh_lists_projects_and_audit(qtbot, tmp_path):
    service = _service(tmp_path)
    owner, _, project, *_ = _project_with_frames(service, tmp_path / "source", with_vectors=False)
    panel = _administration_panel(service, owner)
    qtbot.addWidget(panel)
    panel.show()
    try:
        qtbot.waitUntil(lambda: not panel._admin_tasks.busy)
        QTest.mouseClick(_button(panel, "Обновить каталог"), Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: not panel._admin_tasks.busy)
        projects = panel.findChildren(QComboBox)[0]
        audit = panel.findChild(QTableWidget)
        assert projects.findData(str(project.id)) >= 0
        assert 1 <= audit.rowCount() <= 500
        assert audit.item(0, 2).text()
    finally:
        panel.close()
