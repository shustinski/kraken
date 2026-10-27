from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt

from kraken_hub.electrical_dialog import ElectricalDialog
from tests.test_work_orders import _service

pytest_plugins = ["tests.test_device_candidate_workflow"]


@pytest.mark.parametrize("language", ["English", "Русский"])
def test_device_panel_requires_explicit_kind_roles_and_evidence_then_reopens_reviewed_device(qtbot, device_project, language):
    q = device_project
    dialog = ElectricalDialog(q.service, q.owner, q.project.id, language=language)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=20000)
    dialog.tabs.setCurrentIndex(3)
    assert dialog.device_candidate.count() == 1
    assert "POLY" in dialog.device_candidate.currentText() and "ACTIVE" in dialog.device_candidate.currentText()
    assert dialog.device_source.currentData() is None
    assert dialog.device_drain.currentData() is None
    assert dialog.device_kind.currentData() is None
    assert dialog.device_evidence.currentData() is None
    assert "polarity_not_inferred" in dialog.device_summary.toPlainText()
    revision = dialog.snapshot["electrical"]["revision"]
    qtbot.mouseClick(dialog.realize_device_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=20000)
    assert dialog.snapshot["electrical"]["revision"] == revision
    assert dialog.snapshot["graph"]["devices"] == []
    dialog.device_instance.setText("MX1")
    dialog.device_kind.setCurrentIndex(dialog.device_kind.findData("NMOS"))
    dialog.device_source.setCurrentIndex(dialog.device_source.findData(q.source))
    dialog.device_drain.setCurrentIndex(dialog.device_drain.findData(q.drain))
    dialog.device_evidence.setCurrentIndex(dialog.device_evidence.findData(str(q.process.id)))
    dialog.device_evidence_kind.setCurrentIndex(dialog.device_evidence_kind.findData("process_documentation"))
    dialog.device_note.setText("Independent synthetic process annotation and source/drain assignment reviewed")
    qtbot.mouseClick(dialog.realize_device_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=20000)
    assert not dialog.status.text().startswith(("Error:", "Ошибка:")), dialog.status.text()
    assert dialog.snapshot["graph"]["devices"][0]["kind"] == "NMOS"
    assert dialog.snapshot["graph"]["devices"][0]["bulk_status"] == "unresolved"
    assert len(dialog.snapshot["graph"]["nets"]) == 3
    assert dialog.snapshot["graph"]["connections"] == []
    dialog.tabs.setCurrentIndex(2)
    qtbot.mouseClick(dialog.export_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=20000)
    assert "(instance MX1" in dialog.export_text.toPlainText()
    saved = dialog.snapshot
    dialog.close()
    reopened = ElectricalDialog(_service(q.tmp_path), q.owner, q.project.id, language=language)
    qtbot.addWidget(reopened)
    reopened.show()
    qtbot.waitUntil(lambda: reopened._worker is None, timeout=20000)
    assert reopened.snapshot == saved
    reopened.close()
