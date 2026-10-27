from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt

from kraken_hub.electrical_dialog import ElectricalDialog
from tests.test_work_orders import _service

pytest_plugins = ["tests.test_electrical_workflow"]


def wait(qtbot, dialog):
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=20000)


def click(qtbot, dialog, button):
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    wait(qtbot, dialog)
    assert not dialog.status.text().startswith(("Error:", "Ошибка:")), dialog.status.text()


@pytest.mark.parametrize("language", ["English", "Русский"])
def test_shown_electrical_panel_reviews_material_binds_contact_terminals_exports_and_reopens(qtbot, electrical, language):
    e, q = electrical, electrical.q
    workers = set()
    dialog = ElectricalDialog(q.service, q.owner, q.project.id, language=language, background_workers=workers)
    qtbot.addWidget(dialog)
    dialog.show()
    wait(qtbot, dialog)
    assert dialog.isVisible()
    assert dialog.table.rowCount() == 2
    assert dialog.adjacency.currentData() is None
    assert dialog.foreground.currentData() is None
    assert "M1" in dialog.table.item(0, 0).text()
    for row in range(2):
        dialog.table.selectRow(row)
        assert "SHA256" in dialog.material_evidence.currentText()
        dialog.material_note.setText("Independent synthetic material image reviewed")
        dialog.net_name.setText("SIGNAL")
        click(qtbot, dialog, dialog.confirm_material)
    assert all(node["state"] == "reviewed_conductor" for node in dialog.snapshot["graph"]["nodes"])
    dialog.tabs.setCurrentIndex(1)
    assert dialog.contact.currentData() == e.contact["id"]
    assert "M1" in dialog.contact.currentText() and "M2" in dialog.contact.currentText()
    dialog.contact_note.setText("Explicit component mapping supported by independent contact-layer observation")
    click(qtbot, dialog, dialog.bind_contact_button)
    assert len(dialog.snapshot["graph"]["connections"]) == 1
    for i, (port, direction) in enumerate([("A", "input"), ("Y", "output")]):
        identifier = e.extractions[i]["components"][0]["id"]
        dialog.terminal_component.setCurrentIndex(dialog.terminal_component.findData(identifier))
        dialog.port.setText(port)
        dialog.direction.setCurrentIndex(dialog.direction.findData(direction))
        dialog.terminal_note.setText("Explicit terminal observation reviewed")
        click(qtbot, dialog, dialog.bind_terminal_button)
    dialog.tabs.setCurrentIndex(2)
    click(qtbot, dialog, dialog.export_button)
    assert "KRAKEN_REVIEWED_PARTIAL" in dialog.export_text.toPlainText()
    assert "(portRef A)" in dialog.export_text.toPlainText()
    assert "(portRef Y)" in dialog.export_text.toPlainText()
    assert "KRAKEN_EVIDENCE_JSON_BASE64" in dialog.export_text.toPlainText()
    assert workers == set()
    saved = dialog.snapshot
    dialog.close()
    reopened = ElectricalDialog(_service(q.tmp_path), q.owner, q.project.id, language=language)
    qtbot.addWidget(reopened)
    reopened.show()
    wait(qtbot, reopened)
    assert reopened.snapshot == saved
    assert len(reopened.snapshot["graph"]["nets"]) == 1
    reopened.close()


def test_panel_requires_explicit_foreground_and_adjacency_without_silent_defaults(qtbot, electrical):
    q = electrical.q
    dialog = ElectricalDialog(q.service, q.owner, q.project.id, language="English")
    qtbot.addWidget(dialog)
    dialog.show()
    wait(qtbot, dialog)
    revision = dialog.snapshot["electrical"]["revision"]
    qtbot.mouseClick(dialog.extract_button, Qt.MouseButton.LeftButton)
    wait(qtbot, dialog)
    assert "Error:" in dialog.status.text()
    assert dialog.snapshot["electrical"]["revision"] == revision
    dialog.adjacency.setCurrentIndex(dialog.adjacency.findData(4))
    dialog.foreground.setCurrentIndex(dialog.foreground.findData(255))
    click(qtbot, dialog, dialog.extract_button)
    assert dialog.snapshot["electrical"]["revision"] == revision + 1
    dialog.close()
