from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt

from kraken_hub.multilayer_dialog import MultilayerDialog
from tests.test_work_orders import _service

pytest_plugins = ["tests.test_multilayer_workflow"]


def wait(qtbot, dialog):
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=20000)


def click(qtbot, dialog, button):
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    wait(qtbot, dialog)
    assert "Error:" not in dialog.status.text(), dialog.status.text()
    assert "Ошибка:" not in dialog.status.text(), dialog.status.text()


@pytest.mark.parametrize("language", ["English", "Русский"])
def test_real_dialog_calibration_manual_registration_contact_review_and_reopen(qtbot, physical, language):
    q = physical
    workers = set()
    dialog = MultilayerDialog(q.service, q.owner, q.project.id, language=language, background_workers=workers)
    qtbot.addWidget(dialog)
    dialog.show()
    wait(qtbot, dialog)
    assert dialog.isVisible()
    assert {dialog.stack.item(row, 1).text() for row in range(dialog.stack.rowCount())} == {"M1", "V1", "M2"}
    assert q.layers[0].name in dialog.source_layer.itemText(0)
    assert q.versions[0].sha256[:12] in dialog.calibration_version.currentText()
    dialog.tabs.setCurrentIndex(0)
    click(qtbot, dialog, dialog.stack_button)
    dialog.tabs.setCurrentIndex(1)
    for layer in (q.layers[0], q.layers[2]):
        dialog.calibration_layer.setCurrentIndex(dialog.calibration_layer.findData(str(layer.id)))
        dialog.scale_x.setText("1e-6")
        dialog.scale_y.setText("1e-6")
        dialog.y_axis.setCurrentIndex(dialog.y_axis.findData("up"))
        dialog.uncertainty.setText("1e-9")
        click(qtbot, dialog, dialog.calibrate_button)
    dialog.tabs.setCurrentIndex(2)
    dialog.source_layer.setCurrentIndex(dialog.source_layer.findData(str(q.layers[0].id)))
    dialog.target_layer.setCurrentIndex(dialog.target_layer.findData(str(q.layers[2].id)))
    for point in ((0, 0), (10, 0), (0, 10), (10, 10)):
        dialog.add_point()
        row = dialog.points.rowCount() - 1
        for column, value in enumerate([*point, *point, "1e-9"]):
            dialog.points.item(row, column).setText(str(value))
    dialog.tolerance.setText("1e-9")
    click(qtbot, dialog, dialog.register_button)
    assert dialog.snapshot["model"]["registrations"][-1]["fit"]["status"] == "fitted"
    assert "RMS=" in dialog.fit_summary.text()
    dialog.tabs.setCurrentIndex(3)
    dialog.contact_layer.setCurrentIndex(dialog.contact_layer.findData(str(q.layers[1].id)))
    dialog.contact_x.setText("5e-6")
    dialog.contact_y.setText("5e-6")
    dialog.contact_radius.setText("1e-8")
    click(qtbot, dialog, dialog.propose_button)
    assert dialog.snapshot["model"]["contacts"][-1]["state"] == "candidate"
    assert str(q.versions[1].id) == dialog.evidence.currentData()
    dialog.note.setText("Independent contact-layer observation reviewed")
    click(qtbot, dialog, dialog.confirm_button)
    assert dialog.snapshot["assessment"]["contacts"][-1]["electrical_connectivity"] == "engineer_confirmed"
    saved = dialog.snapshot
    assert workers == set()
    dialog.close()
    reopened = MultilayerDialog(_service(q.tmp_path), q.owner, q.project.id, language=language)
    qtbot.addWidget(reopened)
    reopened.show()
    wait(qtbot, reopened)
    assert reopened.snapshot == saved
    assert reopened.contacts.item(reopened.contacts.rowCount() - 1, 2).text() == "confirmed"
    reopened.close()


def test_dialog_preserves_unknown_degenerate_result_and_reports_invalid_ui_input(qtbot, physical):
    q = physical
    dialog = MultilayerDialog(q.service, q.owner, q.project.id, language="English")
    qtbot.addWidget(dialog)
    dialog.show()
    wait(qtbot, dialog)
    dialog.tabs.setCurrentIndex(2)
    dialog.source_layer.setCurrentIndex(dialog.source_layer.findData(str(q.layers[0].id)))
    dialog.target_layer.setCurrentIndex(dialog.target_layer.findData(str(q.layers[2].id)))
    dialog.tolerance.setText("1e-9")
    # Zero correspondences must produce an auditable insufficient result.
    click(qtbot, dialog, dialog.register_button)
    fit = dialog.snapshot["model"]["registrations"][-1]["fit"]
    assert fit["status"] == "insufficient"
    assert fit["affine"] is None
    assert "at_least_three" in dialog.fit_summary.text()
    revision = dialog.snapshot["model"]["revision"]
    dialog.tolerance.setText("not a measurement")
    qtbot.mouseClick(dialog.register_button, Qt.MouseButton.LeftButton)
    assert "Error:" in dialog.status.text()
    assert dialog.snapshot["model"]["revision"] == revision
    dialog.close()
