"""Validation and cell inspection of the same frames, one after the other, by the run button."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QMessageBox

from karakal.app.main_window import KarakalWidget
from karakal.core.analysis_modes import MODEL_OUTPUT_CONFIDENCE_MODE
from karakal.core.synthetic_masks import synthetic_frame
from karakal.ui.ui_constants import FOLDER_CONFIDENCE_ROLE


def _layer(tmp_path, name: str, *, confidence: bool = False, count: int = 4) -> tuple:
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    confidence_folder = tmp_path / f"{name}_confidence" if confidence else None
    if confidence_folder is not None:
        confidence_folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        mask = synthetic_frame({"bite": 1, "debris": 1}, seed=index).mask
        cv2.imwrite(str(folder / f"F_{index:04d}.png"), mask)
        if confidence_folder is not None:
            cv2.imwrite(str(confidence_folder / f"F_{index:04d}.png"), np.where(mask > 0, 230, 25).astype(np.uint8))
    return folder, confidence_folder


def _widget(tmp_path, qtbot, monkeypatch, *layers):
    shown: list[str] = []
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(
            QMessageBox,
            name,
            staticmethod(lambda *args, **_kwargs: shown.append(str(args[2]) if len(args) > 2 else "") or QMessageBox.StandardButton.Ok),
        )
    widget = KarakalWidget(settings=QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    presenter = widget._presenter
    for folder, confidence_folder in layers:
        item = presenter._append_folder_item(folder, checked=True)
        if confidence_folder is not None:
            item.setData(FOLDER_CONFIDENCE_ROLE, str(confidence_folder))
    presenter._refresh_folder_rows()
    presenter._sync_action_buttons()
    return widget, shown


def _press_run(widget, qtbot) -> None:
    presenter = widget._presenter
    presenter._on_run_button_clicked()
    # A run is a build and then its analysis: wait until both are over.
    for _ in range(3):
        qtbot.waitUntil(lambda: presenter._worker_thread is None, timeout=120000)
        qtbot.wait(300)


def test_confidence_audit_of_one_layer_scores_the_frames(tmp_path, qtbot, monkeypatch) -> None:
    """One layer with a confidence map: the audit keeps its mode (it used to fall back to the pair
    metrics of two models: «no scores for overall_polygon_score», a grey matrix)."""

    widget, shown = _widget(tmp_path, qtbot, monkeypatch, _layer(tmp_path, "layer_a", confidence=True))
    presenter = widget._presenter
    assert widget.analysis_mode_combo.currentData() == MODEL_OUTPUT_CONFIDENCE_MODE
    _press_run(widget, qtbot)
    state = presenter._current_tab_state()
    assert widget.analysis_mode_combo.currentData() == MODEL_OUTPUT_CONFIDENCE_MODE
    assert "overall_polygon_score" not in str(state.metric_key)
    assert all(record.score_ready for record in state.build_result.records)
    assert not shown


def test_validation_keeps_the_cell_check_and_back(tmp_path, qtbot, monkeypatch) -> None:
    widget, shown = _widget(tmp_path, qtbot, monkeypatch, _layer(tmp_path, "layer_a", confidence=True))
    presenter = widget._presenter
    panel = widget.analysis_setup_panel

    panel._mode_buttons["grid_inspection"].click()
    _press_run(widget, qtbot)
    state = presenter._current_tab_state()
    assert state.grid_inspection_results_ready
    cells = dict(state.grid_inspection_payload_by_key)
    assert cells

    panel._mode_buttons["validation"].click()
    assert presenter._run_action()[0] == presenter.RUN_ACTION_COMPUTE
    _press_run(widget, qtbot)
    assert all(record.score_ready for record in state.build_result.records)
    # Validation used to wipe the cell check (and the project then saved without it).
    assert state.grid_inspection_results_ready
    assert set(state.grid_inspection_payload_by_key) == set(cells)

    panel._mode_buttons["grid_inspection"].click()
    assert presenter._run_action()[0] == presenter.RUN_ACTION_RECOMPUTE
    panel._mode_buttons["validation"].click()
    assert presenter._run_action()[0] == presenter.RUN_ACTION_RECOMPUTE
    assert not shown
