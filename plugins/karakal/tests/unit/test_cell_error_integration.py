"""The grid-free analysis inside the app: a run builds the bank in its worker, the matrix and the
frame window show the new defects."""

from __future__ import annotations

import cv2
from PyQt6.QtCore import QSettings

from karakal.app.main_window import KarakalWidget
from karakal.core.domain import BuildResult, FrameRecord, ModelSpec
from karakal.core.grid_anomaly import GRID_DAMAGE_ALGORITHM_VERSION
from karakal.core.synthetic_masks import synthetic_frame


def _build_result(tmp_path) -> BuildResult:
    records = []
    for index in range(4):
        path = tmp_path / f"mask_{index}.png"
        cv2.imwrite(str(path), synthetic_frame({"bite": 2, "debris": 2, "split": 1}, seed=index).mask)
        records.append(FrameRecord(f"frame-{index}", path.name, model_mask_paths={"model": str(path)}))
    return BuildResult(records=tuple(records), model_specs=(ModelSpec("model", "Model", tmp_path),))


def test_run_builds_the_bank_in_the_worker_and_frame_window_uses_it(tmp_path, qtbot) -> None:
    settings = QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat)
    widget = KarakalWidget(settings=settings)
    qtbot.addWidget(widget)
    presenter = widget._presenter
    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("grid_inspection"))
    presenter._on_build_finished(_build_result(tmp_path))
    state = presenter._current_tab_state()
    presenter._start_compute_grid_inspection(state)
    qtbot.waitUntil(
        lambda: state.grid_inspection_results_ready
        and state.grid_inspection_algorithm_version == GRID_DAMAGE_ALGORITHM_VERSION
        and len(state.grid_inspection_payload_by_key) == 4,
        timeout=120000,
    )
    profile = state.grid_inspection_run_profiles.get("model")
    assert profile is not None and profile.normal_bank.ready
    reasons = {
        reason
        for result in state.grid_inspection_payload_by_key.values()
        for cell in result.per_cell_results
        for reason in cell.reasons
    }
    assert {"broken_geometry", "small_artifact", "split_cell"} <= reasons

    record = state.build_result.records[0]
    values = dict(presenter._grid_tuning_values())
    shown, cells = presenter._analyze_grid_record_for_details(record, state, values)
    assert any("broken_geometry" in cell.reasons for cell in cells)
    # The frame window judges by the run's bank, so a softer geometry slider shows no more defects.
    soft, soft_cells = presenter._analyze_grid_record_for_details(record, state, {**values, "geometry_sensitivity": 0})
    assert sum(bool(cell.reasons) for cell in soft_cells) <= sum(bool(cell.reasons) for cell in cells)
    # Let the run's queued UI updates (chunked histogram refresh) finish while the widget is alive,
    # so none of them fires into a later test.
    qtbot.waitUntil(lambda: presenter._worker_thread is None, timeout=30000)
    qtbot.wait(500)
