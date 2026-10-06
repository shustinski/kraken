"""The matrix follows a larger debris size and switched-off types at once and says when it needs a new run."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QSettings

from karakal.app.main_window import KarakalWidget
from karakal.core.domain import BuildResult, FrameRecord, ModelSpec
from karakal.core.grid_anomaly import GRID_DAMAGE_ALGORITHM_VERSION
from karakal.core.image_io import _grayscale_array_to_qimage
from test_grid_narrowing import _frame


def _build_result(tmp_path) -> BuildResult:
    records = []
    for index in range(2):
        mask_path = tmp_path / f"mask_{index}.png"
        confidence_path = tmp_path / f"confidence_{index}.png"
        assert _grayscale_array_to_qimage(_frame()).save(str(mask_path))
        assert _grayscale_array_to_qimage(np.full((800, 1400), 250, dtype=np.uint8)).save(str(confidence_path))
        records.append(
            FrameRecord(
                f"frame-{index}",
                f"frame_{index:04d}.jpg",
                original_path=str(mask_path),
                model_mask_paths={"model": str(mask_path)},
                model_prob_paths={"model": str(confidence_path)},
            )
        )
    return BuildResult(records=tuple(records), model_specs=(ModelSpec("model", "Model", tmp_path),))


def _debris_count(result) -> int:
    return sum(1 for cell in result.per_cell_results if "small_artifact" in cell.reasons)


def test_matrix_recolors_without_a_new_run(tmp_path, qtbot) -> None:
    widget = KarakalWidget(settings=QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    presenter = widget._presenter
    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("grid_inspection"))
    presenter._on_build_finished(_build_result(tmp_path))
    state = presenter._current_tab_state()
    presenter._start_compute_grid_inspection(state)
    qtbot.waitUntil(
        lambda: state.grid_inspection_results_ready
        and state.grid_inspection_algorithm_version == GRID_DAMAGE_ALGORITHM_VERSION
        and len(state.grid_inspection_payload_by_key) == 2,
        timeout=120000,
    )
    qtbot.waitUntil(lambda: presenter._worker_thread is None, timeout=60000)
    view = widget.grid_inspection_matrix_views["unified"]
    run = state.grid_inspection_payload_by_key["frame-0"]
    status = widget.grid_inspection_status_label

    # Larger debris size: fewer debris marks, better score, no new run needed.
    widget.grid_debris_min_area_control.set_value(120)
    widget.grid_debris_min_area_control.valueChanged.emit(120)
    qtbot.waitUntil(lambda: state.grid_inspection_payload_by_key["frame-0"] is not run, timeout=5000)
    larger = state.grid_inspection_payload_by_key["frame-0"]
    assert _debris_count(larger) < _debris_count(run)
    assert larger.damage_score < run.damage_score
    assert view._grid_inspection_payload_by_key["frame-0"][1].damage_score == larger.damage_score
    assert not presenter._grid_matrix_needs_rerun(state)
    assert status.text() != presenter._t("grid_tuning.status_needs_rerun")

    # The details window shows the same narrowed result as the matrix.
    reused = presenter._reusable_grid_matrix_result(state, state.build_result.records[0])
    assert _debris_count(reused) == _debris_count(larger)

    # Debris switched off: no debris marks at all.
    widget.grid_error_type_checks["small_artifact"].setChecked(False)
    qtbot.waitUntil(lambda: _debris_count(state.grid_inspection_payload_by_key["frame-0"]) == 0, timeout=5000)
    assert not presenter._grid_matrix_needs_rerun(state)

    # Back on and smaller than the run: the matrix keeps the run and asks for a new analysis.
    widget.grid_error_type_checks["small_artifact"].setChecked(True)
    widget.grid_debris_min_area_control.set_value(10)
    widget.grid_debris_min_area_control.valueChanged.emit(10)
    qtbot.waitUntil(lambda: status.text() == presenter._t("grid_tuning.status_needs_rerun"), timeout=5000)
    assert _debris_count(state.grid_inspection_payload_by_key["frame-0"]) == _debris_count(run)
