"""Grid matrix calculation stores a result the details window can reuse."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QSettings

from karakal.app.main_window import KarakalWidget
from karakal.core.domain import BuildResult, FrameRecord, ModelSpec
from karakal.core.grid_anomaly import GRID_DAMAGE_ALGORITHM_VERSION, GridFrameAnalysisResult
from karakal.core.image_io import _grayscale_array_to_qimage


def _lattice(seed: int) -> np.ndarray:
    image = np.zeros((96, 96), dtype=np.uint8)
    for row in range(3):
        for col in range(3):
            if seed == 1 and row == 1 and col == 1:
                continue
            y = 16 + row * 28
            x = 16 + col * 28
            image[y : y + 10, x : x + 8] = 255
    if seed == 2:
        image[4:8, 4:7] = 255
    return image


def _build_result(tmp_path) -> BuildResult:
    records = []
    for index in range(3):
        mask_path = tmp_path / f"mask_{index}.png"
        confidence_path = tmp_path / f"confidence_{index}.png"
        assert _grayscale_array_to_qimage(_lattice(index)).save(str(mask_path))
        assert _grayscale_array_to_qimage(np.full((96, 96), 250, dtype=np.uint8)).save(str(confidence_path))
        records.append(
            FrameRecord(
                f"frame-{index}",
                f"frame_{index:04d}.jpg",
                original_path=str(mask_path),
                model_mask_paths={"model": str(mask_path)},
                model_prob_paths={"model": str(confidence_path)},
            )
        )
    return BuildResult(
        records=tuple(records),
        model_specs=(ModelSpec("model", "Model", tmp_path),),
    )


def test_grid_matrix_run_finishes_and_details_reuse_the_result(tmp_path, qtbot) -> None:
    settings = QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat)
    widget = KarakalWidget(settings=settings)
    qtbot.addWidget(widget)
    presenter = widget._presenter
    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("grid_inspection"))
    presenter._on_build_finished(_build_result(tmp_path))
    state = presenter._current_tab_state()
    assert state is not None

    def _refuse_recompute(*_args, **_kwargs):
        raise AssertionError("details recomputed a frame that the matrix already analyzed")

    presenter._start_grid_details_analyze = _refuse_recompute
    presenter._start_compute_grid_inspection(state)
    qtbot.waitUntil(
        lambda: (
            state.grid_inspection_results_ready
            and state.grid_inspection_algorithm_version == GRID_DAMAGE_ALGORITHM_VERSION
            and len(state.grid_inspection_payload_by_key) == 3
        ),
        timeout=60000,
    )

    record = state.build_result.records[1]
    stored = state.grid_inspection_payload_by_key[record.key]
    assert isinstance(stored, GridFrameAnalysisResult)
    presenter._open_record_details(record, state)
    dialog = presenter._details_dialogs[-1]
    qtbot.addWidget(dialog)
    assert dialog._grid_inspection_result is stored
    assert not dialog.grid_preview_status.text()
