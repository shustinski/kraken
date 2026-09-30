"""Multi-model grid inspection: storage, switcher, no-recompute paths.

Covers the pieces added by ``agent_prompt_multi_model.md``:

- Stable model ids come from the folder path, not the display label.
- Clicking a folder while in grid inspection mode never starts a compute.
- Flipping the matrix layer selector never spawns a background worker.
- «Все слои» merges per-model results and normalizes on total cell count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QSettings, Qt

from karakal.app.main_window import KarakalWidget
from karakal.core.domain import BuildOptions, BuildResult, FrameRecord, ModelSpec
from karakal.core.grid_anomaly import GridCellAnalysisResult, GridFrameAnalysisResult
from karakal.core.image_io import _grayscale_array_to_qimage
from karakal.ui.ui_constants import FOLDER_CHECKED_ROLE, FOLDER_LABEL_ROLE


def _lattice() -> np.ndarray:
    image = np.zeros((96, 96), dtype=np.uint8)
    for row in range(3):
        for col in range(3):
            y = 12 + row * 28
            x = 12 + col * 28
            image[y : y + 10, x : x + 8] = 255
    return image


def _build_folder(tmp_path: Path, name: str) -> Path:
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    mask_path = folder / "frame_0001.png"
    assert _grayscale_array_to_qimage(_lattice()).save(str(mask_path))
    return folder


def _widget(tmp_path: Path, qtbot) -> KarakalWidget:
    settings = QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat)
    widget = KarakalWidget(settings=settings)
    qtbot.addWidget(widget)
    return widget


def _add_checked_folder(widget: KarakalWidget, folder: Path, *, label: str | None = None) -> None:
    item = widget._presenter._append_folder_item(folder, checked=True)
    if label is not None:
        item.setData(FOLDER_LABEL_ROLE, label)
    item.setData(FOLDER_CHECKED_ROLE, True)


# --- Stable model ids -------------------------------------------------------


def test_same_display_labels_produce_distinct_stable_ids(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter
    folder_a = _build_folder(tmp_path, "cells_ones")
    folder_b = _build_folder(tmp_path, "cells_zeros")

    _add_checked_folder(widget, folder_a, label="Test")
    _add_checked_folder(widget, folder_b, label="Test")

    specs = presenter._checked_model_specs()
    assert len(specs) == 2
    assert specs[0].display_name == "Test"
    assert specs[1].display_name == "Test"
    assert specs[0].model_id != specs[1].model_id
    assert presenter._stable_model_id_for_folders(folder_a) == specs[0].model_id
    assert presenter._stable_model_id_for_folders(folder_b) == specs[1].model_id


def test_renaming_label_keeps_the_same_stable_id(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter
    folder = _build_folder(tmp_path, "cells")
    _add_checked_folder(widget, folder, label="One")
    id_before = presenter._checked_model_specs()[0].model_id
    presenter.folder_list.item(0).setData(FOLDER_LABEL_ROLE, "Renamed")
    id_after = presenter._checked_model_specs()[0].model_id
    assert id_before == id_after


# --- Folder click never triggers a compute ---------------------------------


def test_folder_click_in_grid_inspection_does_not_start_compute(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter

    folder_a = _build_folder(tmp_path, "cells_ones")
    folder_b = _build_folder(tmp_path, "cells_zeros")
    _add_checked_folder(widget, folder_a, label="Test по единицам")
    _add_checked_folder(widget, folder_b, label="Test по нулям")

    specs = presenter._checked_model_specs()
    records = (
        FrameRecord(
            "frame-0",
            "frame_0001.png",
            model_mask_paths={
                str(specs[0].model_id): str(folder_a / "frame_0001.png"),
                str(specs[1].model_id): str(folder_b / "frame_0001.png"),
            },
        ),
    )
    result = BuildResult(records=records, options=BuildOptions(), model_specs=specs)
    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("grid_inspection"))
    presenter._on_build_finished(result)

    starts: list[str] = []
    original = presenter._start_compute_grid_inspection

    def _tracker(*args, **kwargs):  # pragma: no cover - assertion below is the guard
        starts.append("called")
        original(*args, **kwargs)

    presenter._start_compute_grid_inspection = _tracker  # type: ignore[assignment]

    folder_item = presenter.folder_list.item(1)
    presenter._on_folder_item_clicked(folder_item)

    assert starts == [], "folder click must not recompute grid inspection results"
    assert presenter._worker_thread is None


# --- Layer selector switch is instant and never spawns a worker -------------


def test_matrix_layer_switch_does_not_start_worker(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter

    folder_a = _build_folder(tmp_path, "cells_ones")
    folder_b = _build_folder(tmp_path, "cells_zeros")
    _add_checked_folder(widget, folder_a, label="Test по единицам")
    _add_checked_folder(widget, folder_b, label="Test по нулям")

    specs = presenter._checked_model_specs()
    record = FrameRecord(
        "frame-0",
        "frame_0001.png",
        model_mask_paths={
            str(specs[0].model_id): str(folder_a / "frame_0001.png"),
            str(specs[1].model_id): str(folder_b / "frame_0001.png"),
        },
    )
    result = BuildResult(records=(record,), options=BuildOptions(), model_specs=specs)
    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("grid_inspection"))
    presenter._on_build_finished(result)

    state = presenter._current_tab_state()
    assert state is not None

    def _make_result(model_id: str, defects: int) -> GridFrameAnalysisResult:
        cells = tuple(
            GridCellAnalysisResult(
                0, index, (index * 10, 0, 8, 8), (index * 10 + 4.0, 4.0), None,
                "broken", 0.9, ("filled_cell",),
            )
            for index in range(defects)
        )
        return GridFrameAnalysisResult(
            frame_id="frame-0",
            frame_path=f"{model_id}.png",
            image_width=96,
            image_height=96,
            grid_rows=1,
            grid_cols=9,
            total_expected_cells=9,
            detected_cells=9,
            normal_cells=9 - defects,
            suspicious_cells=0,
            broken_cells=defects,
            missing_cells=0,
            artifact_cells=0,
            damage_score=float(defects) / 9.0,
            severity_level="",
            grid_detected=True,
            per_cell_results=cells,
        )

    state.grid_inspection_payloads_by_model = {
        str(specs[0].model_id): {
            "binary": {"frame-0": _make_result(specs[0].model_id, 2)},
        },
        str(specs[1].model_id): {
            "binary": {"frame-0": _make_result(specs[1].model_id, 4)},
        },
    }
    state.grid_inspection_results_ready = True

    presenter._sync_grid_inspection_matrix_selector(state)
    combo = widget.grid_matrix_layer_combo
    assert combo.count() >= 3  # two models + all layers
    all_layers_index = combo.findData(presenter.ALL_LAYERS_SELECTION_KEY)
    assert all_layers_index >= 0

    started_workers: list[str] = []

    def _reject_start(*_args, **_kwargs):
        started_workers.append("called")

    presenter._start_compute_grid_inspection = _reject_start  # type: ignore[assignment]

    combo.setCurrentIndex(all_layers_index)
    assert started_workers == []
    assert presenter._worker_thread is None

    # Then jump to a specific model — still no compute.
    first_model_index = combo.findData(str(specs[0].model_id))
    assert first_model_index >= 0
    combo.setCurrentIndex(first_model_index)
    assert started_workers == []
    assert presenter._worker_thread is None


# --- All-layers merge score basics -----------------------------------------


def test_all_layers_merge_normalizes_by_total_cell_count(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter
    folder_a = _build_folder(tmp_path, "cells_ones")
    folder_b = _build_folder(tmp_path, "cells_zeros")
    _add_checked_folder(widget, folder_a, label="Ones")
    _add_checked_folder(widget, folder_b, label="Zeros")
    specs = presenter._checked_model_specs()

    def _defective(count: int, offset: int, tag: str) -> GridFrameAnalysisResult:
        cells = tuple(
            GridCellAnalysisResult(
                0,
                index,
                (offset + index * 12, 0, 8, 8),
                (float(offset + index * 12 + 4), 4.0),
                None,
                "broken",
                0.9,
                ("filled_cell",),
            )
            for index in range(count)
        )
        return GridFrameAnalysisResult(
            frame_id="frame-A",
            frame_path=f"{tag}.png",
            image_width=96,
            image_height=96,
            grid_rows=1,
            grid_cols=10,
            total_expected_cells=10,
            detected_cells=10,
            normal_cells=10 - count,
            suspicious_cells=0,
            broken_cells=count,
            missing_cells=0,
            artifact_cells=0,
            damage_score=float(count) / 10.0,
            severity_level="",
            grid_detected=True,
            per_cell_results=cells,
        )

    ones_result = _defective(2, 0, "ones")
    zeros_result = _defective(4, 200, "zeros")
    # Minimal stand-in for tab state — no matrix materialization.
    state = type("State", (), {})()
    state.build_result = BuildResult(
        records=(
            FrameRecord(
                "frame-A",
                "frame_A.png",
                model_mask_paths={
                    str(specs[0].model_id): str(folder_a / "frame_0001.png"),
                    str(specs[1].model_id): str(folder_b / "frame_0001.png"),
                },
            ),
        ),
        options=BuildOptions(),
        model_specs=specs,
    )
    state.grid_inspection_payloads_by_model = {
        str(specs[0].model_id): {"binary": {"frame-A": ones_result}},
        str(specs[1].model_id): {"binary": {"frame-A": zeros_result}},
    }

    merged = presenter._merge_all_layers_frame_result(state, "frame-A", layer="binary")
    assert merged is not None
    assert merged.total_expected_cells == 20
    assert merged.broken_cells == 6
    assert abs(merged.damage_score - 0.3) < 1e-6
    assert merged.damage_score <= max(ones_result.damage_score, zeros_result.damage_score) + 1e-6
    assert merged.damage_score >= min(ones_result.damage_score, zeros_result.damage_score) - 1e-6
    display_names = {specs[0].display_name, specs[1].display_name}
    labels = {str(cell.feature_cluster_label) for cell in merged.per_cell_results}
    assert display_names.issubset(labels)


def test_all_layers_view_projects_merged_payloads_onto_layer_state(tmp_path, qtbot) -> None:
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter
    folder_a = _build_folder(tmp_path, "cells_ones")
    folder_b = _build_folder(tmp_path, "cells_zeros")
    _add_checked_folder(widget, folder_a, label="Ones")
    _add_checked_folder(widget, folder_b, label="Zeros")
    specs = presenter._checked_model_specs()

    def _res(defects: int) -> GridFrameAnalysisResult:
        cell = GridCellAnalysisResult(
            0, 0, (0, 0, 8, 8), (4.0, 4.0), None, "broken", 0.9, ("filled_cell",)
        )
        return GridFrameAnalysisResult(
            frame_id="frame-A",
            frame_path="p.png",
            image_width=96,
            image_height=96,
            grid_rows=1,
            grid_cols=8,
            total_expected_cells=8,
            detected_cells=8,
            normal_cells=8 - defects,
            suspicious_cells=0,
            broken_cells=defects,
            missing_cells=0,
            artifact_cells=0,
            damage_score=float(defects) / 8.0,
            severity_level="",
            grid_detected=True,
            per_cell_results=(cell,) * max(1, defects),
        )

    state = type("State", (), {})()
    state.build_result = BuildResult(
        records=(
            FrameRecord(
                "frame-A",
                "frame_A.png",
                model_mask_paths={
                    str(specs[0].model_id): str(folder_a / "frame_0001.png"),
                    str(specs[1].model_id): str(folder_b / "frame_0001.png"),
                },
            ),
        ),
        options=BuildOptions(),
        model_specs=specs,
    )
    state.grid_inspection_payloads_by_model = {
        str(specs[0].model_id): {"binary": {"frame-A": _res(1)}},
        str(specs[1].model_id): {"binary": {"frame-A": _res(3)}},
    }
    state.grid_inspection_matrix_selection = presenter.ALL_LAYERS_SELECTION_KEY
    layers = {"binary": presenter._build_all_layers_payloads(state, layer="binary"), "confidence": {}}
    state.grid_inspection_payloads_by_layer = layers
    state.grid_inspection_payload_by_key = dict(layers.get("binary") or {})
    state.grid_inspection_results_ready = True

    binary_layer = state.grid_inspection_payloads_by_layer.get("binary") or {}
    assert "frame-A" in binary_layer
    merged = binary_layer["frame-A"]
    assert isinstance(merged, GridFrameAnalysisResult)
    assert merged.total_expected_cells == 16
    assert merged.broken_cells == 4


def test_streamed_multi_model_batches_update_only_their_frames(tmp_path, qtbot) -> None:
    # Re-projecting every finished frame per batch made 20k-frame runs quadratic and froze the UI.
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter
    folder_a = _build_folder(tmp_path, "cells_ones")
    folder_b = _build_folder(tmp_path, "cells_zeros")
    _add_checked_folder(widget, folder_a, label="Ones")
    _add_checked_folder(widget, folder_b, label="Zeros")
    specs = presenter._checked_model_specs()
    model_a, model_b = str(specs[0].model_id), str(specs[1].model_id)
    keys = ("frame-0", "frame-1", "frame-2")
    records = tuple(
        FrameRecord(
            key,
            "frame_0001.png",
            model_mask_paths={
                model_a: str(folder_a / "frame_0001.png"),
                model_b: str(folder_b / "frame_0001.png"),
            },
        )
        for key in keys
    )
    widget.app_mode_combo.setCurrentIndex(widget.app_mode_combo.findData("grid_inspection"))
    presenter._on_build_finished(BuildResult(records=records, options=BuildOptions(), model_specs=specs))
    state = presenter._current_tab_state()
    assert state is not None

    def _res(key: str, defects: int) -> GridFrameAnalysisResult:
        cells = tuple(
            GridCellAnalysisResult(0, index, (index * 10, 0, 8, 8), (index * 10 + 4.0, 4.0), None, "broken", 0.9, ("filled_cell",))
            for index in range(defects)
        )
        return GridFrameAnalysisResult(
            frame_id=key, frame_path=f"{key}.png", image_width=96, image_height=96, grid_rows=1, grid_cols=9,
            total_expected_cells=9, detected_cells=9, normal_cells=9 - defects, suspicious_cells=0,
            broken_cells=defects, missing_cells=0, artifact_cells=0, damage_score=defects / 9.0,
            severity_level="", grid_detected=True, per_cell_results=cells,
        )

    state.grid_inspection_payloads_by_model = {}
    state.grid_inspection_payloads_by_layer = {key: {} for key in presenter._grid_inspection_layer_keys()}
    state.grid_inspection_matrix_selection = presenter.ALL_LAYERS_SELECTION_KEY
    presenter._active_compute_state = state

    full_rebuilds: list[str] = []
    for name, view in presenter._grid_inspection_views().items():
        original = view.set_grid_inspection_payloads

        def _tracked(*args, _original=original, _name=name, **kwargs):
            full_rebuilds.append(_name)
            return _original(*args, **kwargs)

        view.set_grid_inspection_payloads = _tracked

    for index, key in enumerate(keys):
        presenter._on_grid_inspection_multi_model_batch(
            {model_a: {key: {"binary": _res(key, 1 + index)}}, model_b: {key: {"binary": _res(key, 2)}}}
        )
        if index == 0:
            assert full_rebuilds, "the first batch sets up the views"
            full_rebuilds.clear()
    assert full_rebuilds == []

    streamed = {layer: dict(payloads) for layer, payloads in state.grid_inspection_payloads_by_layer.items()}
    presenter._apply_grid_inspection_active_layer_view(state, streaming=False)
    assert set(streamed["binary"]) == set(keys)
    for layer, payloads in state.grid_inspection_payloads_by_layer.items():
        assert streamed.get(layer, {}) == payloads, layer
