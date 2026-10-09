"""Ones/zeros pairs: a place painted by both networks is an error, measured against its cells."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from kraken_core.analysis_protocol import AnalysisProfileKind
from PyQt6.QtCore import QSettings

from karakal.core import grid_anomaly
from karakal.core.image_io import _grayscale_array_to_qimage
from karakal.core.mask_normalization import binarize_mask
from karakal.core.pair_conflict import (
    CONFLICT_AREA_FEATURE,
    CONFLICT_SHARE_FEATURE,
    PAIR_CONFLICT_LAYER,
    ConflictThresholds,
    analyze_class_conflict_frame_paths,
    analyze_class_conflict_masks,
    conflict_features,
    filter_class_conflicts,
    pair_key,
    parse_pair_key,
)

needs_cv2 = pytest.mark.skipif(grid_anomaly.cv2 is None, reason="OpenCV required for conflict components")


def _cells(shape=(64, 96), boxes=()) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for x, y, width, height in boxes:
        mask[y : y + height, x : x + width] = True
    return mask


# Ones cells in the left half, zero cells in the right half: 12x30 cells, 4 px gaps.
ONES = ((4, 10, 12, 30), (20, 10, 12, 30))
ZEROS = ((52, 10, 12, 30), (68, 10, 12, 30))


def test_pair_keys_round_trip() -> None:
    key = pair_key("night_a1", "night_b2")
    assert parse_pair_key(key) == ("night_a1", "night_b2")
    assert parse_pair_key("night_a1") is None
    assert parse_pair_key("pair::only") is None


@needs_cv2
def test_separate_ones_and_zeros_cells_have_no_conflict() -> None:
    result = analyze_class_conflict_masks(_cells(boxes=ONES), _cells(boxes=ZEROS), frame_id="clean")
    assert result.per_cell_results == ()
    assert result.damage_score == 0.0


@needs_cv2
def test_neighbours_touching_by_their_edges_are_not_a_conflict() -> None:
    ones = _cells(boxes=((10, 10, 12, 30),))
    zeros = _cells(boxes=((20, 10, 12, 30),))  # 2 px wide common strip along the edge
    result = filter_class_conflicts(
        analyze_class_conflict_masks(ones, zeros, frame_id="touch"), ConflictThresholds()
    )
    assert result.per_cell_results == ()


@needs_cv2
def test_one_cell_painted_by_both_networks_is_a_conflict() -> None:
    ones = _cells(boxes=(*ONES, ZEROS[0]))
    result = filter_class_conflicts(
        analyze_class_conflict_masks(ones, _cells(boxes=ZEROS), frame_id="both"), ConflictThresholds()
    )
    assert len(result.per_cell_results) == 1
    cell = result.per_cell_results[0]
    assert cell.reasons == ("class_conflict",)
    assert cell.status == "broken"
    features = conflict_features(cell)
    assert features[CONFLICT_AREA_FEATURE] == 12 * 30
    assert features[CONFLICT_SHARE_FEATURE] == pytest.approx(1.0)
    assert cell.bbox == ZEROS[0]
    assert len(cell.outline) >= 4
    assert result.damage_score > 0.0


@needs_cv2
def test_thresholds_hide_small_overlaps_without_a_new_run() -> None:
    zeros = _cells(boxes=((40, 10, 20, 40),))  # 800 px zero cell
    ones = _cells(boxes=((57, 10, 20, 40),))  # an equal ones cell overlapping it by 3 x 40 px: 15 %
    raw = analyze_class_conflict_masks(ones, zeros, frame_id="small")
    assert len(raw.per_cell_results) == 1
    assert conflict_features(raw.per_cell_results[0])[CONFLICT_SHARE_FEATURE] == pytest.approx(120 / 800)
    assert filter_class_conflicts(raw, ConflictThresholds(min_cell_share=0.20)).per_cell_results == ()
    assert len(filter_class_conflicts(raw, ConflictThresholds(min_cell_share=0.10)).per_cell_results) == 1
    assert filter_class_conflicts(raw, ConflictThresholds(min_area_px=200)).per_cell_results == ()


@needs_cv2
def test_a_small_speck_of_the_other_class_inside_a_cell_counts_by_its_own_area() -> None:
    """The share is taken of the smaller of the two cells: a speck covers itself fully."""

    zeros = _cells(boxes=((40, 10, 20, 40),))
    speck = _cells(boxes=((44, 20, 2, 2),))
    blob = _cells(boxes=((44, 20, 8, 8),))
    assert filter_class_conflicts(analyze_class_conflict_masks(speck, zeros), ConflictThresholds()).per_cell_results == ()
    (cell,) = filter_class_conflicts(analyze_class_conflict_masks(blob, zeros), ConflictThresholds()).per_cell_results
    assert conflict_features(cell)[CONFLICT_SHARE_FEATURE] == pytest.approx(1.0)


@needs_cv2
def test_conflict_masks_are_binarised_like_the_cell_analysis(tmp_path) -> None:
    """JPEG-like soft edges: the conflict sees the same cells as the layer analysis."""

    ones = np.zeros((64, 96), dtype=np.uint8)
    zeros = np.zeros((64, 96), dtype=np.uint8)
    for x, y, width, height in (*ONES, ZEROS[0]):
        ones[y : y + height, x : x + width] = 230
        ones[y - 1 : y + height + 1, x - 1 : x + width + 1][ones[y - 1 : y + height + 1, x - 1 : x + width + 1] == 0] = 90
    for x, y, width, height in ZEROS:
        zeros[y : y + height, x : x + width] = 220
    ones_path, zeros_path = tmp_path / "ones.png", tmp_path / "zeros.png"
    assert _grayscale_array_to_qimage(ones).save(str(ones_path))
    assert _grayscale_array_to_qimage(zeros).save(str(zeros_path))
    result = analyze_class_conflict_frame_paths(ones_path, zeros_path, frame_id="soft", use_cache=False)
    expected = analyze_class_conflict_masks(binarize_mask(ones), binarize_mask(zeros), frame_id="soft")
    assert [cell.bbox for cell in result.per_cell_results] == [cell.bbox for cell in expected.per_cell_results]
    assert len(filter_class_conflicts(result, ConflictThresholds()).per_cell_results) == 1


# --- worker ---------------------------------------------------------------------


def _save(folder: Path, name: str, mask: np.ndarray) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    assert _grayscale_array_to_qimage(np.where(mask, 255, 0).astype(np.uint8)).save(str(path))
    return path


@needs_cv2
def test_worker_computes_a_pair_after_the_layers_or_alone(tmp_path, qtbot) -> None:
    from karakal.core.domain import FrameRecord
    from karakal.core.workers import MultiModelGridInspectionWorker

    ones_a = _save(tmp_path / "ones", "F_0001.png", _cells(boxes=(*ONES, ZEROS[0])))
    ones_b = _save(tmp_path / "ones", "F_0002.png", _cells(boxes=ONES))
    zeros_a = _save(tmp_path / "zeros", "F_0001.png", _cells(boxes=ZEROS))
    zeros_b = _save(tmp_path / "zeros", "F_0002.png", _cells(boxes=ZEROS))
    records = (
        FrameRecord("F_0001.png", "F_0001.png", model_mask_paths={"o": str(ones_a), "z": str(zeros_a)}),
        FrameRecord("F_0002.png", "F_0002.png", model_mask_paths={"o": str(ones_b), "z": str(zeros_b)}),
        FrameRecord("F_0003.png", "F_0003.png", model_mask_paths={"o": str(ones_b)}),
    )
    key = pair_key("o", "z")
    worker = MultiModelGridInspectionWorker(records, (), conflict_pairs=((key, "o", "z"),), use_cache=False)
    batches: list[dict] = []
    finished: list[dict] = []
    worker.multiModelResultsReady.connect(batches.append)
    worker.finished.connect(finished.append)
    worker.run()
    assert finished, "the pair-only job finishes"
    stored = finished[0][key]
    # Only frames where both networks have a mask take part.
    assert set(stored) == {"F_0001.png", "F_0002.png"}
    assert stored["F_0001.png"][PAIR_CONFLICT_LAYER].per_cell_results
    assert not stored["F_0002.png"][PAIR_CONFLICT_LAYER].per_cell_results
    assert any(key in batch for batch in batches)


# --- in the program -------------------------------------------------------------


def _project(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "src"
    for name in ("F_0001.png", "F_0002.png"):
        _save(source, name, np.zeros((64, 96), dtype=bool))
    _save(tmp_path / "ones", "F_0001.png", _cells(boxes=(*ONES, ZEROS[0])))
    _save(tmp_path / "ones", "F_0002.png", _cells(boxes=ONES))
    _save(tmp_path / "zeros", "F_0001.png", _cells(boxes=ZEROS))
    _save(tmp_path / "zeros", "F_0002.png", _cells(boxes=ZEROS))
    return tmp_path / "ones", tmp_path / "zeros", source


def _widget(tmp_path: Path, qtbot):
    from karakal.app.main_window import KarakalWidget

    widget = KarakalWidget(settings=QSettings(str(tmp_path / "k.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    return widget


def test_cell_class_of_a_layer_is_saved_and_makes_a_pair(tmp_path, qtbot) -> None:
    from karakal.ui.ui_constants import FOLDER_CELL_CLASS_ROLE, FOLDER_ORIGINAL_ROLE

    ones, zeros, source = _project(tmp_path)
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter
    presenter._on_analysis_profile_changed(AnalysisProfileKind.GRID_DEFECTS.value)
    items = [presenter._append_folder_item(folder, checked=True) for folder in (ones, zeros)]
    for item in items:
        item.setData(FOLDER_ORIGINAL_ROLE, str(source))
    presenter._refresh_folder_rows()
    assert presenter._grid_conflict_pairs() == ()

    presenter._set_folder_item_cell_class(items[0], "ones")
    presenter._set_folder_item_cell_class(items[1], "zeros")
    (pair,) = presenter._grid_conflict_pairs()
    assert (pair.ones_id, pair.zeros_id) == tuple(presenter._model_id_for_folder_item(item) for item in items)
    assert widget.grid_matrix_layer_combo.findData(pair.key) >= 0
    assert pair.key in presenter._grid_pair_buttons
    assert widget.pair_matrix_group.isVisibleTo(widget) or not widget.isVisible()
    row = widget.folder_list.itemWidget(items[0])
    assert row.cell_class_badge.text() == "1"
    assert row.cell_class_combo.currentData() == "ones"
    # Choosing the class in the row stores it.
    widget.folder_list.itemWidget(items[1]).cell_class_combo.setCurrentIndex(0)
    assert presenter._grid_conflict_pairs() == ()
    widget.folder_list.itemWidget(items[1]).cell_class_combo.setCurrentIndex(2)
    assert len(presenter._grid_conflict_pairs()) == 1

    payload = presenter._build_folder_manager_payload()
    assert [entry["cell_class"] for entry in payload["folders"]] == ["ones", "zeros"]

    # Different source photos: different projects, no pair.
    other = tmp_path / "other_src"
    other.mkdir()
    items[1].setData(FOLDER_ORIGINAL_ROLE, str(other))
    assert presenter._grid_conflict_pairs() == ()
    assert items[1].data(FOLDER_CELL_CLASS_ROLE) == "zeros"


@needs_cv2
def test_a_pair_is_computed_with_the_layers_shown_in_the_matrix_and_marked_on_its_layers(tmp_path, qtbot) -> None:
    from karakal.core.analytics import collect_frame_records
    from karakal.core.domain import BuildOptions
    from karakal.ui.ui_constants import FOLDER_ORIGINAL_ROLE

    ones, zeros, source = _project(tmp_path)
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter
    presenter._on_analysis_profile_changed(AnalysisProfileKind.GRID_DEFECTS.value)
    items = [presenter._append_folder_item(folder, checked=True) for folder in (ones, zeros)]
    for item in items:
        item.setData(FOLDER_ORIGINAL_ROLE, str(source))
    presenter._refresh_folder_rows()
    presenter._set_folder_item_cell_class(items[0], "ones")
    presenter._set_folder_item_cell_class(items[1], "zeros")
    (pair,) = presenter._grid_conflict_pairs()

    presenter._on_build_finished(collect_frame_records(presenter._checked_model_specs(), BuildOptions(union_layer_frames=True)))
    qtbot.waitUntil(lambda: presenter._worker_thread is None, timeout=60000)
    state = presenter._current_tab_state()
    if not state.grid_inspection_payloads_by_pair:
        presenter._start_compute_grid_inspection(state)
        qtbot.waitUntil(lambda: presenter._worker_thread is None, timeout=60000)
    assert set(state.grid_inspection_payloads_by_pair[pair.key]) == {"F_0001.png", "F_0002.png"}

    # The ones layer is shown: its own result plus the conflict, named after the zeros network.
    widget.folder_list.setCurrentItem(items[0])
    assert presenter._grid_inspection_matrix_selection(state) == pair.ones_id
    entries = [entry for entry in presenter._iter_grid_inspection_error_payloads(state) if entry["error_type"] == "class_conflict"]
    assert [entry["record_key"] for entry in entries] == ["F_0001.png"]
    assert entries[0]["feature_cluster_label"] == presenter._t("pairs.conflict_with", name=pair.zeros_name)

    # The pair is shown: its frames are coloured by conflicts; no layer row keeps the frame.
    presenter._select_grid_pair(pair.key)
    assert presenter._grid_inspection_matrix_selection(state) == pair.key
    assert widget.folder_list.currentItem() is None
    assert presenter._grid_pair_buttons[pair.key].property("selected") is True
    assert presenter._grid_pair_conflict_frame_count(state, pair) == 1
    shown = state.grid_inspection_payload_by_key
    assert shown["F_0001.png"].per_cell_results and not shown["F_0002.png"].per_cell_results
    assert shown["F_0001.png"].damage_score > shown["F_0002.png"].damage_score
    record = next(record for record in state.build_result.records if record.key == "F_0001.png")
    assert presenter._reusable_grid_matrix_result(state, record) is shown["F_0001.png"]

    # Thresholds are applied on display: no new run.
    widget.grid_conflict_min_area_spin.setValue(2000)
    assert presenter._worker_thread is None
    assert not state.grid_inspection_payload_by_key["F_0001.png"].per_cell_results
    assert presenter._grid_pair_conflict_frame_count(state, pair) == 0
    widget.grid_conflict_min_area_spin.setValue(6)
    assert state.grid_inspection_payload_by_key["F_0001.png"].per_cell_results

    # Back to a layer by its row.
    widget.folder_list.setCurrentItem(items[1])
    assert presenter._grid_inspection_matrix_selection(state) == pair.zeros_id
    assert presenter._grid_pair_buttons[pair.key].property("selected") is False


@needs_cv2
def test_cells_marked_after_the_run_get_a_pass_of_their_own(tmp_path, qtbot) -> None:
    from karakal.core.analytics import collect_frame_records
    from karakal.core.domain import BuildOptions
    from karakal.ui.ui_constants import FOLDER_ORIGINAL_ROLE

    ones, zeros, source = _project(tmp_path)
    widget = _widget(tmp_path, qtbot)
    presenter = widget._presenter
    presenter._on_analysis_profile_changed(AnalysisProfileKind.GRID_DEFECTS.value)
    items = [presenter._append_folder_item(folder, checked=True) for folder in (ones, zeros)]
    for item in items:
        item.setData(FOLDER_ORIGINAL_ROLE, str(source))
    presenter._refresh_folder_rows()
    presenter._on_build_finished(collect_frame_records(presenter._checked_model_specs(), BuildOptions(union_layer_frames=True)))
    qtbot.waitUntil(lambda: presenter._worker_thread is None, timeout=60000)
    state = presenter._current_tab_state()
    if not state.grid_inspection_payloads_by_model:
        presenter._start_compute_grid_inspection(state)
        qtbot.waitUntil(lambda: presenter._worker_thread is None, timeout=60000)
    layers_before = {model: dict(layers) for model, layers in state.grid_inspection_payloads_by_model.items()}
    assert not state.grid_inspection_payloads_by_pair

    presenter._set_folder_item_cell_class(items[0], "ones")
    presenter._set_folder_item_cell_class(items[1], "zeros")
    (pair,) = presenter._grid_conflict_pairs()
    presenter._select_grid_pair(pair.key)
    qtbot.waitUntil(lambda: presenter._worker_thread is None, timeout=60000)
    assert state.grid_inspection_payloads_by_model == layers_before, "the layers are not analysed again"
    assert presenter._grid_pair_conflict_frame_count(state, pair) == 1
    assert state.grid_inspection_payload_by_key["F_0001.png"].per_cell_results
