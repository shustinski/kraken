"""Removed grid filters stay gone, and 1∩0 follows the operation that produced it."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from karakal.app.presenter import KarakalPresenter
from karakal.core import grid_anomaly
from karakal.core.grid_anomaly import GridCellAnalysisResult, GridFrameAnalysisResult
from karakal.ui.ui_constants import (
    GRID_INSPECTION_ERROR_TYPE_OPTIONS,
    GRID_INSPECTION_TUNING_KEYS,
    apply_class_conflict_checkbox,
)

EXPECTED_ERROR_TYPES = (
    "small_artifact",
    "edge_clipped_cell",
    "broken_geometry",
    "merged_contour",
    "class_conflict",
    "conductor_zone",
)


class _Box:
    def __init__(self, *, checked: bool = True, enabled: bool = True) -> None:
        self.checked = checked
        self.enabled = enabled
        self.tooltip = ""
        self._blocked = False

    def blockSignals(self, value: bool) -> bool:
        previous = self._blocked
        self._blocked = bool(value)
        return previous

    def setEnabled(self, value: bool) -> None:
        self.enabled = bool(value)

    def isEnabled(self) -> bool:
        return self.enabled

    def setChecked(self, value: bool) -> None:
        self.checked = bool(value)

    def isChecked(self) -> bool:
        return self.checked

    def setToolTip(self, text: str) -> None:
        self.tooltip = str(text)


def _host(box: _Box, *, wants: bool) -> SimpleNamespace:
    checks = {error: _Box(checked=True) for error in EXPECTED_ERROR_TYPES}
    checks["class_conflict"] = box
    return SimpleNamespace(grid_error_type_checks=checks, _class_conflict_user_wants=wants)


def replace_frame(result: GridFrameAnalysisResult, cells: tuple[GridCellAnalysisResult, ...]) -> GridFrameAnalysisResult:
    return GridFrameAnalysisResult(
        frame_id=result.frame_id,
        frame_path=result.frame_path,
        image_width=result.image_width,
        image_height=result.image_height,
        grid_rows=result.grid_rows,
        grid_cols=result.grid_cols,
        total_expected_cells=result.total_expected_cells,
        detected_cells=len(cells),
        normal_cells=0,
        suspicious_cells=len(cells),
        broken_cells=len(cells),
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.8,
        severity_level="high",
        grid_detected=True,
        per_cell_results=cells,
    )


def _binary() -> GridFrameAnalysisResult:
    return GridFrameAnalysisResult(
        frame_id="overlap",
        frame_path="",
        image_width=64,
        image_height=64,
        grid_rows=1,
        grid_cols=1,
        total_expected_cells=1,
        detected_cells=0,
        normal_cells=1,
        suspicious_cells=0,
        broken_cells=0,
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.0,
        severity_level="OK",
        grid_detected=True,
    )


def test_checkbox_list_matches_remaining_reasons_in_order() -> None:
    assert tuple(error for _label, error in GRID_INSPECTION_ERROR_TYPE_OPTIONS) == EXPECTED_ERROR_TYPES
    assert GRID_INSPECTION_TUNING_KEYS[0] == "disagreement_sensitivity"
    assert "mismatch_sensitivity" not in GRID_INSPECTION_TUNING_KEYS


def test_class_conflict_checkbox_follows_the_operation() -> None:
    box = _Box(checked=True, enabled=True)
    apply_class_conflict_checkbox(box, available=False, user_wants=True, tooltip="not run")
    assert (box.enabled, box.checked, box.tooltip) == (False, False, "not run")
    assert "class_conflict" in KarakalPresenter._selected_grid_error_types(_host(box, wants=True))

    apply_class_conflict_checkbox(box, available=True, user_wants=True, tooltip="not run")
    assert box.enabled is True
    assert box.checked is True
    assert box.tooltip == ""

    state = SimpleNamespace(grid_inspection_payloads_by_layer={"derived_conflict": {"frame": object()}})
    assert KarakalPresenter._class_conflict_ready(SimpleNamespace(), state, "frame") is True
    assert KarakalPresenter._class_conflict_ready(SimpleNamespace(), state, "other") is False
    empty = SimpleNamespace(grid_inspection_payloads_by_layer={})
    assert KarakalPresenter._class_conflict_ready(SimpleNamespace(), empty, "frame") is False


def test_disabled_class_conflict_is_not_stored_as_unchecked() -> None:
    box = _Box(checked=False, enabled=False)
    assert "class_conflict" in KarakalPresenter._selected_grid_error_types(_host(box, wants=True))
    assert "class_conflict" not in KarakalPresenter._selected_grid_error_types(_host(box, wants=False))
    box.setEnabled(True)
    box.setChecked(False)
    assert "class_conflict" not in KarakalPresenter._selected_grid_error_types(_host(box, wants=True))


def test_old_settings_drop_removed_reasons_and_comparison_layer() -> None:
    types = KarakalPresenter._normalize_grid_error_types(
        ["filled_cell", "partial_filled_cell", "small_artifact", "geometry_mismatch", "class_conflict"]
    )
    assert types == ("small_artifact", "class_conflict", "conductor_zone")
    assert KarakalPresenter._normalize_grid_layers(["confidence", "comparison", "binary"]) == ("confidence", "binary")
    softer = KarakalPresenter._grid_damage_config_from_payload({"disagreement_sensitivity": 0})
    stricter = KarakalPresenter._grid_damage_config_from_payload({"disagreement_sensitivity": 100})
    assert stricter.bad_score_threshold < softer.bad_score_threshold


@pytest.mark.skipif(grid_anomaly.cv2 is None, reason="OpenCV required for class-conflict components")
def test_overlapping_masks_show_class_conflict_on_matrix_and_details() -> None:
    ones = np.zeros((64, 64), dtype=bool)
    zeros = np.zeros((64, 64), dtype=bool)
    ones[10:40, 10:40] = True
    zeros[20:35, 20:35] = True
    conflict = grid_anomaly.analyze_class_conflict_masks(ones, zeros, frame_id="overlap")
    merged = KarakalPresenter._merge_grid_layer_results(
        SimpleNamespace(),
        {"binary": _binary(), "derived_conflict": conflict},
    )
    assert merged is not None
    assert any("class_conflict" in cell.reasons for cell in merged.cells)
    state = SimpleNamespace(grid_inspection_payloads_by_layer={"derived_conflict": {"overlap": conflict}})
    shown = KarakalPresenter._with_stored_class_conflict(SimpleNamespace(), _binary(), state, "overlap")
    assert any("class_conflict" in cell.reasons for cell in shown.cells)


def test_unchecked_class_conflict_is_left_out_of_the_matrix() -> None:
    cell = GridCellAnalysisResult(0, 0, (2, 2, 10, 10), (7.0, 7.0), 1, "suspicious", 0.8, ("class_conflict",))
    conflict = replace_frame(_binary(), (cell,))
    box = _Box(checked=False, enabled=True)
    host = _host(box, wants=False)
    host._selected_grid_error_types = lambda: KarakalPresenter._selected_grid_error_types(host)
    host._class_conflict_selected = lambda: KarakalPresenter._class_conflict_selected(host)
    merged = KarakalPresenter._merge_grid_layer_results(host, {"binary": _binary(), "derived_conflict": conflict})
    assert merged is not None
    assert merged.cells == ()
