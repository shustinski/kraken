"""Narrower settings applied to a stored run match a fresh run with those settings."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from karakal.core import grid_anomaly
from karakal.core.grid_anomaly import GridDamageAnalysisConfig, detect_grid_cell_anomalies
from karakal.core.grid_narrowing import GridNarrowing, can_narrow, narrow_grid_frame_result

PITCH = 50
CELL = 35


def _frame() -> np.ndarray:
    image = np.zeros((800, 1400), dtype=np.uint8)
    for row in range(15):
        for col in range(15):
            if (row, col) in {(5, 5), (5, 6), (9, 9)}:
                continue
            x, y = 30 + col * PITCH, 30 + row * PITCH
            cv2.rectangle(image, (x, y), (x + CELL - 1, y + CELL - 1), 255, -1)
    # A scratch across two empty slots, a broken cell and debris pieces of several sizes off the grid.
    cv2.rectangle(image, (280, 295), (369, 298), 255, -1)
    x, y = 30 + 9 * PITCH, 30 + 9 * PITCH
    cv2.rectangle(image, (x, y), (x + CELL - 1, y + 12), 255, -1)
    for index, side in enumerate((4, 6, 8, 11, 15)):
        cv2.rectangle(image, (1000 + 40 * index, 600), (1000 + 40 * index + side - 1, 600 + side - 1), 255, -1)
    return image


def _config(**overrides) -> GridDamageAnalysisConfig:
    return GridDamageAnalysisConfig(cell_representation="binary", scoring_mode="calibrated", **overrides).normalized()


def _marks(result) -> list[tuple[tuple[int, int, int, int], tuple[str, ...]]]:
    return sorted((tuple(cell.bbox), tuple(sorted(cell.reasons))) for cell in result.per_cell_results if cell.reasons)


@pytest.mark.parametrize("size", (30, 60, 120))
def test_raising_debris_size_matches_a_fresh_run(size) -> None:
    image = _frame()
    stored = detect_grid_cell_anomalies(image, config=_config())
    fresh = detect_grid_cell_anomalies(image, config=_config(debris_min_area_px=size))
    narrowed = narrow_grid_frame_result(stored, GridNarrowing(debris_min_area_px=size))
    assert _marks(narrowed) == _marks(fresh)
    assert narrowed.damage_score == pytest.approx(fresh.damage_score)
    assert narrowed.detected_cells == fresh.detected_cells


def test_switching_debris_off_matches_a_fresh_run() -> None:
    image = _frame()
    types = tuple(grid_anomaly.GRID_DAMAGE_REASON_TYPES)
    without = frozenset(reason for reason in types if reason != "small_artifact")
    stored = detect_grid_cell_anomalies(image, config=_config(enabled_reason_types=types))
    fresh = detect_grid_cell_anomalies(image, config=_config(enabled_reason_types=tuple(without)))
    narrowed = narrow_grid_frame_result(
        stored, GridNarrowing(debris_min_area_px=24, enabled_reason_types=without)
    )
    assert _marks(narrowed) == _marks(fresh)
    assert narrowed.damage_score == pytest.approx(fresh.damage_score)


def test_unchanged_settings_keep_the_stored_result() -> None:
    stored = detect_grid_cell_anomalies(_frame(), config=_config())
    assert narrow_grid_frame_result(stored, GridNarrowing(debris_min_area_px=24)) is stored


def test_only_narrower_settings_skip_a_new_run() -> None:
    types = ("small_artifact", "broken_geometry")
    assert can_narrow(run_debris_min_area_px=24, run_reason_types=types, debris_min_area_px=60, enabled_reason_types=types)
    assert can_narrow(
        run_debris_min_area_px=24, run_reason_types=types, debris_min_area_px=24, enabled_reason_types=("broken_geometry",)
    )
    assert not can_narrow(run_debris_min_area_px=24, run_reason_types=types, debris_min_area_px=10, enabled_reason_types=types)
    assert not can_narrow(
        run_debris_min_area_px=24,
        run_reason_types=("broken_geometry",),
        debris_min_area_px=24,
        enabled_reason_types=types,
    )
