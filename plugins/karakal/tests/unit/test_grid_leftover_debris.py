"""Mask pieces that no filter claims are debris, on or off the cell lattice."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from karakal.core import grid_anomaly
from karakal.core.grid_anomaly import (
    LEFTOVER_DEBRIS_FEATURE,
    GridDamageAnalysisConfig,
    detect_grid_cell_anomalies,
    score_confidence_fill_on_binary_result,
)

PITCH = 50
CELL = 35


def _lattice(skip: set[tuple[int, int]] = frozenset(), shape=(800, 1400)) -> np.ndarray:
    image = np.zeros(shape, dtype=np.uint8)
    for row in range(15):
        for col in range(15):
            if (row, col) in skip:
                continue
            x = 30 + col * PITCH
            y = 30 + row * PITCH
            cv2.rectangle(image, (x, y), (x + CELL - 1, y + CELL - 1), 255, -1)
    return image


def _slot(row: int, col: int) -> tuple[int, int]:
    return 30 + col * PITCH, 30 + row * PITCH


def _config(**overrides) -> GridDamageAnalysisConfig:
    return GridDamageAnalysisConfig(cell_representation="binary", scoring_mode="calibrated", **overrides).normalized()


def _debris_boxes(image: np.ndarray, config: GridDamageAnalysisConfig | None = None) -> list[tuple[int, int, int, int]]:
    result = detect_grid_cell_anomalies(image, config=config or _config())
    return [tuple(cell.bbox) for cell in result.per_cell_results if "small_artifact" in cell.reasons]


def _draw_scratch_in_lattice(image: np.ndarray) -> tuple[int, int]:
    x, y = _slot(5, 5)
    cv2.rectangle(image, (x, y + 15), (x + 89, y + 18), 255, -1)
    return x, y + 15


def test_scratch_across_two_slots_is_debris() -> None:
    image = _lattice({(5, 5), (5, 6)})
    x, y = _draw_scratch_in_lattice(image)
    assert (x, y, 90, 4) in _debris_boxes(image)


def test_blob_larger_than_merged_cells_is_debris() -> None:
    image = _lattice({(row, col) for row in range(5, 9) for col in range(5, 9)})
    x, y = _slot(5, 5)
    cv2.rectangle(image, (x, y), (x + 179, y + 179), 255, -1)
    assert (x, y, 180, 180) in _debris_boxes(image)


@pytest.mark.parametrize("box", ((1000, 200, 150, 120), (1000, 500, 150, 4)))
def test_pieces_off_the_lattice_are_debris(box) -> None:
    x, y, w, h = box
    image = _lattice()
    cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), 255, -1)
    assert box in _debris_boxes(image)


def test_crumbs_below_minimum_area_stay_unmarked() -> None:
    image = _lattice()
    cv2.rectangle(image, (1000, 600), (1002, 602), 255, -1)
    assert _debris_boxes(image) == []


def test_clean_lattice_has_no_debris() -> None:
    assert _debris_boxes(_lattice()) == []


def test_disabled_debris_type_hides_leftover_pieces() -> None:
    image = _lattice({(5, 5), (5, 6)})
    _draw_scratch_in_lattice(image)
    enabled = tuple(reason for reason in grid_anomaly.GRID_DAMAGE_REASON_TYPES if reason != "small_artifact")
    assert _debris_boxes(image, _config(enabled_reason_types=enabled)) == []


def test_leftover_debris_is_not_scored_as_filled_cell() -> None:
    image = _lattice({(row, col) for row in range(5, 9) for col in range(5, 9)})
    x, y = _slot(5, 5)
    cv2.rectangle(image, (x, y), (x + 179, y + 179), 255, -1)
    binary = detect_grid_cell_anomalies(image, config=_config())
    blob = [cell for cell in binary.per_cell_results if tuple(cell.bbox) == (x, y, 180, 180)]
    assert blob and LEFTOVER_DEBRIS_FEATURE in dict(blob[0].feature_snapshot)

    # A dark confidence blob under the debris must not read as a filled cell.
    confidence = np.full(image.shape, 252, dtype=np.uint8)
    confidence[y : y + 180, x : x + 180] = 60
    scored = score_confidence_fill_on_binary_result(binary, confidence)
    assert not any(tuple(cell.bbox) == (x, y, 180, 180) for cell in scored.per_cell_results)
    assert not any("filled_cell" in cell.reasons for cell in scored.per_cell_results)


def _piece_off_lattice(side: int) -> np.ndarray:
    image = _lattice()
    cv2.rectangle(image, (1000, 600), (1000 + side - 1, 600 + side - 1), 255, -1)
    return image


def _marks_piece(image: np.ndarray, config: GridDamageAnalysisConfig | None = None) -> bool:
    # Small pieces get a padded box, so match by position.
    return any(x <= 1001 <= x + w and y <= 601 <= y + h for x, y, w, h in _debris_boxes(image, config))


def test_operator_minimum_size_hides_smaller_debris() -> None:
    # Mask smoothing grows a 6x6 square to a contour of about 49 px.
    image = _piece_off_lattice(6)
    assert _marks_piece(image)
    assert not _marks_piece(image, _config(debris_min_area_px=60))


def test_operator_minimum_size_can_mark_crumbs() -> None:
    # A 3x3 square is a contour of about 16 px, below the default 24.
    image = _piece_off_lattice(3)
    assert not _marks_piece(image)
    assert _marks_piece(image, _config(debris_min_area_px=10))


def test_minimum_size_is_clamped() -> None:
    assert _config(debris_min_area_px=-5).debris_min_area_px == 0
    assert _config(debris_min_area_px=10_000).debris_min_area_px == grid_anomaly.DEBRIS_MIN_AREA_MAX_PX
