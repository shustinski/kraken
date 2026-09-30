"""A broken cell is one geometry defect, never debris marks inside the cell."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from karakal.core.grid_anomaly import GridDamageAnalysisConfig, detect_grid_cell_anomalies

PITCH_X, PITCH_Y, CELL_W, CELL_H = 55, 60, 30, 40
SLOT = (5, 7)


def _lattice(draw_slot) -> np.ndarray:
    image = np.zeros((800, 1000), dtype=np.uint8)
    for row in range(12):
        for col in range(16):
            x, y = 40 + col * PITCH_X, 40 + row * PITCH_Y
            if (row, col) == SLOT:
                draw_slot(image, x, y)
            else:
                cv2.rectangle(image, (x, y), (x + CELL_W - 1, y + CELL_H - 1), 255, -1)
    return image


def _box(image, x0, y0, x1, y1, value=255) -> None:
    cv2.rectangle(image, (x0, y0), (x1, y1), value, -1)


def _defects(image) -> list[tuple[tuple[int, int, int, int], tuple[str, ...]]]:
    result = detect_grid_cell_anomalies(
        image, config=GridDamageAnalysisConfig(cell_representation="binary", scoring_mode="calibrated")
    )
    return [(tuple(cell.bbox), tuple(cell.reasons)) for cell in result.per_cell_results if cell.reasons]


def _slot_box() -> tuple[int, int, int, int]:
    return 40 + SLOT[1] * PITCH_X, 40 + SLOT[0] * PITCH_Y, CELL_W, CELL_H


BROKEN_CELLS = {
    # 'П' whose notch closed into a hole: the hole used to be debris, the cell unmarked.
    "closed_notch": lambda i, x, y: (_box(i, x, y, x + 29, y + 39), _box(i, x + 10, y + 14, x + 19, y + 35, 0)),
    "hole_in_center": lambda i, x, y: (_box(i, x, y, x + 29, y + 39), _box(i, x + 11, y + 14, x + 18, y + 25, 0)),
    # 'П' split into bar and legs: each piece used to be debris inside the cell.
    "split_bar_and_legs": lambda i, x, y: (
        _box(i, x, y, x + 29, y + 11),
        _box(i, x, y + 14, x + 8, y + 39),
        _box(i, x + 21, y + 14, x + 29, y + 39),
    ),
    "split_in_halves": lambda i, x, y: (_box(i, x, y, x + 29, y + 17), _box(i, x, y + 21, x + 29, y + 39)),
}


@pytest.mark.parametrize("name", sorted(BROKEN_CELLS))
def test_broken_cell_is_one_geometry_defect_without_debris(name) -> None:
    defects = _defects(_lattice(BROKEN_CELLS[name]))
    assert defects == [(_slot_box(), ("broken_geometry",))]


def test_tiny_hole_below_debris_size_is_ignored() -> None:
    image = _lattice(lambda i, x, y: (_box(i, x, y, x + 29, y + 39), _box(i, x + 13, y + 18, x + 16, y + 21, 0)))
    assert _defects(image) == []


def test_crumbs_in_an_empty_slot_stay_debris() -> None:
    image = _lattice(lambda i, x, y: (_box(i, x + 5, y + 5, x + 10, y + 10), _box(i, x + 14, y + 8, x + 19, y + 13)))
    defects = _defects(image)
    assert defects and all(reasons == ("small_artifact",) for _box_, reasons in defects)


def test_blob_larger_than_a_slot_with_a_hole_stays_debris() -> None:
    def draw(image, x, y) -> None:
        _box(image, x, y, x + 29, y + 39)

    image = _lattice(draw)
    # A 3x2-slot blob off the lattice with a hole inside.
    _box(image, 900, 600, 990, 720)
    _box(image, 930, 640, 960, 680, 0)
    blob = [reasons for bbox, reasons in _defects(image) if bbox[0] >= 900]
    assert blob == [("small_artifact",)]
