"""Broken geometry and frame-edge pieces are judged the same way for any cell shape."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from karakal.core.grid_anomaly import GridDamageAnalysisConfig, detect_grid_cell_anomalies


def _config() -> GridDamageAnalysisConfig:
    return GridDamageAnalysisConfig(cell_representation="binary", scoring_mode="calibrated").normalized()


def _lattice(cell_w: int, cell_h: int, gap_x: int, gap_y: int, *, left: int = 30, rows: int = 10, cols: int = 10):
    pitch_x, pitch_y = cell_w + gap_x, cell_h + gap_y
    image = np.zeros((60 + rows * pitch_y, left + 30 + cols * pitch_x), dtype=np.uint8)
    slots = {}
    for row in range(rows):
        for col in range(cols):
            x, y = left + col * pitch_x, 30 + row * pitch_y
            cv2.rectangle(image, (x, y), (x + cell_w - 1, y + cell_h - 1), 255, -1)
            slots[(row, col)] = (x, y)
    return image, slots


def _reasons_at(result, x: int, y: int) -> set[str]:
    found: set[str] = set()
    for cell in result.per_cell_results:
        bx, by, bw, bh = cell.bbox
        if bx - 2 <= x <= bx + bw + 2 and by - 2 <= y <= by + bh + 2:
            found |= set(cell.reasons)
    return found


@pytest.mark.parametrize("cell", ((23, 66, 17, 29), (35, 35, 15, 15), (66, 23, 17, 29)))
def test_side_bite_of_a_third_of_the_cell_is_broken_geometry(cell) -> None:
    cell_w, cell_h, gap_x, gap_y = cell
    image, slots = _lattice(cell_w, cell_h, gap_x, gap_y)
    assert not any(c.reasons for c in detect_grid_cell_anomalies(image, config=_config()).per_cell_results)
    x, y = slots[(4, 4)]
    if cell_h >= cell_w:
        depth, length = max(3, cell_w // 3), max(6, cell_h // 6)
        cv2.rectangle(image, (x + cell_w - depth, y + cell_h // 2), (x + cell_w - 1, y + cell_h // 2 + length), 0, -1)
    else:
        depth, length = max(3, cell_h // 3), max(6, cell_w // 6)
        cv2.rectangle(image, (x + cell_w // 2, y + cell_h - depth), (x + cell_w // 2 + length, y + cell_h - 1), 0, -1)
    result = detect_grid_cell_anomalies(image, config=_config())
    assert "broken_geometry" in _reasons_at(result, x + 2, y + 2)
    assert [c.bbox for c in result.per_cell_results if c.reasons] == [c.bbox for c in result.per_cell_results if "broken_geometry" in c.reasons]


def test_small_dent_in_the_end_of_a_long_cell_is_not_broken() -> None:
    image, slots = _lattice(23, 66, 17, 29)
    x, y = slots[(4, 4)]
    # A 5 px dent in the short end is 5/66 of the cell along that direction.
    cv2.rectangle(image, (x + 8, y), (x + 14, y + 4), 0, -1)
    result = detect_grid_cell_anomalies(image, config=_config())
    assert "broken_geometry" not in _reasons_at(result, x + 2, y + 20)


def test_cell_cut_by_the_frame_edge_is_not_debris_in_a_sparse_field() -> None:
    # Cells cut by the left frame edge leave thin strips; the field rows are irregular.
    image, _slots = _lattice(23, 66, 17, 29, left=-20, rows=8, cols=8)
    for row in (1, 3, 4, 6):
        image[30 + row * 95 : 30 + row * 95 + 66, :40] = 0
    result = detect_grid_cell_anomalies(image, config=_config())
    strips = [c for c in result.per_cell_results if c.bbox[0] <= 2]
    assert strips
    assert not any("small_artifact" in c.reasons for c in strips)
