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


def _preset(geometry: int, **overrides) -> GridDamageAnalysisConfig:
    """The config the app builds from a preset: strict (geometry 80) or balanced (40)."""

    from dataclasses import replace

    from karakal.app.presenter import KarakalPresenter
    from karakal.ui.ui_constants import GRID_INSPECTION_PRESET_VALUES

    values = dict(GRID_INSPECTION_PRESET_VALUES["strict" if geometry >= 80 else "balanced"])
    config = replace(KarakalPresenter._grid_damage_config_from_payload(values), cell_representation="binary", **overrides)
    return config.normalized()


def _replace_cell(image: np.ndarray, x: int, y: int, cell_w: int, cell_h: int, draw) -> None:
    image[y : y + cell_h, x : x + cell_w] = 0
    draw(image)


def test_short_cell_follows_the_geometry_slider() -> None:
    image, slots = _lattice(23, 66, 17, 29)
    x, y = slots[(4, 4)]
    _replace_cell(image, x, y, 23, 66, lambda img: cv2.rectangle(img, (x, y), (x + 22, y + 44), 255, -1))
    strict = detect_grid_cell_anomalies(image, config=_preset(80))
    balanced = detect_grid_cell_anomalies(image, config=_preset(40))
    assert "broken_geometry" in _reasons_at(strict, x + 2, y + 2)
    assert "broken_geometry" not in _reasons_at(balanced, x + 2, y + 2)


def test_half_cell_is_broken_geometry_not_debris() -> None:
    image, slots = _lattice(23, 66, 17, 29)
    x, y = slots[(4, 4)]
    _replace_cell(image, x, y, 23, 66, lambda img: cv2.rectangle(img, (x, y), (x + 22, y + 32), 255, -1))
    reasons = _reasons_at(detect_grid_cell_anomalies(image, config=_preset(40)), x + 2, y + 2)
    assert "broken_geometry" in reasons
    assert "small_artifact" not in reasons


def test_cut_corner_is_broken_on_strict_only() -> None:
    image, slots = _lattice(23, 66, 17, 29)
    x, y = slots[(4, 4)]
    # A slanted end: the corner cut 11 px along both sides.
    triangle = np.array([[x + 23, y + 66], [x + 12, y + 66], [x + 23, y + 55]], dtype=np.int32)
    cv2.fillPoly(image, [triangle], 0)
    assert "broken_geometry" in _reasons_at(detect_grid_cell_anomalies(image, config=_preset(80)), x + 2, y + 2)
    assert "broken_geometry" not in _reasons_at(detect_grid_cell_anomalies(image, config=_preset(40)), x + 2, y + 2)


def test_geometry_does_not_depend_on_the_debris_size() -> None:
    image, slots = _lattice(23, 66, 17, 29)
    x, y = slots[(4, 4)]
    # A cell split into a top bar and a lower part, with a 3 px crumb in the split.
    _replace_cell(image, x, y, 23, 66, lambda img: None)
    cv2.rectangle(image, (x, y), (x + 22, y + 12), 255, -1)
    cv2.rectangle(image, (x, y + 18), (x + 22, y + 65), 255, -1)
    cv2.rectangle(image, (x + 10, y + 14), (x + 12, y + 16), 255, -1)
    geometry = {}
    for size in (5, 24, 120):
        result = detect_grid_cell_anomalies(image, config=_preset(40, debris_min_area_px=size))
        geometry[size] = sorted(tuple(c.bbox) for c in result.per_cell_results if "broken_geometry" in c.reasons)
    assert geometry[5] == geometry[24] == geometry[120]
