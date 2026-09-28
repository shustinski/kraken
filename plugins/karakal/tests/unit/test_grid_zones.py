"""Conductor zones replace off-lattice grain and stay out of the cell score."""

from __future__ import annotations

import numpy as np
import pytest

from karakal.core.grid_anomaly import detect_grid_cell_anomalies

pytest.importorskip("cv2")
import cv2  # noqa: E402


def _lattice(image: np.ndarray, *, rows: int = 6, cols: int = 8, origin: tuple[int, int] = (20, 20)) -> None:
    x0, y0 = origin
    for row in range(rows):
        for col in range(cols):
            x = x0 + col * 36
            y = y0 + row * 48
            cv2.rectangle(image, (x, y), (x + 22, y + 32), 255, -1)


def _config():
    from karakal.core.grid_anomaly import GridDamageAnalysisConfig

    return GridDamageAnalysisConfig(
        cell_representation="binary",
        scoring_mode="calibrated",
        blur_radius=1,
        min_contour_area=8.0,
        min_cell_size=2,
        fill_sensitivity=60,
        debris_sensitivity=75,
        geometry_sensitivity=40,
        merge_sensitivity=35,
        min_grid_candidate_cells=4,
    )


def _zones(result):
    return [cell for cell in result.cells if "conductor_zone" in cell.reasons]


def _cell_defects(result):
    return [
        cell
        for cell in result.cells
        if cell.reasons and "conductor_zone" not in cell.reasons
    ]


def _centroid_in_zone(cell, zone) -> bool:
    outline = tuple(getattr(zone, "outline", ()) or ())
    if len(outline) >= 3:
        contour = np.asarray(outline, dtype=np.int32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(contour, (float(cell.centroid[0]), float(cell.centroid[1])), False) >= 0
    x, y, width, height = zone.bbox
    return x <= cell.centroid[0] < x + width and y <= cell.centroid[1] < y + height


def test_grain_beside_a_lattice_is_one_conductor_zone() -> None:
    image = np.zeros((520, 420), dtype=np.uint8)
    _lattice(image, origin=(24, 180))
    rng = np.random.default_rng(3)
    for _ in range(80):
        x = int(rng.integers(10, 380))
        y = int(rng.integers(8, 140))
        w = int(rng.integers(3, 9))
        h = int(rng.integers(3, 8))
        cv2.rectangle(image, (x, y), (x + w, y + h), 255, -1)

    result = detect_grid_cell_anomalies(image, config=_config())
    zones = _zones(result)
    assert 1 <= len(zones) <= 3
    assert zones[0].bbox[3] > 80
    inside = [
        cell
        for cell in _cell_defects(result)
        if cell.centroid[1] < 170 and any(_centroid_in_zone(cell, zone) for zone in zones)
    ]
    assert inside == []
    assert result.zone_skipped_components > 0
    lattice_defects = [cell for cell in _cell_defects(result) if cell.centroid[1] >= 170]
    assert lattice_defects == []


def test_a_broken_block_inside_the_lattice_stays_cell_defects() -> None:
    image = np.zeros((360, 400), dtype=np.uint8)
    _lattice(image, rows=5, cols=8, origin=(16, 16))
    for row in range(2):
        for col in range(5):
            x = 16 + col * 36
            y = 16 + row * 48
            cv2.rectangle(image, (x, y), (x + 22, y + 32), 0, -1)
            cv2.rectangle(image, (x, y), (x + 22, y + 32), 255, 2)
            cv2.line(image, (x, y), (x + 22, y + 32), 255, 2)

    result = detect_grid_cell_anomalies(image, config=_config())
    assert _zones(result) == []
    assert _cell_defects(result)


def test_frame_without_a_lattice_is_a_conductor_zone() -> None:
    image = np.zeros((280, 360), dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (70, 40), 255, -1)
    cv2.rectangle(image, (90, 16), (130, 150), 255, -1)
    cv2.rectangle(image, (160, 40), (250, 90), 255, -1)
    cv2.rectangle(image, (20, 160), (200, 250), 255, -1)
    cv2.rectangle(image, (230, 140), (340, 260), 255, -1)
    cv2.rectangle(image, (270, 12), (340, 80), 255, -1)

    result = detect_grid_cell_anomalies(image, config=_config())
    assert _zones(result)
    assert not any("merged_contour" in cell.reasons for cell in result.cells)
    assert result.damage_score == 0.0


def test_l_shaped_zone_does_not_cover_the_corner_array() -> None:
    image = np.zeros((480, 480), dtype=np.uint8)
    _lattice(image, rows=4, cols=4, origin=(280, 280))
    rng = np.random.default_rng(4)
    for _ in range(140):
        if rng.random() < 0.55:
            x = int(rng.integers(8, 450))
            y = int(rng.integers(8, 180))
        else:
            x = int(rng.integers(8, 180))
            y = int(rng.integers(8, 450))
        cv2.rectangle(image, (x, y), (x + int(rng.integers(2, 7)), y + int(rng.integers(2, 6))), 255, -1)

    result = detect_grid_cell_anomalies(image, config=_config())
    zones = _zones(result)
    assert zones
    assert result.zone_skipped_components > 0
    array_cells = [
        cell
        for cell in result.cells
        if cell.centroid[0] > 270 and cell.centroid[1] > 270 and "conductor_zone" not in cell.reasons
    ]
    assert len(array_cells) >= 8
    assert not any(_centroid_in_zone(cell, zone) for cell in array_cells for zone in zones)
    largest = max(zones, key=lambda zone: zone.bbox[2] * zone.bbox[3])
    assert len(largest.outline) >= 6
    contour = np.asarray(largest.outline, dtype=np.int32).reshape(-1, 1, 2)
    assert abs(cv2.contourArea(contour)) < 0.92 * float(largest.bbox[2] * largest.bbox[3])


def test_conductor_zone_does_not_raise_damage_score() -> None:
    clean = np.zeros((360, 340), dtype=np.uint8)
    _lattice(clean, rows=5, cols=7, origin=(16, 16))
    grainy = clean.copy()
    rng = np.random.default_rng(9)
    for _ in range(70):
        x = int(rng.integers(8, 300))
        y = int(rng.integers(280, 350))
        cv2.rectangle(grainy, (x, y), (x + 5, y + 4), 255, -1)
    bare = detect_grid_cell_anomalies(clean, config=_config())
    with_zone = detect_grid_cell_anomalies(grainy, config=_config())
    assert _zones(with_zone)
    assert with_zone.damage_score <= bare.damage_score + 1e-6
