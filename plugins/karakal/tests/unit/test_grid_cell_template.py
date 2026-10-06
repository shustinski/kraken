"""Mask-only geometry against the layer cell template (any cell shape)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from karakal.core import grid_anomaly as ga
from karakal.core.grid_anomaly import GridDamageAnalysisConfig, detect_grid_cell_anomalies, estimate_run_cell_reference_profile

H, W = 900, 1200
PX, PY, CW, CH = 50, 60, 24, 36
CONFIG = GridDamageAnalysisConfig(cell_representation="binary", scoring_mode="calibrated")


def _rect(image, x, y, w=CW, h=CH, rng=None) -> None:
    # Network-like cell: corner rounding and +-1 px edge jitter must stay normal.
    radius, jx, jy, jw, jh = 0, 0, 0, 0, 0
    if rng is not None:
        radius = int(rng.integers(0, 5))
        jx, jy, jw, jh = (int(value) for value in rng.integers(-1, 2, 4))
    x, y, w, h = x + jx, y + jy, w + jw, h + jh
    cv2.rectangle(image, (x + radius, y), (x + w - 1 - radius, y + h - 1), 255, -1)
    cv2.rectangle(image, (x, y + radius), (x + w - 1, y + h - 1 - radius), 255, -1)
    for cx, cy in ((x + radius, y + radius), (x + w - 1 - radius, y + radius), (x + radius, y + h - 1 - radius), (x + w - 1 - radius, y + h - 1 - radius)):
        cv2.circle(image, (cx, cy), radius, 255, -1)


def _triangle(image, x, y, w=CW + 8, h=CH, rng=None) -> None:
    jitter = int(rng.integers(-1, 2)) if rng is not None else 0
    cv2.fillPoly(image, [np.array([[x, y + h + jitter], [x + w // 2, y], [x + w, y + h + jitter]], np.int32)], 255)


def _cut_corner(image, x, y) -> None:
    cv2.fillPoly(image, [np.array([[x, y], [x + 24, y], [x + 24, y + 18], [x + 12, y + 36], [x, y + 36]], np.int32)], 255)


def _frame(cell, *, rows=12, cols=16, bad=None, draw_bad=None, seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros((H, W), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            x, y = 60 + col * PX, 60 + row * PY
            if (row, col) == bad:
                if draw_bad is not None:
                    draw_bad(image, x, y)
            else:
                cell(image, x, y, rng=rng)
    return image


def _profile(tmp_path: Path, cell):
    paths = []
    for index in range(4):
        path = tmp_path / f"{cell.__name__}_{index}.png"
        cv2.imwrite(str(path), _frame(cell, seed=10 + index))
        paths.append(str(path))
    profile = estimate_run_cell_reference_profile(paths, config=CONFIG, sample_limit=0)
    assert profile is not None and profile.shape_template is not None
    return profile


def _defects(image, profile) -> list[tuple[str, ...]]:
    result = detect_grid_cell_anomalies(image, config=CONFIG, reference_profile=profile)
    return [tuple(cell.reasons) for cell in result.per_cell_results if cell.reasons]


@pytest.fixture(scope="module")
def rect_profile(tmp_path_factory):
    return _profile(tmp_path_factory.mktemp("rect"), _rect)


@pytest.fixture(scope="module")
def triangle_profile(tmp_path_factory):
    return _profile(tmp_path_factory.mktemp("triangle"), _triangle)


def test_rounder_or_sharper_edges_stay_normal(rect_profile) -> None:
    assert _defects(_frame(_rect, seed=99), rect_profile) == []


BROKEN_RECT_CELLS = {
    "cut_corner": _cut_corner,
    "two_thirds_height": lambda i, x, y: _rect(i, x, y, h=24),
    "one_and_half_wide": lambda i, x, y: _rect(i, x, y, w=36),
    "turned_horizontal": lambda i, x, y: _rect(i, x - 6, y + 6, w=36, h=24),
}


@pytest.mark.parametrize("name", sorted(BROKEN_RECT_CELLS))
def test_broken_cell_is_geometry(rect_profile, name) -> None:
    image = _frame(_rect, bad=(5, 7), draw_bad=BROKEN_RECT_CELLS[name])
    assert _defects(image, rect_profile) == [("broken_geometry",)]


def test_piece_well_below_a_cell_is_debris(rect_profile) -> None:
    image = _frame(_rect, bad=(5, 7), draw_bad=lambda i, x, y: _rect(i, x, y, h=17))
    assert _defects(image, rect_profile) == [("small_artifact",)]


def test_lone_cell_on_an_empty_frame_is_checked(rect_profile) -> None:
    normal = np.zeros((H, W), dtype=np.uint8)
    _rect(normal, 600, 400)
    result = detect_grid_cell_anomalies(normal, config=CONFIG, reference_profile=rect_profile)
    assert result.detected_cells == 1 and _defects(normal, rect_profile) == []
    broken = np.zeros((H, W), dtype=np.uint8)
    _cut_corner(broken, 600, 400)
    assert _defects(broken, rect_profile) == [("broken_geometry",)]


def test_short_array_is_checked(rect_profile) -> None:
    image = _frame(_rect, rows=2, cols=3, bad=(1, 1), draw_bad=_cut_corner)
    assert _defects(image, rect_profile) == [("broken_geometry",)]


def test_frame_of_crumbs_stays_without_array(rect_profile) -> None:
    rng = np.random.default_rng(3)
    image = np.zeros((H, W), dtype=np.uint8)
    for _ in range(600):
        x, y = (int(value) for value in rng.integers(0, 1180, 2))
        size = int(rng.integers(3, 9))
        cv2.rectangle(image, (x, y % 880), (x + size, y % 880 + size), 255, -1)
    result = detect_grid_cell_anomalies(image, config=CONFIG, reference_profile=rect_profile)
    assert not result.grid_detected and not [cell for cell in result.per_cell_results if cell.reasons]


def test_triangle_layer_uses_its_own_shape(triangle_profile) -> None:
    assert _defects(_frame(_triangle, seed=99), triangle_profile) == []
    squashed = _frame(_triangle, bad=(5, 7), draw_bad=lambda i, x, y: _triangle(i, x, y + 12, h=24))
    assert _defects(squashed, triangle_profile) == [("broken_geometry",)]
    flipped = _frame(
        _triangle,
        bad=(5, 7),
        draw_bad=lambda i, x, y: cv2.fillPoly(i, [np.array([[x, y], [x + 32, y], [x + 16, y + 36]], np.int32)], 255),
    )
    assert _defects(flipped, triangle_profile) == [("broken_geometry",)]
    rectangle = _frame(_triangle, bad=(5, 7), draw_bad=lambda i, x, y: _rect(i, x + 4, y))
    assert _defects(rectangle, triangle_profile) == [("broken_geometry",)]


def test_template_reaches_workers_and_cache_key(rect_profile) -> None:
    import pickle

    restored = pickle.loads(pickle.dumps(rect_profile))
    assert restored.shape_template == rect_profile.shape_template
    assert rect_profile.cache_payload()["shape_template"]["mask_sha1"]


def test_short_cell_is_broken_with_and_without_template(tmp_path) -> None:
    profile = _profile(tmp_path, _rect)
    plain = ga.GridCellReferenceProfile(
        **{name: getattr(profile, name) for name in profile.__slots__ if name != "shape_template"}
    )
    image = _frame(_rect, bad=(5, 7), draw_bad=lambda i, x, y: _rect(i, x, y, h=24))
    # A 2/3-height cell: the template knows it, and so does the size shortfall in geometry.
    assert _defects(image, plain) == [("broken_geometry",)]
    assert _defects(image, profile) == [("broken_geometry",)]


def test_cells_glued_off_the_lattice_pitch_are_merged_not_debris(rect_profile) -> None:
    # Two cells stacked closer than the row pitch: one wide-as-a-cell body two cells tall.
    image = _frame(_rect, bad=(5, 7), draw_bad=lambda i, x, y: cv2.rectangle(i, (x, y), (x + CW - 1, y + 2 * CH + 4), 255, -1))
    assert _defects(image, rect_profile) == [("merged_contour",)]


def test_sparse_field_is_no_conductor_zone_but_grain_is(rect_profile) -> None:
    from karakal.core.grid_zones import ConductorZone, ConductorZoneMap

    view = ga._TemplateView(rect_profile.shape_template, CONFIG.normalized())
    zone = ConductorZone(bbox=(0, 0, 400, 400), area_px=160000)
    zone_map = ConductorZoneMap(zones=(zone,), tile=32, occupied=np.ones((13, 13), dtype=bool))
    sparse = np.zeros((H, W), dtype=np.uint8)
    sparse[100:112, 100:140] = 255  # a broken piece in an empty field: 0.3% of the zone
    grain = np.zeros((H, W), dtype=np.uint8)
    rng = np.random.default_rng(5)
    grain[:400, :400] = (rng.random((400, 400)) < 0.25).astype(np.uint8) * 255  # grain covers ~25%
    assert view.drop_cell_zones(zone_map, [], sparse).zones == ()
    assert view.drop_cell_zones(zone_map, [], grain).zones == (zone,)
