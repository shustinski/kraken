"""Mask components, conductor zones and the normal-cell bank, all without a grid."""

from __future__ import annotations

import numpy as np
import pytest

from karakal.core.cell_components import extract_components
from karakal.core.conductor_ignore import final_ignore, preliminary_zones
from karakal.core.normal_bank import (
    STATUS_INSUFFICIENT,
    STATUS_OK,
    NormalBank,
    build_normal_bank,
    shape_descriptor,
)
from karakal.core.synthetic_masks import SyntheticLayout, add_defect, normal_frame, synthetic_frame


def test_component_features_and_border_flag() -> None:
    image = np.zeros((60, 80), dtype=np.uint8)
    image[10:40, 10:30] = 255
    image[20:25, 15:20] = 0  # a hole
    image[0:12, 60:80] = 255  # cut by the top and right edges
    components = extract_components(image)
    inner, edge = sorted(components.components, key=lambda item: item.touches_border)
    assert inner.bbox == (10, 10, 20, 30)
    assert inner.area == 600 - 25 and inner.hole_area == 25
    assert not inner.touches_border
    assert set(edge.border_sides) == {"top", "right"}
    assert 0.0 < inner.compactness <= 1.0 and len(inner.hu) == 7


def _frames(count: int, **layout) -> list:
    return [extract_components(normal_frame(SyntheticLayout(**layout), seed=seed).mask) for seed in range(count)]


def test_session_bank_from_normal_frames() -> None:
    bank = build_normal_bank(_frames(4))
    assert bank.status == STATUS_OK
    assert bank.cell_width == pytest.approx(18, abs=1) and bank.cell_height == pytest.approx(40, abs=1)
    assert bank.template is not None
    restored = NormalBank.from_payload(bank.to_payload())
    assert restored.bank_id == bank.bank_id
    assert np.allclose(restored.exemplars, bank.exemplars, atol=1e-2)


def test_bank_distance_separates_bites_and_corners_from_normal_cells() -> None:
    bank = build_normal_bank(_frames(4))
    frame = synthetic_frame({"bite": 3, "cut_corner": 3}, seed=21)
    components = extract_components(frame.mask)
    defect_pixels = np.zeros(frame.mask.shape, dtype=bool)
    for defect in frame.defects:
        defect_pixels |= defect.pixels
    normal, broken = [], []
    for item in components.components:
        if item.touches_border:
            continue
        distance = float(bank.normal_distance(shape_descriptor(item, components.labels)[None, :])[0])
        piece, x0, y0 = item.crop(components.labels)
        hit = defect_pixels[y0 : y0 + piece.shape[0], x0 : x0 + piece.shape[1]][piece].any()
        (broken if hit else normal).append(distance)
    assert len(broken) == 6
    assert max(normal) < min(broken)


def test_crumbs_only_give_an_insufficient_bank() -> None:
    rng = np.random.default_rng(0)
    sets = []
    for _ in range(3):
        image = np.zeros((300, 300), dtype=np.uint8)
        for _ in range(300):
            x, y = rng.integers(2, 296, size=2)
            image[y : y + rng.integers(1, 4), x : x + rng.integers(1, 4)] = 255
        sets.append(extract_components(image))
    bank = build_normal_bank(sets)
    assert bank.status == STATUS_INSUFFICIENT
    assert bank.reason


def test_conductor_field_is_found_on_any_side_and_spares_cells_next_to_it() -> None:
    plain = extract_components(normal_frame(seed=3).mask)
    assert not preliminary_zones(plain).found
    for seed in (1, 2, 5):
        frame = add_defect(normal_frame(seed=seed), "conductor", seed=seed)
        components = extract_components(frame.mask)
        zones = preliminary_zones(components)
        field_share = zones.zone[frame.ignore].mean()
        assert field_share > 0.8
        bank = build_normal_bank(_frames(3))
        final = final_ignore(
            components,
            cell_area=bank.cell_area,
            # A cell, or a sliver of one cut by the frame edge (decision Н5).
            is_cell_like=lambda item: (0.6 <= item.area / bank.cell_area <= 1.4 and item.solidity > 0.9)
            or (item.touches_border and max(item.width, item.height) <= 1.25 * bank.cell_height),
        )
        cells = (frame.truth > 0) & ~frame.ignore
        assert not (final.zone & cells).any()
        assert final.zone[frame.ignore & (frame.mask > 0)].mean() > 0.9
