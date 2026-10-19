"""Dirty regions of a frame: found from the mask (and the confidence map) alone, in frame pixels."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from karakal.core.dirt_regions import (
    LEVEL_LIGHT,
    LEVELS,
    DirtConfig,
    analyze_dirt_frame,
    dirt_layers,
    find_dirt_regions,
    region_label_image,
    render_dirt_view,
)

PX, PY = 37.75, 62.9
FRAME_W, FRAME_H = 760, 700


def lattice() -> np.ndarray:
    """A clean array: crisp cells on the pitch, as a network draws them."""

    mask = np.zeros((FRAME_H, FRAME_W), np.uint8)
    for r in range(int(FRAME_H // PY)):
        for c in range(int(FRAME_W // PX)):
            x, y = int(8 + c * PX), int(10 + r * PY)
            cv2.rectangle(mask, (x, y), (x + 20, y + 40), 255, -1)
    return cv2.GaussianBlur(mask, (3, 3), 0)


def smear(mask: np.ndarray, cx: int, cy: int, rx: int, ry: int, seed: int = 0) -> np.ndarray:
    """The network drew a smear: mid-gray, noisy, no crisp shape."""

    rng = np.random.default_rng(seed)
    out = mask.copy()
    ellipse = np.zeros_like(mask)
    cv2.ellipse(ellipse, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    noise = cv2.GaussianBlur(rng.integers(70, 190, mask.shape).astype(np.uint8), (5, 5), 0)
    out[ellipse > 0] = noise[ellipse > 0]
    return out


def sure(mask: np.ndarray) -> np.ndarray:
    """Confidence of a clean network: sure everywhere but a thin ring on every drawn edge."""

    hi = (mask >= 128).astype(np.uint8)
    edge = cv2.morphologyEx(hi, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    confidence = np.full(mask.shape, 255, np.uint8)
    confidence[cv2.dilate(edge, np.ones((3, 3), np.uint8)) > 0] = 120
    return confidence


def test_clean_array_has_no_dirt() -> None:
    mask = lattice()
    assert find_dirt_regions(mask) == ()
    # The low-confidence ring every drawn edge has is not dirt.
    layers = dirt_layers(mask, sure(mask))
    assert not layers.unsure.any()
    assert find_dirt_regions(mask, sure(mask)) == ()


def test_smear_is_one_region_in_frame_pixels() -> None:
    mask = smear(lattice(), 380, 350, 70, 90)
    regions = find_dirt_regions(mask)
    assert len(regions) == 1
    region = regions[0]
    x, y, w, h = region.bbox
    assert x < 380 < x + w and y < 350 < y + h
    # The box hugs the smear (grown by the dirtiness window), not the frame.
    assert w < 3 * 140 and h < 3 * 180
    assert cv2.pointPolygonTest(np.asarray(region.contour, np.int32), (380, 350), False) > 0
    assert region.id == 1 and region.parts == 1 and not region.edge
    assert region.level in LEVELS and 0.0 < region.dirtiness <= 1.0 <= region.peak + 1.0
    assert region.blur_px > 0 and region.dirty_area_px <= region.area_px


def test_scattered_nicks_alone_are_not_a_region() -> None:
    mask = lattice()
    rng = np.random.default_rng(3)
    for _ in range(12):
        x, y = int(rng.integers(20, FRAME_W - 20)), int(rng.integers(20, FRAME_H - 20))
        cv2.circle(mask, (x, y), 3, 0, -1)
    assert find_dirt_regions(mask) == ()


def test_close_clusters_merge_into_one_region() -> None:
    mask = smear(smear(lattice(), 240, 350, 55, 80, seed=1), 520, 350, 55, 80, seed=2)
    apart = find_dirt_regions(mask, config=DirtConfig(merge_gap_cells=0.5))
    assert len(apart) == 2
    together = find_dirt_regions(mask, config=DirtConfig(merge_gap_cells=4.0))
    assert len(together) == 1
    region = together[0]
    assert region.parts == 2 and len(region.part_contours) == 2
    # One outline over both clusters, with the gap between them bridged.
    polygon = np.asarray(region.contour, np.int32)
    assert cv2.pointPolygonTest(polygon, (240, 350), False) > 0
    assert cv2.pointPolygonTest(polygon, (520, 350), False) > 0
    assert region.area_px > region.dirty_area_px


def test_region_at_frame_border_is_flagged_and_reaches_it() -> None:
    mask = smear(lattice(), 30, 350, 60, 90)
    (region,) = find_dirt_regions(mask)
    assert region.edge
    assert region.bbox[0] == 0


def test_network_unsure_away_from_edges_is_dirt() -> None:
    mask = lattice()
    confidence = sure(mask)
    cv2.ellipse(confidence, (380, 350), (80, 100), 0, 0, 360, 90, -1)
    assert find_dirt_regions(mask) == ()
    regions = find_dirt_regions(mask, confidence)
    assert len(regions) == 1 and regions[0].unsure_px > 0


def test_probability_map_saved_as_confidence_is_read_as_certainty() -> None:
    mask = lattice()
    # A class-probability map (the mask itself) is sure everywhere but on the edges.
    assert find_dirt_regions(mask, mask) == ()


def test_higher_sensitivity_finds_weaker_dirt() -> None:
    mask = lattice()
    confidence = sure(mask)
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[:FRAME_H, :FRAME_W]
    spots = (rng.random(mask.shape) < 0.08) & (np.hypot(xx - 380, yy - 350) < 110)
    confidence[spots & (cv2.dilate((mask > 128).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0)] = 60
    assert find_dirt_regions(mask, confidence, DirtConfig(sensitivity=1.0)) == ()
    assert len(find_dirt_regions(mask, confidence, DirtConfig(sensitivity=2.0))) == 1


def test_analyze_frame_from_files_and_label_image(tmp_path) -> None:
    mask = smear(lattice(), 380, 350, 70, 90)
    folder = tmp_path / "Маски"
    folder.mkdir()
    mask_path = folder / "TEST_00001.png"
    confidence_path = folder / "TEST_00001_confidence.png"
    cv2.imencode(".png", mask)[1].tofile(str(mask_path))
    cv2.imencode(".png", sure(mask))[1].tofile(str(confidence_path))
    result = analyze_dirt_frame(mask_path, frame_id="TEST_00001", confidence_path=confidence_path)
    assert result.used_confidence and not result.error
    assert (result.width, result.height) == (FRAME_W, FRAME_H)
    assert len(result.regions) == 1 and result.max_dirtiness == result.regions[0].dirtiness
    assert 0.0 < result.dirty_share < 0.5
    label = region_label_image(result)
    assert label.shape == (FRAME_H, FRAME_W)
    assert label[350, 380] == 1 and label[20, 20] == 0
    view = render_dirt_view(mask, result, dirt_layers(mask))
    assert view.shape == (FRAME_H, FRAME_W, 3)

    no_confidence = analyze_dirt_frame(mask_path, frame_id="TEST_00001")
    assert not no_confidence.used_confidence and no_confidence.confidence_path == ""
    missing = analyze_dirt_frame(tmp_path / "nothing.png", frame_id="x")
    assert missing.error and missing.regions == ()


def test_level_follows_dirtiness() -> None:
    light = find_dirt_regions(smear(lattice(), 380, 350, 45, 60))
    assert light and all(region.level in LEVELS for region in light)
    assert LEVEL_LIGHT in LEVELS


@pytest.mark.parametrize("gap", [0.0, 3.0])
def test_regions_do_not_overlap(gap: float) -> None:
    mask = smear(smear(smear(lattice(), 150, 200, 50, 70, 1), 400, 450, 60, 80, 2), 620, 160, 50, 70, 3)
    regions = find_dirt_regions(mask, config=DirtConfig(merge_gap_cells=gap))
    label = np.zeros(mask.shape, np.int32)
    for region in regions:
        fill = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(fill, [np.asarray(region.contour, np.int32)], 1)
        assert not (label[fill > 0] > 0).any()
        label[fill > 0] = region.id
