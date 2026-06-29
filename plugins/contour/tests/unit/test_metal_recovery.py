from __future__ import annotations

import numpy as np
import pytest

from contour.application.processing import ContourExtractionSettings
from contour.ui.metal_presets import noisy_sem_metal_preset_payload
from contour.vision.metal_recovery import (
    MetalRecoveryConfig,
    detect_metalization,
    estimate_inward_direction_by_gradient_profile,
)
from contour.vision.metal_recovery.detector import (
    _contour_to_polygon,
    _remove_vertices_below_angle,
    _repair_contour_polygon_for_topology,
    _snap_rectilinear,
    _valid_topology,
    build_metal_extraction_mask,
    effective_conductor_width_px,
)
from contour.vision.metal_recovery.settings_bridge import metal_recovery_config_from_settings


def test_effective_conductor_width_blends_with_bbox_minor_axis() -> None:
    assert effective_conductor_width_px(6.0, 12.0, 100.0) == pytest.approx(0.92 * 12.0)
    assert effective_conductor_width_px(10.0, 8.0, 100.0) == 10.0
    assert effective_conductor_width_px(5.0, 0.0, 0.0) == 5.0


def test_detect_metalization_finds_bright_bar() -> None:
    img = np.zeros((120, 160), dtype=np.uint8)
    img[40:45, 20:140] = 240
    cfg = MetalRecoveryConfig(
        min_width_px=3.0,
        min_length_px=20.0,
        min_area=30.0,
        min_perimeter=20.0,
        min_straightness=0.35,
        allowed_angles="free",
        morph_close_radius=2,
        check_contour_validity=False,
    )
    r = detect_metalization(img, cfg)
    assert r.accepted, "expected at least one accepted trace"
    assert r.debug_images.get("metal_binary_mask") is not None


def test_hierarchy_mode_full_preserves_holes() -> None:
    img = np.zeros((120, 120), dtype=np.uint8)
    img[20:100, 20:100] = 240
    img[45:75, 45:75] = 0
    base = dict(
        segmentation_strategy="legacy_otsu",
        min_width_px=2.0,
        min_length_px=10.0,
        min_area=20.0,
        min_perimeter=20.0,
        allowed_angles="free",
        morph_close_radius=1,
        check_contour_validity=True,
    )

    external = detect_metalization(img, MetalRecoveryConfig(**base, retrieval_external_only=True))
    full = detect_metalization(img, MetalRecoveryConfig(**base, retrieval_external_only=False))

    assert not any(polygon.is_hole for polygon in external.accepted)
    holes = [polygon for polygon in full.accepted if polygon.is_hole]
    assert holes
    assert all(hole.parent_id is not None for hole in holes)


def test_estimate_inward_direction_by_profile() -> None:
    img = np.zeros((48, 48), dtype=np.uint8)
    img[8:40, 10:16] = 255
    img[8:40, 16:38] = 90
    inward, conf, prof = estimate_inward_direction_by_gradient_profile(
        img, (14.0, 22.0), (1.0, 0.0), 10, min_confidence=0.01
    )
    assert prof is not None and prof.shape[0] == 21
    assert 0.0 <= conf <= 1.0
    assert inward is not None
    assert inward[0] > 0.2


def test_estimate_inward_direction_flat_is_uncertain() -> None:
    flat = np.full((24, 24), 120, dtype=np.uint8)
    inward, conf, _ = estimate_inward_direction_by_gradient_profile(flat, (12.0, 12.0), (1.0, 0.0), 6)
    assert inward is None
    assert conf < 0.15


def test_wide_gradient_recover_disabled_is_noop() -> None:
    img = np.zeros((120, 160), dtype=np.uint8)
    img[40:45, 20:140] = 240
    cfg = MetalRecoveryConfig(
        min_width_px=3.0,
        min_length_px=20.0,
        min_area=30.0,
        min_perimeter=20.0,
        min_straightness=0.35,
        allowed_angles="free",
        morph_close_radius=2,
        check_contour_validity=False,
        use_wide_conductor_gradient=False,
    )
    r = detect_metalization(img, cfg)
    assert r.accepted
    assert not r.wide_gradient_overlays.get("wide_pairs_suspicious")


def test_wide_gradient_finds_hollow_bar() -> None:
    img = np.full((220, 220), 35, dtype=np.uint8)
    img[50:170, 52:58] = 250
    img[50:170, 102:108] = 250
    cfg = MetalRecoveryConfig(
        min_width_px=8.0,
        max_width_px=80.0,
        min_length_px=40.0,
        min_area=200.0,
        min_perimeter=80.0,
        min_straightness=0.35,
        allowed_angles="free",
        morph_close_radius=2,
        check_contour_validity=False,
        use_wide_conductor_gradient=True,
        wide_gradient_min_pair_length_px=30.0,
        wide_gradient_parallel_tolerance_deg=15.0,
        wide_gradient_min_direction_confidence=0.08,
    )
    r = detect_metalization(img, cfg)
    wide = [p for p in r.accepted if p.category == "metal_wide_gradient"]
    assert wide, "expected wide-gradient conductor"
    assert r.debug_images.get("metal_wide_edge_map") is not None


def test_valid_topology_accepts_dense_simple_outline() -> None:
    """Many vertices on a simple rectangle must not false-fail (old [::step] subsampling could)."""
    pts: list[tuple[float, float]] = []
    for i in range(180):
        pts.append((float(i), 0.0))
    for i in range(180):
        pts.append((179.0, float(i)))
    for i in range(179, -1, -1):
        pts.append((float(i), 179.0))
    for i in range(179, 0, -1):
        pts.append((0.0, float(i)))
    ok, reason = _valid_topology(pts, enabled=True)
    assert ok, reason


def test_topology_repair_accepts_equivalent_valid_open_cv_contour() -> None:
    contour = np.array([[[10, 10]], [[70, 10]], [[70, 40]], [[10, 40]]], dtype=np.int32)
    invalid_display_ring = [(10.0, 10.0), (70.0, 40.0), (10.0, 40.0), (70.0, 10.0)]
    cfg = MetalRecoveryConfig(
        epsilon_simplify=1.0,
        min_polygon_angle_deg=0.0,
        check_contour_validity=True,
    )

    repaired, ok, reason = _repair_contour_polygon_for_topology(contour, invalid_display_ring, cfg, (80, 90))

    assert ok, reason
    assert repaired != invalid_display_ring
    assert _valid_topology(repaired, enabled=True)[0]


def test_min_polygon_angle_removes_vertices_below_threshold() -> None:
    points = [(0.0, 0.0), (10.0, 0.0), (10.5, 2.0), (11.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)]

    filtered = _remove_vertices_below_angle(points, 30.0)

    assert (10.5, 2.0) not in filtered


def test_min_polygon_angle_keeps_vertices_at_or_above_threshold() -> None:
    points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    filtered = _remove_vertices_below_angle(points, 30.0)

    assert filtered == points


def test_edge_mode_separates_adjacent_parallel_traces() -> None:
    """Two bright bars with a narrow dark gap should not merge into one polygon (SEM-like)."""
    img = np.zeros((80, 120), dtype=np.uint8)
    img[35:42, 10:45] = 235
    img[35:42, 52:95] = 235
    cfg = MetalRecoveryConfig(
        segmentation_strategy="edges",
        min_width_px=4.0,
        min_length_px=12.0,
        min_area=40.0,
        min_perimeter=30.0,
        min_straightness=0.35,
        allowed_angles="free",
        morph_close_radius=3,
        check_contour_validity=False,
        edge_watershed_split=True,
    )
    r = detect_metalization(img, cfg)
    assert len(r.accepted) >= 2


def test_max_width_rejects_blob() -> None:
    img = np.zeros((100, 100), dtype=np.uint8)
    img[30:70, 30:70] = 250
    cfg = MetalRecoveryConfig(
        segmentation_strategy="legacy_otsu",
        min_width_px=2.0,
        max_width_px=15.0,
        min_length_px=5.0,
        min_area=50.0,
        min_perimeter=30.0,
        allowed_angles="free",
        morph_close_radius=2,
        check_contour_validity=False,
    )
    r = detect_metalization(img, cfg)
    solid = [p for p in r.accepted if p.category == "conductor" and not p.is_hole]
    assert not solid


def _noisy_sem_synthetic() -> np.ndarray:
    rng = np.random.default_rng(42)
    img = np.zeros((160, 200), dtype=np.uint8)
    img[60:68, 30:170] = 210
    noise = rng.integers(0, 55, size=img.shape, dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return img


def test_contrast_bias_changes_mask_monotonically() -> None:
    img = _noisy_sem_synthetic()
    areas: list[int] = []
    for bias in (-10, -9):
        cfg = MetalRecoveryConfig(
            noise_suppression=70,
            contrast_bias=float(bias),
            segmentation_strategy="sauvola",
            gap_bridge_px=4,
            speckle_removal_px=2,
            min_width_px=3.0,
        )
        mask, _ = build_metal_extraction_mask(img, cfg)
        areas.append(int(np.count_nonzero(mask)))
    assert areas[0] > 0
    delta = abs(areas[1] - areas[0]) / max(areas[0], 1)
    assert delta < 0.15


def test_noisy_sem_preset_finds_trace_on_synthetic() -> None:
    settings = ContourExtractionSettings.from_dict(noisy_sem_metal_preset_payload())
    cfg = metal_recovery_config_from_settings(settings)
    cfg = MetalRecoveryConfig(
        **{**cfg.to_snapshot(), "min_width_px": 4.0, "min_length_px": 20.0, "min_area": 30.0}
    )
    r = detect_metalization(_noisy_sem_synthetic(), cfg)
    assert r.accepted


def test_rectilinear_snap_aligns_near_horizontal_edge() -> None:
    points = [(0.0, 0.0), (100.0, 6.0), (100.0, 50.0), (0.0, 50.0)]
    snapped = _snap_rectilinear(points, allowed_angles="90_only", tolerance_deg=12.0)
    assert abs(snapped[1][1] - snapped[0][1]) < 2.0


def test_legacy_settings_migration_in_contour_settings() -> None:
    settings = ContourExtractionSettings.from_dict(
        {"metal_sensitivity_0_100": 78, "metal_sensitivity": "low", "metal_segmentation_method": "otsu"}
    )
    assert hasattr(settings, "metal_contrast_bias")
    assert settings.metal_segmentation_strategy == "auto"
