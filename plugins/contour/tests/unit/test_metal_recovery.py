from __future__ import annotations

import numpy as np
import pytest

from contour.application.processing import ContourExtractionSettings
from contour.contour_extractor import extract_polygons
from contour.domain.polygon_ring import is_valid_closed_polygon_ring
from contour.ui.metal_presets import noisy_sem_metal_preset_payload, standard_metal_preset_payload
from contour.vision.metal_recovery import MetalRecoveryConfig, detect_metalization
from contour.vision.metal_recovery.detector import (
    build_metal_extraction_mask,
    contour_extraction_settings_from_metal,
)
from contour.vision.metal_recovery.settings_bridge import metal_recovery_config_from_settings


def test_detect_metalization_finds_bright_bar() -> None:
    img = np.zeros((120, 160), dtype=np.uint8)
    img[40:45, 20:140] = 240
    cfg = MetalRecoveryConfig(
        min_width_px=3.0,
        min_length_px=20.0,
        min_area=30.0,
        min_perimeter=20.0,
        gap_bridge_px=2,
        segmentation_strategy="legacy_otsu",
    )
    result = detect_metalization(img, cfg)
    assert result.accepted, "expected at least one accepted trace"
    assert result.debug_images.get("metal_binary_mask") is not None


def test_hierarchy_mode_full_preserves_holes() -> None:
    img = np.zeros((120, 120), dtype=np.uint8)
    img[20:100, 20:100] = 240
    img[45:75, 45:75] = 0
    base = dict(
        segmentation_strategy="legacy_otsu",
        min_width_px=2.0,
        min_area=20.0,
        min_perimeter=20.0,
        gap_bridge_px=1,
        min_polygon_angle_deg=0.0,
    )

    external = detect_metalization(img, MetalRecoveryConfig(**base, retrieval_external_only=True))
    full = detect_metalization(img, MetalRecoveryConfig(**base, retrieval_external_only=False))

    assert not any(polygon.is_hole for polygon in external.accepted)
    holes = [polygon for polygon in full.accepted if polygon.is_hole]
    assert holes
    assert all(hole.parent_id is not None for hole in holes)


def test_full_hierarchy_setting_uses_tree_retrieval() -> None:
    settings = ContourExtractionSettings(
        metal_hierarchy_mode="full",
        retrieval_mode="RETR_EXTERNAL",
    )

    cfg = metal_recovery_config_from_settings(settings)

    assert not cfg.retrieval_external_only
    assert cfg.retrieval_mode == "RETR_TREE"


def test_full_hierarchy_rejects_dark_metal_interior_but_keeps_true_hole() -> None:
    segmentation = np.zeros((140, 260), dtype=np.uint8)
    source = np.full_like(segmentation, 20)
    for x_start in (20, 140):
        segmentation[20:120, x_start : x_start + 100] = 240
        segmentation[45:95, x_start + 25 : x_start + 75] = 0
        source[20:120, x_start : x_start + 100] = 220
    source[45:95, 45:95] = 20
    source[45:95, 165:215] = 170
    config = MetalRecoveryConfig(
        segmentation_strategy="legacy_otsu",
        min_width_px=2.0,
        min_area=20.0,
        min_perimeter=20.0,
        gap_bridge_px=0,
        min_polygon_angle_deg=0.0,
        min_inner_hole_area=20.0,
    )

    result = detect_metalization(segmentation, config, source_image=source)

    holes = [polygon for polygon in result.accepted if polygon.is_hole]
    assert len(holes) == 1
    assert holes[0].bbox[0] < 100


def test_hole_source_contrast_settings_change_accepted_holes() -> None:
    segmentation = np.zeros((140, 140), dtype=np.uint8)
    source = np.full_like(segmentation, 20)
    segmentation[20:120, 20:120] = 240
    segmentation[45:95, 45:95] = 0
    source[20:120, 20:120] = 220
    source[45:95, 45:95] = 20
    base = dict(
        segmentation_strategy="legacy_otsu",
        min_width_px=2.0,
        min_area=20.0,
        min_perimeter=20.0,
        gap_bridge_px=0,
        min_polygon_angle_deg=0.0,
        min_inner_hole_area=20.0,
    )

    strict = detect_metalization(
        segmentation,
        MetalRecoveryConfig(**base, min_hole_source_contrast=250.0),
        source_image=source,
    )
    permissive = detect_metalization(
        segmentation,
        MetalRecoveryConfig(**base, min_hole_source_contrast=8.0),
        source_image=source,
    )

    assert sum(polygon.is_hole for polygon in strict.accepted) == 0
    assert sum(polygon.is_hole for polygon in permissive.accepted) == 1


def test_hole_source_contrast_settings_reach_recovery_config() -> None:
    settings = ContourExtractionSettings(
        metal_min_hole_source_contrast=12.5,
        metal_min_hole_source_contrast_fraction=0.2,
    )

    config = metal_recovery_config_from_settings(settings)

    assert config.min_hole_source_contrast == pytest.approx(12.5)
    assert config.min_hole_source_contrast_fraction == pytest.approx(0.2)


def test_metal_recovery_config_preserves_zero_gap_bridge() -> None:
    settings = ContourExtractionSettings(metal_gap_bridge_px=0, metal_speckle_removal_px=3)
    cfg = metal_recovery_config_from_settings(settings)
    assert cfg.gap_bridge_px == 0
    assert cfg.speckle_removal_px == 3


def test_contour_extraction_settings_map_to_extract_polygons() -> None:
    cfg = MetalRecoveryConfig(
        epsilon_simplify=1.5,
        min_area=50.0,
        min_perimeter=20.0,
        min_width_px=4.0,
        retrieval_external_only=True,
        approximation_enabled=True,
    )
    settings = contour_extraction_settings_from_metal(cfg)
    assert settings.epsilon == pytest.approx(1.5)
    assert settings.retrieval_mode == "RETR_EXTERNAL"
    assert settings.min_polygon_width_px == pytest.approx(4.0)


def test_bowtie_ring_repaired_in_extract_polygons() -> None:
    import cv2

    raw = np.array([[[0, 0]], [[20, 0]], [[20, 20]], [[0, 20]]], dtype=np.int32)
    bad_ring = [(0.0, 0.0), (20.0, 20.0), (0.0, 20.0), (20.0, 0.0)]
    mask = np.zeros((22, 22), dtype=np.uint8)
    cv2.drawContours(mask, [raw], -1, 255, thickness=-1)
    polygons = extract_polygons(
        mask,
        ContourExtractionSettings(
            epsilon=0.0,
            min_polygon_angle=0.0,
            object_type="conductor",
            output_mode="polygon",
        ),
    )
    assert polygons
    assert is_valid_closed_polygon_ring(polygons[0].points)



def test_min_length_filters_short_traces() -> None:
    img = np.zeros((80, 120), dtype=np.uint8)
    img[38:42, 10:70] = 235
    short = detect_metalization(
        img,
        MetalRecoveryConfig(min_length_px=80.0, min_area=5.0, min_perimeter=5.0, min_width_px=1.0),
    )
    long_ok = detect_metalization(
        img,
        MetalRecoveryConfig(min_length_px=20.0, min_area=5.0, min_perimeter=5.0, min_width_px=1.0),
    )
    assert not [p for p in short.accepted if not p.is_hole]
    assert [p for p in long_ok.accepted if not p.is_hole]


def test_max_area_rejects_large_blob() -> None:
    img = np.zeros((100, 100), dtype=np.uint8)
    img[30:70, 30:70] = 250
    cfg = MetalRecoveryConfig(
        segmentation_strategy="legacy_otsu",
        min_width_px=2.0,
        max_area=500.0,
        min_area=50.0,
        min_perimeter=30.0,
        gap_bridge_px=2,
    )
    result = detect_metalization(img, cfg)
    conductors = [polygon for polygon in result.accepted if polygon.category == "conductor" and not polygon.is_hole]
    assert not conductors


def _noisy_sem_synthetic() -> np.ndarray:
    rng = np.random.default_rng(42)
    img = np.zeros((160, 200), dtype=np.uint8)
    img[60:68, 30:170] = 210
    noise = rng.integers(0, 55, size=img.shape, dtype=np.uint8)
    return np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)


def test_minimum_contrast_recovers_otsu_omissions_at_or_above_threshold() -> None:
    from contour.vision.metal_recovery.pipeline_stages import _apply_minimum_contrast
    from contour.vision.schemas import SemPolarity

    gray = np.array([[100, 100, 130, 170]], dtype=np.uint8)
    raw_mask = np.array([[0, 0, 0, 255]], dtype=np.uint8)

    filtered = _apply_minimum_contrast(
        gray,
        raw_mask,
        polarity=SemPolarity.BRIGHT_FOREGROUND,
        min_contrast=30.0,
    )

    assert filtered.tolist() == [[0, 0, 255, 255]]


def test_minimum_contrast_recovers_textured_low_contrast_region() -> None:
    from contour.vision.metal_recovery.pipeline_stages import _apply_minimum_contrast
    from contour.vision.schemas import SemPolarity

    gray = np.full((80, 120), 100, dtype=np.uint8)
    texture = np.indices((40, 80)).sum(axis=0) % 2
    gray[20:60, 20:100] = np.where(texture == 0, 98, 108)
    raw_mask = np.zeros_like(gray)
    raw_mask[20:60, 99] = 255

    filtered = _apply_minimum_contrast(
        gray,
        raw_mask,
        polarity=SemPolarity.BRIGHT_FOREGROUND,
        min_contrast=2.0,
    )

    assert np.count_nonzero(filtered[24:56, 24:96]) / (32 * 72) > 0.95


def test_minimum_contrast_discards_unseeded_noise_components() -> None:
    from contour.vision.metal_recovery.pipeline_stages import _retain_seeded_contrast_components

    candidates = np.zeros((30, 60), dtype=np.uint8)
    candidates[5:20, 5:25] = 255
    candidates[5:20, 35:55] = 255
    raw_mask = np.zeros_like(candidates)
    raw_mask[10, 10:13] = 255

    filtered = _retain_seeded_contrast_components(candidates, raw_mask)

    assert np.all(filtered[5:20, 5:25] == 255)
    assert np.all(filtered[5:20, 35:55] == 0)


def test_lower_minimum_contrast_grows_mask_even_when_fragments_merge() -> None:
    import cv2

    from contour.vision.metal_recovery.pipeline_stages import _apply_minimum_contrast
    from contour.vision.schemas import SemPolarity

    gray = np.full((40, 80), 100, dtype=np.uint8)
    gray[10:30, 10:30] = 170
    gray[10:30, 50:70] = 170
    gray[18:22, 30:50] = 120
    raw_mask = np.zeros_like(gray)
    raw_mask[10:30, 10:30] = 255
    raw_mask[10:30, 50:70] = 255

    strict = _apply_minimum_contrast(
        gray,
        raw_mask,
        polarity=SemPolarity.BRIGHT_FOREGROUND,
        min_contrast=50.0,
    )
    permissive = _apply_minimum_contrast(
        gray,
        raw_mask,
        polarity=SemPolarity.BRIGHT_FOREGROUND,
        min_contrast=10.0,
    )

    strict_components, _ = cv2.connectedComponents((strict > 0).astype(np.uint8), connectivity=8)
    permissive_components, _ = cv2.connectedComponents(
        (permissive > 0).astype(np.uint8), connectivity=8
    )
    assert np.count_nonzero(permissive) > np.count_nonzero(strict)
    assert permissive_components - 1 < strict_components - 1


def test_minimum_contrast_makes_recognition_mask_monotonically_smaller() -> None:
    from contour.vision.metal_recovery.pipeline_stages import clear_metal_segmentation_cache

    clear_metal_segmentation_cache()
    img = _noisy_sem_synthetic()
    areas: list[int] = []
    for min_contrast in (1, 180, 230):
        cfg = MetalRecoveryConfig(
            min_contrast=float(min_contrast),
            gap_bridge_px=4,
            speckle_removal_px=2,
            min_width_px=3.0,
        )
        mask, _ = build_metal_extraction_mask(img, cfg)
        areas.append(int(np.count_nonzero(mask)))
    assert areas[0] > 0
    assert areas[0] >= areas[1] >= areas[2]
    assert areas[0] > areas[2]


def test_topology_stage_reuses_otsu_cache() -> None:
    from contour.vision.metal_recovery.pipeline_stages import (
        _SEGMENTATION_CACHE,
        clear_metal_segmentation_cache,
    )

    clear_metal_segmentation_cache()
    img = _noisy_sem_synthetic()
    base = MetalRecoveryConfig(min_contrast=1.0, gap_bridge_px=1, speckle_removal_px=0, min_width_px=3.0)
    build_metal_extraction_mask(img, base)
    sig = next(iter(_SEGMENTATION_CACHE))
    entry = _SEGMENTATION_CACHE[sig]
    raw_before = entry.raw_segmentation.copy()
    build_metal_extraction_mask(
        img,
        MetalRecoveryConfig(min_contrast=1.0, gap_bridge_px=5, speckle_removal_px=0, min_width_px=3.0),
    )
    assert entry.raw_segmentation is not None
    assert np.array_equal(entry.raw_segmentation, raw_before)
    assert entry.gap_bridge_px == 5


def test_noisy_sem_preset_finds_trace_on_synthetic() -> None:
    settings = ContourExtractionSettings.from_dict(noisy_sem_metal_preset_payload())
    cfg = metal_recovery_config_from_settings(settings)
    cfg = MetalRecoveryConfig(
        **{**cfg.to_snapshot(), "min_width_px": 4.0, "min_area": 30.0}
    )
    result = detect_metalization(_noisy_sem_synthetic(), cfg)
    assert result.accepted


def test_legacy_settings_migration_in_contour_settings() -> None:
    settings = ContourExtractionSettings.from_dict(
        {"metal_sensitivity_0_100": 78, "metal_sensitivity": "low", "metal_segmentation_method": "otsu"}
    )
    assert hasattr(settings, "metal_min_contrast")
    assert settings.metal_segmentation_strategy == "legacy_otsu"


def test_legacy_positive_contrast_bias_migrates_to_minimum_contrast() -> None:
    settings = ContourExtractionSettings.from_dict({"metal_contrast_bias": 25.0})

    config = metal_recovery_config_from_settings(settings)

    assert settings.metal_min_contrast == pytest.approx(25.0)
    assert config.min_contrast == pytest.approx(25.0)


def test_conductor_minimum_contrast_defaults_to_fifty_and_rejects_zero() -> None:
    defaults = ContourExtractionSettings()
    restored = ContourExtractionSettings.from_dict({"metal_min_contrast": 0.0})

    assert defaults.metal_min_contrast == pytest.approx(50.0)
    assert standard_metal_preset_payload()["metal_min_contrast"] == pytest.approx(50.0)
    assert noisy_sem_metal_preset_payload()["metal_min_contrast"] == pytest.approx(50.0)
    assert restored.metal_min_contrast == pytest.approx(1.0)
    assert metal_recovery_config_from_settings(restored).min_contrast == pytest.approx(1.0)
