from __future__ import annotations

import cv2
import numpy as np
import pytest

from contour.application.processing import ContourExtractionSettings, normalize_algorithm_backend
from contour.vision.via_detection.config import HeuristicViaDetectorConfig, ViaPolarity
from contour.vision.via_detection.heuristic_detector import (
    CandidateFeatures,
    _final_candidate_score,
    detect_vias_heuristic,
)
from contour.vision.via_detection.settings_bridge import heuristic_config_from_settings


@pytest.mark.parametrize(
    "polarity",
    [
        ViaPolarity.BRIGHT,
        ViaPolarity.DARK,
        ViaPolarity.RING_LIGHT_RING,
        ViaPolarity.RING_DARK_RING,
    ],
)
@pytest.mark.parametrize("white_enabled,black_enabled", [(True, False), (False, True), (True, True), (False, False)])
def test_explicit_polarity_is_never_overridden_by_ranges(
    polarity: ViaPolarity,
    white_enabled: bool,
    black_enabled: bool,
) -> None:
    settings = ContourExtractionSettings(
        via_heuristic_polarity=str(polarity),
        via_white_range_enabled=white_enabled,
        via_black_range_enabled=black_enabled,
    )
    assert heuristic_config_from_settings(settings).polarity == str(polarity)


@pytest.mark.parametrize(
    ("polarity", "center", "ring", "diameter"),
    [
        ("bright", 230, None, 10),
        ("dark", 25, None, 10),
        ("ring_light_ring", 25, 230, 14),
        ("ring_dark_ring", 230, 25, 14),
    ],
)
def test_all_polarity_shapes_are_detected(
    polarity: str,
    center: int,
    ring: int | None,
    diameter: int,
) -> None:
    image = np.full((96, 96), 128, dtype=np.uint8)
    if ring is None:
        cv2.circle(image, (48, 48), diameter // 2, center, thickness=-1)
    else:
        cv2.circle(image, (48, 48), diameter // 2, ring, thickness=-1)
        cv2.circle(image, (48, 48), 3, center, thickness=-1)
    config = HeuristicViaDetectorConfig(
        diameter_mode="fixed",
        fixed_diameters=[diameter],
        polarity=polarity,
        min_final_score=0.0,
        min_center_contrast=1.0,
        min_peak_prominence=1.0,
        min_compactness=0.01,
        bright_range_enabled=False,
        dark_range_enabled=False,
        max_line_coherence=1.0,
        min_edge_sharpness=0.0,
    )
    result = detect_vias_heuristic(image, config)
    assert len(result.accepted) == 1
    assert result.accepted[0].x == pytest.approx(48.0, abs=0.2)
    assert result.accepted[0].y == pytest.approx(48.0, abs=0.2)
    assert {
        "center_brightness",
        "contrast",
        "prominence",
        "equivalent_diameter",
        "center_drift",
        "compactness",
        "circularity",
        "aspect",
        "line_coherence",
        "edge_snr",
        "edge_sharpness",
        "border_imbalance",
        "line_likeness",
        "binarization_threshold",
        "contribution_contrast",
        "contribution_compactness",
        "penalty_line",
        "penalty_border",
        "final_score",
    } <= result.accepted[0].features.keys()
    assert result.accepted[0].features["center_brightness"] == pytest.approx(center, abs=5.0)


def test_minimum_center_brightness_rejects_dimmer_candidate() -> None:
    image = np.full((96, 96), 100, dtype=np.uint8)
    cv2.circle(image, (48, 48), 5, 220, thickness=-1)
    config = HeuristicViaDetectorConfig(
        diameter_mode="fixed",
        fixed_diameters=[10],
        polarity="bright",
        min_final_score=0.0,
        min_center_brightness=230.0,
        min_center_contrast=1.0,
        min_peak_prominence=1.0,
        min_compactness=0.01,
        bright_range_enabled=False,
        dark_range_enabled=False,
        max_line_coherence=1.0,
        min_edge_sharpness=0.0,
    )

    result = detect_vias_heuristic(image, config)

    assert result.accepted == []
    assert any(
        item.reject_reason == "hard:low_center_brightness"
        for item in result.rejected
    )
    assert result.parameters_snapshot["min_center_brightness"] == pytest.approx(230.0)


def test_score_weights_are_configurable() -> None:
    features = CandidateFeatures(
        center_x=10.0,
        center_y=10.0,
        diameter=8.0,
        contrast=20.0,
        prominence=0.0,
        compactness=0.0,
        circularity=0.0,
        aspect=1.0,
        line_coherence=0.0,
        edge_snr=1.0,
        edge_sharpness=1.0,
        border_imbalance=0.0,
    )
    only_contrast = HeuristicViaDetectorConfig(
        w_contrast=30.0,
        w_prominence=0.0,
        w_size=0.0,
        w_compact=0.0,
        w_round=0.0,
        w_balance=0.0,
        w_line=0.0,
        w_border=0.0,
    )
    no_weights = HeuristicViaDetectorConfig(
        w_contrast=0.0,
        w_prominence=0.0,
        w_size=0.0,
        w_compact=0.0,
        w_round=0.0,
        w_balance=0.0,
        w_line=0.0,
        w_border=0.0,
    )
    assert _final_candidate_score(features, only_contrast, [8], 0.0) == pytest.approx(30.0)
    assert _final_candidate_score(features, no_weights, [8], 0.0) == 0.0


def test_full_frame_sobel_is_computed_once_per_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.full((160, 160), 80, dtype=np.uint8)
    for y in range(20, 141, 20):
        for x in range(20, 141, 20):
            cv2.circle(image, (x, y), 4, 230, thickness=-1)
    calls = 0
    original = cv2.Sobel

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "Sobel", counted)
    detect_vias_heuristic(
        image,
        HeuristicViaDetectorConfig(
            diameter_mode="fixed",
            fixed_diameters=[8],
            polarity="bright",
            bright_range_min=140.0,
        ),
    )
    assert calls == 2


def test_legacy_via_backend_token_is_migrated_and_not_serialized() -> None:
    settings = ContourExtractionSettings.from_dict({"algorithm_backend": "legacy_via"})
    assert normalize_algorithm_backend("legacy_via") == "sem"
    assert settings.algorithm_backend == "sem"
    assert settings.to_dict()["algorithm_backend"] == "sem"


def test_invalid_expert_min_max_pair_is_rejected() -> None:
    config = HeuristicViaDetectorConfig(contrast_score_min=20.0, contrast_score_max=3.0)
    with pytest.raises(ValueError, match="contrast score"):
        config.validate()


def test_minimum_circularity_rejects_nonconforming_candidate() -> None:
    image = np.full((96, 96), 40, dtype=np.uint8)
    cv2.ellipse(image, (48, 48), (5, 1), 0.0, 0.0, 360.0, 230, thickness=-1)
    result = detect_vias_heuristic(
        image,
        HeuristicViaDetectorConfig(
            diameter_mode="fixed",
            fixed_diameters=[10],
            polarity="bright",
            min_final_score=0.0,
            min_center_contrast=1.0,
            min_peak_prominence=1.0,
            min_compactness=0.0,
            min_circularity=0.99,
            size_tolerance_ratio_fixed=0.95,
            max_elongation=20.0,
            max_line_coherence=1.0,
            min_edge_sharpness=0.0,
            bright_range_enabled=False,
        ),
    )

    assert result.accepted == []
    assert any(item.reject_reason == "hard:circularity" for item in result.rejected)
    assert all(item.bbox[2] > 0 and item.bbox[3] > 0 for item in result.rejected)


def test_expert_settings_round_trip() -> None:
    settings = ContourExtractionSettings(
        heuristic_min_center_brightness=121.0,
        heuristic_min_circularity=0.58,
        heuristic_max_line_coherence=0.71,
        heuristic_min_edge_sharpness=0.44,
        heuristic_contrast_score_min=4.0,
        heuristic_contrast_score_max=31.0,
        heuristic_w_contrast=33.0,
        heuristic_w_border=17.0,
        heuristic_seed_percentile=91.4,
    )
    restored = ContourExtractionSettings.from_dict(settings.to_dict())
    assert restored.heuristic_min_center_brightness == pytest.approx(121.0)
    assert restored.heuristic_min_circularity == pytest.approx(0.58)
    assert restored.heuristic_max_line_coherence == pytest.approx(0.71)
    assert restored.heuristic_min_edge_sharpness == pytest.approx(0.44)
    assert restored.heuristic_contrast_score_min == pytest.approx(4.0)
    assert restored.heuristic_contrast_score_max == pytest.approx(31.0)
    assert restored.heuristic_w_contrast == pytest.approx(33.0)
    assert restored.heuristic_w_border == pytest.approx(17.0)
    assert restored.heuristic_seed_percentile == pytest.approx(91.4)
    assert "heuristic_seed_percentile_low" not in restored.to_dict()
    assert "heuristic_seed_percentile_medium" not in restored.to_dict()
    assert "heuristic_seed_percentile_high" not in restored.to_dict()
    assert "via_search_sensitivity" not in restored.to_dict()


def test_old_sensitivity_fields_migrate_to_single_peak_threshold() -> None:
    restored = ContourExtractionSettings.from_dict(
        {
            "via_search_sensitivity": "high",
            "heuristic_seed_percentile_low": 99.2,
            "heuristic_seed_percentile_medium": 93.5,
            "heuristic_seed_percentile_high": 88.0,
        }
    )

    assert restored.heuristic_seed_percentile == pytest.approx(93.5)
    assert restored.heuristic_min_circularity == pytest.approx(0.40)
    payload = restored.to_dict()
    assert payload["heuristic_seed_percentile"] == pytest.approx(93.5)
    assert "via_search_sensitivity" not in payload
