"""benchmark_metal_segmentation: ui recovery config."""
from __future__ import annotations
from collections.abc import Iterable, Mapping
from pathlib import Path
import cv2
import numpy as np
from contour.application.processing import ContourExtractionSettings  # noqa: E402
from contour.domain import PolygonData  # noqa: E402
from contour.ui.metal_presets import standard_metal_preset_payload  # noqa: E402
from contour.vision.metal_recovery import (  # noqa: E402
    metal_recovery_config_from_settings,
)
from contour.vision.metal_recovery.gradient_watershed import (  # noqa: E402
    GradientWatershedConfig,
    build_conductor_seeds,
)

from benchmark_metal_segmentation_parts.rasterize_polygon_labels import _rasterize_polygon_labels

def resolution_probe(source_width: int = 2000, working_width: int = 640) -> list[dict[str, float | int | str]]:
    results: list[dict[str, float | int | str]] = []
    for feature in ("gap", "conductor"):
        background, foreground = (255, 0) if feature == "gap" else (0, 255)
        for width_px in range(1, 9):
            coverage: list[float] = []
            vanished = 0
            for phase in range(10):
                mask = np.full((32, source_width), background, np.uint8)
                start = source_width // 2 + phase
                mask[:, start : start + width_px] = foreground
                working_height = max(1, round(mask.shape[0] * working_width / source_width))
                reduced = cv2.resize(mask, (working_width, working_height), interpolation=cv2.INTER_NEAREST)
                restored = cv2.resize(reduced, (source_width, mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                retained = float(np.mean(restored[:, start : start + width_px] == foreground))
                coverage.append(retained)
                vanished += int(retained == 0.0)
            results.append(
                {
                    "feature": feature,
                    "width_px": width_px,
                    "mean_retained_fraction": float(np.mean(coverage)),
                    "vanished_phases_of_10": vanished,
                }
            )
    return results

def _write_diagnostic_overlay(
    output_path: Path,
    case: BenchmarkCase,
    predicted: np.ndarray,
    config: GradientWatershedConfig,
    *,
    check_presence: bool,
) -> None:
    expected = case.labels > 0
    seeds = build_conductor_seeds(
        case.image,
        config,
        check_presence=check_presence,
    )
    core = np.zeros(case.image.shape, dtype=np.uint8) if seeds is None else seeds.core_seeds
    groove = np.zeros(case.image.shape, dtype=np.uint8) if seeds is None else seeds.groove_seeds
    false_negative = expected & (predicted == 0)
    false_positive = ~expected & (predicted > 0)
    base = cv2.cvtColor(case.image, cv2.COLOR_GRAY2BGR)
    comparison = base.copy()
    comparison[expected] = (0, 170, 0)
    comparison[predicted > 0] = (0, 0, 220)
    seeds_overlay = base.copy()
    seeds_overlay[core > 0] = (255, 0, 255)
    seeds_overlay[groove > 0] = (255, 255, 0)
    errors = base.copy()
    errors[false_negative] = (255, 0, 0)
    errors[false_positive] = (0, 0, 255)
    panel = np.hstack((base, comparison, seeds_overlay, errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), panel):
        raise OSError(f"Could not write diagnostic overlay: {output_path}")

def _rasterize_detected_polygons(
    polygons: Iterable[PolygonData],
    shape: tuple[int, int],
) -> np.ndarray:
    return _rasterize_polygon_labels(polygons, shape)

def _ui_recovery_config(
    strategy: str,
    config: GradientWatershedConfig,
    strategy_parameter_overrides: Mapping[str, object] | None = None,
):
    base = ContourExtractionSettings(
        recognition_mode="conductors",
        extraction_profile="conductors",
        object_type="conductor",
        output_mode="polygon",
        algorithm_backend="legacy",
        metal_structural_pipeline=True,
    )
    settings = ContourExtractionSettings.from_dict(
        {
            **base.to_dict(),
            **standard_metal_preset_payload(),
            "metal_segmentation_strategy": strategy,
            "metal_use_wide_conductor_gradient": strategy == "gradient_watershed",
            "metal_watershed_smoothing_sigma": config.smoothing_sigma,
            "metal_watershed_core_margin": config.core_margin,
            "metal_watershed_groove_margin": config.groove_margin,
            "metal_watershed_rim_probe_px": config.rim_probe_px,
            "metal_watershed_seed_speckle_px": config.seed_speckle_px,
            "metal_watershed_valley_span_px": config.valley_span_px,
            "metal_watershed_valley_depth": config.valley_depth,
            "metal_random_walker_beta": config.random_walker_beta,
            "metal_random_walker_iterations": config.random_walker_iterations,
            "metal_graph_cut_iterations": config.graph_cut_iterations,
            "metal_reconstruction_erode_px": config.reconstruction_erode_px,
            "metal_boundary_relief": config.boundary_relief,
            "metal_boundary_background_sigma": config.boundary_background_sigma,
            **(
                {"metal_strategy_parameters": {strategy: dict(strategy_parameter_overrides)}}
                if strategy_parameter_overrides
                else {}
            ),
        }
    )
    return metal_recovery_config_from_settings(settings)
