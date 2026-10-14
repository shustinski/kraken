"""benchmark_metal_segmentation: seed diagnostics."""
from __future__ import annotations
import cv2
import numpy as np
from contour.vision.metal_recovery.gradient_watershed import (  # noqa: E402
    GradientWatershedConfig,
    analyze_metal_presence,
    build_conductor_seeds,
    intensity_class_limits,
)

from benchmark_metal_segmentation_parts.render_sem import SeedDiagnostics

def seed_diagnostics(
    image: np.ndarray,
    config: GradientWatershedConfig,
    *,
    check_presence: bool = True,
) -> SeedDiagnostics:
    smoothed = cv2.GaussianBlur(
        image,
        (0, 0),
        max(0.1, float(config.smoothing_sigma)),
    )
    substrate_limit, metal_limit = intensity_class_limits(smoothed)
    seeds = build_conductor_seeds(image, config, check_presence=check_presence)
    if seeds is None:
        core = np.zeros(image.shape, dtype=np.uint8)
        groove = np.zeros(image.shape, dtype=np.uint8)
    else:
        core = seeds.core_seeds
        groove = seeds.groove_seeds
    core_count, _core_labels = cv2.connectedComponents(
        (core > 0).astype(np.uint8),
        connectivity=8,
    )
    groove_count, _groove_labels = cv2.connectedComponents(
        (groove > 0).astype(np.uint8),
        connectivity=8,
    )
    gradient_x = cv2.Scharr(smoothed, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(smoothed, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    local_contrast = cv2.absdiff(
        smoothed,
        cv2.GaussianBlur(smoothed, (0, 0), 6.0),
    )
    intensity_percentiles = np.percentile(smoothed, (1, 10, 50, 90, 99))
    gradient_percentiles = np.percentile(gradient, (90, 99))
    contrast_percentiles = np.percentile(local_contrast, (90, 99))
    presence = analyze_metal_presence(
        image,
        smoothing_sigma=float(config.smoothing_sigma),
    )
    return SeedDiagnostics(
        substrate_intensity_limit=float(substrate_limit),
        metal_intensity_limit=float(metal_limit),
        core_seed_fraction=float(np.mean(core > 0)),
        groove_seed_fraction=float(np.mean(groove > 0)),
        core_connected_components=core_count - 1,
        groove_connected_components=groove_count - 1,
        intensity_p01=float(intensity_percentiles[0]),
        intensity_p10=float(intensity_percentiles[1]),
        intensity_p50=float(intensity_percentiles[2]),
        intensity_p90=float(intensity_percentiles[3]),
        intensity_p99=float(intensity_percentiles[4]),
        gradient_mean=float(np.mean(gradient)),
        gradient_p90=float(gradient_percentiles[0]),
        gradient_p99=float(gradient_percentiles[1]),
        local_contrast_mean=float(np.mean(local_contrast)),
        local_contrast_p90=float(contrast_percentiles[0]),
        local_contrast_p99=float(contrast_percentiles[1]),
        presence_has_metal=presence.has_metal,
        presence_robust_intensity_span=presence.robust_intensity_span,
        presence_coherent_contrast_fraction=presence.coherent_contrast_fraction,
        presence_largest_coherent_contrast_fraction=(presence.largest_coherent_contrast_fraction),
        presence_local_contrast_limit=presence.local_contrast_limit,
    )
