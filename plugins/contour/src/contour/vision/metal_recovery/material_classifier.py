"""Deterministic metal/background classification for partition-based methods."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .features import MetalStructuralFeatures, float_map_to_u8


@dataclass(frozen=True, slots=True)
class MaterialClassificationResult:
    mask: np.ndarray
    instance_labels: np.ndarray
    confidence_map: np.ndarray
    region_confidence: np.ndarray
    debug_images: dict[str, np.ndarray]


def _region_mean(values: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    sums = np.bincount(labels.ravel(), weights=values.ravel(), minlength=count)
    areas = np.bincount(labels.ravel(), minlength=count).astype(np.float64)
    return np.divide(sums, np.maximum(areas, 1.0))


def classify_partition_material(
    labels: np.ndarray,
    features: MetalStructuralFeatures,
    parameters: Mapping[str, Any],
) -> MaterialClassificationResult:
    region_labels = np.maximum(labels.astype(np.int32, copy=False), 0)
    count = int(region_labels.max()) + 1 if region_labels.size else 1
    areas = np.bincount(region_labels.ravel(), minlength=count)
    core_mean = _region_mean(features.core_evidence, region_labels, count)
    substrate_mean = _region_mean(features.substrate_evidence, region_labels, count)
    contrast = _region_mean(features.local_contrast, region_labels, count)

    gray = features.denoised.astype(np.float32) / 255.0
    intensity_mean = _region_mean(gray, region_labels, count)
    global_median = float(np.median(gray)) if gray.size else 0.5
    intensity_distance = np.clip(np.abs(intensity_mean - global_median) * 2.0, 0.0, 1.0)

    boundary_pixels = cv2.morphologyEx(
        region_labels.astype(np.float32),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    )
    boundary_mask = (boundary_pixels != 0).astype(np.float32)
    boundary_side = _region_mean(
        features.local_contrast * boundary_mask,
        region_labels,
        count,
    )
    boundary_fraction = _region_mean(boundary_mask, region_labels, count)
    boundary_side = np.divide(boundary_side, np.maximum(boundary_fraction, 1e-3))

    seed_total = core_mean + substrate_mean
    core_signal = np.where(
        seed_total >= 1e-3,
        np.divide(core_mean, np.maximum(seed_total, 1e-6)),
        0.0,
    )
    substrate_signal = np.where(
        seed_total >= 1e-3,
        np.divide(substrate_mean, np.maximum(seed_total, 1e-6)),
        0.0,
    )
    positive = (
        float(parameters.get("core_evidence_weight", 1.0)) * core_signal
        + float(parameters.get("intensity_evidence_weight", 0.45)) * intensity_distance
        + float(parameters.get("local_contrast_evidence_weight", 0.8)) * contrast
        + float(parameters.get("boundary_side_evidence_weight", 0.6)) * boundary_side
    )
    negative = float(parameters.get("substrate_evidence_weight", 1.0)) * substrate_signal
    confidence = np.divide(positive, np.maximum(positive + negative, 1e-6))
    # High-confidence seed evidence is a prior, not just another weak feature.
    # This prevents a large calm substrate region from becoming metal merely
    # because its perimeter contains strong SEM contrast, while still allowing
    # a dark conductor interior to inherit reliable core support from its rim.
    confidence *= 1.0 - 0.85 * substrate_signal * (1.0 - core_signal)
    confidence = np.maximum(
        confidence,
        0.8 * core_signal * (1.0 - substrate_signal),
    )
    # No reliable signal means background, not a fabricated 50% decision.
    confidence[(positive + negative) <= 1e-6] = 0.0
    confidence[0] = 0.0

    metal_threshold = float(parameters.get("minimum_metal_confidence", 0.52))
    background_threshold = min(
        metal_threshold,
        float(parameters.get("minimum_background_confidence", 0.48)),
    )
    metal_regions = confidence >= metal_threshold
    ambiguous = (confidence > background_threshold) & ~metal_regions
    policy = str(parameters.get("ambiguous_region_policy", "background"))
    if policy == "metal":
        metal_regions |= ambiguous
    elif policy == "preserve":
        metal_regions |= ambiguous & (core_signal >= 0.5)
    metal_regions[0] = False
    metal_regions[areas <= 0] = False

    selected = np.where(metal_regions[region_labels], region_labels, 0).astype(np.int32)
    unique = np.unique(selected)
    unique = unique[unique > 0]
    remap = np.zeros(count, dtype=np.int32)
    remap[unique] = np.arange(1, unique.size + 1, dtype=np.int32)
    instances = remap[selected]
    mask = np.where(instances > 0, 255, 0).astype(np.uint8)
    confidence_map = confidence[region_labels].astype(np.float32)
    return MaterialClassificationResult(
        mask=mask,
        instance_labels=instances,
        confidence_map=confidence_map,
        region_confidence=confidence.astype(np.float32),
        debug_images={
            "metal_material_confidence": float_map_to_u8(confidence_map),
            "metal_material_core_evidence": float_map_to_u8(features.core_evidence),
            "metal_material_substrate_evidence": float_map_to_u8(features.substrate_evidence),
        },
    )


__all__ = ["MaterialClassificationResult", "classify_partition_material"]
