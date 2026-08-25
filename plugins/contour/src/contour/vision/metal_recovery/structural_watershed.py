"""Marker-controlled structural segmentation of SEM metallization.

The algorithm extracts interior, ridge/centerline, and boundary evidence from
the SEM frame itself, then partitions with a marker-controlled watershed.  It
does not read ground truth, frame ids, or filenames.

Ablation variants:

* ``s1`` — existing core/groove seeds, binary labels, gradient-magnitude terrain
* ``s2`` — ridge + wide-interior instance markers, gradient-magnitude terrain
* ``s3`` — same markers as ``s2``, orientation-aware boundary-cost terrain
* ``s4`` — ``s3`` with denser multiscale ridge sampling
* ``s5`` — S2 markers, multi-label watershed, instance identity kept
* ``s6`` — S3 markers/cost, multi-label watershed, instance identity kept
* ``s7`` — S2 markers, multi-source geodesic competition, instance identity kept
* ``s8`` — S7 propagation + ridge same-trace linking
* ``s9`` — S7 propagation + wide-region consolidation
* ``s10`` — S7 propagation + ridge linking + wide consolidation
* ``s11`` — S7 + wide combine by internal separator, not ridge count
* ``s12`` — S7 + orientation-aware corridor veto
* ``s13`` — S7 + transverse conductor-band grouping
* ``s14`` — S7 + separator combine + orientation-aware linking + band grouping
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ...utils import ensure_binary_mask, ensure_uint8
from .gradient_watershed import (
    GradientWatershedConfig,
    analyze_metal_presence,
    build_conductor_seeds,
)
from .marker_consolidation import (
    ConsolidationEvidence,
    MarkerConsolidationStats,
    consolidate_markers,
)

STRUCTURAL_WATERSHED_STRATEGY = "structural_watershed"
STRUCTURAL_VARIANTS = (
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14",
)
INSTANCE_IDENTITY_VARIANTS = frozenset(
    {"s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14"}
)
GEODESIC_COMPETITION_VARIANTS = frozenset(
    {"s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14"}
)
RIDGE_LINK_VARIANTS = frozenset({"s8", "s10", "s12", "s14"})
WIDE_CONSOLIDATE_VARIANTS = frozenset({"s9", "s10", "s11", "s14"})
BAND_GROUP_VARIANTS = frozenset({"s13", "s14"})
ORIENTATION_VETO_VARIANTS = frozenset({"s12", "s14"})
SEPARATOR_COMBINE_VARIANTS = frozenset({"s11", "s14"})
_FRANGI_BETA = 0.5
_RIDGE_SCALE_STEP = 1.6
_DENSE_RIDGE_SCALE_STEP = 1.35
_RIBBON_HALF_WIDTHS = (2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 14.0)
_NMS_DIRECTIONS = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
)


def normalize_structural_variant(value: Any) -> str:
    text = str(value or "s2").strip().lower().replace("-", "_")
    aliases = {
        "s1": "s1",
        "existing_markers_gradient": "s1",
        "s2": "s2",
        "ridge_interior_gradient": "s2",
        "s3": "s3",
        "ridge_interior_oriented": "s3",
        "s4": "s4",
        "multiscale_oriented": "s4",
        "s5": "s5",
        "multilabel_gradient": "s5",
        "s6": "s6",
        "multilabel_oriented": "s6",
        "s7": "s7",
        "geodesic_competition": "s7",
        "s8": "s8",
        "ridge_link_geodesic": "s8",
        "s9": "s9",
        "wide_consolidate_geodesic": "s9",
        "s10": "s10",
        "marker_consolidate_geodesic": "s10",
        "s11": "s11",
        "separator_combine_geodesic": "s11",
        "s12": "s12",
        "oriented_veto_geodesic": "s12",
        "s13": "s13",
        "conductor_band_geodesic": "s13",
        "s14": "s14",
        "band_identity_geodesic": "s14",
    }
    return aliases.get(text, "s2")


@dataclass(frozen=True, slots=True)
class StructuralWatershedConfig:
    """Physically named controls for structural marker watershed."""

    variant: str = "s2"
    smoothing_sigma: float = 1.0
    ridge_scale_min: float = 1.0
    ridge_scale_max: float = 6.0
    min_ridge_confidence: float = 0.22
    min_orientation_coherence: float = 0.45
    orientation_smoothing_scale: float = 3.0
    min_marker_length: int = 10
    min_marker_area: int = 16
    gap_link_px: int = 2
    wide_interior_radius: float = 4.0
    boundary_gradient_weight: float = 1.0
    boundary_continuity_weight: float = 0.6
    boundary_rim_weight: float = 0.8
    boundary_orientation_weight: float = 0.4


def clamped_structural_watershed_config(
    *,
    variant: str = "s2",
    smoothing_sigma: float = 1.0,
    ridge_scale_min: float = 1.0,
    ridge_scale_max: float = 6.0,
    min_ridge_confidence: float = 0.22,
    min_orientation_coherence: float = 0.45,
    orientation_smoothing_scale: float = 3.0,
    min_marker_length: int = 10,
    min_marker_area: int = 16,
    gap_link_px: int = 2,
    wide_interior_radius: float = 4.0,
    boundary_gradient_weight: float = 1.0,
    boundary_continuity_weight: float = 0.6,
    boundary_rim_weight: float = 0.8,
    boundary_orientation_weight: float = 0.4,
) -> StructuralWatershedConfig:
    scale_min = max(0.5, min(8.0, float(ridge_scale_min)))
    scale_max = max(scale_min, min(16.0, float(ridge_scale_max)))
    return StructuralWatershedConfig(
        variant=normalize_structural_variant(variant),
        smoothing_sigma=max(0.1, min(8.0, float(smoothing_sigma))),
        ridge_scale_min=scale_min,
        ridge_scale_max=scale_max,
        min_ridge_confidence=max(0.02, min(0.8, float(min_ridge_confidence))),
        min_orientation_coherence=max(0.0, min(1.0, float(min_orientation_coherence))),
        orientation_smoothing_scale=max(1.0, min(16.0, float(orientation_smoothing_scale))),
        min_marker_length=max(2, min(64, int(min_marker_length))),
        min_marker_area=max(2, min(256, int(min_marker_area))),
        gap_link_px=max(0, min(12, int(gap_link_px))),
        wide_interior_radius=max(1.0, min(24.0, float(wide_interior_radius))),
        boundary_gradient_weight=max(0.0, min(4.0, float(boundary_gradient_weight))),
        boundary_continuity_weight=max(0.0, min(4.0, float(boundary_continuity_weight))),
        boundary_rim_weight=max(0.0, min(4.0, float(boundary_rim_weight))),
        boundary_orientation_weight=max(0.0, min(4.0, float(boundary_orientation_weight))),
    )


def structural_watershed_config_from_object(source: object) -> StructuralWatershedConfig:
    return clamped_structural_watershed_config(
        variant=str(getattr(source, "structural_variant", "s2") or "s2"),
        smoothing_sigma=float(
            getattr(
                source,
                "watershed_smoothing_sigma",
                getattr(source, "smoothing_sigma", 1.0),
            )
            or 1.0
        ),
        ridge_scale_min=float(getattr(source, "structural_ridge_scale_min", 1.0) or 1.0),
        ridge_scale_max=float(getattr(source, "structural_ridge_scale_max", 6.0) or 6.0),
        min_ridge_confidence=float(getattr(source, "structural_min_ridge_confidence", 0.22) or 0.22),
        min_orientation_coherence=float(
            getattr(source, "structural_min_orientation_coherence", 0.45) or 0.45
        ),
        orientation_smoothing_scale=float(
            getattr(source, "structural_orientation_smoothing_scale", 3.0) or 3.0
        ),
        min_marker_length=int(getattr(source, "structural_min_marker_length", 10) or 10),
        min_marker_area=int(getattr(source, "structural_min_marker_area", 16) or 16),
        gap_link_px=int(getattr(source, "structural_gap_link_px", 2) or 0),
        wide_interior_radius=float(getattr(source, "structural_wide_interior_radius", 4.0) or 4.0),
        boundary_gradient_weight=float(
            getattr(source, "structural_boundary_gradient_weight", 1.0) or 0.0
        ),
        boundary_continuity_weight=float(
            getattr(source, "structural_boundary_continuity_weight", 0.6) or 0.0
        ),
        boundary_rim_weight=float(getattr(source, "structural_boundary_rim_weight", 0.8) or 0.0),
        boundary_orientation_weight=float(
            getattr(source, "structural_boundary_orientation_weight", 0.4) or 0.0
        ),
    )


@dataclass(frozen=True, slots=True)
class StructuralWatershedResult:
    mask: np.ndarray
    instance_labels: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int32))
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    instance_count: int = 0
    label_fragment_count: int = 0
    consolidation: MarkerConsolidationStats = field(default_factory=MarkerConsolidationStats)


def structural_watershed_mask(
    gray: np.ndarray,
    watershed_config: GradientWatershedConfig | None = None,
    structural_config: StructuralWatershedConfig | None = None,
    *,
    check_presence: bool = True,
) -> np.ndarray:
    return run_structural_watershed(
        gray,
        watershed_config,
        structural_config,
        check_presence=check_presence,
    ).mask


def run_structural_watershed(
    gray: np.ndarray,
    watershed_config: GradientWatershedConfig | None = None,
    structural_config: StructuralWatershedConfig | None = None,
    *,
    check_presence: bool = True,
) -> StructuralWatershedResult:
    source = ensure_uint8(gray)
    empty = np.zeros(source.shape[:2], dtype=np.uint8)
    if source.ndim != 2 or source.size == 0:
        empty_labels = np.zeros(source.shape[:2], dtype=np.int32)
        return StructuralWatershedResult(
            mask=empty,
            instance_labels=empty_labels,
            debug_images=_empty_debug(empty),
        )

    ws_config = watershed_config or GradientWatershedConfig()
    st_config = structural_config or clamped_structural_watershed_config(
        smoothing_sigma=float(ws_config.smoothing_sigma),
    )
    if check_presence:
        presence = analyze_metal_presence(source, smoothing_sigma=float(st_config.smoothing_sigma))
        if not presence.has_metal:
            empty_labels = np.zeros(source.shape[:2], dtype=np.int32)
            return StructuralWatershedResult(
                mask=empty,
                instance_labels=empty_labels,
                debug_images=_empty_debug(empty),
            )

    features = _extract_structural_features(source, st_config)
    seeds = build_conductor_seeds(source, ws_config, check_presence=False)
    core_seeds = empty if seeds is None else seeds.core_seeds
    groove_seeds = empty if seeds is None else seeds.groove_seeds
    metal_limit = 0.0 if seeds is None else float(seeds.metal_limit)

    keep_instances = st_config.variant in INSTANCE_IDENTITY_VARIANTS
    use_geodesic = st_config.variant in GEODESIC_COMPETITION_VARIANTS
    link_ridge = st_config.variant in RIDGE_LINK_VARIANTS
    link_wide = st_config.variant in WIDE_CONSOLIDATE_VARIANTS
    group_bands = st_config.variant in BAND_GROUP_VARIANTS
    orientation_veto = st_config.variant in ORIENTATION_VETO_VARIANTS
    separator_combine = st_config.variant in SEPARATOR_COMBINE_VARIANTS
    ridge_raw = empty
    consolidation = MarkerConsolidationStats()
    extra_debug: dict[str, np.ndarray] = {}
    if st_config.variant == "s1":
        mask, markers, cost, fg_markers, bg_markers, ridge_markers, wide_markers = (
            _segment_existing_markers(
                features,
                core_seeds,
                groove_seeds,
                metal_limit=metal_limit,
            )
        )
        instance_labels = _binary_components_as_labels(mask)
    else:
        ridge_raw, ridge_markers = _ridge_markers(features, st_config)
        wide_markers = _wide_interior_markers(features, core_seeds, st_config)
        fg_labels = None
        if link_ridge or link_wide or group_bands or separator_combine:
            consolidated = consolidate_markers(
                ridge_markers,
                wide_markers,
                ConsolidationEvidence(
                    intensity=features.denoised,
                    ridge_confidence=features.ridge_confidence,
                    ridge_orientation=features.ridge_orientation,
                    structure_orientation=features.structure_orientation,
                    coherence=features.coherence,
                    persistent_edge=features.persistent_edge,
                    magnitude=features.magnitude,
                    rim_response=features.rim_response,
                    gradient_x=features.gradient_x,
                    gradient_y=features.gradient_y,
                ),
                link_ridge=link_ridge,
                link_wide=link_wide,
                group_bands=group_bands,
                orientation_aware_veto=orientation_veto,
                separator_aware_combine=separator_combine,
                min_marker_area=int(st_config.min_marker_area),
                wide_interior_radius=float(st_config.wide_interior_radius),
            )
            fg_labels = consolidated.combined_labels
            fg_markers = np.where(fg_labels > 0, 255, 0).astype(np.uint8)
            consolidation = consolidated.stats
            extra_debug = dict(consolidated.debug_images)
            extra_debug["metal_structural_logical_markers_i32"] = fg_labels.astype(
                np.int32, copy=False
            )
        else:
            fg_markers = _combine_foreground_markers(ridge_markers, wide_markers)
        if keep_instances:
            bg_markers = _sparse_background_markers(features, fg_markers)
        else:
            bg_markers = _background_markers(
                features,
                groove_seeds,
                ridge_markers,
                fg_markers,
                st_config,
            )
        cost = _boundary_cost_map(features, st_config)
        if use_geodesic:
            markers = _geodesic_label_competition(
                fg_markers,
                bg_markers,
                cost,
                foreground_labels=fg_labels,
            )
            instance_labels = _finalize_instance_labels(markers)
            mask = np.where(instance_labels > 0, 255, 0).astype(np.uint8)
        else:
            mask, markers = _instance_watershed(
                fg_markers,
                bg_markers,
                cost,
                grow=keep_instances,
            )
            instance_labels = _finalize_instance_labels(markers)
            if keep_instances:
                mask = np.where(instance_labels > 0, 255, 0).astype(np.uint8)
                mask = _restore_fov_edge(mask)

    instance_count, label_fragment_count = _instance_fragment_stats(instance_labels)
    debug = _debug_images(
        features,
        ridge_raw=ridge_raw,
        ridge_markers=ridge_markers,
        wide_markers=wide_markers,
        fg_markers=fg_markers,
        bg_markers=bg_markers,
        cost=cost,
        watershed_labels=markers,
        instance_labels=instance_labels,
        mask=mask,
    )
    debug.update(extra_debug)
    return StructuralWatershedResult(
        mask=ensure_binary_mask(mask),
        instance_labels=instance_labels.astype(np.int32, copy=False),
        debug_images=debug,
        instance_count=instance_count,
        label_fragment_count=label_fragment_count,
        consolidation=consolidation,
    )


@dataclass(frozen=True, slots=True)
class _StructuralFeatures:
    denoised: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    magnitude: np.ndarray
    structure_orientation: np.ndarray
    coherence: np.ndarray
    ridge_response: np.ndarray
    ridge_orientation: np.ndarray
    ridge_confidence: np.ndarray
    oriented_gradient: np.ndarray
    rim_response: np.ndarray
    persistent_edge: np.ndarray


def _extract_structural_features(
    gray: np.ndarray,
    config: StructuralWatershedConfig,
) -> _StructuralFeatures:
    denoised = cv2.GaussianBlur(gray, (0, 0), max(0.1, float(config.smoothing_sigma)))
    gradient_x = cv2.Scharr(denoised, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(denoised, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    orientation, coherence = _structure_tensor_field(
        gradient_x,
        gradient_y,
        smoothing_sigma=float(config.orientation_smoothing_scale),
    )
    ridge_response, ridge_orientation, ridge_confidence = _multiscale_ridge_evidence(
        denoised,
        config,
    )
    across_x = np.cos(orientation)
    across_y = np.sin(orientation)
    oriented_gradient = np.abs(gradient_x * across_x + gradient_y * across_y)
    rim_response = _bright_rim_response(denoised, magnitude)
    persistent_edge = cv2.GaussianBlur(
        oriented_gradient,
        (0, 0),
        max(1.0, float(config.orientation_smoothing_scale) * 0.5),
    )
    return _StructuralFeatures(
        denoised=denoised,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        magnitude=magnitude,
        structure_orientation=orientation,
        coherence=coherence,
        ridge_response=ridge_response,
        ridge_orientation=ridge_orientation,
        ridge_confidence=ridge_confidence,
        oriented_gradient=oriented_gradient,
        rim_response=rim_response,
        persistent_edge=persistent_edge,
    )


def _structure_tensor_field(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    *,
    smoothing_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (edge orientation in radians, coherence) from the averaged structure tensor."""

    jxx = cv2.GaussianBlur(gradient_x * gradient_x, (0, 0), smoothing_sigma)
    jyy = cv2.GaussianBlur(gradient_y * gradient_y, (0, 0), smoothing_sigma)
    jxy = cv2.GaussianBlur(gradient_x * gradient_y, (0, 0), smoothing_sigma)
    trace = jxx + jyy
    delta = cv2.magnitude(jxx - jyy, 2.0 * jxy)
    lambda_max = 0.5 * (trace + delta)
    coherence = np.divide(delta, np.maximum(trace, 1e-6)).astype(np.float32)
    coherence = np.clip(coherence, 0.0, 1.0)
    orientation = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy).astype(np.float32)
    orientation = np.where(lambda_max <= 1e-6, 0.0, orientation).astype(np.float32)
    return orientation, coherence


def _ridge_scales(config: StructuralWatershedConfig) -> tuple[float, ...]:
    step = _DENSE_RIDGE_SCALE_STEP if config.variant == "s4" else _RIDGE_SCALE_STEP
    scale = float(config.ridge_scale_min)
    limit = float(config.ridge_scale_max)
    values: list[float] = []
    while scale <= limit + 1e-6:
        values.append(float(scale))
        scale *= step
    if not values:
        values.append(float(config.ridge_scale_min))
    if values[-1] < limit - 1e-3:
        values.append(limit)
    return tuple(values)


def _multiscale_ridge_evidence(
    denoised: np.ndarray,
    config: StructuralWatershedConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale-selected Hessian ridge: response, along-ridge orientation, confidence.

    Each scale is scored independently. The pixel keeps the strongest scale;
    binary detections are never OR-combined.
    """

    source = denoised.astype(np.float32)
    best_response = np.zeros(source.shape, dtype=np.float32)
    best_orientation = np.zeros(source.shape, dtype=np.float32)
    for sigma in _ridge_scales(config):
        response, orientation = _hessian_ridge_at_scale(source, sigma)
        stronger = response > best_response
        best_response = np.where(stronger, response, best_response).astype(np.float32)
        best_orientation = np.where(stronger, orientation, best_orientation).astype(np.float32)
    peak = float(np.percentile(best_response, 99.5)) if best_response.size else 0.0
    confidence = np.clip(best_response / max(peak, 1e-6), 0.0, 1.0).astype(np.float32)
    return best_response, best_orientation, confidence


def _hessian_ridge_at_scale(source: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    blurred = cv2.GaussianBlur(source, (0, 0), max(0.4, float(sigma)))
    lx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    ly = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    lxx = cv2.Sobel(lx, cv2.CV_32F, 1, 0, ksize=3)
    lxy = cv2.Sobel(lx, cv2.CV_32F, 0, 1, ksize=3)
    lyy = cv2.Sobel(ly, cv2.CV_32F, 0, 1, ksize=3)
    scale = float(sigma) * float(sigma)
    lxx *= scale
    lxy *= scale
    lyy *= scale
    tmp = cv2.magnitude(lxx - lyy, 2.0 * lxy)
    lambda_pos = 0.5 * (lxx + lyy + tmp)
    lambda_neg = 0.5 * (lxx + lyy - tmp)
    mag_pos = np.abs(lambda_pos)
    mag_neg = np.abs(lambda_neg)
    lambda_across = np.where(mag_pos >= mag_neg, lambda_pos, lambda_neg)
    lambda_along = np.where(mag_pos >= mag_neg, lambda_neg, lambda_pos)
    across_strength = np.abs(lambda_across)
    along_strength = np.abs(lambda_along)
    rb = along_strength / np.maximum(across_strength, 1e-6)
    hessian_norm = cv2.magnitude(lambda_across, lambda_along)
    frangi_c = max(float(np.percentile(hessian_norm, 90.0)), 1e-3)
    vesselness = np.exp(-(rb * rb) / (_FRANGI_BETA * _FRANGI_BETA)) * (
        1.0 - np.exp(-(hessian_norm * hessian_norm) / (frangi_c * frangi_c))
    )
    # Bright ridges only: dark grooves are separators, not conductor markers.
    vesselness = np.where(
        (lambda_across < 0.0) & (across_strength > along_strength),
        vesselness,
        0.0,
    ).astype(np.float32)
    across_x = lxy
    across_y = lambda_across - lxx
    fallback = (np.abs(across_x) + np.abs(across_y)) < 1e-6
    across_x = np.where(fallback, 1.0, across_x)
    across_y = np.where(fallback, 0.0, across_y)
    along_orientation = np.arctan2(across_x, -across_y).astype(np.float32)
    return vesselness, along_orientation


def _bright_rim_response(denoised: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    background = cv2.GaussianBlur(denoised.astype(np.float32), (0, 0), 6.0)
    relief = np.maximum(denoised.astype(np.float32) - background, 0.0)
    return (relief * _unit_range(magnitude)).astype(np.float32)


def _angles_aligned(first: np.ndarray, second: np.ndarray, *, max_delta: float) -> np.ndarray:
    delta = np.abs(np.mod(first - second + np.pi * 0.5, np.pi) - np.pi * 0.5)
    return delta <= max_delta


def _ridge_markers(
    features: _StructuralFeatures,
    config: StructuralWatershedConfig,
) -> tuple[np.ndarray, np.ndarray]:
    ribbon = _paired_rim_center_response(features)
    along = features.structure_orientation + (np.pi * 0.5)
    ribbon_nms = _non_maximum_suppress(ribbon, along)
    ribbon_peak = float(np.percentile(ribbon, 99.5)) if ribbon.size else 0.0
    coherent = features.coherence >= float(config.min_orientation_coherence)
    ribbon_seeds = np.where(
        (ribbon_nms > 0)
        & (ribbon >= float(config.min_ridge_confidence) * max(ribbon_peak, 1e-6))
        & coherent,
        255,
        0,
    ).astype(np.uint8)

    hessian_nms = _non_maximum_suppress(features.ridge_response, features.ridge_orientation)
    along_structure = features.structure_orientation + (np.pi * 0.5)
    aligned = _angles_aligned(
        features.ridge_orientation,
        along_structure,
        max_delta=np.pi / 5.0,
    )
    thin_line = ribbon < 0.25 * max(ribbon_peak, 1e-6)
    hessian_seeds = np.where(
        (hessian_nms > 0)
        & (features.ridge_confidence >= float(config.min_ridge_confidence))
        & coherent
        & aligned
        & thin_line,
        255,
        0,
    ).astype(np.uint8)
    raw = cv2.bitwise_or(ribbon_seeds, hessian_seeds)
    transverse_block = features.persistent_edge >= float(np.percentile(features.persistent_edge, 75.0))
    linked = _link_along_orientation(
        raw,
        along,
        length_px=int(config.gap_link_px),
        block=transverse_block,
    )
    return raw, _filter_short_components(
        linked,
        min_area=int(config.min_marker_area),
        min_length=int(config.min_marker_length),
    )


def _paired_rim_center_response(features: _StructuralFeatures) -> np.ndarray:
    """Scale-selected midline between a pair of bright rims with a bright fill."""

    across_x = np.cos(features.structure_orientation).astype(np.float32)
    across_y = np.sin(features.structure_orientation).astype(np.float32)
    rim = _unit_range(features.rim_response)
    intensity = features.denoised.astype(np.float32)
    height, width = intensity.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    best = np.zeros(intensity.shape, dtype=np.float32)
    for half in _RIBBON_HALF_WIDTHS:
        left_rim = _remap_offset(rim, grid_x, grid_y, across_x * half, across_y * half)
        right_rim = _remap_offset(rim, grid_x, grid_y, -across_x * half, -across_y * half)
        outer_left = _remap_offset(
            intensity,
            grid_x,
            grid_y,
            across_x * (half + 3.0),
            across_y * (half + 3.0),
        )
        outer_right = _remap_offset(
            intensity,
            grid_x,
            grid_y,
            -across_x * (half + 3.0),
            -across_y * (half + 3.0),
        )
        pair = np.minimum(left_rim, right_rim)
        brighter_than_outside = np.minimum(intensity - outer_left, intensity - outer_right)
        score = pair * np.maximum(brighter_than_outside, 0.0) * (1.0 - rim)
        best = np.maximum(best, score)
    return best.astype(np.float32)


def _remap_offset(
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
) -> np.ndarray:
    return cv2.remap(
        values.astype(np.float32),
        grid_x + dx,
        grid_y + dy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _non_maximum_suppress(response: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Keep local maxima of ridge response in the direction across the ridge."""

    across = orientation + (np.pi * 0.5)
    discrete = np.mod(np.round(across / (np.pi / 4.0)).astype(np.int32), 4)
    kept = np.zeros(response.shape, dtype=np.uint8)
    for index, (dx, dy) in enumerate(_NMS_DIRECTIONS):
        direction = discrete == index
        if not np.any(direction):
            continue
        neighbor_a = _shift_no_wrap(response, dx, dy)
        neighbor_b = _shift_no_wrap(response, -dx, -dy)
        local_max = (response >= neighbor_a) & (response >= neighbor_b) & (response > 0)
        kept[direction & local_max] = 255
    return kept


def _shift_no_wrap(values: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(values)
    src_y0 = max(0, -dy)
    src_y1 = values.shape[0] - max(0, dy)
    src_x0 = max(0, -dx)
    src_x1 = values.shape[1] - max(0, dx)
    dst_y0 = max(0, dy)
    dst_y1 = values.shape[0] - max(0, -dy)
    dst_x0 = max(0, dx)
    dst_x1 = values.shape[1] - max(0, -dx)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return shifted
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = values[src_y0:src_y1, src_x0:src_x1]
    return shifted


def _link_along_orientation(
    binary: np.ndarray,
    orientation: np.ndarray,
    *,
    length_px: int,
    block: np.ndarray | None = None,
) -> np.ndarray:
    """Close only short longitudinal gaps; never across a transverse boundary."""

    if length_px <= 0 or not np.any(binary):
        return binary
    linked = binary.copy()
    bins = 8
    blocked = None if block is None else block.astype(bool)
    for index in range(bins):
        angle = np.pi * index / bins
        selected = binary.copy()
        delta = np.abs(np.mod(orientation - angle + np.pi * 0.5, np.pi) - np.pi * 0.5)
        selected[delta > (np.pi / bins)] = 0
        if not np.any(selected):
            continue
        kernel = _oriented_line_kernel(2 * length_px + 1, angle)
        closed = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel)
        grown = (closed > 0) & (binary == 0)
        if blocked is not None:
            closed[grown & blocked] = 0
        linked = cv2.bitwise_or(linked, closed)
    return linked


def _oriented_line_kernel(size: int, angle: float) -> np.ndarray:
    length = max(3, int(size) | 1)
    kernel = np.zeros((length, length), dtype=np.uint8)
    center = length // 2
    dx = float(np.cos(angle))
    dy = float(np.sin(angle))
    for offset in range(-center, center + 1):
        x = int(round(center + offset * dx))
        y = int(round(center + offset * dy))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    kernel[center, center] = 1
    return kernel


def _filter_short_components(mask: np.ndarray, *, min_area: int, min_length: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros(mask.shape, dtype=np.uint8)
    _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(stats.shape[0], dtype=np.uint8)
    for index in range(1, stats.shape[0]):
        area = int(stats[index, cv2.CC_STAT_AREA])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area >= min_area and max(width, height) >= min_length:
            keep[index] = 1
    return np.where(keep[labels] > 0, 255, 0).astype(np.uint8)


def _wide_interior_markers(
    features: _StructuralFeatures,
    existing_core_seeds: np.ndarray,
    config: StructuralWatershedConfig,
) -> np.ndarray:
    """Confident interiors of wide conductors, including dark rim-bounded fills."""

    radius = max(1.0, float(config.wide_interior_radius))
    from_cores = _eroded_wide_cores(existing_core_seeds, radius)
    rim_walls = _rim_walls(features)
    open_space = np.where(rim_walls > 0, 0, 255).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(open_space, connectivity=4)
    if count <= 1:
        return from_cores
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    image_area = float(open_space.size)
    contact = _label_rim_contact(labels, count, rim_walls)
    gradient_means = _label_mean(labels, count, features.magnitude)
    widths = stats[:, cv2.CC_STAT_WIDTH].astype(np.float64)
    heights = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    box_perimeter = np.maximum(2.0 * (widths + heights), 1.0)
    gradient_limit = max(float(np.percentile(features.magnitude, 60.0)), 8.0)
    keep = (
        (areas >= max(float(config.min_marker_area), 16.0 * radius * radius))
        & (areas <= 0.45 * image_area)
        & (contact >= 0.55 * box_perimeter)
        & (np.minimum(widths, heights) >= 2.0 * radius)
        & (gradient_means <= gradient_limit)
    )
    keep[0] = False
    enclosed = np.where(keep[labels], 255, 0).astype(np.uint8)
    enclosed = _erode_binary(enclosed, 1)
    enclosed = _filter_short_components(
        enclosed,
        min_area=int(max(config.min_marker_area, radius * radius)),
        min_length=int(config.min_marker_length),
    )
    return cv2.bitwise_or(from_cores, enclosed)


def _eroded_wide_cores(core_seeds: np.ndarray, radius: float) -> np.ndarray:
    cores = ensure_binary_mask(core_seeds)
    if not np.any(cores):
        return cores
    distance = cv2.distanceTransform((cores > 0).astype(np.uint8), cv2.DIST_L2, 3)
    wide = np.where(distance >= radius, 255, 0).astype(np.uint8)
    if np.any(wide):
        return wide
    return _erode_binary(cores, max(1, int(round(radius))))


def _rim_walls(features: _StructuralFeatures) -> np.ndarray:
    rim_limit = float(np.percentile(features.rim_response, 85.0))
    walls = np.where(features.rim_response >= max(rim_limit, 1e-4), 255, 0).astype(np.uint8)
    return cv2.dilate(walls, np.ones((5, 5), np.uint8))


def _label_rim_contact(labels: np.ndarray, count: int, rim_walls: np.ndarray) -> np.ndarray:
    dilated = cv2.dilate(rim_walls, np.ones((3, 3), np.uint8))
    return np.bincount(
        labels.ravel(),
        weights=(dilated > 0).ravel().astype(np.float64),
        minlength=count,
    )


def _label_mean(labels: np.ndarray, count: int, values: np.ndarray) -> np.ndarray:
    sums = np.bincount(
        labels.ravel(),
        weights=values.astype(np.float64).ravel(),
        minlength=count,
    )
    counts = np.bincount(labels.ravel(), minlength=count).astype(np.float64)
    return np.divide(sums, np.maximum(counts, 1.0))


def _erode_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return ensure_binary_mask(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.erode(ensure_binary_mask(mask), kernel)


def _combine_foreground_markers(ridge_markers: np.ndarray, wide_markers: np.ndarray) -> np.ndarray:
    """Keep simple wide interiors; replace merged interiors with their ridges.

    A wide core that already covers several parallel traces would recreate the
    merge failure. Those cores are dropped and the interior ridge centerlines
    become the instance markers. Ridges that only touch a kept plate are rims
    and are discarded so they cannot split it.
    """

    ridges = ensure_binary_mask(ridge_markers)
    wide = ensure_binary_mask(wide_markers)
    if not np.any(ridges):
        return wide
    if not np.any(wide):
        return ridges

    ridge_count, ridge_labels = cv2.connectedComponents((ridges > 0).astype(np.uint8), connectivity=8)
    wide_count, wide_labels = cv2.connectedComponents((wide > 0).astype(np.uint8), connectivity=8)
    interior = (ridge_labels > 0) & (wide_labels > 0)
    interior_ridge_count = np.zeros(wide_count, dtype=np.int32)
    if np.any(interior):
        pair = wide_labels.astype(np.int64) * int(ridge_count) + ridge_labels.astype(np.int64)
        unique_pairs = np.unique(pair[interior])
        wide_ids = (unique_pairs // int(ridge_count)).astype(np.int32)
        interior_ridge_count = np.bincount(wide_ids, minlength=wide_count).astype(np.int32)

    keep_wide = interior_ridge_count <= 1
    keep_wide[0] = False
    kept_wide = np.where(keep_wide[wide_labels], 255, 0).astype(np.uint8)
    kept_wide = _fill_small_holes(kept_wide, max_area=64)

    blocked = cv2.dilate((kept_wide > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    kept_ridges = ridges.copy()
    kept_ridges[blocked > 0] = 0
    return cv2.bitwise_or(kept_wide, kept_ridges)


def _fill_small_holes(mask: np.ndarray, *, max_area: int) -> np.ndarray:
    binary = ensure_binary_mask(mask)
    if not np.any(binary):
        return binary
    inverted = np.where(binary > 0, 0, 255).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(inverted, connectivity=4)
    filled = binary.copy()
    height, width = binary.shape
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        left = int(stats[index, cv2.CC_STAT_LEFT])
        top = int(stats[index, cv2.CC_STAT_TOP])
        right = left + int(stats[index, cv2.CC_STAT_WIDTH])
        bottom = top + int(stats[index, cv2.CC_STAT_HEIGHT])
        touches_border = left <= 0 or top <= 0 or right >= width or bottom >= height
        if (not touches_border) and area <= max_area:
            filled[labels == index] = 255
    return filled


def _background_markers(
    features: _StructuralFeatures,
    groove_seeds: np.ndarray,
    ridge_markers: np.ndarray,
    foreground_markers: np.ndarray,
    config: StructuralWatershedConfig,
) -> np.ndarray:
    """Background only in high-confidence separators, never inside traces.

    Existing groove seeds are built for a binary core/groove watershed and are
    too dense for instance markers: they occupy conductor interiors and turn
    merges into misses. Inter-ridge valleys handle narrow gaps; distant dark
    substrate handles open fields.
    """

    del groove_seeds
    valleys = _inter_ridge_valleys(features, ridge_markers, config)
    substrate = _confident_substrate(features, foreground_markers)
    background = cv2.bitwise_or(valleys, substrate)
    blocked = cv2.dilate(ensure_binary_mask(foreground_markers), np.ones((3, 3), np.uint8))
    background[blocked > 0] = 0
    return ensure_binary_mask(background)


def _confident_substrate(features: _StructuralFeatures, foreground_markers: np.ndarray) -> np.ndarray:
    """High-confidence substrate: dark, calm, and far from foreground markers."""

    intensity = features.denoised.astype(np.float32)
    dark = intensity <= float(np.percentile(intensity, 30.0))
    calm = features.magnitude <= float(np.percentile(features.magnitude, 35.0))
    far = cv2.dilate(ensure_binary_mask(foreground_markers), np.ones((21, 21), np.uint8)) == 0
    seeds = np.where(dark & calm & far, 255, 0).astype(np.uint8)
    return _filter_short_components(seeds, min_area=32, min_length=8)


def _inter_ridge_valleys(
    features: _StructuralFeatures,
    ridge_markers: np.ndarray,
    config: StructuralWatershedConfig,
) -> np.ndarray:
    ridges = ensure_binary_mask(ridge_markers)
    if not np.any(ridges):
        return np.zeros(ridges.shape, dtype=np.uint8)
    distance = cv2.distanceTransform(cv2.bitwise_not(ridges), cv2.DIST_L2, 3)
    local_max = (
        (distance >= _shift_no_wrap(distance, 1, 0))
        & (distance >= _shift_no_wrap(distance, -1, 0))
        & (distance >= _shift_no_wrap(distance, 0, 1))
        & (distance >= _shift_no_wrap(distance, 0, -1))
        & (distance >= 1.0)
        & (distance <= max(2.0, float(config.wide_interior_radius) * 2.0))
    )
    low_ridge = features.ridge_confidence <= float(config.min_ridge_confidence)
    dark = features.denoised.astype(np.float32) <= float(np.percentile(features.denoised, 30.0))
    valleys = np.where(local_max & low_ridge & dark, 255, 0).astype(np.uint8)
    return _filter_short_components(
        valleys,
        min_area=max(2, int(config.min_marker_area) // 2),
        min_length=max(3, int(config.min_marker_length) // 2),
    )


def _boundary_cost_map(features: _StructuralFeatures, config: StructuralWatershedConfig) -> np.ndarray:
    gradient_term = _unit_range(features.magnitude)
    if config.variant in {"s1", "s2", "s5", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14"}:
        return gradient_term
    cost = (
        float(config.boundary_gradient_weight) * gradient_term
        + float(config.boundary_continuity_weight) * _unit_range(features.persistent_edge)
        + float(config.boundary_rim_weight) * _unit_range(features.rim_response)
        + float(config.boundary_orientation_weight)
        * features.coherence
        * _unit_range(features.oriented_gradient)
    )
    return _unit_range(cost)


def _segment_existing_markers(
    features: _StructuralFeatures,
    core_seeds: np.ndarray,
    groove_seeds: np.ndarray,
    *,
    metal_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    empty = np.zeros(features.denoised.shape, dtype=np.uint8)
    cost = _unit_range(features.magnitude)
    core = ensure_binary_mask(core_seeds)
    groove = ensure_binary_mask(groove_seeds)
    groove = cv2.bitwise_and(groove, cv2.bitwise_not(core))
    if not np.any(core) or not np.any(groove):
        fallback = np.where(features.denoised >= max(float(metal_limit), 1.0), 255, 0).astype(np.uint8)
        markers = np.zeros(features.denoised.shape, dtype=np.int32)
        return fallback, markers, cost, core, groove, empty, empty
    markers = np.zeros(features.denoised.shape, dtype=np.int32)
    markers[core > 0] = 2
    markers[groove > 0] = 1
    terrain = _cost_to_terrain(cost)
    cv2.watershed(cv2.cvtColor(terrain, cv2.COLOR_GRAY2BGR), markers)
    mask = ensure_binary_mask((markers == 2).astype(np.uint8) * 255)
    return _restore_fov_edge(mask), markers, cost, core, groove, empty, empty


def _sparse_background_markers(features: _StructuralFeatures, foreground_markers: np.ndarray) -> np.ndarray:
    """High-confidence substrate only. Narrow gaps stay unlabeled for competition."""

    return _confident_substrate(features, foreground_markers)


def _instance_watershed(
    foreground_markers: np.ndarray,
    background_markers: np.ndarray,
    cost: np.ndarray,
    *,
    grow: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    fg = ensure_binary_mask(foreground_markers)
    bg = cv2.bitwise_and(ensure_binary_mask(background_markers), cv2.bitwise_not(fg))
    markers = _seed_markers(fg, bg)
    if not np.any(fg):
        return np.zeros(fg.shape, dtype=np.uint8), markers
    if (not np.any(bg)) and (not grow):
        return fg, markers
    terrain = _cost_to_terrain(cost)
    cv2.watershed(cv2.cvtColor(terrain, cv2.COLOR_GRAY2BGR), markers)
    mask = ensure_binary_mask((markers > 1).astype(np.uint8) * 255)
    return _restore_fov_edge(mask), markers


def _geodesic_label_competition(
    foreground_markers: np.ndarray,
    background_markers: np.ndarray,
    cost: np.ndarray,
    *,
    foreground_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Multi-source competition: path cost ≈ distance × (1 + boundary)."""

    fg = ensure_binary_mask(foreground_markers)
    bg = cv2.bitwise_and(ensure_binary_mask(background_markers), cv2.bitwise_not(fg))
    if foreground_labels is not None and np.any(foreground_labels > 0):
        markers = _seed_labeled_markers(foreground_labels, bg)
        distance_source = np.where(foreground_labels > 0, 255, 0).astype(np.uint8)
    else:
        markers = _seed_markers(fg, bg)
        distance_source = fg
    if not np.any(distance_source):
        return markers
    distance = cv2.distanceTransform(
        cv2.bitwise_not(distance_source),
        cv2.DIST_L2,
        3,
    ).astype(np.float32)
    safe_cost = np.nan_to_num(np.clip(cost.astype(np.float32), 0.0, 1.0), nan=0.0)
    terrain = _cost_to_terrain(_unit_range(distance * (1.0 + 2.0 * safe_cost)))
    cv2.watershed(cv2.cvtColor(terrain, cv2.COLOR_GRAY2BGR), markers)
    return markers


def _seed_labeled_markers(foreground_labels: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Seed watershed with pre-assigned logical IDs, including disconnected fragments."""

    markers = np.zeros(foreground_labels.shape, dtype=np.int32)
    if np.any(background):
        markers[background > 0] = 1
    positive = foreground_labels > 0
    if np.any(positive):
        markers[positive] = foreground_labels[positive] + 1
    return markers


def _seed_markers(foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
    markers = np.zeros(foreground.shape, dtype=np.int32)
    if np.any(background):
        markers[background > 0] = 1
    if np.any(foreground):
        _count, labels = cv2.connectedComponents((foreground > 0).astype(np.uint8), connectivity=8)
        markers[foreground > 0] = labels[foreground > 0] + 1
    return markers


def _finalize_instance_labels(markers: np.ndarray) -> np.ndarray:
    """Map watershed markers to 0=background, 1..N=instances. Cuts join a neighbor."""

    restored = _restore_instance_fov_edge(markers)
    labels = np.zeros(restored.shape, dtype=np.int32)
    labels[restored > 1] = restored[restored > 1] - 1
    cuts = restored < 0
    if not np.any(cuts) or not np.any(labels):
        labels[cuts] = 0
        return labels
    neighbor = np.zeros_like(labels)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        shifted = _shift_no_wrap(labels, dx, dy)
        take = cuts & (neighbor == 0) & (shifted > 0)
        neighbor[take] = shifted[take]
    labels[cuts] = neighbor[cuts]
    return labels


def _restore_instance_fov_edge(labels: np.ndarray) -> np.ndarray:
    restored = labels.copy()
    inward = (
        (restored[1, :], restored[-2, :], restored[:, 1], restored[:, -2])
    )
    restored[0, :] = np.where(restored[0, :] <= 0, inward[0], restored[0, :])
    restored[-1, :] = np.where(restored[-1, :] <= 0, inward[1], restored[-1, :])
    restored[:, 0] = np.where(restored[:, 0] <= 0, inward[2], restored[:, 0])
    restored[:, -1] = np.where(restored[:, -1] <= 0, inward[3], restored[:, -1])
    return restored


def _binary_components_as_labels(mask: np.ndarray) -> np.ndarray:
    _count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return labels.astype(np.int32)


def _instance_fragment_stats(labels: np.ndarray) -> tuple[int, int]:
    positive = labels > 0
    if not np.any(positive):
        return 0, 0
    cc_count, components = cv2.connectedComponents(positive.astype(np.uint8), connectivity=8)
    encoded = labels.astype(np.int64) * (int(cc_count) + 1) + components.astype(np.int64)
    unique_pairs = np.unique(encoded[positive])
    instance_ids = (unique_pairs // (int(cc_count) + 1)).astype(np.int32)
    per_instance = np.bincount(instance_ids)
    instance_count = int(np.unique(labels[positive]).size)
    fragmented = int(np.count_nonzero(per_instance > 1))
    return instance_count, fragmented


def _label_boundary_map(labels: np.ndarray) -> np.ndarray:
    fg = np.maximum(labels, 0)
    right = _shift_no_wrap(fg, 1, 0)
    down = _shift_no_wrap(fg, 0, 1)
    boundary = ((fg > 0) & (right > 0) & (fg != right)) | ((fg > 0) & (down > 0) & (fg != down))
    return np.where(boundary, 255, 0).astype(np.uint8)


def _restore_fov_edge(mask: np.ndarray) -> np.ndarray:
    """OpenCV watershed paints the image border as a cut; restore FOV-touching metal."""

    restored = ensure_binary_mask(mask)
    restored[0, :] = np.maximum(restored[0, :], restored[1, :])
    restored[-1, :] = np.maximum(restored[-1, :], restored[-2, :])
    restored[:, 0] = np.maximum(restored[:, 0], restored[:, 1])
    restored[:, -1] = np.maximum(restored[:, -1], restored[:, -2])
    return restored


def _cost_to_terrain(cost: np.ndarray) -> np.ndarray:
    return np.clip(cost * 255.0, 0.0, 255.0).astype(np.uint8)


def _unit_range(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    low, high = np.percentile(values, (1.0, 99.5))
    span = float(high - low)
    if span <= 1e-6:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values.astype(np.float32) - float(low)) / span, 0.0, 1.0)


def _to_u8(values: np.ndarray) -> np.ndarray:
    if values.dtype == np.uint8:
        return values
    return _cost_to_terrain(_unit_range(values.astype(np.float32)))


def _orientation_overlay(orientation: np.ndarray, coherence: np.ndarray) -> np.ndarray:
    hsv = np.empty((*orientation.shape, 3), dtype=np.uint8)
    angle = np.mod(orientation, np.pi)
    hsv[:, :, 0] = np.clip(angle * (179.0 / np.pi), 0, 179).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip(coherence * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _label_overlay(labels: np.ndarray) -> np.ndarray:
    color = np.zeros((*labels.shape, 3), dtype=np.uint8)
    positive = labels > 0
    if not np.any(positive):
        return color
    hue = ((labels.astype(np.int32) * 37) % 180).astype(np.uint8)
    hsv = np.zeros((*labels.shape, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.where(positive, 220, 0).astype(np.uint8)
    hsv[labels < 0] = (0, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _empty_debug(empty: np.ndarray) -> dict[str, np.ndarray]:
    empty_f32 = empty.astype(np.float32)
    empty_bgr = np.zeros((*empty.shape, 3), dtype=np.uint8)
    return {
        "metal_structural_denoised": empty,
        "metal_structural_gx": empty,
        "metal_structural_gy": empty,
        "metal_structural_gradient_magnitude": empty,
        "metal_structural_orientation": empty_bgr,
        "metal_structural_coherence": empty,
        "metal_structural_ridge_response": empty,
        "metal_structural_ridge_markers_raw": empty,
        "metal_structural_ridge_markers": empty,
        "metal_structural_wide_interior_markers": empty,
        "metal_structural_foreground_markers": empty,
        "metal_structural_background_markers": empty,
        "metal_structural_boundary_cost": empty,
        "metal_structural_watershed_labels": empty_bgr,
        "metal_structural_instance_labels": empty_bgr,
        "metal_structural_label_boundary": empty,
        "metal_structural_final_mask": empty,
        "metal_structural_gx_f32": empty_f32,
        "metal_structural_gy_f32": empty_f32,
        "metal_structural_ridge_fragments": empty_bgr,
        "metal_structural_wide_fragments": empty_bgr,
        "metal_structural_ridge_links_accepted": empty_bgr,
        "metal_structural_ridge_links_rejected": empty_bgr,
        "metal_structural_ridge_links_boundary_veto": empty_bgr,
        "metal_structural_logical_ridge": empty_bgr,
        "metal_structural_logical_wide": empty_bgr,
        "metal_structural_logical_markers": empty_bgr,
        "metal_structural_conductor_bands": empty_bgr,
        "metal_structural_transverse_samples": empty_bgr,
        "metal_structural_band_groups_accepted": empty_bgr,
        "metal_structural_band_groups_rejected": empty_bgr,
    }


def _debug_images(
    features: _StructuralFeatures,
    *,
    ridge_raw: np.ndarray,
    ridge_markers: np.ndarray,
    wide_markers: np.ndarray,
    fg_markers: np.ndarray,
    bg_markers: np.ndarray,
    cost: np.ndarray,
    watershed_labels: np.ndarray,
    instance_labels: np.ndarray,
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "metal_structural_denoised": ensure_uint8(features.denoised),
        "metal_structural_gx": cv2.convertScaleAbs(features.gradient_x),
        "metal_structural_gy": cv2.convertScaleAbs(features.gradient_y),
        "metal_structural_gradient_magnitude": _to_u8(features.magnitude),
        "metal_structural_orientation": _orientation_overlay(
            features.structure_orientation,
            features.coherence,
        ),
        "metal_structural_coherence": _to_u8(features.coherence),
        "metal_structural_ridge_response": _to_u8(features.ridge_response),
        "metal_structural_ridge_markers_raw": ensure_binary_mask(ridge_raw),
        "metal_structural_ridge_markers": ensure_binary_mask(ridge_markers),
        "metal_structural_wide_interior_markers": ensure_binary_mask(wide_markers),
        "metal_structural_foreground_markers": ensure_binary_mask(fg_markers),
        "metal_structural_background_markers": ensure_binary_mask(bg_markers),
        "metal_structural_boundary_cost": _to_u8(cost),
        "metal_structural_watershed_labels": _label_overlay(watershed_labels),
        "metal_structural_instance_labels": _label_overlay(instance_labels),
        "metal_structural_label_boundary": _label_boundary_map(instance_labels),
        "metal_structural_final_mask": ensure_binary_mask(mask),
        "metal_structural_gx_f32": features.gradient_x,
        "metal_structural_gy_f32": features.gradient_y,
        "metal_structural_instance_labels_i32": instance_labels.astype(np.int32, copy=False),
    }
