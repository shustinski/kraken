"""structural_watershed: run structural watershed."""
from __future__ import annotations
import numpy as np
from contour.utils import ensure_binary_mask, ensure_uint8
from contour.vision.metal_recovery.gradient_watershed import (
    GradientWatershedConfig,
    analyze_metal_presence,
    build_conductor_seeds,
)
from contour.vision.metal_recovery.marker_consolidation import (
    ConsolidationEvidence,
    MarkerConsolidationStats,
    consolidate_markers,
)

from contour.vision.metal_recovery.structural_watershed_parts.module_state import (
    BAND_GROUP_VARIANTS,
    GEODESIC_COMPETITION_VARIANTS,
    INSTANCE_IDENTITY_VARIANTS,
    ORIENTATION_VETO_VARIANTS,
    RIDGE_LINK_VARIANTS,
    SEPARATOR_COMBINE_VARIANTS,
    WIDE_CONSOLIDATE_VARIANTS,
)

from contour.vision.metal_recovery.structural_watershed_parts.clamped_structural_watershed_config import clamped_structural_watershed_config

from contour.vision.metal_recovery.structural_watershed_parts.debug_images import (
    _debug_images,
    _empty_debug,
)

from contour.vision.metal_recovery.structural_watershed_parts.extract_structural_features import _extract_structural_features

from contour.vision.metal_recovery.structural_watershed_parts.geodesic_label_competition import (
    _binary_components_as_labels,
    _boundary_cost_map,
    _finalize_instance_labels,
    _geodesic_label_competition,
    _instance_fragment_stats,
    _instance_watershed,
    _restore_fov_edge,
    _segment_existing_markers,
    _sparse_background_markers,
)

from contour.vision.metal_recovery.structural_watershed_parts.ridge_markers import _ridge_markers

from contour.vision.metal_recovery.structural_watershed_parts.wide_interior_markers import (
    _background_markers,
    _combine_foreground_markers,
    _wide_interior_markers,
)

def run_structural_watershed(
    gray: np.ndarray,
    watershed_config: GradientWatershedConfig | None = None,
    structural_config: StructuralWatershedConfig | None = None,
    *,
    check_presence: bool = True,
) -> StructuralWatershedResult:
    from contour.vision.metal_recovery.structural_watershed_parts.structural_watershed_config_object import StructuralWatershedResult
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
