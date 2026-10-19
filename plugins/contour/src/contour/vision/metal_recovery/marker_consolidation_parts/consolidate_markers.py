"""marker_consolidation: consolidate markers."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from contour.vision.metal_recovery.conductor_bands import (
    evidence_from_consolidation,
    group_ridges_by_conductor_band,
)

from contour.vision.metal_recovery.marker_consolidation_parts.combine_logical_markers import _combine_logical_markers

from contour.vision.metal_recovery.marker_consolidation_parts.consolidate_wide_fragments import _consolidate_wide_fragments

from contour.vision.metal_recovery.marker_consolidation_parts.link_ridge_fragments import _link_ridge_fragments

from contour.vision.metal_recovery.marker_consolidation_parts.render_debug import (
    _connected_with_stats,
    _empty_band_images,
    _positive_id_count,
    _render_debug,
)

from contour.vision.metal_recovery.marker_consolidation_parts.ridge_link import (
    MarkerConsolidationResult,
    MarkerConsolidationStats,
)

def consolidate_markers(
    ridge_mask: np.ndarray,
    wide_mask: np.ndarray,
    evidence: ConsolidationEvidence,
    *,
    link_ridge: bool,
    link_wide: bool,
    group_bands: bool = False,
    orientation_aware_veto: bool = False,
    separator_aware_combine: bool = False,
    min_marker_area: int = 16,
    wide_interior_radius: float = 4.0,
) -> MarkerConsolidationResult:
    ridge_count, ridge_raw, ridge_stats, ridge_centroids = _connected_with_stats(ridge_mask)
    wide_count, wide_raw, wide_stats, wide_centroids = _connected_with_stats(wide_mask)
    accepted: tuple[RidgeLink, ...] = ()
    rejected: tuple[RidgeLink, ...] = ()
    reasons: dict[str, int] = {}

    if link_ridge and ridge_count > 0:
        ridge_logical, accepted, rejected, reasons = _link_ridge_fragments(
            ridge_raw,
            ridge_count,
            ridge_stats,
            ridge_centroids,
            evidence,
            orientation_aware_veto=orientation_aware_veto,
        )
    else:
        ridge_logical = ridge_raw.copy()

    band_debug = _empty_band_images(ridge_raw.shape)
    band_count = 0
    accepted_band_groups = 0
    if group_bands and np.any(ridge_logical):
        grouped, band_stats, band_debug = group_ridges_by_conductor_band(
            ridge_logical,
            evidence_from_consolidation(evidence),
        )
        ridge_logical = grouped
        band_count = int(band_stats.band_count)
        accepted_band_groups = int(band_stats.accepted_band_groups)

    if link_wide and wide_count > 0:
        wide_logical, wide_unions = _consolidate_wide_fragments(
            wide_raw,
            wide_count,
            wide_stats,
            wide_centroids,
            ridge_raw,
            evidence,
            min_marker_area=min_marker_area,
            wide_interior_radius=wide_interior_radius,
        )
    else:
        wide_logical = wide_raw.copy()
        wide_unions = 0

    combined, kept_multi, dropped_sep = _combine_logical_markers(
        ridge_logical,
        wide_logical,
        evidence,
        separator_aware=separator_aware_combine,
    )
    stats = MarkerConsolidationStats(
        raw_ridge_count=ridge_count,
        logical_ridge_count=_positive_id_count(ridge_logical),
        accepted_ridge_links=len(accepted),
        rejected_ridge_links=len(rejected),
        rejection_reasons=reasons,
        raw_wide_count=wide_count,
        logical_wide_count=_positive_id_count(wide_logical),
        wide_unions=wide_unions,
        combined_logical_count=_positive_id_count(combined),
        band_count=band_count,
        accepted_band_groups=accepted_band_groups,
        wide_kept_with_multi_ridge=kept_multi,
        wide_dropped_for_separator=dropped_sep,
    )
    debug = _render_debug(
        evidence.intensity,
        ridge_raw=ridge_raw,
        ridge_logical=ridge_logical,
        wide_raw=wide_raw,
        wide_logical=wide_logical,
        combined=combined,
        accepted=accepted,
        rejected=rejected,
    )
    debug.update(band_debug)
    return MarkerConsolidationResult(
        ridge_raw_labels=ridge_raw,
        ridge_logical_labels=ridge_logical,
        wide_raw_labels=wide_raw,
        wide_logical_labels=wide_logical,
        combined_labels=combined,
        accepted_links=accepted,
        rejected_links=rejected,
        stats=stats,
        debug_images=debug,
    )

if TYPE_CHECKING:
    from contour.vision.metal_recovery.marker_consolidation_parts.ridge_link import ConsolidationEvidence, RidgeLink
