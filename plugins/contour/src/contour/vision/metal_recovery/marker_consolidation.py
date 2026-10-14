"""Instance-aware consolidation of structural marker fragments.

Ridge fragments are linked only as longitudinal same-trace continuations.
Wide-interior fragments are merged only when they sit in one enclosed
conductor basin without a separating boundary.  Parallel conductors stay
distinct: a strong transverse offset or transverse boundary vetoes a link.

Ground truth, frame ids, and filenames are never consulted.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import cv2
import numpy as np
from ...utils import ensure_binary_mask, ensure_uint8
from .conductor_bands import (
    corridor_has_transverse_separator,
    evidence_from_consolidation,
    group_ridges_by_conductor_band,
    region_has_internal_separator,
)

from .marker_consolidation_parts.module_state import (
    RejectionReason,
    _NEIGHBOR_KERNEL,
    _SEARCH_RADIUS_PX,
    _CONE_COSINE,
    _MAX_ANGLE_DELTA,
    _MAX_TRANSVERSE_OFFSET_PX,
    _CORRIDOR_HALF_WIDTH_PX,
    _MIN_CONTINUATION_SCORE,
    _SHORT_GAP_PX,
    _AMBIGUITY_RATIO,
    _BOUNDARY_PERCENTILE,
    _LOCAL_WIDE_RADIUS_PX,
    _WEIGHTS,
)
from .marker_consolidation_parts.ridge_link import (
    ConsolidationEvidence,
    RidgeLink,
    MarkerConsolidationStats,
    MarkerConsolidationResult,
    _Endpoint,
    _Fragment,
)
from .marker_consolidation_parts.consolidate_markers import consolidate_markers
from .marker_consolidation_parts.link_ridge_fragments import _link_ridge_fragments
from .marker_consolidation_parts.extract_ridge_fragments import _extract_ridge_fragments
from .marker_consolidation_parts.fragment_endpoints import (
    _fragment_endpoints,
    _two_farthest_points,
    _principal_extremes,
    _outward_tangent,
    _score_endpoint_pairs,
    _spatial_buckets,
    _bucket_neighbors,
)
from .marker_consolidation_parts.continuation_link import _continuation_link
from .marker_consolidation_parts.corridor_boundary_veto import (
    _corridor_boundary_veto,
    _CorridorSample,
)
from .marker_consolidation_parts.sample_corridor import _sample_corridor
from .marker_consolidation_parts.mutual_best_matches import _mutual_best_matches
from .marker_consolidation_parts.reason import (
    _endpoint_key_a,
    _endpoint_key_b,
    _endpoint_pair_key,
    _with_reason,
)
from .marker_consolidation_parts.consolidate_wide_fragments import _consolidate_wide_fragments
from .marker_consolidation_parts.enclosed_conductor_basins import (
    _enclosed_conductor_basins,
    _basin_has_parallel_ridges,
)
from .marker_consolidation_parts.local_wide_unions import _local_wide_unions
from .marker_consolidation_parts.fragment_majority_labels import _fragment_majority_labels
from .marker_consolidation_parts.combine_logical_markers import _combine_logical_markers
from .marker_consolidation_parts.render_debug import (
    _connected_with_stats,
    _positive_id_count,
    _compact_labels,
    _empty_band_images,
    _find,
    _union,
    _remap_by_parent,
    _acute_angle_delta,
    _label_overlay,
    _draw_links,
    _render_debug,
)

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    RejectionReason,
    _NEIGHBOR_KERNEL,
    _SEARCH_RADIUS_PX,
    _CONE_COSINE,
    _MAX_ANGLE_DELTA,
    _MAX_TRANSVERSE_OFFSET_PX,
    _CORRIDOR_HALF_WIDTH_PX,
    _MIN_CONTINUATION_SCORE,
    _SHORT_GAP_PX,
    _AMBIGUITY_RATIO,
    _BOUNDARY_PERCENTILE,
    _LOCAL_WIDE_RADIUS_PX,
    _WEIGHTS,
    ConsolidationEvidence,
    RidgeLink,
    MarkerConsolidationStats,
    MarkerConsolidationResult,
    _Endpoint,
    _Fragment,
    consolidate_markers,
    _link_ridge_fragments,
    _extract_ridge_fragments,
    _fragment_endpoints,
    _two_farthest_points,
    _principal_extremes,
    _outward_tangent,
    _score_endpoint_pairs,
    _spatial_buckets,
    _bucket_neighbors,
    _continuation_link,
    _corridor_boundary_veto,
    _CorridorSample,
    _sample_corridor,
    _mutual_best_matches,
    _endpoint_key_a,
    _endpoint_key_b,
    _endpoint_pair_key,
    _with_reason,
    _consolidate_wide_fragments,
    _enclosed_conductor_basins,
    _basin_has_parallel_ridges,
    _local_wide_unions,
    _fragment_majority_labels,
    _combine_logical_markers,
    _connected_with_stats,
    _positive_id_count,
    _compact_labels,
    _empty_band_images,
    _find,
    _union,
    _remap_by_parent,
    _acute_angle_delta,
    _label_overlay,
    _draw_links,
    _render_debug,
)
