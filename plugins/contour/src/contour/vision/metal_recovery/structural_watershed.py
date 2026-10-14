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

from .structural_watershed_parts.module_state import (
    STRUCTURAL_WATERSHED_STRATEGY,
    STRUCTURAL_VARIANTS,
    INSTANCE_IDENTITY_VARIANTS,
    GEODESIC_COMPETITION_VARIANTS,
    RIDGE_LINK_VARIANTS,
    WIDE_CONSOLIDATE_VARIANTS,
    BAND_GROUP_VARIANTS,
    ORIENTATION_VETO_VARIANTS,
    SEPARATOR_COMBINE_VARIANTS,
    _FRANGI_BETA,
    _RIDGE_SCALE_STEP,
    _DENSE_RIDGE_SCALE_STEP,
    _RIBBON_HALF_WIDTHS,
    _NMS_DIRECTIONS,
)
from .structural_watershed_parts.clamped_structural_watershed_config import (
    normalize_structural_variant,
    StructuralWatershedConfig,
    clamped_structural_watershed_config,
)
from .structural_watershed_parts.structural_watershed_config_object import (
    structural_watershed_config_from_object,
    StructuralWatershedResult,
    structural_watershed_mask,
)
from .structural_watershed_parts.run_structural_watershed import run_structural_watershed
from .structural_watershed_parts.extract_structural_features import (
    _StructuralFeatures,
    _extract_structural_features,
    _structure_tensor_field,
    _ridge_scales,
    _multiscale_ridge_evidence,
    _hessian_ridge_at_scale,
    _bright_rim_response,
    _angles_aligned,
)
from .structural_watershed_parts.ridge_markers import (
    _ridge_markers,
    _paired_rim_center_response,
    _remap_offset,
    _non_maximum_suppress,
    _shift_no_wrap,
    _link_along_orientation,
    _oriented_line_kernel,
    _filter_short_components,
)
from .structural_watershed_parts.wide_interior_markers import (
    _wide_interior_markers,
    _eroded_wide_cores,
    _rim_walls,
    _label_rim_contact,
    _label_mean,
    _erode_binary,
    _combine_foreground_markers,
    _fill_small_holes,
    _background_markers,
    _confident_substrate,
    _inter_ridge_valleys,
)
from .structural_watershed_parts.geodesic_label_competition import (
    _boundary_cost_map,
    _segment_existing_markers,
    _sparse_background_markers,
    _instance_watershed,
    _geodesic_label_competition,
    _seed_labeled_markers,
    _seed_markers,
    _finalize_instance_labels,
    _restore_instance_fov_edge,
    _binary_components_as_labels,
    _instance_fragment_stats,
    _label_boundary_map,
    _restore_fov_edge,
    _cost_to_terrain,
    _unit_range,
    _to_u8,
)
from .structural_watershed_parts.debug_images import (
    _orientation_overlay,
    _label_overlay,
    _empty_debug,
    _debug_images,
)

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    STRUCTURAL_WATERSHED_STRATEGY,
    STRUCTURAL_VARIANTS,
    INSTANCE_IDENTITY_VARIANTS,
    GEODESIC_COMPETITION_VARIANTS,
    RIDGE_LINK_VARIANTS,
    WIDE_CONSOLIDATE_VARIANTS,
    BAND_GROUP_VARIANTS,
    ORIENTATION_VETO_VARIANTS,
    SEPARATOR_COMBINE_VARIANTS,
    _FRANGI_BETA,
    _RIDGE_SCALE_STEP,
    _DENSE_RIDGE_SCALE_STEP,
    _RIBBON_HALF_WIDTHS,
    _NMS_DIRECTIONS,
    normalize_structural_variant,
    StructuralWatershedConfig,
    clamped_structural_watershed_config,
    structural_watershed_config_from_object,
    StructuralWatershedResult,
    structural_watershed_mask,
    run_structural_watershed,
    _StructuralFeatures,
    _extract_structural_features,
    _structure_tensor_field,
    _ridge_scales,
    _multiscale_ridge_evidence,
    _hessian_ridge_at_scale,
    _bright_rim_response,
    _angles_aligned,
    _ridge_markers,
    _paired_rim_center_response,
    _remap_offset,
    _non_maximum_suppress,
    _shift_no_wrap,
    _link_along_orientation,
    _oriented_line_kernel,
    _filter_short_components,
    _wide_interior_markers,
    _eroded_wide_cores,
    _rim_walls,
    _label_rim_contact,
    _label_mean,
    _erode_binary,
    _combine_foreground_markers,
    _fill_small_holes,
    _background_markers,
    _confident_substrate,
    _inter_ridge_valleys,
    _boundary_cost_map,
    _segment_existing_markers,
    _sparse_background_markers,
    _instance_watershed,
    _geodesic_label_competition,
    _seed_labeled_markers,
    _seed_markers,
    _finalize_instance_labels,
    _restore_instance_fov_edge,
    _binary_components_as_labels,
    _instance_fragment_stats,
    _label_boundary_map,
    _restore_fov_edge,
    _cost_to_terrain,
    _unit_range,
    _to_u8,
    _orientation_overlay,
    _label_overlay,
    _empty_debug,
    _debug_images,
)
