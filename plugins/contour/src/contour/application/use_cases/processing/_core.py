from __future__ import annotations
import cProfile
import hashlib
import io
import json
import pstats
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any
import cv2
import numpy as np
from ....contour_extractor import extract_polygons
from ....domain import PolygonData
from ....domain.polygon_offset import offset_conductor_polygons
from ....edge_detection import (
    build_gradient_elevation,
    normalize_edge_method,
    phase_congruency,
    ridge_response,
    scharr_magnitude,
    structured_edges,
)
from ....infrastructure.profiling import (
    processing_profiling_enabled,
    processing_top_lines,
    try_disable_profiler,
    try_enable_profiler,
    write_profile_report,
)
from ....pipeline import PreprocessingPipeline
from ....serializers import save_result_bundle
from ....utils import ensure_binary_mask, ensure_uint8, load_image_color
from ...preview_cancellation import raise_if_preview_cancelled
from ...processing import (
    ALGORITHM_BACKEND_LEGACY,
    RECOGNITION_MODE_CONDUCTORS,
    RECOGNITION_MODE_DISABLED,
    RECOGNITION_MODE_VIA,
    VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
    VIA_SEARCH_MODE_HEURISTIC,
    VIA_SEARCH_MODE_HYBRID,
    VIA_SEARCH_MODE_TEMPLATE,
    BatchFrameTiming,
    BatchImageResult,
    ContourDebugCandidate,
    ContourExtractionSettings,
    DisplaySettings,
    SaveOptions,
    normalize_algorithm_backend,
    normalize_recognition_mode,
    normalize_via_search_mode,
)

from ._core_parts.module_state import (
    _VIA_DETECTION_CACHE,
    _VIA_DETECTION_CACHE_MAX_ITEMS,
)
from ._core_parts.build_via_vectorization_mask import (
    PreviewProcessingRequest,
    PreparedImageRequest,
    _ViaCandidate,
    _make_image_signature,
    _bounded_cache_set,
    build_preview_request_signature,
    build_prepared_image_signature,
    prepare_image_for_preview,
    _recognition_image,
    apply_via_vectorization_mask,
    build_conductor_vectorization_mask,
    build_via_vectorization_mask,
)
from ._core_parts.build_modern_via_vectorization import _build_modern_via_vectorization_mask
from ._core_parts.build_detection_debug_maps import build_detection_debug_maps
from ._core_parts.refine_conductor_mask_by import (
    _refine_conductor_mask_by_gradient,
    _conductor_gradient_elevation,
    _resolve_conductor_edge_method,
    _resolve_via_edge_method,
    _via_grayscale,
    _range_mask,
    _odd_kernel_size,
    _expected_via_span,
    _clahe_gray,
    _normalize_columns,
    _via_polarity_scans,
    _intensity_mask,
    _intensity_gate,
    _multiscale_blob_response,
)
from ._core_parts.extract_response_peaks import (
    _log_blob_response,
    _ring_template_response,
    _extract_response_peaks,
    _IsolationContext,
    _build_isolation_context,
    _component_extent,
)
from ._core_parts.refine_center import _refine_center
from ._core_parts.verify_peak import _verify_peak
from ._core_parts.edge_ring_metrics import (
    _isolation_score,
    _radial_contrast,
    _edge_ring_metrics,
    _angular_coverage,
    _via_saved_template_arrays,
    _normalize_template_for_matching,
)
from ._core_parts.template_match_candidates import _template_match_candidates
from ._core_parts.apply_via_filters import (
    _apply_via_filters,
    _via_candidate_rejection_reason,
    _candidate_size_limits,
    _via_candidate_radii,
    _iou_nms,
    _candidate_iou,
    _render_via_candidates_mask,
    _rendered_candidate_size,
)
from ._core_parts.filter_vias_by_conductor import (
    _debug_candidates_from_via_candidates,
    _candidate_bbox,
    _should_run_sem_dual_branch,
    _build_via_settings_for_dual_branch,
    _component_map,
    _filter_vias_by_conductor_linearity,
    _merge_dual_branch_polygons,
    _build_metalization_mask,
)
from ._core_parts.metal_preview_batch_result import _metal_preview_batch_result
from ._core_parts.run_structural_metal_recovery import (
    _use_structural_metal_recovery,
    _run_structural_metal_recovery,
)
from ._core_parts.process_image_path import (
    process_image_path,
    process_image_path_timed,
)
from ._core_parts.process_image_path_impl import _process_image_path_impl
from ._core_parts.process_image_path_sem import _process_image_path_sem_backend
from ._core_parts.render_polygon_mask_polygons import _render_polygon_mask_from_polygons
from ._core_parts.debug_candidates_via_debug import (
    _debug_candidates_from_via_hits,
    _debug_candidates_from_via_debug,
)

_EMPTY_ISOLATION: _IsolationContext = _IsolationContext(
    active=False,
    labels=np.zeros((0, 0), dtype=np.int32),
    stats=np.zeros((0, 5), dtype=np.int32),
)

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _VIA_DETECTION_CACHE,
    _VIA_DETECTION_CACHE_MAX_ITEMS,
    PreviewProcessingRequest,
    PreparedImageRequest,
    _ViaCandidate,
    _make_image_signature,
    _bounded_cache_set,
    build_preview_request_signature,
    build_prepared_image_signature,
    prepare_image_for_preview,
    _recognition_image,
    apply_via_vectorization_mask,
    build_conductor_vectorization_mask,
    build_via_vectorization_mask,
    _build_modern_via_vectorization_mask,
    build_detection_debug_maps,
    _refine_conductor_mask_by_gradient,
    _conductor_gradient_elevation,
    _resolve_conductor_edge_method,
    _resolve_via_edge_method,
    _via_grayscale,
    _range_mask,
    _odd_kernel_size,
    _expected_via_span,
    _clahe_gray,
    _normalize_columns,
    _via_polarity_scans,
    _intensity_mask,
    _intensity_gate,
    _multiscale_blob_response,
    _log_blob_response,
    _ring_template_response,
    _extract_response_peaks,
    _IsolationContext,
    _build_isolation_context,
    _component_extent,
    _refine_center,
    _verify_peak,
    _isolation_score,
    _radial_contrast,
    _edge_ring_metrics,
    _angular_coverage,
    _via_saved_template_arrays,
    _normalize_template_for_matching,
    _template_match_candidates,
    _apply_via_filters,
    _via_candidate_rejection_reason,
    _candidate_size_limits,
    _via_candidate_radii,
    _iou_nms,
    _candidate_iou,
    _render_via_candidates_mask,
    _rendered_candidate_size,
    _debug_candidates_from_via_candidates,
    _candidate_bbox,
    _should_run_sem_dual_branch,
    _build_via_settings_for_dual_branch,
    _component_map,
    _filter_vias_by_conductor_linearity,
    _merge_dual_branch_polygons,
    _build_metalization_mask,
    _metal_preview_batch_result,
    _use_structural_metal_recovery,
    _run_structural_metal_recovery,
    process_image_path,
    process_image_path_timed,
    _process_image_path_impl,
    _process_image_path_sem_backend,
    _render_polygon_mask_from_polygons,
    _debug_candidates_from_via_hits,
    _debug_candidates_from_via_debug,
)
