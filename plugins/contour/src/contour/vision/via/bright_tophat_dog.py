"""Bright via detector: two-stage (candidates + scoring), classical OpenCV/NumPy only.

Independent from Qt; tune and test without UI changes.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from math import pi
from typing import Any
import cv2
import numpy as np
from ...application.preview_cancellation import raise_if_preview_cancelled
from ..io_normalize import to_gray_u8
from ..via_detection.binary_mask import detect_binary_components, looks_like_binary_mask

from .bright_tophat_dog_parts.module_state import (
    _COLOR_ACCEPT,
    _COLOR_SOFT,
    _COLOR_HARD,
    _MATCHED_PCT_OFFSET,
    _RADIAL_SMOOTH_SIGMA,
)
from .bright_tophat_dog_parts.center_distance_index import (
    _clip01,
    _CenterDistanceIndex,
)
from .bright_tophat_dog_parts.bright_via_detector_config import BrightViaDetectorConfig
from .bright_tophat_dog_parts.bright_via_detection import (
    BrightViaDetection,
    BrightViaDetectionResult,
    BrightViaPreparedCandidates,
    _CandidateDebugMaps,
    _RawStage1,
    detect_bright_vias,
)
from .bright_tophat_dog_parts.detect_binary_vias import _detect_binary_vias
from .bright_tophat_dog_parts.prepare_bright_via_candidates import prepare_bright_via_candidates
from .bright_tophat_dog_parts.score_bright_via_candidates import score_bright_via_candidates
from .bright_tophat_dog_parts.refine_center_bright_core import (
    _stage1_raw_candidates,
    _contours_from_mask,
    _raw_from_contour,
    _raw_from_peak,
    _contour_center_from_rect,
    _contour_geometry_center,
    _refine_center_on_bright_core,
    _fixed_size_bbox,
)
from .bright_tophat_dog_parts.local_maxima_points import (
    _local_maxima_points,
    _raw_nms,
    _passes_brightness_gate,
    _brightness_gate_reason,
)
from .bright_tophat_dog_parts.score_one_candidate import _score_one_candidate
from .bright_tophat_dog_parts.hard import _hard
from .bright_tophat_dog_parts.composite_final import _composite_final_0_100
from .bright_tophat_dog_parts.via_matched_filter_kernel import (
    radial_symmetry_score,
    _radial_symmetry_score_f32,
    edge_likeness_score,
    _edge_likeness_score_f32,
    line_likeness_score,
    _line_likeness_score_f32,
    bright_center_score,
    _bright_center_score_f32,
    center_mean_score,
    _center_mean_score_f32,
    mask_fraction,
    suppress_close_points,
    _preprocess,
    _white_tophat,
    _dog_response,
    _via_matched_filter_kernel,
    _via_matched_filter_response,
    _percentile_mask,
)
from .bright_tophat_dog_parts.build_isolation_map import (
    _bright_local_peak_mask,
    _tiled_percentile_mask,
    _odd_kernel_size,
    _build_isolation_map,
    _isolation_score,
    _nms_quality,
    _raw_nms_rank,
    _metal_mask,
    _edge_distance_map,
    _expanded_bbox,
    _disk_mean,
    _disk_mean_f32,
    _local_patch,
    _bilinear_sample,
    _sample_is_inside,
    _sample_distance,
)
from .bright_tophat_dog_parts.draw_overlay import (
    _as_gray_f32,
    _paint_score_disk,
    _normalize_u8,
    _normalize_response,
    _normalize_metal_constraint_mode,
    _draw_overlay,
)

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _COLOR_ACCEPT,
    _COLOR_SOFT,
    _COLOR_HARD,
    _MATCHED_PCT_OFFSET,
    _RADIAL_SMOOTH_SIGMA,
    _clip01,
    _CenterDistanceIndex,
    BrightViaDetectorConfig,
    BrightViaDetection,
    BrightViaDetectionResult,
    BrightViaPreparedCandidates,
    _CandidateDebugMaps,
    _RawStage1,
    detect_bright_vias,
    _detect_binary_vias,
    prepare_bright_via_candidates,
    score_bright_via_candidates,
    _stage1_raw_candidates,
    _contours_from_mask,
    _raw_from_contour,
    _raw_from_peak,
    _contour_center_from_rect,
    _contour_geometry_center,
    _refine_center_on_bright_core,
    _fixed_size_bbox,
    _local_maxima_points,
    _raw_nms,
    _passes_brightness_gate,
    _brightness_gate_reason,
    _score_one_candidate,
    _hard,
    _composite_final_0_100,
    radial_symmetry_score,
    _radial_symmetry_score_f32,
    edge_likeness_score,
    _edge_likeness_score_f32,
    line_likeness_score,
    _line_likeness_score_f32,
    bright_center_score,
    _bright_center_score_f32,
    center_mean_score,
    _center_mean_score_f32,
    mask_fraction,
    suppress_close_points,
    _preprocess,
    _white_tophat,
    _dog_response,
    _via_matched_filter_kernel,
    _via_matched_filter_response,
    _percentile_mask,
    _bright_local_peak_mask,
    _tiled_percentile_mask,
    _odd_kernel_size,
    _build_isolation_map,
    _isolation_score,
    _nms_quality,
    _raw_nms_rank,
    _metal_mask,
    _edge_distance_map,
    _expanded_bbox,
    _disk_mean,
    _disk_mean_f32,
    _local_patch,
    _bilinear_sample,
    _sample_is_inside,
    _sample_distance,
    _as_gray_f32,
    _paint_score_disk,
    _normalize_u8,
    _normalize_response,
    _normalize_metal_constraint_mode,
    _draw_overlay,
)
