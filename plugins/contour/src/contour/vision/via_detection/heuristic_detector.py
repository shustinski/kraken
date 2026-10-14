"""Heuristic via detection: local extrema + local component + structural scoring (no top-hat/DoG/ML)."""

from __future__ import annotations
import math
import time
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
import cv2
import numpy as np
from .binary_mask import detect_binary_components, looks_like_binary_mask
from .config import HeuristicViaDetectorConfig, ViaPolarity
from .result import DetectionResult, ViaDetection

from .heuristic_detector_parts.candidate_features import (
    ViaRejectCode,
    PreparedHeuristicFrame,
    CandidateFeatures,
    SegmentedCandidate,
    _hard_reason,
    _det_better,
)
from .heuristic_detector_parts.detect_vias_heuristic import detect_vias_heuristic
from .heuristic_detector_parts.score_seed import (
    analyze_via_at,
    analyze_vias_at,
    _prepare_heuristic_frame,
    _candidate_hypotheses,
    _score_seed,
    _to_gray_u8,
    _preprocess_denoise,
    _normalize01_to_u8,
    _fast_percentile,
    _fast_percentile_u8,
)
from .heuristic_detector_parts.intensity_range_seeds import (
    _local_extrema_seeds,
    _intensity_range_seeds,
    _grid_suppressed_points,
    _merge_seeds,
    _spread_points,
    _dedupe_by_score,
    _nms_simple,
)
from .heuristic_detector_parts.local_structure_metrics import (
    _refine_center_xy,
    _component_geometry_center,
    _coordinate_grids,
    _annulus_masks,
    _mean_mask,
    _center_in_brightness_range,
    _contrast_for_polarity,
    _local_center_background_contrast,
    _local_background_masks,
    _local_structure_metrics,
    _candidate_bbox,
)
from .heuristic_detector_parts.segment_candidate import _segment_candidate
from .heuristic_detector_parts.score_one import _score_one
from .heuristic_detector_parts.candidate_score_metrics import _candidate_score_metrics
from .heuristic_detector_parts.final_candidate_score import (
    _final_candidate_score,
    _scale01,
)
from .heuristic_detector_parts.debug_viz import _debug_viz

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    ViaRejectCode,
    PreparedHeuristicFrame,
    CandidateFeatures,
    SegmentedCandidate,
    _hard_reason,
    _det_better,
    detect_vias_heuristic,
    analyze_via_at,
    analyze_vias_at,
    _prepare_heuristic_frame,
    _candidate_hypotheses,
    _score_seed,
    _to_gray_u8,
    _preprocess_denoise,
    _normalize01_to_u8,
    _fast_percentile,
    _fast_percentile_u8,
    _local_extrema_seeds,
    _intensity_range_seeds,
    _grid_suppressed_points,
    _merge_seeds,
    _spread_points,
    _dedupe_by_score,
    _nms_simple,
    _refine_center_xy,
    _component_geometry_center,
    _coordinate_grids,
    _annulus_masks,
    _mean_mask,
    _center_in_brightness_range,
    _contrast_for_polarity,
    _local_center_background_contrast,
    _local_background_masks,
    _local_structure_metrics,
    _candidate_bbox,
    _segment_candidate,
    _score_one,
    _candidate_score_metrics,
    _final_candidate_score,
    _scale01,
    _debug_viz,
)
