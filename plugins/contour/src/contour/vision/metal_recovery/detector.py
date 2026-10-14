"""Conductor recognition: segmentation mask → findContours + post-filters."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import ceil
from typing import Any
import cv2
import numpy as np
from ...application.preview_cancellation import raise_if_preview_cancelled
from ...application.processing import ContourExtractionSettings
from ...contour_extractor import extract_polygons
from ...domain import PolygonData
from ...utils import ensure_binary_mask, ensure_uint8
from .gradient_watershed import (
    analyze_metal_presence,
    build_conductor_seeds,
    gradient_watershed_config_from_object,
)
from .pipeline_stages import axis_gradient_debug_images, build_metal_segmentation_mask_staged, image_signature
from .segmentation import (
    MetalSegmentationConfig,
    normalize_metal_adaptive_method,
    normalize_metal_segmentation_strategy,
    resolve_metal_segmentation_strategy,
)
from .strategy_registry import IMPLEMENTED_NEW_STRATEGIES, MetalStrategyConfigs

from .detector_parts.module_state import (
    _CONTOUR_CACHE,
    _CONTOUR_CACHE_MAX_ITEMS,
)
from .detector_parts.normalize_metal_extraction_mode import _normalize_metal_extraction_mode
from .detector_parts.metal_recovery_config import MetalRecoveryConfig
from .detector_parts.contour_extraction_settings_metal import (
    MetalPolygonRecord,
    MetalDetectionResult,
    _ContourStageCache,
    clear_metal_contour_cache,
    _requested_segmentation_strategy,
    _segmentation_config_from_recovery,
    _contour_config_signature,
    build_metal_extraction_mask,
    contour_extraction_settings_from_metal,
    _border_touch_points,
    _trace_length_px,
    _trace_width_px,
    _passes_trace_geometry,
)
from .detector_parts.extract_polygons_per_instance import (
    _extract_polygons_per_instance,
    _extract_polygons_cached,
)
from .detector_parts.polygon_interior_median import (
    _renumber_polygons_preserving_parents,
    _polygon_interior_median,
)
from .detector_parts.filter_holes_by_source import _filter_holes_by_source_contrast
from .detector_parts.filter_conductors_by_source import _filter_conductors_by_source_contrast
from .detector_parts.rasterize_accepted_conductors import _rasterize_accepted_conductors
from .detector_parts.detect_metalization_explicit import _detect_metalization_explicit
from .detector_parts.prefer_contrast_refined_legacy import (
    _non_hole_count,
    _mask_iou,
    _prefer_contrast_refined_legacy,
    _extended_separator_mask,
)
from .detector_parts.stable_local_refinement import _stable_local_refinement
from .detector_parts.directionally_bridge_mask import _directionally_bridge_mask
from .detector_parts.refine_local_candidate import _refine_local_candidate
from .detector_parts.detect_metalization import detect_metalization

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _CONTOUR_CACHE,
    _CONTOUR_CACHE_MAX_ITEMS,
    _normalize_metal_extraction_mode,
    MetalRecoveryConfig,
    MetalPolygonRecord,
    MetalDetectionResult,
    _ContourStageCache,
    clear_metal_contour_cache,
    _requested_segmentation_strategy,
    _segmentation_config_from_recovery,
    _contour_config_signature,
    build_metal_extraction_mask,
    contour_extraction_settings_from_metal,
    _border_touch_points,
    _trace_length_px,
    _trace_width_px,
    _passes_trace_geometry,
    _extract_polygons_per_instance,
    _extract_polygons_cached,
    _renumber_polygons_preserving_parents,
    _polygon_interior_median,
    _filter_holes_by_source_contrast,
    _filter_conductors_by_source_contrast,
    _rasterize_accepted_conductors,
    _detect_metalization_explicit,
    _non_hole_count,
    _mask_iou,
    _prefer_contrast_refined_legacy,
    _extended_separator_mask,
    _stable_local_refinement,
    _directionally_bridge_mask,
    _refine_local_candidate,
    detect_metalization,
)
