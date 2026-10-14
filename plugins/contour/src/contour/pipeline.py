from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any
import cv2
import numpy as np
from .application.preview_cancellation import raise_if_preview_cancelled
from .application.processing import OperationParameterSpec, PipelineStepConfig
from .edge_detection import (
    EDGE_METHOD_CHOICES,
    build_gradient_elevation,
    combined_elevation,
)
from .edge_detection import (
    auto_canny as _ed_auto_canny,
)
from .edge_detection import (
    laplacian_of_gaussian as _ed_log,
)
from .edge_detection import (
    phase_congruency as _ed_phase_congruency,
)
from .edge_detection import (
    ridge_response as _ed_ridge_response,
)
from .edge_detection import (
    scharr_magnitude as _ed_scharr_magnitude,
)
from .edge_detection import (
    structured_edges as _ed_structured_edges,
)
from .i18n import choice_label, operation_name, parameter_label, tr
from .utils import ensure_binary_mask, ensure_uint8

from .pipeline_parts.module_state import (
    OperationCallable,
    OperationTimingCallback,
    _OPERATIONS,
    _COMBINED_PRESETS,
)
from .pipeline_parts.operation_descriptor import (
    OperationDescriptor,
    register_operation,
    get_operation_descriptor,
    available_operations,
    get_operation_display_name,
    get_parameter_display_label,
    get_choice_display_label,
)
from .pipeline_parts.preprocessing_pipeline import PreprocessingPipeline
from .pipeline_parts.watershed_split import (
    _odd,
    _operation_timing_bucket,
    _threshold_mode,
    _adaptive_mode,
    _morph_shape,
    _resize_interpolation,
    _as_gray,
    _as_bgr,
    _color_selection_entries,
    _color_binarize,
    _binary_fill_holes,
    _binary_filter_components,
    _binary_filter_area,
    _binary_filter_perimeter,
    _watershed_split,
    _kernel,
    _gaussian_blur,
    _median_blur,
    _bilateral,
)
from .pipeline_parts.edge_contour_refined_mask import (
    _clahe,
    _histogram_equalization,
    _brightness_contrast,
    _brightness,
    _contrast,
    _gamma_correction,
    _threshold,
    _adaptive_threshold,
    _otsu_threshold,
    _binary_threshold_mode,
    _base_threshold_mask,
    _edge_elevation,
    _edge_mask,
    _edge_contour_refined_mask,
)
from .pipeline_parts.edge_guided_threshold import (
    _edge_guided_threshold,
    _morph_op,
    _erode,
    _dilate,
    _canny,
    _invert,
    _resize,
    _scale_resize,
    _crop,
    _sharpen,
    _scharr_edges,
    _auto_canny,
    _log_edges,
    _ridge_edges,
    _structured_edge_operation,
    _phase_congruency_edges,
    _combined_edges,
    _edge_dispatch,
    _denoise,
)
from .pipeline_parts.register_builtin_operations import _register_builtin_operations

_register_builtin_operations()

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    OperationCallable,
    OperationTimingCallback,
    _OPERATIONS,
    _COMBINED_PRESETS,
    OperationDescriptor,
    register_operation,
    get_operation_descriptor,
    available_operations,
    get_operation_display_name,
    get_parameter_display_label,
    get_choice_display_label,
    PreprocessingPipeline,
    _odd,
    _operation_timing_bucket,
    _threshold_mode,
    _adaptive_mode,
    _morph_shape,
    _resize_interpolation,
    _as_gray,
    _as_bgr,
    _color_selection_entries,
    _color_binarize,
    _binary_fill_holes,
    _binary_filter_components,
    _binary_filter_area,
    _binary_filter_perimeter,
    _watershed_split,
    _kernel,
    _gaussian_blur,
    _median_blur,
    _bilateral,
    _clahe,
    _histogram_equalization,
    _brightness_contrast,
    _brightness,
    _contrast,
    _gamma_correction,
    _threshold,
    _adaptive_threshold,
    _otsu_threshold,
    _binary_threshold_mode,
    _base_threshold_mask,
    _edge_elevation,
    _edge_mask,
    _edge_contour_refined_mask,
    _edge_guided_threshold,
    _morph_op,
    _erode,
    _dilate,
    _canny,
    _invert,
    _resize,
    _scale_resize,
    _crop,
    _sharpen,
    _scharr_edges,
    _auto_canny,
    _log_edges,
    _ridge_edges,
    _structured_edge_operation,
    _phase_congruency_edges,
    _combined_edges,
    _edge_dispatch,
    _denoise,
    _register_builtin_operations,
)
