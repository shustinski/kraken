from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from ..domain import PolygonData

from .processing_parts.module_state import (
    VIA_SIZE_MODE_RANGE,
    VIA_SIZE_MODE_FIXED,
    VIA_SEARCH_MODE_HYBRID,
    VIA_SEARCH_MODE_BLOB,
    VIA_SEARCH_MODE_TEMPLATE,
    VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
    VIA_SEARCH_MODE_HEURISTIC,
    VIA_CHANNEL_MODE_COLUMNS,
    VIA_CHANNEL_MODE_GRAYSCALE,
    VIA_CHANNEL_MODE_RED_BLUE,
    ALGORITHM_BACKEND_LEGACY,
    ALGORITHM_BACKEND_SEM,
    VIA_DISPLAY_MODE_RECTANGLE,
    VIA_DISPLAY_MODE_CIRCLE,
    RECOGNITION_MODE_DISABLED,
    RECOGNITION_MODE_VIA,
    RECOGNITION_MODE_METAL,
    RECOGNITION_MODE_CONDUCTORS,
)
from .processing_parts.dict import DictMixin
from .processing_parts.dict_2 import Dict2Mixin
from .processing_parts.pipeline_step_config import (
    normalize_via_display_mode,
    normalize_via_size_mode,
    normalize_via_search_mode,
    normalize_via_channel_mode,
    _normalize_bright_via_mask_mode,
    _normalize_bright_via_metal_constraint_mode,
    _odd_positive,
    normalize_algorithm_backend,
    normalize_recognition_mode,
    normalize_metal_segmentation_strategy,
    normalize_metal_segmentation_method,
    normalize_metal_sensitivity,
    parse_integer_value_list,
    _serialize_template_images,
    _parse_template_images,
    OperationParameterSpec,
    PipelineStepConfig,
    ContourDebugCandidate,
)
from .processing_parts.display_settings import (
    DisplaySettings,
    SaveOptions,
    ImageProcessingState,
    BatchImageResult,
    BatchFrameTiming,
    BatchImageMetadata,
    BatchProcessingOptions,
    base_name_from_path,
)

@dataclass(slots=True)
class ContourExtractionSettings(DictMixin, Dict2Mixin):
    algorithm_backend: str = ALGORITHM_BACKEND_LEGACY
    sem_noise_level: str = "medium"
    sem_polarity: str = "auto"
    sem_preserve_hierarchy: bool = True
    extraction_profile: str = "conductors"
    object_type: str = "conductor"
    output_mode: str = "polygon"
    retrieval_mode: str = "RETR_EXTERNAL"
    approximation_mode: str = "CHAIN_APPROX_SIMPLE"
    epsilon: float = 2.0
    epsilon_relative: bool = False
    preserve_corners: bool = False
    min_polygon_angle: float = 0.0
    min_area: float = 0.0
    max_area: float | None = None
    min_perimeter: float = 10.0
    max_perimeter: float | None = None
    min_points: int = 3
    min_bbox_width: int = 0
    max_bbox_width: int | None = None
    min_bbox_height: int = 0
    max_bbox_height: int | None = None
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float | None = None
    exclude_border_touching: bool = False
    min_solidity: float = 0.0
    min_extent: float = 0.0
    min_polygon_width_px: float = 0.0
    min_via_width: int = 0
    max_via_width: int | None = None
    min_via_height: int = 0
    max_via_height: int | None = None
    via_size_mode: str = VIA_SIZE_MODE_FIXED
    via_search_mode: str = VIA_SEARCH_MODE_HEURISTIC
    fixed_via_widths: list[int] = field(default_factory=list)
    fixed_via_heights: list[int] = field(default_factory=list)
    via_channel_mode: str = VIA_CHANNEL_MODE_GRAYSCALE
    via_white_range_enabled: bool = True
    via_white_range_min: int = 140
    via_white_range_max: int = 255
    via_black_range_enabled: bool = False
    via_black_range_min: int = 0
    via_black_range_max: int = 30
    via_min_roundness: float = 40.0
    via_min_score: float = 0.35
    via_min_contrast: float = 14.0
    via_min_edge_coverage: float = 0.45
    via_spot_line_suppression: float = 0.65
    via_template_min_score: float = 0.35
    via_template_images: list[Any] = field(default_factory=list)
    via_template_min_scores: list[float] = field(default_factory=list)
    via_template_diameters: list[int] = field(default_factory=list)
    via_template_nms_distance: int = 4
    via_template_scale_min: float = 0.9
    via_template_scale_max: float = 1.1
    via_template_scale_step: float = 0.1
    via_heuristic_polarity: str = "auto"
    via_fixed_diameters_text: str = "6, 8, 10"
    heuristic_background_sigma: float = 25.0
    heuristic_analysis_window_scale: float = 3.0
    heuristic_min_center_brightness: float = 0.0
    heuristic_min_center_contrast: float = 50.0
    heuristic_min_peak_prominence: float = 50.0
    heuristic_min_compactness: float = 0.9
    heuristic_min_circularity: float = 0.40
    heuristic_max_elongation: float = 2.5
    heuristic_line_penalty_scale: float = 3.0
    heuristic_border_penalty_scale: float = 1.0
    heuristic_local_binarize_percentile: float = 88.0
    heuristic_min_abs_peak: float = 0.0
    heuristic_use_bilateral: bool = False
    heuristic_size_tolerance_range: float = 0.36
    heuristic_size_tolerance_fixed: float = 0.26
    heuristic_max_center_drift_ratio: float = 0.72
    heuristic_max_line_coherence: float = 0.82
    heuristic_min_edge_sharpness: float = 0.20
    heuristic_contrast_score_min: float = 3.0
    heuristic_contrast_score_max: float = 20.0
    heuristic_prominence_score_min: float = 2.0
    heuristic_prominence_score_max: float = 25.0
    heuristic_edge_snr_score_min: float = 0.70
    heuristic_edge_snr_score_max: float = 2.80
    heuristic_edge_quality_floor: float = 0.55
    heuristic_border_balance_scale: float = 2.0
    heuristic_seed_percentile: float = 90.0
    heuristic_w_contrast: float = 25.0
    heuristic_w_prominence: float = 20.0
    heuristic_w_size: float = 20.0
    heuristic_w_compact: float = 15.0
    heuristic_w_round: float = 10.0
    heuristic_w_balance: float = 10.0
    heuristic_w_line: float = 20.0
    heuristic_w_border: float = 20.0
    bright_via_diameter_min: int = 8
    bright_via_diameter_max: int = 8
    via_output_diameter: int = 8
    bright_via_clahe_clip_limit: float = 2.0
    bright_via_clahe_tile_grid_size: int = 8
    bright_via_median_blur_kernel: int = 3
    bright_via_tophat_kernel_size: int = 11
    bright_via_dog_sigma_small: float = 0.8
    bright_via_dog_sigma_large: float = 2.0
    bright_via_threshold_percentile: float = 99.0
    bright_via_mask_combine_mode: str = "OR"
    bright_via_min_area_factor: float = 0.45
    bright_via_max_area_factor: float = 1.8
    bright_via_min_circularity: float = 0.30
    bright_via_min_aspect: float = 0.45
    bright_via_max_aspect: float = 2.2
    bright_via_bright_center_min_score: float = 140.0
    bright_via_metal_constraint_mode: str = "soft"
    bright_via_use_metal_mask: bool = True
    bright_via_metal_fraction_min: float = 0.3
    bright_via_max_radial_asymmetry: float = 32.0
    bright_via_max_edge_likeness: float = 35.0
    bright_via_max_line_likeness: float = 65.0
    bright_via_nms_distance: int = 5
    bright_via_min_final_score: float = 38.0
    bright_via_show_rejected: bool = True
    bright_via_hard_reject_on_asymmetry: bool = True
    bright_via_hard_reject_on_edge: bool = False
    bright_via_hard_reject_on_line: bool = False
    bright_via_min_isolation_score: float = 0.38
    bright_via_min_annular_contrast: float = 6.0
    bright_via_max_annular_contrast: float = 0.0
    bright_via_tiled_threshold_size: int = 256
    bright_via_scale_kernels_from_diameter: bool = True
    debug_enabled: bool = True
    debug_gradient_map_enabled: bool = False
    min_hierarchy_depth: int = 0
    max_hierarchy_depth: int | None = None
    min_inner_hole_area: float = 100.0
    max_hole_area_ratio: float | None = None
    conductor_gradient_enabled: bool = False
    conductor_gradient_min_strength: float = 18.0
    conductor_gradient_band_radius: int = 3
    edge_method: str = "sobel"
    via_gradient_edge_method: str = ""
    conductor_gradient_edge_method: str = ""
    recognition_mode: str = RECOGNITION_MODE_CONDUCTORS
    via_display_show_detected: bool = True
    via_display_show_candidates: bool = True
    metal_structural_pipeline: bool = False
    metal_preset: str = "standard"
    metal_noise_suppression: int = 20
    metal_min_contrast: float = 50.0
    metal_min_object_source_contrast: float = 12.0
    metal_min_object_rim_contrast: float = 36.0
    metal_min_object_rim_area_fraction: float = 0.001
    # Deprecated persisted setting; retained only for backward-compatible loading.
    metal_contrast_bias: float = 0.0
    metal_min_hole_source_contrast: float = 8.0
    metal_min_hole_source_contrast_fraction: float = 0.35
    metal_segmentation_strategy: str = "auto"
    metal_auto_contrast_step: float = 10.0
    metal_auto_source_contrast_step: float = 4.0
    metal_auto_directional_gap_bridge_px: int = 3
    metal_auto_directional_gap_min_source_intensity: float = 45.0
    metal_gap_bridge_px: int = 1
    metal_speckle_removal_px: int = 1
    metal_contour_smooth_px: float = 0.0
    metal_contour_close_px: int = 0
    metal_min_object_area: float = 30.0
    metal_min_trace_width_px: float = 8.0
    metal_conductor_size_offset_px: float = 0.0
    metal_max_trace_width_px: float | None = None
    metal_min_trace_length_px: float = 8.0
    metal_allowed_angles: str = "free"
    metal_angle_tolerance_deg: float = 7.0
    metal_min_straightness: float = 0.2
    metal_allow_t_junction: bool = True
    metal_border_handling: str = "mark"
    metal_check_contour_validity: bool = True
    metal_hierarchy_mode: str = "full"
    metal_min_area: float = 60.0
    metal_max_area: float | None = None
    metal_min_perimeter: float = 32.0
    metal_max_perimeter: float | None = None
    metal_approximation_enabled: bool = True
    metal_morph_close_radius: int = 1
    metal_morph_open_radius: int = 1
    metal_display_show_conductors: bool = True
    metal_display_show_mask: bool = False
    metal_display_show_contours: bool = True
    metal_display_show_rejected: bool = True
    metal_display_show_suspicious: bool = True
    metal_display_show_border_highlight: bool = False
    metal_debug_visual: str = "overlay"
    metal_overlay_opacity: float = 0.45
    metal_preprocess_subtract_background: bool = True
    metal_preprocess_background_sigma_fraction: float = 0.05
    metal_preprocess_clahe_clip: float = 2.0
    metal_preprocess_clahe_grid: int = 8
    metal_preprocess_denoise: str = "low"
    metal_use_wide_conductor_gradient: bool = False
    metal_watershed_smoothing_sigma: float = 1.2
    metal_watershed_core_margin: float = 23.0
    metal_watershed_groove_margin: float = 19.0
    metal_watershed_rim_probe_px: int = 2
    metal_watershed_seed_speckle_px: int = 1
    metal_watershed_valley_span_px: int = 5
    metal_watershed_valley_depth: float = 65.0
    metal_random_walker_beta: float = 90.0
    metal_random_walker_iterations: int = 160
    metal_graph_cut_iterations: int = 5
    metal_reconstruction_erode_px: int = 0
    metal_boundary_relief: float = 16.0
    metal_boundary_background_sigma: float = 12.0
    metal_structural_variant: str = "s2"
    metal_strategy_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    metal_adaptive_block_size: int = 0
    metal_adaptive_c: float = 0.0
    metal_adaptive_method: str = "gaussian"
    metal_edge_close_cap_px: int = 9
    metal_edge_watershed_split: bool = True
    metal_edge_watershed_dist_peak_frac: float = 0.38
    # 0 or negative = no limit (always run watershed when enabled); default skips on huge frames.
    metal_edge_watershed_max_pixels: int = 3_000_000

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    VIA_SIZE_MODE_RANGE,
    VIA_SIZE_MODE_FIXED,
    VIA_SEARCH_MODE_HYBRID,
    VIA_SEARCH_MODE_BLOB,
    VIA_SEARCH_MODE_TEMPLATE,
    VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
    VIA_SEARCH_MODE_HEURISTIC,
    VIA_CHANNEL_MODE_COLUMNS,
    VIA_CHANNEL_MODE_GRAYSCALE,
    VIA_CHANNEL_MODE_RED_BLUE,
    ALGORITHM_BACKEND_LEGACY,
    ALGORITHM_BACKEND_SEM,
    VIA_DISPLAY_MODE_RECTANGLE,
    VIA_DISPLAY_MODE_CIRCLE,
    RECOGNITION_MODE_DISABLED,
    RECOGNITION_MODE_VIA,
    RECOGNITION_MODE_METAL,
    RECOGNITION_MODE_CONDUCTORS,
    normalize_via_display_mode,
    normalize_via_size_mode,
    normalize_via_search_mode,
    normalize_via_channel_mode,
    _normalize_bright_via_mask_mode,
    _normalize_bright_via_metal_constraint_mode,
    _odd_positive,
    normalize_algorithm_backend,
    normalize_recognition_mode,
    normalize_metal_segmentation_strategy,
    normalize_metal_segmentation_method,
    normalize_metal_sensitivity,
    parse_integer_value_list,
    _serialize_template_images,
    _parse_template_images,
    OperationParameterSpec,
    PipelineStepConfig,
    ContourDebugCandidate,
    DisplaySettings,
    SaveOptions,
    ImageProcessingState,
    BatchImageResult,
    BatchFrameTiming,
    BatchImageMetadata,
    BatchProcessingOptions,
    base_name_from_path,
)
