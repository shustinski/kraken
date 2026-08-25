from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..domain import PolygonData

VIA_SIZE_MODE_RANGE = "range"
VIA_SIZE_MODE_FIXED = "fixed"
VIA_SEARCH_MODE_HYBRID = "hybrid"
VIA_SEARCH_MODE_BLOB = "blob"
VIA_SEARCH_MODE_TEMPLATE = "template"
VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG = "bright_tophat_dog"
VIA_SEARCH_MODE_HEURISTIC = "heuristic"
VIA_CHANNEL_MODE_COLUMNS = "columns"
VIA_CHANNEL_MODE_GRAYSCALE = "grayscale"
VIA_CHANNEL_MODE_RED_BLUE = "red_blue"
ALGORITHM_BACKEND_LEGACY = "legacy"
ALGORITHM_BACKEND_SEM = "sem"
VIA_DISPLAY_MODE_RECTANGLE = "rectangle"
VIA_DISPLAY_MODE_CIRCLE = "circle"


def normalize_via_display_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    return VIA_DISPLAY_MODE_RECTANGLE if text == VIA_DISPLAY_MODE_RECTANGLE else VIA_DISPLAY_MODE_CIRCLE


def normalize_via_size_mode(value: Any) -> str:
    return VIA_SIZE_MODE_FIXED if str(value).strip().lower() == VIA_SIZE_MODE_FIXED else VIA_SIZE_MODE_RANGE


def normalize_via_search_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return VIA_SEARCH_MODE_HEURISTIC
    if text == VIA_SEARCH_MODE_HEURISTIC:
        return VIA_SEARCH_MODE_HEURISTIC
    if text == VIA_SEARCH_MODE_TEMPLATE:
        return VIA_SEARCH_MODE_TEMPLATE
    if text == VIA_SEARCH_MODE_HYBRID:
        return VIA_SEARCH_MODE_HYBRID
    if text == VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG:
        return VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG
    return VIA_SEARCH_MODE_HEURISTIC


def normalize_via_channel_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return VIA_CHANNEL_MODE_GRAYSCALE
    if text == VIA_CHANNEL_MODE_COLUMNS:
        return VIA_CHANNEL_MODE_COLUMNS
    if text in {"gray", "grey", "grayscale"}:
        return VIA_CHANNEL_MODE_GRAYSCALE
    if text in {"rb", "red_blue", "red-blue", "redblue"}:
        return VIA_CHANNEL_MODE_RED_BLUE
    return VIA_CHANNEL_MODE_GRAYSCALE


def _normalize_bright_via_mask_mode(value: Any) -> str:
    return "AND" if str(value or "").strip().upper() == "AND" else "OR"


def _normalize_bright_via_metal_constraint_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"disabled", "off", "none", "false", "0"}:
        return "disabled"
    if text in {"strict", "hard"}:
        return "strict"
    return "soft"


def _odd_positive(value: Any, *, minimum: int) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        size = int(minimum)
    size = max(int(minimum), size)
    if size % 2 == 0:
        size += 1
    return size


def normalize_algorithm_backend(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {ALGORITHM_BACKEND_SEM, "new", "sem_auto", "auto_sem", "legacy_via"}:
        return ALGORITHM_BACKEND_SEM
    return ALGORITHM_BACKEND_LEGACY


RECOGNITION_MODE_DISABLED = "disabled"
RECOGNITION_MODE_VIA = "via"
RECOGNITION_MODE_METAL = "metal"  # legacy value; normalizes to conductors
RECOGNITION_MODE_CONDUCTORS = "conductors"


def normalize_recognition_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {RECOGNITION_MODE_DISABLED, "off", "none", "false", "0"}:
        return RECOGNITION_MODE_DISABLED
    if text in {RECOGNITION_MODE_METAL, "metal", "metallization", "металл"}:
        return RECOGNITION_MODE_CONDUCTORS
    if text in {RECOGNITION_MODE_VIA, "vias", "contact", "contacts"}:
        return RECOGNITION_MODE_VIA
    if text in {RECOGNITION_MODE_CONDUCTORS, "conductor", "wires", "проводники", "traces"}:
        return RECOGNITION_MODE_CONDUCTORS
    return RECOGNITION_MODE_CONDUCTORS


def normalize_metal_segmentation_strategy(value: Any) -> str:
    from ..vision.metal_recovery.segmentation import normalize_metal_segmentation_strategy as _norm

    return _norm(value)


def normalize_metal_segmentation_method(value: Any) -> str:
    """Deprecated alias — maps to segmentation strategy."""
    return normalize_metal_segmentation_strategy(value)


def normalize_metal_sensitivity(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"low", "низкая", "низк"}:
        return "low"
    if text in {"high", "высокая", "высок"}:
        return "high"
    return "medium"


def parse_integer_value_list(payload: Any) -> list[int]:
    if payload in (None, ""):
        return []
    if isinstance(payload, str):
        raw_values = re.split(r"[\s,;]+", payload.strip())
    elif isinstance(payload, (list, tuple, set)):
        raw_values = list(payload)
    else:
        raw_values = [payload]

    values: set[int] = set()
    for raw_value in raw_values:
        if raw_value in (None, ""):
            continue
        text = str(raw_value).strip()
        if not text:
            continue
        try:
            parsed = int(float(text))
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            values.add(parsed)
    return sorted(values)


def _serialize_template_images(payload: Any) -> list[Any]:
    if not isinstance(payload, list):
        return []
    serialized: list[Any] = []
    for item in payload:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, list):
            serialized.append(item)
    return serialized


def _parse_template_images(payload: Any) -> list[Any]:
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, list)]


@dataclass(frozen=True, slots=True)
class OperationParameterSpec:
    name: str
    label: str
    kind: str
    default: Any
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    decimals: int = 3
    options: list[str] = field(default_factory=list)
    tooltip: str = ""


@dataclass(slots=True)
class PipelineStepConfig:
    operation: str
    name: str
    enabled: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> PipelineStepConfig:
        return PipelineStepConfig(
            operation=self.operation,
            name=self.name,
            enabled=self.enabled,
            parameters=dict(self.parameters),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "name": self.name,
            "enabled": self.enabled,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PipelineStepConfig:
        return cls(
            operation=str(payload["operation"]),
            name=str(payload.get("name") or payload["operation"]),
            enabled=bool(payload.get("enabled", True)),
            parameters=dict(payload.get("parameters", {})),
        )


@dataclass(slots=True)
class ContourDebugCandidate:
    contour_index: int
    bbox: tuple[int, int, int, int]
    area: float = 0.0
    perimeter: float = 0.0
    roundness: float = 0.0
    accepted: bool = False
    reason: str = ""
    source: str = ""
    score: float = 0.0
    effective_width: float = 0.0
    width_metric: str = ""
    template_index: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ContourExtractionSettings:
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
    metal_watershed_smoothing_sigma: float = 1.0
    metal_watershed_core_margin: float = 8.0
    metal_watershed_groove_margin: float = 16.0
    metal_watershed_rim_probe_px: int = 6
    metal_watershed_seed_speckle_px: int = 4
    metal_watershed_valley_span_px: int = 5
    metal_watershed_valley_depth: float = 45.0
    metal_random_walker_beta: float = 90.0
    metal_random_walker_iterations: int = 160
    metal_graph_cut_iterations: int = 5
    metal_reconstruction_erode_px: int = 0
    metal_boundary_relief: float = 16.0
    metal_boundary_background_sigma: float = 12.0
    metal_structural_variant: str = "s2"
    metal_adaptive_block_size: int = 0
    metal_adaptive_c: float = 0.0
    metal_adaptive_method: str = "gaussian"
    metal_edge_close_cap_px: int = 9
    metal_edge_watershed_split: bool = True
    metal_edge_watershed_dist_peak_frac: float = 0.38
    # 0 or negative = no limit (always run watershed when enabled); default skips on huge frames.
    metal_edge_watershed_max_pixels: int = 3_000_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "extraction_profile": self.extraction_profile,
            "algorithm_backend": normalize_algorithm_backend(self.algorithm_backend),
            "sem_noise_level": self.sem_noise_level,
            "sem_polarity": self.sem_polarity,
            "sem_preserve_hierarchy": self.sem_preserve_hierarchy,
            "object_type": self.object_type,
            "output_mode": self.output_mode,
            "retrieval_mode": self.retrieval_mode,
            "approximation_mode": self.approximation_mode,
            "epsilon": self.epsilon,
            "epsilon_relative": self.epsilon_relative,
            "preserve_corners": self.preserve_corners,
            "min_polygon_angle": self.min_polygon_angle,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "min_perimeter": self.min_perimeter,
            "min_points": self.min_points,
            "max_perimeter": self.max_perimeter,
            "min_bbox_width": self.min_bbox_width,
            "max_bbox_width": self.max_bbox_width,
            "min_bbox_height": self.min_bbox_height,
            "max_bbox_height": self.max_bbox_height,
            "min_aspect_ratio": self.min_aspect_ratio,
            "max_aspect_ratio": self.max_aspect_ratio,
            "exclude_border_touching": self.exclude_border_touching,
            "min_solidity": self.min_solidity,
            "min_extent": self.min_extent,
            "min_polygon_width_px": self.min_polygon_width_px,
            "min_via_width": self.min_via_width,
            "max_via_width": self.max_via_width,
            "min_via_height": self.min_via_height,
            "max_via_height": self.max_via_height,
            "via_size_mode": normalize_via_size_mode(self.via_size_mode),
            "via_search_mode": normalize_via_search_mode(self.via_search_mode),
            "fixed_via_widths": list(self.fixed_via_widths),
            "fixed_via_heights": list(self.fixed_via_heights),
            "via_channel_mode": normalize_via_channel_mode(self.via_channel_mode),
            "via_white_range_enabled": self.via_white_range_enabled,
            "via_white_range_min": self.via_white_range_min,
            "via_white_range_max": self.via_white_range_max,
            "via_black_range_enabled": self.via_black_range_enabled,
            "via_black_range_min": self.via_black_range_min,
            "via_black_range_max": self.via_black_range_max,
            "via_min_roundness": self.via_min_roundness,
            "via_min_score": self.via_min_score,
            "via_min_contrast": self.via_min_contrast,
            "via_min_edge_coverage": self.via_min_edge_coverage,
            "via_spot_line_suppression": self.via_spot_line_suppression,
            "via_template_min_score": self.via_template_min_score,
            "via_template_images": _serialize_template_images(self.via_template_images),
            "via_template_min_scores": [max(0.0, min(1.0, float(value))) for value in self.via_template_min_scores],
            "via_template_diameters": [max(1, int(value)) for value in self.via_template_diameters],
            "via_template_nms_distance": self.via_template_nms_distance,
            "via_template_scale_min": self.via_template_scale_min,
            "via_template_scale_max": self.via_template_scale_max,
            "via_template_scale_step": self.via_template_scale_step,
            "via_heuristic_polarity": self.via_heuristic_polarity,
            "via_fixed_diameters_text": self.via_fixed_diameters_text,
            "heuristic_background_sigma": self.heuristic_background_sigma,
            "heuristic_analysis_window_scale": self.heuristic_analysis_window_scale,
            "heuristic_min_center_brightness": self.heuristic_min_center_brightness,
            "heuristic_min_center_contrast": self.heuristic_min_center_contrast,
            "heuristic_min_peak_prominence": self.heuristic_min_peak_prominence,
            "heuristic_min_compactness": self.heuristic_min_compactness,
            "heuristic_min_circularity": self.heuristic_min_circularity,
            "heuristic_max_elongation": self.heuristic_max_elongation,
            "heuristic_line_penalty_scale": self.heuristic_line_penalty_scale,
            "heuristic_border_penalty_scale": self.heuristic_border_penalty_scale,
            "heuristic_local_binarize_percentile": self.heuristic_local_binarize_percentile,
            "heuristic_min_abs_peak": self.heuristic_min_abs_peak,
            "heuristic_use_bilateral": self.heuristic_use_bilateral,
            "heuristic_size_tolerance_range": self.heuristic_size_tolerance_range,
            "heuristic_size_tolerance_fixed": self.heuristic_size_tolerance_fixed,
            "heuristic_max_center_drift_ratio": self.heuristic_max_center_drift_ratio,
            "heuristic_max_line_coherence": self.heuristic_max_line_coherence,
            "heuristic_min_edge_sharpness": self.heuristic_min_edge_sharpness,
            "heuristic_contrast_score_min": self.heuristic_contrast_score_min,
            "heuristic_contrast_score_max": self.heuristic_contrast_score_max,
            "heuristic_prominence_score_min": self.heuristic_prominence_score_min,
            "heuristic_prominence_score_max": self.heuristic_prominence_score_max,
            "heuristic_edge_snr_score_min": self.heuristic_edge_snr_score_min,
            "heuristic_edge_snr_score_max": self.heuristic_edge_snr_score_max,
            "heuristic_edge_quality_floor": self.heuristic_edge_quality_floor,
            "heuristic_border_balance_scale": self.heuristic_border_balance_scale,
            "heuristic_seed_percentile": self.heuristic_seed_percentile,
            "heuristic_w_contrast": self.heuristic_w_contrast,
            "heuristic_w_prominence": self.heuristic_w_prominence,
            "heuristic_w_size": self.heuristic_w_size,
            "heuristic_w_compact": self.heuristic_w_compact,
            "heuristic_w_round": self.heuristic_w_round,
            "heuristic_w_balance": self.heuristic_w_balance,
            "heuristic_w_line": self.heuristic_w_line,
            "heuristic_w_border": self.heuristic_w_border,
            "bright_via_diameter_min": self.bright_via_diameter_min,
            "bright_via_diameter_max": self.bright_via_diameter_max,
            "via_output_diameter": self.via_output_diameter,
            "bright_via_clahe_clip_limit": self.bright_via_clahe_clip_limit,
            "bright_via_clahe_tile_grid_size": self.bright_via_clahe_tile_grid_size,
            "bright_via_median_blur_kernel": self.bright_via_median_blur_kernel,
            "bright_via_tophat_kernel_size": self.bright_via_tophat_kernel_size,
            "bright_via_dog_sigma_small": self.bright_via_dog_sigma_small,
            "bright_via_dog_sigma_large": self.bright_via_dog_sigma_large,
            "bright_via_threshold_percentile": self.bright_via_threshold_percentile,
            "bright_via_mask_combine_mode": self.bright_via_mask_combine_mode,
            "bright_via_min_area_factor": self.bright_via_min_area_factor,
            "bright_via_max_area_factor": self.bright_via_max_area_factor,
            "bright_via_min_circularity": self.bright_via_min_circularity,
            "bright_via_min_aspect": self.bright_via_min_aspect,
            "bright_via_max_aspect": self.bright_via_max_aspect,
            "bright_via_bright_center_min_score": self.bright_via_bright_center_min_score,
            "bright_via_metal_constraint_mode": self.bright_via_metal_constraint_mode,
            "bright_via_use_metal_mask": self.bright_via_use_metal_mask,
            "bright_via_metal_fraction_min": self.bright_via_metal_fraction_min,
            "bright_via_max_radial_asymmetry": self.bright_via_max_radial_asymmetry,
            "bright_via_max_edge_likeness": self.bright_via_max_edge_likeness,
            "bright_via_max_line_likeness": self.bright_via_max_line_likeness,
            "bright_via_nms_distance": self.bright_via_nms_distance,
            "bright_via_min_final_score": self.bright_via_min_final_score,
            "bright_via_show_rejected": self.bright_via_show_rejected,
            "bright_via_hard_reject_on_asymmetry": self.bright_via_hard_reject_on_asymmetry,
            "bright_via_hard_reject_on_edge": self.bright_via_hard_reject_on_edge,
            "bright_via_hard_reject_on_line": self.bright_via_hard_reject_on_line,
            "bright_via_min_isolation_score": self.bright_via_min_isolation_score,
            "bright_via_min_annular_contrast": self.bright_via_min_annular_contrast,
            "bright_via_max_annular_contrast": self.bright_via_max_annular_contrast,
            "bright_via_tiled_threshold_size": self.bright_via_tiled_threshold_size,
            "bright_via_scale_kernels_from_diameter": self.bright_via_scale_kernels_from_diameter,
            "debug_enabled": self.debug_enabled,
            "debug_gradient_map_enabled": self.debug_gradient_map_enabled,
            "min_hierarchy_depth": self.min_hierarchy_depth,
            "max_hierarchy_depth": self.max_hierarchy_depth,
            "min_inner_hole_area": self.min_inner_hole_area,
            "max_hole_area_ratio": self.max_hole_area_ratio,
            "conductor_gradient_enabled": self.conductor_gradient_enabled,
            "conductor_gradient_min_strength": self.conductor_gradient_min_strength,
            "conductor_gradient_band_radius": self.conductor_gradient_band_radius,
            "edge_method": self.edge_method,
            "via_gradient_edge_method": self.via_gradient_edge_method,
            "conductor_gradient_edge_method": self.conductor_gradient_edge_method,
            "recognition_mode": normalize_recognition_mode(self.recognition_mode),
            "via_display_show_detected": self.via_display_show_detected,
            "via_display_show_candidates": self.via_display_show_candidates,
            "metal_structural_pipeline": self.metal_structural_pipeline,
            "metal_preset": self.metal_preset,
            "metal_noise_suppression": self.metal_noise_suppression,
            "metal_min_contrast": self.metal_min_contrast,
            "metal_min_object_source_contrast": self.metal_min_object_source_contrast,
            "metal_min_object_rim_contrast": self.metal_min_object_rim_contrast,
            "metal_min_object_rim_area_fraction": self.metal_min_object_rim_area_fraction,
            "metal_contrast_bias": self.metal_contrast_bias,
            "metal_min_hole_source_contrast": self.metal_min_hole_source_contrast,
            "metal_min_hole_source_contrast_fraction": self.metal_min_hole_source_contrast_fraction,
            "metal_segmentation_strategy": normalize_metal_segmentation_strategy(self.metal_segmentation_strategy),
            "metal_auto_contrast_step": self.metal_auto_contrast_step,
            "metal_auto_source_contrast_step": self.metal_auto_source_contrast_step,
            "metal_auto_directional_gap_bridge_px": self.metal_auto_directional_gap_bridge_px,
            "metal_auto_directional_gap_min_source_intensity": (
                self.metal_auto_directional_gap_min_source_intensity
            ),
            "metal_gap_bridge_px": self.metal_gap_bridge_px,
            "metal_speckle_removal_px": self.metal_speckle_removal_px,
            "metal_contour_smooth_px": self.metal_contour_smooth_px,
            "metal_contour_close_px": self.metal_contour_close_px,
            "metal_min_object_area": self.metal_min_object_area,
            "metal_min_trace_width_px": self.metal_min_trace_width_px,
            "metal_max_trace_width_px": self.metal_max_trace_width_px,
            "metal_min_trace_length_px": self.metal_min_trace_length_px,
            "metal_allowed_angles": self.metal_allowed_angles,
            "metal_angle_tolerance_deg": self.metal_angle_tolerance_deg,
            "metal_min_straightness": self.metal_min_straightness,
            "metal_allow_t_junction": self.metal_allow_t_junction,
            "metal_border_handling": self.metal_border_handling,
            "metal_check_contour_validity": self.metal_check_contour_validity,
            "metal_hierarchy_mode": self.metal_hierarchy_mode,
            "metal_min_area": self.metal_min_area,
            "metal_max_area": self.metal_max_area,
            "metal_min_perimeter": self.metal_min_perimeter,
            "metal_max_perimeter": self.metal_max_perimeter,
            "metal_approximation_enabled": self.metal_approximation_enabled,
            "metal_morph_close_radius": self.metal_morph_close_radius,
            "metal_morph_open_radius": self.metal_morph_open_radius,
            "metal_display_show_conductors": self.metal_display_show_conductors,
            "metal_display_show_mask": self.metal_display_show_mask,
            "metal_display_show_contours": self.metal_display_show_contours,
            "metal_display_show_rejected": self.metal_display_show_rejected,
            "metal_display_show_suspicious": self.metal_display_show_suspicious,
            "metal_display_show_border_highlight": self.metal_display_show_border_highlight,
            "metal_debug_visual": self.metal_debug_visual,
            "metal_overlay_opacity": self.metal_overlay_opacity,
            "metal_preprocess_subtract_background": self.metal_preprocess_subtract_background,
            "metal_preprocess_background_sigma_fraction": self.metal_preprocess_background_sigma_fraction,
            "metal_preprocess_clahe_clip": self.metal_preprocess_clahe_clip,
            "metal_preprocess_clahe_grid": self.metal_preprocess_clahe_grid,
            "metal_preprocess_denoise": self.metal_preprocess_denoise,
            "metal_use_wide_conductor_gradient": self.metal_use_wide_conductor_gradient,
            "metal_watershed_smoothing_sigma": self.metal_watershed_smoothing_sigma,
            "metal_watershed_core_margin": self.metal_watershed_core_margin,
            "metal_watershed_groove_margin": self.metal_watershed_groove_margin,
            "metal_watershed_rim_probe_px": self.metal_watershed_rim_probe_px,
            "metal_watershed_seed_speckle_px": self.metal_watershed_seed_speckle_px,
            "metal_watershed_valley_span_px": self.metal_watershed_valley_span_px,
            "metal_watershed_valley_depth": self.metal_watershed_valley_depth,
            "metal_random_walker_beta": self.metal_random_walker_beta,
            "metal_random_walker_iterations": self.metal_random_walker_iterations,
            "metal_graph_cut_iterations": self.metal_graph_cut_iterations,
            "metal_reconstruction_erode_px": self.metal_reconstruction_erode_px,
            "metal_boundary_relief": self.metal_boundary_relief,
            "metal_boundary_background_sigma": self.metal_boundary_background_sigma,
            "metal_structural_variant": self.metal_structural_variant,
            "metal_adaptive_block_size": self.metal_adaptive_block_size,
            "metal_adaptive_c": self.metal_adaptive_c,
            "metal_adaptive_method": self.metal_adaptive_method,
            "metal_edge_close_cap_px": int(self.metal_edge_close_cap_px),
            "metal_edge_watershed_split": bool(self.metal_edge_watershed_split),
            "metal_edge_watershed_dist_peak_frac": float(self.metal_edge_watershed_dist_peak_frac),
            "metal_edge_watershed_max_pixels": int(self.metal_edge_watershed_max_pixels),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ContourExtractionSettings:
        from ..vision.metal_recovery.segmentation import (
            migrate_legacy_metal_settings,
            normalize_metal_adaptive_method,
            resolve_metal_segmentation_strategy,
        )

        legacy_method_only = "metal_segmentation_method" in payload and "metal_segmentation_strategy" not in payload
        payload = migrate_legacy_metal_settings(dict(payload))
        if legacy_method_only:
            payload["metal_segmentation_strategy"] = "legacy_otsu"
        max_area = payload.get("max_area")
        max_perimeter = payload.get("max_perimeter")
        max_bbox_width = payload.get("max_bbox_width")
        max_bbox_height = payload.get("max_bbox_height")
        max_aspect_ratio = payload.get("max_aspect_ratio")
        max_via_width = payload.get("max_via_width")
        max_via_height = payload.get("max_via_height")
        max_hierarchy_depth = payload.get("max_hierarchy_depth")
        min_inner_hole_area = payload.get("min_inner_hole_area")
        max_hole_area_ratio = payload.get("max_hole_area_ratio")
        metal_max_trace_width = payload.get("metal_max_trace_width_px")
        metal_max_area = payload.get("metal_max_area")
        metal_max_perimeter = payload.get("metal_max_perimeter")
        metal_preprocess_denoise = str(payload.get("metal_preprocess_denoise", "low") or "low")
        if metal_preprocess_denoise not in {"low", "medium", "high"}:
            metal_preprocess_denoise = "low"
        white_range_enabled = payload.get("via_white_range_enabled", payload.get("via_white_threshold_enabled", True))
        white_range_min = payload.get("via_white_range_min", payload.get("via_white_threshold", 200))
        white_range_max = payload.get("via_white_range_max", 255)
        black_range_enabled = payload.get("via_black_range_enabled", payload.get("via_black_threshold_enabled", False))
        black_range_min = payload.get("via_black_range_min", 0)
        black_range_max = payload.get("via_black_range_max", payload.get("via_black_threshold", 30))
        if payload.get("via_threshold_range_enabled", False) and "via_white_range_min" not in payload:
            white_range_enabled = True
            white_range_min = payload.get("via_threshold_range_min", white_range_min)
            white_range_max = payload.get("via_threshold_range_max", white_range_max)
        fixed_via_widths = parse_integer_value_list(payload.get("fixed_via_widths"))
        fixed_via_heights = parse_integer_value_list(payload.get("fixed_via_heights"))
        legacy_diameters = [
            max(1, round((width + height) * 0.5))
            for width, height in zip(fixed_via_widths, fixed_via_heights, strict=False)
            if width > 0 and height > 0
        ]
        candidate_diameter_min = int(
            payload.get(
                "bright_via_diameter_min",
                min(legacy_diameters) if legacy_diameters else 6,
            )
        )
        candidate_diameter_max = int(
            payload.get(
                "bright_via_diameter_max",
                max(legacy_diameters) if legacy_diameters else 8,
            )
        )
        output_diameter = int(
            payload.get(
                "via_output_diameter",
                round(
                    sum(legacy_diameters) / len(legacy_diameters)
                    if legacy_diameters
                    else (candidate_diameter_min + candidate_diameter_max) * 0.5
                ),
            )
        )
        return cls(
            extraction_profile=str(payload.get("extraction_profile", "conductors")),
            algorithm_backend=normalize_algorithm_backend(payload.get("algorithm_backend", ALGORITHM_BACKEND_LEGACY)),
            sem_noise_level=str(payload.get("sem_noise_level", "medium") or "medium"),
            sem_polarity=str(payload.get("sem_polarity", "auto") or "auto"),
            sem_preserve_hierarchy=bool(payload.get("sem_preserve_hierarchy", True)),
            object_type=str(payload.get("object_type", "conductor")),
            output_mode=str(payload.get("output_mode", "polygon")),
            retrieval_mode=str(payload.get("retrieval_mode", "RETR_EXTERNAL")),
            approximation_mode=str(payload.get("approximation_mode", "CHAIN_APPROX_SIMPLE")),
            epsilon=float(payload.get("epsilon", 2.0)),
            epsilon_relative=bool(payload.get("epsilon_relative", False)),
            preserve_corners=bool(payload.get("preserve_corners", False)),
            min_polygon_angle=max(0.0, min(180.0, float(payload.get("min_polygon_angle", 0.0)))),
            min_area=float(payload.get("min_area", 0.0)),
            max_area=None if max_area in (None, "", 0, 0.0) else float(max_area),
            min_perimeter=float(payload.get("min_perimeter", 10.0)),
            min_points=max(3, int(payload.get("min_points", 3))),
            max_perimeter=None if max_perimeter in (None, "", 0, 0.0) else float(max_perimeter),
            min_bbox_width=max(0, int(payload.get("min_bbox_width", 0))),
            max_bbox_width=None if max_bbox_width in (None, "", 0, 0.0) else max(1, int(max_bbox_width)),
            min_bbox_height=max(0, int(payload.get("min_bbox_height", 0))),
            max_bbox_height=None if max_bbox_height in (None, "", 0, 0.0) else max(1, int(max_bbox_height)),
            min_aspect_ratio=max(0.0, float(payload.get("min_aspect_ratio", 0.0))),
            max_aspect_ratio=None if max_aspect_ratio in (None, "", 0, 0.0) else float(max_aspect_ratio),
            exclude_border_touching=bool(payload.get("exclude_border_touching", False)),
            min_solidity=max(0.0, float(payload.get("min_solidity", 0.0))),
            min_extent=max(0.0, float(payload.get("min_extent", 0.0))),
            min_polygon_width_px=max(0.0, float(payload.get("min_polygon_width_px", 0.0))),
            min_via_width=max(0, int(payload.get("min_via_width", 0))),
            max_via_width=None if max_via_width in (None, "", 0, 0.0) else max(1, int(max_via_width)),
            min_via_height=max(0, int(payload.get("min_via_height", 0))),
            max_via_height=None if max_via_height in (None, "", 0, 0.0) else max(1, int(max_via_height)),
            via_size_mode=normalize_via_size_mode(payload.get("via_size_mode", VIA_SIZE_MODE_FIXED)),
            via_search_mode=normalize_via_search_mode(payload.get("via_search_mode", VIA_SEARCH_MODE_HEURISTIC)),
            fixed_via_widths=fixed_via_widths,
            fixed_via_heights=fixed_via_heights,
            via_channel_mode=normalize_via_channel_mode(payload.get("via_channel_mode", VIA_CHANNEL_MODE_GRAYSCALE)),
            via_white_range_enabled=bool(white_range_enabled),
            via_white_range_min=max(0, min(255, int(white_range_min))),
            via_white_range_max=max(0, min(255, int(white_range_max))),
            via_black_range_enabled=bool(black_range_enabled),
            via_black_range_min=max(0, min(255, int(black_range_min))),
            via_black_range_max=max(0, min(255, int(black_range_max))),
            via_min_roundness=max(0.0, float(payload.get("via_min_roundness", 40.0))),
            via_min_score=max(0.0, min(1.0, float(payload.get("via_min_score", 0.35)))),
            via_min_contrast=max(0.0, min(255.0, float(payload.get("via_min_contrast", 14.0)))),
            via_min_edge_coverage=max(0.0, min(1.0, float(payload.get("via_min_edge_coverage", 0.45)))),
            via_template_min_score=max(0.0, min(1.0, float(payload.get("via_template_min_score", 0.35)))),
            via_template_images=_parse_template_images(payload.get("via_template_images", [])),
            via_template_min_scores=[
                max(0.0, min(1.0, float(value))) for value in (payload.get("via_template_min_scores", []) or [])
            ],
            via_template_diameters=[max(1, int(value)) for value in (payload.get("via_template_diameters", []) or [])],
            via_template_nms_distance=max(0, int(payload.get("via_template_nms_distance", 4))),
            via_template_scale_min=max(0.1, float(payload.get("via_template_scale_min", 0.9))),
            via_template_scale_max=max(0.1, float(payload.get("via_template_scale_max", 1.1))),
            via_template_scale_step=max(0.01, float(payload.get("via_template_scale_step", 0.1))),
            via_heuristic_polarity=str(payload.get("via_heuristic_polarity", "auto") or "auto"),
            via_fixed_diameters_text=str(payload.get("via_fixed_diameters_text", "6, 8, 10") or "6, 8, 10"),
            heuristic_background_sigma=max(0.1, float(payload.get("heuristic_background_sigma", 25.0))),
            heuristic_analysis_window_scale=max(1.0, float(payload.get("heuristic_analysis_window_scale", 3.0))),
            heuristic_min_center_brightness=max(
                0.0, min(255.0, float(payload.get("heuristic_min_center_brightness", 0.0)))
            ),
            heuristic_min_center_contrast=max(0.0, float(payload.get("heuristic_min_center_contrast", 50.0))),
            heuristic_min_peak_prominence=max(0.0, float(payload.get("heuristic_min_peak_prominence", 50.0))),
            heuristic_min_compactness=max(0.0, float(payload.get("heuristic_min_compactness", 0.9))),
            heuristic_min_circularity=max(
                0.0, min(1.0, float(payload.get("heuristic_min_circularity", 0.40)))
            ),
            heuristic_max_elongation=max(1.0, float(payload.get("heuristic_max_elongation", 2.5))),
            heuristic_line_penalty_scale=max(0.0, float(payload.get("heuristic_line_penalty_scale", 3.0))),
            heuristic_border_penalty_scale=max(0.0, float(payload.get("heuristic_border_penalty_scale", 1.0))),
            heuristic_local_binarize_percentile=max(
                1.0, min(99.0, float(payload.get("heuristic_local_binarize_percentile", 88.0)))
            ),
            heuristic_min_abs_peak=max(0.0, float(payload.get("heuristic_min_abs_peak", 0.0))),
            heuristic_use_bilateral=bool(payload.get("heuristic_use_bilateral", False)),
            heuristic_size_tolerance_range=max(
                0.05, min(0.95, float(payload.get("heuristic_size_tolerance_range", 0.36)))
            ),
            heuristic_size_tolerance_fixed=max(
                0.05, min(0.95, float(payload.get("heuristic_size_tolerance_fixed", 0.26)))
            ),
            heuristic_max_center_drift_ratio=max(
                0.1, min(1.5, float(payload.get("heuristic_max_center_drift_ratio", 0.72)))
            ),
            heuristic_max_line_coherence=max(
                0.0, min(1.0, float(payload.get("heuristic_max_line_coherence", 0.82)))
            ),
            heuristic_min_edge_sharpness=max(0.0, float(payload.get("heuristic_min_edge_sharpness", 0.20))),
            heuristic_contrast_score_min=max(0.0, float(payload.get("heuristic_contrast_score_min", 3.0))),
            heuristic_contrast_score_max=max(0.0, float(payload.get("heuristic_contrast_score_max", 20.0))),
            heuristic_prominence_score_min=max(
                0.0, float(payload.get("heuristic_prominence_score_min", 2.0))
            ),
            heuristic_prominence_score_max=max(
                0.0, float(payload.get("heuristic_prominence_score_max", 25.0))
            ),
            heuristic_edge_snr_score_min=max(0.0, float(payload.get("heuristic_edge_snr_score_min", 0.70))),
            heuristic_edge_snr_score_max=max(0.0, float(payload.get("heuristic_edge_snr_score_max", 2.80))),
            heuristic_edge_quality_floor=max(
                0.0, min(1.0, float(payload.get("heuristic_edge_quality_floor", 0.55)))
            ),
            heuristic_border_balance_scale=max(
                0.0, float(payload.get("heuristic_border_balance_scale", 2.0))
            ),
            heuristic_seed_percentile=max(
                0.0,
                min(
                    100.0,
                    float(
                        payload.get(
                            "heuristic_seed_percentile",
                            payload.get("heuristic_seed_percentile_medium", 90.0),
                        )
                    ),
                ),
            ),
            heuristic_w_contrast=max(0.0, float(payload.get("heuristic_w_contrast", 25.0))),
            heuristic_w_prominence=max(0.0, float(payload.get("heuristic_w_prominence", 20.0))),
            heuristic_w_size=max(0.0, float(payload.get("heuristic_w_size", 20.0))),
            heuristic_w_compact=max(0.0, float(payload.get("heuristic_w_compact", 15.0))),
            heuristic_w_round=max(0.0, float(payload.get("heuristic_w_round", 10.0))),
            heuristic_w_balance=max(0.0, float(payload.get("heuristic_w_balance", 10.0))),
            heuristic_w_line=max(0.0, float(payload.get("heuristic_w_line", 20.0))),
            heuristic_w_border=max(0.0, float(payload.get("heuristic_w_border", 20.0))),
            via_spot_line_suppression=max(0.0, min(1.0, float(payload.get("via_spot_line_suppression", 0.65)))),
            bright_via_diameter_min=max(1, candidate_diameter_min),
            bright_via_diameter_max=max(1, candidate_diameter_max),
            via_output_diameter=max(1, output_diameter),
            bright_via_clahe_clip_limit=max(0.01, float(payload.get("bright_via_clahe_clip_limit", 2.0))),
            bright_via_clahe_tile_grid_size=max(1, int(payload.get("bright_via_clahe_tile_grid_size", 8))),
            bright_via_median_blur_kernel=_odd_positive(payload.get("bright_via_median_blur_kernel", 3), minimum=1),
            bright_via_tophat_kernel_size=_odd_positive(payload.get("bright_via_tophat_kernel_size", 11), minimum=3),
            bright_via_dog_sigma_small=max(0.01, float(payload.get("bright_via_dog_sigma_small", 0.8))),
            bright_via_dog_sigma_large=max(0.02, float(payload.get("bright_via_dog_sigma_large", 2.0))),
            bright_via_threshold_percentile=max(
                90.0, min(99.9, float(payload.get("bright_via_threshold_percentile", 99.0)))
            ),
            bright_via_mask_combine_mode=_normalize_bright_via_mask_mode(
                payload.get("bright_via_mask_combine_mode", "OR")
            ),
            bright_via_min_area_factor=max(0.01, float(payload.get("bright_via_min_area_factor", 0.45))),
            bright_via_max_area_factor=max(0.01, float(payload.get("bright_via_max_area_factor", 1.8))),
            bright_via_min_circularity=max(0.0, float(payload.get("bright_via_min_circularity", 0.30))),
            bright_via_min_aspect=max(0.01, float(payload.get("bright_via_min_aspect", 0.45))),
            bright_via_max_aspect=max(0.01, float(payload.get("bright_via_max_aspect", 2.2))),
            bright_via_bright_center_min_score=max(
                0.0, float(payload.get("bright_via_bright_center_min_score", 140.0))
            ),
            bright_via_metal_constraint_mode=_normalize_bright_via_metal_constraint_mode(
                payload.get(
                    "bright_via_metal_constraint_mode",
                    "soft" if bool(payload.get("bright_via_use_metal_mask", True)) else "disabled",
                )
            ),
            bright_via_use_metal_mask=bool(payload.get("bright_via_use_metal_mask", True)),
            bright_via_metal_fraction_min=max(0.0, min(1.0, float(payload.get("bright_via_metal_fraction_min", 0.3)))),
            bright_via_max_radial_asymmetry=max(0.0, float(payload.get("bright_via_max_radial_asymmetry", 32.0))),
            bright_via_max_edge_likeness=max(0.0, float(payload.get("bright_via_max_edge_likeness", 35.0))),
            bright_via_max_line_likeness=max(0.0, float(payload.get("bright_via_max_line_likeness", 65.0))),
            bright_via_nms_distance=max(0, int(payload.get("bright_via_nms_distance", 5))),
            bright_via_min_final_score=max(0.0, min(100.0, float(payload.get("bright_via_min_final_score", 38.0)))),
            bright_via_show_rejected=bool(payload.get("bright_via_show_rejected", True)),
            bright_via_hard_reject_on_asymmetry=bool(payload.get("bright_via_hard_reject_on_asymmetry", True)),
            bright_via_hard_reject_on_edge=bool(payload.get("bright_via_hard_reject_on_edge", False)),
            bright_via_hard_reject_on_line=bool(payload.get("bright_via_hard_reject_on_line", False)),
            bright_via_min_isolation_score=float(payload.get("bright_via_min_isolation_score", 0.38)),
            bright_via_min_annular_contrast=float(payload.get("bright_via_min_annular_contrast", 6.0)),
            bright_via_max_annular_contrast=float(payload.get("bright_via_max_annular_contrast", 0.0)),
            bright_via_tiled_threshold_size=max(32, int(payload.get("bright_via_tiled_threshold_size", 256))),
            bright_via_scale_kernels_from_diameter=bool(payload.get("bright_via_scale_kernels_from_diameter", True)),
            debug_enabled=bool(payload.get("debug_enabled", True)),
            debug_gradient_map_enabled=bool(payload.get("debug_gradient_map_enabled", False)),
            min_hierarchy_depth=max(0, int(payload.get("min_hierarchy_depth", 0))),
            max_hierarchy_depth=None if max_hierarchy_depth in (None, "", 0, 0.0) else max(0, int(max_hierarchy_depth)),
            min_inner_hole_area=max(
                0.0,
                float(100.0 if min_inner_hole_area in (None, "") else min_inner_hole_area),
            ),
            max_hole_area_ratio=None
            if max_hole_area_ratio in (None, "", 0, 0.0)
            else max(0.0, float(max_hole_area_ratio)),
            conductor_gradient_enabled=bool(payload.get("conductor_gradient_enabled", False)),
            conductor_gradient_min_strength=max(
                0.0, min(255.0, float(payload.get("conductor_gradient_min_strength", 18.0)))
            ),
            conductor_gradient_band_radius=max(0, min(25, int(payload.get("conductor_gradient_band_radius", 3)))),
            edge_method=str(payload.get("edge_method", "sobel") or "sobel"),
            via_gradient_edge_method=str(payload.get("via_gradient_edge_method", "") or ""),
            conductor_gradient_edge_method=str(payload.get("conductor_gradient_edge_method", "") or ""),
            recognition_mode=normalize_recognition_mode(payload.get("recognition_mode", RECOGNITION_MODE_CONDUCTORS)),
            via_display_show_detected=bool(payload.get("via_display_show_detected", True)),
            via_display_show_candidates=bool(payload.get("via_display_show_candidates", True)),
            metal_structural_pipeline=bool(payload.get("metal_structural_pipeline", False)),
            metal_preset=str(payload.get("metal_preset", "standard") or "standard"),
            metal_noise_suppression=max(0, min(100, int(payload.get("metal_noise_suppression", 20)))),
            metal_min_contrast=max(
                1.0,
                min(
                    255.0,
                    float(
                        payload.get(
                            "metal_min_contrast",
                            max(1.0, float(payload.get("metal_contrast_bias", 50.0))),
                        )
                    ),
                ),
            ),
            metal_contrast_bias=max(-50.0, min(50.0, float(payload.get("metal_contrast_bias", 0.0)))),
            metal_min_hole_source_contrast=max(
                0.0,
                min(255.0, float(payload.get("metal_min_hole_source_contrast", 8.0))),
            ),
            metal_min_hole_source_contrast_fraction=max(
                0.0,
                min(1.0, float(payload.get("metal_min_hole_source_contrast_fraction", 0.35))),
            ),
            metal_segmentation_strategy=resolve_metal_segmentation_strategy(
                payload.get("metal_segmentation_strategy", "legacy_otsu"),
                use_wide_conductor_gradient=bool(payload.get("metal_use_wide_conductor_gradient", False)),
            ),
            metal_auto_contrast_step=max(
                0.0,
                min(255.0, float(payload.get("metal_auto_contrast_step", 10.0))),
            ),
            metal_auto_source_contrast_step=max(
                0.0,
                min(255.0, float(payload.get("metal_auto_source_contrast_step", 4.0))),
            ),
            metal_auto_directional_gap_bridge_px=max(
                0,
                min(64, int(payload.get("metal_auto_directional_gap_bridge_px", 3))),
            ),
            metal_auto_directional_gap_min_source_intensity=max(
                0.0,
                min(
                    255.0,
                    float(payload.get("metal_auto_directional_gap_min_source_intensity", 45.0)),
                ),
            ),
            metal_gap_bridge_px=max(
                0, int(payload.get("metal_gap_bridge_px", payload.get("metal_morph_close_radius", 1)) or 0)
            ),
            metal_speckle_removal_px=max(
                0, int(payload.get("metal_speckle_removal_px", payload.get("metal_morph_open_radius", 1)) or 0)
            ),
            metal_contour_smooth_px=max(0.0, float(payload.get("metal_contour_smooth_px", 0.0) or 0.0)),
            metal_contour_close_px=max(0, int(payload.get("metal_contour_close_px", 0) or 0)),
            metal_min_object_area=max(0.0, float(payload.get("metal_min_object_area", 30.0))),
            metal_min_trace_width_px=max(0.5, float(payload.get("metal_min_trace_width_px", 8.0) or 8.0)),
            metal_max_trace_width_px=None
            if metal_max_trace_width in (None, "", 0, 0.0)
            else max(0.5, float(metal_max_trace_width)),
            metal_min_trace_length_px=max(1.0, float(payload.get("metal_min_trace_length_px", 8.0) or 8.0)),
            metal_allowed_angles=str(payload.get("metal_allowed_angles", "free") or "free"),
            metal_angle_tolerance_deg=max(0.5, float(payload.get("metal_angle_tolerance_deg", 7.0) or 7.0)),
            metal_min_straightness=max(0.05, min(1.0, float(payload.get("metal_min_straightness", 0.2) or 0.2))),
            metal_allow_t_junction=bool(payload.get("metal_allow_t_junction", True)),
            metal_border_handling=str(payload.get("metal_border_handling", "mark") or "mark"),
            metal_check_contour_validity=bool(payload.get("metal_check_contour_validity", True)),
            metal_hierarchy_mode=str(payload.get("metal_hierarchy_mode", "full") or "full"),
            metal_min_area=max(
                0.0,
                float(
                    payload.get(
                        "metal_min_area",
                        payload.get("metal_min_object_area", 60.0),
                    )
                ),
            ),
            metal_max_area=None if metal_max_area in (None, "", 0, 0.0) else float(metal_max_area),
            metal_min_perimeter=max(0.0, float(payload.get("metal_min_perimeter", 32.0) or 32.0)),
            metal_max_perimeter=None if metal_max_perimeter in (None, "", 0, 0.0) else float(metal_max_perimeter),
            metal_approximation_enabled=bool(payload.get("metal_approximation_enabled", True)),
            metal_morph_close_radius=max(1, int(payload.get("metal_morph_close_radius", 1) or 1)),
            metal_morph_open_radius=max(0, int(payload.get("metal_morph_open_radius", 1) or 0)),
            metal_display_show_conductors=bool(payload.get("metal_display_show_conductors", True)),
            metal_display_show_mask=bool(payload.get("metal_display_show_mask", False)),
            metal_display_show_contours=bool(payload.get("metal_display_show_contours", True)),
            metal_display_show_rejected=bool(payload.get("metal_display_show_rejected", True)),
            metal_display_show_suspicious=bool(payload.get("metal_display_show_suspicious", True)),
            metal_display_show_border_highlight=bool(payload.get("metal_display_show_border_highlight", False)),
            metal_debug_visual=str(payload.get("metal_debug_visual", "overlay") or "overlay"),
            metal_overlay_opacity=max(0.05, min(1.0, float(payload.get("metal_overlay_opacity", 0.45) or 0.45))),
            metal_use_wide_conductor_gradient=bool(payload.get("metal_use_wide_conductor_gradient", False)),
            metal_watershed_smoothing_sigma=max(
                0.1, min(8.0, float(payload.get("metal_watershed_smoothing_sigma", 1.0) or 1.0))
            ),
            metal_watershed_core_margin=max(
                0.0, min(40.0, float(payload.get("metal_watershed_core_margin", 8.0) or 0.0))
            ),
            metal_watershed_groove_margin=max(
                0.0, min(40.0, float(payload.get("metal_watershed_groove_margin", 16.0) or 0.0))
            ),
            metal_watershed_rim_probe_px=max(1, min(32, int(payload.get("metal_watershed_rim_probe_px", 6) or 1))),
            metal_watershed_seed_speckle_px=max(
                0, min(8, int(payload.get("metal_watershed_seed_speckle_px", 4) or 0))
            ),
            metal_watershed_valley_span_px=max(
                0, min(16, int(payload.get("metal_watershed_valley_span_px", 5) or 0))
            ),
            metal_watershed_valley_depth=max(
                0.0, min(160.0, float(payload.get("metal_watershed_valley_depth", 45.0) or 0.0))
            ),
            metal_random_walker_beta=max(
                1.0, min(400.0, float(payload.get("metal_random_walker_beta", 90.0) or 90.0))
            ),
            metal_random_walker_iterations=max(
                8, min(400, int(payload.get("metal_random_walker_iterations", 160) or 160))
            ),
            metal_graph_cut_iterations=max(
                1, min(16, int(payload.get("metal_graph_cut_iterations", 5) or 5))
            ),
            metal_reconstruction_erode_px=max(
                0, min(16, int(payload.get("metal_reconstruction_erode_px", 0) or 0))
            ),
            metal_boundary_relief=max(
                1.0, min(80.0, float(payload.get("metal_boundary_relief", 16.0) or 16.0))
            ),
            metal_boundary_background_sigma=max(
                2.0,
                min(60.0, float(payload.get("metal_boundary_background_sigma", 12.0) or 12.0)),
            ),
            metal_structural_variant=str(payload.get("metal_structural_variant", "s2") or "s2"),
            metal_adaptive_block_size=max(
                0, min(255, int(payload.get("metal_adaptive_block_size", 0) or 0))
            ),
            metal_adaptive_c=max(
                -64.0, min(64.0, float(payload.get("metal_adaptive_c", 0.0) or 0.0))
            ),
            metal_adaptive_method=normalize_metal_adaptive_method(
                payload.get("metal_adaptive_method", "gaussian")
            ),
            metal_preprocess_subtract_background=bool(
                payload.get("metal_preprocess_subtract_background", True)
            ),
            metal_preprocess_background_sigma_fraction=max(
                0.005,
                min(
                    0.25,
                    float(payload.get("metal_preprocess_background_sigma_fraction", 0.05) or 0.05),
                ),
            ),
            metal_min_object_source_contrast=max(
                0.0,
                min(255.0, float(payload.get("metal_min_object_source_contrast", 12.0))),
            ),
            metal_min_object_rim_contrast=max(
                0.0,
                min(255.0, float(payload.get("metal_min_object_rim_contrast", 36.0))),
            ),
            metal_min_object_rim_area_fraction=max(
                0.000001,
                min(1.0, float(payload.get("metal_min_object_rim_area_fraction", 0.001))),
            ),
            metal_preprocess_clahe_clip=max(
                0.1,
                min(20.0, float(payload.get("metal_preprocess_clahe_clip", 2.0) or 2.0)),
            ),
            metal_preprocess_clahe_grid=max(
                2,
                min(64, int(payload.get("metal_preprocess_clahe_grid", 8) or 8)),
            ),
            metal_preprocess_denoise=metal_preprocess_denoise,
            metal_edge_close_cap_px=max(5, min(21, int(payload.get("metal_edge_close_cap_px", 9) or 9) | 1)),
            metal_edge_watershed_split=bool(payload.get("metal_edge_watershed_split", True)),
            metal_edge_watershed_dist_peak_frac=max(
                0.22,
                min(0.55, float(payload.get("metal_edge_watershed_dist_peak_frac", 0.38) or 0.38)),
            ),
            metal_edge_watershed_max_pixels=int(payload.get("metal_edge_watershed_max_pixels", 3_000_000) or 3_000_000),
        )


@dataclass(slots=True)
class DisplaySettings:
    external_color: str = "#28C76F"
    hole_color: str = "#FF9F43"
    selected_color: str = "#00CFE8"
    via_selection_color: str = "#FACC15"
    conductor_hover_highlight_color: str = "#FB923C"
    vertex_color: str = "#FF4D6D"
    line_width: float = 2.0
    vertex_size: float = 7.0
    fill_opacity: float = 0.18
    show_vertices: bool = True
    show_labels: bool = False
    via_display_mode: str = VIA_DISPLAY_MODE_CIRCLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "external_color": self.external_color,
            "hole_color": self.hole_color,
            "selected_color": self.selected_color,
            "via_selection_color": self.via_selection_color,
            "conductor_hover_highlight_color": self.conductor_hover_highlight_color,
            "vertex_color": self.vertex_color,
            "line_width": self.line_width,
            "vertex_size": self.vertex_size,
            "fill_opacity": self.fill_opacity,
            "show_vertices": self.show_vertices,
            "show_labels": self.show_labels,
            "via_display_mode": normalize_via_display_mode(self.via_display_mode),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DisplaySettings:
        return cls(
            external_color=str(payload.get("external_color", "#28C76F")),
            hole_color=str(payload.get("hole_color", "#FF9F43")),
            selected_color=str(payload.get("selected_color", "#00CFE8")),
            via_selection_color=str(payload.get("via_selection_color", "#FACC15")),
            conductor_hover_highlight_color=str(payload.get("conductor_hover_highlight_color", "#FB923C")),
            vertex_color=str(payload.get("vertex_color", "#FF4D6D")),
            line_width=float(payload.get("line_width", 2.0)),
            vertex_size=float(payload.get("vertex_size", 7.0)),
            fill_opacity=float(payload.get("fill_opacity", 0.18)),
            show_vertices=bool(payload.get("show_vertices", True)),
            show_labels=bool(payload.get("show_labels", False)),
            via_display_mode=normalize_via_display_mode(payload.get("via_display_mode", VIA_DISPLAY_MODE_CIRCLE)),
        )


@dataclass(slots=True)
class SaveOptions:
    save_cif: bool = True
    save_cv: bool = False
    save_json: bool = False
    save_csv: bool = False
    save_txt: bool = False
    save_svg: bool = False
    save_preview: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "save_cif": self.save_cif,
            "save_cv": self.save_cv,
            "save_json": self.save_json,
            "save_csv": self.save_csv,
            "save_txt": self.save_txt,
            "save_svg": self.save_svg,
            "save_preview": self.save_preview,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SaveOptions:
        return cls(
            save_cif=bool(payload.get("save_cif", True)),
            save_cv=bool(payload.get("save_cv", False)),
            save_json=bool(payload.get("save_json", False)),
            save_csv=bool(payload.get("save_csv", False)),
            save_txt=bool(payload.get("save_txt", False)),
            save_svg=bool(payload.get("save_svg", False)),
            save_preview=bool(payload.get("save_preview", False)),
        )


@dataclass(slots=True)
class ImageProcessingState:
    image_path: str
    source_image: Any | None = None
    preprocessed_image: Any | None = None
    pipeline_config: dict[str, Any] | None = None
    mask_image: Any | None = None
    polygons: list[PolygonData] = field(default_factory=list)
    debug_candidates: list[ContourDebugCandidate] = field(default_factory=list)
    debug_gradient_maps: dict[str, Any] = field(default_factory=dict)
    metal_overlay_polygons: dict[str, list[PolygonData]] = field(default_factory=dict)
    loaded_cif_path: str | None = None
    reference_polygons: list[PolygonData] = field(default_factory=list)
    #: Cached ``image_has_changes``; ``None`` means recompute on next query.
    polygons_dirty: bool | None = None


@dataclass(slots=True)
class BatchImageResult:
    image_path: str
    source_image: Any | None
    preprocessed_image: Any | None
    pipeline_config: dict[str, Any] | None
    mask_image: Any | None
    polygons: list[PolygonData]
    debug_candidates: list[ContourDebugCandidate] = field(default_factory=list)
    debug_gradient_maps: dict[str, Any] = field(default_factory=dict)
    metal_overlay_polygons: dict[str, list[PolygonData]] = field(default_factory=dict)
    saved_files: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class BatchFrameTiming:
    image_loading_ms: float = 0.0
    morphology_ms: float = 0.0
    threshold_ms: float = 0.0
    contour_extraction_ms: float = 0.0
    postprocessing_ms: float = 0.0
    saving_ms: float = 0.0
    total_frame_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "image_loading_ms": float(self.image_loading_ms),
            "morphology_ms": float(self.morphology_ms),
            "threshold_ms": float(self.threshold_ms),
            "contour_extraction_ms": float(self.contour_extraction_ms),
            "postprocessing_ms": float(self.postprocessing_ms),
            "saving_ms": float(self.saving_ms),
            "total_frame_ms": float(self.total_frame_ms),
        }


@dataclass(slots=True)
class BatchImageMetadata:
    image_path: str
    polygon_count: int = 0
    saved_files: dict[str, str] = field(default_factory=dict)
    timing: BatchFrameTiming = field(default_factory=BatchFrameTiming)
    worker_pid: int | None = None
    error: str | None = None


@dataclass(slots=True)
class BatchProcessingOptions:
    max_workers: int = 4
    chunk_size: int = 16
    output_directory: str | None = None
    save_options: SaveOptions = field(default_factory=SaveOptions)


def base_name_from_path(path: str) -> str:
    return Path(path).stem
