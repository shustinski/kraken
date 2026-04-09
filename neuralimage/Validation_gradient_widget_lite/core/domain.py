"""Domain models for the extended validation gradient widget."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from pathlib import Path


class ComparisonMode(str, Enum):
    """Define supported overlay operations in the details dialog."""

    OVERLAY_ONLY = "overlay_only"
    XOR = "xor"
    FIRST_MINUS_SECOND = "first_minus_second"
    SECOND_MINUS_FIRST = "second_minus_first"
    DISAGREEMENT = "disagreement"
    GRAYSCALE_DIFF = "grayscale_diff"

    @property
    def label(self) -> str:
        labels = {
            self.OVERLAY_ONLY: "Overlay only",
            self.XOR: "XOR",
            self.FIRST_MINUS_SECOND: "First - second",
            self.SECOND_MINUS_FIRST: "Second - first",
            self.DISAGREEMENT: "Disagreement",
            self.GRAYSCALE_DIFF: "Grayscale difference",
        }
        return labels[self]


class GeometryMode(str, Enum):
    """Select which geometry interpretation drives frame analytics."""

    MASK = "mask"
    POINT = "point"
    AUTO = "auto"

    @property
    def label(self) -> str:
        labels = {
            self.MASK: "Polygons",
            self.POINT: "Points",
            self.AUTO: "Auto",
        }
        return labels[self]


@dataclass(frozen=True, slots=True)
class FolderSpec:
    """Describe one filesystem folder used by the widget."""

    path: Path
    label: str


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Describe one model input branch."""

    model_id: str
    display_name: str
    mask_folder: Path
    prob_folder: Path | None = None
    threshold: float = 0.5


@dataclass(frozen=True, slots=True)
class BuildOptions:
    """Store backend options used to index frames and compute analytics."""

    thumbnail_size: int = 64
    recursive: bool = True
    file_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    max_workers: int = max(1, os.cpu_count() or 4)
    progress_update_interval: int = 32
    cache_enabled: bool = True
    analysis_max_side: int = 1024
    comparison_mode: ComparisonMode = ComparisonMode.DISAGREEMENT
    geometry_mode: GeometryMode = GeometryMode.MASK
    mask_threshold: float = 0.5
    boundary_radius: int = 1
    point_match_radius: float = 3.0
    point_extraction_mode: str = "component_centroids"
    confidence_uncertainty_delta: float = 0.10
    point_confidence_radius: int = 3
    polygon_confidence_summary: str = "weighted"
    export_top_k: int = 32
    export_neighbor_radius: int = 1


@dataclass(frozen=True, slots=True)
class FrameIdentity:
    """Store stable frame identifiers and matrix coordinates for one tile."""

    frame_id: int
    base_id: int | None = None
    tile_x: int | None = None
    tile_y: int | None = None
    source_key: str | None = None
    sequence_id: str | None = None


@dataclass(slots=True)
class LabeledModelMetrics:
    """Store GT-based polygon metrics for one model on one frame."""

    soft_dice: float
    soft_iou: float
    ssim: float
    dice: float
    iou: float
    precision: float
    recall: float
    count_error: float
    connected_component_error: float
    hausdorff_distance: float
    hausdorff_similarity: float
    centroid_distance: float
    centroid_similarity: float
    mae: float
    rmse: float
    boundary_f1: float | None
    delta_connected_components: float
    break_count: int
    false_bridge_count: int
    skeleton_length_delta: float | None
    quality_score: float
    error_score: float


@dataclass(slots=True)
class ModelDiagnosticMetrics:
    """Store non-GT diagnostics for one mask/polygon model on one frame."""

    area_fraction: float
    component_count: int
    skeleton_length: float
    proxy_score: float


@dataclass(slots=True)
class PointDiagnosticMetrics:
    """Store non-GT diagnostics for one point-based model on one frame."""

    point_count: int
    mean_radius: float
    mean_peak_intensity: float
    false_spot_ratio: float
    proxy_score: float


@dataclass(slots=True)
class PointLabeledMetrics:
    """Store GT-based point metrics for one model on one frame."""

    precision_at_radius: float
    recall_at_radius: float
    f1_at_radius: float
    mean_localization_error: float
    localization_score: float
    chamfer_score: float
    hausdorff_score: float
    count_error: float
    matched_count: int
    predicted_count: int
    target_count: int
    quality_score: float
    error_score: float


@dataclass(slots=True)
class MaskAgreementMetrics:
    """Store pairwise model-agreement metrics for mask/polygon geometry."""

    soft_dice: float
    soft_iou: float
    ssim: float
    dice: float
    iou: float
    hausdorff_distance: float
    hausdorff_similarity: float
    centroid_distance: float
    centroid_similarity: float
    mae: float
    rmse: float
    count_agreement: float
    agreement_score: float


@dataclass(slots=True)
class PointAgreementMetrics:
    """Store pairwise model-agreement metrics for point geometry."""

    precision_at_radius: float
    recall_at_radius: float
    f1_at_radius: float
    mean_localization_error: float
    localization_agreement: float
    count_agreement: float
    matched_count: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    point_count_a: int
    point_count_b: int
    agreement_score: float


@dataclass(frozen=True, slots=True)
class PolygonConfidencePipelineConfig:
    """Store configurable parameters for polygon confidence extraction from raw probability maps."""

    gaussian_sigma: float = 0.35
    median_radius: int = 0
    local_normalization_radius: int = 9
    local_normalization_strength: float = 0.45
    hysteresis_low_ratio: float = 0.60
    hysteresis_low_floor: float = 0.08
    elongated_vertical_radius: int = 5
    elongated_horizontal_radius: int = 1
    elongated_min_aspect_ratio: float = 2.2
    elongated_min_area: int = 6
    dominant_min_area: int = 12
    dominant_min_mean_probability: float = 0.52
    dominant_min_aspect_ratio: float = 3.0
    dominant_min_extent: float = 0.72
    dominant_large_area: int = 64
    dominant_lock_radius: int = 1
    large_polygon_low_scale: float = 0.78
    large_polygon_min_area: int = 18
    large_polygon_min_major_span: int = 10
    large_polygon_min_extent: float = 0.38
    large_polygon_min_aspect_ratio: float = 1.6
    large_polygon_band_expand: int = 3
    large_polygon_roi_padding: int = 4
    large_polygon_seed_low_scale: float = 0.58
    large_polygon_major_close_radius: int = 2
    large_polygon_minor_close_radius: int = 1
    large_polygon_barrier_delta: float = 0.08
    large_polygon_barrier_coverage_min: float = 0.78
    small_low_scale: float = 0.90
    small_high_scale: float = 0.90
    small_min_area: int = 1
    small_max_area: int = 128
    small_peak_floor: float = 0.10
    small_mean_floor: float = 0.05
    adaptive_radius: int = 9
    adaptive_low_offset: float = 0.015
    adaptive_high_offset: float = 0.05
    separation_core_min_area: int = 2
    separation_roi_padding: int = 2
    separation_boundary_low_weight: float = 0.55
    separation_boundary_contrast_weight: float = 0.25
    separation_boundary_uncertainty_weight: float = 0.20
    separation_barrier_threshold: float = 0.48
    separation_barrier_dilate_radius: int = 0
    separation_bridge_probability_max: float = 0.42
    separation_bridge_barrier_threshold: float = 0.52
    merge_iou_threshold: float = 0.25
    merge_distance: int = 2
    enable_watershed: bool = True
    watershed_seed_min_area: int = 2
    hole_probability_scale: float = 0.90
    hole_probability_max: float = 0.30
    hole_min_area: int = 4
    spill_large_area_fraction: float = 0.12
    spill_large_extent: float = 0.72
    spill_low_texture_max: float = 0.06
    spill_trim_delta: float = 0.04
    spill_boundary_separation_max: float = 0.10
    spill_peak_margin_max: float = 0.10
    spill_ribbon_aspect_min: float = 3.0
    spill_border_coverage_min: float = 0.90
    spill_mean_probability_max: float = 0.82
    spill_cross_axis_max: float = 0.18
    spill_prominence_min: float = 0.035
    spill_strong_axis_coverage_min: float = 0.55
    spill_strong_area_fraction_min: float = 0.22
    boundary_snap_min_aspect: float = 3.0
    boundary_snap_profile_quantile: float = 0.35
    boundary_snap_min_drop: float = 0.08
    boundary_snap_min_retained_fraction: float = 0.65
    valley_minor_coverage_min: float = 0.70


@dataclass(slots=True)
class PolygonConfidenceDebugCandidate:
    """Store debug metadata for one candidate object proposed by a confidence branch."""

    object_id: int
    branch: str
    source_branches: tuple[str, ...]
    accepted: bool
    area: int
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    aspect_ratio: float
    elongation: float
    peak_probability: float
    mean_probability: float
    extent: float
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class PolygonConfidenceDebugData:
    """Store intermediate masks, candidate metadata, and stage timings for polygon confidence debugging."""

    preprocessed_probability: Any | None = None
    locally_normalized_probability: Any | None = None
    boundary_cues: Any | None = None
    low_mask: Any | None = None
    high_mask: Any | None = None
    adaptive_low_mask: Any | None = None
    adaptive_high_mask: Any | None = None
    core_seed_mask: Any | None = None
    candidate_region_mask: Any | None = None
    thin_barrier_map: Any | None = None
    bridge_cut_mask: Any | None = None
    barrier_stop_mask: Any | None = None
    branch_masks: dict[str, Any] = field(default_factory=dict)
    merged_mask: Any | None = None
    object_labels: Any | None = None
    candidate_rows: tuple[PolygonConfidenceDebugCandidate, ...] = ()
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PolygonObjectConfidence:
    """Store object-level confidence statistics for one predicted polygon."""

    object_id: int
    area: int
    area_fraction: float
    centroid_x: float
    centroid_y: float
    core_confidence: float
    boundary_uncertainty: float
    weighted_confidence: float
    summary_confidence: float
    mean_probability: float
    mean_confidence: float
    median_confidence: float
    min_confidence: float
    max_confidence: float
    low_percentile_confidence: float
    uncertain_fraction: float
    transition_width_mean: float
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    aspect_ratio: float
    elongation: float
    source_branch: str = "merged"
    source_branches: tuple[str, ...] = ()


@dataclass(slots=True)
class PolygonConfidenceMetrics:
    """Store internal polygon confidence derived from one grayscale mask."""

    frame_uncertainty_score: float
    uncertain_support_fraction: float
    top_uncertainty_mean: float
    largest_uncertain_region_fraction: float
    mean_object_confidence: float
    mean_core_confidence: float
    mean_boundary_uncertainty: float
    mean_weighted_confidence: float
    mean_object_probability: float
    uncertain_fraction: float
    mean_transition_width: float
    object_area_fraction: float
    polygon_count: int
    summary_metric: str
    objects: tuple[PolygonObjectConfidence, ...] = ()
    debug_data: PolygonConfidenceDebugData | None = None


@dataclass(slots=True)
class PointObjectConfidence:
    """Store object-level confidence statistics for one detected point."""

    object_id: int
    x: float
    y: float
    radius: float
    point_probability: float
    center_confidence: float
    local_confidence: float
    local_contrast: float


@dataclass(slots=True)
class PointConfidenceMetrics:
    """Store internal point confidence derived from one grayscale mask."""

    frame_uncertainty_score: float
    uncertain_support_fraction: float
    top_uncertainty_mean: float
    largest_uncertain_region_fraction: float
    mean_point_confidence: float
    mean_center_confidence: float
    mean_local_confidence: float
    mean_point_probability: float
    mean_point_contrast: float
    point_count: int
    objects: tuple[PointObjectConfidence, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelAggregateScore:
    """Store one model-level supervised aggregation over labeled frames."""

    model_id: str
    display_name: str
    labeled_frame_count: int
    mean_supervised_score: float
    median_supervised_score: float
    rank: int


@dataclass(slots=True)
class FrameAnalysisSummary:
    """Store the full analytics summary used by the matrix and export flows."""

    is_labeled: bool
    disagreement_score: float
    temporal_instability: float
    structural_anomaly: float
    labeled_best_quality: float | None
    labeled_mean_quality: float | None
    export_priority_score: float
    metric_values: dict[str, float] = field(default_factory=dict)
    model_metrics: dict[str, Any] = field(default_factory=dict)
    model_confidence: dict[str, Any] = field(default_factory=dict)
    model_diagnostics: dict[str, Any] = field(default_factory=dict)
    pairwise_metrics: tuple[dict[str, Any], ...] = ()
    notes: tuple[str, ...] = ()
    frame_type: str = "polygon"


@dataclass(slots=True)
class FrameRecord:
    """Store one matched frame and its derived analytics."""

    key: str
    display_name: str
    identity: FrameIdentity | None = None
    score: float = 0.0
    absolute_score: float | None = None
    relative_score: float | None = None
    score_percentile: float | None = None
    score_ready: bool = False
    # Legacy lite fields retained for backward compatibility.
    first_path: str = ""
    second_path: str = ""
    base_path: str | None = None
    original_path: str | None = None
    gt_path: str | None = None
    model_mask_paths: dict[str, str] = field(default_factory=dict)
    model_prob_paths: dict[str, str] = field(default_factory=dict)
    summary: FrameAnalysisSummary | None = None


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Store the indexed matrix and analytics payload."""

    records: tuple[FrameRecord, ...] = ()
    model_specs: tuple[ModelSpec, ...] = ()
    original_folder: FolderSpec | None = None
    gt_folder: FolderSpec | None = None
    # Legacy lite aliases retained for compatibility with older call sites.
    first_folder: FolderSpec | None = None
    second_folder: FolderSpec | None = None
    base_folder: FolderSpec | None = None
    options: BuildOptions = field(default_factory=BuildOptions)
    min_score: float = 0.0
    max_score: float = 0.0
    eligible_key_count: int = 0
    scores_computed: bool = False
    best_match_key: str | None = None
    min_absolute_score: float | None = None
    max_absolute_score: float | None = None
    selected_metric_key: str = "overall_frame_score"
    model_ranking: tuple[ModelAggregateScore, ...] = ()
    available_metric_keys: tuple[str, ...] = (
        "overall_frame_score",
        "export_priority_score",
        "model_model_score",
        "model_labeled_score",
        "disagreement_score",
        "labeled_best_quality",
        "labeled_mean_quality",
    )
