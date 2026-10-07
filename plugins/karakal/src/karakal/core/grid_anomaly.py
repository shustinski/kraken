"""Detect damaged cells in regular grid-like frames with OpenCV."""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from .algorithm_version import GRID_DAMAGE_ALGORITHM_VERSION
from .backend_constants import CACHE_DIR
from .cache_utils import atomic_pickle_dump, trim_directory_by_bytes
from .grid_packed import pack_features, pack_points
from .grid_template import CellShapeTemplate, CellTemplateMatcher, build_cell_shape_template, geometry_thresholds
from .performance import load_performance_config
from .profiling import current_profiler, profile_stage

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is optional at runtime
    cv2 = None


# Bump the number in algorithm_version.py when results change; it also sets the app version.
GRID_DAMAGE_CACHE_DIR = CACHE_DIR / "grid_damage"
GRID_DAMAGE_CACHE_MAX_FILES = 20000
GRID_DAMAGE_CACHE_TRIM_INTERVAL_SECONDS = 300.0
_grid_damage_cache_last_trim = 0.0
# JPEG crumbs are a few pixels; real debris is a painted piece at least this large.
DEBRIS_MIN_AREA_PX = 24.0
# Upper bound of the operator's minimum debris size setting.
DEBRIS_MIN_AREA_MAX_PX = 200
# The analysis itself keeps debris down to this size, whatever the operator's setting:
# fragment grouping, gates and verdicts on cells never depend on that setting.
# The operator's size only hides debris marks at the very end (and on display).
DEBRIS_ANALYSIS_FLOOR_PX = 4.0
# Score of a piece that is debris only because no other filter claimed it.
LEFTOVER_DEBRIS_SCORE = 0.75
# feature_snapshot flag on such pieces: they are not cells and skip confidence fill.
LEFTOVER_DEBRIS_FEATURE = "leftover_debris"
# feature_snapshot key with the contour area in pixels, the value the debris size gate compares.
# Not a calibration distance key, so it does not move operator examples.
CONTOUR_AREA_FEATURE = "contour_area_px"
# A hole this much larger (share of the cell area) than holes of normal cells breaks the cell.
HOLE_EXCESS_RATIO = 0.03
# Score of a lattice slot cut by the frame edge.
EDGE_CLIP_SCORE = 0.75
# Template seed cells taken from one frame, so a few frames do not dominate the shape.
TEMPLATE_SEEDS_PER_FRAME = 40
# A piece below this share of the template cell area is debris, not a cell.
CELL_MIN_AREA_RATIO = 0.60
# Pieces of one slot are one broken cell only if together they stay this close to the template.
FRAGMENT_UNION_MAX_DEVIATION = 0.45
# Pixels that must differ deeper than the depth limit to call the shape broken (single pixels are noise).
DEEP_PIXELS_MIN = 4
# Conductor grain covers 12-30% of its zone with mask; below this a "zone" is a sparse field.
ZONE_MIN_DENSITY = 0.08
_LOGGER = logging.getLogger(__name__)
GRID_DAMAGE_REASON_TYPES = (
    "filled_cell",
    "partial_filled_cell",
    "small_artifact",
    "broken_geometry",
    "merged_contour",
    "split_cell",
    "unknown_anomaly",
    "edge_clipped_cell",
    "class_conflict",
    "conductor_zone",
)
_GRID_DAMAGE_REASON_TYPE_SET = set(GRID_DAMAGE_REASON_TYPES)
GRID_CELL_REPRESENTATION_MODES = ("confidence", "binary")
GRID_DERIVED_LAYER_KEYS = ("derived_conflict",)
GRID_CLASS_CONFLICT_MIN_AREA = 6
GRID_CLASS_CONFLICT_ALGORITHM_VERSION = "grid_class_conflict.v2"
# Back-compat aliases for earlier derived-XOR naming.
GRID_XOR_RESIDUAL_MIN_AREA = GRID_CLASS_CONFLICT_MIN_AREA
GRID_XOR_RESIDUAL_ALGORITHM_VERSION = GRID_CLASS_CONFLICT_ALGORITHM_VERSION


@dataclass(frozen=True, slots=True)
class GridDamageSeverityThresholds:
    ok: float = 0.05
    low: float = 0.15
    medium: float = 0.35
    high: float = 0.60

    def level_for_score(self, score: float) -> str:
        value = float(np.clip(score, 0.0, 1.0))
        if value <= self.ok:
            return "OK"
        if value <= self.low:
            return "LOW"
        if value <= self.medium:
            return "MEDIUM"
        if value <= self.high:
            return "HIGH"
        return "CRITICAL"


@dataclass(frozen=True, slots=True)
class GridDamageAnalysisConfig:
    cell_representation: str = "confidence"
    threshold_mode: str = "otsu"
    adaptive_block_size: int = 31
    adaptive_c: float = -2.0
    blur_radius: int = 3
    morphology_open: int = 1
    morphology_close: int = 1
    min_contour_area: float = 12.0
    min_cell_size: int = 4
    axis_cluster_tolerance_ratio: float = 0.55
    slot_match_tolerance_ratio: float = 0.62
    filled_ratio_delta: float = 0.22
    filled_ratio_absolute: float = 0.54
    bad_score_threshold: float = 0.72
    merged_size_ratio: float = 1.58
    merged_area_ratio: float = 1.55
    geometry_solidity_limit: float = 0.70
    geometry_iou_threshold: float = 0.58
    centroid_mismatch_ratio: float = 0.48
    min_grid_candidate_cells: int = 6
    expected_rows: int | None = None
    expected_cols: int | None = None
    severity_thresholds: GridDamageSeverityThresholds = field(default_factory=GridDamageSeverityThresholds)
    include_debug_payload: bool = False
    debug: bool = False
    debug_dir: str | None = None
    enabled_reason_types: tuple[str, ...] | None = GRID_DAMAGE_REASON_TYPES
    scoring_mode: str = "legacy"
    fill_sensitivity: int = 60
    debris_sensitivity: int = 75
    geometry_sensitivity: int = 40
    merge_sensitivity: int = 35
    # Operator setting: mask pieces smaller than this are not debris.
    debris_min_area_px: int = int(DEBRIS_MIN_AREA_PX)
    calibration_reference: tuple[tuple[str, float], ...] = ()
    calibration_examples: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    example_influence: float = 0.5
    calibration_fingerprint: str = ""

    def normalized(self) -> "GridDamageAnalysisConfig":
        block_size = max(3, int(self.adaptive_block_size) | 1)
        enabled_reason_types = None
        if self.enabled_reason_types is not None:
            enabled_reason_types = tuple(
                reason
                for reason in GRID_DAMAGE_REASON_TYPES
                if reason in {str(item) for item in self.enabled_reason_types}
            )
        return GridDamageAnalysisConfig(
            cell_representation=(
                "binary" if str(self.cell_representation or "confidence").strip().lower() == "binary" else "confidence"
            ),
            threshold_mode=str(self.threshold_mode or "otsu").strip().lower(),
            adaptive_block_size=block_size,
            adaptive_c=float(self.adaptive_c),
            blur_radius=max(0, int(self.blur_radius)),
            morphology_open=max(0, int(self.morphology_open)),
            morphology_close=max(0, int(self.morphology_close)),
            min_contour_area=max(1.0, float(self.min_contour_area)),
            min_cell_size=max(2, int(self.min_cell_size)),
            axis_cluster_tolerance_ratio=max(0.05, float(self.axis_cluster_tolerance_ratio)),
            slot_match_tolerance_ratio=max(0.10, float(self.slot_match_tolerance_ratio)),
            filled_ratio_delta=max(0.0, float(self.filled_ratio_delta)),
            filled_ratio_absolute=max(0.05, float(self.filled_ratio_absolute)),
            bad_score_threshold=max(0.05, min(1.0, float(self.bad_score_threshold))),
            merged_size_ratio=max(1.05, float(self.merged_size_ratio)),
            merged_area_ratio=max(1.05, float(self.merged_area_ratio)),
            geometry_solidity_limit=max(0.20, min(0.98, float(self.geometry_solidity_limit))),
            geometry_iou_threshold=max(0.05, min(0.98, float(self.geometry_iou_threshold))),
            centroid_mismatch_ratio=max(0.05, min(0.98, float(self.centroid_mismatch_ratio))),
            min_grid_candidate_cells=max(1, int(self.min_grid_candidate_cells)),
            expected_rows=None if self.expected_rows is None else max(1, int(self.expected_rows)),
            expected_cols=None if self.expected_cols is None else max(1, int(self.expected_cols)),
            severity_thresholds=self.severity_thresholds,
            include_debug_payload=bool(self.include_debug_payload),
            debug=bool(self.debug),
            debug_dir=self.debug_dir,
            enabled_reason_types=enabled_reason_types,
            scoring_mode="calibrated" if str(self.scoring_mode or "legacy").strip().lower() == "calibrated" else "legacy",
            fill_sensitivity=max(0, min(100, int(self.fill_sensitivity))),
            debris_sensitivity=max(0, min(100, int(self.debris_sensitivity))),
            geometry_sensitivity=max(0, min(100, int(self.geometry_sensitivity))),
            merge_sensitivity=max(0, min(100, int(self.merge_sensitivity))),
            debris_min_area_px=max(0, min(DEBRIS_MIN_AREA_MAX_PX, int(self.debris_min_area_px))),
            calibration_reference=tuple(
                (str(key), float(value))
                for key, value in self.calibration_reference
                if str(key)
            ),
            calibration_examples=tuple(self.calibration_examples),
            example_influence=max(0.0, min(1.0, float(self.example_influence))),
            calibration_fingerprint=str(self.calibration_fingerprint or ""),
        )

    def cache_payload(self) -> dict[str, Any]:
        data = asdict(self.normalized())
        data["include_debug_payload"] = False
        data["debug"] = False
        data["debug_dir"] = None
        return data


@dataclass(frozen=True, slots=True)
class GridCellReferenceProfile:
    median_width: float
    median_height: float
    median_area: float
    median_fill: float
    median_interior_fill: float
    median_center_fill: float
    median_aspect: float
    candidate_count: int = 0
    seed_count: int = 0
    frame_id: str = ""
    frame_path: str = ""
    # Mean silhouette of the layer cell; None keeps the size-only scoring.
    shape_template: CellShapeTemplate | None = None
    # Bank of normal cells of the run (normal_bank.NormalBank). With it, frames go through the
    # grid-free cell-error analysis (cell_error_analysis); without it, through the lattice analysis.
    normal_bank: Any = None

    def cache_payload(self) -> dict[str, Any]:
        return {
            "normal_bank_id": None if self.normal_bank is None else str(self.normal_bank.bank_id),
            "shape_template": None if self.shape_template is None else self.shape_template.cache_payload(),
            "median_width": round(float(self.median_width), 6),
            "median_height": round(float(self.median_height), 6),
            "median_area": round(float(self.median_area), 6),
            "median_fill": round(float(self.median_fill), 6),
            "median_interior_fill": round(float(self.median_interior_fill), 6),
            "median_center_fill": round(float(self.median_center_fill), 6),
            "median_aspect": round(float(self.median_aspect), 6),
            "candidate_count": int(self.candidate_count),
            "seed_count": int(self.seed_count),
            "frame_id": str(self.frame_id or ""),
            "frame_path": str(self.frame_path or ""),
        }


@dataclass(frozen=True, slots=True)
class GridCellAnalysisResult:
    row: int
    col: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    contour_id: int | None
    status: str
    score: float
    reasons: tuple[str, ...] = ()
    feature_cluster_id: int | None = None
    feature_cluster_label: str = ""
    consistency_score: float = 0.0
    consistency_reasons: tuple[str, ...] = ()
    mean_confidence: float | None = None
    uncertain_pixel_ratio: float | None = None
    border_uncertainty: float | None = None
    feature_snapshot: tuple[tuple[str, float], ...] = ()
    outline: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        # A run keeps every cell of every frame; tuples of pairs made a frame
        # cost megabytes. Packed values read back as the same pairs.
        object.__setattr__(self, "feature_snapshot", pack_features(self.feature_snapshot))
        object.__setattr__(self, "outline", pack_points(self.outline))

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            type(self),
            (
                self.row,
                self.col,
                self.bbox,
                self.centroid,
                self.contour_id,
                self.status,
                self.score,
                self.reasons,
                self.feature_cluster_id,
                self.feature_cluster_label,
                self.consistency_score,
                self.consistency_reasons,
                self.mean_confidence,
                self.uncertain_pixel_ratio,
                self.border_uncertainty,
                self.feature_snapshot,
                self.outline,
            ),
        )

    @property
    def column(self) -> int:
        return self.col

    @property
    def center_x(self) -> float:
        return float(self.centroid[0])

    @property
    def center_y(self) -> float:
        return float(self.centroid[1])

    @property
    def left(self) -> int:
        return int(self.bbox[0])

    @property
    def top(self) -> int:
        return int(self.bbox[1])

    @property
    def width(self) -> int:
        return int(self.bbox[2])

    @property
    def height(self) -> int:
        return int(self.bbox[3])


@dataclass(frozen=True, slots=True)
class GridDebugPayload:
    threshold: np.ndarray | None = None
    contours: tuple[tuple[tuple[int, int], ...], ...] = ()

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return type(self), (self.threshold, self.contours)


@dataclass(frozen=True, slots=True)
class GridCellFeatureCluster:
    cluster_id: int
    label: str
    member_count: int
    cell_indexes: tuple[int, ...]
    contour_ids: tuple[int, ...]
    bbox: tuple[int, int, int, int]
    mean_score: float
    max_score: float
    dominant_status: str
    dominant_reason: str
    feature_mean: tuple[float, ...] = ()

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            type(self),
            (
                self.cluster_id,
                self.label,
                self.member_count,
                self.cell_indexes,
                self.contour_ids,
                self.bbox,
                self.mean_score,
                self.max_score,
                self.dominant_status,
                self.dominant_reason,
                self.feature_mean,
            ),
        )


@dataclass(frozen=True, slots=True)
class GridFrameAnalysisResult:
    frame_id: str
    frame_path: str
    image_width: int
    image_height: int
    grid_rows: int
    grid_cols: int
    total_expected_cells: int
    detected_cells: int
    normal_cells: int
    suspicious_cells: int
    broken_cells: int
    missing_cells: int
    artifact_cells: int
    damage_score: float
    severity_level: str
    grid_detected: bool = False
    per_cell_results: tuple[GridCellAnalysisResult, ...] = ()
    component_count: int = 0
    x_axes: tuple[float, ...] = ()
    y_axes: tuple[float, ...] = ()
    cell_width: int = 0
    cell_height: int = 0
    feature_clusters: tuple[GridCellFeatureCluster, ...] = ()
    debug: GridDebugPayload | None = None
    zone_skipped_components: int = 0
    model_file_status: str = ""

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            type(self),
            (
                self.frame_id,
                self.frame_path,
                self.image_width,
                self.image_height,
                self.grid_rows,
                self.grid_cols,
                self.total_expected_cells,
                self.detected_cells,
                self.normal_cells,
                self.suspicious_cells,
                self.broken_cells,
                self.missing_cells,
                self.artifact_cells,
                self.damage_score,
                self.severity_level,
                self.grid_detected,
                self.per_cell_results,
                self.component_count,
                self.x_axes,
                self.y_axes,
                self.cell_width,
                self.cell_height,
                self.feature_clusters,
                self.debug,
                self.zone_skipped_components,
                self.model_file_status,
            ),
        )

    @property
    def cells(self) -> tuple[GridCellAnalysisResult, ...]:
        return self.per_cell_results

    @property
    def score(self) -> float:
        return float(self.damage_score)

    @property
    def bad_cells(self) -> int:
        return int(self.suspicious_cells + self.broken_cells + self.missing_cells + self.artifact_cells)


# Backward-compatible names used by the old grid-check UI.
GridCellAnomaly = GridCellAnalysisResult
GridCellAnomalyResult = GridFrameAnalysisResult


def _dedupe_nested_grid_cells(cells: list[GridCellAnalysisResult]) -> list[GridCellAnalysisResult]:
    """Keep the larger contour when an inner hole shares almost the same box."""

    if len(cells) <= 1:
        return list(cells)
    ordered = sorted(cells, key=lambda cell: float(cell.bbox[2]) * float(cell.bbox[3]), reverse=True)
    points = np.array([[float(cell.centroid[0]), float(cell.centroid[1])] for cell in ordered], dtype=np.float64)
    sizes = np.array([max(float(cell.bbox[2]), float(cell.bbox[3]), 1.0) for cell in ordered], dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
    except Exception:
        tree = None
    kept: list[GridCellAnalysisResult] = []
    kept_indexes: set[int] = set()
    for index, cell in enumerate(ordered):
        if tree is None:
            candidates = list(kept_indexes)
        else:
            candidates = [other for other in tree.query_ball_point(points[index], r=float(sizes[index]) * 0.85) if other in kept_indexes]
        if any(_bbox_iou(cell.bbox, ordered[other].bbox) >= 0.70 for other in candidates):
            continue
        kept.append(cell)
        kept_indexes.add(index)
    return kept


def _bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    first_left, first_top, first_width, first_height = (int(value) for value in first[:4])
    second_left, second_top, second_width, second_height = (int(value) for value in second[:4])
    intersection_width = max(
        0, min(first_left + first_width, second_left + second_width) - max(first_left, second_left)
    )
    intersection_height = max(0, min(first_top + first_height, second_top + second_height) - max(first_top, second_top))
    intersection = int(intersection_width * intersection_height)
    union = int(first_width * first_height + second_width * second_height - intersection)
    return 0.0 if union <= 0 else float(intersection / union)


def _bbox_intersection_area(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> int:
    first_left, first_top, first_width, first_height = (int(value) for value in first[:4])
    second_left, second_top, second_width, second_height = (int(value) for value in second[:4])
    intersection_width = max(
        0, min(first_left + first_width, second_left + second_width) - max(first_left, second_left)
    )
    intersection_height = max(0, min(first_top + first_height, second_top + second_height) - max(first_top, second_top))
    return int(intersection_width * intersection_height)


_DEFECT_REASON_PRIORITY: dict[str, int] = {
    "merged_contour": 90,
    "filled_cell": 88,
    "broken_geometry": 86,
    "edge_clipped_cell": 84,
    "partial_filled_cell": 82,
    "defect_disagreement": 60,
    "geometry_mismatch": 58,
    "class_conflict": 56,
    "xor_residual": 56,
    "small_artifact": 30,
    "confidence_only_cell": 28,
    "binary_only_cell": 26,
}


def _defect_priority(reasons: tuple[str, ...] | list[str] | None) -> int:
    if not reasons:
        return 0
    return max((_DEFECT_REASON_PRIORITY.get(str(reason), 10) for reason in reasons), default=0)


def _suppress_overlapping_defect_boxes(
    per_cell: list[GridCellAnalysisResult],
    *,
    containment_ratio: float = 0.55,
    iou_threshold: float = 0.35,
) -> list[GridCellAnalysisResult]:
    """Drop weaker/smaller defect boxes nested inside or heavily overlapping stronger ones."""

    if len(per_cell) <= 1:
        return list(per_cell)

    normals = [cell for cell in per_cell if not tuple(cell.reasons or ())]
    defects = [cell for cell in per_cell if tuple(cell.reasons or ())]
    if len(defects) <= 1:
        return list(per_cell)

    ordered = sorted(
        defects,
        key=lambda cell: (
            _defect_priority(cell.reasons),
            int(cell.bbox[2]) * int(cell.bbox[3]),
            float(cell.score),
        ),
        reverse=True,
    )
    kept: list[GridCellAnalysisResult] = []
    for candidate in ordered:
        candidate_area = max(1, int(candidate.bbox[2]) * int(candidate.bbox[3]))
        suppressed = False
        for winner in kept:
            intersection = _bbox_intersection_area(candidate.bbox, winner.bbox)
            if intersection <= 0:
                continue
            contained = float(intersection) / float(candidate_area) >= float(containment_ratio)
            overlapped = _bbox_iou(candidate.bbox, winner.bbox) >= float(iou_threshold)
            if contained or overlapped:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)

    merged = list(normals)
    merged.extend(kept)
    merged.sort(key=lambda cell: (float(cell.centroid[1]), float(cell.centroid[0]), int(cell.contour_id)))
    return [replace(cell, row=int(index), col=0) for index, cell in enumerate(merged)]


def _grid_comparison_pairs(
    confidence_cells: list[GridCellAnalysisResult],
    binary_cells: list[GridCellAnalysisResult],
    match_distance: float,
) -> list[tuple[int, int]]:
    """Candidate pairs via cKDTree on centroids. Same radius semantics as the old bucket scan."""

    confidence_count = len(confidence_cells)
    binary_count = len(binary_cells)
    if confidence_count == 0 or binary_count == 0:
        return []
    if confidence_count * binary_count <= 64:
        return [(left, right) for left in range(confidence_count) for right in range(binary_count)]
    radius = max(1.0, float(match_distance))
    points = np.array(
        [[float(cell.center_x), float(cell.center_y)] for cell in binary_cells],
        dtype=np.float64,
    )
    queries = np.array(
        [[float(cell.center_x), float(cell.center_y)] for cell in confidence_cells],
        dtype=np.float64,
    )
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        neighbors = tree.query_ball_point(queries, r=radius * 1.35)
    except Exception:
        neighbors = []
        for query in queries:
            deltas = points - query
            distances = np.hypot(deltas[:, 0], deltas[:, 1])
            neighbors.append(np.flatnonzero(distances <= radius * 1.35).tolist())
    pairs: list[tuple[int, int]] = []
    for confidence_index, found in enumerate(neighbors):
        for binary_index in found:
            pairs.append((confidence_index, int(binary_index)))
    return pairs


def compare_grid_cell_analyses(
    confidence_result: GridFrameAnalysisResult,
    binary_result: GridFrameAnalysisResult,
    *,
    centroid_tolerance_ratio: float = 0.72,
    geometry_iou_threshold: float = 0.58,
    centroid_mismatch_ratio: float = 0.48,
) -> GridFrameAnalysisResult:
    """Compare semantic cell geometry after each representation was analyzed independently."""

    confidence_cells = list(confidence_result.per_cell_results)
    binary_cells = list(binary_result.per_cell_results)
    cell_width = max(1.0, float(confidence_result.cell_width or binary_result.cell_width or 1))
    cell_height = max(1.0, float(confidence_result.cell_height or binary_result.cell_height or 1))
    match_distance = max(1.0, float(np.hypot(cell_width, cell_height)) * float(centroid_tolerance_ratio))
    candidates: list[tuple[float, float, int, int]] = []
    for confidence_index, binary_index in _grid_comparison_pairs(confidence_cells, binary_cells, match_distance):
        confidence_cell = confidence_cells[confidence_index]
        binary_cell = binary_cells[binary_index]
        distance = float(
            np.hypot(
                confidence_cell.center_x - binary_cell.center_x,
                confidence_cell.center_y - binary_cell.center_y,
            )
        )
        iou = _bbox_iou(confidence_cell.bbox, binary_cell.bbox)
        if distance <= match_distance or iou >= 0.08:
            candidates.append((distance / match_distance, 1.0 - iou, confidence_index, binary_index))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    confidence_matches: dict[int, int] = {}
    binary_matches: set[int] = set()
    for _distance_ratio, _iou_loss, confidence_index, binary_index in candidates:
        if confidence_index in confidence_matches or binary_index in binary_matches:
            continue
        confidence_matches[confidence_index] = binary_index
        binary_matches.add(binary_index)

    mismatches: list[GridCellAnalysisResult] = []
    for confidence_index, confidence_cell in enumerate(confidence_cells):
        binary_index = confidence_matches.get(confidence_index)
        if binary_index is None:
            mismatches.append(
                replace(
                    confidence_cell,
                    row=len(mismatches),
                    col=0,
                    status="broken",
                    score=1.0,
                    reasons=("confidence_only_cell",),
                    consistency_score=1.0,
                    consistency_reasons=("confidence_only_cell",),
                )
            )
            continue
        binary_cell = binary_cells[binary_index]
        distance_ratio = float(
            np.hypot(
                confidence_cell.center_x - binary_cell.center_x,
                confidence_cell.center_y - binary_cell.center_y,
            )
            / match_distance
        )
        iou = _bbox_iou(confidence_cell.bbox, binary_cell.bbox)
        reasons: list[str] = []
        if iou < float(geometry_iou_threshold) or distance_ratio > float(centroid_mismatch_ratio):
            reasons.append("geometry_mismatch")
        confidence_bad = str(confidence_cell.status) != "normal"
        binary_bad = str(binary_cell.status) != "normal"
        if confidence_bad != binary_bad:
            reasons.append("defect_disagreement")
        if not reasons:
            continue
        geometry_score = max(0.0, 1.0 - iou, min(1.0, distance_ratio))
        status_score = (
            max(float(confidence_cell.score), float(binary_cell.score), 0.78) if confidence_bad != binary_bad else 0.0
        )
        score = float(np.clip(max(geometry_score, status_score), 0.0, 1.0))
        left = min(confidence_cell.left, binary_cell.left)
        top = min(confidence_cell.top, binary_cell.top)
        right = max(confidence_cell.left + confidence_cell.width, binary_cell.left + binary_cell.width)
        bottom = max(confidence_cell.top + confidence_cell.height, binary_cell.top + binary_cell.height)
        mismatches.append(
            GridCellAnalysisResult(
                row=len(mismatches),
                col=0,
                bbox=(left, top, max(1, right - left), max(1, bottom - top)),
                centroid=(
                    (confidence_cell.center_x + binary_cell.center_x) * 0.5,
                    (confidence_cell.center_y + binary_cell.center_y) * 0.5,
                ),
                contour_id=confidence_cell.contour_id,
                status="broken",
                score=score,
                reasons=tuple(reasons),
                consistency_score=score,
                consistency_reasons=tuple(reasons),
            )
        )

    for binary_index, binary_cell in enumerate(binary_cells):
        if binary_index in binary_matches:
            continue
        mismatches.append(
            replace(
                binary_cell,
                row=len(mismatches),
                col=0,
                status="broken",
                score=1.0,
                reasons=("binary_only_cell",),
                consistency_score=1.0,
                consistency_reasons=("binary_only_cell",),
            )
        )

    total_expected = max(1, confidence_result.total_expected_cells, binary_result.total_expected_cells)
    damage_score = _damage_score(mismatches, total_expected) if mismatches else 0.0
    severity = GridDamageSeverityThresholds().level_for_score(damage_score)
    return GridFrameAnalysisResult(
        frame_id=str(confidence_result.frame_id or binary_result.frame_id),
        frame_path=str(confidence_result.frame_path or binary_result.frame_path),
        image_width=max(confidence_result.image_width, binary_result.image_width),
        image_height=max(confidence_result.image_height, binary_result.image_height),
        grid_rows=max(confidence_result.grid_rows, binary_result.grid_rows),
        grid_cols=max(confidence_result.grid_cols, binary_result.grid_cols),
        total_expected_cells=total_expected,
        detected_cells=len(mismatches),
        normal_cells=max(0, total_expected - len(mismatches)),
        suspicious_cells=0,
        broken_cells=len(mismatches),
        missing_cells=0,
        artifact_cells=0,
        damage_score=damage_score,
        severity_level=severity,
        grid_detected=bool(confidence_result.grid_detected and binary_result.grid_detected),
        per_cell_results=tuple(mismatches),
        component_count=len(mismatches),
        cell_width=int(round(cell_width)),
        cell_height=int(round(cell_height)),
    )


@dataclass(slots=True)
class _ContourCandidate:
    contour_id: int
    contour: Any
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: float
    bbox_area: float
    aspect_ratio: float
    extent: float
    solidity: float
    perimeter: float
    approx_vertices: int
    fill_ratio: float
    interior_fill_ratio: float
    center_fill_ratio: float
    outline_min_side_coverage: float
    outline_mean_side_coverage: float
    outline_side_imbalance: float
    inner_hole_ratio: float
    child_count: int
    touches_border: bool
    # A hole inside a mask piece. Kept for frame-level counts, never scored as a cell.
    is_hole: bool = False


def detect_grid_cell_anomalies(
    image: np.ndarray,
    *,
    frame_id: str = "",
    frame_path: str = "",
    config: GridDamageAnalysisConfig | None = None,
    reference_profile: GridCellReferenceProfile | None = None,
    confidence_map: np.ndarray | None = None,
    conductor_zones=None,
) -> GridFrameAnalysisResult:
    """Analyze one image and return compact grid damage metrics."""

    cfg = (config or GridDamageAnalysisConfig()).normalized()
    started = perf_counter()
    with profile_stage("validation.grid.grayscale", frame_id=frame_id):
        gray = _normalize_grayscale(image)
    probability = _as_probability_map(confidence_map)
    if probability is None and str(cfg.cell_representation) == "confidence":
        probability = _as_probability_map(gray)
    if probability is not None and probability.shape != gray.shape:
        if cv2 is not None:
            probability = cv2.resize(
                probability, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32, copy=False)
        else:
            probability = None
    height, width = gray.shape
    empty_result = GridFrameAnalysisResult(
        frame_id=str(frame_id or ""),
        frame_path=str(frame_path or ""),
        image_width=int(width),
        image_height=int(height),
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=0,
        detected_cells=0,
        normal_cells=0,
        suspicious_cells=0,
        broken_cells=0,
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.0,
        severity_level="OK",
        grid_detected=False,
        per_cell_results=(),
        component_count=0,
    )
    if gray.size <= 1 or cv2 is None:
        return empty_result

    with profile_stage("validation.grid.threshold_morphology", frame_id=frame_id):
        threshold = _threshold_grid(gray, cfg)
    with profile_stage("validation.grid.contours.find", frame_id=frame_id):
        contours_result = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
        hierarchy = contours_result[1] if len(contours_result) == 2 else contours_result[2]
    zone_skipped = 0
    zone_supplied = conductor_zones is not None
    with profile_stage("validation.grid.contours.features", frame_id=frame_id):
        if conductor_zones is not None:
            zone_map = conductor_zones
        elif str(cfg.cell_representation) == "confidence":
            # Rings on a confidence map are cell outlines, not conductor grain.
            zone_map = _empty_conductor_zone_map()
        else:
            zone_map = _binary_conductor_zone_map(gray, threshold, contours, hierarchy, cfg)
        shape_template = (
            getattr(reference_profile, "shape_template", None)
            if reference_profile is not None and str(cfg.scoring_mode) == "calibrated"
            else None
        )
        template_view = _TemplateView(shape_template, cfg) if shape_template is not None else None
        skipped_box = [0]
        if template_view is not None and zone_map.zones:
            # Zones are found before the pieces, so every piece inside a zone used to vanish.
            # In a sparse field the off-lattice cells themselves made the zone; a zone made of
            # cells is no zone, and the debris and broken cells in it must be checked.
            all_candidates = _extract_candidates(contours, hierarchy, threshold, gray.shape, cfg)
            zone_map = template_view.drop_cell_zones(zone_map, all_candidates, threshold)
            candidates = [
                item for item in all_candidates if not zone_map.contains(item.centroid[0], item.centroid[1])
            ]
            skipped_box[0] = sum(1 for item in all_candidates if not item.is_hole) - sum(
                1 for item in candidates if not item.is_hole
            )
        else:
            candidates = _extract_candidates(
                contours,
                hierarchy,
                threshold,
                gray.shape,
                cfg,
                zone_map=zone_map if zone_map.zones else None,
                zone_skip_count=skipped_box,
            )
        zone_skipped = int(skipped_box[0])
    if zone_skipped:
        _LOGGER.info(
            "conductor zone skipped %d components frame=%s",
            zone_skipped,
            frame_id or frame_path or "<array>",
        )
    if not candidates and not zone_map.zones:
        return empty_result

    min_required_cells = int(cfg.min_grid_candidate_cells)
    # With a layer template one cell is enough to check: a lone cell is still a cell.
    if len(candidates) < min_required_cells and not (template_view is not None and candidates):
        if not zone_map.zones:
            return empty_result
        return _analyze_conductor_only_frame(
            candidates,
            frame_id=frame_id,
            frame_path=frame_path,
            width=width,
            height=height,
            config=cfg,
            started=started,
            zone_map=zone_map,
            zone_skipped=zone_skipped,
            zone_supplied=zone_supplied,
        )

    with profile_stage("validation.grid.reference_profile", frame_id=frame_id):
        if str(cfg.scoring_mode) == "calibrated" and not cfg.calibration_reference:
            if reference_profile is not None:
                normal_seed = _normal_seed_candidates(candidates, config=cfg)
                profile = reference_profile
            else:
                profile, normal_seed = _calibrated_size_profile(candidates, frame_id=frame_id, frame_path=frame_path)
        elif str(cfg.scoring_mode) == "calibrated":
            profile, normal_seed = _calibrated_size_profile(candidates, frame_id=frame_id, frame_path=frame_path)
        else:
            normal_seed = _normal_seed_candidates(candidates, config=cfg)
            profile = reference_profile or _grid_cell_reference_profile_from_candidates(
                candidates,
                config=cfg,
                frame_id=frame_id,
                frame_path=frame_path,
            )
            if profile is None:
                # Outline drawings fail the filled-cell seed. A repeated size is still a grid.
                profile, sized = _calibrated_size_profile(
                    candidates, frame_id=frame_id, frame_path=frame_path
                )
                if sized:
                    normal_seed = sized
    if profile is None:
        return _analyze_conductor_only_frame(
            candidates,
            frame_id=frame_id,
            frame_path=frame_path,
            width=width,
            height=height,
            config=cfg,
            started=started,
            zone_map=zone_map,
            zone_skipped=zone_skipped,
            zone_supplied=zone_supplied,
        )
    # Run-wide cell size: skip crumb-only frames (no array). Without a run profile,
    # keep legacy single-frame behaviour so synthetic unit tests stay valid.
    if template_view is not None:
        matched = template_view.count_cells(candidates)
        # A frame of crumbs (no array) still throws a few cell-like blobs among thousands.
        if matched < 1 or (len(candidates) >= max(120, 4 * matched) and matched < 80):
            return _no_cell_array_result(
                frame_id=frame_id,
                frame_path=frame_path,
                width=width,
                height=height,
                component_count=len(candidates),
                zone_map=zone_map,
                zone_skipped=zone_skipped,
                cell_width=float(reference_profile.median_width),
                cell_height=float(reference_profile.median_height),
                config=cfg,
            )
    elif reference_profile is not None:
        min_matched = max(24, int(cfg.min_grid_candidate_cells) * 3)
        local_profile, _local_seed = _calibrated_size_profile(
            candidates, frame_id=frame_id, frame_path=frame_path
        )
        # Frame has no array when its *own* size cluster is crumb-scale, even if a few
        # blobs accidentally match the run-wide cell size (classic frame 184).
        if not _profile_looks_like_cell_array(local_profile):
            return _no_cell_array_result(
                frame_id=frame_id,
                frame_path=frame_path,
                width=width,
                height=height,
                component_count=len(candidates),
                zone_map=zone_map,
                zone_skipped=zone_skipped,
                cell_width=float(reference_profile.median_width),
                cell_height=float(reference_profile.median_height),
                config=cfg,
            )
        if not _frame_has_reliable_cell_array(candidates, reference_profile, min_matched=min_matched):
            return _no_cell_array_result(
                frame_id=frame_id,
                frame_path=frame_path,
                width=width,
                height=height,
                component_count=len(candidates),
                zone_map=zone_map,
                zone_skipped=zone_skipped,
                cell_width=float(reference_profile.median_width),
                cell_height=float(reference_profile.median_height),
                config=cfg,
            )
    median_width = float(profile.median_width)
    median_height = float(profile.median_height)
    median_area = float(profile.median_area)
    median_fill = float(profile.median_fill)
    median_interior_fill = float(profile.median_interior_fill)
    median_center_fill = float(profile.median_center_fill)
    median_aspect = float(profile.median_aspect)

    normal_seed_ids = {int(item.contour_id) for item in normal_seed}
    per_cell: list[GridCellAnalysisResult] = []
    defect_entries: list[tuple[int, _ContourCandidate]] = []
    ordered_candidates = sorted(
        (item for item in candidates if not item.is_hole),
        key=lambda item: (float(item.centroid[1]), float(item.centroid[0]), int(item.contour_id)),
    )
    with profile_stage("validation.grid.cells.classify", frame_id=frame_id):
        from .grid_calibration import (
            apply_example_correction,
            example_distance_matrix,
            local_neighbor_medians,
            reference_from_normal_examples,
        )
        from .grid_scoring import (
            NORMAL_SCORE_CEILING,
            CalibratedScores,
            geometry_agrees_with_neighbors,
            score_debris,
            score_edge,
            score_fill,
            score_geometry,
            score_size_shortfall,
            slider_threshold,
        )

        reference_solidity = float(np.median([float(item.solidity) for item in (normal_seed or candidates)]))
        reference_extent = float(np.median([float(item.extent) for item in (normal_seed or candidates)]))
        # Local shape measures are scored only in calibrated mode.
        local_shapes: dict[int, tuple[float, float, float]] = {}
        reference_window_fill, reference_notch_depth, reference_corner_fill = 1.0, 0.0, 1.0
        if str(cfg.scoring_mode) == "calibrated":
            local_shapes = {id(item): _local_shape_measures(item) for item in ordered_candidates}
            reference_pool = [
                local_shapes.get(id(item)) or _local_shape_measures(item) for item in (normal_seed or candidates)
            ]
            if reference_pool:
                reference_window_fill = float(np.median([shape[0] for shape in reference_pool]))
                reference_notch_depth = float(np.median([shape[1] for shape in reference_pool]))
                reference_corner_fill = float(np.median([shape[2] for shape in reference_pool]))
        shared_reference = dict(cfg.calibration_reference)
        if shared_reference:
            median_width = float(shared_reference.get("width", median_width))
            median_height = float(shared_reference.get("height", median_height))
            median_area = float(shared_reference.get("area", median_area))
            median_fill = float(shared_reference.get("fill", median_fill))
            median_interior_fill = float(shared_reference.get("interior_fill", median_interior_fill))
            median_center_fill = float(shared_reference.get("center_fill", median_center_fill))
            median_aspect = float(shared_reference.get("aspect", median_aspect))
            reference_solidity = float(shared_reference.get("solidity", reference_solidity))
            reference_extent = float(shared_reference.get("extent", reference_extent))
        example_reference = reference_from_normal_examples(
            [{"label": label, "features": dict(features)} for label, features in cfg.calibration_examples if features]
        )
        if example_reference:
            # Operator "normal" marks move the shape reference, not only nearby hits.
            median_interior_fill = float(example_reference.get("interior_fill", median_interior_fill))
            median_fill = float(example_reference.get("fill", median_fill))
            median_center_fill = float(example_reference.get("center_fill", median_center_fill))
            reference_solidity = float(example_reference.get("solidity", reference_solidity))
            reference_extent = float(example_reference.get("extent", reference_extent))
            if "width" in example_reference:
                median_width = float(example_reference["width"])
            if "height" in example_reference:
                median_height = float(example_reference["height"])
            if "area" in example_reference:
                median_area = float(example_reference["area"])
        calibrated = str(cfg.scoring_mode) == "calibrated"
        reference_hole = float(np.median([float(item.inner_hole_ratio) for item in (normal_seed or candidates)]))
        local_medians, neighbor_ids = local_neighbor_medians(ordered_candidates) if calibrated else ([], [])
        lattice_neighbor_counts: list[int] = []
        if calibrated:
            local_medians, lattice_neighbor_counts = _retarget_local_medians(
                ordered_candidates,
                local_medians,
                neighbor_ids,
                modal_width=median_width,
                modal_height=median_height,
                modal_area=median_area,
                modal_solidity=reference_solidity,
                modal_extent=reference_extent,
                modal_interior=median_interior_fill,
            )
        edge_enabled = cfg.enabled_reason_types is None or "edge_clipped_cell" in set(cfg.enabled_reason_types or ())
        edge_columns, edge_rows, edge_pitch_x, edge_pitch_y = _lattice_grid_axes(
            ordered_candidates,
            modal_width=median_width,
            modal_height=median_height,
            modal_area=median_area,
        )
        if calibrated:
            prepared_scores: list[CalibratedScores] = []
            prepared_features: list[dict[str, float]] = []
            for candidate_index, candidate in enumerate(ordered_candidates):
                local = local_medians[candidate_index]
                width_ratio = float(candidate.bbox[2]) / max(1.0, float(local["width"]))
                height_ratio = float(candidate.bbox[3]) / max(1.0, float(local["height"]))
                area_ratio = float(candidate.area) / max(1.0, float(local["area"]))
                prepared_features.append(
                    {
                        "width_ratio": width_ratio,
                        "height_ratio": height_ratio,
                        "area_ratio": area_ratio,
                        "interior_fill": float(candidate.interior_fill_ratio),
                        "reference_interior": float(median_interior_fill),
                        "solidity": float(candidate.solidity),
                        "reference_solidity": float(reference_solidity),
                        "extent": float(candidate.extent),
                        "reference_extent": float(reference_extent),
                    }
                )
                window_fill, notch_depth, corner_fill = local_shapes[id(candidate)]
                prepared_scores.append(
                    CalibratedScores(
                        fill=score_fill(candidate.interior_fill_ratio, median_interior_fill),
                        geometry=score_geometry(
                            candidate.solidity,
                            candidate.extent,
                            reference_solidity,
                            reference_extent,
                            window_fill=window_fill,
                            reference_window_fill=reference_window_fill,
                            notch_depth=notch_depth,
                            reference_notch_depth=reference_notch_depth,
                            corner_fill=corner_fill,
                            reference_corner_fill=reference_corner_fill,
                            size_ratio=min(width_ratio, height_ratio),
                        ),
                        merge=_score_merge_for_candidate(
                            candidate,
                            cell_width=median_width,
                            cell_height=median_height,
                            merge_sensitivity=cfg.merge_sensitivity,
                        ),
                        debris=score_debris(area_ratio, max(width_ratio, height_ratio)),
                        edge=score_edge(bool(candidate.touches_border), min(width_ratio, height_ratio)),
                    )
                )
            other_mark = []
            for scores in prepared_scores:
                preliminary = scores.reasons(
                    fill=slider_threshold(cfg.fill_sensitivity),
                    geometry=1.0,
                    merge=slider_threshold(cfg.merge_sensitivity),
                    debris=slider_threshold(cfg.debris_sensitivity),
                    edge=1.0,
                    edge_enabled=False,
                )
                other_mark.append(bool(preliminary))
            for candidate_index, candidate in enumerate(ordered_candidates):
                ids = neighbor_ids[candidate_index]
                clean = [index for index in ids if not other_mark[index]]
                neighbors_unmarked = len(ids) >= 3 and len(clean) * 2 >= len(ids)
                local = local_medians[candidate_index]
                neighbor_shapes = [local_shapes[id(ordered_candidates[index])] for index in ids]
                window_fill, notch_depth, corner_fill = local_shapes[id(candidate)]
                if geometry_agrees_with_neighbors(
                    candidate.solidity,
                    candidate.extent,
                    local["solidity"],
                    local["extent"],
                    neighbors_unmarked=neighbors_unmarked,
                    window_fill=window_fill,
                    neighbor_window_fill=float(np.median([shape[0] for shape in neighbor_shapes])) if neighbor_shapes else None,
                    notch_depth=notch_depth,
                    neighbor_notch_depth=float(np.median([shape[1] for shape in neighbor_shapes])) if neighbor_shapes else None,
                    corner_fill=corner_fill,
                    neighbor_corner_fill=float(np.median([shape[2] for shape in neighbor_shapes])) if neighbor_shapes else None,
                ):
                    # A shared shape is not broken; a cell shorter than its neighbours still is.
                    prepared_features_row = prepared_features[candidate_index]
                    size_score = score_size_shortfall(
                        min(prepared_features_row["width_ratio"], prepared_features_row["height_ratio"])
                    )
                    prepared_scores[candidate_index] = replace(
                        prepared_scores[candidate_index],
                        geometry=max(min(prepared_scores[candidate_index].geometry, NORMAL_SCORE_CEILING), size_score),
                    )
            frame_examples = [
                {"label": label, "features": dict(features)}
                for label, features in cfg.calibration_examples
                if features
            ]
            example_distances = example_distance_matrix(prepared_features, frame_examples) if frame_examples else None
        else:
            frame_examples = []
            example_distances = None
        for candidate_index, candidate in enumerate(ordered_candidates):
            template_features: tuple[tuple[str, float], ...] = ()
            local = local_medians[candidate_index] if calibrated else {}
            if calibrated:
                local_width = float(local.get("width", median_width))
                local_height = float(local.get("height", median_height))
                local_area = float(local.get("area", median_area))
                width_ratio = float(candidate.bbox[2]) / max(1.0, local_width)
                height_ratio = float(candidate.bbox[3]) / max(1.0, local_height)
                area_ratio = float(candidate.area) / max(1.0, local_area)
                scores = prepared_scores[candidate_index]
                reasons = scores.reasons(
                    fill=slider_threshold(cfg.fill_sensitivity),
                    geometry=slider_threshold(cfg.geometry_sensitivity),
                    merge=slider_threshold(cfg.merge_sensitivity),
                    debris=slider_threshold(cfg.debris_sensitivity),
                    edge=0.55,
                    edge_enabled=edge_enabled,
                )
                reasons = apply_example_correction(
                    reasons,
                    {"fill": scores.fill, "geometry": scores.geometry, "merge": scores.merge, "debris": scores.debris},
                    {
                        "fill": slider_threshold(cfg.fill_sensitivity),
                        "geometry": slider_threshold(cfg.geometry_sensitivity),
                        "merge": slider_threshold(cfg.merge_sensitivity),
                        "debris": slider_threshold(cfg.debris_sensitivity),
                    },
                    prepared_features[candidate_index],
                    frame_examples,
                    example_influence=float(cfg.example_influence),
                    distance_row=None if example_distances is None else example_distances[candidate_index],
                )
                reasons = _gate_calibrated_reasons(
                    reasons,
                    width_ratio=width_ratio,
                    height_ratio=height_ratio,
                    area_ratio=area_ratio,
                    area=float(candidate.area),
                    in_lattice=len(lattice_neighbor_counts) > candidate_index
                    and lattice_neighbor_counts[candidate_index] >= 3,
                    debris_min_area=DEBRIS_ANALYSIS_FLOOR_PX,
                    solidity=float(candidate.solidity),
                )
                array_area_ratio = float(candidate.area) / max(1.0, float(median_area))
                array_width_ratio = float(candidate.bbox[2]) / max(1.0, float(median_width))
                array_height_ratio = float(candidate.bbox[3]) / max(1.0, float(median_height))
                on_column = _on_grid_axis(
                    float(candidate.centroid[0]),
                    edge_columns,
                    edge_pitch_x,
                    max(8.0, 0.55 * float(median_width)),
                )
                on_row = _on_grid_axis(
                    float(candidate.centroid[1]),
                    edge_rows,
                    edge_pitch_y,
                    max(8.0, 0.55 * float(median_height)),
                )
                edge_y = float(getattr(zone_map, "boundary_y", -1.0))
                if edge_y >= 0.0:
                    near_conductor_edge = edge_y - 4.0 <= float(candidate.centroid[1]) <= edge_y + max(
                        20.0, 1.2 * float(median_height)
                    )
                else:
                    near_conductor_edge = True
                before_short = reasons
                reasons = _promote_short_lattice_cell(
                    reasons,
                    on_lattice_node=(
                        template_view is None
                        and on_column
                        and on_row
                        and not candidate.touches_border
                        and near_conductor_edge
                    ),
                    area_ratio=array_area_ratio,
                    width_ratio=array_width_ratio,
                    height_ratio=array_height_ratio,
                )
                forced_geometry = reasons != before_short and "broken_geometry" in reasons
                reasons = _prefer_edge_slot_over_debris(
                    reasons,
                    candidate,
                    frame_width=int(width),
                    frame_height=int(height),
                    modal_width=median_width,
                    modal_height=median_height,
                    column_centers=edge_columns,
                    row_centers=edge_rows,
                    pitch_x=edge_pitch_x,
                    pitch_y=edge_pitch_y,
                )
                score = max(scores.fill, scores.geometry, scores.merge, scores.debris) if reasons else 0.0
                if "edge_clipped_cell" in reasons:
                    # The slot check already proved the clip. Without this a clean clipped
                    # cell scored 0 and fell back to normal, so edge clips came and went.
                    score = max(float(score), float(scores.edge), EDGE_CLIP_SCORE)
                if forced_geometry and "broken_geometry" in reasons:
                    score = max(float(score), 0.80)
                hole_area = float(candidate.inner_hole_ratio) * float(candidate.area)
                slot_sized = (
                    0.55 <= area_ratio <= 1.90
                    and min(width_ratio, height_ratio) >= 0.50
                    and max(width_ratio, height_ratio) <= 1.85
                )
                if (
                    slot_sized
                    and hole_area >= DEBRIS_MIN_AREA_PX
                    and float(candidate.inner_hole_ratio) >= reference_hole + HOLE_EXCESS_RATIO
                    and "merged_contour" not in reasons
                ):
                    # A hole the normal cells of this layer do not have breaks the cell shape.
                    # Blobs larger than a slot stay debris even with holes.
                    reasons = tuple(
                        dict.fromkeys((*(reason for reason in reasons if reason != "small_artifact"), "broken_geometry"))
                    )
                    score = max(float(score), 0.80)
                    hole_broken = True
                else:
                    hole_broken = False
                if template_view is not None:
                    score, reasons, template_features = template_view.judge(
                        candidate, reasons, score, hole_broken=hole_broken
                    )
            else:
                score, reasons = _classify_detected_cell(
                    candidate,
                    median_width=median_width,
                    median_height=median_height,
                    median_area=median_area,
                    median_fill=median_fill,
                    median_interior_fill=median_interior_fill,
                    median_center_fill=median_center_fill,
                    median_aspect=median_aspect,
                    config=cfg,
                    confidence_map=probability,
                )
            score, reasons = _filter_disabled_grid_reasons(score, reasons, cfg)
            detached_edge = False
            leftover_debris = False
            if reasons and set(reasons) <= {"small_artifact", "broken_geometry", "edge_clipped_cell"} and _is_detached_confidence_cell_edge(
                candidate,
                candidates,
                median_width=median_width,
                median_height=median_height,
                median_area=median_area,
                config=cfg,
            ):
                score, reasons = 0.0, ()
                detached_edge = True
            is_bad = _is_bad_grid_cell(score, reasons, cfg)
            if calibrated and template_view is not None:
                is_cell_like = template_view.is_cell_sized(candidate)
            elif calibrated:
                is_cell_like = bool(
                    0.50 <= min(width_ratio, height_ratio)
                    and max(width_ratio, height_ratio) <= 1.85
                    and 0.35 <= area_ratio <= 2.40
                    and float(candidate.solidity) >= 0.45
                )
            else:
                is_cell_like = int(candidate.contour_id) in normal_seed_ids or _is_cell_like_candidate(
                    candidate,
                    median_width=median_width,
                    median_height=median_height,
                    median_area=median_area,
                    config=cfg,
                )
            if (
                calibrated
                and not is_bad
                and not is_cell_like
                and not detached_edge
                and float(candidate.area) >= DEBRIS_ANALYSIS_FLOOR_PX
                # A frame-cut piece on a grid row/column, or one that fits a cell slot, is a
                # clipped cell, not debris.
                and not (
                    candidate.touches_border
                    and (
                        on_column
                        or on_row
                        or _fits_edge_slot(
                            candidate,
                            frame_width=int(width),
                            frame_height=int(height),
                            modal_width=median_width,
                            modal_height=median_height,
                            column_centers=edge_columns,
                            row_centers=edge_rows,
                            pitch_x=edge_pitch_x,
                            pitch_y=edge_pitch_y,
                        )
                    )
                )
            ):
                # No filter claimed a piece that is not a cell: scratches, blobs larger
                # than merged cells, pieces off the lattice. They are debris, not skipped,
                # unless an operator "normal" example sits closer than a debris one.
                leftover = apply_example_correction(
                    ("small_artifact",),
                    {"debris": float(scores.debris)},
                    {"debris": slider_threshold(cfg.debris_sensitivity)},
                    prepared_features[candidate_index],
                    frame_examples,
                    example_influence=float(cfg.example_influence),
                    distance_row=None if example_distances is None else example_distances[candidate_index],
                )
                score, reasons = _filter_disabled_grid_reasons(
                    max(LEFTOVER_DEBRIS_SCORE, float(scores.debris)),
                    tuple(reason for reason in leftover if reason == "small_artifact"),
                    cfg,
                )
                is_bad = bool(reasons)
                leftover_debris = is_bad
            if not is_bad and not is_cell_like:
                continue
            if not calibrated and not is_bad and _is_ignored_fragment(
                candidate, median_width=median_width, median_height=median_height, median_area=median_area
            ):
                continue
            index = len(per_cell)
            if is_bad:
                defect_entries.append((index, candidate))
            uncertainty = _cell_model_uncertainty(probability, candidate.bbox) if calibrated else None
            outline: tuple[tuple[int, int], ...] = ()
            contour = getattr(candidate, "contour", None)
            if contour is not None and len(contour) >= 3:
                points = np.asarray(contour, dtype=np.int32).reshape(-1, 2)
                step = max(1, int(len(points) // 96))
                outline = tuple((int(x), int(y)) for x, y in points[::step])
            per_cell.append(
                GridCellAnalysisResult(
                    row=int(index),
                    col=0,
                    bbox=candidate.bbox,
                    centroid=candidate.centroid,
                    contour_id=candidate.contour_id,
                    status=_status_for_reasons(reasons) if is_bad else "normal",
                    score=float(max(0.0, min(1.0, score if is_bad else 0.0))),
                    reasons=tuple(reasons if is_bad else ()),
                    mean_confidence=None if uncertainty is None else uncertainty[0],
                    uncertain_pixel_ratio=None if uncertainty is None else uncertainty[1],
                    border_uncertainty=None if uncertainty is None else uncertainty[2],
                    feature_snapshot=(
                        (
                            ("width_ratio", float(width_ratio)),
                            ("height_ratio", float(height_ratio)),
                            ("area_ratio", float(area_ratio)),
                            ("interior_fill", float(candidate.interior_fill_ratio)),
                            ("reference_interior", float(median_interior_fill)),
                            ("solidity", float(candidate.solidity)),
                            ("reference_solidity", float(reference_solidity)),
                            ("extent", float(candidate.extent)),
                            ("reference_extent", float(reference_extent)),
                            ("fill_score", float(scores.fill)),
                            ("geometry_score", float(scores.geometry)),
                            ("merge_score", float(scores.merge)),
                            ("debris_score", float(scores.debris)),
                            ("edge_score", float(scores.edge)),
                            (CONTOUR_AREA_FEATURE, float(candidate.area)),
                        )
                        + ((LEFTOVER_DEBRIS_FEATURE, 1.0),) * int(leftover_debris)
                        + (template_features if template_view is not None else ())
                        if calibrated
                        else (
                            ("width", float(candidate.bbox[2])),
                            ("height", float(candidate.bbox[3])),
                            ("area", float(candidate.area)),
                            ("fill", float(candidate.fill_ratio)),
                            ("interior_fill", float(candidate.interior_fill_ratio)),
                            ("center_fill", float(candidate.center_fill_ratio)),
                            ("aspect", float(candidate.aspect_ratio)),
                            ("solidity", float(candidate.solidity)),
                            ("extent", float(candidate.extent)),
                        )
                    ),
                    outline=outline,
                )
            )

    if (
        not zone_map.zones
        and not zone_supplied
        and str(cfg.cell_representation) != "confidence"
    ):
        zone_map = _conductor_zone_map(
            candidates,
            width=int(width),
            height=int(height),
            columns=edge_columns,
            rows=edge_rows,
            pitch_x=edge_pitch_x,
            pitch_y=edge_pitch_y,
            modal_area=median_area,
            modal_width=median_width,
            modal_height=median_height,
            confidence=probability,
            foreground=threshold,
        )
        if template_view is not None:
            # A zone made of pieces that look like the layer cell is cells, not conductor:
            # a lone cell or a short array has no lattice around it but must be checked.
            zone_map = template_view.drop_cell_zones(zone_map, candidates, threshold)
    if zone_map.zones:
        per_cell = [cell for cell in per_cell if not zone_map.contains(cell.centroid[0], cell.centroid[1])]
    if str(cfg.scoring_mode) == "calibrated":
        per_cell = _dedupe_nested_grid_cells(per_cell)
        per_cell = _group_slot_fragments(
            per_cell,
            median_width=median_width,
            median_height=median_height,
            columns=edge_columns,
            rows=edge_rows,
            pitch_x=edge_pitch_x,
            pitch_y=edge_pitch_y,
            config=cfg,
            template_view=template_view,
        )
    per_cell = _collapse_conductor_regions(
        per_cell,
        median_width=median_width,
        median_height=median_height,
    )
    per_cell = _suppress_overlapping_defect_boxes(per_cell)
    emit_zones = cfg.enabled_reason_types is None or "conductor_zone" in set(cfg.enabled_reason_types)
    if emit_zones and zone_map.zones:
        next_contour_id = max((int(cell.contour_id or 0) for cell in per_cell), default=0) + 1
        for zone in zone_map.zones:
            per_cell.append(
                _zone_cell_record(zone, row=len(per_cell), contour_id=next_contour_id)
            )
            next_contour_id += 1
    by_contour = {int(item.contour_id): item for item in candidates}
    defect_entries = []
    for index, cell in enumerate(per_cell):
        if not cell.reasons or "conductor_zone" in cell.reasons:
            continue
        candidate = by_contour.get(int(cell.contour_id))
        if candidate is None:
            box_w = max(1, int(cell.bbox[2]))
            box_h = max(1, int(cell.bbox[3]))
            candidate = _ContourCandidate(
                contour_id=int(cell.contour_id),
                contour=None,
                bbox=tuple(int(value) for value in cell.bbox[:4]),
                centroid=cell.centroid,
                area=float(box_w * box_h) * 0.45,
                bbox_area=float(box_w * box_h),
                aspect_ratio=float(box_w) / float(box_h),
                extent=0.55,
                solidity=0.70,
                perimeter=float(2 * (box_w + box_h)),
                approx_vertices=10,
                fill_ratio=0.45,
                interior_fill_ratio=0.20,
                center_fill_ratio=0.15,
                outline_min_side_coverage=0.2,
                outline_mean_side_coverage=0.3,
                outline_side_imbalance=0.2,
                inner_hole_ratio=0.0,
                child_count=0,
                touches_border=False,
            )
        defect_entries.append((index, candidate))

    with profile_stage("validation.grid.clustering", frame_id=frame_id):
        feature_clusters, feature_cluster_by_cell = _cluster_defective_cell_features(
            per_cell,
            defect_entries,
            median_width=median_width,
            median_height=median_height,
            median_area=median_area,
        )
    if feature_cluster_by_cell:
        per_cell = [
            replace(
                cell,
                feature_cluster_id=feature_cluster_by_cell[index][0],
                feature_cluster_label=feature_cluster_by_cell[index][1],
            )
            if index in feature_cluster_by_cell
            else cell
            for index, cell in enumerate(per_cell)
        ]

    cell_rows = [cell for cell in per_cell if "conductor_zone" not in cell.reasons]
    counts = {
        "normal": sum(1 for cell in cell_rows if cell.status == "normal"),
        "suspicious": sum(1 for cell in cell_rows if cell.status == "suspicious"),
        "broken": sum(1 for cell in cell_rows if cell.status == "broken"),
        "missing": sum(1 for cell in cell_rows if cell.status == "missing"),
        "artifact": sum(1 for cell in cell_rows if cell.status == "artifact"),
    }
    if len(cell_rows) < min_required_cells and not zone_map.zones and template_view is None:
        return empty_result
    total_expected = max(1, len(cell_rows)) if cell_rows else 0
    damage_score = _damage_score(cell_rows, max(1, len(cell_rows))) if cell_rows else 0.0
    severity = cfg.severity_thresholds.level_for_score(damage_score)
    debug_payload = None
    if cfg.debug or cfg.include_debug_payload:
        debug_payload = GridDebugPayload(
            threshold=np.asarray(threshold, dtype=np.uint8), contours=_compact_contours(contours)
        )
    if cfg.debug:
        _write_debug_images(gray, threshold, contours, per_cell, (), (), frame_id, cfg)

    elapsed_ms = (perf_counter() - started) * 1000.0
    if cfg.debug:
        _LOGGER.info(
            "cell damage frame=%s contours=%d candidates=%d zone_skipped=%d cells=%d normal=%d suspicious=%d broken=%d missing=%d artifact=%d score=%.4f time_ms=%.1f",
            frame_id or frame_path or "<array>",
            len(contours),
            len(candidates),
            zone_skipped,
            total_expected,
            counts.get("normal", 0),
            counts.get("suspicious", 0),
            counts.get("broken", 0),
            counts.get("missing", 0),
            counts.get("artifact", 0),
            damage_score,
            elapsed_ms,
        )

    result = GridFrameAnalysisResult(
        frame_id=str(frame_id or ""),
        frame_path=str(frame_path or ""),
        image_width=int(width),
        image_height=int(height),
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=int(total_expected),
        detected_cells=int(len(cell_rows)),
        normal_cells=int(counts.get("normal", 0)),
        suspicious_cells=int(counts.get("suspicious", 0)),
        broken_cells=int(counts.get("broken", 0)),
        missing_cells=int(counts.get("missing", 0)),
        artifact_cells=int(counts.get("artifact", 0)),
        damage_score=float(damage_score),
        severity_level=str(severity),
        grid_detected=True,
        per_cell_results=tuple(sorted(per_cell, key=lambda item: (item.row, -item.score))),
        component_count=int(len(candidates)),
        x_axes=(),
        y_axes=(),
        cell_width=int(round(median_width)),
        cell_height=int(round(median_height)),
        feature_clusters=tuple(feature_clusters),
        debug=debug_payload,
        zone_skipped_components=int(zone_skipped),
    )
    if str(cfg.scoring_mode) == "calibrated" and float(cfg.debris_min_area_px) > DEBRIS_ANALYSIS_FLOOR_PX:
        # The operator's debris size is a last filter on debris marks only.
        from .grid_narrowing import GridNarrowing, narrow_grid_frame_result

        result = narrow_grid_frame_result(result, GridNarrowing(debris_min_area_px=int(cfg.debris_min_area_px)))
    return result


def uses_normal_bank(
    reference_profile: GridCellReferenceProfile | None, config: GridDamageAnalysisConfig | None = None
) -> bool:
    """Whether frames of this run go through the grid-free analysis with a bank of normal cells."""

    if reference_profile is None or getattr(reference_profile, "normal_bank", None) is None:
        return False
    return config is None or str(config.normalized().scoring_mode) == "calibrated"


def _cell_error_config(cfg: GridDamageAnalysisConfig):
    from .cell_error_analysis import CellErrorConfig

    return CellErrorConfig(
        geometry_sensitivity=int(cfg.geometry_sensitivity),
        merge_sensitivity=int(cfg.merge_sensitivity),
        debris_min_area_px=int(DEBRIS_ANALYSIS_FLOOR_PX),
    )


def _analyze_with_normal_bank(
    path_obj: Path,
    *,
    frame_id: str,
    cfg: GridDamageAnalysisConfig,
    reference_profile: GridCellReferenceProfile,
    confidence_obj: Path | None,
) -> GridFrameAnalysisResult | None:
    """Grid-free analysis of one mask, shaped as the old result for the matrix and the frame window."""

    from .cell_error_adapter import to_grid_frame_result
    from .cell_error_analysis import analyze_cell_errors
    from .grid_narrowing import GridNarrowing, narrow_grid_frame_result

    image = _load_cv2_grayscale_image(path_obj)
    if image is None:
        return None
    confidence = _load_cv2_grayscale_image(confidence_obj) if confidence_obj is not None else None
    analysis = analyze_cell_errors(
        image,
        reference_profile.normal_bank,
        confidence=confidence,
        config=_cell_error_config(cfg),
        frame_id=frame_id,
        frame_path=str(path_obj),
    )
    result = to_grid_frame_result(analysis, severity=cfg.severity_thresholds)
    enabled = None if cfg.enabled_reason_types is None else frozenset(cfg.enabled_reason_types)
    if enabled is not None and "conductor_zone" not in enabled:
        result = replace(
            result,
            per_cell_results=tuple(cell for cell in result.per_cell_results if "conductor_zone" not in cell.reasons),
        )
    narrowed = narrow_grid_frame_result(
        result,
        GridNarrowing(debris_min_area_px=max(int(DEBRIS_ANALYSIS_FLOOR_PX), int(cfg.debris_min_area_px)), enabled_reason_types=enabled),
    )
    return narrowed if isinstance(narrowed, GridFrameAnalysisResult) else result


def analyze_grid_frame_path(
    path: Path | str,
    *,
    frame_id: str = "",
    config: GridDamageAnalysisConfig | None = None,
    reference_profile: GridCellReferenceProfile | None = None,
    use_cache: bool = True,
    read_cache: bool | None = None,
    write_cache: bool | None = None,
    confidence_path: Path | str | None = None,
    conductor_zones=None,
    zone_cache_token: str = "",
) -> GridFrameAnalysisResult | None:
    """Load one image, analyze it, and cache only compact analysis data."""

    path_obj = Path(path)
    cfg = (config or GridDamageAnalysisConfig()).normalized()
    should_read_cache = bool(use_cache) if read_cache is None else bool(read_cache)
    should_write_cache = bool(use_cache) if write_cache is None else bool(write_cache)
    confidence_obj = Path(confidence_path) if confidence_path else None
    if uses_normal_bank(reference_profile, cfg):
        # The confidence map moves only scores here; it is part of the cache key, not a reason to skip it.
        cfg = replace(cfg, cell_representation="binary").normalized()
        token = "cell_errors" + (
            "|" + "|".join(str(part) for part in _grid_cache_identity(confidence_obj)) if confidence_obj is not None else ""
        )
        if should_read_cache:
            cached = _load_cached_grid_result(
                path_obj, frame_id=frame_id, config=cfg, reference_profile=reference_profile, zone_cache_token=token
            )
            if cached is not None:
                return cached
        with profile_stage("validation.grid.frame", frame_id=frame_id, frame_count=1):
            result = _analyze_with_normal_bank(
                path_obj, frame_id=frame_id, cfg=cfg, reference_profile=reference_profile, confidence_obj=confidence_obj
            )
        if result is not None and should_write_cache:
            _store_cached_grid_result(
                path_obj, frame_id=frame_id, config=cfg, reference_profile=reference_profile, result=result, zone_cache_token=token
            )
        profiler = current_profiler()
        if profiler is not None:
            profiler.increment("frames.processed")
        return result
    # External confidence fusion changes classify outcomes; skip shared path cache for those runs.
    can_use_shared_cache = confidence_obj is None
    if should_read_cache and can_use_shared_cache:
        with profile_stage("validation.cache.disk.read", frame_id=frame_id):
            cached = _load_cached_grid_result(
                path_obj,
                frame_id=frame_id,
                config=cfg,
                reference_profile=reference_profile,
                zone_cache_token=zone_cache_token,
            )
        if cached is not None:
            profiler = current_profiler()
            if profiler is not None:
                profiler.increment("cache.disk.hits")
                profiler.increment("frames.processed")
            return cached
        profiler = current_profiler()
        if profiler is not None:
            profiler.increment("cache.disk.misses")
    if cv2 is None:
        return None
    with profile_stage("validation.image.read_decode", frame_id=frame_id):
        image = _load_cv2_grayscale_image(path_obj)
        confidence_image = _load_cv2_grayscale_image(confidence_obj) if confidence_obj is not None else None
    if image is None:
        return None
    with profile_stage("validation.grid.frame", frame_id=frame_id, frame_count=1):
        result = detect_grid_cell_anomalies(
            image,
            frame_id=frame_id,
            frame_path=str(path_obj),
            config=cfg,
            reference_profile=reference_profile,
            confidence_map=confidence_image,
            conductor_zones=conductor_zones,
        )
    if should_write_cache and can_use_shared_cache:
        with profile_stage("validation.cache.disk.write", frame_id=frame_id):
            _store_cached_grid_result(
                path_obj,
                frame_id=frame_id,
                config=cfg,
                reference_profile=reference_profile,
                result=result,
                zone_cache_token=zone_cache_token,
            )
    profiler = current_profiler()
    if profiler is not None:
        profiler.increment("frames.processed")
    return result


def build_grid_cell_reference_profile(
    image: np.ndarray,
    *,
    frame_id: str = "",
    frame_path: str = "",
    config: GridDamageAnalysisConfig | None = None,
) -> GridCellReferenceProfile | None:
    """Compute the baseline cell profile from a user-selected reference frame."""

    cfg = (config or GridDamageAnalysisConfig()).normalized()
    gray = _normalize_grayscale(image)
    if gray.size <= 1 or cv2 is None:
        return None
    threshold = _threshold_grid(gray, cfg)
    contours_result = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
    hierarchy = contours_result[1] if len(contours_result) == 2 else contours_result[2]
    candidates = _extract_candidates(contours, hierarchy, threshold, gray.shape, cfg)
    return _grid_cell_reference_profile_from_candidates(
        candidates, config=cfg, frame_id=frame_id, frame_path=frame_path
    )


def build_grid_cell_reference_profile_path(
    path: Path | str,
    *,
    frame_id: str = "",
    config: GridDamageAnalysisConfig | None = None,
) -> GridCellReferenceProfile | None:
    if cv2 is None:
        return None
    path_obj = Path(path)
    image = _load_cv2_grayscale_image(path_obj)
    if image is None:
        return None
    return build_grid_cell_reference_profile(image, frame_id=frame_id, frame_path=str(path_obj), config=config)


@dataclass(frozen=True, slots=True)
class RunProfileRequest:
    """What a worker needs to build the run's normal-cell profile in its own thread.

    Building the bank reads and splits dozens of masks; the UI thread only collects the paths.
    """

    mask_paths: tuple[str, ...]
    sample_limit: int = 32
    frame_id: str = "run"
    min_frames: int = 1

    def resolve(self, config: GridDamageAnalysisConfig | None = None) -> GridCellReferenceProfile | None:
        return estimate_run_normal_profile(
            self.mask_paths,
            config=config,
            sample_limit=int(self.sample_limit),
            frame_id=self.frame_id,
            min_frames=int(self.min_frames),
        )


def resolve_reference_profile(value, config: GridDamageAnalysisConfig | None = None):
    """A ready profile as it is; a request built now (call it in a worker thread)."""

    return value.resolve(config) if isinstance(value, RunProfileRequest) else value


def estimate_run_normal_profile(
    mask_paths: Sequence[str | Path],
    *,
    config: GridDamageAnalysisConfig | None = None,
    sample_limit: int = 32,
    frame_id: str = "run",
    min_frames: int = 1,
) -> GridCellReferenceProfile | None:
    """Bank of normal cells from a sample of the run's masks, wrapped as the run's reference profile.

    The bank is built in two passes over the sample (normal_bank). Its status travels with it:
    a run without a reliable normal cell still gets a profile, and its frames report
    ``insufficient_normal_model`` instead of guessed defects.
    """

    from .cell_components import extract_components
    from .normal_bank import build_normal_bank

    paths = [Path(path) for path in mask_paths if str(path)]
    if sample_limit > 0 and len(paths) > sample_limit:
        step = max(1, len(paths) // sample_limit)
        paths = paths[::step][:sample_limit]
    component_sets = []
    for path_obj in paths:
        image = _load_cv2_grayscale_image(path_obj)
        if image is not None:
            component_sets.append(extract_components(image))
    if len(component_sets) < max(1, int(min_frames)):
        return None
    bank = build_normal_bank(component_sets)
    return GridCellReferenceProfile(
        median_width=float(bank.cell_width),
        median_height=float(bank.cell_height),
        median_area=float(bank.cell_area),
        median_fill=1.0,
        median_interior_fill=1.0,
        median_center_fill=1.0,
        median_aspect=float(bank.cell_width) / max(1.0, float(bank.cell_height)),
        candidate_count=int(bank.candidates),
        seed_count=int(len(bank.exemplars)),
        frame_id=str(frame_id or "run"),
        frame_path="",
        shape_template=bank.template,
        normal_bank=bank,
    )


def estimate_run_cell_reference_profile(
    mask_paths: Sequence[str | Path],
    *,
    config: GridDamageAnalysisConfig | None = None,
    sample_limit: int = 32,
    frame_id: str = "run",
    min_frames: int = 2,
) -> GridCellReferenceProfile | None:
    """Modal cell size across a run of masks for one model.

    Frames without a real lattice (crumbs only) do not contribute. Callers may
    later override this with an explicit project cell size when that setting exists.
    """

    if cv2 is None:
        return None
    cfg = (config or GridDamageAnalysisConfig()).normalized()
    binary_cfg = replace(cfg, cell_representation="binary").normalized()
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    fills: list[float] = []
    interiors: list[float] = []
    centers: list[float] = []
    aspects: list[float] = []
    seeds = 0
    template_seeds: list[tuple[Any, float, float, float]] = []
    paths = [Path(path) for path in mask_paths if str(path)]
    if sample_limit > 0 and len(paths) > sample_limit:
        step = max(1, len(paths) // sample_limit)
        paths = paths[::step][:sample_limit]
    for path_obj in paths:
        image = _load_cv2_grayscale_image(path_obj)
        if image is None:
            continue
        gray = _normalize_grayscale(image)
        threshold = _threshold_grid(gray, binary_cfg)
        contours_result = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
        hierarchy = contours_result[1] if len(contours_result) == 2 else contours_result[2]
        candidates = _extract_candidates(contours, hierarchy, threshold, gray.shape, binary_cfg)
        profile, cluster = _calibrated_size_profile(candidates, frame_id=str(path_obj.stem), frame_path=str(path_obj))
        if profile is None or len(cluster) < 8:
            continue
        if float(profile.median_area) < 80.0:
            continue
        frame_seeds = [
            (item.contour, float(item.bbox[2]), float(item.bbox[3]), float(item.area))
            for item in cluster
            if not item.touches_border
            and not item.is_hole
            and item.contour is not None
            and 0.80 <= float(item.bbox[2]) / max(1.0, float(profile.median_width)) <= 1.25
            and 0.80 <= float(item.bbox[3]) / max(1.0, float(profile.median_height)) <= 1.25
        ]
        if len(frame_seeds) > TEMPLATE_SEEDS_PER_FRAME:
            step_seed = len(frame_seeds) / float(TEMPLATE_SEEDS_PER_FRAME)
            frame_seeds = [frame_seeds[int(index * step_seed)] for index in range(TEMPLATE_SEEDS_PER_FRAME)]
        template_seeds.extend(frame_seeds)
        widths.append(float(profile.median_width))
        heights.append(float(profile.median_height))
        areas.append(float(profile.median_area))
        fills.append(float(profile.median_fill))
        interiors.append(float(profile.median_interior_fill))
        centers.append(float(profile.median_center_fill))
        aspects.append(float(profile.median_aspect))
        seeds += int(profile.seed_count)
    if len(areas) < max(1, int(min_frames)):
        return None
    run_w, run_h = float(np.median(widths)), float(np.median(heights))
    template = build_cell_shape_template(
        [
            seed
            for seed in template_seeds
            if 0.80 <= seed[1] / max(1.0, run_w) <= 1.25 and 0.80 <= seed[2] / max(1.0, run_h) <= 1.25
        ]
    )
    return GridCellReferenceProfile(
        median_width=float(np.median(widths)),
        median_height=float(np.median(heights)),
        median_area=float(np.median(areas)),
        median_fill=float(np.median(fills)),
        median_interior_fill=float(np.median(interiors)),
        median_center_fill=float(np.median(centers)),
        median_aspect=float(np.median(aspects)),
        candidate_count=int(len(areas)),
        seed_count=int(seeds),
        frame_id=str(frame_id or "run"),
        frame_path="",
        shape_template=template,
    )


def load_cached_grid_frame_result(
    path: Path | str,
    *,
    frame_id: str = "",
    config: GridDamageAnalysisConfig | None = None,
    reference_profile: GridCellReferenceProfile | None = None,
) -> GridFrameAnalysisResult | None:
    """Read one valid compact result without starting image analysis."""

    return _load_cached_grid_result(
        Path(path),
        frame_id=frame_id,
        config=(config or GridDamageAnalysisConfig()).normalized(),
        reference_profile=reference_profile,
    )


def configure_grid_worker_process(opencv_threads: int = 1) -> None:
    """Configure OpenCV inside a spawned grid-analysis worker process."""

    if cv2 is not None:
        cv2.setNumThreads(max(1, int(opencv_threads)))


def analyze_grid_frame_chunk(
    entries: tuple[tuple[str, str], ...],
    config: GridDamageAnalysisConfig,
    use_cache: bool = True,
    *,
    reference_profile: GridCellReferenceProfile | None = None,
    read_cache: bool | None = None,
    write_cache: bool | None = None,
) -> tuple[dict[str, GridFrameAnalysisResult], dict[str, str]]:
    """Analyze a compact chunk without importing or returning Qt objects."""

    payloads: dict[str, GridFrameAnalysisResult] = {}
    errors: dict[str, str] = {}
    for key, path_text in entries:
        try:
            result = analyze_grid_frame_path(
                path_text,
                frame_id=key,
                config=config,
                reference_profile=reference_profile,
                use_cache=use_cache,
                read_cache=read_cache,
                write_cache=write_cache,
            )
        except Exception as error:
            errors[str(key)] = f"{type(error).__name__}: {error}"
            continue
        if isinstance(result, GridFrameAnalysisResult):
            payloads[str(key)] = result
        else:
            errors[str(key)] = "decode_error"
    return payloads, errors


def classify_confidence_map_kind(image: np.ndarray) -> str:
    """Classify a grayscale map as class probability (bimodal) or confidence (mostly high)."""

    gray = _normalize_grayscale(image)
    values = np.asarray(gray, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return "confidence"
    step = max(1, int(values.size // 120_000))
    sample = values[::step]
    p10, p50, p90 = (float(v) for v in np.percentile(sample, [10, 50, 90]))
    low_share = float(np.mean(sample < 40.0))
    high_share = float(np.mean(sample > 200.0))
    if low_share >= 0.12 and high_share >= 0.12 and p10 < 50.0 and p90 > 180.0:
        return "class_probability"
    if p10 >= 170.0 or (high_share >= 0.70 and p50 >= 220.0):
        return "confidence"
    if low_share >= 0.20 and high_share >= 0.20:
        return "class_probability"
    return "confidence"


def _confidence_needs_binary_message_result(
    binary_result: GridFrameAnalysisResult | None,
    *,
    frame_id: str,
    frame_path: str,
) -> GridFrameAnalysisResult:
    """Empty confidence-layer result explaining that a binary mask is required."""

    width = int(getattr(binary_result, "image_width", 0) or 0) if binary_result is not None else 0
    height = int(getattr(binary_result, "image_height", 0) or 0) if binary_result is not None else 0
    return GridFrameAnalysisResult(
        frame_id=str(frame_id or ""),
        frame_path=str(frame_path or ""),
        image_width=width,
        image_height=height,
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=0,
        detected_cells=0,
        normal_cells=0,
        suspicious_cells=0,
        broken_cells=0,
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.0,
        severity_level="needs_binary_for_confidence",
        grid_detected=bool(getattr(binary_result, "grid_detected", False)) if binary_result is not None else False,
        cell_width=int(getattr(binary_result, "cell_width", 0) or 0) if binary_result is not None else 0,
        cell_height=int(getattr(binary_result, "cell_height", 0) or 0) if binary_result is not None else 0,
        model_file_status="needs_binary",
    )


def _cell_interior_mask_from_outline(
    outline: Sequence[tuple[int, int]] | tuple[tuple[int, int], ...],
    shape: tuple[int, int],
    *,
    erode_px: int = 3,
) -> tuple[int, int, np.ndarray] | None:
    """Return ``(x0, y0, mask)`` for the cell interior, cropped to the outline.

    A full-frame mask per cell costs cells x frame pixels of memory and time.
    The crop keeps a margin wider than the erosion kernel, so the eroded pixels
    match a full-frame erosion exactly.
    """

    if cv2 is None or not outline:
        return None
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        return None
    points = np.asarray([(int(x), int(y)) for x, y in outline], dtype=np.int32)
    if points.ndim != 2 or points.shape[0] < 3:
        return None
    margin = max(0, int(erode_px)) + 1
    x0 = max(0, int(points[:, 0].min()) - margin)
    y0 = max(0, int(points[:, 1].min()) - margin)
    x1 = min(width, int(points[:, 0].max()) + margin + 1)
    y1 = min(height, int(points[:, 1].max()) + margin + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(mask, [(points - (x0, y0)).astype(np.int32).reshape(-1, 1, 2)], 255)
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        eroded = cv2.erode(mask, kernel, iterations=1)
        if int(np.count_nonzero(eroded)) >= 8:
            mask = eroded
    return x0, y0, mask


def score_confidence_fill_on_binary_result(
    binary_result: GridFrameAnalysisResult,
    confidence_image: np.ndarray,
    config: GridDamageAnalysisConfig | None = None,
    *,
    erode_px: int = 3,
) -> GridFrameAnalysisResult:
    """Score filled/partial fill using binary cell geometry and confidence interior values."""

    cfg = (config or GridDamageAnalysisConfig()).normalized()
    conf = _normalize_grayscale(confidence_image)
    if conf.ndim != 2 or binary_result.image_height <= 0 or binary_result.image_width <= 0:
        return replace(binary_result, per_cell_results=(), damage_score=0.0, severity_level="OK")
    if conf.shape[0] != int(binary_result.image_height) or conf.shape[1] != int(binary_result.image_width):
        if cv2 is not None:
            conf = cv2.resize(
                conf,
                (int(binary_result.image_width), int(binary_result.image_height)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            return replace(binary_result, per_cell_results=(), damage_score=0.0, severity_level="OK")

    # Soft → harder thresholds: fill_sensitivity 0..100 → filled_floor 0.70..0.50, partial 0.20..0.10
    sens = float(np.clip(float(cfg.fill_sensitivity), 0.0, 100.0)) / 100.0
    filled_floor = float(0.70 - 0.20 * sens)
    partial_floor = float(0.20 - 0.10 * sens)
    min_drop_levels = 5.0
    mad_k = 3.0

    # Pieces marked debris only because no filter claimed them are not cells: a dark
    # blob must not read as a filled cell or move the interior norm. The binary layer keeps them.
    cells = [
        cell
        for cell in (binary_result.per_cell_results or ())
        if LEFTOVER_DEBRIS_FEATURE not in dict(cell.feature_snapshot or ())
    ]
    frame_w = int(binary_result.image_width)
    frame_h = int(binary_result.image_height)
    interior_means: list[float] = []
    interior_masks: list[tuple[int, int, np.ndarray] | None] = []
    for cell in cells:
        x, y, w, h = (int(v) for v in cell.bbox[:4])
        touches_border = x <= 1 or y <= 1 or (x + w) >= (frame_w - 1) or (y + h) >= (frame_h - 1)
        if touches_border:
            interior_masks.append(None)
            interior_means.append(float("nan"))
            continue
        crop = _cell_interior_mask_from_outline(cell.outline, conf.shape, erode_px=erode_px)
        interior_masks.append(crop)
        if crop is None or not np.any(crop[2]):
            interior_means.append(float("nan"))
            continue
        mx, my, mask = crop
        values = conf[my : my + mask.shape[0], mx : mx + mask.shape[1]][mask > 0].astype(np.float64)
        interior_means.append(float(np.median(values)) if values.size else float("nan"))

    valid_means = [value for value in interior_means if np.isfinite(value)]
    if len(valid_means) >= 5:
        norm = float(np.median(valid_means))
        mad = float(np.median(np.abs(np.asarray(valid_means, dtype=np.float64) - norm)))
    elif valid_means:
        norm = float(np.median(valid_means))
        mad = 0.0
    else:
        norm = 253.0
        mad = 0.0
    # Floor of 8 gray levels: prompt min is 5, but real maps sit at 250–253 with 1–3 noise.
    drop = max(8.0, min_drop_levels, mad_k * mad)
    threshold = float(norm - drop)

    scored: list[GridCellAnalysisResult] = []
    zone_cells = [
        cell
        for cell in cells
        if "conductor_zone" in set(str(reason) for reason in (cell.reasons or ()))
    ]
    analysis_cells = [
        cell
        for cell in cells
        if "conductor_zone" not in set(str(reason) for reason in (cell.reasons or ()))
    ]
    # Rebuild masks/means only for non-zone cells (zones were skipped above as touches_border/NaN).
    analysis_masks = []
    analysis_means = []
    for cell, crop, mean_value in zip(cells, interior_masks, interior_means):
        if "conductor_zone" in set(str(reason) for reason in (cell.reasons or ())):
            continue
        analysis_masks.append(crop)
        analysis_means.append(mean_value)
        if crop is None or not np.any(crop[2]) or not np.isfinite(mean_value):
            scored.append(
                replace(
                    cell,
                    status="normal",
                    score=0.0,
                    reasons=(),
                    mean_confidence=None,
                    uncertain_pixel_ratio=None,
                )
            )
            continue
        mx, my, mask = crop
        ys, xs = np.where(mask > 0)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        local_mask = mask[y0:y1, x0:x1] > 0
        local_conf = conf[my + y0 : my + y1, mx + x0 : mx + x1]
        uncertain = local_mask & (local_conf < threshold)
        interior_count = float(np.count_nonzero(local_mask))
        uncertain_ratio = float(np.count_nonzero(uncertain)) / max(1.0, interior_count)
        if cv2 is not None and np.any(uncertain):
            labels = np.zeros(uncertain.shape, dtype=np.int32)
            n_labels, labels = cv2.connectedComponents(uncertain.astype(np.uint8), connectivity=8)
            if n_labels > 1:
                sizes = np.bincount(labels.reshape(-1))
                largest = float(np.max(sizes[1:])) if sizes.size > 1 else 0.0
            else:
                largest = 0.0
            blob_fraction = largest / max(1.0, interior_count)
        else:
            blob_fraction = 0.0
        reasons: list[str] = []
        score = 0.0
        # Require the cell median itself to sit below the threshold so bright cells with
        # a few dark edge leftovers after erosion do not fire.
        median_low = float(mean_value) <= threshold
        if median_low and uncertain_ratio >= filled_floor:
            reasons.append("filled_cell")
            score = min(1.0, 0.78 + 0.35 * (uncertain_ratio - filled_floor))
        elif median_low and blob_fraction >= partial_floor:
            reasons.append("partial_filled_cell")
            score = min(0.92, 0.62 + 0.5 * blob_fraction)
        elif blob_fraction >= partial_floor and uncertain_ratio >= partial_floor:
            # Local blotch inside an otherwise bright cell (typical smear column).
            reasons.append("partial_filled_cell")
            score = min(0.90, 0.60 + 0.45 * blob_fraction)
        score, reasons_t = _filter_disabled_grid_reasons(score, tuple(reasons), cfg)
        is_bad = bool(reasons_t)
        scored.append(
            replace(
                cell,
                status=_status_for_reasons(reasons_t) if is_bad else "normal",
                score=float(score) if is_bad else 0.0,
                reasons=reasons_t if is_bad else (),
                mean_confidence=float(mean_value) / 255.0,
                uncertain_pixel_ratio=float(uncertain_ratio),
                feature_snapshot=(
                    ("confidence_interior_median", float(mean_value)),
                    ("confidence_threshold", float(threshold)),
                    ("uncertain_ratio", float(uncertain_ratio)),
                    ("uncertain_blob_fraction", float(blob_fraction)),
                    ("confidence_norm", float(norm)),
                ),
            )
        )

    scored.extend(zone_cells)
    _ = analysis_cells  # kept for clarity; scoring uses analysis_masks/means above
    bad = [cell for cell in scored if cell.status != "normal" and "conductor_zone" not in set(cell.reasons or ())]
    damage = _damage_score([cell for cell in scored if "conductor_zone" not in set(cell.reasons or ())], max(1, len(analysis_cells))) if analysis_cells else 0.0
    severity = cfg.severity_thresholds.level_for_score(damage)
    return replace(
        binary_result,
        detected_cells=len(analysis_cells),
        normal_cells=len(analysis_cells) - len(bad),
        suspicious_cells=sum(1 for cell in bad if cell.status == "suspicious"),
        broken_cells=sum(1 for cell in bad if cell.status == "broken"),
        missing_cells=0,
        artifact_cells=sum(1 for cell in bad if cell.status == "artifact"),
        damage_score=float(damage),
        severity_level=str(severity),
        per_cell_results=tuple(sorted(scored, key=lambda item: (item.row, -item.score))),
        component_count=int(binary_result.component_count),
        feature_clusters=(),
        debug=None,
        model_file_status="",
    )


def analyze_grid_frame_pair_chunk(
    entries: tuple[tuple[str, str, str], ...],
    config: GridDamageAnalysisConfig,
    use_cache: bool = True,
    *,
    reference_profile: GridCellReferenceProfile | None = None,
    requested_layers: tuple[str, ...] | None = None,
) -> tuple[dict[str, dict[str, GridFrameAnalysisResult]], dict[str, str]]:
    """Analyze linked confidence/binary files and return selected matrix payloads."""

    payloads: dict[str, dict[str, GridFrameAnalysisResult]] = {}
    errors: dict[str, str] = {}
    wanted = {
        str(layer)
        for layer in (requested_layers or ("confidence", "binary"))
        if str(layer) in {"confidence", "binary"}
    }
    if not wanted:
        wanted = {"confidence", "binary"}
    need_confidence = "confidence" in wanted
    need_binary = "binary" in wanted
    binary_config = replace(config, cell_representation="binary").normalized()
    for key, confidence_path, binary_path in entries:
        binary_obj = Path(binary_path) if binary_path else None
        if binary_obj is None or not binary_obj.is_file():
            missing = _missing_model_file_result(str(key), str(binary_path or ""))
            payloads[str(key)] = {layer: missing for layer in sorted(wanted)}
            continue
        try:
            binary_image = _load_cv2_grayscale_image(binary_obj)
            if binary_image is None:
                errors[str(key)] = "binary_decode_error"
                continue
            shared_zone = conductor_zone_map_from_mask_image(binary_image, binary_config)
            zone_token = "|".join(str(part) for part in _grid_cache_identity(binary_obj))
            confidence_file = Path(confidence_path) if confidence_path else None
            # Binary geometry is required for confidence fill (cells come from the mask).
            binary_result = analyze_grid_frame_path(
                binary_obj,
                frame_id=key,
                config=binary_config,
                reference_profile=reference_profile,
                use_cache=use_cache,
                conductor_zones=shared_zone,
                zone_cache_token=zone_token,
            )
            confidence_result = None
            if need_confidence:
                if confidence_file is None or not confidence_file.is_file():
                    confidence_result = _missing_model_file_result(str(key), str(confidence_path or ""))
                elif not isinstance(binary_result, GridFrameAnalysisResult):
                    confidence_result = _confidence_needs_binary_message_result(
                        None, frame_id=str(key), frame_path=str(confidence_path or "")
                    )
                else:
                    confidence_image = _load_cv2_grayscale_image(confidence_file)
                    if confidence_image is None:
                        errors[str(key)] = "confidence_decode_error"
                        continue
                    map_kind = classify_confidence_map_kind(confidence_image)
                    if map_kind == "class_probability":
                        # Threshold to a synthetic binary, then score the confidence map on those cells.
                        binary_from_prob = (confidence_image >= 128).astype(np.uint8) * 255
                        synthetic = detect_grid_cell_anomalies(
                            binary_from_prob,
                            frame_id=key,
                            frame_path=str(confidence_file),
                            config=binary_config,
                            reference_profile=reference_profile,
                            conductor_zones=shared_zone,
                        )
                        confidence_result = score_confidence_fill_on_binary_result(
                            synthetic, confidence_image, config
                        )
                    else:
                        confidence_result = score_confidence_fill_on_binary_result(
                            binary_result, confidence_image, config
                        )
        except Exception as error:
            errors[str(key)] = f"{type(error).__name__}: {error}"
            continue
        if need_confidence and not isinstance(confidence_result, GridFrameAnalysisResult):
            errors[str(key)] = "confidence_decode_error"
            continue
        if need_binary and not isinstance(binary_result, GridFrameAnalysisResult):
            errors[str(key)] = "binary_decode_error"
            continue
        frame_payload: dict[str, GridFrameAnalysisResult] = {}
        if "confidence" in wanted and isinstance(confidence_result, GridFrameAnalysisResult):
            frame_payload["confidence"] = confidence_result
        if "binary" in wanted and isinstance(binary_result, GridFrameAnalysisResult):
            frame_payload["binary"] = binary_result
        if frame_payload:
            payloads[str(key)] = frame_payload
    return payloads, errors


def analyze_grid_frame_single_source_path(
    path: Path | str,
    *,
    frame_id: str = "",
    config: GridDamageAnalysisConfig | None = None,
    reference_profile: GridCellReferenceProfile | None = None,
    use_cache: bool = True,
) -> tuple[str, GridFrameAnalysisResult] | None:
    """Infer whether one untyped source is a confidence outline or a binary result."""

    base_config = (config or GridDamageAnalysisConfig()).normalized()
    if uses_normal_bank(reference_profile, base_config):
        result = analyze_grid_frame_path(
            path, frame_id=frame_id, config=base_config, reference_profile=reference_profile, use_cache=use_cache
        )
        return None if result is None else ("binary", result)
    confidence_result = analyze_grid_frame_path(
        path,
        frame_id=frame_id,
        config=replace(base_config, cell_representation="confidence"),
        reference_profile=reference_profile,
        use_cache=use_cache,
    )
    binary_result = analyze_grid_frame_path(
        path,
        frame_id=frame_id,
        config=replace(base_config, cell_representation="binary"),
        reference_profile=None,
        use_cache=use_cache,
    )
    candidates = [
        ("confidence", confidence_result),
        ("binary", binary_result),
    ]
    valid = [(layer, result) for layer, result in candidates if isinstance(result, GridFrameAnalysisResult)]
    if not valid:
        return None

    def quality(item: tuple[str, GridFrameAnalysisResult]) -> tuple[int, float, int, int]:
        layer, result = item
        return (
            1 if result.grid_detected else 0,
            -float(result.damage_score),
            int(result.detected_cells),
            # Untyped mask paths are usually binary fills; outlines already win via damage_score.
            1 if layer == "binary" else 0,
        )

    return max(valid, key=quality)


def analyze_grid_frame_sources_chunk(
    entries: tuple[tuple[str, str, str], ...],
    config: GridDamageAnalysisConfig,
    use_cache: bool = True,
    *,
    reference_profile: GridCellReferenceProfile | None = None,
    single_source_layer: str | None = None,
    requested_layers: tuple[str, ...] | None = None,
) -> tuple[dict[str, dict[str, GridFrameAnalysisResult]], dict[str, str]]:
    """Analyze complete pairs and gracefully handle either source on its own."""

    payloads: dict[str, dict[str, GridFrameAnalysisResult]] = {}
    errors: dict[str, str] = {}
    confidence_config = replace(config, cell_representation="confidence").normalized()
    wanted = {
        str(layer)
        for layer in (requested_layers or ("confidence", "binary"))
        if str(layer) in {"confidence", "binary"}
    }
    if not wanted:
        wanted = {"confidence", "binary"}
    if uses_normal_bank(reference_profile, config):
        # One analysis of the mask; the confidence map only adds scores (decision Н2), so both
        # layers show the same objects.
        for key, confidence_path, binary_path in entries:
            mask_path = binary_path or confidence_path
            missing = next((item for item in (binary_path, confidence_path) if item and not Path(item).is_file()), None)
            if missing is not None and (not binary_path or missing == binary_path):
                payloads[str(key)] = {layer: _missing_model_file_result(str(key), str(missing)) for layer in sorted(wanted)}
                continue
            try:
                result = analyze_grid_frame_path(
                    mask_path,
                    frame_id=key,
                    config=config,
                    reference_profile=reference_profile,
                    use_cache=use_cache,
                    confidence_path=confidence_path if binary_path and confidence_path and Path(confidence_path).is_file() else None,
                )
            except Exception as error:
                errors[str(key)] = f"{type(error).__name__}: {error}"
                continue
            if result is None:
                errors[str(key)] = "decode_error"
                continue
            payloads[str(key)] = {layer: result for layer in sorted(wanted)}
        return payloads, errors
    for key, confidence_path, binary_path in entries:
        if confidence_path and binary_path:
            pair_payloads, pair_errors = analyze_grid_frame_pair_chunk(
                ((key, confidence_path, binary_path),),
                config,
                use_cache,
                reference_profile=reference_profile,
                requested_layers=tuple(wanted),
            )
            payloads.update(pair_payloads)
            errors.update(pair_errors)
            continue
        try:
            if confidence_path and not Path(confidence_path).is_file():
                payloads[str(key)] = {
                    layer: _missing_model_file_result(str(key), str(confidence_path)) for layer in sorted(wanted)
                }
                continue
            if binary_path and not Path(binary_path).is_file():
                payloads[str(key)] = {
                    layer: _missing_model_file_result(str(key), str(binary_path)) for layer in sorted(wanted)
                }
                continue
            if confidence_path:
                if "confidence" not in wanted:
                    continue
                result = analyze_grid_frame_path(
                    confidence_path,
                    frame_id=key,
                    config=confidence_config,
                    reference_profile=reference_profile,
                    use_cache=use_cache,
                )
                selected = ("confidence", result) if isinstance(result, GridFrameAnalysisResult) else None
            elif binary_path:
                forced_layer = str(single_source_layer or "")
                if forced_layer not in GRID_CELL_REPRESENTATION_MODES:
                    if "binary" in wanted:
                        forced_layer = "binary"
                    elif "confidence" in wanted:
                        forced_layer = "confidence"
                if forced_layer in GRID_CELL_REPRESENTATION_MODES:
                    forced_result = analyze_grid_frame_path(
                        binary_path,
                        frame_id=key,
                        config=replace(config, cell_representation=forced_layer),
                        reference_profile=reference_profile if forced_layer == "confidence" else None,
                        use_cache=use_cache,
                    )
                    selected = (
                        (forced_layer, forced_result) if isinstance(forced_result, GridFrameAnalysisResult) else None
                    )
                else:
                    selected = analyze_grid_frame_single_source_path(
                        binary_path,
                        frame_id=key,
                        config=config,
                        reference_profile=reference_profile,
                        use_cache=use_cache,
                    )
            else:
                selected = None
        except Exception as error:
            errors[str(key)] = f"{type(error).__name__}: {error}"
            continue
        if selected is None:
            errors[str(key)] = "decode_error"
            continue
        layer_key, result = selected
        if str(layer_key) not in wanted:
            continue
        payloads[str(key)] = {str(layer_key): result}
    return payloads, errors


def _as_probability_map(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    values = np.asarray(image)
    if values.ndim == 3:
        values = values[..., :3].mean(axis=2)
    if values.ndim != 2 or values.size == 0:
        return None
    if values.dtype == np.uint8:
        return (np.asarray(values, dtype=np.float32) / 255.0).astype(np.float32, copy=False)
    finite = np.nan_to_num(values.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    if float(np.nanmax(finite)) > 1.0 + 1e-6:
        finite = finite / 255.0
    return np.clip(finite, 0.0, 1.0).astype(np.float32, copy=False)


def _mask_from_grid_gray(image: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    gray = _normalize_grayscale(image)
    level = int(round(max(0.0, min(1.0, float(threshold))) * 255.0))
    return np.asarray(gray >= level, dtype=bool)


def _is_conflict_contour_jitter(
    conflict_roi: np.ndarray,
    union_boundary_roi: np.ndarray,
    *,
    area: int,
    width: int,
    height: int,
) -> bool:
    """Thin fringe between oversized detections is contour jitter, not a real class clash."""

    if area <= 0 or width <= 0 or height <= 0:
        return True
    min_side = min(int(width), int(height))
    max_side = max(int(width), int(height))
    # Hairline strips (1–2 px thick) are contour jitter even when they fill their bbox.
    if min_side <= 2 and max_side >= 6:
        return True
    if min_side <= 3 and max_side >= 8 and float(area) <= 0.45 * float(width * height):
        return True
    if min_side <= 2 and area <= 24:
        return True
    boundary = np.asarray(union_boundary_roi, dtype=bool)
    conflict = np.asarray(conflict_roi, dtype=bool)
    if not np.any(conflict):
        return True
    if np.any(boundary):
        overlap = float(np.count_nonzero(conflict & boundary)) / float(max(1, area))
        if overlap >= 0.62 and min_side <= 5:
            return True
        if overlap >= 0.78 and area <= 40:
            return True
    return False


def analyze_class_conflict_masks(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    frame_id: str = "",
    frame_path: str = "",
    min_area: int = GRID_CLASS_CONFLICT_MIN_AREA,
) -> GridFrameAnalysisResult:
    """Mark AND-overlap blobs where mutually exclusive classes fire together (e.g. ones ∩ zeros)."""

    first = np.asarray(mask_a, dtype=bool)
    second = np.asarray(mask_b, dtype=bool)
    if first.shape != second.shape:
        raise ValueError(f"Conflict masks have different shapes: {first.shape} vs {second.shape}")
    height, width = first.shape
    empty = GridFrameAnalysisResult(
        frame_id=str(frame_id or ""),
        frame_path=str(frame_path or ""),
        image_width=int(width),
        image_height=int(height),
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=0,
        detected_cells=0,
        normal_cells=0,
        suspicious_cells=0,
        broken_cells=0,
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.0,
        severity_level="OK",
        grid_detected=True,
        per_cell_results=(),
        component_count=0,
    )
    conflict = np.logical_and(first, second)
    if not np.any(conflict) or cv2 is None:
        return empty

    union = first | second
    union_u8 = np.asarray(union, dtype=np.uint8)
    eroded = cv2.erode(union_u8, np.ones((3, 3), dtype=np.uint8), iterations=1)
    union_boundary = union_u8.astype(bool) & ~eroded.astype(bool)

    conflict_u8 = np.asarray(conflict, dtype=np.uint8) * 255
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(conflict_u8, connectivity=8)
    findings: list[GridCellAnalysisResult] = []
    for index in range(1, int(count)):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        bw = int(stats[index, cv2.CC_STAT_WIDTH])
        bh = int(stats[index, cv2.CC_STAT_HEIGHT])
        component = labels[y : y + bh, x : x + bw] == index
        boundary_roi = union_boundary[y : y + bh, x : x + bw]
        if _is_conflict_contour_jitter(component, boundary_roi, area=area, width=bw, height=bh):
            continue
        cx = float(centroids[index][0])
        cy = float(centroids[index][1])
        score = float(min(1.0, 0.78 + 0.0025 * float(area)))
        findings.append(
            GridCellAnalysisResult(
                row=len(findings),
                col=0,
                bbox=(x, y, bw, bh),
                centroid=(cx, cy),
                contour_id=int(index),
                status="suspicious",
                score=score,
                reasons=("class_conflict",),
            )
        )

    if not findings:
        return empty
    findings = _suppress_overlapping_defect_boxes(findings)
    if not findings:
        return empty
    damage_score = _damage_score(findings, max(1, len(findings)))
    severity = GridDamageSeverityThresholds().level_for_score(damage_score)
    return GridFrameAnalysisResult(
        frame_id=str(frame_id or ""),
        frame_path=str(frame_path or ""),
        image_width=int(width),
        image_height=int(height),
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=int(len(findings)),
        detected_cells=int(len(findings)),
        normal_cells=0,
        suspicious_cells=int(len(findings)),
        broken_cells=0,
        missing_cells=0,
        artifact_cells=0,
        damage_score=float(damage_score),
        severity_level=str(severity),
        grid_detected=True,
        per_cell_results=tuple(findings),
        component_count=int(max(0, count - 1)),
    )


def _class_conflict_cache_key(
    path_a: Path,
    path_b: Path,
    *,
    frame_id: str,
    model_a_id: str,
    model_b_id: str,
    min_area: int,
) -> tuple[Any, ...]:
    return (
        GRID_CLASS_CONFLICT_ALGORITHM_VERSION,
        str(frame_id or ""),
        str(model_a_id),
        str(model_b_id),
        int(min_area),
        _grid_cache_identity(path_a),
        _grid_cache_identity(path_b),
    )


def analyze_class_conflict_frame_paths(
    path_a: Path | str,
    path_b: Path | str,
    *,
    frame_id: str = "",
    model_a_id: str = "",
    model_b_id: str = "",
    threshold: float = 0.5,
    min_area: int = GRID_CLASS_CONFLICT_MIN_AREA,
    use_cache: bool = True,
) -> GridFrameAnalysisResult | None:
    """Load two binary sources, compute filtered ones∩zeros conflicts, and cache the result."""

    first_path = Path(path_a)
    second_path = Path(path_b)
    cache_key = _class_conflict_cache_key(
        first_path,
        second_path,
        frame_id=frame_id,
        model_a_id=model_a_id,
        model_b_id=model_b_id,
        min_area=min_area,
    )
    cache_path = _grid_damage_cache_path(cache_key)
    if use_cache and cache_path.is_file():
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, GridFrameAnalysisResult):
                return payload
        except Exception as error:
            _LOGGER.warning("Ignoring corrupt class-conflict cache entry %s: %s", cache_path, error)

    image_a = _load_cv2_grayscale_image(first_path)
    image_b = _load_cv2_grayscale_image(second_path)
    if image_a is None or image_b is None:
        return None
    if image_a.shape != image_b.shape:
        if cv2 is None:
            return None
        image_b = cv2.resize(image_b, (image_a.shape[1], image_a.shape[0]), interpolation=cv2.INTER_NEAREST)
    result = analyze_class_conflict_masks(
        _mask_from_grid_gray(image_a, threshold=threshold),
        _mask_from_grid_gray(image_b, threshold=threshold),
        frame_id=frame_id,
        frame_path=f"{first_path.name}|{second_path.name}",
        min_area=min_area,
    )
    if use_cache:
        try:
            GRID_DAMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            atomic_pickle_dump(cache_path, result)
        except (OSError, pickle.PickleError, TypeError, ValueError) as error:
            _LOGGER.warning("Could not store class-conflict cache for %s: %s", frame_id, error)
    return result


def analyze_class_conflict_chunk(
    entries: tuple[tuple[str, str, str], ...],
    *,
    model_a_id: str,
    model_b_id: str,
    threshold: float = 0.5,
    min_area: int = GRID_CLASS_CONFLICT_MIN_AREA,
    use_cache: bool = True,
) -> tuple[dict[str, dict[str, GridFrameAnalysisResult]], dict[str, str]]:
    """Analyze class conflicts for (frame_key, path_a, path_b) batches."""

    payloads: dict[str, dict[str, GridFrameAnalysisResult]] = {}
    errors: dict[str, str] = {}
    for key, path_a, path_b in entries:
        try:
            result = analyze_class_conflict_frame_paths(
                path_a,
                path_b,
                frame_id=key,
                model_a_id=model_a_id,
                model_b_id=model_b_id,
                threshold=threshold,
                min_area=min_area,
                use_cache=use_cache,
            )
        except Exception as error:
            errors[str(key)] = f"{type(error).__name__}: {error}"
            continue
        if result is None:
            errors[str(key)] = "decode_error"
            continue
        payloads[str(key)] = {"derived_conflict": result}
    return payloads, errors


# Back-compat aliases (previous derived-XOR API).
_is_xor_contour_jitter = _is_conflict_contour_jitter
analyze_xor_residual_masks = analyze_class_conflict_masks
analyze_xor_residual_frame_paths = analyze_class_conflict_frame_paths
analyze_xor_residual_chunk = analyze_class_conflict_chunk


def _load_cv2_grayscale_image(path: Path | str) -> np.ndarray | None:
    """Load a grayscale image through OpenCV while supporting non-ASCII Windows paths."""

    if cv2 is None:
        return None
    path_obj = Path(path)
    profiler = current_profiler()
    if profiler is not None:
        profiler.increment("files.read")
    try:
        encoded = np.fromfile(str(path_obj), dtype=np.uint8)
    except Exception:
        encoded = np.asarray([], dtype=np.uint8)
    if encoded.size > 0:
        try:
            image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                if profiler is not None:
                    profiler.increment("files.decoded")
                    profiler.increment("bytes.read", int(encoded.nbytes))
                return np.asarray(image, dtype=np.uint8)
        except Exception as error:
            _LOGGER.debug("OpenCV imdecode failed for %s: %s", path_obj, error)
    try:
        image = cv2.imread(str(path_obj), cv2.IMREAD_GRAYSCALE)
    except Exception:
        image = None
    if image is None:
        return None
    if profiler is not None:
        profiler.increment("files.decoded")
    return np.asarray(image, dtype=np.uint8)


def _normalize_grayscale(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 3:
        values = values[..., :3].mean(axis=2)
    if values.ndim != 2:
        return np.zeros((1, 1), dtype=np.uint8)
    if values.dtype == np.uint8:
        return np.ascontiguousarray(values)
    if values.size > 0 and float(np.nanmax(values)) <= 1.0:
        values = values * 255.0
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0), 0.0, 255.0).astype(np.uint8)


def _threshold_grid(gray: np.ndarray, config: GridDamageAnalysisConfig) -> np.ndarray:
    values = np.ascontiguousarray(gray)
    if config.blur_radius > 1 and cv2 is not None:
        kernel = int(config.blur_radius) | 1
        values = cv2.GaussianBlur(values, (kernel, kernel), 0)
    if str(config.threshold_mode) == "adaptive":
        threshold = cv2.adaptiveThreshold(
            values,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            int(config.adaptive_block_size),
            float(config.adaptive_c),
        )
        otsu_level = 127.0
    else:
        otsu_level, threshold = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground_ratio = float(np.count_nonzero(threshold)) / max(1.0, float(threshold.size))
    representation = str(getattr(config, "cell_representation", "confidence") or "confidence")
    border = np.concatenate((threshold[0, :], threshold[-1, :], threshold[:, 0], threshold[:, -1]))
    border_foreground_ratio = float(np.count_nonzero(border)) / max(1.0, float(border.size))
    should_invert = border_foreground_ratio > 0.72 if representation == "binary" else foreground_ratio > 0.62
    if should_invert:
        threshold = cv2.bitwise_not(threshold)
        values_for_dirt = cv2.bitwise_not(values) if representation == "binary" else values
    else:
        values_for_dirt = values
    # Recover grainy/dim debris that a high Otsu split into sub-speckle fragments.
    if representation == "binary" and cv2 is not None:
        threshold = _recover_binary_dirt_components(values_for_dirt, threshold, float(otsu_level))
    if config.morphology_open > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(config.morphology_open), int(config.morphology_open)))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    if config.morphology_close > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(config.morphology_close), int(config.morphology_close)))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    return np.ascontiguousarray(threshold)


def _recover_binary_dirt_components(
    values: np.ndarray,
    threshold: np.ndarray,
    otsu_level: float,
) -> np.ndarray:
    """OR in compact low-threshold blobs that Otsu alone would fragment away."""

    if cv2 is None:
        return threshold
    low_level = float(max(8.0, min(20.0, float(otsu_level) * 0.18)))
    if low_level >= float(otsu_level) - 1.0:
        return threshold
    _level, low = cv2.threshold(np.ascontiguousarray(values), low_level, 255, cv2.THRESH_BINARY)
    n_labels, labels, stats, _cents = cv2.connectedComponentsWithStats(low, 8)
    if n_labels <= 1:
        return threshold
    accepted: list[int] = []
    for index in range(1, n_labels):
        area = int(stats[index, cv2.CC_STAT_AREA])
        bw = int(stats[index, cv2.CC_STAT_WIDTH])
        bh = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area < 8 or area > 1200:
            continue
        if max(bw, bh) > 64 or min(bw, bh) < 2:
            continue
        if float(area) < 0.22 * float(max(1, bw * bh)) and area < 24:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        roi = threshold[y : y + bh, x : x + bw]
        label_roi = labels[y : y + bh, x : x + bw] == index
        overlap = float(np.count_nonzero(roi[label_roi])) / max(1.0, float(area))
        # Keep blobs that Otsu only partially captured (grainy dirt), skip solid cells.
        if overlap >= 0.62:
            continue
        accepted.append(index)
    if not accepted:
        return threshold
    keep = np.zeros(n_labels, dtype=np.uint8)
    keep[np.asarray(accepted, dtype=np.int32)] = 255
    recovered = keep[labels]
    # Close only the recovered dirt mask so grain reconnects without bridging real cells.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    recovered = cv2.morphologyEx(recovered, cv2.MORPH_CLOSE, kernel)
    return cv2.bitwise_or(threshold, recovered)


def _foreground_count_tables(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integral image plus column/row prefixes of a nonzero mask."""

    binary = np.asarray(mask > 0, dtype=np.uint8)
    height, width = binary.shape[:2]
    if cv2 is not None and binary.size:
        integral = cv2.integral(binary)
        if isinstance(integral, tuple):
            integral = integral[0]
        integral = np.asarray(integral)
    else:
        integral = np.zeros((height + 1, width + 1), dtype=np.int64)
        if binary.size:
            integral[1:, 1:] = np.cumsum(np.cumsum(binary, axis=0, dtype=np.int64), axis=1)
    column_prefix = np.zeros((height + 1, width), dtype=np.int64)
    row_prefix = np.zeros((height, width + 1), dtype=np.int64)
    if height and width:
        np.cumsum(binary, axis=0, out=column_prefix[1:], dtype=np.int64)
        np.cumsum(binary, axis=1, out=row_prefix[:, 1:], dtype=np.int64)
    return integral, column_prefix, row_prefix


def _rect_nonzero_count(integral: np.ndarray, x: int, y: int, width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        return 0
    return int(integral[y + height, x + width] - integral[y, x + width] - integral[y + height, x] + integral[y, x])


def _outline_side_coverages_from_tables(
    column_prefix: np.ndarray,
    row_prefix: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    band = max(1, int(round(min(width, height) * 0.18)))
    band = min(band, width, height)
    top = float(np.count_nonzero(column_prefix[y + band, x : x + width] - column_prefix[y, x : x + width]) / width)
    bottom = float(
        np.count_nonzero(
            column_prefix[y + height, x : x + width] - column_prefix[y + height - band, x : x + width]
        )
        / width
    )
    left = float(np.count_nonzero(row_prefix[y : y + height, x + band] - row_prefix[y : y + height, x]) / height)
    right = float(
        np.count_nonzero(row_prefix[y : y + height, x + width] - row_prefix[y : y + height, x + width - band]) / height
    )
    return (top, bottom, left, right)


@dataclass(frozen=True, slots=True)
class _ZoneProbe:
    """Area and centroid only. Enough to place the conductor mask before the expensive features."""

    contour_id: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: float


def _empty_conductor_zone_map():
    from .grid_zones import _empty_map

    return _empty_map()


def _zone_probe_candidates(contours, hierarchy, shape: tuple[int, int], config: GridDamageAnalysisConfig) -> list[_ZoneProbe]:
    """Same admission rules as candidate extraction, without hull, perimeter, or fill features."""

    if hierarchy is None:
        hierarchy_array = np.zeros((0, 4), dtype=np.int32)
    else:
        hierarchy_array = np.asarray(hierarchy).reshape(-1, 4)
    height, width = shape
    calibrated = str(getattr(config, "scoring_mode", "legacy") or "legacy") == "calibrated"
    min_side = 2 if calibrated else int(config.min_cell_size)
    min_area = 2.0 if calibrated else float(config.min_contour_area)
    probes: list[_ZoneProbe] = []
    areas: dict[int, float] = {}
    rects: dict[int, tuple[int, int, int, int]] = {}

    def contour_area(contour_index: int) -> float:
        cached = areas.get(contour_index)
        if cached is None:
            cached = float(abs(cv2.contourArea(contours[contour_index])))
            areas[contour_index] = cached
        return cached

    def contour_rect(contour_index: int) -> tuple[int, int, int, int]:
        cached = rects.get(contour_index)
        if cached is None:
            x_value, y_value, width_value, height_value = cv2.boundingRect(contours[contour_index])
            cached = (int(x_value), int(y_value), int(width_value), int(height_value))
            rects[contour_index] = cached
        return cached

    for index, contour in enumerate(contours):
        parent_idx = int(hierarchy_array[index][3]) if index < len(hierarchy_array) else -1
        x, y, w, h = contour_rect(index)
        if w < min_side or h < min_side:
            continue
        if w >= max(1, width - 1) and h >= max(1, height - 1):
            continue
        area = contour_area(index)
        if area < min_area:
            continue
        if parent_idx >= 0:
            if parent_idx >= len(contours):
                continue
            _px, _py, parent_w, parent_h = contour_rect(parent_idx)
            parent_area = max(1.0, contour_area(parent_idx))
            axis_ratio = max(float(w) / max(1.0, float(parent_w)), float(h) / max(1.0, float(parent_h)))
            if area >= 0.40 * parent_area or axis_ratio >= 0.70:
                continue
            if max(w, h) > 64 and area > max(120.0, 0.18 * parent_area):
                continue
        moments = cv2.moments(contour)
        if abs(float(moments.get("m00", 0.0))) > 1e-6:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx = float(x + w / 2.0)
            cy = float(y + h / 2.0)
        probes.append(
            _ZoneProbe(
                contour_id=int(index),
                bbox=(int(x), int(y), int(w), int(h)),
                centroid=(cx, cy),
                area=float(area),
            )
        )
    return probes


@dataclass(frozen=True, slots=True)
class _ZoneSizeSample:
    area: float
    bbox: tuple[int, int, int, int]
    solidity: float
    extent: float
    fill_ratio: float
    interior_fill_ratio: float
    center_fill_ratio: float
    aspect_ratio: float


def _zone_size_hint(
    probes: list[_ZoneProbe], contours, threshold: np.ndarray
) -> tuple[float, float, float]:
    """Same size cluster the calibrated profile uses, without feature extraction on speckles."""

    from .grid_calibration import robust_frame_reference

    sized = [item for item in probes if float(item.area) >= 40.0]
    if len(sized) < 8 or cv2 is None:
        return 0.0, 0.0, 0.0
    integral, _column_prefix, _row_prefix = _foreground_count_tables(threshold)
    samples: list[_ZoneSizeSample] = []
    for item in sized:
        x, y, width, height = item.bbox
        hull_area = max(1.0, float(abs(cv2.contourArea(cv2.convexHull(contours[item.contour_id])))))
        bbox_area = float(max(1, width * height))
        fill_ratio = float(_rect_nonzero_count(integral, x, y, width, height)) / bbox_area
        border = max(2, int(round(min(width, height) * 0.22)))
        if width > border * 2 + 1 and height > border * 2 + 1:
            inner_width = width - 2 * border
            inner_height = height - 2 * border
            interior = float(
                _rect_nonzero_count(integral, x + border, y + border, inner_width, inner_height)
            ) / max(1.0, float(inner_width * inner_height))
        else:
            interior = fill_ratio
        center_border = max(border + 1, int(round(min(width, height) * 0.36)))
        if width > center_border * 2 + 1 and height > center_border * 2 + 1:
            center_width = width - 2 * center_border
            center_height = height - 2 * center_border
            center = float(
                _rect_nonzero_count(integral, x + center_border, y + center_border, center_width, center_height)
            ) / max(1.0, float(center_width * center_height))
        else:
            center = interior
        samples.append(
            _ZoneSizeSample(
                area=float(item.area),
                bbox=item.bbox,
                solidity=float(item.area) / hull_area,
                extent=float(item.area) / bbox_area,
                fill_ratio=fill_ratio,
                interior_fill_ratio=interior,
                center_fill_ratio=center,
                aspect_ratio=float(width) / max(1.0, float(height)),
            )
        )
    reference = robust_frame_reference(samples)
    if not reference or float(reference.get("area", 0.0)) < 36.0:
        return 0.0, 0.0, 0.0
    return float(reference["area"]), float(reference["width"]), float(reference["height"])


def _zone_cell_record(zone, *, row: int, contour_id: int) -> GridCellAnalysisResult:
    outline = tuple((int(point[0]), int(point[1])) for point in getattr(zone, "contour", ()) or ())
    centroid = _outline_centroid(outline, zone.bbox)
    return GridCellAnalysisResult(
        row=int(row),
        col=0,
        bbox=tuple(int(value) for value in zone.bbox[:4]),
        centroid=centroid,
        contour_id=int(contour_id),
        status="zone",
        score=0.0,
        reasons=("conductor_zone",),
        outline=outline,
    )


def _outline_centroid(
    outline: tuple[tuple[int, int], ...], bbox: tuple[int, int, int, int]
) -> tuple[float, float]:
    if cv2 is not None and len(outline) >= 3:
        array = np.asarray(outline, dtype=np.int32).reshape(-1, 1, 2)
        moments = cv2.moments(array)
        if abs(float(moments.get("m00", 0.0))) > 1e-6:
            return (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))
    return (
        float(bbox[0]) + 0.5 * float(bbox[2]),
        float(bbox[1]) + 0.5 * float(bbox[3]),
    )


def _extract_candidates(
    contours,
    hierarchy,
    threshold: np.ndarray,
    shape: tuple[int, int],
    config: GridDamageAnalysisConfig,
    zone_map=None,
    zone_skip_count: list[int] | None = None,
) -> list[_ContourCandidate]:
    if hierarchy is None:
        hierarchy_array = np.zeros((0, 4), dtype=np.int32)
    else:
        hierarchy_array = np.asarray(hierarchy).reshape(-1, 4)
    height, width = shape
    integral, column_prefix, row_prefix = _foreground_count_tables(threshold)
    area_by_index: dict[int, float] = {}
    rect_by_index: dict[int, tuple[int, int, int, int]] = {}

    def contour_area(contour_index: int) -> float:
        cached = area_by_index.get(contour_index)
        if cached is None:
            cached = float(abs(cv2.contourArea(contours[contour_index])))
            area_by_index[contour_index] = cached
        return cached

    def contour_rect(contour_index: int) -> tuple[int, int, int, int]:
        cached = rect_by_index.get(contour_index)
        if cached is None:
            x_value, y_value, width_value, height_value = cv2.boundingRect(contours[contour_index])
            cached = (int(x_value), int(y_value), int(width_value), int(height_value))
            rect_by_index[contour_index] = cached
        return cached

    depth_by_index: dict[int, int] = {}

    def nesting_depth(contour_index: int) -> int:
        chain: list[int] = []
        current = contour_index
        while current >= 0 and current not in depth_by_index and current < len(hierarchy_array):
            chain.append(current)
            current = int(hierarchy_array[current][3])
        depth = depth_by_index.get(current, -1) if current >= 0 else -1
        for item in reversed(chain):
            depth += 1
            depth_by_index[item] = depth
        return depth_by_index.get(contour_index, 0)

    candidates: list[_ContourCandidate] = []
    for index, contour in enumerate(contours):
        parent_idx = int(hierarchy_array[index][3]) if index < len(hierarchy_array) else -1
        calibrated = str(getattr(config, "scoring_mode", "legacy") or "legacy") == "calibrated"
        # Odd depth is a hole in the mask, not a piece of it. A hole counts against
        # its cell (the hole check in scoring), not as debris.
        is_hole = bool(calibrated and parent_idx >= 0 and nesting_depth(index) % 2 == 1)
        min_side = 2 if calibrated else int(config.min_cell_size)
        min_area = 2.0 if calibrated else float(config.min_contour_area)
        x, y, w, h = contour_rect(index)
        if w < min_side or h < min_side:
            continue
        if w >= max(1, width - 1) and h >= max(1, height - 1):
            continue
        area = contour_area(index)
        if area < min_area:
            continue
        if parent_idx >= 0:
            # Nested contours are usually holes; keep only compact debris inside a parent cell.
            if parent_idx >= len(contours):
                continue
            _px, _py, parent_w, parent_h = contour_rect(parent_idx)
            parent_area = max(1.0, contour_area(parent_idx))
            axis_ratio = max(float(w) / max(1.0, float(parent_w)), float(h) / max(1.0, float(parent_h)))
            if area >= 0.40 * parent_area or axis_ratio >= 0.70:
                continue
            if max(w, h) > 64 and area > max(120.0, 0.18 * parent_area):
                continue
        bbox_area = float(max(1, w * h))
        moments = cv2.moments(contour)
        if abs(float(moments.get("m00", 0.0))) > 1e-6:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx = float(x + w / 2.0)
            cy = float(y + h / 2.0)
        if zone_map is not None and zone_map.contains(cx, cy):
            if zone_skip_count is not None:
                zone_skip_count[0] += 1
            continue
        hull = cv2.convexHull(contour)
        hull_area = max(1.0, float(abs(cv2.contourArea(hull))))
        perimeter = float(cv2.arcLength(contour, True))
        epsilon = max(0.5, 0.025 * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        child_areas: list[float] = []
        child = int(hierarchy_array[index][2]) if index < len(hierarchy_array) else -1
        while child >= 0 and child < len(contours):
            child_areas.append(contour_area(child))
            next_child = int(hierarchy_array[child][0]) if child < len(hierarchy_array) else -1
            if next_child == child:
                break
            child = next_child
        fill_ratio = float(_rect_nonzero_count(integral, x, y, w, h)) / bbox_area
        side_coverages = _outline_side_coverages_from_tables(column_prefix, row_prefix, x, y, w, h)
        outline_min_side_coverage = float(min(side_coverages, default=0.0))
        outline_mean_side_coverage = float(sum(side_coverages) / len(side_coverages)) if side_coverages else 0.0
        outline_side_imbalance = float(max(side_coverages, default=0.0) - min(side_coverages, default=0.0))
        border = max(2, int(round(min(w, h) * 0.22)))
        if w > border * 2 + 1 and h > border * 2 + 1:
            inner_width = w - 2 * border
            inner_height = h - 2 * border
            interior_fill_ratio = float(
                _rect_nonzero_count(integral, x + border, y + border, inner_width, inner_height)
            ) / max(1.0, float(inner_width * inner_height))
        else:
            interior_fill_ratio = fill_ratio
        center_border = max(border + 1, int(round(min(w, h) * 0.36)))
        if w > center_border * 2 + 1 and h > center_border * 2 + 1:
            center_width = w - 2 * center_border
            center_height = h - 2 * center_border
            center_fill_ratio = float(
                _rect_nonzero_count(integral, x + center_border, y + center_border, center_width, center_height)
            ) / max(1.0, float(center_width * center_height))
        else:
            center_fill_ratio = interior_fill_ratio
        inner_hole_ratio = max(child_areas, default=0.0) / max(1.0, area)
        border_margin = max(2, min(6, int(round(max(1, int(config.min_cell_size)) * 0.75))))
        candidates.append(
            _ContourCandidate(
                contour_id=int(index),
                contour=contour,
                bbox=(int(x), int(y), int(w), int(h)),
                centroid=(float(cx), float(cy)),
                area=float(area),
                bbox_area=bbox_area,
                aspect_ratio=float(w / max(1.0, h)),
                extent=float(area / bbox_area),
                solidity=float(area / hull_area),
                perimeter=perimeter,
                approx_vertices=int(len(approx)),
                fill_ratio=fill_ratio,
                interior_fill_ratio=interior_fill_ratio,
                center_fill_ratio=center_fill_ratio,
                outline_min_side_coverage=outline_min_side_coverage,
                outline_mean_side_coverage=outline_mean_side_coverage,
                outline_side_imbalance=outline_side_imbalance,
                inner_hole_ratio=float(inner_hole_ratio),
                child_count=int(len(child_areas)),
                touches_border=bool(
                    x <= border_margin
                    or y <= border_margin
                    or x + w >= width - border_margin
                    or y + h >= height - border_margin
                ),
                is_hole=is_hole,
            )
        )
    if str(getattr(config, "scoring_mode", "legacy") or "legacy") == "calibrated" and len(candidates) > 4000:
        candidates.sort(key=lambda item: float(item.area), reverse=True)
        del candidates[4000:]
    return candidates


def _outline_side_coverages(roi: np.ndarray) -> tuple[float, float, float, float]:
    height, width = np.asarray(roi).shape[:2]
    if width <= 0 or height <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    band = max(1, int(round(min(width, height) * 0.18)))
    band = min(band, max(1, width), max(1, height))
    top = float(np.count_nonzero(np.any(roi[:band, :], axis=0)) / width)
    bottom = float(np.count_nonzero(np.any(roi[height - band :, :], axis=0)) / width)
    left = float(np.count_nonzero(np.any(roi[:, :band], axis=1)) / height)
    right = float(np.count_nonzero(np.any(roi[:, width - band :], axis=1)) / height)
    return (top, bottom, left, right)


def _normal_seed_candidates(
    candidates: list[_ContourCandidate],
    *,
    config: GridDamageAnalysisConfig | None = None,
) -> list[_ContourCandidate]:
    representation = str(getattr(config, "cell_representation", "confidence") or "confidence")
    if representation == "binary":
        # Prefer near-square filled cells so bad aspect ratios do not poison the
        # same-frame median when a reference frame is not available.
        preferred = [
            item
            for item in candidates
            if 0.78 <= item.aspect_ratio <= 1.30
            and 0.42 <= item.fill_ratio <= 0.98
            and item.interior_fill_ratio >= 0.38
            and item.solidity >= 0.76
            and item.approx_vertices <= 14
            and float(item.area) >= 40.0
            and min(int(item.bbox[2]), int(item.bbox[3])) >= 8
        ]
        if len(preferred) >= 3:
            return preferred
        filtered = [
            item
            for item in candidates
            if 0.45 <= item.aspect_ratio <= 2.25
            and 0.42 <= item.fill_ratio <= 0.98
            and item.interior_fill_ratio >= 0.38
            and item.solidity >= 0.76
            and item.approx_vertices <= 14
            and float(item.area) >= 40.0
            and min(int(item.bbox[2]), int(item.bbox[3])) >= 8
        ]
        if len(filtered) >= 3:
            return filtered
        return [
            item
            for item in candidates
            if 0.35 <= item.aspect_ratio <= 2.80
            and 0.28 <= item.fill_ratio <= 1.0
            and item.solidity >= 0.55
            and float(item.area) >= 40.0
            and min(int(item.bbox[2]), int(item.bbox[3])) >= 8
        ]
    filtered = [
        item
        for item in candidates
        if 0.45 <= item.aspect_ratio <= 2.20
        and 0.03 <= item.fill_ratio <= 0.70
        and item.interior_fill_ratio <= 0.18
        and item.center_fill_ratio <= 0.12
        and item.solidity >= 0.76
        and item.approx_vertices <= 14
        and float(item.area) >= 24.0
        and min(int(item.bbox[2]), int(item.bbox[3])) >= 6
    ]
    if len(filtered) >= 3:
        return filtered
    return [
        item
        for item in candidates
        if 0.35 <= item.aspect_ratio <= 2.80
        and item.interior_fill_ratio <= 0.26
        and item.center_fill_ratio <= 0.18
        and item.solidity >= 0.55
        and float(item.area) >= 24.0
        and min(int(item.bbox[2]), int(item.bbox[3])) >= 6
    ]


def _retarget_local_medians(
    candidates: list[_ContourCandidate],
    local_medians: list[dict[str, float]],
    neighbor_ids: list[tuple[int, ...]],
    *,
    modal_width: float,
    modal_height: float,
    modal_area: float,
    modal_solidity: float,
    modal_extent: float,
    modal_interior: float,
) -> tuple[list[dict[str, float]], list[int]]:
    """Size each cell against the lattice, not against nearby crumbs or conductor fragments."""

    if not local_medians:
        return local_medians, []
    area_lo = 0.55 * max(1.0, float(modal_area))
    area_hi = 1.85 * max(1.0, float(modal_area))
    count = len(candidates)
    width_values = np.asarray([float(item.bbox[2]) for item in candidates], dtype=np.float64)
    height_values = np.asarray([float(item.bbox[3]) for item in candidates], dtype=np.float64)
    area_values = np.asarray([float(item.area) for item in candidates], dtype=np.float64)
    interior_values = np.asarray([float(item.interior_fill_ratio) for item in candidates], dtype=np.float64)
    solidity_values = np.asarray([float(item.solidity) for item in candidates], dtype=np.float64)
    extent_values = np.asarray([float(item.extent) for item in candidates], dtype=np.float64)
    max_neighbors = max((len(ids) for ids in neighbor_ids), default=0)
    if max_neighbors == 0:
        band_counts = [0] * count
    else:
        neighbor_index = np.full((count, max_neighbors), -1, dtype=np.int32)
        for index, ids in enumerate(neighbor_ids):
            if ids:
                neighbor_index[index, : len(ids)] = ids
        valid = neighbor_index >= 0
        safe_index = np.clip(neighbor_index, 0, max(0, count - 1))
        in_band = valid & (area_values[safe_index] >= area_lo) & (area_values[safe_index] <= area_hi)
        band_counts = [int(value) for value in np.count_nonzero(in_band, axis=1)]

        def _band_median(values: np.ndarray) -> np.ndarray:
            masked = np.where(in_band, values[safe_index], np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return np.nanmedian(masked, axis=1)

        width_median = _band_median(width_values)
        height_median = _band_median(height_values)
        area_median = _band_median(area_values)
        interior_median = _band_median(interior_values)
        solidity_median = _band_median(solidity_values)
        extent_median = _band_median(extent_values)
    fallback = {
        "width": max(1.0, float(modal_width)),
        "height": max(1.0, float(modal_height)),
        "area": max(1.0, float(modal_area)),
        "interior_fill": float(modal_interior),
        "solidity": float(modal_solidity),
        "extent": float(modal_extent),
    }
    retargeted: list[dict[str, float]] = []
    for index, band_count in enumerate(band_counts):
        if max_neighbors == 0 or int(band_count) < 3 or not np.isfinite(width_median[index]):
            retargeted.append(dict(fallback))
            continue
        local = {
            "width": max(1.0, float(width_median[index])),
            "height": max(1.0, float(height_median[index])),
            "area": max(1.0, float(area_median[index])),
            "interior_fill": float(interior_median[index]),
            "solidity": float(solidity_median[index]),
            "extent": float(extent_median[index]),
        }
        # A whole damaged row is its own median. Use it only when it still
        # matches the array. A clump far from the array is scored against the array.
        if _local_shape_matches_array(local, fallback):
            retargeted.append(local)
        else:
            retargeted.append(dict(fallback))
    return retargeted, band_counts


def _local_shape_matches_array(local: dict[str, float], array: dict[str, float]) -> bool:
    for key in ("width", "height", "area"):
        modal = float(array[key])
        value = float(local[key])
        if modal <= 1.0 or value <= 0.0:
            continue
        ratio = value / modal
        if ratio < 0.75 or ratio > 1.35:
            return False
    return True


def _score_merge_for_candidate(
    candidate: _ContourCandidate,
    *,
    cell_width: float,
    cell_height: float,
    merge_sensitivity: int,
) -> float:
    """Core-based merge score. Tiny blobs skip the distance transform."""

    from .grid_merge_cores import score_merge_cores

    cell_area = max(4.0, float(cell_width) * float(cell_height))
    if float(candidate.area) < 1.4 * cell_area:
        return 0.0
    return float(
        score_merge_cores(
            candidate.contour,
            candidate.bbox,
            area=float(candidate.area),
            cell_width=float(cell_width),
            cell_height=float(cell_height),
            merge_sensitivity=int(merge_sensitivity),
        )
    )


def _gate_calibrated_reasons(
    reasons: tuple[str, ...],
    *,
    width_ratio: float,
    height_ratio: float,
    area_ratio: float,
    area: float,
    in_lattice: bool,
    debris_min_area: float = DEBRIS_MIN_AREA_PX,
    solidity: float = 1.0,
) -> tuple[str, ...]:
    """Keep geometry/debris on slot-sized lattice cells. Merge uses cores, not the lattice."""

    largest = max(float(width_ratio), float(height_ratio))
    smallest = min(float(width_ratio), float(height_ratio))
    found = list(reasons)
    if not in_lattice:
        # Noise and conductor masses have no cell-sized neighbors. They are not slot defects.
        # merged_contour is decided by cores inside the blob and is not gated here.
        found = [
            reason
            for reason in found
            if reason
            not in {
                "broken_geometry",
                "small_artifact",
                "edge_clipped_cell",
                "filled_cell",
                "partial_filled_cell",
            }
        ]
    slot_like = 0.55 <= float(area_ratio) <= 1.90 and smallest >= 0.50 and largest <= 1.85
    # A cell drawn only in part: one side of a cell kept, the other cut short, and the
    # piece itself solid like a cell (noise blobs of a cell's width are not).
    partial_cell = (
        in_lattice
        and 0.80 <= largest <= 1.25
        and 0.30 <= smallest < 0.85
        and float(area_ratio) >= 0.25
        and float(solidity) >= 0.85
    )
    if "broken_geometry" in found and not (slot_like or partial_cell):
        found = [reason for reason in found if reason != "broken_geometry"]
    if partial_cell and "broken_geometry" in found:
        found = [reason for reason in found if reason != "small_artifact"]
    if "edge_clipped_cell" in found and (smallest < 0.35 or float(area) < 24.0):
        found = [reason for reason in found if reason != "edge_clipped_cell"]
    # JPEG crumbs are a few pixels. The operator sets how large real debris starts.
    if "small_artifact" in found and float(area) < float(debris_min_area):
        found = [reason for reason in found if reason != "small_artifact"]
    return tuple(dict.fromkeys(found))


def _promote_short_lattice_cell(
    reasons: tuple[str, ...],
    *,
    on_lattice_node: bool,
    area_ratio: float,
    width_ratio: float,
    height_ratio: float,
) -> tuple[str, ...]:
    """A short or smeared cell on a lattice node is broken geometry.

    The node is a column and a row of the array, on the conductor edge when
    that edge exists. A frame-edge clip is not this case. Debris stays off
    the node, or when the piece is under 0.30 of a cell. A flat double-width
    piece with a normal area is the squashed cell, not two full cells.
    """

    if not on_lattice_node:
        return reasons
    smallest = min(float(width_ratio), float(height_ratio))
    short = 0.30 <= float(area_ratio) <= 0.55 and smallest <= 0.80
    smeared = (
        float(width_ratio) >= 1.65
        and float(height_ratio) <= 1.15
        and 0.70 <= float(area_ratio) <= 1.15
    )
    if not short and not smeared:
        return reasons
    found = [reason for reason in reasons if reason != "small_artifact"]
    if "broken_geometry" not in found:
        found.append("broken_geometry")
    return tuple(found)


def _cluster_centers(coords: list[float], gap: float) -> list[float]:
    if not coords:
        return []
    ordered = sorted(float(value) for value in coords)
    groups: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(np.median(group)) for group in groups]


def _axis_pitch(centers: list[float]) -> float | None:
    """Typical one-cell step. Gaps between separate arrays are wider and are ignored."""

    if len(centers) < 2:
        return None
    gaps = np.diff(np.asarray(sorted(centers), dtype=np.float64))
    positive = gaps[gaps > 1.0]
    if positive.size == 0:
        return None
    cutoff = float(np.percentile(positive, 40))
    tight = positive[positive <= max(cutoff * 1.35, cutoff + 1.0)]
    if tight.size == 0:
        tight = positive
    return float(np.median(tight))


def _lattice_grid_axes(
    candidates: list[_ContourCandidate],
    *,
    modal_width: float,
    modal_height: float,
    modal_area: float,
) -> tuple[list[float], list[float], float | None, float | None]:
    """Column and row centers of the repeated cell, used to continue the lattice to the frame edge."""

    area_lo = 0.55 * max(1.0, float(modal_area))
    area_hi = 1.85 * max(1.0, float(modal_area))
    xs: list[float] = []
    ys: list[float] = []
    for item in candidates:
        if area_lo <= float(item.area) <= area_hi:
            xs.append(float(item.centroid[0]))
            ys.append(float(item.centroid[1]))
    columns = _cluster_centers(xs, max(4.0, 0.45 * float(modal_width)))
    rows = _cluster_centers(ys, max(4.0, 0.45 * float(modal_height)))
    return columns, rows, _axis_pitch(columns), _axis_pitch(rows)


def _on_grid_axis(value: float, centers: list[float], pitch: float | None, tolerance: float) -> bool:
    if not centers:
        return False
    nearest = min(centers, key=lambda center: abs(center - value))
    delta = abs(float(value) - float(nearest))
    if delta <= tolerance:
        return True
    if pitch is None or pitch <= 1.0:
        return False
    return abs(delta - float(pitch)) <= tolerance


def _inside_slot_band(
    start: float,
    end: float,
    centers: list[float],
    pitch: float | None,
    cell_size: float,
) -> bool:
    """Whether [start, end] along the frame edge stays inside one slot of the lattice."""

    if not centers:
        return False
    middle = (float(start) + float(end)) / 2.0
    nearest = min(centers, key=lambda center: abs(center - middle))
    if pitch is not None and pitch > 1.0:
        steps = round((middle - nearest) / float(pitch))
        nearest = nearest + steps * float(pitch)
    half = 0.5 * float(cell_size) + max(2.0, 0.15 * float(cell_size))
    return nearest - half <= float(start) and float(end) <= nearest + half


def _local_shape_measures(candidate: _ContourCandidate) -> tuple[float, float, float]:
    """(worst square-window fill, deepest notch, emptiest corner) of a contour.

    The window is a square with the short side, slid along the long side; its worst fill
    is the extent of the most damaged stretch. The notch is the deepest convexity defect
    as a share of the cell size in the direction it cuts in. Neither depends on how
    elongated the cell is; for a square cell both reduce to whole-cell measures. The corner is a
    square of half the short side at each bbox corner; a slanted or cut end empties one.
    """

    contour = getattr(candidate, "contour", None)
    x, y, width, height = (int(value) for value in candidate.bbox)
    short = min(width, height)
    if contour is None or cv2 is None or short < 4 or len(contour) < 4:
        return float(candidate.extent), 0.0, 1.0
    points = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
    local = points - np.array([[[x, y]]], dtype=np.int32)
    filled = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(filled, [local], -1, 1, thickness=-1)
    if height >= width:
        line = filled.sum(axis=1, dtype=np.float64)
    else:
        line = filled.sum(axis=0, dtype=np.float64)
    sums = np.convolve(line, np.ones(short, dtype=np.float64), mode="valid")
    window_fill = float(sums.min()) / float(short * short) if sums.size else float(candidate.extent)
    corner = max(2, int(round(short / 2.0)))
    corner_fill = min(
        float(filled[:corner, :corner].mean()),
        float(filled[:corner, -corner:].mean()),
        float(filled[-corner:, :corner].mean()),
        float(filled[-corner:, -corner:].mean()),
    )
    notch = 0.0
    try:
        hull = cv2.convexHull(points, returnPoints=False)
        defects = cv2.convexityDefects(points, hull) if hull is not None and len(hull) >= 3 else None
    except cv2.error:
        defects = None
    if defects is not None and len(defects):
        flat = points.reshape(-1, 2).astype(np.float64)
        for start, end, _far, depth in np.asarray(defects).reshape(-1, 4):
            edge = flat[int(end)] - flat[int(start)]
            length = float(np.hypot(edge[0], edge[1]))
            if length <= 0.0:
                continue
            # The notch points across the hull edge; measure it in cell sizes along that
            # direction: a bite into the side over the width, a dent in the end over the length.
            normal_x, normal_y = -edge[1] / length, edge[0] / length
            reach = float(np.hypot(normal_x / float(width), normal_y / float(height)))
            notch = max(notch, float(depth) / 256.0 * reach)
    return min(1.0, window_fill), notch, corner_fill


def _fits_edge_slot(
    candidate: _ContourCandidate,
    *,
    frame_width: int,
    frame_height: int,
    modal_width: float,
    modal_height: float,
    column_centers: list[float] | None = None,
    row_centers: list[float] | None = None,
    pitch_x: float | None = None,
    pitch_y: float | None = None,
) -> bool:
    """A frame-cut piece that fits inside one cell slot.

    Across the cut side it is thinner than a cell; along that side it is no longer than
    a cell and at least a quarter of one, so a small blob at the edge stays debris.
    It must also be a strip along the edge, or, where the lattice knows the rows
    (left/right edge) or columns (top/bottom edge), lie in one of them: a round crumb
    in the gap between slots is debris. The cut-off column or row itself is not
    needed, so sparse fields and elongated cells work too. Sizes are in cell sides of each axis, so the cell shape does not matter.
    """

    if not candidate.touches_border:
        return False
    x, y, width, height = (int(value) for value in candidate.bbox)
    margin = 2
    horizontal = x <= margin or x + width >= int(frame_width) - margin
    vertical = y <= margin or y + height >= int(frame_height) - margin
    width_fraction = float(width) / max(1.0, float(modal_width))
    height_fraction = float(height) / max(1.0, float(modal_height))
    rows = list(row_centers or ())
    columns = list(column_centers or ())

    def _in_rows() -> bool:
        return len(rows) < 2 or _inside_slot_band(float(y), float(y + height), rows, pitch_y, float(modal_height))

    def _in_columns() -> bool:
        return len(columns) < 2 or _inside_slot_band(float(x), float(x + width), columns, pitch_x, float(modal_width))

    if horizontal and vertical:
        return width_fraction <= 0.95 and height_fraction <= 0.95
    # A cell cut by the frame leaves a strip along the edge: at least twice as long
    # along it as across. That holds where the lattice rows or columns are irregular.
    if horizontal:
        strip = height >= 2 * width
        return width_fraction <= 0.95 and 0.25 <= height_fraction <= 1.25 and (strip or _in_rows())
    if vertical:
        strip = width >= 2 * height
        return height_fraction <= 0.95 and 0.25 <= width_fraction <= 1.25 and (strip or _in_columns())
    return False


def _prefer_edge_slot_over_debris(
    reasons: tuple[str, ...],
    candidate: _ContourCandidate,
    *,
    frame_width: int,
    frame_height: int,
    modal_width: float,
    modal_height: float,
    column_centers: list[float],
    row_centers: list[float],
    pitch_x: float | None,
    pitch_y: float | None,
) -> tuple[str, ...]:
    """A lattice slot cut by the frame is an edge clip, not debris.

    The visible piece is compared with the slot it continues. Geometry is left
    alone only when at least 60% of the slot is still in frame.
    """

    if not candidate.touches_border:
        return reasons
    # Debris that fits one cell slot at the frame edge is a cut cell even where the
    # lattice has no row or column there (sparse fields, elongated cells).
    free_fit = "small_artifact" in reasons and _fits_edge_slot(
        candidate,
        frame_width=frame_width,
        frame_height=frame_height,
        modal_width=modal_width,
        modal_height=modal_height,
        column_centers=column_centers,
        row_centers=row_centers,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
    )
    lattice = len(column_centers) >= 2 and len(row_centers) >= 2
    if not lattice and not free_fit:
        return reasons
    x, y, width, height = (int(value) for value in candidate.bbox)
    margin = 2
    sides: set[str] = set()
    if x <= margin:
        sides.add("left")
    if y <= margin:
        sides.add("top")
    if x + width >= int(frame_width) - margin:
        sides.add("right")
    if y + height >= int(frame_height) - margin:
        sides.add("bottom")
    if not sides:
        return reasons
    modal_w = max(1.0, float(modal_width))
    modal_h = max(1.0, float(modal_height))
    width_fraction = float(width) / modal_w
    height_fraction = float(height) / modal_h
    if max(width_fraction, height_fraction) > 1.70 and not free_fit:
        return reasons
    visible_x = float(x) + float(width) / 2.0
    visible_y = float(y) + float(height) / 2.0
    center_x = visible_x
    center_y = visible_y
    if "left" in sides and "right" not in sides:
        center_x -= (modal_w - float(width)) / 2.0
    elif "right" in sides and "left" not in sides:
        center_x += (modal_w - float(width)) / 2.0
    if "top" in sides and "bottom" not in sides:
        center_y -= (modal_h - float(height)) / 2.0
    elif "bottom" in sides and "top" not in sides:
        center_y += (modal_h - float(height)) / 2.0
    tolerance_x = max(4.0, 0.40 * modal_w)
    tolerance_y = max(4.0, 0.40 * modal_h)
    horizontal = "left" in sides or "right" in sides
    vertical = "top" in sides or "bottom" in sides

    def _on_slot(point_x: float, point_y: float) -> bool:
        return _on_grid_axis(point_x, column_centers, pitch_x, tolerance_x) and _on_grid_axis(
            point_y, row_centers, pitch_y, tolerance_y
        )

    on_slot = lattice and (_on_slot(center_x, center_y) or _on_slot(visible_x, visible_y))
    # A one-edge sliver as tall or as wide as a cell continues the border line
    # even when that whole row is missing from the full-cell cluster.
    if (
        lattice
        and horizontal
        and not vertical
        and width_fraction <= 0.40
        and 0.75 <= height_fraction <= 1.25
        and (
            _on_grid_axis(visible_x, column_centers, pitch_x, tolerance_x)
            or _on_grid_axis(center_x, column_centers, pitch_x, tolerance_x)
            or _on_grid_axis(visible_y, row_centers, pitch_y, tolerance_y)
        )
    ):
        on_slot = True
    if (
        lattice
        and vertical
        and not horizontal
        and height_fraction <= 0.40
        and 0.75 <= width_fraction <= 1.25
        and (
            _on_grid_axis(visible_y, row_centers, pitch_y, tolerance_y)
            or _on_grid_axis(center_y, row_centers, pitch_y, tolerance_y)
            or _on_grid_axis(visible_x, column_centers, pitch_x, tolerance_x)
        )
    ):
        on_slot = True
    # A sliver of a cell that lies mostly outside the frame, drawn only in part:
    # it sits inside the visible part of an edge slot, whatever its length.
    sliver = False
    if lattice and horizontal and not vertical and width_fraction <= 0.60 and height_fraction <= 1.25:
        inner_x = float(x + width) - modal_w / 2.0 if "left" in sides else float(x) + modal_w / 2.0
        sliver = _on_grid_axis(inner_x, column_centers, pitch_x, tolerance_x) and _inside_slot_band(
            float(y), float(y + height), row_centers, pitch_y, modal_h
        )
    if lattice and vertical and not horizontal and height_fraction <= 0.60 and width_fraction <= 1.25:
        inner_y = float(y + height) - modal_h / 2.0 if "top" in sides else float(y) + modal_h / 2.0
        sliver = _on_grid_axis(inner_y, row_centers, pitch_y, tolerance_y) and _inside_slot_band(
            float(x), float(x + width), column_centers, pitch_x, modal_w
        )
    if not on_slot and not sliver and not free_fit:
        return reasons
    cropped = (horizontal and width_fraction <= 0.95) or (vertical and height_fraction <= 0.95)
    if not cropped:
        return reasons
    if not sliver and not free_fit:
        if horizontal and not vertical and not 0.75 <= height_fraction <= 1.25:
            return reasons
        if vertical and not horizontal and not 0.75 <= width_fraction <= 1.25:
            return reasons
    if horizontal and vertical:
        visible = width_fraction * height_fraction
    elif horizontal:
        visible = width_fraction
    else:
        visible = height_fraction
    found = [reason for reason in reasons if reason != "small_artifact"]
    if visible < 0.60:
        found = [
            reason
            for reason in found
            if reason not in {"broken_geometry", "filled_cell", "partial_filled_cell"}
        ]
    if "edge_clipped_cell" not in found:
        found.append("edge_clipped_cell")
    return tuple(dict.fromkeys(found))


def _calibrated_size_profile(
    candidates: list[_ContourCandidate],
    *,
    frame_id: str = "",
    frame_path: str = "",
) -> tuple[GridCellReferenceProfile | None, list[_ContourCandidate]]:
    """Size cluster for calibrated scoring when the legacy fill seed does not match outline cells."""

    from .grid_calibration import modal_size_cluster, robust_frame_reference

    reference = robust_frame_reference(candidates)
    cluster = modal_size_cluster(candidates)
    if reference is None or len(cluster) < 4:
        return None, []
    profile = GridCellReferenceProfile(
        median_width=float(reference["width"]),
        median_height=float(reference["height"]),
        median_area=float(reference["area"]),
        median_fill=float(reference.get("fill", 0.0)),
        median_interior_fill=float(reference.get("interior_fill", 0.0)),
        median_center_fill=float(reference.get("center_fill", 0.0)),
        median_aspect=float(reference.get("aspect", 1.0)),
        candidate_count=int(len(candidates)),
        seed_count=int(len(cluster)),
        frame_id=str(frame_id or ""),
        frame_path=str(frame_path or ""),
    )
    return profile, cluster


def _grid_cell_reference_profile_from_candidates(
    candidates: list[_ContourCandidate],
    *,
    config: GridDamageAnalysisConfig,
    frame_id: str = "",
    frame_path: str = "",
) -> GridCellReferenceProfile | None:
    if len(candidates) < int(config.min_grid_candidate_cells):
        return None
    normal_seed = _normal_seed_candidates(candidates, config=config)
    # Never fall back to the full candidate list: grainy debris would poison the median
    # and turn large conductor masses into false merged_contour hits.
    if len(normal_seed) < 3:
        return None
    seed = normal_seed
    median_area = float(np.median([item.area for item in seed]))
    # Tiny "seed" from grain speckles is not a cell grid.
    if median_area < 36.0:
        return None
    return GridCellReferenceProfile(
        median_width=float(np.median([item.bbox[2] for item in seed])),
        median_height=float(np.median([item.bbox[3] for item in seed])),
        median_area=median_area,
        median_fill=float(np.median([item.fill_ratio for item in seed])),
        median_interior_fill=float(np.median([item.interior_fill_ratio for item in seed])),
        median_center_fill=float(np.median([item.center_fill_ratio for item in seed])),
        median_aspect=float(np.median([item.aspect_ratio for item in seed])),
        candidate_count=int(len(candidates)),
        seed_count=int(len(seed)),
        frame_id=str(frame_id or ""),
        frame_path=str(frame_path or ""),
    )


def _is_cell_like_candidate(
    candidate: _ContourCandidate,
    *,
    median_width: float,
    median_height: float,
    median_area: float,
    config: GridDamageAnalysisConfig | None = None,
) -> bool:
    width_ratio = float(candidate.bbox[2]) / max(1.0, float(median_width))
    height_ratio = float(candidate.bbox[3]) / max(1.0, float(median_height))
    area_ratio = float(candidate.area) / max(1.0, float(median_area))
    representation = str(getattr(config, "cell_representation", "confidence") or "confidence")
    if representation == "binary":
        filled = (
            0.52 <= min(width_ratio, height_ratio)
            and max(width_ratio, height_ratio) <= 1.62
            and 0.34 <= area_ratio <= 2.35
            and candidate.fill_ratio >= 0.28
            and candidate.interior_fill_ratio >= 0.20
            and candidate.solidity >= 0.58
        )
        # A mask file can still be an outline drawing. A repeated hollow cell is the grid.
        outline = (
            0.52 <= min(width_ratio, height_ratio)
            and max(width_ratio, height_ratio) <= 1.62
            and 0.15 <= area_ratio <= 2.35
            and candidate.child_count > 0
            and candidate.interior_fill_ratio <= 0.34
        )
        return bool(filled or outline)
    return bool(
        0.56 <= min(width_ratio, height_ratio)
        and max(width_ratio, height_ratio) <= 1.55
        and 0.38 <= area_ratio <= 2.10
        and candidate.child_count > 0
        and candidate.interior_fill_ratio <= 0.34
        and candidate.center_fill_ratio <= 0.20
    )


def _status_for_reasons(_reasons: tuple[str, ...]) -> str:
    """Every detected error is a broken cell, including debris on its own.

    Older results may still carry status ``suspicious`` or ``artifact``. Those
    values stay readable; new analysis does not emit them for debris.
    """

    return "broken"


def _is_ignored_fragment(
    candidate: _ContourCandidate,
    *,
    median_width: float,
    median_height: float,
    median_area: float,
) -> bool:
    width_ratio = float(candidate.bbox[2]) / max(1.0, float(median_width))
    height_ratio = float(candidate.bbox[3]) / max(1.0, float(median_height))
    area_ratio = float(candidate.area) / max(1.0, float(median_area))
    thin_axis = min(width_ratio, height_ratio)
    long_axis = max(width_ratio, height_ratio)
    if thin_axis < 0.42 and long_axis < 1.30:
        return True
    if area_ratio < 0.30 and thin_axis < 0.58:
        return True
    if area_ratio < 0.36 and width_ratio < 0.72 and height_ratio < 0.72:
        return True
    return False


def _cell_feature_vector(
    candidate: _ContourCandidate, seed_medians: tuple[float, float, float, float, float]
) -> np.ndarray:
    median_width, median_height, median_area, median_fill, median_aspect = seed_medians
    width_ratio = float(candidate.bbox[2]) / max(1.0, median_width)
    height_ratio = float(candidate.bbox[3]) / max(1.0, median_height)
    area_ratio = float(candidate.area) / max(1.0, median_area)
    aspect_deviation = abs(float(candidate.aspect_ratio) - float(median_aspect)) / max(0.1, float(median_aspect))
    fill_delta = float(candidate.fill_ratio) - float(median_fill)
    return np.asarray(
        [
            np.log(max(0.05, width_ratio)),
            np.log(max(0.05, height_ratio)),
            np.log(max(0.05, area_ratio)),
            aspect_deviation,
            fill_delta,
            float(candidate.interior_fill_ratio),
            1.0 - min(1.0, max(0.0, float(candidate.solidity))),
            abs(float(candidate.extent) - 0.74),
        ],
        dtype=np.float32,
    )


def _wrong_cell_cluster_ids(candidates: list[_ContourCandidate], seed: list[_ContourCandidate]) -> set[int]:
    if len(candidates) < 4:
        return set()
    medians = (
        float(np.median([item.bbox[2] for item in seed])),
        float(np.median([item.bbox[3] for item in seed])),
        float(np.median([item.area for item in seed])),
        float(np.median([item.fill_ratio for item in seed])),
        float(np.median([item.aspect_ratio for item in seed])),
    )
    matrix = np.vstack([_cell_feature_vector(item, medians) for item in candidates]).astype(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    spans = np.maximum(matrix.std(axis=0), 1e-4)
    normalized = matrix / spans
    normal_center = np.median(normalized, axis=0)
    distances = np.linalg.norm(normalized - normal_center, axis=1)
    wrong_center = normalized[int(np.argmax(distances))].copy()
    labels = np.zeros(len(candidates), dtype=np.int32)
    for _iteration in range(10):
        normal_dist = np.linalg.norm(normalized - normal_center, axis=1)
        wrong_dist = np.linalg.norm(normalized - wrong_center, axis=1)
        next_labels = (wrong_dist < normal_dist).astype(np.int32)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        if np.any(labels == 0):
            normal_center = normalized[labels == 0].mean(axis=0)
        if np.any(labels == 1):
            wrong_center = normalized[labels == 1].mean(axis=0)
    cluster_scores: list[float] = []
    for cluster_id in (0, 1):
        cluster = [candidate for candidate, label in zip(candidates, labels) if int(label) == cluster_id]
        if not cluster:
            cluster_scores.append(float("inf"))
            continue
        size_deviation = float(
            np.mean(
                [
                    abs(item.bbox[2] / max(1.0, medians[0]) - 1.0) + abs(item.bbox[3] / max(1.0, medians[1]) - 1.0)
                    for item in cluster
                ]
            )
        )
        fill = float(np.mean([item.fill_ratio for item in cluster]))
        interior = float(np.mean([item.interior_fill_ratio for item in cluster]))
        cluster_scores.append(size_deviation + max(0.0, fill - medians[3]) + interior * 2.2)
    wrong_label = int(np.argmax(cluster_scores))
    if abs(float(cluster_scores[0]) - float(cluster_scores[1])) < 0.12:
        return set()
    return {int(candidate.contour_id) for candidate, label in zip(candidates, labels) if int(label) == wrong_label}


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = (int(a[0]), int(a[1]), int(a[2]), int(a[3]))
    bx, by, bw, bh = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    horizontal = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    vertical = max(0, max(ay, by) - min(ay + ah, by + bh))
    if horizontal == 0 and vertical == 0:
        return 0.0
    if horizontal == 0:
        return float(vertical)
    if vertical == 0:
        return float(horizontal)
    return float(np.hypot(horizontal, vertical))


def _binary_conductor_zone_map(gray: np.ndarray, threshold: np.ndarray, contours, hierarchy, cfg: GridDamageAnalysisConfig):
    """Zone from a binary mask only. Confidence maps must not call this on themselves."""

    probes = _zone_probe_candidates(contours, hierarchy, gray.shape, cfg)
    hint_area, hint_width, hint_height = _zone_size_hint(probes, contours, threshold)
    if hint_area <= 0.0:
        return _empty_conductor_zone_map()
    pitch_columns, pitch_rows, pitch_x, pitch_y = _lattice_grid_axes(
        probes,
        modal_width=hint_width,
        modal_height=hint_height,
        modal_area=hint_area,
    )
    height, width = gray.shape
    return _conductor_zone_map(
        probes,
        width=int(width),
        height=int(height),
        columns=pitch_columns,
        rows=pitch_rows,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        modal_area=hint_area,
        modal_width=hint_width,
        modal_height=hint_height,
        confidence=None,
        foreground=threshold,
    )


def conductor_zone_map_from_mask_image(
    image: np.ndarray,
    config: GridDamageAnalysisConfig | None = None,
):
    """Build the frame zone from a binary model mask."""

    cfg = replace((config or GridDamageAnalysisConfig()).normalized(), cell_representation="binary")
    gray = _normalize_grayscale(image)
    if gray.size <= 1 or cv2 is None:
        return _empty_conductor_zone_map()
    threshold = _threshold_grid(gray, cfg)
    contours_result = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
    hierarchy = contours_result[1] if len(contours_result) == 2 else contours_result[2]
    return _binary_conductor_zone_map(gray, threshold, contours, hierarchy, cfg)


def _missing_model_file_result(frame_id: str, frame_path: str = "") -> GridFrameAnalysisResult:
    return GridFrameAnalysisResult(
        frame_id=str(frame_id or ""),
        frame_path=str(frame_path or ""),
        image_width=0,
        image_height=0,
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=0,
        detected_cells=0,
        normal_cells=0,
        suspicious_cells=0,
        broken_cells=0,
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.0,
        severity_level="missing_model_file",
        grid_detected=False,
        model_file_status="missing",
    )


def _count_profile_matched_cells(
    candidates: Sequence[_ContourCandidate],
    profile: GridCellReferenceProfile,
) -> int:
    """How many contours look like the run/modal cell size (not crumbs)."""

    width = max(1.0, float(profile.median_width))
    height = max(1.0, float(profile.median_height))
    area = max(1.0, float(profile.median_area))
    matched = 0
    for item in candidates:
        box_w = float(item.bbox[2])
        box_h = float(item.bbox[3])
        if not (0.70 * width <= box_w <= 1.35 * width):
            continue
        if not (0.70 * height <= box_h <= 1.35 * height):
            continue
        if not (0.55 * area <= float(item.area) <= 1.80 * area):
            continue
        if float(item.solidity) < 0.70:
            continue
        matched += 1
    return int(matched)


def _profile_looks_like_cell_array(profile: GridCellReferenceProfile | None) -> bool:
    if profile is None:
        return False
    # Real SEM cells are ~20×26 (area≈500). Synthetic/unit lattices may be ~12×12.
    # Reject only crumb-scale norms (a few pixels).
    return (
        float(profile.median_area) >= 80.0
        and float(profile.median_width) >= 10.0
        and float(profile.median_height) >= 10.0
    )


def _frame_has_reliable_cell_array(
    candidates: Sequence[_ContourCandidate],
    profile: GridCellReferenceProfile | None,
    *,
    min_matched: int = 24,
) -> bool:
    """Reject crumb-only frames (e.g. no array) that would invent a tiny cell norm."""

    if not _profile_looks_like_cell_array(profile):
        return False
    matched = _count_profile_matched_cells(candidates, profile)
    if matched < max(8, int(min_matched)):
        return False
    # No-array frames still throw a few coincidental cell-sized blobs among thousands of crumbs.
    if len(candidates) >= max(120, 4 * matched) and matched < 80:
        return False
    return True


def _no_cell_array_result(
    *,
    frame_id: str,
    frame_path: str,
    width: int,
    height: int,
    component_count: int = 0,
    zone_map=None,
    zone_skipped: int = 0,
    cell_width: float = 0.0,
    cell_height: float = 0.0,
    config: GridDamageAnalysisConfig | None = None,
) -> GridFrameAnalysisResult:
    """Frame without a reliable cell lattice: zero cell defects, optional conductor zones."""

    cfg = (config or GridDamageAnalysisConfig()).normalized()
    zones: list[GridCellAnalysisResult] = []
    enabled = set(cfg.enabled_reason_types or GRID_DAMAGE_REASON_TYPES)
    if zone_map is not None and getattr(zone_map, "zones", ()) and "conductor_zone" in enabled:
        zones = [_zone_cell_record(zone, row=index, contour_id=index + 1) for index, zone in enumerate(zone_map.zones)]
    return GridFrameAnalysisResult(
        frame_id=str(frame_id or ""),
        frame_path=str(frame_path or ""),
        image_width=int(width),
        image_height=int(height),
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=0,
        detected_cells=0,
        normal_cells=0,
        suspicious_cells=0,
        broken_cells=0,
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.0,
        severity_level="no_cell_array",
        grid_detected=False,
        per_cell_results=tuple(zones),
        component_count=int(component_count),
        cell_width=int(round(float(cell_width or 0.0))),
        cell_height=int(round(float(cell_height or 0.0))),
        zone_skipped_components=int(zone_skipped),
        model_file_status="no_cell_array",
    )


def _conductor_zone_map(
    candidates: list[_ContourCandidate],
    *,
    width: int,
    height: int,
    columns: list[float],
    rows: list[float],
    pitch_x: float | None,
    pitch_y: float | None,
    modal_area: float,
    modal_width: float,
    modal_height: float,
    confidence: np.ndarray | None,
    foreground: np.ndarray | None,
):
    from .grid_zones import find_conductor_zones

    return find_conductor_zones(
        width=width,
        height=height,
        candidates=candidates,
        columns=columns,
        rows=rows,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        modal_area=modal_area,
        modal_width=modal_width,
        modal_height=modal_height,
        confidence=confidence,
        foreground=foreground,
    )


class _TemplateView:
    """Layer template applied to the pieces of one frame (calibrated scoring)."""

    def __init__(self, template: CellShapeTemplate, config: GridDamageAnalysisConfig) -> None:
        self.template = template
        self.matcher = CellTemplateMatcher(template)
        self.deviation_limit, self.width_tolerance, self.height_tolerance, self.depth_limit = geometry_thresholds(
            template, int(config.geometry_sensitivity)
        )
        self._measure_by_id: dict[int, tuple[float, int]] = {}

    def ratios(self, candidate: _ContourCandidate) -> tuple[float, float, float]:
        t = self.template
        return (
            float(candidate.bbox[2]) / max(1.0, float(t.cell_width)),
            float(candidate.bbox[3]) / max(1.0, float(t.cell_height)),
            float(candidate.area) / max(1.0, float(t.cell_area)),
        )

    def is_cell_sized(self, candidate: _ContourCandidate) -> bool:
        width_ratio, height_ratio, area_ratio = self.ratios(candidate)
        return bool(
            area_ratio >= CELL_MIN_AREA_RATIO
            and min(width_ratio, height_ratio) >= 0.50
            and max(width_ratio, height_ratio) <= 1.85
            and area_ratio <= 2.40
        )

    def measure(self, candidate: _ContourCandidate) -> tuple[float, int]:
        key = int(candidate.contour_id)
        cached = self._measure_by_id.get(key)
        if cached is None:
            cached = (
                self.matcher.measure_points(candidate.contour, self.depth_limit)
                if candidate.contour is not None
                else (1.0, 10**6)
            )
            self._measure_by_id[key] = cached
        return cached

    def deviation(self, candidate: _ContourCandidate) -> float:
        return self.measure(candidate)[0]

    def is_cells_in_a_row(self, candidate: _ContourCandidate) -> bool:
        """One cell wide, 1.6-4 cells long, with the area of 1.5-3.8 cells and a solid body."""

        width_ratio, height_ratio, area_ratio = self.ratios(candidate)
        short, long = sorted((width_ratio, height_ratio))
        return bool(
            0.75 <= short <= 1.40
            and 1.60 <= long <= 4.20
            and 1.50 <= area_ratio <= 3.80
            and float(candidate.solidity) >= 0.80
        )

    def shape_is_off(self, candidate: _ContourCandidate) -> bool:
        deviation, deep = self.measure(candidate)
        return deviation > self.deviation_limit or deep >= DEEP_PIXELS_MIN

    def matches(self, candidate: _ContourCandidate) -> bool:
        width_ratio, height_ratio, area_ratio = self.ratios(candidate)
        if candidate.is_hole or candidate.touches_border or not 0.75 <= area_ratio <= 1.40:
            return False
        if abs(width_ratio - 1.0) > self.width_tolerance or abs(height_ratio - 1.0) > self.height_tolerance:
            return False
        return not self.shape_is_off(candidate)

    def count_cells(self, candidates: Sequence[_ContourCandidate]) -> int:
        """Pieces that are cells, broken ones included; a lone broken cell still makes a frame to check."""

        return sum(1 for candidate in candidates if not candidate.touches_border and self.looks_like_cell(candidate))

    def judge(
        self,
        candidate: _ContourCandidate,
        reasons: tuple[str, ...],
        score: float,
        *,
        hole_broken: bool,
    ) -> tuple[float, tuple[str, ...], tuple[tuple[str, float], ...]]:
        """Geometry against the template; size-only and shape-only scores give way to it."""

        if candidate.touches_border or "merged_contour" in reasons or "edge_clipped_cell" in reasons:
            return score, reasons, ()
        width_ratio, height_ratio, area_ratio = self.ratios(candidate)
        if self.is_cells_in_a_row(candidate):
            # Two or more cells glued along a row or column, at any spacing (also off the
            # lattice pitch, where the core check does not look).
            reasons = tuple(dict.fromkeys((*(r for r in reasons if r not in {"small_artifact", "broken_geometry"}), "merged_contour")))
            return max(float(score), 0.86), reasons, (("template_cells_in_row", float(max(width_ratio, height_ratio))),)
        if not self.is_cell_sized(candidate):
            # A piece well below a cell is debris (the leftover rule marks it), not a broken cell.
            if area_ratio < CELL_MIN_AREA_RATIO and not hole_broken:
                reasons = tuple(reason for reason in reasons if reason != "broken_geometry")
            return (score if reasons else 0.0), reasons, ()
        deviation, deep = self.measure(candidate)
        features = (
            ("template_deviation", float(deviation)),
            ("template_deep_pixels", float(deep)),
            ("template_width_ratio", float(width_ratio)),
            ("template_height_ratio", float(height_ratio)),
        )
        # Solidity/extent geometry reacts to rounder or sharper edges; the template does not.
        reasons = tuple(reason for reason in reasons if reason != "broken_geometry" or hole_broken)
        off_shape = deviation > self.deviation_limit or deep >= DEEP_PIXELS_MIN
        off_size = abs(width_ratio - 1.0) > self.width_tolerance or abs(height_ratio - 1.0) > self.height_tolerance
        if off_shape or off_size:
            reasons = tuple(dict.fromkeys((*(reason for reason in reasons if reason != "small_artifact"), "broken_geometry")))
            score = max(float(score), 0.80)
        elif not reasons:
            score = 0.0
        return score, reasons, features

    def looks_like_cell(self, candidate: _ContourCandidate) -> bool:
        """Cell-sized and loosely the template shape: a cell, perhaps a broken one."""

        return (
            not candidate.is_hole
            and self.is_cell_sized(candidate)
            and self.deviation(candidate) <= FRAGMENT_UNION_MAX_DEVIATION
        )

    def drop_cell_zones(self, zone_map, candidates: Sequence[_ContourCandidate], foreground: np.ndarray | None = None):
        zones = tuple(getattr(zone_map, "zones", ()) or ())
        if not zones:
            return zone_map
        kept = []
        dropped = []
        for zone in zones:
            x, y, w, h = (int(value) for value in zone.bbox[:4])
            inside = [
                item
                for item in candidates
                if not item.is_hole
                and not item.touches_border
                and float(item.area) >= DEBRIS_MIN_AREA_PX
                and x <= float(item.centroid[0]) <= x + w
                and y <= float(item.centroid[1]) <= y + h
            ]
            # Only pieces that match the template outright count: conductor grain has
            # plenty of cell-sized, roughly square blobs, clean cells of a sparse field do not differ.
            cells = sum(1 for item in inside if self.matches(item))
            # A lone cell, broken or not, is never grain: grain is many pieces.
            lone_cells = 0 < len(inside) <= 3 and all(self.looks_like_cell(item) for item in inside)
            # Conductor grain fills its zone densely; a sparse field with a few cells or
            # broken pieces does not, and what is in it must be checked.
            sparse = False
            if foreground is not None and w > 0 and h > 0:
                sparse = float(np.count_nonzero(foreground[y : y + h, x : x + w])) / float(w * h) < ZONE_MIN_DENSITY
            (dropped if (cells and cells * 2 >= len(inside)) or lone_cells or sparse else kept).append(zone)
        if not dropped:
            return zone_map
        occupied = np.array(zone_map.occupied, copy=True)
        tile = int(getattr(zone_map, "tile", 0) or 0)
        if tile > 0 and occupied.size:
            for zone in dropped:
                x, y, w, h = (int(value) for value in zone.bbox[:4])
                occupied[max(0, y // tile) : (y + h) // tile + 1, max(0, x // tile) : (x + w) // tile + 1] = False
        return replace(zone_map, zones=tuple(kept), occupied=occupied)

    def union_looks_like_cell(self, outlines: Sequence[Any]) -> bool:
        points = [np.asarray(outline, dtype=np.int32).reshape(-1, 2) for outline in outlines if outline is not None and len(outline) >= 3]
        if not points or cv2 is None:
            return False
        stacked = np.concatenate(points)
        x0, y0 = stacked.min(axis=0)
        x1, y1 = stacked.max(axis=0)
        crop = np.zeros((int(y1 - y0 + 1), int(x1 - x0 + 1)), dtype=np.uint8)
        for item in points:
            cv2.fillPoly(crop, [(item - (x0, y0)).reshape(-1, 1, 2)], 1)
        return self.matcher.deviation_of_crop(crop) <= FRAGMENT_UNION_MAX_DEVIATION


def _group_slot_fragments(
    per_cell: list[GridCellAnalysisResult],
    *,
    median_width: float,
    median_height: float,
    columns: list[float],
    rows: list[float],
    pitch_x: float | None,
    pitch_y: float | None,
    config: GridDamageAnalysisConfig,
    template_view: "_TemplateView | None" = None,
) -> list[GridCellAnalysisResult]:
    """One broken cell for defect pieces that together fill one empty lattice slot.

    A cell the network drew as a split 'П' comes out as a bar and legs. Scored
    one by one they read as debris inside the cell; together they are the cell.
    """

    enabled = config.enabled_reason_types
    if enabled is not None and "broken_geometry" not in set(enabled):
        return per_cell
    if not columns or not rows:
        return per_cell
    cell_w = max(1.0, float(median_width))
    cell_h = max(1.0, float(median_height))
    fragment_reasons = {"small_artifact", "broken_geometry"}
    fragments = [
        index
        for index, cell in enumerate(per_cell)
        if cell.reasons
        and set(cell.reasons) <= fragment_reasons
        and cell.bbox[2] <= 1.35 * cell_w
        and cell.bbox[3] <= 1.35 * cell_h
    ]
    if len(fragments) < 2:
        return per_cell
    gap_limit = max(2.0, 0.35 * min(cell_w, cell_h))
    parent = {index: index for index in fragments}
    box = {index: tuple(int(value) for value in per_cell[index].bbox[:4]) for index in fragments}
    members = {index: [index] for index in fragments}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def bucket(index: int) -> tuple[int, int]:
        x, y, w, h = per_cell[index].bbox[:4]
        return int((x + w / 2.0) // cell_w), int((y + h / 2.0) // cell_h)

    buckets: dict[tuple[int, int], list[int]] = {}
    for index in fragments:
        buckets.setdefault(bucket(index), []).append(index)
    for index in fragments:
        bx, by = bucket(index)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in buckets.get((bx + dx, by + dy), ()):
                    left, right = find(index), find(other)
                    if left == right or _bbox_gap(box[left], box[right]) > gap_limit:
                        continue
                    ax, ay, aw, ah = box[left]
                    ox, oy, ow, oh = box[right]
                    x0, y0 = min(ax, ox), min(ay, oy)
                    x1, y1 = max(ax + aw, ox + ow), max(ay + ah, oy + oh)
                    if x1 - x0 > 1.35 * cell_w or y1 - y0 > 1.35 * cell_h:
                        continue
                    parent[right] = left
                    box[left] = (x0, y0, x1 - x0, y1 - y0)
                    members[left].extend(members.pop(right))

    grouped: set[int] = set()
    merged_cells: list[GridCellAnalysisResult] = []
    next_contour_id = max((int(cell.contour_id or 0) for cell in per_cell), default=0) + 1
    for root, group in members.items():
        if len(group) < 2:
            continue
        x, y, w, h = box[root]
        if w < 0.6 * cell_w or h < 0.6 * cell_h:
            continue
        cx, cy = x + w / 2.0, y + h / 2.0
        if not (
            _on_grid_axis(cx, columns, pitch_x, 0.45 * cell_w) and _on_grid_axis(cy, rows, pitch_y, 0.45 * cell_h)
        ):
            continue
        member_set = set(group)
        if any(
            index not in member_set and x <= float(cell.centroid[0]) <= x + w and y <= float(cell.centroid[1]) <= y + h
            for index, cell in enumerate(per_cell)
        ):
            continue
        # Pieces of a cell together still look like the cell; neighbouring grain does not.
        if template_view is not None and not template_view.union_looks_like_cell([per_cell[index].outline for index in group]):
            continue
        points = [point for index in group for point in (per_cell[index].outline or ())]
        outline: tuple[tuple[int, int], ...] = ()
        if len(points) >= 3 and cv2 is not None:
            hull = cv2.convexHull(np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)).reshape(-1, 2)
            outline = tuple((int(px), int(py)) for px, py in hull)
        merged_cells.append(
            GridCellAnalysisResult(
                row=0,
                col=0,
                bbox=(int(x), int(y), int(w), int(h)),
                centroid=(float(cx), float(cy)),
                contour_id=int(next_contour_id),
                status="broken",
                score=float(max(0.80, max(float(per_cell[index].score) for index in group))),
                reasons=("broken_geometry",),
                feature_snapshot=(("fragment_count", float(len(group))),),
                outline=outline,
            )
        )
        next_contour_id += 1
        grouped.update(member_set)
    if not grouped:
        return per_cell
    kept = [cell for index, cell in enumerate(per_cell) if index not in grouped]
    kept.extend(merged_cells)
    return kept


def _collapse_conductor_regions(
    per_cell: list[GridCellAnalysisResult],
    *,
    median_width: float,
    median_height: float,
) -> list[GridCellAnalysisResult]:
    """Merge adjacent core-based merged_contour hits into one rectangle per blob.

    Conductor residue keeps its own reason — folding it into merged_contour made
    false merges on grain / no-array frames.
    """

    conductor_indexes = [
        index
        for index, cell in enumerate(per_cell)
        if "merged_contour" in tuple(cell.reasons or ())
    ]
    if len(conductor_indexes) <= 1:
        return per_cell

    gap_limit = max(6.0, 0.85 * max(float(median_width), float(median_height)))
    parent = {index: index for index in conductor_indexes}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def unite(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for position, left in enumerate(conductor_indexes):
        left_box = tuple(int(value) for value in per_cell[left].bbox[:4])
        for right in conductor_indexes[position + 1 :]:
            right_box = tuple(int(value) for value in per_cell[right].bbox[:4])
            if _bbox_gap(left_box, right_box) <= gap_limit:
                unite(left, right)

    groups: dict[int, list[int]] = {}
    for index in conductor_indexes:
        groups.setdefault(find(index), []).append(index)

    drop_indexes: set[int] = set()
    replacements: list[GridCellAnalysisResult] = []
    next_contour_id = max((int(cell.contour_id) for cell in per_cell), default=0) + 1
    for members in groups.values():
        if len(members) <= 1:
            continue
        boxes = [tuple(int(value) for value in per_cell[index].bbox[:4]) for index in members]
        left = min(box[0] for box in boxes)
        top = min(box[1] for box in boxes)
        right = max(box[0] + box[2] for box in boxes)
        bottom = max(box[1] + box[3] for box in boxes)
        width = max(1, right - left)
        height = max(1, bottom - top)
        score = max(float(per_cell[index].score) for index in members)
        replacements.append(
            GridCellAnalysisResult(
                row=0,
                col=0,
                bbox=(int(left), int(top), int(width), int(height)),
                centroid=(float(left) + 0.5 * float(width), float(top) + 0.5 * float(height)),
                contour_id=int(next_contour_id),
                status="broken",
                score=float(min(1.0, max(0.86, score))),
                reasons=("merged_contour",),
            )
        )
        next_contour_id += 1
        drop_indexes.update(members)

    if not replacements:
        return per_cell
    kept = [cell for index, cell in enumerate(per_cell) if index not in drop_indexes]
    kept.extend(replacements)
    return [
        replace(cell, row=int(index), col=0)
        for index, cell in enumerate(sorted(kept, key=lambda item: (float(item.centroid[1]), float(item.centroid[0]))))
    ]


def _analyze_conductor_only_frame(
    candidates: list[_ContourCandidate],
    *,
    frame_id: str,
    frame_path: str,
    width: int,
    height: int,
    config: GridDamageAnalysisConfig,
    started: float,
    zone_map=None,
    zone_skipped: int = 0,
    zone_supplied: bool = False,
) -> GridFrameAnalysisResult:
    """One conductor zone per off-grid mass when the frame has no cell lattice."""

    empty = GridFrameAnalysisResult(
        frame_id=str(frame_id or ""),
        frame_path=str(frame_path or ""),
        image_width=int(width),
        image_height=int(height),
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=0,
        detected_cells=0,
        normal_cells=0,
        suspicious_cells=0,
        broken_cells=0,
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.0,
        severity_level="OK",
        grid_detected=False,
        per_cell_results=(),
        component_count=int(len(candidates)),
        zone_skipped_components=int(zone_skipped),
    )
    if not zone_supplied and (zone_map is None or not getattr(zone_map, "zones", ())):
        zone_map = _conductor_zone_map(
            candidates,
            width=int(width),
            height=int(height),
            columns=[],
            rows=[],
            pitch_x=None,
            pitch_y=None,
            modal_area=0.0,
            modal_width=0.0,
            modal_height=0.0,
            confidence=None,
            foreground=None,
        )
    enabled = set(config.enabled_reason_types or GRID_DAMAGE_REASON_TYPES)
    if not zone_map.zones or "conductor_zone" not in enabled:
        return empty
    zones = [_zone_cell_record(zone, row=index, contour_id=index + 1) for index, zone in enumerate(zone_map.zones)]
    _ = started
    return GridFrameAnalysisResult(
        frame_id=str(frame_id or ""),
        frame_path=str(frame_path or ""),
        image_width=int(width),
        image_height=int(height),
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=0,
        detected_cells=0,
        normal_cells=0,
        suspicious_cells=0,
        broken_cells=0,
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.0,
        severity_level="OK",
        grid_detected=False,
        per_cell_results=tuple(zones),
        component_count=int(len(candidates)),
        zone_skipped_components=int(zone_skipped),
    )


def _border_merged_cell_signal(
    candidate: _ContourCandidate,
    *,
    median_width: float,
    median_height: float,
    median_area: float,
) -> bool:
    """Recognize two grid cells joined into one contour at a frame edge."""

    if not bool(candidate.touches_border):
        return False
    width_ratio = float(candidate.bbox[2]) / max(1.0, float(median_width))
    height_ratio = float(candidate.bbox[3]) / max(1.0, float(median_height))
    area_ratio = float(candidate.area) / max(1.0, float(median_area))
    bbox_area_ratio = float(candidate.bbox_area) / max(1.0, float(median_width) * float(median_height))
    largest_axis = max(width_ratio, height_ratio)
    smallest_axis = min(width_ratio, height_ratio)
    two_slot_geometry = (
        1.55 <= largest_axis <= 2.70
        and 0.45 <= smallest_axis <= 1.40
        and 1.05 <= bbox_area_ratio <= 3.25
        and 0.20 <= area_ratio <= 3.00
    )
    multiple_chambers = int(candidate.child_count) >= 2
    clipped_outline = (
        float(candidate.fill_ratio) <= 0.54
        and float(candidate.interior_fill_ratio) <= 0.32
        and float(candidate.center_fill_ratio) <= 0.26
        and float(candidate.outline_mean_side_coverage) >= 0.42
        and (
            float(candidate.outline_side_imbalance) >= 0.12
            or float(candidate.outline_min_side_coverage) <= 0.68
            or int(candidate.approx_vertices) >= 8
        )
    )
    return bool(two_slot_geometry and (multiple_chambers or clipped_outline))


def _local_confidence_stats(
    confidence_map: np.ndarray | None,
    bbox: tuple[int, int, int, int],
) -> tuple[float, float] | None:
    """Return (mean, min) probability inside a candidate bbox, or None when unavailable."""

    if confidence_map is None:
        return None
    values = np.asarray(confidence_map, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        return None
    x, y, width, height = (int(value) for value in bbox[:4])
    if width <= 0 or height <= 0:
        return None
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(int(values.shape[1]), x + width)
    y1 = min(int(values.shape[0]), y + height)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = values[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    return float(np.mean(roi, dtype=np.float64)), float(np.min(roi))


def _cell_model_uncertainty(
    confidence_map: np.ndarray | None,
    bbox: tuple[int, int, int, int],
) -> tuple[float, float, float] | None:
    """Mean confidence, fraction of uncertain pixels, and the same fraction on the border band."""

    if confidence_map is None:
        return None
    values = np.asarray(confidence_map, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        return None
    x, y, width, height = (int(value) for value in bbox[:4])
    if width <= 0 or height <= 0:
        return None
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(int(values.shape[1]), x + width)
    y1 = min(int(values.shape[0]), y + height)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = values[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    if float(np.max(roi)) > 1.5:
        roi = roi / 255.0
    uncertain = (roi >= 0.35) & (roi <= 0.65)
    band = max(1, int(round(min(roi.shape) * 0.18)))
    border = np.concatenate(
        (
            roi[:band, :].reshape(-1),
            roi[-band:, :].reshape(-1),
            roi[:, :band].reshape(-1),
            roi[:, -band:].reshape(-1),
        )
    )
    border_uncertain = (border >= 0.35) & (border <= 0.65)
    return (
        float(np.mean(roi, dtype=np.float64)),
        float(np.mean(uncertain)),
        float(np.mean(border_uncertain)),
    )


def _apply_confidence_geometry_boost(
    candidate: _ContourCandidate,
    score: float,
    reasons: list[str],
    *,
    median_width: float,
    median_height: float,
    median_area: float,
    confidence_map: np.ndarray | None,
) -> tuple[float, list[str]]:
    """Promote broken geometry / debris when the local confidence map is uncertain."""

    stats = _local_confidence_stats(confidence_map, candidate.bbox)
    if stats is None:
        return score, reasons
    mean_prob, min_prob = stats
    width_ratio = float(candidate.bbox[2]) / max(1.0, float(median_width))
    height_ratio = float(candidate.bbox[3]) / max(1.0, float(median_height))
    area_ratio = float(candidate.area) / max(1.0, float(median_area))
    smallest = min(width_ratio, height_ratio)
    largest = max(width_ratio, height_ratio)
    slot_like = 0.22 <= smallest and largest <= 1.85 and 0.10 <= area_ratio <= 2.40
    if not slot_like:
        return score, reasons
    # The repeated cell is the norm, even when the drawing is an outline.
    if 0.85 <= width_ratio <= 1.15 and 0.85 <= height_ratio <= 1.15 and 0.75 <= area_ratio <= 1.25:
        return score, reasons
    irregular = (
        float(candidate.solidity) <= 0.90
        or abs(float(candidate.extent) - 0.74) >= 0.10
        or int(candidate.approx_vertices) >= 8
        or float(candidate.outline_min_side_coverage) <= 0.58
        or float(candidate.outline_side_imbalance) >= 0.18
    )
    uncertain = mean_prob <= 0.42 or min_prob <= 0.18
    strongly_uncertain = mean_prob <= 0.30 or min_prob <= 0.10
    if not uncertain:
        return score, reasons
    if irregular or strongly_uncertain:
        next_reasons = list(reasons)
        if "edge_clipped_cell" not in next_reasons:
            if "broken_geometry" not in next_reasons:
                next_reasons.append("broken_geometry")
            score = max(score, min(0.94, 0.78 + 0.30 * max(0.0, 0.45 - mean_prob)))
        if strongly_uncertain and float(candidate.area) <= max(48.0, 0.55 * float(median_area)):
            if "small_artifact" not in next_reasons and "merged_contour" not in next_reasons:
                next_reasons.append("small_artifact")
                score = max(score, 0.74)
        return score, next_reasons
    return score, reasons


def _classify_binary_cell(
    candidate: _ContourCandidate,
    *,
    median_width: float,
    median_height: float,
    median_area: float,
    median_fill: float,
    median_interior_fill: float,
    median_aspect: float,
    config: GridDamageAnalysisConfig,
    confidence_map: np.ndarray | None = None,
) -> tuple[float, tuple[str, ...]]:
    width_ratio = float(candidate.bbox[2]) / max(1.0, float(median_width))
    height_ratio = float(candidate.bbox[3]) / max(1.0, float(median_height))
    area_ratio = float(candidate.area) / max(1.0, float(median_area))
    smallest_axis = min(width_ratio, height_ratio)
    largest_axis = max(width_ratio, height_ratio)
    reasons: list[str] = []
    score = 0.0

    border_merged = _border_merged_cell_signal(
        candidate,
        median_width=median_width,
        median_height=median_height,
        median_area=median_area,
    )
    merged = (
        border_merged
        or (largest_axis > float(config.merged_size_ratio) and area_ratio > float(config.merged_area_ratio))
        or (largest_axis >= 1.55 and smallest_axis >= 0.60 and area_ratio >= 1.45)
    )
    if merged:
        reasons.append("merged_contour")
        score = max(score, min(1.0, 0.82 + 0.12 * max(0.0, largest_axis - 1.35)))

    # Allow slightly undersized / elongated slot occupants so wrong geometry is
    # still scored when the same-frame seed is built without a reference frame.
    slot_sized = 0.28 <= smallest_axis and largest_axis <= 1.85 and 0.12 <= area_ratio <= 2.40
    interior = float(candidate.interior_fill_ratio)
    center = float(candidate.center_fill_ratio)
    fill_delta = float(config.filled_ratio_delta)
    fill_absolute = float(config.filled_ratio_absolute)
    normal_interior = max(0.0, float(median_interior_fill))
    # A cell is filled only when its interior is clearly denser than the
    # normal cells on this same mask. Similar neighbors stay unmarked.
    fill_gap = max(0.14, fill_delta)
    filled_floor = max(fill_absolute, normal_interior + fill_gap, 0.62)
    partial_floor = max(0.48, normal_interior + max(0.18, fill_delta), fill_absolute * 0.9)
    if slot_sized and interior >= filled_floor and center >= filled_floor - 0.08 and interior >= normal_interior + fill_gap:
        reasons.append("filled_cell")
        score = max(score, min(1.0, 0.80 + (interior - filled_floor)))
    elif (
        slot_sized
        and "merged_contour" not in reasons
        and interior >= partial_floor
        and center >= partial_floor - 0.10
        and interior >= normal_interior + max(0.16, fill_delta)
    ):
        reasons.append("partial_filled_cell")
        score = max(score, min(0.90, 0.70 + (interior - partial_floor)))
    fill_loss = max(
        0.0,
        max(0.34, float(median_fill) * 0.64) - float(candidate.fill_ratio),
        max(0.28, float(median_interior_fill) * 0.58) - float(candidate.interior_fill_ratio),
    )
    aspect = float(candidate.aspect_ratio)
    aspect_ref = max(0.1, float(median_aspect) if float(median_aspect) > 0.0 else 1.0)
    aspect_mismatch = abs(aspect - aspect_ref) >= max(0.18, aspect_ref * 0.20)
    extent_delta = abs(float(candidate.extent) - 0.74)
    solidity_limit = float(config.geometry_solidity_limit)
    # Higher slider raises the limit and admits milder shape faults. A plain
    # outline with a few extra corners is not broken geometry.
    shape_broken = float(candidate.solidity) < solidity_limit and (
        extent_delta >= max(0.08, 0.34 - 0.28 * min(1.0, max(0.0, (solidity_limit - 0.50) / 0.40)))
        or int(candidate.approx_vertices) >= (8 if solidity_limit >= 0.82 else 12)
        or fill_loss > (0.10 if solidity_limit >= 0.82 else 0.22)
    )
    severely_bitten = float(candidate.solidity) <= min(0.62, solidity_limit - 0.04) and extent_delta >= 0.16
    broken = slot_sized and (shape_broken or severely_bitten) and not (
        float(candidate.solidity) >= max(0.84, solidity_limit)
        and extent_delta < 0.12
        and int(candidate.approx_vertices) <= 10
    )
    flat_slot = (
        slot_sized
        and smallest_axis <= 0.55
        and largest_axis >= 0.75
        and float(area_ratio) <= 0.70
        and float(candidate.solidity) >= 0.85
    )
    if broken or flat_slot:
        reasons.append("broken_geometry")
        score = max(
            score,
            min(
                0.96,
                0.74
                + 0.55 * fill_loss
                + 0.35 * max(0.0, 0.82 - float(candidate.solidity))
                + 0.015 * max(0, int(candidate.approx_vertices) - 8)
                + (0.12 if aspect_mismatch else 0.0),
            ),
        )

    small_artifact = _small_artifact_signal(
        candidate,
        median_width=median_width,
        median_height=median_height,
        median_area=median_area,
        median_fill=median_fill,
        median_interior_fill=median_interior_fill,
        median_center_fill=0.0,
        config=config,
    )
    if small_artifact:
        reasons.append("small_artifact")
        score = max(score, 0.76)
        # Mid dirt must stay in the debris group, not double-count as geometry.
        reasons = [reason for reason in reasons if reason != "broken_geometry"]

    edge_clipped = bool(candidate.touches_border) and (
        (smallest_axis < 0.82 and largest_axis <= 1.70 and area_ratio <= 1.25)
        or (
            smallest_axis < 0.90
            and largest_axis <= 1.55
            and aspect_mismatch
            and area_ratio <= 1.05
        )
    )
    if edge_clipped:
        reasons.append("edge_clipped_cell")
        score = max(score, min(0.96, 0.76 + 0.18 * max(0.0, 1.0 - smallest_axis)))
        # Border crop explains the warped silhouette — keep the dedicated group.
        reasons = [reason for reason in reasons if reason != "broken_geometry"]
    score, reasons = _apply_confidence_geometry_boost(
        candidate,
        score,
        reasons,
        median_width=median_width,
        median_height=median_height,
        median_area=median_area,
        confidence_map=confidence_map,
    )
    if not reasons:
        return 0.0, ()
    return float(np.clip(score, 0.0, 1.0)), tuple(dict.fromkeys(reasons))


def _classify_detected_cell(
    candidate: _ContourCandidate,
    *,
    median_width: float,
    median_height: float,
    median_area: float,
    median_fill: float,
    median_interior_fill: float,
    median_center_fill: float,
    median_aspect: float,
    config: GridDamageAnalysisConfig,
    confidence_map: np.ndarray | None = None,
) -> tuple[float, tuple[str, ...]]:
    score = 0.0
    reasons: list[str] = []
    has_hole_signal = candidate.child_count > 0 and candidate.inner_hole_ratio >= 0.02
    width_ratio = candidate.bbox[2] / max(1.0, median_width)
    height_ratio = candidate.bbox[3] / max(1.0, median_height)
    area_ratio = candidate.area / max(1.0, median_area)
    bbox_area_ratio = candidate.bbox_area / max(1.0, float(median_width) * float(median_height))
    aspect_ratio = float(candidate.aspect_ratio)
    smallest_axis_ratio = min(float(width_ratio), float(height_ratio))
    largest_axis_ratio = max(float(width_ratio), float(height_ratio))
    if str(config.cell_representation) == "binary":
        return _classify_binary_cell(
            candidate,
            median_width=median_width,
            median_height=median_height,
            median_area=median_area,
            median_fill=median_fill,
            median_interior_fill=median_interior_fill,
            median_aspect=median_aspect,
            config=config,
            confidence_map=confidence_map,
        )
    fill_limit = max(
        float(config.filled_ratio_absolute),
        min(0.86, float(median_fill) + max(0.10, float(config.filled_ratio_delta))),
    )
    interior_limit = max(0.34, min(0.72, fill_limit * 0.64))
    center_fill = float(candidate.center_fill_ratio)
    center_limit = max(0.22, min(0.58, interior_limit * 0.72))
    normal_interior = max(0.0, float(median_interior_fill))
    normal_center = max(0.0, float(median_center_fill))
    fill_margin = max(0.035, min(0.12, float(config.filled_ratio_delta) * 0.35))
    interior_margin = max(0.045, min(0.12, float(config.filled_ratio_delta) * 0.45))
    center_margin = max(0.035, min(0.10, float(config.filled_ratio_delta) * 0.35))
    strict_fill_mode = float(config.filled_ratio_delta) <= 0.08
    strict_merge_mode = float(config.merged_size_ratio) <= 1.32 or float(config.merged_area_ratio) <= 1.32
    off_center_fill_signal = (
        candidate.interior_fill_ratio >= max(0.16, normal_interior + interior_margin)
        and candidate.fill_ratio >= max(float(median_fill) + fill_margin, 0.30)
        and float(area_ratio) >= 0.32
        and smallest_axis_ratio >= 0.54
        and largest_axis_ratio <= 1.62
    )
    edge_partial_fill_signal = (
        (smallest_axis_ratio >= 0.54 and largest_axis_ratio <= 1.64 and 0.30 <= float(area_ratio) <= 2.20)
        and (
            candidate.interior_fill_ratio >= max(0.10, normal_interior + interior_margin * 0.72)
            or center_fill >= max(0.08, normal_center + center_margin)
        )
        and (
            candidate.fill_ratio >= max(float(median_fill) + fill_margin * 0.72, 0.28)
            or candidate.extent >= 0.42
            or candidate.solidity <= 0.84
            or abs(float(candidate.extent) - 0.74) >= 0.18
        )
        and (
            candidate.outline_side_imbalance >= 0.18
            or candidate.outline_min_side_coverage <= 0.58
            or candidate.approx_vertices >= 8
            or candidate.child_count == 0
        )
    )
    filled_by_center = (
        center_fill >= center_limit
        and candidate.interior_fill_ratio >= max(0.24, interior_limit * 0.78)
        and candidate.fill_ratio >= max(float(median_fill) + 0.10, fill_limit * 0.62)
    )
    centered_partial_fill_signal = (
        center_fill >= max(0.11, normal_center + center_margin)
        and candidate.interior_fill_ratio >= max(0.14, normal_interior + interior_margin)
        and candidate.fill_ratio >= max(float(median_fill) + fill_margin, 0.24)
        and float(area_ratio) >= 0.32
        and smallest_axis_ratio >= 0.56
        and largest_axis_ratio <= 1.58
        and not _is_cell_like_candidate(
            candidate,
            median_width=median_width,
            median_height=median_height,
            median_area=median_area,
            config=config,
        )
    )
    partial_fill_signal = centered_partial_fill_signal or off_center_fill_signal or edge_partial_fill_signal
    slot_sized_geometry = (
        0.38 <= smallest_axis_ratio and largest_axis_ratio <= 1.65 and 0.22 <= float(area_ratio) <= 2.20
    )
    outline_min = float(candidate.outline_min_side_coverage)
    outline_mean = float(candidate.outline_mean_side_coverage)
    outline_imbalance = float(candidate.outline_side_imbalance)
    low_fill_cell = (
        candidate.interior_fill_ratio <= max(0.30, float(median_fill) + 0.12)
        and center_fill <= max(0.20, float(median_fill) + 0.08)
        and candidate.fill_ratio >= max(float(median_fill) * 0.70, 0.08)
    )
    faint_inner_fill_signal = candidate.interior_fill_ratio >= max(
        0.095, normal_interior + max(0.030, interior_margin * 0.62)
    ) or center_fill >= max(0.065, normal_center + max(0.022, center_margin * 0.55))
    faint_shape_evidence = (
        outline_imbalance >= 0.16
        or outline_min <= 0.58
        or candidate.solidity <= 0.88
        or abs(float(candidate.extent) - 0.74) >= 0.11
        or candidate.approx_vertices >= 10
        or candidate.child_count == 0
    )
    faint_partial_fill_signal = (
        strict_fill_mode
        and slot_sized_geometry
        and faint_inner_fill_signal
        and faint_shape_evidence
        and (
            candidate.fill_ratio >= max(0.105, float(median_fill) + max(0.018, fill_margin * 0.34))
            or candidate.child_count == 0
            or candidate.inner_hole_ratio <= 0.34
        )
    )
    outline_damage_signal = (
        slot_sized_geometry
        and low_fill_cell
        and (
            (outline_min <= 0.34 and outline_mean >= 0.50 and outline_imbalance >= 0.34)
            or (outline_min <= 0.25 and outline_mean >= 0.42)
        )
        and (
            candidate.solidity <= 0.86
            or candidate.approx_vertices >= 10
            or candidate.child_count == 0
            or abs(float(candidate.extent) - 0.74) >= 0.12
        )
    )
    strict_shape_distortion_signal = (
        (strict_fill_mode or strict_merge_mode)
        and slot_sized_geometry
        and (
            (outline_min <= 0.48 and outline_mean >= 0.36 and outline_imbalance >= 0.18)
            or (
                candidate.solidity <= 0.89
                and abs(float(candidate.extent) - 0.74) >= 0.075
                and candidate.approx_vertices >= 6
            )
            or (
                candidate.approx_vertices >= 9
                and (
                    outline_imbalance >= 0.14
                    or abs(float(candidate.extent) - 0.74) >= 0.09
                    or candidate.solidity <= 0.91
                )
            )
        )
    )
    pinched_geometry_signal = slot_sized_geometry and (
        (
            candidate.solidity <= 0.84
            and candidate.approx_vertices >= 8
            and (abs(float(candidate.extent) - 0.74) >= 0.10 or outline_imbalance >= 0.16 or outline_min <= 0.60)
        )
        or (candidate.solidity <= 0.78 and candidate.approx_vertices >= 6)
        or (
            candidate.approx_vertices >= 12
            and (outline_imbalance >= 0.22 or abs(float(candidate.extent) - 0.74) >= 0.14)
        )
    )
    warped_slot_signal = (
        slot_sized_geometry
        and low_fill_cell
        and (
            candidate.child_count == 0
            or candidate.inner_hole_ratio <= 0.020
            or outline_min <= 0.68
            or outline_imbalance >= 0.12
            or abs(aspect_ratio - median_aspect) >= max(0.11, median_aspect * 0.13)
        )
        and (
            candidate.solidity <= 0.94
            or abs(float(candidate.extent) - 0.74) >= 0.055
            or candidate.approx_vertices >= 7
            or outline_min <= 0.58
        )
        and not (
            candidate.child_count > 0
            and candidate.inner_hole_ratio >= 0.030
            and outline_min >= 0.58
            and outline_imbalance <= 0.16
            and abs(aspect_ratio - median_aspect) <= max(0.09, median_aspect * 0.10)
        )
    )
    broken_geometry_signal = (
        slot_sized_geometry
        and (
            (
                candidate.child_count == 0
                and candidate.solidity <= float(config.geometry_solidity_limit)
                and candidate.approx_vertices >= 8
            )
            or (candidate.solidity <= 0.66 and abs(float(candidate.extent) - 0.74) >= 0.22)
            or (
                candidate.approx_vertices >= 16
                and candidate.solidity <= 0.82
                and abs(float(candidate.extent) - 0.74) >= 0.14
            )
            or outline_damage_signal
            or pinched_geometry_signal
            or strict_shape_distortion_signal
            or warped_slot_signal
        )
    ) or (
        # Severely warped mid-size fragments that fall below the normal slot band.
        0.26 <= smallest_axis_ratio < 0.48
        and largest_axis_ratio <= 1.50
        and 0.16 <= float(area_ratio) <= 1.20
        and candidate.solidity <= 0.74
        and (
            candidate.approx_vertices >= 10
            or outline_imbalance >= 0.26
            or outline_min <= 0.36
            or abs(float(candidate.extent) - 0.74) >= 0.18
        )
    )
    small_artifact_signal = _small_artifact_signal(
        candidate,
        median_width=median_width,
        median_height=median_height,
        median_area=median_area,
        median_fill=median_fill,
        median_interior_fill=median_interior_fill,
        median_center_fill=median_center_fill,
        config=config,
    )
    border_merged_signal = _border_merged_cell_signal(
        candidate,
        median_width=median_width,
        median_height=median_height,
        median_area=median_area,
    )
    if (
        (not has_hole_signal or candidate.interior_fill_ratio >= max(0.24, interior_limit * 0.78))
        and filled_by_center
        and smallest_axis_ratio >= 0.58
        and largest_axis_ratio <= 1.55
    ):
        fill_excess = max(
            center_fill - center_limit,
            candidate.interior_fill_ratio - interior_limit,
            candidate.fill_ratio - fill_limit,
            0.0,
        )
        score = max(score, min(1.0, 0.80 + fill_excess))
        reasons.append("filled_cell")
    elif partial_fill_signal or faint_partial_fill_signal:
        fill_excess = max(
            center_fill - max(0.16, float(median_fill) + 0.08),
            candidate.interior_fill_ratio - max(0.22, float(median_fill) + 0.08),
            candidate.fill_ratio - max(float(median_fill) + 0.10, 0.24),
            0.0,
        )
        base_score = 0.68 if faint_partial_fill_signal and not partial_fill_signal else 0.74
        score = max(score, min(0.90, base_score + fill_excess * 0.80))
        reasons.append("partial_filled_cell")
    elif broken_geometry_signal:
        concavity = max(0.0, 0.82 - float(candidate.solidity))
        extent_delta = abs(float(candidate.extent) - 0.74)
        vertex_excess = max(0.0, float(candidate.approx_vertices) - 8.0) / 24.0
        outline_loss = max(0.0, 0.55 - outline_min)
        outline_skew = max(0.0, outline_imbalance - 0.25)
        warp_bonus = 0.08 if warped_slot_signal else 0.0
        score = max(
            score,
            min(
                0.92,
                0.76
                + warp_bonus
                + 0.38 * concavity
                + 0.20 * extent_delta
                + 0.08 * vertex_excess
                + 0.16 * outline_loss
                + 0.10 * outline_skew,
            ),
        )
        reasons.append("broken_geometry")

    near_merged_signal = (
        strict_merge_mode
        and largest_axis_ratio >= max(1.28, float(config.merged_size_ratio) - 0.08)
        and smallest_axis_ratio >= 0.50
        and float(area_ratio) >= max(1.05, float(config.merged_area_ratio) - 0.20)
        and (
            candidate.child_count >= 2
            or (
                abs(aspect_ratio - median_aspect) > max(0.35, median_aspect * 0.40)
                and largest_axis_ratio >= 1.55
            )
            or (candidate.inner_hole_ratio >= 0.050 and outline_imbalance >= 0.22)
        )
    )
    bridge_connected_signal = (
        strict_merge_mode
        and largest_axis_ratio >= max(1.55, float(config.merged_size_ratio) + 0.18)
        and smallest_axis_ratio >= 0.34
        and float(bbox_area_ratio) >= 1.45
        and float(area_ratio) >= 0.40
        and (
            outline_mean >= 0.28
            or candidate.fill_ratio >= max(0.14, float(median_fill) * 0.70)
            or candidate.interior_fill_ratio >= max(0.060, normal_interior + 0.020)
            or candidate.approx_vertices >= 10
        )
        and (
            outline_imbalance >= 0.16
            or outline_min <= 0.55
            or abs(aspect_ratio - median_aspect) > max(0.18, median_aspect * 0.22)
            or candidate.solidity <= 0.86
        )
    )
    single_intact_cell_signal = (
        0.68 <= smallest_axis_ratio
        and largest_axis_ratio <= 1.42
        and float(area_ratio) <= 2.05
        and abs(aspect_ratio - median_aspect) <= max(0.18, median_aspect * 0.22)
        and candidate.child_count <= 1
        and candidate.inner_hole_ratio >= 0.010
        and outline_min >= 0.42
        and outline_imbalance <= 0.34
        and candidate.interior_fill_ratio <= max(0.18, normal_interior + 0.12)
        and center_fill <= max(0.14, normal_center + 0.08)
    )
    merged_contour_signal = (
        (
            max(width_ratio, height_ratio) > float(config.merged_size_ratio)
            and area_ratio > float(config.merged_area_ratio)
        )
        or (
            max(width_ratio, height_ratio) > max(1.05, float(config.merged_size_ratio) - 0.23)
            and area_ratio > max(1.05, float(config.merged_area_ratio) - 0.20)
            and abs(aspect_ratio - median_aspect) > max(0.18, median_aspect * 0.22)
        )
        or (
            largest_axis_ratio >= max(1.30, float(config.merged_size_ratio) - 0.36)
            and smallest_axis_ratio >= 0.46
            and float(area_ratio) >= max(0.92, float(config.merged_area_ratio) - 0.46)
            and (
                outline_mean >= 0.36
                or candidate.child_count >= 2
                or candidate.inner_hole_ratio >= 0.05
                or candidate.fill_ratio >= max(0.18, float(median_fill) * 0.82)
            )
            and (abs(aspect_ratio - median_aspect) > max(0.14, median_aspect * 0.16) or largest_axis_ratio >= 1.55)
        )
        or (
            largest_axis_ratio >= 1.72
            and smallest_axis_ratio >= 0.38
            and float(area_ratio) >= 0.62
            and (
                candidate.interior_fill_ratio >= max(0.14, float(median_fill) + 0.04)
                or candidate.fill_ratio >= max(0.26, float(median_fill) + 0.08)
            )
        )
        or area_ratio > max(1.35, float(config.merged_area_ratio) * 2.03)
        or near_merged_signal
        or bridge_connected_signal
        or border_merged_signal
    )
    if (
        merged_contour_signal
        and not (single_intact_cell_signal and not bridge_connected_signal)
    ):
        score = max(score, 0.86 if bridge_connected_signal else (0.78 if near_merged_signal else 0.88))
        reasons.append("merged_contour")
    if small_artifact_signal:
        area_ratio_clipped = min(1.0, max(0.0, float(area_ratio) / 0.45))
        score = max(score, min(0.86, 0.72 + area_ratio_clipped * 0.12))
        reasons.append("small_artifact")
        reasons = [reason for reason in reasons if reason != "broken_geometry"]
    edge_filled_signal = (
        center_fill >= max(center_limit, 0.42)
        and candidate.interior_fill_ratio >= max(0.32, interior_limit * 0.86)
        and candidate.fill_ratio >= max(float(median_fill) + 0.14, fill_limit * 0.72)
    )
    edge_size_clip_signal = (
        getattr(candidate, "touches_border", False)
        and 0.14 <= smallest_axis_ratio <= 0.88
        and largest_axis_ratio <= 1.75
        and float(area_ratio) <= 1.30
        and (
            smallest_axis_ratio < 0.78
            or abs(aspect_ratio - median_aspect) >= max(0.16, median_aspect * 0.18)
            or float(area_ratio) <= 0.88
        )
    )
    edge_filled_clip_signal = (
        getattr(candidate, "touches_border", False)
        and not _is_cell_like_candidate(
            candidate,
            median_width=median_width,
            median_height=median_height,
            median_area=median_area,
            config=config,
        )
        and 0.22 <= smallest_axis_ratio <= 1.35
        and largest_axis_ratio <= 1.70
        and edge_filled_signal
    )
    if edge_size_clip_signal or edge_filled_clip_signal:
        edge_loss = max(0.0, 1.0 - smallest_axis_ratio)
        area_loss = max(0.0, 0.74 - float(area_ratio))
        score = max(score, min(0.94, 0.72 + 0.18 * edge_loss + 0.10 * area_loss))
        reasons.append("edge_clipped_cell")
        reasons = [reason for reason in reasons if reason != "broken_geometry"]
    score, reasons = _apply_confidence_geometry_boost(
        candidate,
        score,
        reasons,
        median_width=median_width,
        median_height=median_height,
        median_area=median_area,
        confidence_map=confidence_map,
    )
    if not reasons:
        return 0.0, ()
    return float(max(0.0, min(1.0, score))), tuple(dict.fromkeys(reasons))


def _small_artifact_signal(
    candidate: _ContourCandidate,
    *,
    median_width: float,
    median_height: float,
    median_area: float,
    median_fill: float,
    median_interior_fill: float,
    median_center_fill: float,
    config: GridDamageAnalysisConfig | None = None,
) -> bool:
    width = float(candidate.bbox[2])
    height = float(candidate.bbox[3])
    width_ratio = width / max(1.0, float(median_width))
    height_ratio = height / max(1.0, float(median_height))
    # Prefer slot (bbox) area: confidence hollow outlines report contourArea ≈ full slot.
    slot_area = max(1.0, float(median_width) * float(median_height))
    area_ratio = float(candidate.area) / slot_area
    bbox_area_ratio = float(candidate.bbox_area) / slot_area
    smallest_axis_ratio = min(width_ratio, height_ratio)
    largest_axis_ratio = max(width_ratio, height_ratio)
    if width < 2.0 or height < 2.0:
        return False
    binary = config is not None and str(config.cell_representation) == "binary"
    # Ignore true speckles; keep small-but-critical dirt (~2.5–55% of a cell slot).
    min_area = max(6.0, slot_area * 0.014)
    if binary:
        max_largest_axis = 1.45
        elongated_largest_cap = 1.28
        compact_largest_cap = 0.85
    else:
        max_largest_axis = 1.28
        elongated_largest_cap = 1.12
        compact_largest_cap = 0.72
    if float(candidate.area) < min_area:
        return False
    # Hard speckle gate: 1–2 px chips stay ignored even when slot medians are tiny.
    if float(candidate.area) < 8.0 and max(width, height) <= 3.0:
        return False
    if area_ratio < 0.025:
        return False
    if candidate.child_count > 0:
        # Confidence outlines use children as holes; binary grain dirt often has speck holes.
        if not binary:
            return False
        if (
            smallest_axis_ratio >= 0.42
            and largest_axis_ratio >= 0.52
            and area_ratio >= 0.28
            and float(candidate.inner_hole_ratio) >= 0.035
        ):
            return False
    if smallest_axis_ratio < 0.035 or largest_axis_ratio > max_largest_axis:
        return False
    line_like_fragment = (
        smallest_axis_ratio <= (0.14 if binary else 0.18)
        and largest_axis_ratio >= 0.38
        and float(candidate.area) <= max(18.0, float(median_area) * 0.42)
    )
    if line_like_fragment:
        return False
    # Dense strip along one cell axis is a polygon edge fragment, not dirt.
    cell_edge_strip = (
        smallest_axis_ratio <= 0.38
        and 0.55 <= largest_axis_ratio <= 1.20
        and float(candidate.solidity) >= 0.82
        and float(candidate.fill_ratio) >= 0.45
        and bbox_area_ratio <= 0.48
    )
    if cell_edge_strip:
        return False
    if area_ratio > 0.58 or bbox_area_ratio > 0.92:
        return False
    normal_interior = max(0.0, float(median_interior_fill))
    normal_center = max(0.0, float(median_center_fill))
    outline_fragment = (
        not binary
        and largest_axis_ratio >= 0.42
        and smallest_axis_ratio <= 0.62
        and float(candidate.outline_mean_side_coverage) >= 0.12
        and float(candidate.interior_fill_ratio) <= max(0.14, normal_interior + 0.055)
        and float(candidate.center_fill_ratio) <= max(0.10, normal_center + 0.040)
        and float(candidate.fill_ratio) <= min(0.40, max(0.28, float(median_fill) + 0.08))
    )
    if outline_fragment:
        return False
    cell_sized_fragment = (
        smallest_axis_ratio >= 0.48
        and largest_axis_ratio >= 0.58
        and bbox_area_ratio >= 0.22
        and area_ratio >= 0.40
        and (0.55 <= float(candidate.aspect_ratio) <= 1.85 or float(candidate.outline_mean_side_coverage) >= 0.10)
    )
    if cell_sized_fragment:
        return False
    border_cell_fragment = bool(
        getattr(candidate, "touches_border", False)
        and largest_axis_ratio >= 0.34
        and smallest_axis_ratio <= 0.56
        and bbox_area_ratio >= 0.045
        and (height_ratio >= 0.46 or width_ratio >= 0.46 or float(candidate.outline_mean_side_coverage) >= 0.18)
    )
    if border_cell_fragment:
        return False
    compact_debris = (
        smallest_axis_ratio >= 0.06
        and largest_axis_ratio <= compact_largest_cap
        and 0.025 <= area_ratio <= 0.55
        and bbox_area_ratio <= 0.65
    )
    elongated_or_blurred_debris = (
        largest_axis_ratio <= elongated_largest_cap
        and 0.025 <= area_ratio <= 0.52
        and bbox_area_ratio <= 0.65
        and (
            smallest_axis_ratio <= 0.36
            or float(candidate.solidity) <= 0.76
            or float(candidate.extent) <= 0.42
            or candidate.approx_vertices >= 7
        )
    )
    mid_inaccurate_blob = (
        0.025 <= area_ratio <= 0.55
        and largest_axis_ratio <= 0.95
        and smallest_axis_ratio <= 0.88
        and float(candidate.solidity) >= 0.35
        and (
            float(candidate.fill_ratio) >= 0.22
            or float(candidate.extent) <= 0.75
            or candidate.approx_vertices >= 6
        )
    )
    if not (compact_debris or elongated_or_blurred_debris or mid_inaccurate_blob):
        return False
    foreground_signal = (
        candidate.fill_ratio >= max(0.16, float(median_fill) * 0.55)
        or candidate.extent >= 0.38
        or candidate.solidity >= 0.58
        or elongated_or_blurred_debris
        or mid_inaccurate_blob
    )
    return bool(foreground_signal)


def _is_detached_confidence_cell_edge(
    candidate: _ContourCandidate,
    candidates: list[_ContourCandidate],
    *,
    median_width: float,
    median_height: float,
    median_area: float,
    config: GridDamageAnalysisConfig,
) -> bool:
    if str(config.cell_representation) != "confidence":
        return False
    width = float(candidate.bbox[2])
    height = float(candidate.bbox[3])
    width_ratio = width / max(1.0, float(median_width))
    height_ratio = height / max(1.0, float(median_height))
    area_ratio = float(candidate.area) / max(1.0, float(median_area))
    bbox_area_ratio = float(candidate.bbox_area) / max(1.0, float(median_width) * float(median_height))
    vertical_edge = width_ratio <= 0.36 and 0.55 <= height_ratio <= 1.22
    horizontal_edge = height_ratio <= 0.36 and 0.55 <= width_ratio <= 1.22
    clean_edge_shape = (
        area_ratio <= 0.42
        and bbox_area_ratio <= 0.46
        and float(candidate.solidity) >= 0.80
        and float(candidate.extent) >= 0.50
        and int(candidate.approx_vertices) <= 8
        and int(candidate.child_count) == 0
    )
    if not clean_edge_shape or not (vertical_edge or horizontal_edge):
        return False

    x, y, candidate_width, candidate_height = candidate.bbox
    for companion in candidates:
        if int(companion.contour_id) == int(candidate.contour_id) or int(companion.child_count) > 0:
            continue
        if float(companion.solidity) > 0.78 and float(companion.extent) > 0.52:
            continue
        other_x, other_y, other_width, other_height = companion.bbox
        left = min(int(x), int(other_x))
        top = min(int(y), int(other_y))
        right = max(int(x + candidate_width), int(other_x + other_width))
        bottom = max(int(y + candidate_height), int(other_y + other_height))
        union_width_ratio = float(right - left) / max(1.0, float(median_width))
        union_height_ratio = float(bottom - top) / max(1.0, float(median_height))
        if not (0.72 <= union_width_ratio <= 1.34 and 0.72 <= union_height_ratio <= 1.34):
            continue
        horizontal_gap = max(0, max(int(x), int(other_x)) - min(int(x + candidate_width), int(other_x + other_width)))
        vertical_gap = max(0, max(int(y), int(other_y)) - min(int(y + candidate_height), int(other_y + other_height)))
        horizontal_overlap = max(
            0, min(int(x + candidate_width), int(other_x + other_width)) - max(int(x), int(other_x))
        )
        vertical_overlap = max(
            0, min(int(y + candidate_height), int(other_y + other_height)) - max(int(y), int(other_y))
        )
        if vertical_edge:
            aligned = (
                horizontal_gap <= max(2.0, float(median_width) * 0.16)
                and vertical_overlap >= min(int(candidate_height), int(other_height)) * 0.55
                and float(other_width) >= float(median_width) * 0.42
            )
        else:
            aligned = (
                vertical_gap <= max(2.0, float(median_height) * 0.16)
                and horizontal_overlap >= min(int(candidate_width), int(other_width)) * 0.55
                and float(other_height) >= float(median_height) * 0.42
            )
        if aligned:
            return True
    return False


def _defect_feature_summary(
    candidate: _ContourCandidate,
    *,
    median_width: float,
    median_height: float,
    median_area: float,
) -> np.ndarray:
    width_ratio = float(candidate.bbox[2]) / max(1.0, float(median_width))
    height_ratio = float(candidate.bbox[3]) / max(1.0, float(median_height))
    area_ratio = float(candidate.area) / max(1.0, float(median_area))
    bbox_area_ratio = float(candidate.bbox_area) / max(1.0, float(median_width) * float(median_height))
    return np.asarray(
        [
            width_ratio,
            height_ratio,
            area_ratio,
            bbox_area_ratio,
            float(candidate.fill_ratio),
            float(candidate.interior_fill_ratio),
            float(candidate.center_fill_ratio),
            float(candidate.solidity),
            abs(float(candidate.extent) - 0.74),
            float(candidate.approx_vertices),
            float(candidate.inner_hole_ratio),
            1.0 if bool(getattr(candidate, "touches_border", False)) else 0.0,
        ],
        dtype=np.float32,
    )


def _defect_cluster_vector(summary: np.ndarray) -> np.ndarray:
    values = np.asarray(summary, dtype=np.float32).reshape(-1)
    if values.size < 12:
        return np.zeros((12,), dtype=np.float32)
    width_ratio, height_ratio, area_ratio, bbox_area_ratio = (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
    )
    return np.asarray(
        [
            np.log(max(0.05, width_ratio)),
            np.log(max(0.05, height_ratio)),
            np.log(max(0.05, area_ratio)),
            np.log(max(0.05, bbox_area_ratio)),
            float(values[4]),
            float(values[5]),
            float(values[6]),
            1.0 - min(1.0, max(0.0, float(values[7]))),
            float(values[8]),
            min(1.0, max(0.0, float(values[9]) / 20.0)),
            min(1.0, max(0.0, float(values[10]))),
            min(1.0, max(0.0, float(values[11]))),
        ],
        dtype=np.float32,
    )


def _normalize_defect_cluster_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.size <= 0:
        return values.reshape((0, 0))
    center = np.median(values, axis=0)
    spread = np.percentile(values, 75.0, axis=0) - np.percentile(values, 25.0, axis=0)
    spread = np.maximum(spread, np.maximum(values.std(axis=0), 1e-4))
    return np.nan_to_num((values - center) / spread, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _initial_defect_cluster_centers(normalized: np.ndarray, cluster_count: int) -> np.ndarray:
    rows = int(normalized.shape[0])
    if rows <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    first = int(np.argmin(np.linalg.norm(normalized, axis=1)))
    selected = [first]
    while len(selected) < int(cluster_count):
        selected_values = normalized[np.asarray(selected, dtype=np.int32)]
        distances = np.min(np.linalg.norm(normalized[:, None, :] - selected_values[None, :, :], axis=2), axis=1)
        for index in selected:
            distances[int(index)] = -1.0
        next_index = int(np.argmax(distances))
        if next_index in selected or float(distances[next_index]) <= 1e-6:
            break
        selected.append(next_index)
    return normalized[np.asarray(selected, dtype=np.int32)].copy()


def _kmeans_defect_clusters(normalized: np.ndarray, cluster_count: int) -> np.ndarray:
    rows = int(normalized.shape[0])
    if rows <= 0:
        return np.zeros((0,), dtype=np.int32)
    centers = _initial_defect_cluster_centers(normalized, max(1, min(rows, int(cluster_count))))
    if centers.shape[0] <= 1:
        return np.zeros((rows,), dtype=np.int32)
    labels = np.zeros((rows,), dtype=np.int32)
    for _iteration in range(24):
        distances = np.linalg.norm(normalized[:, None, :] - centers[None, :, :], axis=2)
        next_labels = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for cluster_id in range(int(centers.shape[0])):
            members = normalized[labels == cluster_id]
            if members.size > 0:
                centers[cluster_id] = members.mean(axis=0)
    return labels.astype(np.int32)


def _reason_counts(cells: list[GridCellAnalysisResult], indexes: tuple[int, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index in indexes:
        if not (0 <= int(index) < len(cells)):
            continue
        for reason in cells[int(index)].reasons:
            key = str(reason)
            counts[key] = int(counts.get(key, 0) + 1)
    return counts


def _dominant_value(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "")
        if not key:
            continue
        counts[key] = int(counts.get(key, 0) + 1)
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _defect_feature_cluster_label(summary_mean: np.ndarray, reason_counts: dict[str, int]) -> str:
    values = np.asarray(summary_mean, dtype=np.float32).reshape(-1)
    if values.size < 12:
        return "mixed_defect"
    width_ratio = float(values[0])
    height_ratio = float(values[1])
    area_ratio = float(values[2])
    bbox_area_ratio = float(values[3])
    fill_ratio = float(values[4])
    interior_fill = float(values[5])
    center_fill = float(values[6])
    solidity = float(values[7])
    extent_delta = float(values[8])
    vertices = float(values[9])
    largest_axis = max(width_ratio, height_ratio)
    smallest_axis = min(width_ratio, height_ratio)
    scores = {
        "filled_like": max((fill_ratio - 0.24) / 0.48, (interior_fill - 0.12) / 0.42, (center_fill - 0.08) / 0.34),
        "debris_like": max((0.66 - area_ratio) / 0.66, (0.62 - bbox_area_ratio) / 0.62, (0.55 - largest_axis) / 0.55),
        "broken_shape": max((0.90 - solidity) / 0.36, (vertices - 7.0) / 12.0, (extent_delta - 0.08) / 0.26),
        "merged_like": max((largest_axis - 1.18) / 0.82, (area_ratio - 1.20) / 1.25, (bbox_area_ratio - 1.25) / 1.35),
    }
    reason_boosts = {
        "filled_cell": "filled_like",
        "partial_filled_cell": "filled_like",
        "small_artifact": "debris_like",
        "broken_geometry": "broken_shape",
        "edge_clipped_cell": "broken_shape",
        "merged_contour": "merged_like",
    }
    for reason, label in reason_boosts.items():
        if int(reason_counts.get(reason, 0)) > 0:
            scores[label] = float(scores.get(label, 0.0) + 0.18)
    if smallest_axis <= 0.28 and area_ratio <= 0.74:
        scores["debris_like"] = float(scores["debris_like"] + 0.10)
    if largest_axis >= 1.45:
        scores["merged_like"] = float(scores["merged_like"] + 0.10)
    label, score = max(scores.items(), key=lambda item: (float(item[1]), item[0]))
    return str(label if float(score) >= 0.18 else "mixed_defect")


def _union_bbox(cells: list[GridCellAnalysisResult], indexes: tuple[int, ...]) -> tuple[int, int, int, int]:
    boxes = [
        tuple(int(value) for value in cells[int(index)].bbox[:4]) for index in indexes if 0 <= int(index) < len(cells)
    ]
    if not boxes:
        return (0, 0, 0, 0)
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return (int(left), int(top), int(max(0, right - left)), int(max(0, bottom - top)))


def _cluster_defective_cell_features(
    cells: list[GridCellAnalysisResult],
    entries: list[tuple[int, _ContourCandidate]],
    *,
    median_width: float,
    median_height: float,
    median_area: float,
    max_clusters: int = 6,
) -> tuple[tuple[GridCellFeatureCluster, ...], dict[int, tuple[int, str]]]:
    if not entries:
        return (), {}
    summaries = np.vstack(
        [
            _defect_feature_summary(
                candidate,
                median_width=median_width,
                median_height=median_height,
                median_area=median_area,
            )
            for _cell_index, candidate in entries
        ]
    ).astype(np.float32)
    vectors = np.vstack([_defect_cluster_vector(summary) for summary in summaries]).astype(np.float32)
    rows = int(vectors.shape[0])
    if rows <= 4:
        labels = np.arange(rows, dtype=np.int32)
    else:
        cluster_count = int(np.clip(round(np.sqrt(float(rows)) * 1.35), 2, max(2, min(rows, int(max_clusters)))))
        labels = _kmeans_defect_clusters(_normalize_defect_cluster_matrix(vectors), cluster_count)

    raw_clusters = []
    for source_cluster_id in sorted(set(int(item) for item in labels.tolist())):
        local_indexes = tuple(int(index) for index in np.where(labels == int(source_cluster_id))[0].tolist())
        if not local_indexes:
            continue
        cell_indexes = tuple(int(entries[local_index][0]) for local_index in local_indexes)
        contour_ids = tuple(int(entries[local_index][1].contour_id) for local_index in local_indexes)
        summary_mean = summaries[np.asarray(local_indexes, dtype=np.int32)].mean(axis=0)
        reasons = _reason_counts(cells, cell_indexes)
        scores = [float(cells[index].score) for index in cell_indexes if 0 <= int(index) < len(cells)]
        statuses = [str(cells[index].status) for index in cell_indexes if 0 <= int(index) < len(cells)]
        dominant_reason = _dominant_value([reason for reason, count in reasons.items() for _ in range(int(count))])
        raw_clusters.append(
            (
                int(source_cluster_id),
                cell_indexes,
                contour_ids,
                summary_mean,
                float(np.mean(scores)) if scores else 0.0,
                float(np.max(scores)) if scores else 0.0,
                _dominant_value(statuses),
                dominant_reason,
                tuple(float(value) for value in summary_mean.tolist()),
            )
        )

    ordered = sorted(
        raw_clusters,
        key=lambda item: (
            -float(item[5]),
            str(_defect_feature_cluster_label(item[3], _reason_counts(cells, item[1]))),
            item[0],
        ),
    )
    clusters: list[GridCellFeatureCluster] = []
    cluster_by_cell: dict[int, tuple[int, str]] = {}
    for new_cluster_id, (
        _source_cluster_id,
        cell_indexes,
        contour_ids,
        summary_mean,
        mean_score,
        max_score,
        dominant_status,
        dominant_reason,
        feature_mean,
    ) in enumerate(ordered):
        reasons = _reason_counts(cells, cell_indexes)
        label = _defect_feature_cluster_label(summary_mean, reasons)
        clusters.append(
            GridCellFeatureCluster(
                cluster_id=int(new_cluster_id),
                label=str(label),
                member_count=int(len(cell_indexes)),
                cell_indexes=tuple(cell_indexes),
                contour_ids=tuple(contour_ids),
                bbox=_union_bbox(cells, cell_indexes),
                mean_score=float(mean_score),
                max_score=float(max_score),
                dominant_status=str(dominant_status),
                dominant_reason=str(dominant_reason),
                feature_mean=tuple(feature_mean),
            )
        )
        for cell_index in cell_indexes:
            cluster_by_cell[int(cell_index)] = (int(new_cluster_id), str(label))
    return tuple(clusters), cluster_by_cell


def _damage_score(cells: list[GridCellAnalysisResult], total_expected: int) -> float:
    weights = {
        "normal": 0.0,
        "suspicious": 0.35,
        "broken": 0.90,
        "missing": 1.00,
        "artifact": 0.75,
    }
    weighted = 0.0
    bad_scores: list[float] = []
    for cell in cells:
        if tuple(cell.reasons or ()) == ("edge_clipped_cell",):
            # A cell cut by the frame is not a network error; it stays listed but does not
            # make a clean frame look damaged.
            continue
        status_weight = weights.get(str(cell.status), 0.50)
        weighted += status_weight * max(0.25 if status_weight else 0.0, float(cell.score))
        if cell.status != "normal":
            bad_scores.append(float(cell.score))
    base = weighted / max(1.0, float(total_expected))
    severity_tail = 0.12 * (float(np.mean(bad_scores)) if bad_scores else 0.0)
    return float(np.clip(base + severity_tail, 0.0, 1.0))


def _filter_disabled_grid_reasons(
    score: float,
    reasons: tuple[str, ...],
    config: GridDamageAnalysisConfig,
) -> tuple[float, tuple[str, ...]]:
    enabled = config.enabled_reason_types
    if enabled is None:
        return float(score), tuple(reasons)
    enabled_set = {str(reason) for reason in enabled}
    filtered = tuple(str(reason) for reason in reasons if str(reason) in enabled_set)
    if not filtered:
        return 0.0, ()
    return float(score), filtered


def _is_bad_grid_cell(score: float, reasons: tuple[str, ...], config: GridDamageAnalysisConfig | None = None) -> bool:
    threshold = 0.72 if config is None else float(config.bad_score_threshold)
    if score >= threshold:
        return True
    reason_set = set(reasons)
    strong_reasons = {
        "filled_cell",
        "partial_filled_cell",
        "broken_geometry",
        "merged_contour",
        "edge_clipped_cell",
        "small_artifact",
    }
    strong_reason_threshold = min(0.70, max(0.05, threshold - 0.16))
    if reason_set & strong_reasons and score >= strong_reason_threshold:
        return True
    return False


def _compact_contours(contours) -> tuple[tuple[tuple[int, int], ...], ...]:
    compact = []
    for contour in contours:
        points = np.asarray(contour).reshape(-1, 2)
        if len(points) > 64:
            points = points[:: max(1, len(points) // 64)]
        compact.append(tuple((int(x), int(y)) for x, y in points))
    return tuple(compact)


def _write_debug_images(
    gray: np.ndarray,
    threshold: np.ndarray,
    contours,
    cells: list[GridCellAnalysisResult],
    x_axes: tuple[float, ...],
    y_axes: tuple[float, ...],
    frame_id: str,
    config: GridDamageAnalysisConfig,
) -> None:
    if cv2 is None:
        return
    debug_dir = Path(config.debug_dir or os.getenv("KARAKAL_GRID_DEBUG_DIR", GRID_DAMAGE_CACHE_DIR / "debug"))
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_debug_stem(frame_id or "frame")
        cv2.imwrite(str(debug_dir / f"{stem}_threshold.png"), threshold)
        contour_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_image, contours, -1, (0, 220, 255), 1)
        cv2.imwrite(str(debug_dir / f"{stem}_contours.png"), contour_image)
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x in x_axes:
            cv2.line(overlay, (int(round(x)), 0), (int(round(x)), overlay.shape[0] - 1), (255, 220, 0), 1)
        for y in y_axes:
            cv2.line(overlay, (0, int(round(y))), (overlay.shape[1] - 1, int(round(y))), (255, 220, 0), 1)
        for cell in cells:
            color = {
                "normal": (70, 210, 90),
                "suspicious": (0, 190, 255),
                "broken": (40, 40, 255),
                "missing": (255, 80, 220),
                "artifact": (255, 0, 180),
            }.get(cell.status, (255, 255, 255))
            x, y, w, h = cell.bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
        cv2.imwrite(str(debug_dir / f"{stem}_overlay.png"), overlay)
    except Exception as error:
        _LOGGER.debug("Could not write grid debug images for %s: %s", frame_id, error)
        return


def _safe_debug_stem(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return safe[:80] or "frame"


def _grid_cache_identity(path: Path) -> tuple[str, int, int]:
    try:
        stat = path.stat()
        return str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)
    except Exception as error:
        _LOGGER.debug("Could not resolve grid cache identity for %s: %s", path, error)
        return str(path), 0, 0


def _grid_damage_cache_key(
    path: Path,
    *,
    frame_id: str,
    config: GridDamageAnalysisConfig,
    reference_profile: GridCellReferenceProfile | None = None,
    zone_cache_token: str = "",
) -> tuple[Any, ...]:
    return (
        GRID_DAMAGE_ALGORITHM_VERSION,
        str(frame_id or ""),
        _grid_cache_identity(path),
        config.cache_payload(),
        None if reference_profile is None else reference_profile.cache_payload(),
        bool(getattr(config, "calibration_fingerprint", "")),
        str(getattr(config, "calibration_fingerprint", "") or ""),
        str(zone_cache_token or ""),
    )


def _grid_damage_cache_path(cache_key: tuple[Any, ...]) -> Path:
    digest = hashlib.sha1(repr(cache_key).encode("utf-8", errors="ignore")).hexdigest()
    return GRID_DAMAGE_CACHE_DIR / f"{digest}.pickle"


def _load_cached_grid_result(
    path: Path,
    *,
    frame_id: str,
    config: GridDamageAnalysisConfig,
    reference_profile: GridCellReferenceProfile | None = None,
    zone_cache_token: str = "",
) -> GridFrameAnalysisResult | None:
    cache_path = _grid_damage_cache_path(
        _grid_damage_cache_key(
            path,
            frame_id=frame_id,
            config=config,
            reference_profile=reference_profile,
            zone_cache_token=zone_cache_token,
        )
    )
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, GridFrameAnalysisResult):
            return payload
    except Exception as error:
        _LOGGER.warning("Ignoring corrupt grid cache entry %s: %s", cache_path, error)
        return None
    return None


def _store_cached_grid_result(
    path: Path,
    *,
    frame_id: str,
    config: GridDamageAnalysisConfig,
    reference_profile: GridCellReferenceProfile | None = None,
    result: GridFrameAnalysisResult,
    zone_cache_token: str = "",
) -> None:
    global _grid_damage_cache_last_trim
    try:
        GRID_DAMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _grid_damage_cache_path(
            _grid_damage_cache_key(
                path,
                frame_id=frame_id,
                config=config,
                reference_profile=reference_profile,
                zone_cache_token=zone_cache_token,
            )
        )
        atomic_pickle_dump(cache_path, result)
        now = time.monotonic()
        if now - _grid_damage_cache_last_trim >= GRID_DAMAGE_CACHE_TRIM_INTERVAL_SECONDS:
            _grid_damage_cache_last_trim = now
            profiler = current_profiler()
            performance = profiler.config if profiler is not None else load_performance_config()
            trim_directory_by_bytes(
                GRID_DAMAGE_CACHE_DIR,
                max_bytes=int(performance.disk_cache_limit_mb) * 1024 * 1024,
                max_files=GRID_DAMAGE_CACHE_MAX_FILES,
            )
    except (OSError, pickle.PickleError, TypeError, ValueError) as error:
        _LOGGER.warning("Could not store grid cache result for %s: %s", path, error)
        return


__all__ = [
    "GridCellAnalysisResult",
    "GridCellAnomaly",
    "GridCellAnomalyResult",
    "GridCellFeatureCluster",
    "GridCellReferenceProfile",
    "GridDamageAnalysisConfig",
    "GRID_DAMAGE_REASON_TYPES",
    "GridDamageSeverityThresholds",
    "GridFrameAnalysisResult",
    "analyze_grid_frame_chunk",
    "analyze_grid_frame_pair_chunk",
    "analyze_grid_frame_path",
    "analyze_grid_frame_single_source_path",
    "analyze_grid_frame_sources_chunk",
    "analyze_class_conflict_chunk",
    "analyze_class_conflict_frame_paths",
    "analyze_class_conflict_masks",
    "analyze_xor_residual_chunk",
    "analyze_xor_residual_frame_paths",
    "analyze_xor_residual_masks",
    "build_grid_cell_reference_profile",
    "build_grid_cell_reference_profile_path",
    "compare_grid_cell_analyses",
    "configure_grid_worker_process",
    "detect_grid_cell_anomalies",
    "load_cached_grid_frame_result",
    "GRID_DERIVED_LAYER_KEYS",
    "GRID_XOR_RESIDUAL_MIN_AREA",
]
