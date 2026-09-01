"""Detect damaged cells in regular grid-like frames with OpenCV."""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .backend_constants import CACHE_DIR
from .cache_utils import atomic_pickle_dump, trim_directory_by_bytes
from .performance import load_performance_config
from .profiling import current_profiler, profile_stage

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is optional at runtime
    cv2 = None


GRID_DAMAGE_ALGORITHM_VERSION = "grid_damage_v64_compact_cache"
GRID_DAMAGE_CACHE_DIR = CACHE_DIR / "grid_damage"
GRID_DAMAGE_CACHE_MAX_FILES = 20000
GRID_DAMAGE_CACHE_TRIM_INTERVAL_SECONDS = 300.0
_grid_damage_cache_last_trim = 0.0
_LOGGER = logging.getLogger(__name__)
GRID_DAMAGE_REASON_TYPES = (
    "filled_cell",
    "partial_filled_cell",
    "small_artifact",
    "conductor_residue",
    "broken_geometry",
    "merged_contour",
    "edge_clipped_cell",
    "confidence_only_cell",
    "binary_only_cell",
    "geometry_mismatch",
    "defect_disagreement",
)
_GRID_DAMAGE_REASON_TYPE_SET = set(GRID_DAMAGE_REASON_TYPES)
GRID_CELL_REPRESENTATION_MODES = ("confidence", "binary")


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
    min_grid_candidate_cells: int = 6
    expected_rows: int | None = None
    expected_cols: int | None = None
    severity_thresholds: GridDamageSeverityThresholds = field(default_factory=GridDamageSeverityThresholds)
    include_debug_payload: bool = False
    debug: bool = False
    debug_dir: str | None = None
    enabled_reason_types: tuple[str, ...] | None = GRID_DAMAGE_REASON_TYPES

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
            min_grid_candidate_cells=max(1, int(self.min_grid_candidate_cells)),
            expected_rows=None if self.expected_rows is None else max(1, int(self.expected_rows)),
            expected_cols=None if self.expected_cols is None else max(1, int(self.expected_cols)),
            severity_thresholds=self.severity_thresholds,
            include_debug_payload=bool(self.include_debug_payload),
            debug=bool(self.debug),
            debug_dir=self.debug_dir,
            enabled_reason_types=enabled_reason_types,
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

    def cache_payload(self) -> dict[str, Any]:
        return {
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


def compare_grid_cell_analyses(
    confidence_result: GridFrameAnalysisResult,
    binary_result: GridFrameAnalysisResult,
    *,
    centroid_tolerance_ratio: float = 0.72,
    geometry_iou_threshold: float = 0.58,
) -> GridFrameAnalysisResult:
    """Compare semantic cell geometry after each representation was analyzed independently."""

    confidence_cells = list(confidence_result.per_cell_results)
    binary_cells = list(binary_result.per_cell_results)
    cell_width = max(1.0, float(confidence_result.cell_width or binary_result.cell_width or 1))
    cell_height = max(1.0, float(confidence_result.cell_height or binary_result.cell_height or 1))
    match_distance = max(1.0, float(np.hypot(cell_width, cell_height)) * float(centroid_tolerance_ratio))
    candidates: list[tuple[float, float, int, int]] = []
    for confidence_index, confidence_cell in enumerate(confidence_cells):
        for binary_index, binary_cell in enumerate(binary_cells):
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
        if iou < float(geometry_iou_threshold) or distance_ratio > 0.48:
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


def detect_grid_cell_anomalies(
    image: np.ndarray,
    *,
    frame_id: str = "",
    frame_path: str = "",
    config: GridDamageAnalysisConfig | None = None,
    reference_profile: GridCellReferenceProfile | None = None,
) -> GridFrameAnalysisResult:
    """Analyze one image and return compact grid damage metrics."""

    cfg = (config or GridDamageAnalysisConfig()).normalized()
    started = perf_counter()
    with profile_stage("validation.grid.grayscale", frame_id=frame_id):
        gray = _normalize_grayscale(image)
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
    with profile_stage("validation.grid.contours.features", frame_id=frame_id):
        candidates = _extract_candidates(contours, hierarchy, threshold, gray.shape, cfg)
    if not candidates:
        return empty_result

    min_required_cells = int(cfg.min_grid_candidate_cells)
    if len(candidates) < min_required_cells:
        return empty_result

    with profile_stage("validation.grid.reference_profile", frame_id=frame_id):
        normal_seed = _normal_seed_candidates(candidates, config=cfg)
        profile = reference_profile or _grid_cell_reference_profile_from_candidates(
            candidates,
            config=cfg,
            frame_id=frame_id,
            frame_path=frame_path,
        )
    if profile is None:
        return empty_result
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
        candidates, key=lambda item: (float(item.centroid[1]), float(item.centroid[0]), int(item.contour_id))
    )
    with profile_stage("validation.grid.cells.classify", frame_id=frame_id):
        for candidate in ordered_candidates:
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
            )
            score, reasons = _filter_disabled_grid_reasons(score, reasons, cfg)
            if reasons == ("small_artifact",) and _is_detached_confidence_cell_edge(
                candidate,
                candidates,
                median_width=median_width,
                median_height=median_height,
                median_area=median_area,
                config=cfg,
            ):
                score, reasons = 0.0, ()
            is_bad = _is_bad_grid_cell(score, reasons, cfg)
            is_cell_like = int(candidate.contour_id) in normal_seed_ids or _is_cell_like_candidate(
                candidate,
                median_width=median_width,
                median_height=median_height,
                median_area=median_area,
                config=cfg,
            )
            if not is_bad and not is_cell_like:
                continue
            if not is_bad and _is_ignored_fragment(
                candidate, median_width=median_width, median_height=median_height, median_area=median_area
            ):
                continue
            index = len(per_cell)
            if is_bad:
                defect_entries.append((index, candidate))
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
                )
            )

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

    counts = {
        "normal": sum(1 for cell in per_cell if cell.status == "normal"),
        "suspicious": sum(1 for cell in per_cell if cell.status == "suspicious"),
        "broken": sum(1 for cell in per_cell if cell.status == "broken"),
        "missing": sum(1 for cell in per_cell if cell.status == "missing"),
        "artifact": sum(1 for cell in per_cell if cell.status == "artifact"),
    }
    if len(per_cell) < min_required_cells:
        return empty_result
    total_expected = max(1, len(per_cell))
    damage_score = _damage_score(per_cell, total_expected)
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
            "cell damage frame=%s contours=%d candidates=%d cells=%d normal=%d suspicious=%d broken=%d missing=%d artifact=%d score=%.4f time_ms=%.1f",
            frame_id or frame_path or "<array>",
            len(contours),
            len(candidates),
            total_expected,
            counts.get("normal", 0),
            counts.get("suspicious", 0),
            counts.get("broken", 0),
            counts.get("missing", 0),
            counts.get("artifact", 0),
            damage_score,
            elapsed_ms,
        )

    return GridFrameAnalysisResult(
        frame_id=str(frame_id or ""),
        frame_path=str(frame_path or ""),
        image_width=int(width),
        image_height=int(height),
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=int(total_expected),
        detected_cells=int(len(per_cell)),
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
    )


def analyze_grid_frame_path(
    path: Path | str,
    *,
    frame_id: str = "",
    config: GridDamageAnalysisConfig | None = None,
    reference_profile: GridCellReferenceProfile | None = None,
    use_cache: bool = True,
    read_cache: bool | None = None,
    write_cache: bool | None = None,
) -> GridFrameAnalysisResult | None:
    """Load one image, analyze it, and cache only compact analysis data."""

    path_obj = Path(path)
    cfg = (config or GridDamageAnalysisConfig()).normalized()
    should_read_cache = bool(use_cache) if read_cache is None else bool(read_cache)
    should_write_cache = bool(use_cache) if write_cache is None else bool(write_cache)
    if should_read_cache:
        with profile_stage("validation.cache.disk.read", frame_id=frame_id):
            cached = _load_cached_grid_result(
                path_obj, frame_id=frame_id, config=cfg, reference_profile=reference_profile
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
    if image is None:
        return None
    with profile_stage("validation.grid.frame", frame_id=frame_id, frame_count=1):
        result = detect_grid_cell_anomalies(
            image,
            frame_id=frame_id,
            frame_path=str(path_obj),
            config=cfg,
            reference_profile=reference_profile,
        )
    if should_write_cache:
        with profile_stage("validation.cache.disk.write", frame_id=frame_id):
            _store_cached_grid_result(
                path_obj, frame_id=frame_id, config=cfg, reference_profile=reference_profile, result=result
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


def analyze_grid_frame_pair_chunk(
    entries: tuple[tuple[str, str, str], ...],
    config: GridDamageAnalysisConfig,
    use_cache: bool = True,
    *,
    reference_profile: GridCellReferenceProfile | None = None,
) -> tuple[dict[str, dict[str, GridFrameAnalysisResult]], dict[str, str]]:
    """Analyze linked confidence/binary files and return all three matrix payloads."""

    payloads: dict[str, dict[str, GridFrameAnalysisResult]] = {}
    errors: dict[str, str] = {}
    confidence_config = replace(config, cell_representation="confidence").normalized()
    binary_config = replace(config, cell_representation="binary").normalized()
    for key, confidence_path, binary_path in entries:
        try:
            confidence_result = analyze_grid_frame_path(
                confidence_path,
                frame_id=key,
                config=confidence_config,
                reference_profile=reference_profile,
                use_cache=use_cache,
            )
            binary_result = analyze_grid_frame_path(
                binary_path,
                frame_id=key,
                config=binary_config,
                reference_profile=None,
                use_cache=use_cache,
            )
        except Exception as error:
            errors[str(key)] = f"{type(error).__name__}: {error}"
            continue
        if not isinstance(confidence_result, GridFrameAnalysisResult):
            errors[str(key)] = "confidence_decode_error"
            continue
        if not isinstance(binary_result, GridFrameAnalysisResult):
            errors[str(key)] = "binary_decode_error"
            continue
        payloads[str(key)] = {
            "confidence": confidence_result,
            "binary": binary_result,
            "comparison": compare_grid_cell_analyses(confidence_result, binary_result),
        }
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
            1 if layer == "confidence" else 0,
        )

    return max(valid, key=quality)


def analyze_grid_frame_sources_chunk(
    entries: tuple[tuple[str, str, str], ...],
    config: GridDamageAnalysisConfig,
    use_cache: bool = True,
    *,
    reference_profile: GridCellReferenceProfile | None = None,
    single_source_layer: str | None = None,
) -> tuple[dict[str, dict[str, GridFrameAnalysisResult]], dict[str, str]]:
    """Analyze complete pairs and gracefully handle either source on its own."""

    payloads: dict[str, dict[str, GridFrameAnalysisResult]] = {}
    errors: dict[str, str] = {}
    confidence_config = replace(config, cell_representation="confidence").normalized()
    for key, confidence_path, binary_path in entries:
        if confidence_path and binary_path:
            pair_payloads, pair_errors = analyze_grid_frame_pair_chunk(
                ((key, confidence_path, binary_path),),
                config,
                use_cache,
                reference_profile=reference_profile,
            )
            payloads.update(pair_payloads)
            errors.update(pair_errors)
            continue
        try:
            if confidence_path:
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
        payloads[str(key)] = {str(layer_key): result}
    return payloads, errors


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
    else:
        _level, threshold = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground_ratio = float(np.count_nonzero(threshold)) / max(1.0, float(threshold.size))
    representation = str(getattr(config, "cell_representation", "confidence") or "confidence")
    border = np.concatenate((threshold[0, :], threshold[-1, :], threshold[:, 0], threshold[:, -1]))
    border_foreground_ratio = float(np.count_nonzero(border)) / max(1.0, float(border.size))
    should_invert = border_foreground_ratio > 0.72 if representation == "binary" else foreground_ratio > 0.62
    if should_invert:
        threshold = cv2.bitwise_not(threshold)
    if config.morphology_open > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(config.morphology_open), int(config.morphology_open)))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    if config.morphology_close > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(config.morphology_close), int(config.morphology_close)))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    return np.ascontiguousarray(threshold)


def _extract_candidates(
    contours, hierarchy, threshold: np.ndarray, shape: tuple[int, int], config: GridDamageAnalysisConfig
) -> list[_ContourCandidate]:
    if hierarchy is None:
        hierarchy_array = np.zeros((0, 4), dtype=np.int32)
    else:
        hierarchy_array = np.asarray(hierarchy).reshape(-1, 4)
    height, width = shape
    candidates: list[_ContourCandidate] = []
    for index, contour in enumerate(contours):
        if index < len(hierarchy_array) and int(hierarchy_array[index][3]) >= 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < config.min_cell_size or h < config.min_cell_size:
            continue
        if w >= max(1, width - 1) and h >= max(1, height - 1):
            continue
        area = float(abs(cv2.contourArea(contour)))
        if area < float(config.min_contour_area):
            continue
        bbox_area = float(max(1, w * h))
        moments = cv2.moments(contour)
        if abs(float(moments.get("m00", 0.0))) > 1e-6:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx = float(x + w / 2.0)
            cy = float(y + h / 2.0)
        hull = cv2.convexHull(contour)
        hull_area = max(1.0, float(abs(cv2.contourArea(hull))))
        perimeter = float(cv2.arcLength(contour, True))
        epsilon = max(0.5, 0.025 * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        child_areas: list[float] = []
        child = int(hierarchy_array[index][2]) if index < len(hierarchy_array) else -1
        while child >= 0 and child < len(contours):
            child_areas.append(float(abs(cv2.contourArea(contours[child]))))
            next_child = int(hierarchy_array[child][0]) if child < len(hierarchy_array) else -1
            if next_child == child:
                break
            child = next_child
        roi = threshold[y : y + h, x : x + w]
        fill_ratio = float(np.count_nonzero(roi)) / bbox_area
        side_coverages = _outline_side_coverages(roi)
        outline_min_side_coverage = float(min(side_coverages, default=0.0))
        outline_mean_side_coverage = float(sum(side_coverages) / len(side_coverages)) if side_coverages else 0.0
        outline_side_imbalance = float(max(side_coverages, default=0.0) - min(side_coverages, default=0.0))
        border = max(2, int(round(min(w, h) * 0.22)))
        if w > border * 2 + 1 and h > border * 2 + 1:
            inner_roi = roi[border : h - border, border : w - border]
            interior_fill_ratio = float(np.count_nonzero(inner_roi)) / max(1.0, float(inner_roi.size))
        else:
            interior_fill_ratio = fill_ratio
        center_border = max(border + 1, int(round(min(w, h) * 0.36)))
        if w > center_border * 2 + 1 and h > center_border * 2 + 1:
            center_roi = roi[center_border : h - center_border, center_border : w - center_border]
            center_fill_ratio = float(np.count_nonzero(center_roi)) / max(1.0, float(center_roi.size))
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
            )
        )
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
        filtered = [
            item
            for item in candidates
            if 0.45 <= item.aspect_ratio <= 2.25
            and 0.42 <= item.fill_ratio <= 0.98
            and item.interior_fill_ratio >= 0.38
            and item.solidity >= 0.76
            and item.approx_vertices <= 14
        ]
        if len(filtered) >= 3:
            return filtered
        return [
            item
            for item in candidates
            if 0.35 <= item.aspect_ratio <= 2.80 and 0.28 <= item.fill_ratio <= 1.0 and item.solidity >= 0.55
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
    ]


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
    seed = normal_seed if len(normal_seed) >= 3 else candidates
    if not seed:
        return None
    return GridCellReferenceProfile(
        median_width=float(np.median([item.bbox[2] for item in seed])),
        median_height=float(np.median([item.bbox[3] for item in seed])),
        median_area=float(np.median([item.area for item in seed])),
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
        return bool(
            0.52 <= min(width_ratio, height_ratio)
            and max(width_ratio, height_ratio) <= 1.62
            and 0.34 <= area_ratio <= 2.35
            and candidate.fill_ratio >= 0.28
            and candidate.interior_fill_ratio >= 0.20
            and candidate.solidity >= 0.58
        )
    return bool(
        0.56 <= min(width_ratio, height_ratio)
        and max(width_ratio, height_ratio) <= 1.55
        and 0.38 <= area_ratio <= 2.10
        and candidate.child_count > 0
        and candidate.interior_fill_ratio <= 0.34
        and candidate.center_fill_ratio <= 0.20
    )


def _status_for_reasons(reasons: tuple[str, ...]) -> str:
    if {"small_artifact", "conductor_residue"}.intersection(reasons):
        return "artifact"
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


def _conductor_residue_signal(
    candidate: _ContourCandidate,
    *,
    median_width: float,
    median_height: float,
    median_area: float,
) -> bool:
    """Detect large, semantically irrelevant conductor remnants at the frame edge."""

    if not bool(candidate.touches_border):
        return False
    width_ratio = float(candidate.bbox[2]) / max(1.0, float(median_width))
    height_ratio = float(candidate.bbox[3]) / max(1.0, float(median_height))
    area_ratio = float(candidate.area) / max(1.0, float(median_area))
    bbox_area_ratio = float(candidate.bbox_area) / max(1.0, float(median_width) * float(median_height))
    largest_axis = max(width_ratio, height_ratio)
    smallest_axis = min(width_ratio, height_ratio)
    very_large = largest_axis >= 2.75 or bbox_area_ratio >= 3.40 or area_ratio >= 3.10
    large = largest_axis >= 1.70 and (bbox_area_ratio >= 1.85 or area_ratio >= 1.55)
    irregular = (
        float(candidate.solidity) <= 0.88
        or float(candidate.extent) <= 0.68
        or int(candidate.approx_vertices) >= 8
        or smallest_axis <= 0.46
        or largest_axis / max(0.05, smallest_axis) >= 2.45
    )
    return bool(very_large or (large and irregular))


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


def _classify_binary_cell(
    candidate: _ContourCandidate,
    *,
    median_width: float,
    median_height: float,
    median_area: float,
    median_fill: float,
    median_interior_fill: float,
    config: GridDamageAnalysisConfig,
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
    conductor_residue = not border_merged and _conductor_residue_signal(
        candidate,
        median_width=median_width,
        median_height=median_height,
        median_area=median_area,
    )
    if conductor_residue:
        residue_scale = max(largest_axis / 2.75, area_ratio / 3.10)
        return float(np.clip(0.80 + 0.12 * min(1.0, residue_scale), 0.0, 0.96)), ("conductor_residue",)

    merged = (
        border_merged
        or (largest_axis > float(config.merged_size_ratio) and area_ratio > float(config.merged_area_ratio))
        or (largest_axis >= 1.42 and smallest_axis >= 0.55 and area_ratio >= 1.32)
    )
    if merged:
        reasons.append("merged_contour")
        score = max(score, min(1.0, 0.82 + 0.12 * max(0.0, largest_axis - 1.35)))

    slot_sized = 0.50 <= smallest_axis and largest_axis <= 1.62 and 0.30 <= area_ratio <= 2.25
    fill_loss = max(
        0.0,
        max(0.34, float(median_fill) * 0.64) - float(candidate.fill_ratio),
        max(0.28, float(median_interior_fill) * 0.58) - float(candidate.interior_fill_ratio),
    )
    broken = slot_sized and (
        float(candidate.solidity) < 0.80 or int(candidate.approx_vertices) >= 12 or fill_loss > 0.08
    )
    if broken:
        reasons.append("broken_geometry")
        score = max(
            score,
            min(
                0.96,
                0.74
                + 0.55 * fill_loss
                + 0.35 * max(0.0, 0.82 - float(candidate.solidity))
                + 0.015 * max(0, int(candidate.approx_vertices) - 8),
            ),
        )

    small_artifact = _small_artifact_signal(
        candidate,
        median_width=median_width,
        median_height=median_height,
        median_area=median_area,
        median_fill=median_fill,
        median_interior_fill=median_interior_fill,
        median_center_fill=max(0.0, float(candidate.center_fill_ratio)),
    )
    if small_artifact:
        reasons.append("small_artifact")
        score = max(score, 0.76)

    if bool(candidate.touches_border) and smallest_axis < 0.78 and largest_axis <= 1.55:
        reasons.append("edge_clipped_cell")
        score = max(score, min(0.96, 0.76 + 0.18 * max(0.0, 1.0 - smallest_axis)))
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
            config=config,
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
        0.52 <= smallest_axis_ratio and largest_axis_ratio <= 1.58 and 0.34 <= float(area_ratio) <= 2.15
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
    broken_geometry_signal = slot_sized_geometry and (
        (candidate.child_count == 0 and candidate.solidity <= 0.70 and candidate.approx_vertices >= 8)
        or (candidate.solidity <= 0.66 and abs(float(candidate.extent) - 0.74) >= 0.22)
        or (
            candidate.approx_vertices >= 18
            and candidate.solidity <= 0.82
            and abs(float(candidate.extent) - 0.74) >= 0.16
        )
        or outline_damage_signal
        or pinched_geometry_signal
        or strict_shape_distortion_signal
        or warped_slot_signal
    )
    small_artifact_signal = _small_artifact_signal(
        candidate,
        median_width=median_width,
        median_height=median_height,
        median_area=median_area,
        median_fill=median_fill,
        median_interior_fill=median_interior_fill,
        median_center_fill=median_center_fill,
    )
    border_merged_signal = _border_merged_cell_signal(
        candidate,
        median_width=median_width,
        median_height=median_height,
        median_area=median_area,
    )
    conductor_residue_signal = not border_merged_signal and _conductor_residue_signal(
        candidate,
        median_width=median_width,
        median_height=median_height,
        median_area=median_area,
    )
    if conductor_residue_signal:
        residue_scale = max(largest_axis_ratio / 2.75, float(area_ratio) / 3.10)
        score = max(score, min(0.96, 0.80 + 0.12 * min(1.0, residue_scale)))
        reasons.append("conductor_residue")
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
        and largest_axis_ratio >= max(1.08, float(config.merged_size_ratio) - 0.16)
        and smallest_axis_ratio >= 0.42
        and float(area_ratio) >= max(0.80, float(config.merged_area_ratio) - 0.34)
        and (
            abs(aspect_ratio - median_aspect) > max(0.08, median_aspect * 0.10)
            or outline_imbalance >= 0.14
            or outline_min <= 0.58
            or candidate.child_count >= 2
            or candidate.inner_hole_ratio >= 0.035
        )
        and (
            largest_axis_ratio >= 1.16
            or float(area_ratio) >= 1.02
            or candidate.fill_ratio >= max(0.16, float(median_fill) + 0.035)
            or candidate.solidity <= 0.90
        )
    )
    bridge_connected_signal = (
        strict_merge_mode
        and largest_axis_ratio >= max(1.42, float(config.merged_size_ratio) + 0.10)
        and smallest_axis_ratio >= 0.28
        and float(bbox_area_ratio) >= 1.28
        and float(area_ratio) >= 0.22
        and (
            outline_mean >= 0.20
            or candidate.fill_ratio >= max(0.10, float(median_fill) * 0.60)
            or candidate.interior_fill_ratio >= max(0.045, normal_interior + 0.012)
            or candidate.approx_vertices >= 8
        )
        and (
            outline_imbalance >= 0.10
            or outline_min <= 0.64
            or abs(aspect_ratio - median_aspect) > max(0.10, median_aspect * 0.12)
            or candidate.solidity <= 0.92
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
        and not conductor_residue_signal
        and not (single_intact_cell_signal and not bridge_connected_signal)
    ):
        score = max(score, 0.86 if bridge_connected_signal else (0.78 if near_merged_signal else 0.88))
        reasons.append("merged_contour")
    if small_artifact_signal:
        area_ratio_clipped = min(1.0, max(0.0, float(area_ratio) / 0.45))
        score = max(score, min(0.86, 0.72 + area_ratio_clipped * 0.12))
        reasons.append("small_artifact")
    edge_filled_signal = (
        center_fill >= max(center_limit, 0.42)
        and candidate.interior_fill_ratio >= max(0.32, interior_limit * 0.86)
        and candidate.fill_ratio >= max(float(median_fill) + 0.14, fill_limit * 0.72)
    )
    if (
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
    ):
        edge_loss = max(0.0, 1.0 - smallest_axis_ratio)
        area_loss = max(0.0, 0.74 - float(area_ratio))
        score = max(score, min(0.94, 0.72 + 0.18 * edge_loss + 0.10 * area_loss))
        reasons.append("edge_clipped_cell")
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
) -> bool:
    width = float(candidate.bbox[2])
    height = float(candidate.bbox[3])
    width_ratio = width / max(1.0, float(median_width))
    height_ratio = height / max(1.0, float(median_height))
    area_ratio = float(candidate.area) / max(1.0, float(median_area))
    bbox_area_ratio = float(candidate.bbox_area) / max(1.0, float(median_width) * float(median_height))
    smallest_axis_ratio = min(width_ratio, height_ratio)
    largest_axis_ratio = max(width_ratio, height_ratio)
    if width <= 2.0 and height <= 2.0:
        return False
    if float(candidate.area) < max(5.0, min(14.0, float(median_area) * 0.035)):
        return False
    if candidate.child_count > 0:
        return False
    if smallest_axis_ratio < 0.06 or largest_axis_ratio > 1.32:
        return False
    line_like_fragment = (
        smallest_axis_ratio <= 0.20
        and largest_axis_ratio >= 0.38
        and float(candidate.area) <= max(18.0, float(median_area) * 0.42)
    )
    if line_like_fragment:
        return False
    if area_ratio > 0.62 or bbox_area_ratio > 0.92:
        return False
    normal_interior = max(0.0, float(median_interior_fill))
    normal_center = max(0.0, float(median_center_fill))
    outline_fragment = (
        largest_axis_ratio >= 0.42
        and smallest_axis_ratio <= 0.62
        and float(candidate.outline_mean_side_coverage) >= 0.12
        and float(candidate.interior_fill_ratio) <= max(0.14, normal_interior + 0.055)
        and float(candidate.center_fill_ratio) <= max(0.10, normal_center + 0.040)
        and float(candidate.fill_ratio) <= max(0.32, float(median_fill) + 0.12)
    )
    if outline_fragment:
        return False
    cell_sized_fragment = (
        smallest_axis_ratio >= 0.42
        and largest_axis_ratio >= 0.52
        and bbox_area_ratio >= 0.16
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
        smallest_axis_ratio >= 0.12 and largest_axis_ratio <= 0.58 and area_ratio <= 0.34 and bbox_area_ratio <= 0.36
    )
    elongated_or_blurred_debris = (
        largest_axis_ratio <= 0.96
        and area_ratio <= 0.42
        and bbox_area_ratio <= 0.48
        and (
            smallest_axis_ratio <= 0.30
            or float(candidate.solidity) <= 0.72
            or float(candidate.extent) <= 0.34
            or candidate.approx_vertices >= 8
        )
    )
    if not (compact_debris or elongated_or_blurred_debris):
        return False
    foreground_signal = (
        candidate.fill_ratio >= max(0.18, float(median_fill) + 0.04)
        or candidate.extent >= 0.42
        or candidate.solidity >= 0.62
        or elongated_or_blurred_debris
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
    if int(reason_counts.get("conductor_residue", 0)) > 0:
        return "conductor_residue"
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
) -> tuple[Any, ...]:
    return (
        GRID_DAMAGE_ALGORITHM_VERSION,
        str(frame_id or ""),
        _grid_cache_identity(path),
        config.cache_payload(),
        None if reference_profile is None else reference_profile.cache_payload(),
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
) -> GridFrameAnalysisResult | None:
    cache_path = _grid_damage_cache_path(
        _grid_damage_cache_key(path, frame_id=frame_id, config=config, reference_profile=reference_profile)
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
) -> None:
    global _grid_damage_cache_last_trim
    try:
        GRID_DAMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _grid_damage_cache_path(
            _grid_damage_cache_key(path, frame_id=frame_id, config=config, reference_profile=reference_profile)
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
    "build_grid_cell_reference_profile",
    "build_grid_cell_reference_profile_path",
    "compare_grid_cell_analyses",
    "configure_grid_worker_process",
    "detect_grid_cell_anomalies",
    "load_cached_grid_frame_result",
]
