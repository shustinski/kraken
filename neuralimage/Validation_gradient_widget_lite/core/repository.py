"""Repository and analytics helpers for the extended validation gradient widget."""
from __future__ import annotations

import hashlib
import os
import json
import math
import pickle
import shutil
from collections import OrderedDict
from time import perf_counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage

try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

try:
    import cv2
except Exception:
    cv2 = None

from .backend_constants import (
    ANALYSIS_CACHE_DIR,
    ANALYSIS_CACHE_VERSION,
    BCE_SCORE_CAP,
    DETAIL_CACHE_DIR,
    IMAGE_CACHE_SIZE,
    INTER_MODEL_POINT_SCORE_WEIGHTS,
    INTER_MODEL_POLYGON_SCORE_WEIGHTS,
    NATURAL_SPLIT_PATTERN,
    MASK_AGREEMENT_SCORE_WEIGHTS,
    MASK_SUPERVISED_SCORE_WEIGHTS,
    MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    MODEL_RISK_TOP_UNCERTAIN_FRACTION,
    MODEL_RISK_UNCERTAINTY_THRESHOLD,
    MODEL_RISK_WEIGHT_CLUSTER,
    MODEL_RISK_WEIGHT_FRACTION,
    MODEL_RISK_WEIGHT_TOP,
    POINT_AGREEMENT_SCORE_WEIGHTS,
    POINT_CONFIDENCE_NEIGHBOR_RADIUS,
    POINT_SUPPORT_THRESHOLD,
    POINT_SUPERVISED_SCORE_WEIGHTS,
    POLYGON_SUPPORT_THRESHOLD,
    POLYGON_CONFIDENCE_HYSTERESIS_FLOOR,
    POLYGON_CONFIDENCE_HYSTERESIS_LOW_RATIO,
    POLYGON_CONFIDENCE_COMPLETION_AXIS_RATIO,
    POLYGON_CONFIDENCE_COMPLETION_BRIDGE_RADIUS,
    POLYGON_CONFIDENCE_COMPLETION_LOW_RATIO,
    POLYGON_CONFIDENCE_COMPLETION_MAJOR_SCALE,
    POLYGON_CONFIDENCE_COMPLETION_WEAK_RATIO,
    POLYGON_CONFIDENCE_PREPROC_GAUSSIAN_SIGMA,
    POLYGON_CONFIDENCE_PREPROC_MEDIAN_RADIUS,
    POLYGON_CONFIDENCE_LOCAL_NORMALIZATION_RADIUS,
    POLYGON_CONFIDENCE_LOCAL_NORMALIZATION_STRENGTH,
    POLYGON_CONFIDENCE_ELONGATED_VERTICAL_RADIUS,
    POLYGON_CONFIDENCE_ELONGATED_HORIZONTAL_RADIUS,
    POLYGON_CONFIDENCE_ELONGATED_MIN_ASPECT_RATIO,
    POLYGON_CONFIDENCE_ELONGATED_MIN_AREA,
    POLYGON_CONFIDENCE_DOMINANT_MIN_AREA,
    POLYGON_CONFIDENCE_DOMINANT_MIN_MEAN_PROBABILITY,
    POLYGON_CONFIDENCE_DOMINANT_MIN_ASPECT_RATIO,
    POLYGON_CONFIDENCE_DOMINANT_MIN_EXTENT,
    POLYGON_CONFIDENCE_DOMINANT_LARGE_AREA,
    POLYGON_CONFIDENCE_DOMINANT_LOCK_RADIUS,
    POLYGON_CONFIDENCE_LARGE_POLYGON_LOW_SCALE,
    POLYGON_CONFIDENCE_LARGE_POLYGON_MIN_AREA,
    POLYGON_CONFIDENCE_LARGE_POLYGON_MIN_MAJOR_SPAN,
    POLYGON_CONFIDENCE_LARGE_POLYGON_MIN_EXTENT,
    POLYGON_CONFIDENCE_LARGE_POLYGON_MIN_ASPECT_RATIO,
    POLYGON_CONFIDENCE_LARGE_POLYGON_BAND_EXPAND,
    POLYGON_CONFIDENCE_LARGE_POLYGON_ROI_PADDING,
    POLYGON_CONFIDENCE_LARGE_POLYGON_SEED_LOW_SCALE,
    POLYGON_CONFIDENCE_LARGE_POLYGON_MAJOR_CLOSE_RADIUS,
    POLYGON_CONFIDENCE_LARGE_POLYGON_MINOR_CLOSE_RADIUS,
    POLYGON_CONFIDENCE_LARGE_POLYGON_BARRIER_DELTA,
    POLYGON_CONFIDENCE_LARGE_POLYGON_BARRIER_COVERAGE_MIN,
    POLYGON_CONFIDENCE_SMALL_LOW_SCALE,
    POLYGON_CONFIDENCE_SMALL_HIGH_SCALE,
    POLYGON_CONFIDENCE_SMALL_MAX_AREA,
    POLYGON_CONFIDENCE_ADAPTIVE_RADIUS,
    POLYGON_CONFIDENCE_ADAPTIVE_LOW_OFFSET,
    POLYGON_CONFIDENCE_ADAPTIVE_HIGH_OFFSET,
    POLYGON_CONFIDENCE_SEPARATION_CORE_MIN_AREA,
    POLYGON_CONFIDENCE_SEPARATION_ROI_PADDING,
    POLYGON_CONFIDENCE_SEPARATION_BOUNDARY_LOW_WEIGHT,
    POLYGON_CONFIDENCE_SEPARATION_BOUNDARY_CONTRAST_WEIGHT,
    POLYGON_CONFIDENCE_SEPARATION_BOUNDARY_UNCERTAINTY_WEIGHT,
    POLYGON_CONFIDENCE_SEPARATION_BARRIER_THRESHOLD,
    POLYGON_CONFIDENCE_SEPARATION_BARRIER_DILATE_RADIUS,
    POLYGON_CONFIDENCE_SEPARATION_BRIDGE_PROBABILITY_MAX,
    POLYGON_CONFIDENCE_SEPARATION_BRIDGE_BARRIER_THRESHOLD,
    POLYGON_CONFIDENCE_ENABLE_WATERSHED,
    POLYGON_CONFIDENCE_MERGE_DISTANCE,
    POLYGON_CONFIDENCE_MERGE_IOU_THRESHOLD,
    POLYGON_CONFIDENCE_PROPOSAL_MEAN_FLOOR,
    POLYGON_CONFIDENCE_PROPOSAL_MIN_AREA,
    POLYGON_CONFIDENCE_PROPOSAL_PEAK_FLOOR,
    POLYGON_CONFIDENCE_PROPOSAL_PEAK_RATIO,
    POLYGON_CONFIDENCE_SUMMARY_CORE,
    POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
    POLYGON_CONFIDENCE_WATERSHED_SEED_MIN_AREA,
    POLYGON_CONFIDENCE_HOLE_PROBABILITY_SCALE,
    POLYGON_CONFIDENCE_HOLE_PROBABILITY_MAX,
    POLYGON_CONFIDENCE_HOLE_MIN_AREA,
    POLYGON_CONFIDENCE_SPILL_LARGE_AREA_FRACTION,
    POLYGON_CONFIDENCE_SPILL_LARGE_EXTENT,
    POLYGON_CONFIDENCE_SPILL_LOW_TEXTURE_MAX,
    POLYGON_CONFIDENCE_SPILL_TRIM_DELTA,
    POLYGON_CONFIDENCE_SPILL_BOUNDARY_SEPARATION_MAX,
    POLYGON_CONFIDENCE_SPILL_PEAK_MARGIN_MAX,
    POLYGON_CONFIDENCE_SPILL_RIBBON_ASPECT_MIN,
    POLYGON_CONFIDENCE_SPILL_BORDER_COVERAGE_MIN,
    POLYGON_CONFIDENCE_SPILL_MEAN_PROBABILITY_MAX,
    POLYGON_CONFIDENCE_SPILL_CROSS_AXIS_MAX,
    POLYGON_CONFIDENCE_SPILL_PROMINENCE_MIN,
    POLYGON_CONFIDENCE_SPILL_STRONG_AXIS_COVERAGE_MIN,
    POLYGON_CONFIDENCE_SPILL_STRONG_AREA_FRACTION_MIN,
    POLYGON_CONFIDENCE_BOUNDARY_SNAP_MIN_ASPECT,
    POLYGON_CONFIDENCE_BOUNDARY_SNAP_PROFILE_QUANTILE,
    POLYGON_CONFIDENCE_BOUNDARY_SNAP_MIN_DROP,
    POLYGON_CONFIDENCE_BOUNDARY_SNAP_MIN_RETAINED_FRACTION,
    POLYGON_CONFIDENCE_VALLEY_MINOR_COVERAGE_MIN,
    EXPORT_SELECTION_MODE_COUNT,
    EXPORT_SELECTION_MODE_PERCENT,
    EXPORT_SELECTION_MODE_PERCENTILE,
)
from .domain import (
    BuildOptions,
    BuildResult,
    ComparisonMode,
    FolderSpec,
    FrameAnalysisSummary,
    FrameIdentity,
    FrameRecord,
    GeometryMode,
    LabeledModelMetrics,
    MaskAgreementMetrics,
    ModelAggregateScore,
    ModelDiagnosticMetrics,
    ModelSpec,
    PointAgreementMetrics,
    PointConfidenceMetrics,
    PointDiagnosticMetrics,
    PointLabeledMetrics,
    PointObjectConfidence,
    PolygonConfidenceDebugCandidate,
    PolygonConfidenceDebugData,
    PolygonConfidenceMetrics,
    PolygonConfidencePipelineConfig,
    PolygonObjectConfidence,
)

EPS = 1e-8




class BuildCancelledError(RuntimeError):
    """Signal cooperative cancellation during matrix build or analytics."""


@dataclass(frozen=True, slots=True)
class _PredictionPoint:
    """Compact point feature used by point geometry mode."""

    x: float
    y: float
    score: float
    peak_intensity: float
    local_contrast: float
    blob_score: float
    local_snr: float
    radius: float
    spot_area: float


@dataclass(frozen=True, slots=True)
class _PredictionRegionSummary:
    """Compact region summary used for geometry auto-selection."""

    area_fraction: float
    mean_area: float


@dataclass(frozen=True, slots=True)
class _PredictionView:
    """Self-contained prediction view for the extended widget."""

    model_name: str
    pred_gray: np.ndarray
    pred_bin: np.ndarray
    points: tuple[_PredictionPoint, ...]
    region_summary: _PredictionRegionSummary


@dataclass(frozen=True, slots=True)
class _OriginalFrameFeatures:
    """Scalar features extracted from the original grayscale frame."""

    mean_brightness: float
    contrast: float
    entropy: float
    blur_score: float
    noise_score: float
    edge_density: float
    local_peak_density: float
    dynamic_range: float
    saturation_ratio: float


def extract_frame_id(value: str) -> int:
    stem = Path(str(value)).stem
    token = stem.rsplit("_", 1)[-1]
    return int(token) if token.isdigit() else 0


def build_frame_identity(key: str, fallback_frame_id: int) -> FrameIdentity:
    path = Path(key)
    sequence_id = path.parent.as_posix() if path.parent.as_posix() not in {"", "."} else None
    try:
        frame_id = extract_frame_id(path.name)
    except Exception:
        frame_id = int(fallback_frame_id)
    return FrameIdentity(
        frame_id=frame_id,
        base_id=frame_id,
        tile_x=None,
        tile_y=None,
        source_key=key,
        sequence_id=sequence_id,
    )


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, float(value))))


def _normalize_ratio(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return 0.0
    return float(value / (1.0 + value))


def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
    valid = [(float(v), float(w)) for v, w in pairs if np.isfinite(v) and np.isfinite(w) and w > 0.0]
    if not valid:
        return 0.0
    numerator = sum(v * w for v, w in valid)
    denominator = sum(w for _v, w in valid)
    return float(numerator / max(EPS, denominator))


def _resolve_aux_path(key: str, index: dict[str, Path]) -> Path | None:
    exact = index.get(key)
    if exact is not None:
        return exact
    name = Path(key).name.lower()
    stem = Path(key).stem.lower()
    name_matches = [path for path in index.values() if path.name.lower() == name]
    if len(name_matches) == 1:
        return name_matches[0]
    stem_matches = [path for path in index.values() if path.stem.lower() == stem]
    if len(stem_matches) == 1:
        return stem_matches[0]
    return None


def natural_sort_key(value: str) -> tuple[object, ...]:
    """Split a string into digit and text chunks for natural sorting."""

    parts = NATURAL_SPLIT_PATTERN.split(str(value).lower())
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part)
    return tuple(key)


def iter_image_paths(folder: Path, *, recursive: bool, extensions: tuple[str, ...]) -> list[Path]:
    """Return image paths from one folder using natural sorting."""

    normalized_extensions = {str(ext).lower() for ext in extensions}
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    paths = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in normalized_extensions
    ]
    return sorted(paths, key=lambda item: natural_sort_key(item.as_posix()))


def build_folder_index(
    folder: Path,
    *,
    recursive: bool,
    extensions: tuple[str, ...],
    cancel_check=None,
) -> dict[str, Path]:
    """Index frame files in one folder by relative path."""

    index: dict[str, Path] = {}
    for image_path in iter_image_paths(folder, recursive=recursive, extensions=extensions):
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError("Build cancelled")
        index[image_path.relative_to(folder).as_posix()] = image_path
    return index


def _qimage_to_grayscale_array(image: QImage) -> np.ndarray:
    """Convert Qt image to contiguous grayscale ndarray."""

    grayscale = image.convertToFormat(QImage.Format.Format_Grayscale8)
    pointer = grayscale.bits()
    pointer.setsize(grayscale.height() * grayscale.bytesPerLine())
    buffer = np.frombuffer(pointer, dtype=np.uint8).reshape((grayscale.height(), grayscale.bytesPerLine()))
    return buffer[:, : grayscale.width()].copy()


def _grayscale_array_to_qimage(array: np.ndarray) -> QImage:
    """Convert contiguous grayscale ndarray to Qt image."""

    contiguous = np.ascontiguousarray(array.astype(np.uint8))
    height, width = contiguous.shape
    image = QImage(contiguous.data, width, height, contiguous.strides[0], QImage.Format.Format_Grayscale8)
    return image.copy()


def _load_grayscale_image_raw(path: Path) -> np.ndarray:
    image = QImage(str(path))
    if image.isNull():
        raise ValueError(f"Unable to decode image: {path}")
    return _qimage_to_grayscale_array(image)


def load_grayscale_image(path: Path) -> np.ndarray:
    """Load one image as grayscale ndarray."""

    return _load_grayscale_image_raw(Path(path))


def resize_grayscale_image(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize grayscale image using Qt fast transformation."""

    target_height, target_width = int(target_shape[0]), int(target_shape[1])
    source = np.asarray(array, dtype=np.uint8)
    if source.shape == (target_height, target_width):
        return source.copy()
    image = _grayscale_array_to_qimage(source)
    scaled = image.scaled(
        target_width,
        target_height,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )
    return _qimage_to_grayscale_array(scaled)


def _mean_filter_3x3(image: np.ndarray) -> np.ndarray:
    """Compute 3x3 mean filter without SciPy dependency."""

    image_f = np.asarray(image, dtype=np.float32)
    padded = np.pad(image_f, 1, mode="edge")
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0


def extract_original_frame_features(image: np.ndarray | None) -> _OriginalFrameFeatures | None:
    """Extract scalar quality features from original grayscale frame."""

    if image is None:
        return None
    grayscale = np.asarray(image, dtype=np.uint8)
    image_f = grayscale.astype(np.float32)
    normalized = image_f / 255.0
    mean_brightness = float(normalized.mean())
    contrast = float(normalized.std())

    histogram = np.bincount(grayscale.ravel(), minlength=256).astype(np.float64)
    histogram /= max(1.0, histogram.sum())
    non_zero = histogram > 0.0
    entropy = float(-(histogram[non_zero] * np.log2(histogram[non_zero])).sum())

    padded = np.pad(image_f, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    laplacian = (
        -4.0 * center
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
    )
    blur_score = float(laplacian.var() / (255.0 * 255.0))

    smoothed = _mean_filter_3x3(grayscale)
    noise_score = float(np.mean(np.abs(image_f - smoothed)) / 255.0)

    gradient_x = padded[1:-1, 2:] - padded[1:-1, :-2]
    gradient_y = padded[2:, 1:-1] - padded[:-2, 1:-1]
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    edge_density = float(np.mean(gradient_magnitude >= 20.0))

    neighbors = [
        padded[:-2, :-2],
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
        padded[2:, :-2],
        padded[2:, 1:-1],
        padded[2:, 2:],
    ]
    strict_peaks = np.logical_and.reduce([center > neighbor for neighbor in neighbors])
    local_peak_density = float(np.mean(strict_peaks))

    dynamic_range = float((float(grayscale.max()) - float(grayscale.min())) / 255.0)
    saturation_ratio = float(np.mean((grayscale <= 5) | (grayscale >= 250)))

    return _OriginalFrameFeatures(
        mean_brightness=mean_brightness,
        contrast=contrast,
        entropy=entropy,
        blur_score=blur_score,
        noise_score=noise_score,
        edge_density=edge_density,
        local_peak_density=local_peak_density,
        dynamic_range=dynamic_range,
        saturation_ratio=saturation_ratio,
    )


def _extract_patch(image: np.ndarray, center_x: float, center_y: float, radius: int = 2) -> np.ndarray:
    height, width = image.shape
    x = int(round(center_x))
    y = int(round(center_y))
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    return image[y0:y1, x0:x1]


def build_prediction_view_from_gray(model_name: str, pred_gray: np.ndarray, *, threshold: int = 128) -> _PredictionView:
    """Build lightweight prediction view from one grayscale output."""

    grayscale = np.asarray(pred_gray, dtype=np.uint8)
    pred_bin = grayscale >= int(threshold)
    labels, count = _label_components(pred_bin)
    points: list[_PredictionPoint] = []
    areas: list[int] = []
    for label in range(1, int(count) + 1):
        ys, xs = np.where(labels == label)
        if xs.size == 0:
            continue
        area = int(xs.size)
        areas.append(area)
        centroid_x = float(np.mean(xs, dtype=np.float64))
        centroid_y = float(np.mean(ys, dtype=np.float64))
        px = int(np.clip(round(centroid_x), 0, grayscale.shape[1] - 1))
        py = int(np.clip(round(centroid_y), 0, grayscale.shape[0] - 1))
        patch = _extract_patch(grayscale, centroid_x, centroid_y)
        peak = float(grayscale[py, px] / 255.0)
        patch_mean = float(patch.mean(dtype=np.float32) / 255.0) if patch.size else peak
        patch_std = float(patch.std(dtype=np.float32) / 255.0) if patch.size else 0.0
        local_contrast = float(max(0.0, peak - patch_mean))
        local_snr = float(local_contrast / max(1e-6, patch_std))
        radius = float(np.sqrt(area / np.pi))
        boundary = _boundary_mask(labels == label)
        perimeter = float(max(1, np.count_nonzero(boundary)))
        compactness = float(np.clip((4.0 * np.pi * area) / max(EPS, perimeter * perimeter), 0.0, 1.0))
        points.append(
            _PredictionPoint(
                x=centroid_x,
                y=centroid_y,
                score=peak,
                peak_intensity=peak * 255.0,
                local_contrast=local_contrast,
                blob_score=compactness,
                local_snr=local_snr,
                radius=radius,
                spot_area=float(area),
            )
        )
    region_summary = _PredictionRegionSummary(
        area_fraction=float(np.count_nonzero(pred_bin) / max(1, pred_bin.size)),
        mean_area=float(np.mean(np.asarray(areas, dtype=np.float64))) if areas else 0.0,
    )
    return _PredictionView(
        model_name=str(model_name),
        pred_gray=grayscale,
        pred_bin=np.asarray(pred_bin, dtype=bool),
        points=tuple(points),
        region_summary=region_summary,
    )


def _candidate_paths_for_known_key(folder: Path, key: str, extensions: tuple[str, ...]) -> list[Path]:
    relative = Path(key)
    candidates: list[Path] = []
    direct = folder / relative
    candidates.append(direct)
    stem_name = relative.stem
    parent = folder / relative.parent
    seen: set[str] = {str(direct).lower()}
    for extension in extensions:
        normalized = str(extension or '').strip()
        if not normalized:
            continue
        if not normalized.startswith('.'):
            normalized = f'.{normalized}'
        candidate = parent / f'{stem_name}{normalized}'
        marker = str(candidate).lower()
        if marker in seen:
            continue
        seen.add(marker)
        candidates.append(candidate)
    return candidates


def _resolve_model_path_from_known_key(folder: Path, key: str, extensions: tuple[str, ...]) -> Path | None:
    for candidate in _candidate_paths_for_known_key(folder, key, extensions):
        if candidate.is_file():
            return candidate
    return None


def _resolve_model_path_for_key(
    folder: Path,
    key: str,
    extensions: tuple[str, ...],
    fallback_index_cache: dict[str, dict[str, Path]],
    *,
    recursive: bool,
    cancel_check=None,
) -> Path | None:
    resolved = _resolve_model_path_from_known_key(folder, key, extensions)
    if resolved is not None:
        return resolved
    cache_key = str(folder.resolve())
    index = fallback_index_cache.get(cache_key)
    if index is None:
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError('Build cancelled')
        index = build_folder_index(folder, recursive=recursive, extensions=extensions, cancel_check=cancel_check)
        fallback_index_cache[cache_key] = index
    return _resolve_aux_path(key, index)


def _fit_shape_to_max_side(shape: tuple[int, int], max_side: int | None) -> tuple[int, int]:
    height, width = (int(shape[0]), int(shape[1]))
    limit = int(max_side or 0)
    if limit <= 0 or max(height, width) <= limit:
        return height, width
    scale = float(limit) / float(max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _image_signature(path: Path) -> tuple[str, int, int]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=IMAGE_CACHE_SIZE)
def _load_grayscale_image_cached(path_text: str, _mtime_ns: int, _size: int) -> np.ndarray:
    return load_grayscale_image(Path(path_text))


@lru_cache(maxsize=IMAGE_CACHE_SIZE)
def _load_resized_grayscale_image_cached(path_text: str, _mtime_ns: int, _size: int, target_shape: tuple[int, int]) -> np.ndarray:
    source = load_grayscale_image(Path(path_text))
    return resize_grayscale_image(source, target_shape)


def _clear_runtime_image_caches() -> None:
    """Release in-memory grayscale caches used during batch analytics."""

    _load_grayscale_image_cached.cache_clear()
    _load_resized_grayscale_image_cached.cache_clear()


def _load_optional_gray(path_text: str | None, target_shape: tuple[int, int] | None = None, max_side: int | None = None) -> np.ndarray | None:
    if not path_text:
        return None
    source_path = Path(path_text)
    source = _load_grayscale_image_cached(*_image_signature(source_path))
    limited_shape = _fit_shape_to_max_side(tuple(int(v) for v in source.shape), max_side)
    final_shape = tuple(int(v) for v in target_shape) if target_shape is not None else limited_shape
    if tuple(int(v) for v in source.shape) == final_shape:
        return np.asarray(source, dtype=np.uint8)
    return np.asarray(_load_resized_grayscale_image_cached(*_image_signature(source_path), final_shape), dtype=np.uint8)


def _path_signature(path_text: str | None) -> tuple[str, int, int] | None:
    if not path_text:
        return None
    return _image_signature(Path(path_text))


def _record_payload_cache_key(
    record: FrameRecord,
    model_specs: tuple[ModelSpec, ...],
    analysis_max_side: int | None,
    geometry_mode: GeometryMode,
    point_match_radius: float,
    boundary_radius: int,
    confidence_uncertainty_delta: float,
    point_confidence_radius: int,
    polygon_confidence_summary: str,
) -> str:
    payload = {
        "version": ANALYSIS_CACHE_VERSION,
        "record_key": record.key,
        "analysis_max_side": int(analysis_max_side or 0),
        "geometry_mode": geometry_mode.value,
        "point_match_radius": float(point_match_radius),
        "boundary_radius": int(boundary_radius),
        "confidence_uncertainty_delta": float(confidence_uncertainty_delta),
        "point_confidence_radius": int(point_confidence_radius),
        "polygon_confidence_summary": str(polygon_confidence_summary),
        "original": _path_signature(record.original_path),
        "gt": _path_signature(record.gt_path),
        "models": [
            {
                "model_id": spec.model_id,
                "threshold": float(spec.threshold),
                "mask": _path_signature(record.model_mask_paths.get(spec.model_id)),
                "prob": _path_signature(record.model_prob_paths.get(spec.model_id)),
            }
            for spec in model_specs
        ],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _record_payload_cache_path(cache_key: str) -> Path:
    ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_CACHE_DIR / f"{cache_key}.pickle"


def _load_cached_record_payload(cache_key: str) -> dict[str, object] | None:
    cache_path = _record_payload_cache_path(cache_key)
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _store_cached_record_payload(cache_key: str, payload: dict[str, object]) -> None:
    cache_path = _record_payload_cache_path(cache_key)
    tmp_path = cache_path.with_suffix(".tmp")
    try:
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


DETAIL_PAYLOAD_CACHE_SIZE = 32
_DETAIL_PAYLOAD_MEMORY_CACHE: OrderedDict[str, object] = OrderedDict()


def _detail_payload_cache_path(cache_key: str) -> Path:
    DETAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(cache_key).encode("utf-8")).hexdigest()
    return DETAIL_CACHE_DIR / f"{digest}.pickle"


def _load_cached_detail_payload(cache_key: str) -> object | None:
    payload = _DETAIL_PAYLOAD_MEMORY_CACHE.get(cache_key)
    if payload is not None:
        _DETAIL_PAYLOAD_MEMORY_CACHE.move_to_end(cache_key)
        return payload
    cache_path = _detail_payload_cache_path(cache_key)
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    _DETAIL_PAYLOAD_MEMORY_CACHE[cache_key] = payload
    _DETAIL_PAYLOAD_MEMORY_CACHE.move_to_end(cache_key)
    while len(_DETAIL_PAYLOAD_MEMORY_CACHE) > DETAIL_PAYLOAD_CACHE_SIZE:
        _DETAIL_PAYLOAD_MEMORY_CACHE.popitem(last=False)
    return payload


def _store_cached_detail_payload(cache_key: str, payload: object) -> None:
    _DETAIL_PAYLOAD_MEMORY_CACHE[cache_key] = payload
    _DETAIL_PAYLOAD_MEMORY_CACHE.move_to_end(cache_key)
    while len(_DETAIL_PAYLOAD_MEMORY_CACHE) > DETAIL_PAYLOAD_CACHE_SIZE:
        _DETAIL_PAYLOAD_MEMORY_CACHE.popitem(last=False)
    cache_path = _detail_payload_cache_path(cache_key)
    tmp_path = cache_path.with_suffix(".tmp")
    try:
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _detail_payload_cache_key(record: FrameRecord, build_result: BuildResult, max_side: int | None, model_id: str | None) -> str:
    return _record_payload_cache_key(
        record,
        build_result.model_specs,
        max_side,
        build_result.options.geometry_mode,
        float(build_result.options.point_match_radius),
        int(getattr(build_result.options, 'boundary_radius', 1) or 1),
        float(getattr(build_result.options, 'confidence_uncertainty_delta', MODEL_CONFIDENCE_UNCERTAIN_DELTA)),
        int(getattr(build_result.options, 'point_confidence_radius', POINT_CONFIDENCE_NEIGHBOR_RADIUS) or POINT_CONFIDENCE_NEIGHBOR_RADIUS),
        str(getattr(build_result.options, 'polygon_confidence_summary', POLYGON_CONFIDENCE_SUMMARY_WEIGHTED) or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED),
    ) + f"::detail::{str(model_id or '')}"


def _detail_base_payload_cache_key(record: FrameRecord, build_result: BuildResult, max_side: int | None) -> str:
    return _record_payload_cache_key(
        record,
        build_result.model_specs,
        max_side,
        build_result.options.geometry_mode,
        float(build_result.options.point_match_radius),
        int(getattr(build_result.options, 'boundary_radius', 1) or 1),
        float(getattr(build_result.options, 'confidence_uncertainty_delta', MODEL_CONFIDENCE_UNCERTAIN_DELTA)),
        int(getattr(build_result.options, 'point_confidence_radius', POINT_CONFIDENCE_NEIGHBOR_RADIUS) or POINT_CONFIDENCE_NEIGHBOR_RADIUS),
        str(getattr(build_result.options, 'polygon_confidence_summary', POLYGON_CONFIDENCE_SUMMARY_WEIGHTED) or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED),
    ) + "::detail_base"


def _detail_confidence_cache_key(record: FrameRecord, build_result: BuildResult, max_side: int | None, model_id: str | None) -> str:
    return _detail_base_payload_cache_key(record, build_result, max_side) + f"::confidence::{str(model_id or '')}"


def _detail_confidence_payload_ready(confidence_row, geometry_mode: str) -> bool:
    if confidence_row is None:
        return False
    if geometry_mode == GeometryMode.POINT.value:
        return hasattr(confidence_row, 'mean_point_confidence')
    return hasattr(confidence_row, 'mean_object_confidence')


def _with_selected_detail_payload(payload: dict[str, object], target_model_id: str | None) -> dict[str, object]:
    detail = dict(payload)
    probabilities = detail.get('model_probabilities') or {}
    masks = detail.get('model_masks') or {}
    selected_model_id = target_model_id if target_model_id in probabilities else (next(iter(probabilities.keys()), None))
    fallback_prob = np.zeros_like(next(iter(probabilities.values()))) if probabilities else np.zeros((1, 1), dtype=np.float32)
    selected_prob = probabilities.get(selected_model_id, fallback_prob)
    selected_mask = masks.get(selected_model_id, np.asarray(selected_prob >= 0.5, dtype=bool))
    detail['selected_model_id'] = selected_model_id
    detail['selected_prob'] = np.asarray(selected_prob, dtype=np.float32)
    detail['selected_mask'] = np.asarray(selected_mask, dtype=bool)
    return detail


def _mask_from_gray(gray: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    value = int(round(max(0.0, min(1.0, threshold)) * 255.0))
    return np.asarray(gray >= value, dtype=bool)


def _prob_from_gray(gray: np.ndarray) -> np.ndarray:
    return np.asarray(gray, dtype=np.float32) / 255.0


def _is_binary_like_probability(probability: np.ndarray, *, tolerance: float = 1e-4) -> bool:
    prob = np.asarray(probability, dtype=np.float32)
    if prob.size == 0:
        return False
    finite = prob[np.isfinite(prob)]
    if finite.size == 0:
        return False
    distance_to_binary = np.minimum(np.abs(finite), np.abs(finite - 1.0))
    return bool(np.max(distance_to_binary, initial=0.0) <= float(max(EPS, tolerance)))


def _internal_confidence_probability_map(probability: np.ndarray, *, support_mask: np.ndarray | None = None) -> np.ndarray:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    mask_bool = np.asarray(support_mask, dtype=bool) if support_mask is not None else np.asarray(prob >= 0.5, dtype=bool)
    if prob.size == 0 or not _is_binary_like_probability(prob):
        return prob
    if mask_bool.shape != prob.shape:
        mask_bool = np.asarray(prob >= 0.5, dtype=bool)
    if not np.any(mask_bool):
        return prob
    inside = _distance_transform(mask_bool)
    outside = _distance_transform(~mask_bool)
    signed_distance = np.asarray(inside - outside, dtype=np.float32)
    scale = float(np.percentile(np.abs(signed_distance), 95.0))
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.max(np.abs(signed_distance))) if signed_distance.size else 1.0
    scale = max(scale, 1.0)
    proxy = np.clip(0.5 + 0.5 * signed_distance / scale, 0.0, 1.0).astype(np.float32)
    proxy[mask_bool] = np.maximum(proxy[mask_bool], 0.5 + 0.5 * np.clip(inside[mask_bool] / scale, 0.0, 1.0))
    proxy[~mask_bool] = np.minimum(proxy[~mask_bool], 0.5 - 0.5 * np.clip(outside[~mask_bool] / scale, 0.0, 1.0))
    return np.clip(proxy, 0.0, 1.0).astype(np.float32)


def _confidence_map_from_probability(probability: np.ndarray) -> np.ndarray:
    probability_array = np.asarray(probability, dtype=np.float32)
    return np.clip(2.0 * np.abs(probability_array - 0.5), 0.0, 1.0).astype(np.float32)


def _support_weights_from_probability(probability: np.ndarray, support_threshold: float) -> np.ndarray:
    """Map probabilities into support weights used by confidence overlays and scores."""

    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    threshold = float(max(0.0, min(0.999, support_threshold)))
    weights = np.zeros_like(prob, dtype=np.float32)
    if prob.size == 0:
        return weights
    support_mask = prob >= threshold
    if not np.any(support_mask):
        return weights
    weights[support_mask] = (prob[support_mask] - threshold) / max(EPS, 1.0 - threshold)
    return np.clip(weights, 0.0, 1.0).astype(np.float32)


def _uncertainty_map_from_probability(probability: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - _confidence_map_from_probability(probability), 0.0, 1.0).astype(np.float32)


def _top_weighted_uncertainty_mean(uncertainty_values: np.ndarray, weight_values: np.ndarray, top_fraction: float) -> float:
    uncertainties = np.asarray(uncertainty_values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weight_values, dtype=np.float64).reshape(-1)
    if uncertainties.size == 0 or weights.size == 0:
        return 0.0
    valid_mask = np.isfinite(uncertainties) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid_mask):
        return 0.0
    uncertainties = uncertainties[valid_mask]
    weights = weights[valid_mask]
    count = max(1, int(math.ceil(float(uncertainties.size) * max(0.0, float(top_fraction)))))
    count = min(count, int(uncertainties.size))
    if count <= 0:
        return 0.0
    if count >= int(uncertainties.size):
        return float(np.sum(uncertainties * weights, dtype=np.float64) / max(EPS, float(np.sum(weights, dtype=np.float64))))
    top_indices = np.argpartition(-uncertainties, count - 1)[:count]
    top_uncertainty = uncertainties[top_indices]
    top_weights = weights[top_indices]
    return float(np.sum(top_uncertainty * top_weights, dtype=np.float64) / max(EPS, float(np.sum(top_weights, dtype=np.float64))))


def _largest_uncertain_region_fraction(
    uncertainty: np.ndarray,
    support_mask: np.ndarray,
    *,
    uncertainty_threshold: float,
) -> float:
    support = np.asarray(support_mask, dtype=bool)
    if support.size == 0 or not np.any(support):
        return 0.0
    uncertainty_map = np.asarray(uncertainty, dtype=np.float32)
    uncertain_support = support & np.isfinite(uncertainty_map) & (uncertainty_map > float(uncertainty_threshold))
    if not np.any(uncertain_support):
        return 0.0
    labels, component_count = _label_components(uncertain_support)
    if component_count <= 0:
        return 0.0
    largest_area = 0
    for label_id in range(1, int(component_count) + 1):
        largest_area = max(largest_area, int(np.count_nonzero(labels == label_id)))
    support_pixels = int(np.count_nonzero(support))
    return float(largest_area / max(1, support_pixels))


def _frame_uncertainty_components_from_probability(
    probability: np.ndarray,
    *,
    support_threshold: float,
    uncertainty_threshold: float = MODEL_RISK_UNCERTAINTY_THRESHOLD,
    top_fraction: float = MODEL_RISK_TOP_UNCERTAIN_FRACTION,
    risk_weight_fraction: float = MODEL_RISK_WEIGHT_FRACTION,
    risk_weight_top: float = MODEL_RISK_WEIGHT_TOP,
    risk_weight_cluster: float = MODEL_RISK_WEIGHT_CLUSTER,
) -> tuple[float, float, float, float]:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    support_weights = _support_weights_from_probability(prob, support_threshold)
    support_mask = np.asarray(support_weights > 0.0, dtype=bool)
    if prob.ndim != 2 or prob.size == 0 or not np.any(support_mask):
        return 0.0, 0.0, 0.0, 0.0
    uncertainty = _uncertainty_map_from_probability(prob)
    weights = np.asarray(support_weights[support_mask], dtype=np.float64)
    uncertainty_values = np.asarray(uncertainty[support_mask], dtype=np.float64)
    weight_total = float(np.sum(weights, dtype=np.float64))
    if weight_total <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    uncertain_fraction = float(
        np.sum(weights * np.asarray(uncertainty_values > float(uncertainty_threshold), dtype=np.float64), dtype=np.float64)
        / max(EPS, weight_total)
    )
    top_uncertainty_mean = _top_weighted_uncertainty_mean(uncertainty_values, weights, float(top_fraction))
    largest_region_fraction = _largest_uncertain_region_fraction(
        uncertainty,
        support_mask,
        uncertainty_threshold=float(uncertainty_threshold),
    )
    denominator = max(EPS, float(risk_weight_fraction + risk_weight_top + risk_weight_cluster))
    score = float(
        (
            float(risk_weight_fraction) * uncertain_fraction
            + float(risk_weight_top) * top_uncertainty_mean
            + float(risk_weight_cluster) * largest_region_fraction
        )
        / denominator
    )
    return float(np.clip(score, 0.0, 1.0)), uncertain_fraction, top_uncertainty_mean, largest_region_fraction


def _point_uncertainty_cluster_fraction(
    coordinates: tuple[tuple[float, float, float], ...],
    uncertainty_values: np.ndarray,
    support_mask: np.ndarray,
    *,
    uncertainty_threshold: float,
) -> float:
    support = np.asarray(support_mask, dtype=bool).reshape(-1)
    uncertainties = np.asarray(uncertainty_values, dtype=np.float64).reshape(-1)
    if not coordinates or support.size == 0 or not np.any(support):
        return 0.0
    uncertain_indices = [
        index
        for index, (is_supported, uncertainty) in enumerate(zip(support.tolist(), uncertainties.tolist()))
        if is_supported and np.isfinite(uncertainty) and float(uncertainty) > float(uncertainty_threshold)
    ]
    if not uncertain_indices:
        return 0.0
    support_count = int(np.count_nonzero(support))
    adjacency: dict[int, list[int]] = {index: [] for index in uncertain_indices}
    for offset, left_index in enumerate(uncertain_indices):
        left_x, left_y, left_radius = coordinates[left_index]
        left_radius = max(1.0, float(left_radius))
        for right_index in uncertain_indices[offset + 1:]:
            right_x, right_y, right_radius = coordinates[right_index]
            right_radius = max(1.0, float(right_radius))
            max_distance = left_radius + right_radius + 1.0
            if math.hypot(float(left_x) - float(right_x), float(left_y) - float(right_y)) <= max_distance:
                adjacency[left_index].append(right_index)
                adjacency[right_index].append(left_index)
    visited: set[int] = set()
    largest_cluster = 0
    for start_index in uncertain_indices:
        if start_index in visited:
            continue
        stack = [start_index]
        visited.add(start_index)
        cluster_size = 0
        while stack:
            current = stack.pop()
            cluster_size += 1
            for neighbor in adjacency.get(current, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        largest_cluster = max(largest_cluster, cluster_size)
    return float(largest_cluster / max(1, support_count))


def _frame_uncertainty_components_from_points(
    point_probabilities: np.ndarray,
    point_coordinates: tuple[tuple[float, float, float], ...],
    *,
    support_threshold: float = POINT_SUPPORT_THRESHOLD,
    uncertainty_threshold: float = MODEL_RISK_UNCERTAINTY_THRESHOLD,
    top_fraction: float = MODEL_RISK_TOP_UNCERTAIN_FRACTION,
    risk_weight_fraction: float = MODEL_RISK_WEIGHT_FRACTION,
    risk_weight_top: float = MODEL_RISK_WEIGHT_TOP,
    risk_weight_cluster: float = MODEL_RISK_WEIGHT_CLUSTER,
) -> tuple[float, float, float, float]:
    probabilities = np.clip(np.asarray(point_probabilities, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if probabilities.size == 0 or not point_coordinates:
        return 0.0, 0.0, 0.0, 0.0
    support_weights = np.zeros_like(probabilities, dtype=np.float32)
    support_mask = probabilities >= float(support_threshold)
    if np.any(support_mask):
        support_weights[support_mask] = (probabilities[support_mask] - float(support_threshold)) / max(EPS, 1.0 - float(support_threshold))
    support_weights = np.clip(support_weights, 0.0, 1.0).astype(np.float32)
    if not np.any(support_weights > 0.0):
        return 0.0, 0.0, 0.0, 0.0
    uncertainty = 1.0 - np.clip(2.0 * np.abs(probabilities - 0.5), 0.0, 1.0)
    positive_mask = support_weights > 0.0
    weights = np.asarray(support_weights[positive_mask], dtype=np.float64)
    uncertainty_values = np.asarray(uncertainty[positive_mask], dtype=np.float64)
    weight_total = float(np.sum(weights, dtype=np.float64))
    if weight_total <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    uncertain_fraction = float(
        np.sum(weights * np.asarray(uncertainty_values > float(uncertainty_threshold), dtype=np.float64), dtype=np.float64)
        / max(EPS, weight_total)
    )
    top_uncertainty_mean = _top_weighted_uncertainty_mean(uncertainty_values, weights, float(top_fraction))
    largest_region_fraction = _point_uncertainty_cluster_fraction(
        point_coordinates,
        uncertainty,
        positive_mask,
        uncertainty_threshold=float(uncertainty_threshold),
    )
    denominator = max(EPS, float(risk_weight_fraction + risk_weight_top + risk_weight_cluster))
    score = float(
        (
            float(risk_weight_fraction) * uncertain_fraction
            + float(risk_weight_top) * top_uncertainty_mean
            + float(risk_weight_cluster) * largest_region_fraction
        )
        / denominator
    )
    return float(np.clip(score, 0.0, 1.0)), uncertain_fraction, top_uncertainty_mean, largest_region_fraction


def _probability_gradient(probability: np.ndarray) -> np.ndarray:
    """Compute simple central-difference gradient magnitude for a probability map."""

    prob = np.asarray(probability, dtype=np.float32)
    if prob.ndim != 2 or prob.size == 0:
        return np.zeros_like(prob, dtype=np.float32)
    padded = np.pad(prob, 1, mode='edge')
    grad_x = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    grad_y = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    return np.hypot(grad_x, grad_y).astype(np.float32)


def _polygon_frame_confidence(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    uncertainty_delta: float = MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    summary_metric: str = POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
    include_debug: bool = False,
) -> PolygonConfidenceMetrics:
    """Compute lightweight frame-level polygon confidence without object-aware geometry refinement."""

    strong = np.asarray(mask, dtype=bool)
    prob = _internal_confidence_probability_map(probability, support_mask=strong)
    if strong.shape != prob.shape:
        strong = np.asarray(strong, dtype=bool)
        if strong.shape != prob.shape:
            strong = np.zeros_like(prob, dtype=bool)
    confidence = _confidence_map_from_probability(prob)
    uncertainty = _uncertainty_map_from_probability(prob)
    support_weights = _support_weights_from_probability(prob, POLYGON_SUPPORT_THRESHOLD)
    support_mask = support_weights > 0.0
    focus_confidence = np.asarray(confidence[support_mask], dtype=np.float32)
    focus_probability = np.asarray(prob[support_mask], dtype=np.float32)
    uncertain_mask = np.abs(prob - 0.5) <= float(max(0.0, uncertainty_delta))
    uncertain_fraction = float(np.mean(uncertain_mask[support_mask], dtype=np.float64)) if np.any(support_mask) else 0.0
    boundary_mask = _boundary_mask(strong) if np.any(strong) else _boundary_mask(support_mask)
    boundary_uncertainty = float(np.mean(uncertainty[boundary_mask], dtype=np.float64)) if np.any(boundary_mask) else 0.0
    core_threshold = max(0.65, float(np.quantile(focus_probability, 0.65)) if focus_probability.size > 0 else 0.65)
    core_mask = support_mask & (prob >= core_threshold)
    core_confidence = float(np.mean(confidence[core_mask], dtype=np.float64)) if np.any(core_mask) else 0.0
    labels, polygon_count = _label_components(strong if np.any(strong) else support_mask)
    weight_total = float(np.sum(np.asarray(support_weights, dtype=np.float64), dtype=np.float64))
    mean_confidence = float(np.sum(np.asarray(support_weights * confidence, dtype=np.float64), dtype=np.float64) / max(EPS, weight_total)) if weight_total > 0.0 else 0.0
    mean_probability = float(np.sum(np.asarray(support_weights * prob, dtype=np.float64), dtype=np.float64) / max(EPS, weight_total)) if weight_total > 0.0 else 0.0
    frame_uncertainty_score, uncertain_support_fraction, top_uncertainty_mean, largest_uncertain_region_fraction = _frame_uncertainty_components_from_probability(
        prob,
        support_threshold=POLYGON_SUPPORT_THRESHOLD,
    )
    area_fraction = float(np.count_nonzero(support_mask) / max(1, support_mask.size))
    debug_data = None
    if include_debug:
        debug_data = PolygonConfidenceDebugData(
            preprocessed_probability=np.asarray(prob, dtype=np.float32),
            low_mask=np.asarray(support_mask, dtype=bool),
            high_mask=np.asarray(strong, dtype=bool),
            merged_mask=np.asarray(strong if np.any(strong) else support_mask, dtype=bool),
            object_labels=np.asarray(labels, dtype=np.int32),
            candidate_rows=(),
            timings_ms={},
        )
    return PolygonConfidenceMetrics(
        frame_uncertainty_score=frame_uncertainty_score,
        uncertain_support_fraction=uncertain_support_fraction,
        top_uncertainty_mean=top_uncertainty_mean,
        largest_uncertain_region_fraction=largest_uncertain_region_fraction,
        mean_object_confidence=mean_confidence,
        mean_core_confidence=core_confidence,
        mean_boundary_uncertainty=boundary_uncertainty,
        mean_weighted_confidence=mean_confidence,
        mean_object_probability=mean_probability,
        uncertain_fraction=uncertain_fraction,
        mean_transition_width=0.0,
        object_area_fraction=area_fraction,
        polygon_count=int(polygon_count),
        summary_metric=str(summary_metric),
        objects=tuple(),
        debug_data=debug_data,
    )


def _polygon_confidence_config() -> PolygonConfidencePipelineConfig:
    """Return the default configuration for polygon confidence extraction."""

    return PolygonConfidencePipelineConfig(
        gaussian_sigma=float(POLYGON_CONFIDENCE_PREPROC_GAUSSIAN_SIGMA),
        median_radius=int(POLYGON_CONFIDENCE_PREPROC_MEDIAN_RADIUS),
        local_normalization_radius=int(POLYGON_CONFIDENCE_LOCAL_NORMALIZATION_RADIUS),
        local_normalization_strength=float(POLYGON_CONFIDENCE_LOCAL_NORMALIZATION_STRENGTH),
        hysteresis_low_ratio=float(POLYGON_CONFIDENCE_HYSTERESIS_LOW_RATIO),
        hysteresis_low_floor=float(POLYGON_CONFIDENCE_HYSTERESIS_FLOOR),
        elongated_vertical_radius=int(POLYGON_CONFIDENCE_ELONGATED_VERTICAL_RADIUS),
        elongated_horizontal_radius=int(POLYGON_CONFIDENCE_ELONGATED_HORIZONTAL_RADIUS),
        elongated_min_aspect_ratio=float(POLYGON_CONFIDENCE_ELONGATED_MIN_ASPECT_RATIO),
        elongated_min_area=int(POLYGON_CONFIDENCE_ELONGATED_MIN_AREA),
        dominant_min_area=int(POLYGON_CONFIDENCE_DOMINANT_MIN_AREA),
        dominant_min_mean_probability=float(POLYGON_CONFIDENCE_DOMINANT_MIN_MEAN_PROBABILITY),
        dominant_min_aspect_ratio=float(POLYGON_CONFIDENCE_DOMINANT_MIN_ASPECT_RATIO),
        dominant_min_extent=float(POLYGON_CONFIDENCE_DOMINANT_MIN_EXTENT),
        dominant_large_area=int(POLYGON_CONFIDENCE_DOMINANT_LARGE_AREA),
        dominant_lock_radius=int(POLYGON_CONFIDENCE_DOMINANT_LOCK_RADIUS),
        large_polygon_low_scale=float(POLYGON_CONFIDENCE_LARGE_POLYGON_LOW_SCALE),
        large_polygon_min_area=int(POLYGON_CONFIDENCE_LARGE_POLYGON_MIN_AREA),
        large_polygon_min_major_span=int(POLYGON_CONFIDENCE_LARGE_POLYGON_MIN_MAJOR_SPAN),
        large_polygon_min_extent=float(POLYGON_CONFIDENCE_LARGE_POLYGON_MIN_EXTENT),
        large_polygon_min_aspect_ratio=float(POLYGON_CONFIDENCE_LARGE_POLYGON_MIN_ASPECT_RATIO),
        large_polygon_band_expand=int(POLYGON_CONFIDENCE_LARGE_POLYGON_BAND_EXPAND),
        large_polygon_roi_padding=int(POLYGON_CONFIDENCE_LARGE_POLYGON_ROI_PADDING),
        large_polygon_seed_low_scale=float(POLYGON_CONFIDENCE_LARGE_POLYGON_SEED_LOW_SCALE),
        large_polygon_major_close_radius=int(POLYGON_CONFIDENCE_LARGE_POLYGON_MAJOR_CLOSE_RADIUS),
        large_polygon_minor_close_radius=int(POLYGON_CONFIDENCE_LARGE_POLYGON_MINOR_CLOSE_RADIUS),
        large_polygon_barrier_delta=float(POLYGON_CONFIDENCE_LARGE_POLYGON_BARRIER_DELTA),
        large_polygon_barrier_coverage_min=float(POLYGON_CONFIDENCE_LARGE_POLYGON_BARRIER_COVERAGE_MIN),
        small_low_scale=float(POLYGON_CONFIDENCE_SMALL_LOW_SCALE),
        small_high_scale=float(POLYGON_CONFIDENCE_SMALL_HIGH_SCALE),
        small_min_area=max(1, int(POLYGON_CONFIDENCE_PROPOSAL_MIN_AREA)),
        small_max_area=int(POLYGON_CONFIDENCE_SMALL_MAX_AREA),
        small_peak_floor=float(POLYGON_CONFIDENCE_PROPOSAL_PEAK_FLOOR),
        small_mean_floor=float(POLYGON_CONFIDENCE_PROPOSAL_MEAN_FLOOR),
        adaptive_radius=int(POLYGON_CONFIDENCE_ADAPTIVE_RADIUS),
        adaptive_low_offset=float(POLYGON_CONFIDENCE_ADAPTIVE_LOW_OFFSET),
        adaptive_high_offset=float(POLYGON_CONFIDENCE_ADAPTIVE_HIGH_OFFSET),
        separation_core_min_area=int(POLYGON_CONFIDENCE_SEPARATION_CORE_MIN_AREA),
        separation_roi_padding=int(POLYGON_CONFIDENCE_SEPARATION_ROI_PADDING),
        separation_boundary_low_weight=float(POLYGON_CONFIDENCE_SEPARATION_BOUNDARY_LOW_WEIGHT),
        separation_boundary_contrast_weight=float(POLYGON_CONFIDENCE_SEPARATION_BOUNDARY_CONTRAST_WEIGHT),
        separation_boundary_uncertainty_weight=float(POLYGON_CONFIDENCE_SEPARATION_BOUNDARY_UNCERTAINTY_WEIGHT),
        separation_barrier_threshold=float(POLYGON_CONFIDENCE_SEPARATION_BARRIER_THRESHOLD),
        separation_barrier_dilate_radius=int(POLYGON_CONFIDENCE_SEPARATION_BARRIER_DILATE_RADIUS),
        separation_bridge_probability_max=float(POLYGON_CONFIDENCE_SEPARATION_BRIDGE_PROBABILITY_MAX),
        separation_bridge_barrier_threshold=float(POLYGON_CONFIDENCE_SEPARATION_BRIDGE_BARRIER_THRESHOLD),
        merge_iou_threshold=float(POLYGON_CONFIDENCE_MERGE_IOU_THRESHOLD),
        merge_distance=int(POLYGON_CONFIDENCE_MERGE_DISTANCE),
        enable_watershed=bool(POLYGON_CONFIDENCE_ENABLE_WATERSHED),
        watershed_seed_min_area=int(POLYGON_CONFIDENCE_WATERSHED_SEED_MIN_AREA),
        hole_probability_scale=float(POLYGON_CONFIDENCE_HOLE_PROBABILITY_SCALE),
        hole_probability_max=float(POLYGON_CONFIDENCE_HOLE_PROBABILITY_MAX),
        hole_min_area=int(POLYGON_CONFIDENCE_HOLE_MIN_AREA),
        spill_large_area_fraction=float(POLYGON_CONFIDENCE_SPILL_LARGE_AREA_FRACTION),
        spill_large_extent=float(POLYGON_CONFIDENCE_SPILL_LARGE_EXTENT),
        spill_low_texture_max=float(POLYGON_CONFIDENCE_SPILL_LOW_TEXTURE_MAX),
        spill_trim_delta=float(POLYGON_CONFIDENCE_SPILL_TRIM_DELTA),
        spill_boundary_separation_max=float(POLYGON_CONFIDENCE_SPILL_BOUNDARY_SEPARATION_MAX),
        spill_peak_margin_max=float(POLYGON_CONFIDENCE_SPILL_PEAK_MARGIN_MAX),
        spill_ribbon_aspect_min=float(POLYGON_CONFIDENCE_SPILL_RIBBON_ASPECT_MIN),
        spill_border_coverage_min=float(POLYGON_CONFIDENCE_SPILL_BORDER_COVERAGE_MIN),
        spill_mean_probability_max=float(POLYGON_CONFIDENCE_SPILL_MEAN_PROBABILITY_MAX),
        spill_cross_axis_max=float(POLYGON_CONFIDENCE_SPILL_CROSS_AXIS_MAX),
        spill_prominence_min=float(POLYGON_CONFIDENCE_SPILL_PROMINENCE_MIN),
        spill_strong_axis_coverage_min=float(POLYGON_CONFIDENCE_SPILL_STRONG_AXIS_COVERAGE_MIN),
        spill_strong_area_fraction_min=float(POLYGON_CONFIDENCE_SPILL_STRONG_AREA_FRACTION_MIN),
        boundary_snap_min_aspect=float(POLYGON_CONFIDENCE_BOUNDARY_SNAP_MIN_ASPECT),
        boundary_snap_profile_quantile=float(POLYGON_CONFIDENCE_BOUNDARY_SNAP_PROFILE_QUANTILE),
        boundary_snap_min_drop=float(POLYGON_CONFIDENCE_BOUNDARY_SNAP_MIN_DROP),
        boundary_snap_min_retained_fraction=float(POLYGON_CONFIDENCE_BOUNDARY_SNAP_MIN_RETAINED_FRACTION),
        valley_minor_coverage_min=float(POLYGON_CONFIDENCE_VALLEY_MINOR_COVERAGE_MIN),
    )


@dataclass(slots=True)
class _PolygonConfidenceCandidate:
    """Internal candidate mask proposed by one confidence extraction branch."""

    mask: np.ndarray
    source_branches: tuple[str, ...]
    area: int
    bbox: tuple[int, int, int, int]
    aspect_ratio: float
    elongation: float
    extent: float
    peak_probability: float
    mean_probability: float


def _polygon_confidence_weak_threshold(strong_threshold: float, config: PolygonConfidencePipelineConfig | None = None) -> float:
    cfg = config or _polygon_confidence_config()
    strong = float(max(0.0, min(1.0, strong_threshold)))
    weak = max(float(cfg.hysteresis_low_floor), strong * float(cfg.hysteresis_low_ratio))
    return float(min(strong, weak))


def _polygon_confidence_completion_threshold(strong_threshold: float, weak_threshold: float) -> float:
    strong = float(max(0.0, min(1.0, strong_threshold)))
    weak = float(max(0.0, min(1.0, weak_threshold)))
    completion = max(
        POLYGON_CONFIDENCE_HYSTERESIS_FLOOR,
        min(weak * POLYGON_CONFIDENCE_COMPLETION_WEAK_RATIO, strong * POLYGON_CONFIDENCE_COMPLETION_LOW_RATIO),
    )
    return float(min(strong, completion))


def _normalize_probability_map(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0).astype(np.float32)


def _smooth_probability_map(probability: np.ndarray, config: PolygonConfidencePipelineConfig) -> np.ndarray:
    prob = _normalize_probability_map(probability)
    if ndi is None:
        return prob
    result = prob
    median_radius = max(0, int(config.median_radius))
    if median_radius > 0:
        size = 2 * median_radius + 1
        result = np.asarray(ndi.median_filter(result, size=size, mode='nearest'), dtype=np.float32)
    sigma = max(0.0, float(config.gaussian_sigma))
    if sigma > EPS:
        result = np.asarray(ndi.gaussian_filter(result, sigma=sigma, mode='nearest'), dtype=np.float32)
    return _normalize_probability_map(result)


def _locally_normalized_probability_map(probability: np.ndarray, config: PolygonConfidencePipelineConfig) -> np.ndarray:
    prob = _normalize_probability_map(probability)
    radius = max(1, int(config.local_normalization_radius))
    mean_map = _local_mean_map(prob, radius=radius)
    sq_mean_map = _local_mean_map(np.square(prob, dtype=np.float32), radius=radius)
    variance_map = np.clip(sq_mean_map - np.square(mean_map, dtype=np.float32), 0.0, None)
    std_map = np.sqrt(variance_map + EPS, dtype=np.float32)
    normalized = np.clip(0.5 + 0.25 * ((prob - mean_map) / np.maximum(std_map, 0.05)), 0.0, 1.0)
    strength = float(max(0.0, min(1.0, config.local_normalization_strength)))
    enhanced = np.clip(prob + strength * (normalized - 0.5), 0.0, 1.0)
    return np.maximum(prob, np.asarray(enhanced, dtype=np.float32)).astype(np.float32)


def _preprocess_polygon_probability(probability: np.ndarray, config: PolygonConfidencePipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    prob = _normalize_probability_map(probability)
    smoothed = np.maximum(prob, _smooth_probability_map(prob, config))
    locally_normalized = _locally_normalized_probability_map(prob, config)
    preprocessed = np.maximum(smoothed, locally_normalized).astype(np.float32)
    return _normalize_probability_map(preprocessed), _normalize_probability_map(locally_normalized)


def _binary_erode_rect(mask: np.ndarray, radius_y: int = 1, radius_x: int = 1) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    ry = max(0, int(radius_y))
    rx = max(0, int(radius_x))
    if ry <= 0 and rx <= 0:
        return mask_bool.copy()
    if ndi is not None:
        structure = np.ones((2 * ry + 1, 2 * rx + 1), dtype=bool)
        return np.asarray(ndi.binary_erosion(mask_bool, structure=structure), dtype=bool)
    padded = np.pad(mask_bool, ((ry, ry), (rx, rx)), mode='constant', constant_values=False)
    result = np.ones_like(mask_bool, dtype=bool)
    for row_offset in range(2 * ry + 1):
        for column_offset in range(2 * rx + 1):
            result &= padded[row_offset:row_offset + mask_bool.shape[0], column_offset:column_offset + mask_bool.shape[1]]
    return result


def _binary_close_rect(mask: np.ndarray, radius_y: int = 1, radius_x: int = 1) -> np.ndarray:
    dilated = _binary_dilate_rect(mask, radius_y, radius_x)
    return _binary_erode_rect(dilated, radius_y, radius_x)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size == 0 or xs.size == 0:
        return 0, 0, 0, 0
    y0 = int(np.min(ys))
    y1 = int(np.max(ys)) + 1
    x0 = int(np.min(xs))
    x1 = int(np.max(xs)) + 1
    return x0, y0, x1 - x0, y1 - y0


def _mask_geometry(mask: np.ndarray) -> tuple[int, tuple[int, int, int, int], float, float, float]:
    mask_bool = np.asarray(mask, dtype=bool)
    area = int(np.count_nonzero(mask_bool))
    bbox = _mask_bbox(mask_bool)
    width = max(1, int(bbox[2]))
    height = max(1, int(bbox[3]))
    aspect_ratio = float(max(width, height) / max(1, min(width, height)))
    extent = float(area / max(1, width * height))
    elongation = aspect_ratio
    ys, xs = np.nonzero(mask_bool)
    if ys.size >= 3:
        coords = np.column_stack((ys.astype(np.float32), xs.astype(np.float32)))
        centered = coords - np.mean(coords, axis=0, keepdims=True)
        covariance = np.matmul(centered.T, centered) / max(1.0, float(coords.shape[0] - 1))
        eigvals = np.linalg.eigvalsh(covariance)
        if eigvals.size >= 2:
            major = float(max(eigvals[-1], EPS))
            minor = float(max(eigvals[0], EPS))
            elongation = float(np.sqrt(major / minor))
    return area, bbox, aspect_ratio, elongation, extent


def _make_polygon_candidate(
    probability: np.ndarray,
    mask: np.ndarray,
    source_branches: tuple[str, ...],
    *,
    roi_bbox: tuple[int, int, int, int] | None = None,
) -> _PolygonConfidenceCandidate | None:
    """Build one candidate while optionally restricting geometry extraction to a known ROI."""

    mask_bool = np.asarray(mask, dtype=bool)
    if roi_bbox is not None:
        roi_x0, roi_y0, roi_width, roi_height = roi_bbox
        roi_x0 = max(0, int(roi_x0))
        roi_y0 = max(0, int(roi_y0))
        roi_x1 = min(mask_bool.shape[1], roi_x0 + max(0, int(roi_width)))
        roi_y1 = min(mask_bool.shape[0], roi_y0 + max(0, int(roi_height)))
        roi_mask = mask_bool[roi_y0:roi_y1, roi_x0:roi_x1]
    else:
        roi_x0 = 0
        roi_y0 = 0
        roi_mask = mask_bool

    area = int(np.count_nonzero(roi_mask))
    if area <= 0:
        return None

    ys, xs = np.nonzero(roi_mask)
    if ys.size == 0 or xs.size == 0:
        return None
    bbox_x0 = roi_x0 + int(np.min(xs))
    bbox_y0 = roi_y0 + int(np.min(ys))
    bbox_x1 = roi_x0 + int(np.max(xs)) + 1
    bbox_y1 = roi_y0 + int(np.max(ys)) + 1
    bbox = (bbox_x0, bbox_y0, bbox_x1 - bbox_x0, bbox_y1 - bbox_y0)
    width = max(1, int(bbox[2]))
    height = max(1, int(bbox[3]))
    aspect_ratio = float(max(width, height) / max(1, min(width, height)))
    extent = float(area / max(1, width * height))
    elongation = aspect_ratio
    if ys.size >= 3:
        coords = np.column_stack((ys.astype(np.float32), xs.astype(np.float32)))
        centered = coords - np.mean(coords, axis=0, keepdims=True)
        covariance = np.matmul(centered.T, centered) / max(1.0, float(coords.shape[0] - 1))
        eigvals = np.linalg.eigvalsh(covariance)
        if eigvals.size >= 2:
            major = float(max(eigvals[-1], EPS))
            minor = float(max(eigvals[0], EPS))
            elongation = float(np.sqrt(major / minor))

    object_prob = np.asarray(probability[bbox_y0:bbox_y1, bbox_x0:bbox_x1], dtype=np.float32)[mask_bool[bbox_y0:bbox_y1, bbox_x0:bbox_x1]]
    if object_prob.size == 0:
        return None
    return _PolygonConfidenceCandidate(
        mask=mask_bool,
        source_branches=tuple(sorted({str(branch) for branch in source_branches if branch})),
        area=area,
        bbox=bbox,
        aspect_ratio=float(aspect_ratio),
        elongation=float(elongation),
        extent=float(extent),
        peak_probability=float(np.max(object_prob)),
        mean_probability=float(np.mean(object_prob, dtype=np.float64)),
    )


def _candidate_object_labels(
    candidates: Sequence[_PolygonConfidenceCandidate],
    shape: tuple[int, int],
) -> np.ndarray:
    """Build a label map that preserves final-candidate identity even when masks touch."""

    labels = np.zeros(shape, dtype=np.int32)
    for object_id, candidate in enumerate(candidates, start=1):
        candidate_mask = np.asarray(candidate.mask, dtype=bool)
        if candidate_mask.shape != shape or not np.any(candidate_mask):
            continue
        assignable = candidate_mask & (labels == 0)
        if np.any(assignable):
            labels[assignable] = int(object_id)
    return labels


def _append_debug_candidate(rows: list[PolygonConfidenceDebugCandidate], candidate_id: int, branch: str, candidate: _PolygonConfidenceCandidate, *, accepted: bool, notes: tuple[str, ...] = ()) -> None:
    rows.append(PolygonConfidenceDebugCandidate(
        object_id=int(candidate_id),
        branch=str(branch),
        source_branches=tuple(candidate.source_branches),
        accepted=bool(accepted),
        area=int(candidate.area),
        bbox_x=int(candidate.bbox[0]),
        bbox_y=int(candidate.bbox[1]),
        bbox_width=int(candidate.bbox[2]),
        bbox_height=int(candidate.bbox[3]),
        aspect_ratio=float(candidate.aspect_ratio),
        elongation=float(candidate.elongation),
        peak_probability=float(candidate.peak_probability),
        mean_probability=float(candidate.mean_probability),
        extent=float(candidate.extent),
        notes=tuple(str(note) for note in notes),
    ))


def _extract_branch_candidates(
    probability: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
    *,
    branch: str,
    min_area: int = 1,
    max_area: int | None = None,
    require_high_core: bool = True,
    acceptance_fn=None,
    start_candidate_id: int = 1,
) -> tuple[list[_PolygonConfidenceCandidate], np.ndarray, tuple[PolygonConfidenceDebugCandidate, ...], int]:
    accepted: list[_PolygonConfidenceCandidate] = []
    debug_rows: list[PolygonConfidenceDebugCandidate] = []
    accepted_mask = np.zeros_like(np.asarray(low_mask, dtype=bool), dtype=bool)
    labels, count = _label_components(low_mask)
    candidate_id = int(start_candidate_id)
    for label_id in range(1, int(count) + 1):
        component_mask = labels == label_id
        candidate = _make_polygon_candidate(probability, component_mask, (branch,))
        if candidate is None:
            continue
        has_high_core = bool(np.any(np.asarray(high_mask, dtype=bool) & component_mask))
        notes: list[str] = []
        accepted_flag = candidate.area >= max(1, int(min_area))
        if max_area is not None:
            accepted_flag = accepted_flag and candidate.area <= max(1, int(max_area))
            if candidate.area > max(1, int(max_area)):
                notes.append('area_above_max')
        if candidate.area < max(1, int(min_area)):
            notes.append('area_below_min')
        if require_high_core and not has_high_core:
            accepted_flag = False
            notes.append('missing_high_core')
        if acceptance_fn is not None:
            branch_ok, branch_notes = acceptance_fn(candidate, has_high_core)
            accepted_flag = accepted_flag and bool(branch_ok)
            notes.extend(tuple(branch_notes))
        _append_debug_candidate(debug_rows, candidate_id, branch, candidate, accepted=accepted_flag, notes=tuple(notes))
        if accepted_flag:
            accepted.append(candidate)
            accepted_mask |= candidate.mask
        candidate_id += 1
    return accepted, accepted_mask, tuple(debug_rows), candidate_id


def _candidate_iou(first: _PolygonConfidenceCandidate, second: _PolygonConfidenceCandidate) -> float:
    x0_a, y0_a, w_a, h_a = first.bbox
    x0_b, y0_b, w_b, h_b = second.bbox
    x1_a = x0_a + w_a
    y1_a = y0_a + h_a
    x1_b = x0_b + w_b
    y1_b = y0_b + h_b
    ix0 = max(x0_a, x0_b)
    iy0 = max(y0_a, y0_b)
    ix1 = min(x1_a, x1_b)
    iy1 = min(y1_a, y1_b)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = int(np.count_nonzero(first.mask[iy0:iy1, ix0:ix1] & second.mask[iy0:iy1, ix0:ix1]))
    if intersection <= 0:
        return 0.0
    union = int(first.area + second.area - intersection)
    return float(intersection / max(1, union))


def _bbox_gap(first_bbox: tuple[int, int, int, int], second_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x0_a, y0_a, w_a, h_a = first_bbox
    x0_b, y0_b, w_b, h_b = second_bbox
    x1_a = x0_a + w_a
    y1_a = y0_a + h_a
    x1_b = x0_b + w_b
    y1_b = y0_b + h_b
    gap_x = max(0, max(x0_b - x1_a, x0_a - x1_b))
    gap_y = max(0, max(y0_b - y1_a, y0_a - y1_b))
    return int(gap_y), int(gap_x)


def _touches_within_distance(first: _PolygonConfidenceCandidate, second: _PolygonConfidenceCandidate, distance: int) -> bool:
    gap = max(0, int(distance))
    if gap <= 0:
        iy, ix = _bbox_gap(first.bbox, second.bbox)
        if iy > 0 or ix > 0:
            return False
        x0 = max(int(first.bbox[0]), int(second.bbox[0]))
        y0 = max(int(first.bbox[1]), int(second.bbox[1]))
        x1 = min(int(first.bbox[0] + first.bbox[2]), int(second.bbox[0] + second.bbox[2]))
        y1 = min(int(first.bbox[1] + first.bbox[3]), int(second.bbox[1] + second.bbox[3]))
        if x1 <= x0 or y1 <= y0:
            return False
        return bool(np.any(first.mask[y0:y1, x0:x1] & second.mask[y0:y1, x0:x1]))
    gap_y, gap_x = _bbox_gap(first.bbox, second.bbox)
    if gap_y > gap or gap_x > gap:
        return False
    x0 = max(0, min(int(first.bbox[0]), int(second.bbox[0])) - gap)
    y0 = max(0, min(int(first.bbox[1]), int(second.bbox[1])) - gap)
    x1 = min(first.mask.shape[1], max(int(first.bbox[0] + first.bbox[2]), int(second.bbox[0] + second.bbox[2])) + gap)
    y1 = min(first.mask.shape[0], max(int(first.bbox[1] + first.bbox[3]), int(second.bbox[1] + second.bbox[3])) + gap)
    roi_first = np.asarray(first.mask[y0:y1, x0:x1], dtype=bool)
    roi_second = np.asarray(second.mask[y0:y1, x0:x1], dtype=bool)
    return bool(np.any(_binary_dilate(roi_first, gap) & roi_second))


def _axis_gap_close(first: _PolygonConfidenceCandidate, second: _PolygonConfidenceCandidate, config: PolygonConfidencePipelineConfig) -> bool:
    x0_a, y0_a, w_a, h_a = first.bbox
    x0_b, y0_b, w_b, h_b = second.bbox
    x1_a = x0_a + w_a
    x1_b = x0_b + w_b
    y1_a = y0_a + h_a
    y1_b = y0_b + h_b
    overlap_x = min(x1_a, x1_b) - max(x0_a, x0_b)
    overlap_y = min(y1_a, y1_b) - max(y0_a, y0_b)
    vertical_gap = max(0, max(y0_b - y1_a, y0_a - y1_b))
    horizontal_gap = max(0, max(x0_b - x1_a, x0_a - x1_b))
    major_gap = max(1, int(config.merge_distance) * max(1, int(POLYGON_CONFIDENCE_COMPLETION_MAJOR_SCALE)))

    def _dominant_axis(candidate: _PolygonConfidenceCandidate) -> str:
        bbox_width = max(1, int(candidate.bbox[2]))
        bbox_height = max(1, int(candidate.bbox[3]))
        if bbox_width == bbox_height:
            return "compact"
        return "horizontal" if bbox_width >= bbox_height else "vertical"

    first_axis = _dominant_axis(first)
    second_axis = _dominant_axis(second)
    if first_axis == "compact" and second_axis != "compact":
        first_axis = second_axis
    elif second_axis == "compact" and first_axis != "compact":
        second_axis = first_axis
    if first_axis != second_axis:
        return False
    if first_axis == "vertical":
        return overlap_x > 0 and vertical_gap <= major_gap
    if first_axis == "horizontal":
        return overlap_y > 0 and horizontal_gap <= major_gap
    return False


def _filter_bridge_region(
    probability: np.ndarray,
    bridge_mask: np.ndarray,
    first_support: np.ndarray,
    second_support: np.ndarray,
    *,
    floor_threshold: float,
    trim_delta: float,
) -> np.ndarray:
    """Keep only bridge pixels that have enough support to justify connecting two regions."""

    bridge_bool = np.asarray(bridge_mask, dtype=bool)
    if not np.any(bridge_bool):
        return bridge_bool
    first_bool = np.asarray(first_support, dtype=bool)
    second_bool = np.asarray(second_support, dtype=bool)
    prob = np.asarray(probability, dtype=np.float32)
    first_values = np.asarray(prob[first_bool], dtype=np.float32)
    second_values = np.asarray(prob[second_bool], dtype=np.float32)
    if first_values.size == 0 or second_values.size == 0:
        return np.zeros_like(bridge_bool, dtype=bool)
    reference_mean = min(
        float(np.mean(first_values, dtype=np.float64)),
        float(np.mean(second_values, dtype=np.float64)),
    )
    bridge_threshold = max(
        float(floor_threshold),
        min(reference_mean - float(trim_delta), reference_mean * 0.92),
    )
    keep = bridge_bool & (prob >= bridge_threshold)
    if not np.any(keep):
        return np.zeros_like(bridge_bool, dtype=bool)
    kept_fraction = float(np.count_nonzero(keep) / max(1, np.count_nonzero(bridge_bool)))
    if kept_fraction < 0.35:
        return np.zeros_like(bridge_bool, dtype=bool)
    keep_touch = np.asarray(_binary_dilate(keep, 1), dtype=bool)
    if not (np.any(keep_touch & first_bool) and np.any(keep_touch & second_bool)):
        return np.zeros_like(bridge_bool, dtype=bool)
    return np.asarray(keep, dtype=bool)


def _merge_candidate_pair(
    probability: np.ndarray,
    first: _PolygonConfidenceCandidate,
    second: _PolygonConfidenceCandidate,
    *,
    bridge_radius: int = 0,
    floor_threshold: float = POLYGON_CONFIDENCE_HYSTERESIS_FLOOR,
    trim_delta: float = POLYGON_CONFIDENCE_SPILL_TRIM_DELTA,
) -> _PolygonConfidenceCandidate | None:
    first_mask = np.asarray(first.mask, dtype=bool)
    second_mask = np.asarray(second.mask, dtype=bool)
    merged_mask = first_mask | second_mask
    gap_radius = max(0, int(bridge_radius))
    if gap_radius > 0 and not np.any(first_mask & second_mask):
        x0 = max(0, min(int(first.bbox[0]), int(second.bbox[0])) - gap_radius)
        y0 = max(0, min(int(first.bbox[1]), int(second.bbox[1])) - gap_radius)
        x1 = min(first_mask.shape[1], max(int(first.bbox[0] + first.bbox[2]), int(second.bbox[0] + second.bbox[2])) + gap_radius)
        y1 = min(first_mask.shape[0], max(int(first.bbox[1] + first.bbox[3]), int(second.bbox[1] + second.bbox[3])) + gap_radius)
        first_roi = first_mask[y0:y1, x0:x1]
        second_roi = second_mask[y0:y1, x0:x1]
        bridge_roi = _binary_dilate(first_roi, gap_radius) & _binary_dilate(second_roi, gap_radius)
        if np.any(bridge_roi):
            bridge_roi = _filter_bridge_region(
                np.asarray(probability[y0:y1, x0:x1], dtype=np.float32),
                bridge_roi,
                first_roi,
                second_roi,
                floor_threshold=float(floor_threshold),
                trim_delta=float(trim_delta),
            )
        if np.any(bridge_roi):
            merged_mask = merged_mask.copy()
            merged_mask[y0:y1, x0:x1] |= bridge_roi
    merged_bbox = (
        max(0, min(int(first.bbox[0]), int(second.bbox[0])) - gap_radius),
        max(0, min(int(first.bbox[1]), int(second.bbox[1])) - gap_radius),
        min(first_mask.shape[1], max(int(first.bbox[0] + first.bbox[2]), int(second.bbox[0] + second.bbox[2])) + gap_radius)
        - max(0, min(int(first.bbox[0]), int(second.bbox[0])) - gap_radius),
        min(first_mask.shape[0], max(int(first.bbox[1] + first.bbox[3]), int(second.bbox[1] + second.bbox[3])) + gap_radius)
        - max(0, min(int(first.bbox[1]), int(second.bbox[1])) - gap_radius),
    )
    return _make_polygon_candidate(
        probability,
        merged_mask,
        tuple(sorted(set(first.source_branches) | set(second.source_branches))),
        roi_bbox=merged_bbox,
    )


def _merge_search_margin(config: PolygonConfidencePipelineConfig) -> int:
    """Return the maximum bbox margin needed for merge neighborhood search."""

    base_gap = max(0, int(config.merge_distance))
    axis_gap = max(1, base_gap * max(1, int(POLYGON_CONFIDENCE_COMPLETION_MAJOR_SCALE)))
    return max(base_gap, axis_gap)


@lru_cache(maxsize=64)
def _merge_tile_size(search_margin: int) -> int:
    """Return a stable tile size for spatial candidate indexing."""

    return max(16, 2 * max(1, int(search_margin)) + 8)


def _candidate_neighbor_pairs(
    candidates: list[_PolygonConfidenceCandidate],
    config: PolygonConfidencePipelineConfig,
) -> list[tuple[int, int]]:
    """Return unique candidate pairs that are spatially close enough to merit exact merge checks."""

    if len(candidates) <= 1:
        return []
    search_margin = _merge_search_margin(config)
    tile_size = _merge_tile_size(search_margin)
    bins: dict[tuple[int, int], list[int]] = {}
    pairs: set[tuple[int, int]] = set()
    for index, candidate in enumerate(candidates):
        x0, y0, width, height = candidate.bbox
        x1 = x0 + width
        y1 = y0 + height
        gx0 = (max(0, int(x0) - search_margin)) // tile_size
        gy0 = (max(0, int(y0) - search_margin)) // tile_size
        gx1 = (max(0, int(x1) + search_margin - 1)) // tile_size
        gy1 = (max(0, int(y1) + search_margin - 1)) // tile_size
        for grid_y in range(gy0, gy1 + 1):
            for grid_x in range(gx0, gx1 + 1):
                cell = (grid_x, grid_y)
                for other_index in bins.get(cell, ()): 
                    if other_index == index:
                        continue
                    pair = (other_index, index) if other_index < index else (index, other_index)
                    pairs.add(pair)
                bins.setdefault(cell, []).append(index)
    return sorted(pairs)


def _merge_connected_candidate_groups(
    probability: np.ndarray,
    candidates: list[_PolygonConfidenceCandidate],
    pair_indices: list[tuple[int, int]],
    config: PolygonConfidencePipelineConfig,
) -> list[_PolygonConfidenceCandidate]:
    """Merge all candidate groups connected by exact merge predicates for one round."""

    if not pair_indices:
        return list(candidates)

    parent = list(range(len(candidates)))

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _union(left: int, right: int) -> None:
        root_left = _find(left)
        root_right = _find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    merge_threshold = float(config.merge_iou_threshold)
    merge_distance = int(config.merge_distance)
    for index_a, index_b in pair_indices:
        first = candidates[index_a]
        second = candidates[index_b]
        overlap = _candidate_iou(first, second)
        if overlap >= merge_threshold:
            _union(index_a, index_b)
            continue
        if _touches_within_distance(first, second, merge_distance) or _axis_gap_close(first, second, config):
            _union(index_a, index_b)

    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(_find(index), []).append(index)

    if all(len(indices) == 1 for indices in groups.values()):
        return list(candidates)

    next_candidates: list[_PolygonConfidenceCandidate] = []
    for indices in groups.values():
        ordered = sorted(indices)
        if len(ordered) == 1:
            next_candidates.append(candidates[ordered[0]])
            continue
        merged_candidate = candidates[ordered[0]]
        for current_index in ordered[1:]:
            merged_candidate = _merge_candidate_pair(
                probability,
                merged_candidate,
                candidates[current_index],
                bridge_radius=max(1, merge_distance),
                floor_threshold=float(config.hysteresis_low_floor),
                trim_delta=float(config.spill_trim_delta),
            )
            if merged_candidate is None:
                merged_candidate = candidates[ordered[0]]
                break
        next_candidates.append(merged_candidate)
    return next_candidates


def _merge_polygon_candidates(
    probability: np.ndarray,
    candidates: list[_PolygonConfidenceCandidate],
    config: PolygonConfidencePipelineConfig,
) -> list[_PolygonConfidenceCandidate]:
    """Merge polygon candidates using spatial shortlists instead of repeated all-pairs scans."""

    merged = list(candidates)
    while True:
        pair_indices = _candidate_neighbor_pairs(merged, config)
        if not pair_indices:
            return merged
        next_candidates = _merge_connected_candidate_groups(probability, merged, pair_indices, config)
        if len(next_candidates) == len(merged):
            return merged
        merged = next_candidates


def _complete_polygon_candidates(
    probability: np.ndarray,
    candidates: list[_PolygonConfidenceCandidate],
    *,
    strong_threshold: float,
    weak_threshold: float,
    high_seed_mask: np.ndarray | None = None,
) -> list[_PolygonConfidenceCandidate]:
    if not candidates:
        return []
    completion_threshold = _polygon_confidence_completion_threshold(float(strong_threshold), float(weak_threshold))
    completion_mask = np.asarray(probability >= completion_threshold, dtype=bool)
    if not np.any(completion_mask):
        return list(candidates)
    completion_labels, completion_count = _label_components(completion_mask)
    if completion_count <= 0:
        return list(candidates)
    high_labels = None
    if high_seed_mask is not None:
        high_labels, _high_count = _label_components(np.asarray(high_seed_mask, dtype=bool) & completion_mask)
    bridge_radius = max(0, int(POLYGON_CONFIDENCE_COMPLETION_BRIDGE_RADIUS))
    completed: list[_PolygonConfidenceCandidate] = []
    for candidate in candidates:
        seed_mask = np.asarray(candidate.mask, dtype=bool)
        reach_radius_y, reach_radius_x = _completion_radii_for_mask(seed_mask, bridge_radius)
        reach_mask = _binary_dilate_rect(seed_mask, reach_radius_y, reach_radius_x) if (reach_radius_y > 0 or reach_radius_x > 0) else seed_mask
        grown_mask = np.asarray(seed_mask, dtype=bool)
        completion_ids = [int(candidate_id) for candidate_id in np.unique(completion_labels[reach_mask]) if int(candidate_id) > 0]
        for completion_id in completion_ids:
            component_mask = completion_labels == completion_id
            component_add = component_mask
            if high_labels is not None:
                component_high_ids = np.unique(high_labels[component_mask])
                component_high_ids = component_high_ids[component_high_ids > 0]
                if component_high_ids.size > 1:
                    current_high_ids = np.unique(high_labels[component_mask & seed_mask])
                    current_high_ids = current_high_ids[current_high_ids > 0]
                    if current_high_ids.size > 0:
                        valid_seed_mask = np.isin(high_labels, component_high_ids, assume_unique=True) & component_mask
                        if np.any(valid_seed_mask):
                            _distances, nearest_indices = ndi.distance_transform_edt(~valid_seed_mask, return_indices=True)
                            nearest_seed_labels = high_labels[tuple(nearest_indices)]
                            component_add = component_mask & np.isin(nearest_seed_labels, current_high_ids, assume_unique=True)
                            component_add &= (
                                np.asarray(probability, dtype=np.float32) >= max(
                                    float(POLYGON_CONFIDENCE_HYSTERESIS_FLOOR),
                                    float(weak_threshold) + 0.02,
                                )
                            ) | seed_mask
            grown_mask |= component_add
            if (reach_radius_y > 0 or reach_radius_x > 0) and not np.any(component_mask & seed_mask):
                bridge_path = _binary_dilate_rect(seed_mask, reach_radius_y, reach_radius_x) & _binary_dilate_rect(component_mask, reach_radius_y, reach_radius_x)
                bridge_path = _filter_bridge_region(
                    probability,
                    bridge_path,
                    seed_mask,
                    component_mask,
                    floor_threshold=max(float(weak_threshold), float(POLYGON_CONFIDENCE_HYSTERESIS_FLOOR)),
                    trim_delta=float(POLYGON_CONFIDENCE_SPILL_TRIM_DELTA),
                )
                if np.any(bridge_path):
                    grown_mask |= bridge_path
        completed_candidate = _make_polygon_candidate(probability, grown_mask, candidate.source_branches)
        completed.append(completed_candidate if completed_candidate is not None else candidate)
    return completed


def _split_polygon_candidate_by_seeds(
    probability: np.ndarray,
    candidate: _PolygonConfidenceCandidate,
    high_mask: np.ndarray,
    config: PolygonConfidencePipelineConfig,
) -> tuple[_PolygonConfidenceCandidate, ...]:
    """Backward-compatible wrapper around the barrier-aware seed separation stage."""

    split_candidates, _debug_payload = _split_polygon_candidate_by_barriers(
        probability,
        candidate,
        high_mask,
        low_threshold=_polygon_confidence_weak_threshold(float(np.max(np.asarray(probability[candidate.mask], dtype=np.float32))) if np.any(candidate.mask) else 0.5, config),
        strong_threshold=float(np.max(np.asarray(probability[candidate.mask], dtype=np.float32))) if np.any(candidate.mask) else 0.5,
        config=config,
        include_debug=False,
    )
    return split_candidates


def _candidate_boundary_separation(probability: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return 0.0
    boundary = _boundary_mask(mask_bool)
    outer_ring = _binary_dilate(mask_bool, 1) & ~mask_bool
    if not np.any(boundary) or not np.any(outer_ring):
        return 0.0
    inner_mean = float(np.mean(np.asarray(probability[boundary], dtype=np.float32), dtype=np.float64))
    outer_mean = float(np.mean(np.asarray(probability[outer_ring], dtype=np.float32), dtype=np.float64))
    return float(_clip01(inner_mean - outer_mean))


def _candidate_axis_coverage(mask: np.ndarray) -> tuple[float, float]:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return 0.0, 0.0
    height, width = mask_bool.shape
    row_coverage = np.count_nonzero(mask_bool, axis=1).astype(np.float32) / max(1, width)
    col_coverage = np.count_nonzero(mask_bool, axis=0).astype(np.float32) / max(1, height)
    active_row = row_coverage[row_coverage > 0.0]
    active_col = col_coverage[col_coverage > 0.0]
    row_p90 = float(np.quantile(active_row, 0.9)) if active_row.size else 0.0
    col_p90 = float(np.quantile(active_col, 0.9)) if active_col.size else 0.0
    return row_p90, col_p90



def _candidate_border_span_features(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    *,
    cross_axis_max: float,
) -> dict[str, float | bool]:
    """Return border-touch and span metrics shared by spill rejection and final trimming."""

    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    x0, y0, width, height = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    x1 = x0 + width
    y1 = y0 + height
    touches_lr = x0 <= 0 and x1 >= image_width
    touches_tb = y0 <= 0 and y1 >= image_height
    span_x = float(width / max(1, image_width))
    span_y = float(height / max(1, image_height))
    thin_cross_axis = (
        (touches_lr and span_y <= float(cross_axis_max))
        or (touches_tb and span_x <= float(cross_axis_max))
    )
    row_coverage_p90, col_coverage_p90 = _candidate_axis_coverage(mask)
    axis_coverage = 0.0
    if touches_lr:
        axis_coverage = max(axis_coverage, row_coverage_p90)
    if touches_tb:
        axis_coverage = max(axis_coverage, col_coverage_p90)
    return {
        'touches_lr': bool(touches_lr),
        'touches_tb': bool(touches_tb),
        'span_x': float(span_x),
        'span_y': float(span_y),
        'thin_cross_axis': bool(thin_cross_axis),
        'row_coverage_p90': float(row_coverage_p90),
        'col_coverage_p90': float(col_coverage_p90),
        'axis_coverage': float(axis_coverage),
    }


def _candidate_axis_support_strength(
    candidate_mask: np.ndarray,
    support_mask: np.ndarray,
    *,
    touches_lr: bool,
    touches_tb: bool,
) -> tuple[float, float]:
    """Return major-axis support coverage and support area fraction for a candidate."""

    candidate_bool = np.asarray(candidate_mask, dtype=bool)
    support_bool = np.asarray(support_mask, dtype=bool) & candidate_bool
    candidate_area = int(np.count_nonzero(candidate_bool))
    if candidate_area <= 0 or not np.any(support_bool):
        return 0.0, 0.0
    row_coverage_p90, col_coverage_p90 = _candidate_axis_coverage(support_bool)
    axis_coverage = 0.0
    if touches_lr:
        axis_coverage = max(axis_coverage, row_coverage_p90)
    if touches_tb:
        axis_coverage = max(axis_coverage, col_coverage_p90)
    support_area_fraction = float(np.count_nonzero(support_bool) / max(1, candidate_area))
    return float(axis_coverage), float(support_area_fraction)


def _candidate_has_compact_core(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    core_threshold: float,
    max_aspect_ratio: float,
) -> bool:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return False
    core_mask = mask_bool & (np.asarray(probability, dtype=np.float32) >= float(core_threshold))
    if not np.any(core_mask):
        return False
    labels, count = _label_components(core_mask)
    for label_id in range(1, int(count) + 1):
        component_mask = labels == label_id
        area = int(np.count_nonzero(component_mask))
        if area < 4:
            continue
        ys, xs = np.nonzero(component_mask)
        if ys.size == 0 or xs.size == 0:
            continue
        width = int(np.max(xs) - np.min(xs) + 1)
        height = int(np.max(ys) - np.min(ys) + 1)
        aspect_ratio = float(max(width, height) / max(1, min(width, height)))
        if aspect_ratio <= float(max_aspect_ratio):
            return True
    return False


def _compact_core_mask(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    core_threshold: float,
    max_aspect_ratio: float,
    min_area: int = 4,
) -> np.ndarray:
    """Return compact high-probability cores inside a candidate."""

    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return np.zeros_like(mask_bool, dtype=bool)
    core_mask = mask_bool & (np.asarray(probability, dtype=np.float32) >= float(core_threshold))
    if not np.any(core_mask):
        return np.zeros_like(mask_bool, dtype=bool)
    labels, count = _label_components(core_mask)
    compact = np.zeros_like(mask_bool, dtype=bool)
    for label_id in range(1, int(count) + 1):
        component_mask = labels == label_id
        area = int(np.count_nonzero(component_mask))
        if area < max(1, int(min_area)):
            continue
        ys, xs = np.nonzero(component_mask)
        if ys.size == 0 or xs.size == 0:
            continue
        width = int(np.max(xs) - np.min(xs) + 1)
        height = int(np.max(ys) - np.min(ys) + 1)
        aspect_ratio = float(max(width, height) / max(1, min(width, height)))
        if aspect_ratio <= float(max_aspect_ratio):
            compact |= component_mask
    return compact


def _local_prominence_map(probability: np.ndarray, radius: int) -> np.ndarray:
    """Estimate local prominence over a broader neighborhood."""

    prob = np.asarray(probability, dtype=np.float32)
    local_mean = _local_mean_map(prob, radius=max(1, int(radius)))
    prominence = np.clip(prob - local_mean, 0.0, 1.0)
    return np.asarray(prominence, dtype=np.float32)


def _retain_core_supported_region(
    probability: np.ndarray,
    prominence_map: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    core_threshold: float,
    grow_threshold: float,
    prominence_threshold: float,
    max_core_aspect_ratio: float,
) -> np.ndarray:
    """Trim a spill-like candidate to regions supported by compact high-prominence cores."""

    mask_bool = np.asarray(candidate_mask, dtype=bool)
    compact_core = _compact_core_mask(
        probability,
        mask_bool,
        core_threshold=core_threshold,
        max_aspect_ratio=max_core_aspect_ratio,
    )
    if not np.any(compact_core):
        return np.zeros_like(mask_bool, dtype=bool)
    support_mask = mask_bool & (
        (np.asarray(probability, dtype=np.float32) >= float(grow_threshold))
        | (np.asarray(prominence_map, dtype=np.float32) >= float(prominence_threshold))
    )
    if not np.any(support_mask):
        return compact_core
    core_labels, core_count = _label_components(compact_core)
    retained = np.zeros_like(mask_bool, dtype=bool)
    for core_id in range(1, int(core_count) + 1):
        core_component = core_labels == core_id
        if not np.any(core_component):
            continue
        core_bbox = _mask_bbox(core_component)
        core_width = max(1, int(core_bbox[2]))
        core_height = max(1, int(core_bbox[3]))
        if core_width >= core_height:
            core_window = _binary_dilate_rect(
                core_component,
                radius_y=max(1, min(4, core_height)),
                radius_x=max(2, min(4, core_width // 3 + 1)),
            )
        else:
            core_window = _binary_dilate_rect(
                core_component,
                radius_y=max(2, min(4, core_height // 3 + 1)),
                radius_x=max(1, min(4, core_width)),
            )
        local_support = support_mask & core_window
        if not np.any(local_support):
            retained |= core_component
            continue
        labels, count = _label_components(local_support)
        retained_core = False
        for label_id in range(1, int(count) + 1):
            component_mask = labels == label_id
            if np.any(component_mask & core_component):
                retained |= component_mask
                retained_core = True
        if not retained_core:
            retained |= core_component
    return retained if np.any(retained) else compact_core


def _connected_support_from_seed(support_mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    """Keep only support components connected to the provided seed mask."""

    support_bool = np.asarray(support_mask, dtype=bool)
    seed_bool = np.asarray(seed_mask, dtype=bool) & support_bool
    if not np.any(support_bool) or not np.any(seed_bool):
        return np.zeros_like(support_bool, dtype=bool)
    labels, count = _label_components(support_bool)
    if count <= 0:
        return np.zeros_like(support_bool, dtype=bool)
    seed_labels = np.unique(labels[seed_bool])
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size <= 0:
        return np.zeros_like(support_bool, dtype=bool)
    return np.isin(labels, seed_labels, assume_unique=True)


def _separation_core_mask(
    candidate_mask: np.ndarray,
    high_mask: np.ndarray,
    *,
    min_area: int,
) -> np.ndarray:
    """Return stable seed cores for one candidate, removing tiny fragments."""

    candidate_bool = np.asarray(candidate_mask, dtype=bool)
    core_mask = candidate_bool & np.asarray(high_mask, dtype=bool)
    if not np.any(core_mask):
        return np.zeros_like(candidate_bool, dtype=bool)
    labels, count = _label_components(core_mask)
    if count <= 0:
        return np.zeros_like(candidate_bool, dtype=bool)
    area_counts = np.bincount(labels.ravel(), minlength=count + 1)
    valid_ids = np.flatnonzero(area_counts >= max(1, int(min_area)))
    valid_ids = valid_ids[valid_ids > 0]
    if valid_ids.size <= 0:
        return np.zeros_like(candidate_bool, dtype=bool)
    return np.isin(labels, valid_ids, assume_unique=True)


def _max_filter_map(array: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    local_radius = max(1, int(radius))
    if ndi is not None:
        size = 2 * local_radius + 1
        return np.asarray(ndi.maximum_filter(values, size=size, mode='nearest'), dtype=np.float32)
    padded = np.pad(values, local_radius, mode='edge')
    result = np.empty_like(values, dtype=np.float32)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            patch = padded[y:y + 2 * local_radius + 1, x:x + 2 * local_radius + 1]
            result[y, x] = float(np.max(patch))
    return result


def _boundary_cue_map(
    probability: np.ndarray,
    candidate_region: np.ndarray,
    *,
    low_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> np.ndarray:
    """Build boundary cues that combine valleys, local contrast, and transition uncertainty."""

    candidate_bool = np.asarray(candidate_region, dtype=bool)
    if not np.any(candidate_bool):
        return np.zeros_like(candidate_bool, dtype=np.float32)
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    local_contrast = _local_contrast_map(prob, radius=1)
    _width_map, _inverse_local_contrast, transition_uncertainty = _polygon_transition_uncertainty_maps(
        prob,
        candidate_bool,
        contrast_radius=1,
    )
    candidate_values = np.asarray(prob[candidate_bool], dtype=np.float32)
    candidate_low = float(np.quantile(candidate_values, 0.20)) if candidate_values.size else float(low_threshold)
    valley_reference = max(float(low_threshold), candidate_low)
    valley_scale = max(0.04, 1.0 - valley_reference)
    low_score = np.clip((valley_reference - prob) / valley_scale, 0.0, 1.0).astype(np.float32)
    cue = np.clip(
        float(config.separation_boundary_low_weight) * low_score
        + float(config.separation_boundary_contrast_weight) * local_contrast
        + float(config.separation_boundary_uncertainty_weight) * transition_uncertainty,
        0.0,
        1.0,
    ).astype(np.float32)
    cue[~candidate_bool] = 0.0
    return cue


def _thin_barrier_map_from_cues(
    boundary_cues: np.ndarray,
    candidate_region: np.ndarray,
    *,
    threshold: float,
    dilate_radius: int,
) -> np.ndarray:
    """Convert wide boundary cues into a thin separating barrier."""

    candidate_bool = np.asarray(candidate_region, dtype=bool)
    cue = np.asarray(boundary_cues, dtype=np.float32)
    if not np.any(candidate_bool):
        return np.zeros_like(candidate_bool, dtype=bool)
    ridge_zone = candidate_bool & (cue >= float(threshold))
    if not np.any(ridge_zone):
        return np.zeros_like(candidate_bool, dtype=bool)
    local_max = _max_filter_map(cue, radius=1)
    ridge_center = ridge_zone & (cue >= (local_max - 1e-6))
    skeleton = skeletonize(ridge_zone)
    thin = (ridge_center | skeleton) & candidate_bool
    radius = max(0, int(dilate_radius))
    if radius > 0:
        thin = _binary_dilate(thin, radius) & candidate_bool
    return np.asarray(thin, dtype=bool)


def _bridge_cut_mask(
    probability: np.ndarray,
    candidate_region: np.ndarray,
    boundary_cues: np.ndarray,
    seed_mask: np.ndarray,
    *,
    config: PolygonConfidencePipelineConfig,
) -> np.ndarray:
    """Break weak narrow bridges that connect multiple nearby seed regions."""

    candidate_bool = np.asarray(candidate_region, dtype=bool)
    seed_bool = np.asarray(seed_mask, dtype=bool) & candidate_bool
    if not np.any(candidate_bool) or not np.any(seed_bool):
        return np.zeros_like(candidate_bool, dtype=bool)
    seed_labels, seed_count = _label_components(seed_bool)
    if seed_count <= 1:
        return np.zeros_like(candidate_bool, dtype=bool)
    thin_bridge = np.asarray(_thin_bridge_map(candidate_bool) > 0.0, dtype=bool)
    if not np.any(thin_bridge):
        return np.zeros_like(candidate_bool, dtype=bool)
    bridge_mask = thin_bridge & (
        np.asarray(probability, dtype=np.float32) <= float(config.separation_bridge_probability_max)
    ) & (
        np.asarray(boundary_cues, dtype=np.float32) >= float(config.separation_bridge_barrier_threshold)
    )
    if not np.any(bridge_mask):
        return np.zeros_like(candidate_bool, dtype=bool)
    labels, count = _label_components(bridge_mask)
    if count <= 0:
        return np.zeros_like(candidate_bool, dtype=bool)
    result = np.zeros_like(candidate_bool, dtype=bool)
    for label_id in range(1, int(count) + 1):
        component = labels == label_id
        if not np.any(component):
            continue
        expanded = _binary_dilate(component, 1) & candidate_bool
        touched_seeds = np.unique(seed_labels[expanded])
        touched_seeds = touched_seeds[touched_seeds > 0]
        if touched_seeds.size >= 2:
            result |= component
    return result


def _assign_candidate_region_to_seeds(
    candidate_region: np.ndarray,
    seed_labels: np.ndarray,
    blocked_mask: np.ndarray,
) -> np.ndarray:
    """Assign candidate pixels to seed instances while respecting blocked barriers."""

    candidate_bool = np.asarray(candidate_region, dtype=bool)
    labels = np.asarray(seed_labels, dtype=np.int32)
    blocked = np.asarray(blocked_mask, dtype=bool) & candidate_bool
    if not np.any(candidate_bool) or not np.any(labels > 0):
        return np.zeros_like(labels, dtype=np.int32)
    available = candidate_bool & ~blocked
    available |= labels > 0
    if not np.any(available):
        return np.zeros_like(labels, dtype=np.int32)
    component_labels, component_count = _label_components(available)
    if component_count <= 0:
        return np.zeros_like(labels, dtype=np.int32)
    assigned = np.zeros_like(labels, dtype=np.int32)
    for component_id in range(1, int(component_count) + 1):
        component = component_labels == component_id
        if not np.any(component):
            continue
        touched = np.unique(labels[component])
        touched = touched[touched > 0]
        if touched.size == 1:
            assigned[component] = int(touched[0])
            continue
        if touched.size <= 1:
            continue
        valid_seed_mask = np.isin(labels, touched, assume_unique=True)
        if not np.any(valid_seed_mask):
            continue
        _distances, nearest_indices = ndi.distance_transform_edt(~valid_seed_mask, return_indices=True)
        nearest_seed_labels = labels[tuple(nearest_indices)]
        assigned_component = nearest_seed_labels[component]
        positive = assigned_component > 0
        if np.any(positive):
            assigned_values = assigned_component[positive]
            component_indices = np.argwhere(component)
            assigned[component_indices[positive, 0], component_indices[positive, 1]] = assigned_values
    return assigned


def _split_polygon_candidate_by_barriers(
    probability: np.ndarray,
    candidate: _PolygonConfidenceCandidate,
    high_mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
    include_debug: bool = False,
) -> tuple[tuple[_PolygonConfidenceCandidate, ...], dict[str, np.ndarray] | None]:
    """Split one candidate via thin barriers and bridge cuts, then grow from stable seed cores."""

    if not bool(config.enable_watershed) or ndi is None:
        return (candidate,), None

    candidate_mask = np.asarray(candidate.mask, dtype=bool)
    if not np.any(candidate_mask):
        return (candidate,), None
    pad = max(1, int(config.separation_roi_padding))
    x0, y0, width, height = candidate.bbox
    roi_x0 = max(0, x0 - pad)
    roi_y0 = max(0, y0 - pad)
    roi_x1 = min(candidate_mask.shape[1], x0 + width + pad)
    roi_y1 = min(candidate_mask.shape[0], y0 + height + pad)
    candidate_roi = np.asarray(candidate_mask[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    if not np.any(candidate_roi):
        return (candidate,), None
    probability_roi = np.asarray(probability[roi_y0:roi_y1, roi_x0:roi_x1], dtype=np.float32)
    high_roi = np.asarray(high_mask[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    core_roi = _separation_core_mask(
        candidate_roi,
        high_roi,
        min_area=max(1, int(config.separation_core_min_area)),
    )
    seed_labels, seed_count = _label_components(core_roi)
    debug_payload = None
    if include_debug:
        debug_payload = {
            "candidate_region": np.zeros_like(candidate_mask, dtype=bool),
            "core_seeds": np.zeros_like(candidate_mask, dtype=bool),
            "thin_barrier": np.zeros_like(candidate_mask, dtype=bool),
            "bridge_cuts": np.zeros_like(candidate_mask, dtype=bool),
            "barrier_blocked": np.zeros_like(candidate_mask, dtype=bool),
            "boundary_cues": np.zeros_like(probability, dtype=np.float32),
        }
        debug_payload["candidate_region"][roi_y0:roi_y1, roi_x0:roi_x1] = candidate_roi
        debug_payload["core_seeds"][roi_y0:roi_y1, roi_x0:roi_x1] = core_roi
    if seed_count <= 1:
        return (candidate,), debug_payload

    boundary_cues_roi = _boundary_cue_map(
        probability_roi,
        candidate_roi,
        low_threshold=low_threshold,
        config=config,
    )
    thin_barrier_roi = _thin_barrier_map_from_cues(
        boundary_cues_roi,
        candidate_roi,
        threshold=float(config.separation_barrier_threshold),
        dilate_radius=int(config.separation_barrier_dilate_radius),
    )
    bridge_cut_roi = _bridge_cut_mask(
        probability_roi,
        candidate_roi,
        boundary_cues_roi,
        core_roi,
        config=config,
    )
    if not np.any(thin_barrier_roi) and not np.any(bridge_cut_roi):
        return (candidate,), debug_payload
    blocked_roi = (thin_barrier_roi | bridge_cut_roi) & ~core_roi
    assigned_labels_roi = _assign_candidate_region_to_seeds(candidate_roi, seed_labels, blocked_roi)
    split_candidates: list[_PolygonConfidenceCandidate] = []
    for seed_id in range(1, int(seed_count) + 1):
        component_roi = assigned_labels_roi == seed_id
        if not np.any(component_roi):
            continue
        component_mask = np.zeros_like(candidate_mask, dtype=bool)
        component_mask[roi_y0:roi_y1, roi_x0:roi_x1] = component_roi
        component_candidate = _make_polygon_candidate(
            probability,
            component_mask,
            candidate.source_branches,
            roi_bbox=(roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0),
        )
        if component_candidate is not None:
            split_candidates.append(component_candidate)
    if include_debug and debug_payload is not None:
        debug_payload["thin_barrier"][roi_y0:roi_y1, roi_x0:roi_x1] = thin_barrier_roi
        debug_payload["bridge_cuts"][roi_y0:roi_y1, roi_x0:roi_x1] = bridge_cut_roi
        debug_payload["barrier_blocked"][roi_y0:roi_y1, roi_x0:roi_x1] = blocked_roi
        debug_payload["boundary_cues"][roi_y0:roi_y1, roi_x0:roi_x1] = boundary_cues_roi
    return tuple(split_candidates) if split_candidates else (candidate,), debug_payload


def _tighten_candidate_with_barrier_support(
    probability: np.ndarray,
    mask: np.ndarray,
    high_mask: np.ndarray,
    *,
    low_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> np.ndarray:
    """Tighten inflated small/medium candidate boundaries using thin barrier evidence."""

    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return mask_bool
    candidate = _make_polygon_candidate(probability, mask_bool, ('refine',))
    if candidate is None or candidate.area < 6:
        return mask_bool
    if float(max(candidate.aspect_ratio, candidate.elongation)) >= float(config.boundary_snap_min_aspect):
        return mask_bool
    pad = max(1, int(config.separation_roi_padding))
    x0, y0, width, height = candidate.bbox
    roi_x0 = max(0, x0 - pad)
    roi_y0 = max(0, y0 - pad)
    roi_x1 = min(mask_bool.shape[1], x0 + width + pad)
    roi_y1 = min(mask_bool.shape[0], y0 + height + pad)
    candidate_roi = np.asarray(mask_bool[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    high_roi = np.asarray(high_mask[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    core_roi = _separation_core_mask(
        candidate_roi,
        high_roi,
        min_area=max(1, int(config.separation_core_min_area)),
    )
    if not np.any(core_roi):
        return mask_bool
    boundary_cues_roi = _boundary_cue_map(
        np.asarray(probability[roi_y0:roi_y1, roi_x0:roi_x1], dtype=np.float32),
        candidate_roi,
        low_threshold=low_threshold,
        config=config,
    )
    thin_barrier_roi = _thin_barrier_map_from_cues(
        boundary_cues_roi,
        candidate_roi,
        threshold=max(float(config.separation_barrier_threshold), 0.56),
        dilate_radius=0,
    )
    if not np.any(thin_barrier_roi):
        return mask_bool
    blocked_roi = thin_barrier_roi & ~_binary_dilate(core_roi, 1)
    if not np.any(blocked_roi):
        return mask_bool
    tightened_roi = _connected_support_from_seed(candidate_roi & ~blocked_roi, core_roi)
    if not np.any(tightened_roi):
        return mask_bool
    retained_fraction = float(np.count_nonzero(tightened_roi) / max(1, np.count_nonzero(candidate_roi)))
    if retained_fraction < 0.68:
        return mask_bool
    tightened_mask = mask_bool.copy()
    tightened_mask[roi_y0:roi_y1, roi_x0:roi_x1] = tightened_roi
    return tightened_mask


def _spanning_barrier_mask(
    score_map: np.ndarray,
    band_mask: np.ndarray,
    *,
    threshold: float,
    horizontal: bool,
    coverage_min: float,
) -> np.ndarray:
    """Find low-score separator components that span the minor axis inside a local band."""

    band_bool = np.asarray(band_mask, dtype=bool)
    if not np.any(band_bool):
        return np.zeros_like(band_bool, dtype=bool)
    valley_mask = band_bool & (np.asarray(score_map, dtype=np.float32) <= float(threshold))
    if not np.any(valley_mask):
        return np.zeros_like(band_bool, dtype=bool)
    labels, count = _label_components(valley_mask)
    if count <= 0:
        return np.zeros_like(band_bool, dtype=bool)
    active_minor = np.any(band_bool, axis=1 if horizontal else 0)
    minor_span = max(1, int(np.count_nonzero(active_minor)))
    spanning = np.zeros_like(band_bool, dtype=bool)
    for label_id in range(1, int(count) + 1):
        component = labels == label_id
        if not np.any(component):
            continue
        ys, xs = np.nonzero(component)
        if ys.size == 0 or xs.size == 0:
            continue
        component_minor_span = int(np.max(ys) - np.min(ys) + 1) if horizontal else int(np.max(xs) - np.min(xs) + 1)
        if float(component_minor_span / max(1, minor_span)) >= float(coverage_min):
            spanning |= component
    return spanning


def _reconstruct_large_polygon_candidate(
    probability: np.ndarray,
    local_normalized_probability: np.ndarray,
    seed_mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> _PolygonConfidenceCandidate | None:
    """Reconstruct one large polygon from a strong seed while keeping internal texture separate."""

    seed_bool = np.asarray(seed_mask, dtype=bool)
    if not np.any(seed_bool):
        return None
    seed_bbox = _mask_bbox(seed_bool)
    if seed_bbox[2] <= 0 or seed_bbox[3] <= 0:
        return None

    score_map = np.maximum(
        np.asarray(probability, dtype=np.float32),
        np.asarray(local_normalized_probability, dtype=np.float32),
    ).astype(np.float32)
    image_height, image_width = score_map.shape
    seed_x0, seed_y0, seed_w, seed_h = seed_bbox
    band_expand = max(
        int(config.large_polygon_band_expand),
        min(max(seed_w, seed_h), max(seed_w if seed_w < seed_h else seed_h, 1)),
    )
    roi_padding = max(1, int(config.large_polygon_roi_padding))
    base_support_threshold = max(
        float(config.hysteresis_low_floor),
        float(low_threshold) * float(config.large_polygon_low_scale),
    )
    seed_support_threshold = max(
        float(config.hysteresis_low_floor),
        min(
            float(base_support_threshold),
            float(low_threshold) * float(config.large_polygon_seed_low_scale),
            float(strong_threshold) * 0.40,
        ),
    )
    barrier_threshold = max(
        float(config.hysteresis_low_floor),
        float(base_support_threshold) - float(config.large_polygon_barrier_delta),
    )

    def _candidate_for_orientation(horizontal: bool) -> _PolygonConfidenceCandidate | None:
        envelope_support = np.asarray(score_map >= float(seed_support_threshold), dtype=bool)
        envelope_support |= seed_bool
        connected_envelope = _connected_support_from_seed(envelope_support, seed_bool)
        envelope_bbox = _mask_bbox(connected_envelope) if np.any(connected_envelope) else seed_bbox
        env_x0, env_y0, env_w, env_h = envelope_bbox
        env_x1 = env_x0 + env_w
        env_y1 = env_y0 + env_h
        if horizontal:
            roi_y0 = max(0, min(seed_y0, env_y0) - band_expand)
            roi_y1 = min(image_height, max(seed_y0 + seed_h, env_y1) + band_expand)
            roi_x0 = max(0, env_x0 - roi_padding)
            roi_x1 = min(image_width, env_x1 + roi_padding)
        else:
            roi_x0 = max(0, min(seed_x0, env_x0) - band_expand)
            roi_x1 = min(image_width, max(seed_x0 + seed_w, env_x1) + band_expand)
            roi_y0 = max(0, env_y0 - roi_padding)
            roi_y1 = min(image_height, env_y1 + roi_padding)
        if roi_x1 <= roi_x0 or roi_y1 <= roi_y0:
            return None
        score_roi = np.asarray(score_map[roi_y0:roi_y1, roi_x0:roi_x1], dtype=np.float32)
        seed_roi = np.asarray(seed_bool[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
        if not np.any(seed_roi):
            return None

        support_roi = np.asarray(score_roi >= float(base_support_threshold), dtype=bool)
        if horizontal:
            support_roi = _binary_close_rect(
                support_roi,
                radius_y=max(0, int(config.large_polygon_minor_close_radius)),
                radius_x=max(1, int(config.large_polygon_major_close_radius)),
            )
        else:
            support_roi = _binary_close_rect(
                support_roi,
                radius_y=max(1, int(config.large_polygon_major_close_radius)),
                radius_x=max(0, int(config.large_polygon_minor_close_radius)),
            )
        support_roi |= seed_roi

        barrier_roi = _spanning_barrier_mask(
            score_roi,
            support_roi,
            threshold=barrier_threshold,
            horizontal=horizontal,
            coverage_min=float(config.large_polygon_barrier_coverage_min),
        )
        if np.any(barrier_roi):
            if horizontal:
                barrier_roi = _binary_dilate_rect(barrier_roi, radius_y=0, radius_x=1)
            else:
                barrier_roi = _binary_dilate_rect(barrier_roi, radius_y=1, radius_x=0)
            support_roi &= ~barrier_roi
            support_roi |= seed_roi

        connected_roi = _connected_support_from_seed(support_roi, seed_roi)
        if not np.any(connected_roi):
            return None
        candidate_mask = np.zeros_like(seed_bool, dtype=bool)
        candidate_mask[roi_y0:roi_y1, roi_x0:roi_x1] = connected_roi
        return _make_polygon_candidate(probability, candidate_mask, ('large_polygon',), roi_bbox=(roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0))

    candidates = tuple(
        candidate
        for candidate in (
            _candidate_for_orientation(horizontal=True),
            _candidate_for_orientation(horizontal=False),
        )
        if candidate is not None
    )
    if not candidates:
        return None

    def _rank(candidate: _PolygonConfidenceCandidate) -> tuple[float, float, float, float]:
        bbox_width = max(1, int(candidate.bbox[2]))
        bbox_height = max(1, int(candidate.bbox[3]))
        major_span = float(max(bbox_width, bbox_height))
        return (
            major_span,
            float(candidate.extent),
            float(max(candidate.aspect_ratio, candidate.elongation)),
            float(candidate.mean_probability),
        )

    return max(candidates, key=_rank)


def _extract_large_polygon_candidates(
    probability: np.ndarray,
    local_normalized_probability: np.ndarray,
    high_mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
    start_candidate_id: int = 1,
) -> tuple[list[_PolygonConfidenceCandidate], np.ndarray, tuple[PolygonConfidenceDebugCandidate, ...], int]:
    """Extract large polygon candidates from strong seeds before small-detail processing."""

    accepted: list[_PolygonConfidenceCandidate] = []
    debug_rows: list[PolygonConfidenceDebugCandidate] = []
    accepted_mask = np.zeros_like(np.asarray(high_mask, dtype=bool), dtype=bool)
    high_labels, high_count = _label_components(high_mask)
    candidate_id = int(start_candidate_id)
    for label_id in range(1, int(high_count) + 1):
        seed_component = high_labels == label_id
        seed_candidate = _make_polygon_candidate(probability, seed_component, ('large_polygon_seed',))
        if seed_candidate is None:
            continue
        reconstructed = _reconstruct_large_polygon_candidate(
            probability,
            local_normalized_probability,
            seed_component,
            low_threshold=low_threshold,
            strong_threshold=strong_threshold,
            config=config,
        )
        candidate = reconstructed or seed_candidate
        bbox_width = max(1, int(candidate.bbox[2]))
        bbox_height = max(1, int(candidate.bbox[3]))
        major_span = max(bbox_width, bbox_height)
        geometry_ok = (
            candidate.area >= max(1, int(config.large_polygon_min_area))
            and major_span >= max(1, int(config.large_polygon_min_major_span))
            and candidate.extent >= float(config.large_polygon_min_extent)
            and max(candidate.aspect_ratio, candidate.elongation) >= float(config.large_polygon_min_aspect_ratio)
        )
        notes: list[str] = []
        if candidate.area < max(1, int(config.large_polygon_min_area)):
            notes.append('large_area_below_min')
        if major_span < max(1, int(config.large_polygon_min_major_span)):
            notes.append('large_major_span_too_small')
        if candidate.extent < float(config.large_polygon_min_extent):
            notes.append('large_extent_too_small')
        if max(candidate.aspect_ratio, candidate.elongation) < float(config.large_polygon_min_aspect_ratio):
            notes.append('large_ratio_too_small')
        has_high_core = bool(np.any(np.asarray(high_mask, dtype=bool) & np.asarray(candidate.mask, dtype=bool)))
        accepted_flag = bool(geometry_ok and has_high_core)
        if not has_high_core:
            notes.append('large_missing_high_core')
        _append_debug_candidate(debug_rows, candidate_id, 'large_polygon', candidate, accepted=accepted_flag, notes=tuple(notes))
        if accepted_flag:
            accepted.append(candidate)
            accepted_mask |= np.asarray(candidate.mask, dtype=bool)
        candidate_id += 1
    return accepted, accepted_mask, tuple(debug_rows), candidate_id


def _should_reject_branch_spill(
    probability: np.ndarray,
    contrast_map: np.ndarray,
    candidate: _PolygonConfidenceCandidate,
    *,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> tuple[bool, tuple[str, ...]]:
    image_height, image_width = probability.shape
    border_features = _candidate_border_span_features(
        candidate.mask,
        candidate.bbox,
        probability.shape,
        cross_axis_max=float(config.spill_cross_axis_max),
    )
    touches_lr = bool(border_features['touches_lr'])
    touches_tb = bool(border_features['touches_tb'])
    if not (touches_lr or touches_tb):
        return False, ()
    area_fraction = float(candidate.area / max(1, probability.size))
    thin_cross_axis = bool(border_features['thin_cross_axis'])
    if area_fraction < float(config.spill_large_area_fraction) and not thin_cross_axis:
        return False, ()
    if float(candidate.extent) < float(config.spill_large_extent):
        return False, ()
    if float(max(candidate.aspect_ratio, candidate.elongation)) < float(config.spill_ribbon_aspect_min):
        return False, ()
    axis_coverage = float(border_features['axis_coverage'])
    if axis_coverage < float(config.spill_border_coverage_min):
        return False, ()
    mask_bool = np.asarray(candidate.mask, dtype=bool)
    boundary_separation = _candidate_boundary_separation(probability, mask_bool)
    if boundary_separation >= float(config.spill_boundary_separation_max):
        return False, ()
    interior_mask = _binary_erode(mask_bool, 1) & mask_bool
    texture_region = interior_mask if np.any(interior_mask) else mask_bool
    texture_mean = float(np.mean(np.asarray(contrast_map[texture_region], dtype=np.float32), dtype=np.float64)) if np.any(texture_region) else 0.0
    peak_margin = float(max(0.0, float(candidate.peak_probability) - float(candidate.mean_probability)))
    core_threshold = max(
        float(strong_threshold),
        float(candidate.mean_probability) + max(
            float(config.spill_trim_delta),
            0.35 * peak_margin,
        ),
    )
    has_compact_core = _candidate_has_compact_core(
        probability,
        mask_bool,
        core_threshold=core_threshold,
        max_aspect_ratio=max(1.6, float(config.spill_ribbon_aspect_min) * 0.55),
    )
    reject = (
        peak_margin <= float(config.spill_peak_margin_max)
        and (
            texture_mean <= float(config.spill_low_texture_max)
            or float(candidate.mean_probability) <= float(config.spill_mean_probability_max)
        )
        and not has_compact_core
    )
    if not reject:
        return False, ()
    return True, ('branch_spill_reject',)


def _carve_enclosed_low_probability_holes(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    low_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return mask_bool
    hole_threshold = min(float(config.hole_probability_max), max(float(config.hysteresis_low_floor), float(low_threshold) * float(config.hole_probability_scale)))
    low_inside = mask_bool & (np.asarray(probability, dtype=np.float32) <= hole_threshold)
    if not np.any(low_inside):
        return mask_bool
    hole_labels, hole_count = _label_components(low_inside)
    if hole_count <= 0:
        return mask_bool
    object_boundary = _boundary_mask(mask_bool)
    carved = mask_bool.copy()
    min_hole_area = max(1, int(config.hole_min_area))
    for label_id in range(1, int(hole_count) + 1):
        hole_mask = hole_labels == label_id
        hole_area = int(np.count_nonzero(hole_mask))
        if hole_area < min_hole_area:
            continue
        if np.any(hole_mask & object_boundary):
            continue
        surround = (_binary_dilate(hole_mask, 1) & mask_bool) & ~hole_mask
        if not np.any(surround):
            continue
        hole_mean = float(np.mean(np.asarray(probability[hole_mask], dtype=np.float32), dtype=np.float64))
        surround_mean = float(np.mean(np.asarray(probability[surround], dtype=np.float32), dtype=np.float64))
        if surround_mean <= hole_mean + 0.05:
            continue
        carved[hole_mask] = False
    return carved


def _split_candidate_by_spanning_valleys(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
    source_branches: tuple[str, ...],
) -> tuple[_PolygonConfidenceCandidate, ...]:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return ()
    candidate = _make_polygon_candidate(probability, mask_bool, source_branches)
    if candidate is None:
        return ()
    x0, y0, width, height = candidate.bbox
    if width < 6 and height < 6:
        return (candidate,)
    roi_mask = np.asarray(mask_bool[y0:y0 + height, x0:x0 + width], dtype=bool)
    roi_prob = np.asarray(probability[y0:y0 + height, x0:x0 + width], dtype=np.float32)
    horizontal = width >= height
    elongated_like = float(max(candidate.aspect_ratio, candidate.elongation)) >= float(config.boundary_snap_min_aspect)
    peak_margin = float(max(0.0, float(candidate.peak_probability) - float(candidate.mean_probability)))
    valley_threshold = max(
        float(config.hysteresis_low_floor),
        min(
            float(strong_threshold) - 0.08,
            float(low_threshold) + max(0.05, float(config.spill_trim_delta) + 0.02),
        ),
    )
    if 'large_polygon' in set(source_branches):
        valley_threshold = max(
            float(config.hysteresis_low_floor),
            min(
                float(strong_threshold) - 0.08,
                float(candidate.mean_probability) - max(0.04, 0.50 * peak_margin),
            ),
        )
    valley_mask = roi_mask & (roi_prob <= valley_threshold)
    if not np.any(valley_mask):
        return (candidate,)
    valley_labels, valley_count = _label_components(valley_mask)
    if valley_count <= 0:
        return (candidate,)
    carved_roi = roi_mask.copy()
    coverage_min = float(config.spill_border_coverage_min)
    minor_coverage_min = float(config.valley_minor_coverage_min)
    removed_any = False
    for label_id in range(1, int(valley_count) + 1):
        component = valley_labels == label_id
        if not np.any(component):
            continue
        ys, xs = np.nonzero(component)
        if ys.size == 0 or xs.size == 0:
            continue
        comp_width = int(np.max(xs) - np.min(xs) + 1)
        comp_height = int(np.max(ys) - np.min(ys) + 1)
        cover_x = float(comp_width / max(1, width))
        cover_y = float(comp_height / max(1, height))
        is_horizontal_separator = cover_x >= coverage_min and cover_y <= 0.35
        is_vertical_separator = cover_y >= coverage_min and cover_x <= 0.35
        if elongated_like and horizontal:
            is_vertical_separator = is_vertical_separator or (cover_y >= minor_coverage_min and cover_x <= 0.25)
        elif elongated_like and not horizontal:
            is_horizontal_separator = is_horizontal_separator or (cover_x >= minor_coverage_min and cover_y <= 0.25)
        if not (is_horizontal_separator or is_vertical_separator):
            continue
        # Slight dilation makes the separator robust to weak single-pixel bridges.
        if horizontal and is_vertical_separator:
            carved_roi &= ~_binary_dilate_rect(component, 1, 0)
        elif (not horizontal) and is_horizontal_separator:
            carved_roi &= ~_binary_dilate_rect(component, 0, 1)
        else:
            carved_roi &= ~_binary_dilate(component, 1)
        removed_any = True
    if not removed_any or not np.any(carved_roi):
        return (candidate,)
    split_labels, split_count = _label_components(carved_roi)
    if split_count <= 1:
        carved_mask = mask_bool.copy()
        carved_mask[y0:y0 + height, x0:x0 + width] = carved_roi
        split_candidate = _make_polygon_candidate(probability, carved_mask, source_branches)
        return (split_candidate,) if split_candidate is not None else (candidate,)
    split_candidates: list[_PolygonConfidenceCandidate] = []
    strong_mask = np.asarray(probability, dtype=np.float32) >= float(strong_threshold)
    for label_id in range(1, int(split_count) + 1):
        component_roi = split_labels == label_id
        if not np.any(component_roi):
            continue
        component_mask = np.zeros_like(mask_bool, dtype=bool)
        component_mask[y0:y0 + height, x0:x0 + width] = component_roi
        if not np.any(component_mask & strong_mask):
            continue
        component_candidate = _make_polygon_candidate(probability, component_mask, source_branches)
        if component_candidate is not None:
            split_candidates.append(component_candidate)
    return tuple(split_candidates) if split_candidates else (candidate,)


def _tighten_candidate_with_boundary_barriers(
    probability: np.ndarray,
    mask: np.ndarray,
    high_mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
    source_branches: tuple[str, ...],
) -> np.ndarray:
    """Trim local boundary overgrowth while preserving core-supported geometry."""

    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return mask_bool
    candidate = _make_polygon_candidate(probability, mask_bool, source_branches)
    if candidate is None:
        return mask_bool
    if candidate.area < max(6, int(config.separation_core_min_area) * 3):
        return mask_bool

    pad = max(1, int(config.separation_roi_padding))
    x0, y0, width, height = candidate.bbox
    roi_x0 = max(0, x0 - pad)
    roi_y0 = max(0, y0 - pad)
    roi_x1 = min(mask_bool.shape[1], x0 + width + pad)
    roi_y1 = min(mask_bool.shape[0], y0 + height + pad)

    candidate_roi = np.asarray(mask_bool[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    if not np.any(candidate_roi):
        return mask_bool
    probability_roi = np.asarray(probability[roi_y0:roi_y1, roi_x0:roi_x1], dtype=np.float32)
    high_roi = np.asarray(high_mask[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    core_roi = _separation_core_mask(
        candidate_roi,
        high_roi,
        min_area=max(1, int(config.separation_core_min_area)),
    )
    if not np.any(core_roi):
        return mask_bool
    _seed_labels, seed_count = _label_components(core_roi)

    boundary_cues_roi = _boundary_cue_map(
        probability_roi,
        candidate_roi,
        low_threshold=low_threshold,
        config=config,
    )
    cue_values = np.asarray(boundary_cues_roi[candidate_roi], dtype=np.float32)
    if cue_values.size <= 0:
        return mask_bool
    tighten_threshold = min(
        0.95,
        max(
            float(config.separation_barrier_threshold) + 0.08,
            float(np.quantile(cue_values, 0.72)),
        ),
    )
    thin_barrier_roi = _thin_barrier_map_from_cues(
        boundary_cues_roi,
        candidate_roi,
        threshold=tighten_threshold,
        dilate_radius=0,
    )
    if not np.any(thin_barrier_roi):
        return mask_bool

    boundary_seed = thin_barrier_roi & _binary_dilate(_boundary_mask(candidate_roi), 1)
    edge_barrier_roi = (
        _connected_support_from_seed(thin_barrier_roi, boundary_seed)
        if np.any(boundary_seed)
        else np.zeros_like(candidate_roi, dtype=bool)
    )
    bridge_cut_roi = _bridge_cut_mask(
        probability_roi,
        candidate_roi,
        boundary_cues_roi,
        core_roi,
        config=config,
    )
    thin_bridge_roi = np.asarray(_thin_bridge_map(candidate_roi) > 0.0, dtype=bool)
    internal_conflict_roi = (
        _binary_dilate(thin_barrier_roi & _binary_dilate(thin_bridge_roi, 1), 1) & candidate_roi
        if seed_count > 1 and np.any(thin_bridge_roi)
        else np.zeros_like(candidate_roi, dtype=bool)
    )
    protected_roi = core_roi
    blocked_roi = (edge_barrier_roi | internal_conflict_roi | bridge_cut_roi) & ~protected_roi
    if not np.any(blocked_roi):
        return mask_bool

    tightened_roi = _connected_support_from_seed(candidate_roi & ~blocked_roi, core_roi)
    if not np.any(tightened_roi):
        return mask_bool
    original_area = int(np.count_nonzero(candidate_roi))
    tightened_area = int(np.count_nonzero(tightened_roi))
    retained_fraction = float(tightened_area / max(1, original_area))
    min_retained_fraction = max(0.72, float(config.boundary_snap_min_retained_fraction))
    if retained_fraction < min_retained_fraction:
        return mask_bool
    if tightened_area >= original_area:
        return mask_bool

    refined_mask = mask_bool.copy()
    refined_mask[roi_y0:roi_y1, roi_x0:roi_x1] = tightened_roi
    return refined_mask


def _snap_elongated_candidate_boundaries(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
    source_branches: tuple[str, ...],
) -> np.ndarray:
    """Refine large elongated candidates by snapping the minor-axis boundaries to local profiles."""

    def _smooth_trace(values: np.ndarray, valid_mask: np.ndarray, window: int) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32).copy()
        valid = np.asarray(valid_mask, dtype=bool)
        if not np.any(valid):
            return result
        coords = np.flatnonzero(valid)
        samples = np.asarray(result[coords], dtype=np.float32)
        if samples.size >= 3:
            kernel = max(3, int(window) | 1)
            if ndi is not None:
                samples = np.asarray(ndi.median_filter(samples, size=kernel, mode='nearest'), dtype=np.float32)
            else:
                radius = kernel // 2
                padded = np.pad(samples, (radius, radius), mode='edge')
                smoothed = np.empty_like(samples)
                for index in range(samples.size):
                    smoothed[index] = float(np.median(padded[index:index + kernel]))
                samples = smoothed
        result[coords] = samples
        return result

    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return mask_bool
    candidate = _make_polygon_candidate(probability, mask_bool, source_branches)
    if candidate is None:
        return mask_bool
    if float(max(candidate.aspect_ratio, candidate.elongation)) < float(config.boundary_snap_min_aspect):
        return mask_bool
    x0, y0, width, height = candidate.bbox
    if width < 6 or height < 6:
        return mask_bool

    pad_y = max(2, min(6, height))
    pad_x = max(2, min(6, width))
    roi_x0 = max(0, x0 - pad_x)
    roi_y0 = max(0, y0 - pad_y)
    roi_x1 = min(mask_bool.shape[1], x0 + width + pad_x)
    roi_y1 = min(mask_bool.shape[0], y0 + height + pad_y)

    roi_mask = np.asarray(mask_bool[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    roi_prob = np.asarray(probability[roi_y0:roi_y1, roi_x0:roi_x1], dtype=np.float32)
    inner_x0 = x0 - roi_x0
    inner_y0 = y0 - roi_y0
    inner_x1 = inner_x0 + width
    inner_y1 = inner_y0 + height
    inner_mask = np.asarray(roi_mask[inner_y0:inner_y1, inner_x0:inner_x1], dtype=bool)
    object_prob = np.asarray(roi_prob[roi_mask], dtype=np.float32)
    if object_prob.size == 0:
        return mask_bool

    refined_roi = np.zeros_like(roi_mask, dtype=bool)
    horizontal = width >= height
    if horizontal:
        active_cols = np.any(inner_mask, axis=0)
        if not np.any(active_cols):
            return mask_bool
        center_profile = np.mean(roi_prob[:, inner_x0:inner_x1][:, active_cols], axis=1, dtype=np.float32)
        active_rows = np.any(inner_mask, axis=1)
        if np.count_nonzero(active_rows) <= 2:
            return mask_bool
        active_row_indices = inner_y0 + np.flatnonzero(active_rows)
        edge_drop = float(np.max(center_profile[active_row_indices])) - float(max(center_profile[active_row_indices[0]], center_profile[active_row_indices[-1]]))
        if edge_drop < float(config.boundary_snap_min_drop):
            return mask_bool
        center_row = int(np.argmax(center_profile))
        top_trace = np.full(width, np.nan, dtype=np.float32)
        bottom_trace = np.full(width, np.nan, dtype=np.float32)
        valid_trace = np.zeros(width, dtype=bool)
        for local_col in np.flatnonzero(active_cols):
            col = inner_x0 + int(local_col)
            col_profile = np.asarray(roi_prob[:, col], dtype=np.float32)
            baseline = float(np.quantile(col_profile, 0.2))
            peak_row = int(np.argmax(col_profile))
            peak_value = float(col_profile[peak_row])
            profile_threshold = max(
                float(low_threshold),
                min(
                    float(strong_threshold),
                    baseline + 0.45 * max(0.0, peak_value - baseline),
                ),
            )
            keep_rows = col_profile >= profile_threshold
            if not np.any(keep_rows):
                continue
            row_labels, row_count = _label_components(keep_rows[:, None])
            selected_rows = None
            preferred_row = peak_row if keep_rows[peak_row] else center_row
            for label_id in range(1, int(row_count) + 1):
                component = (row_labels[:, 0] == label_id)
                if component[preferred_row]:
                    selected_rows = component
                    break
            if selected_rows is None:
                continue
            selected_indices = np.flatnonzero(selected_rows & roi_mask[:, col])
            if selected_indices.size == 0:
                continue
            top_trace[int(local_col)] = float(selected_indices[0])
            bottom_trace[int(local_col)] = float(selected_indices[-1])
            valid_trace[int(local_col)] = True
        if not np.any(valid_trace):
            return mask_bool
        active_col_indices = np.flatnonzero(active_cols)
        valid_col_indices = np.flatnonzero(valid_trace)
        top_trace = np.interp(active_col_indices, valid_col_indices, top_trace[valid_col_indices]).astype(np.float32)
        bottom_trace = np.interp(active_col_indices, valid_col_indices, bottom_trace[valid_col_indices]).astype(np.float32)
        smooth_window = max(3, min(11, width // 6 if width >= 6 else 3))
        top_smoothed = _smooth_trace(top_trace, np.ones_like(top_trace, dtype=bool), smooth_window)
        bottom_smoothed = _smooth_trace(bottom_trace, np.ones_like(bottom_trace, dtype=bool), smooth_window)
        for position, local_col in enumerate(active_col_indices.tolist()):
            col = inner_x0 + int(local_col)
            top = int(np.clip(round(float(top_smoothed[position])), 0, roi_mask.shape[0] - 1))
            bottom = int(np.clip(round(float(bottom_smoothed[position])), top, roi_mask.shape[0] - 1))
            refined_roi[top:bottom + 1, col] = roi_mask[top:bottom + 1, col]
    else:
        active_rows = np.any(inner_mask, axis=1)
        if not np.any(active_rows):
            return mask_bool
        center_profile = np.mean(roi_prob[inner_y0:inner_y1, :][active_rows, :], axis=0, dtype=np.float32)
        active_cols = np.any(inner_mask, axis=0)
        if np.count_nonzero(active_cols) <= 2:
            return mask_bool
        active_col_indices = inner_x0 + np.flatnonzero(active_cols)
        edge_drop = float(np.max(center_profile[active_col_indices])) - float(max(center_profile[active_col_indices[0]], center_profile[active_col_indices[-1]]))
        if edge_drop < float(config.boundary_snap_min_drop):
            return mask_bool
        center_col = int(np.argmax(center_profile))
        left_trace = np.full(height, np.nan, dtype=np.float32)
        right_trace = np.full(height, np.nan, dtype=np.float32)
        valid_trace = np.zeros(height, dtype=bool)
        for local_row in np.flatnonzero(active_rows):
            row = inner_y0 + int(local_row)
            row_profile = np.asarray(roi_prob[row, :], dtype=np.float32)
            baseline = float(np.quantile(row_profile, 0.2))
            peak_col = int(np.argmax(row_profile))
            peak_value = float(row_profile[peak_col])
            profile_threshold = max(
                float(low_threshold),
                min(
                    float(strong_threshold),
                    baseline + 0.45 * max(0.0, peak_value - baseline),
                ),
            )
            keep_cols = row_profile >= profile_threshold
            if not np.any(keep_cols):
                continue
            col_labels, col_count = _label_components(keep_cols[None, :])
            selected_cols = None
            preferred_col = peak_col if keep_cols[peak_col] else center_col
            for label_id in range(1, int(col_count) + 1):
                component = (col_labels[0, :] == label_id)
                if component[preferred_col]:
                    selected_cols = component
                    break
            if selected_cols is None:
                continue
            selected_indices = np.flatnonzero(selected_cols & roi_mask[row, :])
            if selected_indices.size == 0:
                continue
            left_trace[int(local_row)] = float(selected_indices[0])
            right_trace[int(local_row)] = float(selected_indices[-1])
            valid_trace[int(local_row)] = True
        if not np.any(valid_trace):
            return mask_bool
        active_row_indices = np.flatnonzero(active_rows)
        valid_row_indices = np.flatnonzero(valid_trace)
        left_trace = np.interp(active_row_indices, valid_row_indices, left_trace[valid_row_indices]).astype(np.float32)
        right_trace = np.interp(active_row_indices, valid_row_indices, right_trace[valid_row_indices]).astype(np.float32)
        smooth_window = max(3, min(11, height // 6 if height >= 6 else 3))
        left_smoothed = _smooth_trace(left_trace, np.ones_like(left_trace, dtype=bool), smooth_window)
        right_smoothed = _smooth_trace(right_trace, np.ones_like(right_trace, dtype=bool), smooth_window)
        for position, local_row in enumerate(active_row_indices.tolist()):
            row = inner_y0 + int(local_row)
            left = int(np.clip(round(float(left_smoothed[position])), 0, roi_mask.shape[1] - 1))
            right = int(np.clip(round(float(right_smoothed[position])), left, roi_mask.shape[1] - 1))
            refined_roi[row, left:right + 1] = roi_mask[row, left:right + 1]

    if not np.any(refined_roi):
        return mask_bool
    original_area = int(np.count_nonzero(roi_mask))
    refined_area = int(np.count_nonzero(refined_roi))
    if refined_area <= 0:
        return mask_bool
    retained_fraction = float(refined_area / max(1, original_area))
    if retained_fraction < float(config.boundary_snap_min_retained_fraction):
        return mask_bool
    refined_mask = mask_bool.copy()
    refined_mask[roi_y0:roi_y1, roi_x0:roi_x1] = refined_roi
    return refined_mask


def _refine_final_polygon_candidates(
    probability: np.ndarray,
    candidates: list[_PolygonConfidenceCandidate],
    *,
    strong_mask: np.ndarray,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> list[_PolygonConfidenceCandidate]:
    if not candidates:
        return []
    low_threshold = _polygon_confidence_weak_threshold(float(strong_threshold), config)
    contrast_map = _local_contrast_map(probability, radius=1)
    prominence_map = _local_prominence_map(probability, radius=max(2, int(config.local_normalization_radius)))
    image_height, image_width = probability.shape
    refined_candidates: list[_PolygonConfidenceCandidate] = []
    for candidate in candidates:
        source_branch_set = set(candidate.source_branches)
        if 'large_polygon' in source_branch_set:
            # Large polygons may contain broad low-confidence texture that should remain
            # inside the polygon as uncertainty, not be carved into geometric holes.
            refined_mask = np.asarray(candidate.mask, dtype=bool)
        else:
            refined_mask = _carve_enclosed_low_probability_holes(
                probability,
                candidate.mask,
                low_threshold=low_threshold,
                config=config,
            )
            refined_mask = _snap_elongated_candidate_boundaries(
                probability,
                refined_mask,
                low_threshold=low_threshold,
                strong_threshold=strong_threshold,
                config=config,
                source_branches=candidate.source_branches,
            )
            refined_mask = _tighten_candidate_with_boundary_barriers(
                probability,
                refined_mask,
                strong_mask,
                low_threshold=low_threshold,
                strong_threshold=strong_threshold,
                config=config,
                source_branches=candidate.source_branches,
            )
        split_after_valleys = _split_candidate_by_spanning_valleys(
            probability,
            refined_mask,
            low_threshold=low_threshold,
            strong_threshold=strong_threshold,
            config=config,
            source_branches=candidate.source_branches,
        )
        for split_candidate in split_after_valleys:
            candidate_mask = np.asarray(split_candidate.mask, dtype=bool)
            candidate_after_holes = _make_polygon_candidate(probability, candidate_mask, split_candidate.source_branches)
            if candidate_after_holes is None:
                continue
            source_branch_set = set(split_candidate.source_branches)
            if 'large_polygon' in source_branch_set:
                # Large-polygon geometry is already reconstructed from a seed-connected local ROI
                # and optionally split by spanning valleys. Re-applying generic spill trimming here
                # collapses weak but valid wide polygons to bright core fragments.
                refined_candidates.append(candidate_after_holes)
                continue
            border_features = _candidate_border_span_features(
                candidate_mask,
                candidate_after_holes.bbox,
                probability.shape,
                cross_axis_max=float(config.spill_cross_axis_max),
            )
            touches_lr = bool(border_features['touches_lr'])
            touches_tb = bool(border_features['touches_tb'])
            touches_opposite_borders = touches_lr or touches_tb
            area_fraction = float(candidate_after_holes.area / max(1, probability.size))
            thin_cross_axis = bool(border_features['thin_cross_axis'])
            if touches_opposite_borders and (area_fraction >= float(config.spill_large_area_fraction) or thin_cross_axis) and float(candidate_after_holes.extent) >= float(config.spill_large_extent):
                interior_mask = _binary_erode(candidate_mask, 1) & candidate_mask
                texture_region = interior_mask if np.any(interior_mask) else candidate_mask
                texture_mean = float(np.mean(np.asarray(contrast_map[texture_region], dtype=np.float32), dtype=np.float64)) if np.any(texture_region) else 0.0
                peak_margin = float(max(0.0, float(candidate_after_holes.peak_probability) - float(candidate_after_holes.mean_probability)))
                boundary_separation = _candidate_boundary_separation(probability, candidate_mask)
                is_ribbon = float(max(candidate_after_holes.aspect_ratio, candidate_after_holes.elongation)) >= float(config.spill_ribbon_aspect_min)
                axis_coverage = float(border_features['axis_coverage'])
                object_prob = np.asarray(probability[candidate_mask], dtype=np.float32)
                local_quantile = float(np.quantile(object_prob, 0.80)) if object_prob.size else float(strong_threshold)
                core_threshold = max(
                    float(strong_threshold),
                    float(candidate_after_holes.mean_probability) + max(
                        float(config.spill_trim_delta),
                        0.35 * peak_margin,
                    ),
                    local_quantile,
                )
                strong_axis_coverage, strong_area_fraction = _candidate_axis_support_strength(
                    candidate_mask,
                    strong_mask,
                    touches_lr=touches_lr,
                    touches_tb=touches_tb,
                )
                fallback_only = source_branch_set == {'strong_mask'}
                preserve_compact_core = _candidate_has_compact_core(
                    probability,
                    candidate_mask,
                    core_threshold=core_threshold,
                    max_aspect_ratio=max(2.5, float(config.spill_ribbon_aspect_min)),
                )
                has_strong_axis_support = (
                    not fallback_only
                    and preserve_compact_core
                    and peak_margin > float(config.spill_peak_margin_max)
                    and (
                        strong_axis_coverage >= float(config.spill_strong_axis_coverage_min)
                        or strong_area_fraction >= float(config.spill_strong_area_fraction_min)
                    )
                )
                spill_like = (
                    is_ribbon
                    and axis_coverage >= float(config.spill_border_coverage_min)
                    and (peak_margin <= float(config.spill_peak_margin_max) or thin_cross_axis)
                    and (
                        texture_mean <= float(config.spill_low_texture_max)
                        or float(candidate_after_holes.mean_probability) <= float(config.spill_mean_probability_max)
                    )
                    and boundary_separation <= float(config.spill_boundary_separation_max)
                    and not has_strong_axis_support
                )
                if spill_like:
                    grow_threshold = max(
                        float(strong_threshold),
                        float(candidate_after_holes.mean_probability) + 0.5 * float(config.spill_trim_delta),
                    )
                    trimmed_mask = _retain_core_supported_region(
                        probability,
                        prominence_map,
                        candidate_mask,
                        core_threshold=core_threshold,
                        grow_threshold=grow_threshold,
                        prominence_threshold=float(config.spill_prominence_min),
                        max_core_aspect_ratio=max(2.5, float(config.spill_ribbon_aspect_min)),
                    )
                    trimmed_candidate = _make_polygon_candidate(probability, trimmed_mask, split_candidate.source_branches)
                    if trimmed_candidate is not None:
                        trimmed_border_features = _candidate_border_span_features(
                            trimmed_mask,
                            trimmed_candidate.bbox,
                            probability.shape,
                            cross_axis_max=float(config.spill_cross_axis_max),
                        )
                        trimmed_touches_opposite_borders = bool(trimmed_border_features['touches_lr']) or bool(trimmed_border_features['touches_tb'])
                        trimmed_axis_coverage = float(trimmed_border_features['axis_coverage'])
                        trimmed_peak_margin = float(max(0.0, float(trimmed_candidate.peak_probability) - float(trimmed_candidate.mean_probability)))
                        trimmed_boundary_separation = _candidate_boundary_separation(probability, trimmed_mask)
                        trimmed_is_ribbon = float(max(trimmed_candidate.aspect_ratio, trimmed_candidate.elongation)) >= float(config.spill_ribbon_aspect_min)
                        trimmed_spill_like = (
                            trimmed_is_ribbon
                            and trimmed_touches_opposite_borders
                            and trimmed_axis_coverage >= float(config.spill_border_coverage_min)
                            and trimmed_peak_margin <= float(config.spill_peak_margin_max)
                            and trimmed_boundary_separation <= float(config.spill_boundary_separation_max)
                        )
                        if trimmed_spill_like:
                            continue
                        candidate_after_holes = trimmed_candidate
                    else:
                        continue
            refined_candidates.append(candidate_after_holes)
    return refined_candidates


def _polygon_branch_debug_mask(shape: tuple[int, int], candidates: tuple[_PolygonConfidenceCandidate, ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        mask |= np.asarray(candidate.mask, dtype=bool)
    return mask


def _polygon_confidence_pipeline(
    probability: np.ndarray,
    strong_mask: np.ndarray,
    *,
    strong_threshold: float = 0.5,
    config: PolygonConfidencePipelineConfig | None = None,
    include_debug: bool = False,
) -> tuple[np.ndarray, tuple[_PolygonConfidenceCandidate, ...], PolygonConfidenceDebugData | None]:
    cfg = config or _polygon_confidence_config()
    stage_started = perf_counter()
    stage_timings_ms: dict[str, float] = {}

    raw_prob = _normalize_probability_map(probability)
    preprocessed_prob, local_normalized_prob = _preprocess_polygon_probability(raw_prob, cfg)
    high_threshold = float(max(0.0, min(1.0, strong_threshold)))
    low_threshold = _polygon_confidence_weak_threshold(high_threshold, cfg)
    stage_timings_ms['preprocess'] = 1000.0 * (perf_counter() - stage_started)

    branch_started = perf_counter()
    global_low_mask = np.asarray(preprocessed_prob >= low_threshold, dtype=bool)
    global_high_mask = np.asarray(preprocessed_prob >= high_threshold, dtype=bool) | np.asarray(strong_mask, dtype=bool)
    branch_contrast_map = _local_contrast_map(preprocessed_prob, radius=1)

    debug_rows: list[PolygonConfidenceDebugCandidate] = []
    next_debug_id = 1

    dominant_high_threshold = max(high_threshold, float(cfg.dominant_min_mean_probability))
    dominant_high_mask = np.asarray(preprocessed_prob >= dominant_high_threshold, dtype=bool) | np.asarray(strong_mask, dtype=bool)

    large_polygon_candidates, large_polygon_mask, large_polygon_debug, next_debug_id = _extract_large_polygon_candidates(
        preprocessed_prob,
        local_normalized_prob,
        dominant_high_mask,
        low_threshold=low_threshold,
        strong_threshold=high_threshold,
        config=cfg,
        start_candidate_id=next_debug_id,
    )
    debug_rows.extend(large_polygon_debug)
    dominant_candidates = list(large_polygon_candidates)
    dominant_mask = np.asarray(large_polygon_mask, dtype=bool)
    dominant_lock_mask = (
        _binary_dilate(dominant_mask, max(0, int(cfg.dominant_lock_radius)))
        if np.any(dominant_mask)
        else np.zeros_like(global_low_mask, dtype=bool)
    )

    def _global_accept(candidate: _PolygonConfidenceCandidate, has_high_core: bool) -> tuple[bool, tuple[str, ...]]:
        reject, reject_notes = _should_reject_branch_spill(
            preprocessed_prob,
            branch_contrast_map,
            candidate,
            strong_threshold=high_threshold,
            config=cfg,
        )
        return (False, reject_notes) if reject else (True, ())

    global_candidates, global_mask, global_debug, next_debug_id = _extract_branch_candidates(
        preprocessed_prob,
        global_low_mask & ~dominant_lock_mask,
        global_high_mask & ~dominant_lock_mask,
        branch='global_hysteresis',
        min_area=1,
        require_high_core=True,
        acceptance_fn=_global_accept,
        start_candidate_id=next_debug_id,
    )
    debug_rows.extend(global_debug)

    elongated_low_mask = _binary_close_rect(global_low_mask, int(cfg.elongated_vertical_radius), int(cfg.elongated_horizontal_radius))
    elongated_low_mask |= _binary_close_rect(global_low_mask, int(cfg.elongated_horizontal_radius), int(cfg.elongated_vertical_radius))

    def _elongated_accept(candidate: _PolygonConfidenceCandidate, has_high_core: bool) -> tuple[bool, tuple[str, ...]]:
        notes: list[str] = []
        if candidate.area < max(1, int(cfg.elongated_min_area)):
            notes.append('elongated_area_below_min')
            return False, tuple(notes)
        if max(candidate.aspect_ratio, candidate.elongation) < float(cfg.elongated_min_aspect_ratio):
            notes.append('elongated_ratio_too_small')
            return False, tuple(notes)
        if not has_high_core:
            notes.append('missing_high_core')
            return False, tuple(notes)
        reject, reject_notes = _should_reject_branch_spill(
            preprocessed_prob,
            branch_contrast_map,
            candidate,
            strong_threshold=high_threshold,
            config=cfg,
        )
        if reject:
            notes.extend(reject_notes)
            return False, tuple(notes)
        return True, tuple(notes)

    elongated_candidates, elongated_mask, elongated_debug, next_debug_id = _extract_branch_candidates(
        preprocessed_prob,
        elongated_low_mask & ~dominant_lock_mask,
        global_high_mask & ~dominant_lock_mask,
        branch='elongated',
        min_area=int(cfg.elongated_min_area),
        require_high_core=True,
        acceptance_fn=_elongated_accept,
        start_candidate_id=next_debug_id,
    )
    debug_rows.extend(elongated_debug)

    small_low_threshold = max(float(cfg.small_mean_floor), low_threshold * float(cfg.small_low_scale))
    small_high_threshold = max(float(cfg.small_peak_floor), high_threshold * float(cfg.small_high_scale))
    small_low_mask = np.asarray(preprocessed_prob >= small_low_threshold, dtype=bool)
    small_high_mask = np.asarray(preprocessed_prob >= small_high_threshold, dtype=bool)

    def _small_accept(candidate: _PolygonConfidenceCandidate, has_high_core: bool) -> tuple[bool, tuple[str, ...]]:
        notes: list[str] = []
        peak_ok = candidate.peak_probability >= max(float(cfg.small_peak_floor), small_high_threshold)
        mean_ok = candidate.mean_probability >= float(cfg.small_mean_floor)
        if not (has_high_core or peak_ok):
            notes.append('small_missing_peak')
            return False, tuple(notes)
        if not mean_ok:
            notes.append('small_mean_too_low')
            return False, tuple(notes)
        return True, tuple(notes)

    small_candidates, small_mask, small_debug, next_debug_id = _extract_branch_candidates(
        preprocessed_prob,
        small_low_mask & ~dominant_lock_mask,
        small_high_mask & ~dominant_lock_mask,
        branch='small_weak',
        min_area=max(1, int(cfg.small_min_area)),
        max_area=max(1, int(cfg.small_max_area)),
        require_high_core=False,
        acceptance_fn=_small_accept,
        start_candidate_id=next_debug_id,
    )
    debug_rows.extend(small_debug)

    adaptive_background = _local_mean_map(preprocessed_prob, radius=max(1, int(cfg.adaptive_radius)))
    adaptive_response = np.asarray(preprocessed_prob - adaptive_background, dtype=np.float32)
    adaptive_low_mask = np.asarray(adaptive_response >= float(cfg.adaptive_low_offset), dtype=bool)
    adaptive_high_mask = np.asarray(adaptive_response >= float(cfg.adaptive_high_offset), dtype=bool)

    def _adaptive_accept(candidate: _PolygonConfidenceCandidate, has_high_core: bool) -> tuple[bool, tuple[str, ...]]:
        notes: list[str] = []
        if not has_high_core:
            notes.append('adaptive_missing_high_core')
            return False, tuple(notes)
        if candidate.mean_probability < max(float(cfg.small_mean_floor) * 0.9, 0.12) and candidate.peak_probability < max(float(cfg.small_peak_floor), 0.18):
            notes.append('adaptive_signal_too_low')
            return False, tuple(notes)
        reject, reject_notes = _should_reject_branch_spill(
            preprocessed_prob,
            branch_contrast_map,
            candidate,
            strong_threshold=high_threshold,
            config=cfg,
        )
        if reject:
            notes.extend(reject_notes)
            return False, tuple(notes)
        return True, tuple(notes)

    adaptive_candidates, adaptive_mask, adaptive_debug, next_debug_id = _extract_branch_candidates(
        preprocessed_prob,
        adaptive_low_mask & ~dominant_lock_mask,
        adaptive_high_mask & ~dominant_lock_mask,
        branch='adaptive_local',
        min_area=max(1, int(cfg.small_min_area)),
        require_high_core=True,
        acceptance_fn=_adaptive_accept,
        start_candidate_id=next_debug_id,
    )
    debug_rows.extend(adaptive_debug)

    candidates = list(large_polygon_candidates) + list(global_candidates) + list(elongated_candidates) + list(small_candidates) + list(adaptive_candidates)
    if not candidates and np.any(np.asarray(strong_mask, dtype=bool)):
        fallback_candidate = _make_polygon_candidate(raw_prob, np.asarray(strong_mask, dtype=bool), ('strong_mask',))
        if fallback_candidate is not None:
            candidates = [fallback_candidate]
    stage_timings_ms['branch_extract'] = 1000.0 * (perf_counter() - branch_started)

    merge_started = perf_counter()
    merged_candidates = _merge_polygon_candidates(raw_prob, candidates, cfg)
    stage_timings_ms['initial_merge'] = 1000.0 * (perf_counter() - merge_started)

    completion_started = perf_counter()
    seed_mask = global_high_mask | small_high_mask | adaptive_high_mask | np.asarray(strong_mask, dtype=bool)
    completed_candidates = _complete_polygon_candidates(
        raw_prob,
        merged_candidates,
        strong_threshold=high_threshold,
        weak_threshold=low_threshold,
        high_seed_mask=seed_mask,
    )
    stage_timings_ms['completion'] = 1000.0 * (perf_counter() - completion_started)

    split_started = perf_counter()
    split_candidates: list[_PolygonConfidenceCandidate] = []
    separation_boundary_cues = np.zeros_like(preprocessed_prob, dtype=np.float32)
    separation_core_mask = np.zeros_like(global_low_mask, dtype=bool)
    separation_candidate_region = np.zeros_like(global_low_mask, dtype=bool)
    separation_thin_barrier = np.zeros_like(global_low_mask, dtype=bool)
    separation_bridge_cuts = np.zeros_like(global_low_mask, dtype=bool)
    separation_barrier_stops = np.zeros_like(global_low_mask, dtype=bool)
    for candidate in completed_candidates:
        if 'large_polygon' in set(candidate.source_branches):
            split_candidates.append(candidate)
            continue
        separated, separation_debug = _split_polygon_candidate_by_barriers(
            raw_prob,
            candidate,
            seed_mask,
            low_threshold=low_threshold,
            strong_threshold=high_threshold,
            config=cfg,
            include_debug=include_debug,
        )
        split_candidates.extend(separated)
        if include_debug and separation_debug is not None:
            separation_boundary_cues = np.maximum(
                separation_boundary_cues,
                np.asarray(separation_debug.get("boundary_cues"), dtype=np.float32),
            )
            separation_core_mask |= np.asarray(separation_debug.get("core_seeds"), dtype=bool)
            separation_candidate_region |= np.asarray(separation_debug.get("candidate_region"), dtype=bool)
            separation_thin_barrier |= np.asarray(separation_debug.get("thin_barrier"), dtype=bool)
            separation_bridge_cuts |= np.asarray(separation_debug.get("bridge_cuts"), dtype=bool)
            separation_barrier_stops |= np.asarray(separation_debug.get("barrier_blocked"), dtype=bool)
    stage_timings_ms['split_refine'] = 1000.0 * (perf_counter() - split_started)

    final_merge_started = perf_counter()
    final_candidates = _merge_polygon_candidates(raw_prob, split_candidates or completed_candidates, cfg)
    final_candidates = _refine_final_polygon_candidates(
        raw_prob,
        final_candidates,
        strong_mask=np.asarray(strong_mask, dtype=bool),
        strong_threshold=high_threshold,
        config=cfg,
    )
    stage_timings_ms['final_merge'] = 1000.0 * (perf_counter() - final_merge_started)

    final_mask = np.zeros_like(np.asarray(strong_mask, dtype=bool), dtype=bool)
    for candidate in final_candidates:
        final_mask |= np.asarray(candidate.mask, dtype=bool)

    if not np.any(final_mask) and np.any(np.asarray(strong_mask, dtype=bool)) and not candidates:
        final_mask = np.asarray(strong_mask, dtype=bool).copy()
        fallback_candidate = _make_polygon_candidate(raw_prob, final_mask, ('strong_mask',))
        final_candidates = [fallback_candidate] if fallback_candidate is not None else []

    object_labels = _candidate_object_labels(final_candidates, final_mask.shape) if final_candidates else np.zeros(final_mask.shape, dtype=np.int32)

    debug_data = None
    if include_debug:
        final_debug_rows = list(debug_rows)
        for object_index, candidate in enumerate(final_candidates, start=1):
            final_debug_rows.append(PolygonConfidenceDebugCandidate(
                object_id=int(object_index),
                branch='merged',
                source_branches=tuple(candidate.source_branches),
                accepted=True,
                area=int(candidate.area),
                bbox_x=int(candidate.bbox[0]),
                bbox_y=int(candidate.bbox[1]),
                bbox_width=int(candidate.bbox[2]),
                bbox_height=int(candidate.bbox[3]),
                aspect_ratio=float(candidate.aspect_ratio),
                elongation=float(candidate.elongation),
                peak_probability=float(candidate.peak_probability),
                mean_probability=float(candidate.mean_probability),
                extent=float(candidate.extent),
                notes=('final_object',),
            ))
        debug_data = PolygonConfidenceDebugData(
            preprocessed_probability=np.asarray(preprocessed_prob, dtype=np.float32),
            locally_normalized_probability=np.asarray(local_normalized_prob, dtype=np.float32),
            boundary_cues=np.asarray(separation_boundary_cues, dtype=np.float32),
            low_mask=np.asarray(global_low_mask, dtype=bool),
            high_mask=np.asarray(global_high_mask, dtype=bool),
            adaptive_low_mask=np.asarray(adaptive_low_mask, dtype=bool),
            adaptive_high_mask=np.asarray(adaptive_high_mask, dtype=bool),
            core_seed_mask=np.asarray(separation_core_mask, dtype=bool),
            candidate_region_mask=np.asarray(separation_candidate_region, dtype=bool),
            thin_barrier_map=np.asarray(separation_thin_barrier, dtype=bool),
            bridge_cut_mask=np.asarray(separation_bridge_cuts, dtype=bool),
            barrier_stop_mask=np.asarray(separation_barrier_stops, dtype=bool),
            branch_masks={
                'large_polygon': np.asarray(large_polygon_mask, dtype=bool),
                'dominant_clear': np.asarray(dominant_mask, dtype=bool),
                'global_hysteresis': np.asarray(global_mask, dtype=bool),
                'elongated': np.asarray(elongated_mask, dtype=bool),
                'small_weak': np.asarray(small_mask, dtype=bool),
                'adaptive_local': np.asarray(adaptive_mask, dtype=bool),
                'core_seeds': np.asarray(separation_core_mask, dtype=bool),
                'candidate_region': np.asarray(separation_candidate_region, dtype=bool),
                'thin_barrier': np.asarray(separation_thin_barrier, dtype=bool),
                'bridge_cuts': np.asarray(separation_bridge_cuts, dtype=bool),
                'barrier_stops': np.asarray(separation_barrier_stops, dtype=bool),
            },
            merged_mask=np.asarray(final_mask, dtype=bool),
            object_labels=np.asarray(object_labels, dtype=np.int32),
            candidate_rows=tuple(final_debug_rows),
            timings_ms={name: float(value) for name, value in stage_timings_ms.items()},
        )

    return np.asarray(final_mask, dtype=bool), tuple(final_candidates), debug_data


def _polygon_confidence_support_mask(
    probability: np.ndarray,
    strong_mask: np.ndarray,
    *,
    strong_threshold: float = 0.5,
    config: PolygonConfidencePipelineConfig | None = None,
) -> np.ndarray:
    final_mask, _candidates, _debug = _polygon_confidence_pipeline(
        probability,
        strong_mask,
        strong_threshold=float(strong_threshold),
        config=config,
        include_debug=False,
    )
    return np.asarray(final_mask, dtype=bool)


def _local_contrast_map(probability: np.ndarray, radius: int = 1) -> np.ndarray:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    local_radius = max(1, int(radius))
    if ndi is not None:
        size = 2 * local_radius + 1
        local_max = np.asarray(ndi.maximum_filter(prob, size=size, mode='nearest'), dtype=np.float32)
        local_min = np.asarray(ndi.minimum_filter(prob, size=size, mode='nearest'), dtype=np.float32)
        return np.clip(local_max - local_min, 0.0, 1.0).astype(np.float32)
    padded = np.pad(prob, local_radius, mode='edge')
    local_max = np.empty_like(prob, dtype=np.float32)
    local_min = np.empty_like(prob, dtype=np.float32)
    for y in range(prob.shape[0]):
        for x in range(prob.shape[1]):
            patch = padded[y:y + 2 * local_radius + 1, x:x + 2 * local_radius + 1]
            local_max[y, x] = float(np.max(patch))
            local_min[y, x] = float(np.min(patch))
    return np.clip(local_max - local_min, 0.0, 1.0).astype(np.float32)


def _local_mean_map(array: np.ndarray, radius: int = 1) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    local_radius = max(1, int(radius))
    if ndi is not None:
        size = 2 * local_radius + 1
        return np.asarray(ndi.uniform_filter(values, size=size, mode='nearest'), dtype=np.float32)
    padded = np.pad(values, local_radius, mode='edge')
    result = np.empty_like(values, dtype=np.float32)
    patch_size = float((2 * local_radius + 1) ** 2)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            patch = padded[y:y + 2 * local_radius + 1, x:x + 2 * local_radius + 1]
            result[y, x] = float(np.sum(patch, dtype=np.float64) / patch_size)
    return result


def _polygon_transition_uncertainty_maps(probability: np.ndarray, support_mask: np.ndarray, *, contrast_radius: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    support = np.asarray(support_mask, dtype=bool)
    local_radius = max(1, int(contrast_radius))
    uncertainty_map = _uncertainty_map_from_probability(prob)
    local_contrast = _local_contrast_map(prob, radius=local_radius)
    width_raw = _local_mean_map(uncertainty_map, radius=local_radius)
    sample_mask = np.asarray(_binary_dilate(support, radius=local_radius), dtype=bool) if np.any(support) else np.ones_like(support, dtype=bool)

    def _scale(values: np.ndarray) -> float:
        valid = np.asarray(values, dtype=np.float32)
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            return 1.0
        scale = float(np.percentile(valid, 95.0))
        if scale <= EPS:
            scale = float(np.max(valid)) if valid.size else 1.0
        return max(scale, EPS)

    width_scale = _scale(width_raw[sample_mask])
    contrast_scale = _scale(local_contrast[sample_mask])
    transition_width_map = np.clip(width_raw / width_scale, 0.0, 1.0).astype(np.float32)
    inverse_local_contrast = (1.0 - np.clip(local_contrast / contrast_scale, 0.0, 1.0)).astype(np.float32)
    transition_uncertainty_map = np.clip(
        uncertainty_map * transition_width_map * (0.35 + 0.65 * inverse_local_contrast),
        0.0,
        1.0,
    ).astype(np.float32)
    return transition_width_map, inverse_local_contrast, transition_uncertainty_map


def _polygon_internal_confidence(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    boundary_radius: int = 1,
    uncertainty_delta: float = MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    summary_metric: str = POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
    strong_threshold: float = 0.5,
    include_objects: bool = True,
    include_debug: bool = False,
    config: PolygonConfidencePipelineConfig | None = None,
) -> PolygonConfidenceMetrics:
    prob = _internal_confidence_probability_map(probability, support_mask=np.asarray(mask, dtype=bool))
    pipeline_config = config or _polygon_confidence_config()
    mask_bool, final_candidates, debug_data = _polygon_confidence_pipeline(
        prob,
        np.asarray(mask, dtype=bool),
        strong_threshold=float(strong_threshold),
        config=pipeline_config,
        include_debug=include_debug,
    )
    confidence_map = _confidence_map_from_probability(prob)
    object_count = int(np.count_nonzero(mask_bool))
    area_fraction = float(object_count / max(1, mask_bool.size))
    frame_uncertainty_score, uncertain_support_fraction, top_uncertainty_mean, largest_uncertain_region_fraction = _frame_uncertainty_components_from_probability(
        prob,
        support_threshold=POLYGON_SUPPORT_THRESHOLD,
    )
    normalized_summary_metric = str(summary_metric or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED).strip().lower()
    if normalized_summary_metric != POLYGON_CONFIDENCE_SUMMARY_CORE:
        normalized_summary_metric = POLYGON_CONFIDENCE_SUMMARY_WEIGHTED
    if object_count <= 0:
        return PolygonConfidenceMetrics(
            frame_uncertainty_score=0.0,
            uncertain_support_fraction=0.0,
            top_uncertainty_mean=0.0,
            largest_uncertain_region_fraction=0.0,
            mean_object_confidence=0.0,
            mean_core_confidence=0.0,
            mean_boundary_uncertainty=0.0,
            mean_weighted_confidence=0.0,
            mean_object_probability=0.0,
            uncertain_fraction=0.0,
            mean_transition_width=0.0,
            object_area_fraction=area_fraction,
            polygon_count=0,
            summary_metric=normalized_summary_metric,
            objects=(),
            debug_data=debug_data if include_debug else None,
        )

    object_rows: list[PolygonObjectConfidence] = []
    aggregate_summary: list[tuple[float, float]] = []
    aggregate_core: list[tuple[float, float]] = []
    aggregate_boundary: list[tuple[float, float]] = []
    aggregate_weighted: list[tuple[float, float]] = []
    aggregate_probability: list[tuple[float, float]] = []
    aggregate_uncertain: list[tuple[float, float]] = []
    aggregate_transition_width: list[tuple[float, float]] = []
    object_total = 0
    delta = float(max(EPS, uncertainty_delta))
    erosion_radius = max(0, int(boundary_radius))
    transition_width_map, _inverse_local_contrast, transition_uncertainty_map = _polygon_transition_uncertainty_maps(
        prob,
        mask_bool,
        contrast_radius=max(1, erosion_radius or 1),
    )

    candidate_entries: list[tuple[int, np.ndarray, tuple[str, ...], tuple[int, int, int, int], float, float]] = []
    if final_candidates:
        for object_id, candidate in enumerate(final_candidates, start=1):
            object_mask = np.asarray(candidate.mask, dtype=bool) & mask_bool
            if not np.any(object_mask):
                continue
            candidate_entries.append((
                int(object_id),
                object_mask,
                tuple(candidate.source_branches) or ('merged',),
                tuple(candidate.bbox),
                float(candidate.aspect_ratio),
                float(candidate.elongation),
            ))
    else:
        labels, label_count = _label_components(mask_bool)
        for label_id in range(1, int(label_count) + 1):
            object_mask = labels == label_id
            if not np.any(object_mask):
                continue
            bbox = _mask_bbox(object_mask)
            _area, _bbox, aspect_ratio, elongation, _extent = _mask_geometry(object_mask)
            candidate_entries.append((
                int(label_id),
                object_mask,
                ('merged',),
                tuple(bbox),
                float(aspect_ratio),
                float(elongation),
            ))

    for object_id, object_mask, source_branches, bbox, aspect_ratio, elongation in candidate_entries:
        area = int(np.count_nonzero(object_mask))
        if area <= 0:
            continue
        object_total += 1
        morphological_interior = _binary_erode(object_mask, erosion_radius) & object_mask if erosion_radius > 0 else object_mask.copy()

        object_conf = np.asarray(confidence_map[object_mask], dtype=np.float32)
        object_prob = np.asarray(prob[object_mask], dtype=np.float32)
        object_peak = float(np.max(object_prob)) if object_prob.size > 0 else 0.0
        object_mean = float(np.mean(object_prob, dtype=np.float64)) if object_prob.size > 0 else 0.0
        plateau_threshold = max(0.5, object_mean + 0.25 * max(0.0, object_peak - object_mean))
        plateau_mask = object_mask & (prob >= plateau_threshold)
        if np.any(morphological_interior & plateau_mask):
            interior = morphological_interior & plateau_mask
        elif np.any(plateau_mask):
            interior = plateau_mask
        elif np.any(morphological_interior):
            interior = morphological_interior
        else:
            interior = object_mask.copy()

        boundary_band = object_mask & np.logical_not(interior)
        if not np.any(boundary_band):
            boundary_band = object_mask.copy()

        core_confidence = float(np.mean(confidence_map[interior], dtype=np.float64))
        mean_confidence = float(np.mean(object_conf, dtype=np.float64)) if object_conf.size > 0 else 0.0
        median_confidence = float(np.median(object_conf)) if object_conf.size > 0 else 0.0
        min_confidence = float(np.min(object_conf)) if object_conf.size > 0 else 0.0
        max_confidence = float(np.max(object_conf)) if object_conf.size > 0 else 0.0
        low_percentile_confidence = float(np.percentile(object_conf, 25.0)) if object_conf.size > 0 else 0.0
        boundary_transition_uncertainty = np.asarray(transition_uncertainty_map[boundary_band], dtype=np.float32)
        boundary_uncertainty = float(np.mean(boundary_transition_uncertainty, dtype=np.float64))
        boundary_transition_width = np.asarray(transition_width_map[boundary_band], dtype=np.float32)
        if boundary_transition_width.size > 0 and boundary_transition_uncertainty.size > 0:
            transition_width_mean = float(np.mean(boundary_transition_width * boundary_transition_uncertainty, dtype=np.float64))
        else:
            transition_width_mean = 0.0
        if ndi is not None:
            weights = np.asarray(ndi.distance_transform_edt(object_mask), dtype=np.float32)[object_mask]
        else:
            weights = np.ones(area, dtype=np.float32)
        if weights.size == 0 or float(np.max(weights)) <= EPS:
            normalized_weights = np.ones(area, dtype=np.float32)
        else:
            normalized_weights = np.clip(weights / float(np.max(weights)), EPS, 1.0).astype(np.float32)
        weighted_confidence = float(np.average(object_conf, weights=normalized_weights))
        mean_probability = float(np.mean(object_prob, dtype=np.float64))
        uncertain_fraction = float(np.mean(np.abs(object_prob - 0.5) < delta, dtype=np.float64))
        ys, xs = np.nonzero(object_mask)
        centroid_x = float(np.mean(xs, dtype=np.float64)) if xs.size else 0.0
        centroid_y = float(np.mean(ys, dtype=np.float64)) if ys.size else 0.0
        source_branch = source_branches[0] if len(source_branches) == 1 else 'merged'
        summary_confidence = weighted_confidence if normalized_summary_metric == POLYGON_CONFIDENCE_SUMMARY_WEIGHTED else core_confidence
        if include_objects:
            object_rows.append(PolygonObjectConfidence(
                object_id=int(object_id),
                area=area,
                area_fraction=float(area / max(1, mask_bool.size)),
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                core_confidence=core_confidence,
                boundary_uncertainty=boundary_uncertainty,
                weighted_confidence=weighted_confidence,
                summary_confidence=summary_confidence,
                mean_probability=mean_probability,
                mean_confidence=mean_confidence,
                median_confidence=median_confidence,
                min_confidence=min_confidence,
                max_confidence=max_confidence,
                low_percentile_confidence=low_percentile_confidence,
                uncertain_fraction=uncertain_fraction,
                transition_width_mean=transition_width_mean,
                bbox_x=int(bbox[0]),
                bbox_y=int(bbox[1]),
                bbox_width=int(bbox[2]),
                bbox_height=int(bbox[3]),
                aspect_ratio=float(aspect_ratio),
                elongation=float(elongation),
                source_branch=source_branch,
                source_branches=tuple(source_branches),
            ))
        weight = float(area)
        aggregate_summary.append((summary_confidence, weight))
        aggregate_core.append((core_confidence, weight))
        aggregate_boundary.append((boundary_uncertainty, weight))
        aggregate_weighted.append((weighted_confidence, weight))
        aggregate_probability.append((mean_probability, weight))
        aggregate_uncertain.append((uncertain_fraction, weight))
        aggregate_transition_width.append((transition_width_mean, weight))

    return PolygonConfidenceMetrics(
        frame_uncertainty_score=frame_uncertainty_score,
        uncertain_support_fraction=uncertain_support_fraction,
        top_uncertainty_mean=top_uncertainty_mean,
        largest_uncertain_region_fraction=largest_uncertain_region_fraction,
        mean_object_confidence=_weighted_mean(aggregate_summary),
        mean_core_confidence=_weighted_mean(aggregate_core),
        mean_boundary_uncertainty=_weighted_mean(aggregate_boundary),
        mean_weighted_confidence=_weighted_mean(aggregate_weighted),
        mean_object_probability=_weighted_mean(aggregate_probability),
        uncertain_fraction=_weighted_mean(aggregate_uncertain),
        mean_transition_width=_weighted_mean(aggregate_transition_width),
        object_area_fraction=area_fraction,
        polygon_count=int(object_total),
        summary_metric=normalized_summary_metric,
        objects=tuple(object_rows),
        debug_data=debug_data if include_debug else None,
    )


def _point_confidence_patch(array: np.ndarray, x: float, y: float, radius: int) -> np.ndarray:
    patch_radius = max(0, int(radius))
    px = int(round(float(x)))
    py = int(round(float(y)))
    if array.ndim != 2 or array.size == 0 or py < 0 or py >= array.shape[0] or px < 0 or px >= array.shape[1]:
        return np.zeros((0, 0), dtype=np.float32)
    y0 = max(0, py - patch_radius)
    y1 = min(array.shape[0], py + patch_radius + 1)
    x0 = max(0, px - patch_radius)
    x1 = min(array.shape[1], px + patch_radius + 1)
    return np.asarray(array[y0:y1, x0:x1], dtype=np.float32)


def _point_local_contrast(probability: np.ndarray, x: float, y: float, radius: int = POINT_CONFIDENCE_NEIGHBOR_RADIUS) -> float:
    prob = np.asarray(probability, dtype=np.float32)
    px = int(round(float(x)))
    py = int(round(float(y)))
    if prob.ndim != 2 or prob.size == 0 or py < 0 or py >= prob.shape[0] or px < 0 or px >= prob.shape[1]:
        return 0.0
    patch = _point_confidence_patch(prob, x, y, radius)
    if patch.size <= 1:
        return 0.0
    center_value = float(prob[py, px])
    patch_sum = float(np.sum(patch, dtype=np.float64)) - center_value
    neighbor_count = max(1, int(patch.size - 1))
    return float(center_value - patch_sum / float(neighbor_count))


def _point_internal_confidence(prediction_view: object, *, neighborhood_radius: int = POINT_CONFIDENCE_NEIGHBOR_RADIUS, include_objects: bool = True) -> PointConfidenceMetrics:
    probability = _internal_confidence_probability_map(
        _prob_from_gray(np.asarray(getattr(prediction_view, 'pred_gray'), dtype=np.uint8)),
        support_mask=np.asarray(getattr(prediction_view, 'pred_bin'), dtype=bool),
    )
    confidence_map = _confidence_map_from_probability(probability)
    points = tuple(getattr(prediction_view, 'points', ()))
    if not points:
        return PointConfidenceMetrics(
            frame_uncertainty_score=0.0,
            uncertain_support_fraction=0.0,
            top_uncertainty_mean=0.0,
            largest_uncertain_region_fraction=0.0,
            mean_point_confidence=0.0,
            mean_center_confidence=0.0,
            mean_local_confidence=0.0,
            mean_point_probability=0.0,
            mean_point_contrast=0.0,
            point_count=0,
            objects=(),
        )
    object_rows: list[PointObjectConfidence] = []
    center_confidences: list[float] = []
    local_confidences: list[float] = []
    point_probs: list[float] = []
    point_contrasts: list[float] = []
    point_weights: list[float] = []
    point_coordinates: list[tuple[float, float, float]] = []
    local_radius = max(0, int(neighborhood_radius))
    for index, point in enumerate(points, start=1):
        x = float(getattr(point, 'x', 0.0))
        y = float(getattr(point, 'y', 0.0))
        px = int(round(x))
        py = int(round(y))
        if py < 0 or py >= probability.shape[0] or px < 0 or px >= probability.shape[1]:
            point_prob = 0.0
            center_confidence = 0.0
        else:
            point_prob = float(probability[py, px])
            center_confidence = float(confidence_map[py, px])
        local_patch = _point_confidence_patch(confidence_map, x, y, local_radius)
        local_confidence = float(np.mean(local_patch, dtype=np.float64)) if local_patch.size > 0 else center_confidence
        local_contrast = _point_local_contrast(probability, x, y, radius=local_radius)
        radius = max(1.0, float(getattr(point, 'radius', 0.0)))
        if include_objects:
            object_rows.append(PointObjectConfidence(
                object_id=int(index),
                x=x,
                y=y,
                radius=radius,
                point_probability=point_prob,
                center_confidence=center_confidence,
                local_confidence=local_confidence,
                local_contrast=local_contrast,
            ))
        center_confidences.append(center_confidence)
        local_confidences.append(local_confidence)
        point_probs.append(point_prob)
        point_contrasts.append(local_contrast)
        point_weights.append(float(max(0.0, min(1.0, (point_prob - float(POINT_SUPPORT_THRESHOLD)) / max(EPS, 1.0 - float(POINT_SUPPORT_THRESHOLD))))) if point_prob >= float(POINT_SUPPORT_THRESHOLD) else 0.0)
        point_coordinates.append((x, y, radius))
    confidence_array = np.asarray(center_confidences, dtype=np.float64)
    weight_array = np.asarray(point_weights, dtype=np.float64)
    weight_sum = float(np.sum(weight_array, dtype=np.float64))
    frame_uncertainty_score, uncertain_support_fraction, top_uncertainty_mean, largest_uncertain_region_fraction = _frame_uncertainty_components_from_points(
        np.asarray(point_probs, dtype=np.float32),
        tuple(point_coordinates),
        support_threshold=POINT_SUPPORT_THRESHOLD,
    )
    return PointConfidenceMetrics(
        frame_uncertainty_score=frame_uncertainty_score,
        uncertain_support_fraction=uncertain_support_fraction,
        top_uncertainty_mean=top_uncertainty_mean,
        largest_uncertain_region_fraction=largest_uncertain_region_fraction,
        mean_point_confidence=float(np.sum(confidence_array * weight_array, dtype=np.float64) / max(EPS, weight_sum)) if weight_sum > 0.0 else 0.0,
        mean_center_confidence=float(np.mean(confidence_array, dtype=np.float64)),
        mean_local_confidence=float(np.mean(np.asarray(local_confidences, dtype=np.float64))),
        mean_point_probability=float(np.mean(np.asarray(point_probs, dtype=np.float64))),
        mean_point_contrast=float(np.mean(np.asarray(point_contrasts, dtype=np.float64))),
        point_count=int(len(points)),
        objects=tuple(object_rows),
    )


def _resize_like(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if tuple(int(v) for v in array.shape) == tuple(int(v) for v in target_shape):
        return np.asarray(array)
    return np.asarray(resize_grayscale_image(np.asarray(array, dtype=np.uint8), target_shape))


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.size == 0 or not np.any(mask_bool):
        return np.zeros_like(mask_bool, dtype=np.int32), 0
    if ndi is not None:
        labels, count = ndi.label(mask_bool, structure=np.ones((3, 3), dtype=np.uint8))
        return np.asarray(labels, dtype=np.int32), int(count)
    if cv2 is not None:
        count, labels = cv2.connectedComponents(np.asarray(mask_bool, dtype=np.uint8), connectivity=8)
        return np.asarray(labels, dtype=np.int32), max(0, int(count) - 1)
    height, width = mask_bool.shape
    labels = np.zeros((height, width), dtype=np.int32)
    next_label = 1
    for row in range(height):
        for column in range(width):
            if not mask_bool[row, column] or labels[row, column] != 0:
                continue
            queue = [(row, column)]
            labels[row, column] = next_label
            while queue:
                current_row, current_column = queue.pop()
                for neighbor_row in range(max(0, current_row - 1), min(height, current_row + 2)):
                    for neighbor_column in range(max(0, current_column - 1), min(width, current_column + 2)):
                        if not mask_bool[neighbor_row, neighbor_column] or labels[neighbor_row, neighbor_column] != 0:
                            continue
                        labels[neighbor_row, neighbor_column] = next_label
                        queue.append((neighbor_row, neighbor_column))
            next_label += 1
    return labels, next_label - 1


def _binary_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask_bool.copy()
    if ndi is not None:
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        return np.asarray(ndi.binary_erosion(mask_bool, structure=structure), dtype=bool)
    padded = np.pad(mask_bool, radius, mode="constant", constant_values=False)
    result = np.ones_like(mask_bool, dtype=bool)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            result &= padded[row_offset:row_offset + mask_bool.shape[0], column_offset:column_offset + mask_bool.shape[1]]
    return result


def _binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask_bool.copy()
    if ndi is not None:
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        return np.asarray(ndi.binary_dilation(mask_bool, structure=structure), dtype=bool)
    padded = np.pad(mask_bool, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask_bool)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            result |= padded[row_offset:row_offset + mask_bool.shape[0], column_offset:column_offset + mask_bool.shape[1]]
    return result


def _binary_dilate_rect(mask: np.ndarray, radius_y: int = 1, radius_x: int = 1) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    ry = max(0, int(radius_y))
    rx = max(0, int(radius_x))
    if ry <= 0 and rx <= 0:
        return mask_bool.copy()
    if ndi is not None:
        structure = np.ones((2 * ry + 1, 2 * rx + 1), dtype=bool)
        return np.asarray(ndi.binary_dilation(mask_bool, structure=structure), dtype=bool)
    padded = np.pad(mask_bool, ((ry, ry), (rx, rx)), mode="constant", constant_values=False)
    result = np.zeros_like(mask_bool)
    for row_offset in range(2 * ry + 1):
        for column_offset in range(2 * rx + 1):
            result |= padded[row_offset:row_offset + mask_bool.shape[0], column_offset:column_offset + mask_bool.shape[1]]
    return result


def _completion_radii_for_mask(mask: np.ndarray, base_radius: int) -> tuple[int, int]:
    radius = max(0, int(base_radius))
    if radius <= 0:
        return 0, 0
    mask_bool = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0 or xs.size == 0:
        return radius, radius
    height = int(np.max(ys) - np.min(ys) + 1)
    width = int(np.max(xs) - np.min(xs) + 1)
    axis_ratio = float(max(height, width) / max(1, min(height, width)))
    major_scale = max(1, int(POLYGON_CONFIDENCE_COMPLETION_MAJOR_SCALE))
    if axis_ratio >= float(POLYGON_CONFIDENCE_COMPLETION_AXIS_RATIO):
        if height >= width:
            return radius * major_scale, radius
        return radius, radius * major_scale
    return radius, radius


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.size == 0:
        return np.zeros_like(mask_bool)
    padded = np.pad(mask_bool, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    interior = center & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return center & np.logical_not(interior)


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if ndi is not None:
        return np.asarray(ndi.distance_transform_edt(mask_bool), dtype=np.float32)
    if cv2 is not None:
        distances = cv2.distanceTransform(np.asarray(mask_bool, dtype=np.uint8), cv2.DIST_L2, 5)
        return np.asarray(distances, dtype=np.float32)
    raise RuntimeError("Distance transform backend is unavailable")


_NEIGHBOR_SLICE_ORDER = (
    (slice(0, -2), slice(0, -2)),
    (slice(0, -2), slice(1, -1)),
    (slice(0, -2), slice(2, None)),
    (slice(1, -1), slice(0, -2)),
    (slice(1, -1), slice(2, None)),
    (slice(2, None), slice(0, -2)),
    (slice(2, None), slice(1, -1)),
    (slice(2, None), slice(2, None)),
)


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=np.uint8), 1, mode="constant", constant_values=0)
    neighbors = np.zeros_like(mask, dtype=np.uint8)
    for row_slice, column_slice in _NEIGHBOR_SLICE_ORDER:
        neighbors += padded[row_slice, column_slice]
    return neighbors


def _transition_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=np.uint8), 1, mode="constant", constant_values=0)
    p2 = padded[:-2, 1:-1]
    p3 = padded[:-2, 2:]
    p4 = padded[1:-1, 2:]
    p5 = padded[2:, 2:]
    p6 = padded[2:, 1:-1]
    p7 = padded[2:, :-2]
    p8 = padded[1:-1, :-2]
    p9 = padded[:-2, :-2]
    sequence = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
    transitions = np.zeros_like(mask, dtype=np.uint8)
    for left, right in zip(sequence[:-1], sequence[1:]):
        transitions += ((left == 0) & (right == 1)).astype(np.uint8)
    return transitions


def skeletonize(mask: np.ndarray) -> np.ndarray:
    image = np.asarray(mask, dtype=np.uint8).copy()
    if image.size == 0:
        return image.astype(bool)
    changed = True
    while changed:
        changed = False
        neighbors = _neighbor_count(image)
        transitions = _transition_count(image)
        padded = np.pad(image, 1, mode="constant", constant_values=0)
        p2 = padded[:-2, 1:-1]
        p4 = padded[1:-1, 2:]
        p6 = padded[2:, 1:-1]
        p8 = padded[1:-1, :-2]
        remove = (image == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
        if np.any(remove):
            image[remove] = 0
            changed = True
        neighbors = _neighbor_count(image)
        transitions = _transition_count(image)
        padded = np.pad(image, 1, mode="constant", constant_values=0)
        p2 = padded[:-2, 1:-1]
        p4 = padded[1:-1, 2:]
        p6 = padded[2:, 1:-1]
        p8 = padded[1:-1, :-2]
        remove = (image == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
        if np.any(remove):
            image[remove] = 0
            changed = True
    return image.astype(bool)


def _endpoint_count(skeleton: np.ndarray) -> int:
    if not np.any(skeleton):
        return 0
    neighbors = _neighbor_count(skeleton)
    return int(np.count_nonzero(skeleton & (neighbors == 1)))


def _branchpoint_count(skeleton: np.ndarray) -> int:
    if not np.any(skeleton):
        return 0
    neighbors = _neighbor_count(skeleton)
    return int(np.count_nonzero(skeleton & (neighbors >= 3)))


def _component_area_stats(labels: np.ndarray, count: int) -> tuple[list[float], float]:
    if count <= 0:
        return [], 0.0
    area_counts = np.bincount(np.asarray(labels, dtype=np.int32).ravel(), minlength=count + 1)
    component_areas = [float(value) for value in area_counts[1:count + 1]]
    mean_component_area = float(np.mean(component_areas, dtype=np.float64)) if component_areas else 0.0
    return component_areas, mean_component_area


def _mask_structure(mask: np.ndarray, *, include_skeleton: bool = True) -> dict[str, object]:
    mask_bool = np.asarray(mask, dtype=bool)
    labels, count = _label_components(mask_bool)
    if include_skeleton:
        skeleton = skeletonize(mask_bool)
        if np.any(skeleton):
            skeleton_neighbors = _neighbor_count(skeleton)
            endpoint_count = int(np.count_nonzero(skeleton & (skeleton_neighbors == 1)))
            branchpoint_count = int(np.count_nonzero(skeleton & (skeleton_neighbors >= 3)))
        else:
            skeleton_neighbors = np.zeros_like(mask_bool, dtype=np.uint8)
            endpoint_count = 0
            branchpoint_count = 0
    else:
        skeleton = np.zeros_like(mask_bool, dtype=bool)
        skeleton_neighbors = np.zeros_like(mask_bool, dtype=np.uint8)
        endpoint_count = 0
        branchpoint_count = 0
    _component_areas, mean_component_area = _component_area_stats(labels, count)
    area = float(np.count_nonzero(mask_bool))
    return {
        "labels": labels,
        "component_count": int(count),
        "area_fraction": float(area / max(1, mask_bool.size)),
        "mean_component_area": mean_component_area,
        "has_skeleton": bool(include_skeleton),
        "skeleton": skeleton,
        "skeleton_neighbors": skeleton_neighbors,
        "skeleton_length": float(np.count_nonzero(skeleton)),
        "endpoint_count": endpoint_count,
        "branchpoint_count": branchpoint_count,
    }


def _tiny_component_map_from_structure(structure: dict[str, object], area_threshold: int = 12) -> np.ndarray:
    labels = np.asarray(structure["labels"], dtype=np.int32)
    count = int(structure["component_count"])
    result = np.zeros(labels.shape, dtype=np.float32)
    if count <= 0:
        return result
    area_counts = np.bincount(labels.ravel(), minlength=count + 1)
    small_labels = np.flatnonzero(area_counts[1:count + 1] <= int(area_threshold)) + 1
    for label_id in small_labels:
        result[labels == int(label_id)] = 1.0
    return result


def _tiny_component_map(mask: np.ndarray, area_threshold: int = 12) -> np.ndarray:
    return _tiny_component_map_from_structure(_mask_structure(mask), area_threshold=area_threshold)


def _thin_bridge_map_from_structure(structure: dict[str, object]) -> np.ndarray:
    skeleton = np.asarray(structure["skeleton"], dtype=bool)
    if not np.any(skeleton):
        return np.zeros_like(skeleton, dtype=np.float32)
    neighbors = np.asarray(structure["skeleton_neighbors"], dtype=np.uint8)
    bridge = skeleton & (neighbors == 2)
    return np.asarray(_binary_dilate(bridge, radius=1), dtype=np.float32)


def _thin_bridge_map(mask: np.ndarray) -> np.ndarray:
    return _thin_bridge_map_from_structure(_mask_structure(mask))


def _branchpoint_map_from_structure(structure: dict[str, object]) -> np.ndarray:
    skeleton = np.asarray(structure["skeleton"], dtype=bool)
    neighbors = np.asarray(structure["skeleton_neighbors"], dtype=np.uint8)
    return np.asarray(_binary_dilate(skeleton & (neighbors >= 3), radius=1), dtype=np.float32)


def _branchpoint_map(mask: np.ndarray) -> np.ndarray:
    return _branchpoint_map_from_structure(_mask_structure(mask))


def _endpoint_map_from_structure(structure: dict[str, object]) -> np.ndarray:
    skeleton = np.asarray(structure["skeleton"], dtype=bool)
    neighbors = np.asarray(structure["skeleton_neighbors"], dtype=np.uint8)
    return np.asarray(_binary_dilate(skeleton & (neighbors == 1), radius=1), dtype=np.float32)


def _endpoint_map(mask: np.ndarray) -> np.ndarray:
    return _endpoint_map_from_structure(_mask_structure(mask))


def _boundary_f1(pred: np.ndarray, gt: np.ndarray, radius: int = 1) -> float:
    pred_boundary = _boundary_mask(pred)
    gt_boundary = _boundary_mask(gt)
    if not np.any(pred_boundary) and not np.any(gt_boundary):
        return 1.0
    pred_d = _binary_dilate(pred_boundary, radius=radius)
    gt_d = _binary_dilate(gt_boundary, radius=radius)
    precision = float(np.count_nonzero(pred_boundary & gt_d) / max(1, np.count_nonzero(pred_boundary)))
    recall = float(np.count_nonzero(gt_boundary & pred_d) / max(1, np.count_nonzero(gt_boundary)))
    if precision + recall <= EPS:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = np.asarray(pred, dtype=bool)
    gt_bool = np.asarray(gt, dtype=bool)
    intersection = int(np.count_nonzero(pred_bool & gt_bool))
    total = int(np.count_nonzero(pred_bool)) + int(np.count_nonzero(gt_bool))
    if total == 0:
        return 1.0
    return float((2.0 * intersection + EPS) / (total + EPS))


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = np.asarray(pred, dtype=bool)
    gt_bool = np.asarray(gt, dtype=bool)
    intersection = int(np.count_nonzero(pred_bool & gt_bool))
    union = int(np.count_nonzero(pred_bool | gt_bool))
    if union == 0:
        return 1.0
    return float((intersection + EPS) / (union + EPS))


def _precision(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = np.asarray(pred, dtype=bool)
    gt_bool = np.asarray(gt, dtype=bool)
    tp = int(np.count_nonzero(pred_bool & gt_bool))
    fp = int(np.count_nonzero(pred_bool & np.logical_not(gt_bool)))
    return float((tp + EPS) / (tp + fp + EPS))


def _recall(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool = np.asarray(pred, dtype=bool)
    gt_bool = np.asarray(gt, dtype=bool)
    tp = int(np.count_nonzero(pred_bool & gt_bool))
    fn = int(np.count_nonzero(np.logical_not(pred_bool) & gt_bool))
    return float((tp + EPS) / (tp + fn + EPS))


def _soft_dice(first: np.ndarray, second: np.ndarray) -> float:
    first_arr = np.clip(np.asarray(first, dtype=np.float32), 0.0, 1.0)
    second_arr = np.clip(np.asarray(second, dtype=np.float32), 0.0, 1.0)
    intersection = float(np.sum(first_arr * second_arr, dtype=np.float64))
    total = float(np.sum(first_arr, dtype=np.float64) + np.sum(second_arr, dtype=np.float64))
    if total <= EPS:
        return 1.0
    return float((2.0 * intersection + EPS) / (total + EPS))


def _soft_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_arr = np.clip(np.asarray(first, dtype=np.float32), 0.0, 1.0)
    second_arr = np.clip(np.asarray(second, dtype=np.float32), 0.0, 1.0)
    intersection = float(np.sum(first_arr * second_arr, dtype=np.float64))
    union = float(np.sum(first_arr, dtype=np.float64) + np.sum(second_arr, dtype=np.float64) - intersection)
    if union <= EPS:
        return 1.0
    return float((intersection + EPS) / (union + EPS))


def _ssim_score(first: np.ndarray, second: np.ndarray) -> float:
    first_arr = np.clip(np.asarray(first, dtype=np.float64), 0.0, 1.0)
    second_arr = np.clip(np.asarray(second, dtype=np.float64), 0.0, 1.0)
    if first_arr.size == 0 and second_arr.size == 0:
        return 1.0
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_first = float(first_arr.mean())
    mu_second = float(second_arr.mean())
    sigma_first = float(np.mean((first_arr - mu_first) ** 2, dtype=np.float64))
    sigma_second = float(np.mean((second_arr - mu_second) ** 2, dtype=np.float64))
    sigma_cross = float(np.mean((first_arr - mu_first) * (second_arr - mu_second), dtype=np.float64))
    denominator = (mu_first ** 2 + mu_second ** 2 + c1) * (sigma_first + sigma_second + c2)
    if denominator <= EPS:
        return 1.0 if np.allclose(first_arr, second_arr) else 0.0
    numerator = (2.0 * mu_first * mu_second + c1) * (2.0 * sigma_cross + c2)
    return float(_clip01(numerator / denominator))


def _mae(first: np.ndarray, second: np.ndarray) -> float:
    first_arr = np.asarray(first, dtype=np.float32)
    second_arr = np.asarray(second, dtype=np.float32)
    return float(np.mean(np.abs(first_arr - second_arr), dtype=np.float64)) if first_arr.size else 0.0


def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    first_arr = np.asarray(first, dtype=np.float32)
    second_arr = np.asarray(second, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(first_arr - second_arr), dtype=np.float64))) if first_arr.size else 0.0


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size <= 0:
        return None
    return float(cols.mean(dtype=np.float64)), float(rows.mean(dtype=np.float64))


def _frame_diagonal(shape: tuple[int, int]) -> float:
    return float(max(EPS, math.hypot(float(shape[0]), float(shape[1]))))


def _centroid_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_center = _mask_centroid(first)
    second_center = _mask_centroid(second)
    diagonal = _frame_diagonal(tuple(int(v) for v in np.asarray(first).shape))
    if first_center is None and second_center is None:
        return 0.0
    if first_center is None or second_center is None:
        return diagonal
    return float(math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1]))


def _distance_similarity(distance: float, shape: tuple[int, int]) -> float:
    return float(1.0 - min(1.0, float(distance) / _frame_diagonal(shape)))


def _nearest_distances_between_coordinate_sets(coords_a: np.ndarray, coords_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords_a = np.asarray(coords_a, dtype=np.float32)
    coords_b = np.asarray(coords_b, dtype=np.float32)
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return np.zeros(coords_a.shape[0], dtype=np.float64), np.zeros(coords_b.shape[0], dtype=np.float64)

    def _chunked_nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        pair_budget = 1_000_000
        chunk_size = max(1, min(int(source.shape[0]), int(max(1, pair_budget // max(1, int(target.shape[0]))))))
        nearest = np.empty(source.shape[0], dtype=np.float64)
        for start in range(0, source.shape[0], chunk_size):
            stop = min(source.shape[0], start + chunk_size)
            diff = np.asarray(source[start:stop, None, :] - target[None, :, :], dtype=np.float32)
            squared = np.sum(np.square(diff, dtype=np.float32), axis=2, dtype=np.float64)
            nearest[start:stop] = np.sqrt(np.min(squared, axis=1, initial=np.inf))
        return nearest

    if cKDTree is not None:
        tree_a = cKDTree(coords_a)
        tree_b = cKDTree(coords_b)
        nearest_a = np.asarray(tree_b.query(coords_a, k=1)[0], dtype=np.float64)
        nearest_b = np.asarray(tree_a.query(coords_b, k=1)[0], dtype=np.float64)
        return nearest_a, nearest_b
    return _chunked_nearest_distances(coords_a, coords_b), _chunked_nearest_distances(coords_b, coords_a)


def _hausdorff_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_boundary = _boundary_mask(first)
    second_boundary = _boundary_mask(second)
    shape = tuple(int(v) for v in np.asarray(first_boundary).shape)
    diagonal = _frame_diagonal(shape)
    if not np.any(first_boundary) and not np.any(second_boundary):
        return 0.0
    if not np.any(first_boundary) or not np.any(second_boundary):
        return diagonal
    if ndi is not None or cv2 is not None:
        dist_to_second = _distance_transform(~second_boundary)
        dist_to_first = _distance_transform(~first_boundary)
        directed_first = float(np.max(dist_to_second[first_boundary])) if np.any(first_boundary) else 0.0
        directed_second = float(np.max(dist_to_first[second_boundary])) if np.any(second_boundary) else 0.0
        return float(max(directed_first, directed_second))
    first_points = np.argwhere(first_boundary)
    second_points = np.argwhere(second_boundary)
    if first_points.size == 0 and second_points.size == 0:
        return 0.0
    if first_points.size == 0 or second_points.size == 0:
        return diagonal
    nearest_first, nearest_second = _nearest_distances_between_coordinate_sets(first_points, second_points)
    directed_first = float(np.max(nearest_first, initial=0.0))
    directed_second = float(np.max(nearest_second, initial=0.0))
    return float(max(directed_first, directed_second))


def _structural_penalties(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    pred_structure: dict[str, object] | None = None,
    gt_structure: dict[str, object] | None = None,
) -> tuple[float, int, int, float]:
    current_pred_structure = pred_structure or _mask_structure(pred)
    current_gt_structure = gt_structure or _mask_structure(gt)
    gt_count = int(current_gt_structure["component_count"])
    pred_count = int(current_pred_structure["component_count"])
    delta_cc = float(abs(pred_count - gt_count) / max(1, gt_count))
    if not bool(current_pred_structure.get("has_skeleton", True)) or not bool(current_gt_structure.get("has_skeleton", True)):
        return delta_cc, 0, 0, 0.0

    skeleton_gt = np.asarray(current_gt_structure["skeleton"], dtype=bool)
    supported_gt = skeleton_gt & _binary_dilate(pred, radius=1)
    _supported_labels, supported_count = _label_components(supported_gt)
    _gt_skel_labels, gt_skel_count = _label_components(skeleton_gt)
    break_count = max(0, int(supported_count - gt_skel_count))

    pred_labels = np.asarray(current_pred_structure["labels"], dtype=np.int32)
    false_bridge_count = 0
    if pred_count > 0 and gt_count > 1:
        expanded_gt = _binary_dilate(gt, radius=1)
        expanded_gt_labels, _ = _label_components(expanded_gt)
        for label_id in range(1, pred_count + 1):
            component_mask = pred_labels == label_id
            touched = set(int(value) for value in np.unique(expanded_gt_labels[component_mask]) if int(value) > 0)
            if len(touched) >= 2:
                false_bridge_count += len(touched) - 1

    length_gt = float(current_gt_structure["skeleton_length"])
    length_pred = float(current_pred_structure["skeleton_length"])
    skeleton_delta = float(abs(length_pred - length_gt) / max(1.0, length_gt))
    return delta_cc, int(break_count), int(false_bridge_count), skeleton_delta


def _labeled_quality(
    pred_prob: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    *,
    pred_structure: dict[str, object] | None = None,
    gt_structure: dict[str, object] | None = None,
    boundary_radius: int = 1,
) -> LabeledModelMetrics:
    current_pred_structure = pred_structure or _mask_structure(pred_mask)
    current_gt_structure = gt_structure or _mask_structure(gt_mask)
    gt_float = np.asarray(gt_mask, dtype=np.float32)
    pred_float = np.clip(np.asarray(pred_prob, dtype=np.float32), 0.0, 1.0)

    soft_dice = _soft_dice(pred_float, gt_float)
    soft_iou = _soft_iou(pred_float, gt_float)
    ssim = _ssim_score(pred_float, gt_float)
    dice = _dice(pred_mask, gt_mask)
    iou = _iou(pred_mask, gt_mask)
    precision = _precision(pred_mask, gt_mask)
    recall = _recall(pred_mask, gt_mask)
    boundary_f1 = _boundary_f1(pred_mask, gt_mask, radius=max(1, int(boundary_radius)))
    hausdorff_distance = _hausdorff_distance(pred_mask, gt_mask)
    centroid_distance = _centroid_distance(pred_mask, gt_mask)
    frame_shape = tuple(int(v) for v in np.asarray(pred_mask).shape)
    hausdorff_similarity = _distance_similarity(hausdorff_distance, frame_shape)
    centroid_similarity = _distance_similarity(centroid_distance, frame_shape)
    mae = _mae(pred_float, gt_float)
    rmse = _rmse(pred_float, gt_float)

    delta_cc, break_count, false_bridge_count, skeleton_delta = _structural_penalties(
        pred_mask,
        gt_mask,
        pred_structure=current_pred_structure,
        gt_structure=current_gt_structure,
    )
    pred_count = int(current_pred_structure["component_count"])
    gt_count = int(current_gt_structure["component_count"])
    count_error = float(_clip01(abs(pred_count - gt_count) / max(1.0, float(gt_count))))
    connected_component_error = float(_clip01(abs(pred_count - gt_count) / max(1.0, float(gt_count))))
    quality = _weighted_mean([
        (float(soft_dice), MASK_SUPERVISED_SCORE_WEIGHTS["soft_dice"]),
        (float(soft_iou), MASK_SUPERVISED_SCORE_WEIGHTS["soft_iou"]),
        (float(ssim), MASK_SUPERVISED_SCORE_WEIGHTS["ssim"]),
        (float(dice), MASK_SUPERVISED_SCORE_WEIGHTS["dice"]),
        (float(iou), MASK_SUPERVISED_SCORE_WEIGHTS["iou"]),
        (float(hausdorff_similarity), MASK_SUPERVISED_SCORE_WEIGHTS["hausdorff_term"]),
        (float(centroid_similarity), MASK_SUPERVISED_SCORE_WEIGHTS["centroid_term"]),
    ])
    error = float(1.0 - quality)
    return LabeledModelMetrics(
        soft_dice=float(soft_dice),
        soft_iou=float(soft_iou),
        ssim=float(ssim),
        dice=float(dice),
        iou=float(iou),
        precision=float(precision),
        recall=float(recall),
        count_error=float(count_error),
        connected_component_error=float(connected_component_error),
        hausdorff_distance=float(hausdorff_distance),
        hausdorff_similarity=float(hausdorff_similarity),
        centroid_distance=float(centroid_distance),
        centroid_similarity=float(centroid_similarity),
        mae=float(mae),
        rmse=float(rmse),
        boundary_f1=float(boundary_f1),
        delta_connected_components=float(delta_cc),
        break_count=int(break_count),
        false_bridge_count=int(false_bridge_count),
        skeleton_length_delta=float(skeleton_delta),
        quality_score=float(quality),
        error_score=float(error),
    )


def _point_coordinates(points: tuple[object, ...]) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([[float(getattr(point, "x", 0.0)), float(getattr(point, "y", 0.0))] for point in points], dtype=np.float32)


def _point_match_threshold(point_a: object, point_b: object, base_radius: float) -> float:
    _ = point_a, point_b
    return float(max(0.0, float(base_radius)))


def _match_point_sets(points_a: tuple[object, ...], points_b: tuple[object, ...], base_radius: float) -> tuple[list[float], set[int], set[int]]:
    candidate_pairs: list[tuple[float, int, int]] = []
    if cKDTree is not None and points_a and points_b:
        coords_a = _point_coordinates(points_a)
        coords_b = _point_coordinates(points_b)
        radius = max(0.0, float(base_radius))
        if coords_a.shape[0] > 0 and coords_b.shape[0] > 0 and radius >= 0.0:
            tree_b = cKDTree(coords_b)
            neighbors = tree_b.query_ball_point(coords_a, r=radius)
            for index_a, neighbor_indices in enumerate(neighbors):
                point_a = points_a[index_a]
                for index_b in neighbor_indices:
                    point_b = points_b[int(index_b)]
                    distance = float(np.hypot(
                        float(getattr(point_a, "x", 0.0)) - float(getattr(point_b, "x", 0.0)),
                        float(getattr(point_a, "y", 0.0)) - float(getattr(point_b, "y", 0.0)),
                    ))
                    if distance <= _point_match_threshold(point_a, point_b, base_radius):
                        candidate_pairs.append((distance, index_a, int(index_b)))
    else:
        for index_a, point_a in enumerate(points_a):
            for index_b, point_b in enumerate(points_b):
                distance = float(np.hypot(float(getattr(point_a, "x", 0.0)) - float(getattr(point_b, "x", 0.0)), float(getattr(point_a, "y", 0.0)) - float(getattr(point_b, "y", 0.0))))
                if distance <= _point_match_threshold(point_a, point_b, base_radius):
                    candidate_pairs.append((distance, index_a, index_b))
    candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    matched_distances: list[float] = []
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    for distance, index_a, index_b in candidate_pairs:
        if index_a in matched_a or index_b in matched_b:
            continue
        matched_a.add(index_a)
        matched_b.add(index_b)
        matched_distances.append(float(distance))
    return matched_distances, matched_a, matched_b


def _point_distance_scores(points_a: tuple[object, ...], points_b: tuple[object, ...], frame_shape: tuple[int, int]) -> tuple[float, float]:
    coords_a = _point_coordinates(points_a)
    coords_b = _point_coordinates(points_b)
    if coords_a.shape[0] == 0 and coords_b.shape[0] == 0:
        return 0.0, 0.0
    diagonal = float(max(EPS, math.hypot(float(frame_shape[0]), float(frame_shape[1]))))
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return diagonal, diagonal
    nearest_a, nearest_b = _nearest_distances_between_coordinate_sets(coords_a, coords_b)
    chamfer = float((nearest_a.mean(dtype=np.float64) + nearest_b.mean(dtype=np.float64)) / 2.0)
    hausdorff = float(max(float(nearest_a.max(initial=0.0)), float(nearest_b.max(initial=0.0))))
    return chamfer, hausdorff


def _point_count_error(predicted_count: int, target_count: int) -> float:
    return float(_clip01(abs(int(predicted_count) - int(target_count)) / max(1.0, float(target_count))))


def _point_count_agreement(first_count: int, second_count: int) -> float:
    return float(1.0 - _clip01(abs(int(first_count) - int(second_count)) / max(1.0, float(max(first_count, second_count, 1)))))


def _mask_count_agreement(first_count: int, second_count: int) -> float:
    return float(1.0 - _clip01(abs(int(first_count) - int(second_count)) / max(1.0, float(max(first_count, second_count, 1)))))


def _symmetric_binary_cross_entropy(first_prob: np.ndarray, second_prob: np.ndarray) -> float:
    """Compute symmetric BCE between two probability maps."""

    first = np.clip(np.asarray(first_prob, dtype=np.float64), EPS, 1.0 - EPS)
    second = np.clip(np.asarray(second_prob, dtype=np.float64), EPS, 1.0 - EPS)
    forward = -(first * np.log(second) + (1.0 - first) * np.log(1.0 - second))
    backward = -(second * np.log(first) + (1.0 - second) * np.log(1.0 - first))
    return float(np.mean((forward + backward) * 0.5, dtype=np.float64)) if forward.size else 0.0


def _polygon_bce_score(bce_value: float) -> float:
    capped = min(float(bce_value) / max(EPS, float(BCE_SCORE_CAP)), 1.0)
    return float(max(0.0, 100.0 * (1.0 - capped)))


def _aggregate_inter_model_polygon_scores(pairwise_rows: tuple[dict[str, object], ...]) -> dict[str, float]:
    """Aggregate pairwise polygon comparison metrics into one frame-level score bundle."""

    if not pairwise_rows:
        return {}
    iou = float(np.mean(np.asarray([float(row.get("iou", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    dice = float(np.mean(np.asarray([float(row.get("dice", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    bce = float(np.mean(np.asarray([float(row.get("bce", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    iou_score = float(max(0.0, min(iou, 1.0)) * 100.0)
    dice_score = float(max(0.0, min(dice, 1.0)) * 100.0)
    bce_score = _polygon_bce_score(bce)
    weight_sum = float(sum(INTER_MODEL_POLYGON_SCORE_WEIGHTS.values()))
    overall = (
        iou_score * float(INTER_MODEL_POLYGON_SCORE_WEIGHTS["iou"]) +
        dice_score * float(INTER_MODEL_POLYGON_SCORE_WEIGHTS["dice"]) +
        bce_score * float(INTER_MODEL_POLYGON_SCORE_WEIGHTS["bce"])
    ) / max(EPS, weight_sum)
    return {
        "iou": iou,
        "dice": dice,
        "bce": bce,
        "iou_score": iou_score,
        "dice_score": dice_score,
        "polygon_bce_score": bce_score,
        "overall_polygon_score": float(overall),
    }


def _aggregate_inter_model_point_scores(pairwise_rows: tuple[dict[str, object], ...], point_match_radius: float) -> dict[str, float]:
    """Aggregate pairwise point comparison metrics into one frame-level score bundle."""

    if not pairwise_rows:
        return {}
    precision = float(np.mean(np.asarray([float(row.get("precision", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    recall = float(np.mean(np.asarray([float(row.get("recall", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    f1 = float(np.mean(np.asarray([float(row.get("f1", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    mean_localization_distance = float(np.mean(np.asarray([float(row.get("mean_localization_error", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    tp = float(np.mean(np.asarray([float(row.get("matched_count", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    fp = float(np.mean(np.asarray([max(0.0, float(row.get("point_count_a", 0.0)) - float(row.get("matched_count", 0.0))) for row in pairwise_rows], dtype=np.float64)))
    fn = float(np.mean(np.asarray([max(0.0, float(row.get("point_count_b", 0.0)) - float(row.get("matched_count", 0.0))) for row in pairwise_rows], dtype=np.float64)))
    precision_score = float(max(0.0, min(precision, 1.0)) * 100.0)
    recall_score = float(max(0.0, min(recall, 1.0)) * 100.0)
    f1_score = float(max(0.0, min(f1, 1.0)) * 100.0)
    localization_score = float(max(0.0, 100.0 * (1.0 - min(mean_localization_distance / max(EPS, float(point_match_radius)), 1.0))))
    weight_sum = float(sum(INTER_MODEL_POINT_SCORE_WEIGHTS.values()))
    overall = (
        precision_score * float(INTER_MODEL_POINT_SCORE_WEIGHTS["precision"]) +
        recall_score * float(INTER_MODEL_POINT_SCORE_WEIGHTS["recall"]) +
        f1_score * float(INTER_MODEL_POINT_SCORE_WEIGHTS["f1"]) +
        localization_score * float(INTER_MODEL_POINT_SCORE_WEIGHTS["localization"])
    ) / max(EPS, weight_sum)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_localization_distance": mean_localization_distance,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
        "localization_score": localization_score,
        "overall_point_score": float(overall),
    }


def _paint_disk(target: np.ndarray, center_x: float, center_y: float, radius: float, value: float) -> None:
    if target.ndim != 2 or target.size == 0:
        return
    effective_radius = max(1.0, float(radius))
    x = int(round(float(center_x)))
    y = int(round(float(center_y)))
    y0 = max(0, int(math.floor(y - effective_radius)))
    y1 = min(target.shape[0], int(math.ceil(y + effective_radius + 1.0)))
    x0 = max(0, int(math.floor(x - effective_radius)))
    x1 = min(target.shape[1], int(math.ceil(x + effective_radius + 1.0)))
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = np.sqrt((yy - float(center_y)) ** 2 + (xx - float(center_x)) ** 2, dtype=np.float64)
    disk = distance <= effective_radius
    target[y0:y1, x0:x1] = np.maximum(target[y0:y1, x0:x1], np.asarray(disk, dtype=np.float32) * float(value))


def _point_map_from_view(prediction_view: object, shape: tuple[int, int] | None = None, *, scale: float = 1.0) -> np.ndarray:
    base_shape = tuple(int(v) for v in (shape or getattr(prediction_view, 'pred_gray').shape))
    result = np.zeros(base_shape, dtype=np.float32)
    for point in tuple(getattr(prediction_view, 'points', ())):
        radius = max(1.0, float(getattr(point, 'radius', 0.0)))
        score = float(getattr(point, 'score', 1.0) or 1.0)
        _paint_disk(result, float(getattr(point, 'x', 0.0)), float(getattr(point, 'y', 0.0)), radius + 1.0, float(scale) * float(max(0.25, min(1.0, score))))
    return np.clip(result, 0.0, 1.0)


def _point_labeled_error_map(prediction_view: object, gt_view: object, point_match_radius: float) -> np.ndarray:
    shape = tuple(int(v) for v in getattr(prediction_view, 'pred_gray').shape)
    result = np.zeros(shape, dtype=np.float32)
    pred_points = tuple(getattr(prediction_view, 'points', ()))
    gt_points = tuple(getattr(gt_view, 'points', ()))
    matched_distances, matched_pred, matched_gt = _match_point_sets(pred_points, gt_points, point_match_radius)
    diagonal = float(max(EPS, math.hypot(float(shape[0]), float(shape[1]))))
    matched_pairs: list[tuple[int, int, float]] = []
    candidate_pairs: list[tuple[float, int, int]] = []
    for index_a, point_a in enumerate(pred_points):
        for index_b, point_b in enumerate(gt_points):
            distance = float(np.hypot(float(getattr(point_a, 'x', 0.0)) - float(getattr(point_b, 'x', 0.0)), float(getattr(point_a, 'y', 0.0)) - float(getattr(point_b, 'y', 0.0))))
            if distance <= _point_match_threshold(point_a, point_b, point_match_radius):
                candidate_pairs.append((distance, index_a, index_b))
    candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    for distance, index_a, index_b in candidate_pairs:
        if index_a in used_a or index_b in used_b:
            continue
        used_a.add(index_a)
        used_b.add(index_b)
        matched_pairs.append((index_a, index_b, distance))
    for index_a, index_b, distance in matched_pairs:
        severity = float(min(1.0, distance / max(EPS, point_match_radius, diagonal * 0.05)))
        point_a = pred_points[index_a]
        point_b = gt_points[index_b]
        _paint_disk(result, float(getattr(point_a, 'x', 0.0)), float(getattr(point_a, 'y', 0.0)), max(1.0, float(getattr(point_a, 'radius', 0.0))) + 1.0, severity)
        _paint_disk(result, float(getattr(point_b, 'x', 0.0)), float(getattr(point_b, 'y', 0.0)), max(1.0, float(getattr(point_b, 'radius', 0.0))) + 1.0, severity)
    for index, point in enumerate(pred_points):
        if index not in matched_pred:
            _paint_disk(result, float(getattr(point, 'x', 0.0)), float(getattr(point, 'y', 0.0)), max(1.0, float(getattr(point, 'radius', 0.0))) + 1.0, 1.0)
    for index, point in enumerate(gt_points):
        if index not in matched_gt:
            _paint_disk(result, float(getattr(point, 'x', 0.0)), float(getattr(point, 'y', 0.0)), max(1.0, float(getattr(point, 'radius', 0.0))) + 1.0, 0.85)
    return np.clip(result, 0.0, 1.0)



def _point_labeled_quality(prediction_view: object, gt_view: object, point_match_radius: float) -> PointLabeledMetrics:
    points_pred = tuple(getattr(prediction_view, "points", ()))
    points_gt = tuple(getattr(gt_view, "points", ()))
    matched_distances, matched_pred, matched_gt = _match_point_sets(points_pred, points_gt, point_match_radius)
    predicted_count = len(points_pred)
    target_count = len(points_gt)
    matched_count = len(matched_distances)
    if predicted_count == 0 and target_count == 0:
        precision = 1.0
        recall = 1.0
    else:
        precision = float(matched_count / max(1, predicted_count))
        recall = float(matched_count / max(1, target_count))
    f1 = 0.0 if precision + recall <= EPS else float((2.0 * precision * recall) / (precision + recall))
    mean_localization_error = float(np.mean(np.asarray(matched_distances, dtype=np.float64))) if matched_count > 0 else 0.0
    if matched_count <= 0:
        localization_score = 1.0 if predicted_count == 0 and target_count == 0 else 0.0
    else:
        localization_score = float(1.0 - min(1.0, mean_localization_error / max(EPS, float(point_match_radius))))
    chamfer_distance, hausdorff_distance = _point_distance_scores(points_pred, points_gt, tuple(int(v) for v in prediction_view.pred_gray.shape))
    diagonal = float(max(EPS, math.hypot(float(prediction_view.pred_gray.shape[0]), float(prediction_view.pred_gray.shape[1]))))
    chamfer_score = float(1.0 - min(1.0, chamfer_distance / diagonal))
    hausdorff_score = float(1.0 - min(1.0, hausdorff_distance / diagonal))
    count_error = _point_count_error(predicted_count, target_count)
    quality = _weighted_mean([
        (f1, POINT_SUPERVISED_SCORE_WEIGHTS["f1"]),
        (localization_score, POINT_SUPERVISED_SCORE_WEIGHTS["localization"]),
        (1.0 - count_error, POINT_SUPERVISED_SCORE_WEIGHTS["count_term"]),
    ])
    return PointLabeledMetrics(
        precision_at_radius=float(precision),
        recall_at_radius=float(recall),
        f1_at_radius=float(f1),
        mean_localization_error=float(mean_localization_error),
        localization_score=float(localization_score),
        chamfer_score=float(chamfer_score),
        hausdorff_score=float(hausdorff_score),
        count_error=float(count_error),
        matched_count=int(matched_count),
        predicted_count=int(predicted_count),
        target_count=int(target_count),
        quality_score=float(quality),
        error_score=float(1.0 - quality),
    )


def _point_model_agreement(first_view: object, second_view: object, point_match_radius: float) -> PointAgreementMetrics:
    points_a = tuple(getattr(first_view, "points", ()))
    points_b = tuple(getattr(second_view, "points", ()))
    matched_distances, _matched_a, _matched_b = _match_point_sets(points_a, points_b, point_match_radius)
    count_a = len(points_a)
    count_b = len(points_b)
    matched_count = len(matched_distances)
    false_positive_count = max(0, int(count_a - matched_count))
    false_negative_count = max(0, int(count_b - matched_count))
    if count_a == 0 and count_b == 0:
        precision = 1.0
        recall = 1.0
    else:
        precision = float(matched_count / max(1, count_a))
        recall = float(matched_count / max(1, count_b))
    f1 = 0.0 if precision + recall <= EPS else float((2.0 * precision * recall) / (precision + recall))
    mean_localization_error = float(np.mean(np.asarray(matched_distances, dtype=np.float64))) if matched_count > 0 else 0.0
    if matched_count <= 0:
        localization_agreement = 1.0 if count_a == 0 and count_b == 0 else 0.0
    else:
        localization_agreement = float(1.0 - min(1.0, mean_localization_error / max(EPS, float(point_match_radius))))
    count_agreement = _point_count_agreement(count_a, count_b)
    agreement_score = _weighted_mean([
        (f1, POINT_AGREEMENT_SCORE_WEIGHTS["f1"]),
        (localization_agreement, POINT_AGREEMENT_SCORE_WEIGHTS["localization"]),
        (count_agreement, POINT_AGREEMENT_SCORE_WEIGHTS["count_agreement"]),
    ])
    return PointAgreementMetrics(
        precision_at_radius=float(precision),
        recall_at_radius=float(recall),
        f1_at_radius=float(f1),
        mean_localization_error=float(mean_localization_error),
        localization_agreement=float(localization_agreement),
        count_agreement=float(count_agreement),
        matched_count=int(matched_count),
        true_positive_count=int(matched_count),
        false_positive_count=int(false_positive_count),
        false_negative_count=int(false_negative_count),
        point_count_a=int(count_a),
        point_count_b=int(count_b),
        agreement_score=float(agreement_score),
    )

def _prepare_mask_pairwise_descriptors(
    probabilities_by_model: dict[str, np.ndarray],
    masks_by_model: dict[str, np.ndarray],
    model_structures: dict[str, dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    """Precompute reusable per-model structures for symmetric mask agreement metrics."""

    descriptors: dict[str, dict[str, object]] = {}
    for model_id, probability in probabilities_by_model.items():
        prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
        mask = np.asarray(masks_by_model.get(model_id), dtype=bool)
        boundary = _boundary_mask(mask)
        dist_to_boundary = _distance_transform(~boundary) if np.any(boundary) and (ndi is not None or cv2 is not None) else None
        current_structure = (model_structures or {}).get(str(model_id)) or _mask_structure(mask)
        descriptors[str(model_id)] = {
            'prob': prob,
            'prob_sum': float(np.sum(prob, dtype=np.float64)),
            'prob_sq_sum': float(np.sum(np.square(prob, dtype=np.float32), dtype=np.float64)),
            'mask': mask,
            'mask_area': int(np.count_nonzero(mask)),
            'boundary': boundary,
            'boundary_dist': dist_to_boundary,
            'centroid': _mask_centroid(mask),
            'structure': current_structure,
            'shape': tuple(int(v) for v in mask.shape),
        }
    return descriptors


def _pairwise_mask_metrics(
    first: dict[str, object],
    second: dict[str, object],
) -> dict[str, float]:
    """Compute all symmetric mask agreement metrics from precomputed per-model descriptors."""

    first_prob = np.asarray(first['prob'], dtype=np.float32)
    second_prob = np.asarray(second['prob'], dtype=np.float32)
    first_mask = np.asarray(first['mask'], dtype=bool)
    second_mask = np.asarray(second['mask'], dtype=bool)
    shape = tuple(int(v) for v in first['shape'])

    intersection_mask = int(np.count_nonzero(first_mask & second_mask))
    area_first = int(first['mask_area'])
    area_second = int(second['mask_area'])
    union_mask = int(area_first + area_second - intersection_mask)

    prob_intersection = float(np.sum(first_prob * second_prob, dtype=np.float64))
    prob_sum_first = float(first['prob_sum'])
    prob_sum_second = float(second['prob_sum'])
    prob_union = float(prob_sum_first + prob_sum_second - prob_intersection)

    soft_dice = 1.0 if (prob_sum_first + prob_sum_second) <= EPS else float((2.0 * prob_intersection + EPS) / (prob_sum_first + prob_sum_second + EPS))
    soft_iou = 1.0 if prob_union <= EPS else float((prob_intersection + EPS) / (prob_union + EPS))

    first_prob64 = np.asarray(first_prob, dtype=np.float64)
    second_prob64 = np.asarray(second_prob, dtype=np.float64)
    mu_first = float(first_prob64.mean())
    mu_second = float(second_prob64.mean())
    sigma_first = float(np.mean((first_prob64 - mu_first) ** 2, dtype=np.float64))
    sigma_second = float(np.mean((second_prob64 - mu_second) ** 2, dtype=np.float64))
    sigma_cross = float(np.mean((first_prob64 - mu_first) * (second_prob64 - mu_second), dtype=np.float64))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    denominator = (mu_first ** 2 + mu_second ** 2 + c1) * (sigma_first + sigma_second + c2)
    ssim = 1.0 if denominator <= EPS and np.allclose(first_prob64, second_prob64) else (0.0 if denominator <= EPS else float(_clip01(((2.0 * mu_first * mu_second + c1) * (2.0 * sigma_cross + c2)) / denominator)))

    dice = 1.0 if (area_first + area_second) == 0 else float((2.0 * intersection_mask + EPS) / (area_first + area_second + EPS))
    iou = 1.0 if union_mask == 0 else float((intersection_mask + EPS) / (union_mask + EPS))

    diagonal = _frame_diagonal(shape)
    first_center = first['centroid']
    second_center = second['centroid']
    if first_center is None and second_center is None:
        centroid_distance = 0.0
    elif first_center is None or second_center is None:
        centroid_distance = diagonal
    else:
        centroid_distance = float(math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1]))
    centroid_similarity = _distance_similarity(centroid_distance, shape)

    first_boundary = np.asarray(first['boundary'], dtype=bool)
    second_boundary = np.asarray(second['boundary'], dtype=bool)
    if not np.any(first_boundary) and not np.any(second_boundary):
        hausdorff_distance = 0.0
    elif not np.any(first_boundary) or not np.any(second_boundary):
        hausdorff_distance = diagonal
    elif first.get('boundary_dist') is not None and second.get('boundary_dist') is not None:
        directed_first = float(np.max(np.asarray(second['boundary_dist'], dtype=np.float32)[first_boundary])) if np.any(first_boundary) else 0.0
        directed_second = float(np.max(np.asarray(first['boundary_dist'], dtype=np.float32)[second_boundary])) if np.any(second_boundary) else 0.0
        hausdorff_distance = float(max(directed_first, directed_second))
    else:
        hausdorff_distance = _hausdorff_distance(first_mask, second_mask)
    hausdorff_similarity = _distance_similarity(hausdorff_distance, shape)

    diff = first_prob - second_prob
    mae = float(np.mean(np.abs(diff), dtype=np.float64)) if diff.size else 0.0
    rmse = float(np.sqrt(np.mean(np.square(diff, dtype=np.float32), dtype=np.float64))) if diff.size else 0.0
    bce = _symmetric_binary_cross_entropy(first_prob, second_prob)

    count_agreement = _mask_count_agreement(
        int(np.asarray(first['structure']['component_count']).item() if hasattr(first['structure']['component_count'], 'item') else first['structure']['component_count']),
        int(np.asarray(second['structure']['component_count']).item() if hasattr(second['structure']['component_count'], 'item') else second['structure']['component_count']),
    )
    agreement_score = _weighted_mean([
        (float(soft_dice), MASK_AGREEMENT_SCORE_WEIGHTS['soft_dice']),
        (float(soft_iou), MASK_AGREEMENT_SCORE_WEIGHTS['soft_iou']),
        (float(ssim), MASK_AGREEMENT_SCORE_WEIGHTS['ssim']),
        (float(dice), MASK_AGREEMENT_SCORE_WEIGHTS['dice']),
        (float(iou), MASK_AGREEMENT_SCORE_WEIGHTS['iou']),
        (float(hausdorff_similarity), MASK_AGREEMENT_SCORE_WEIGHTS['hausdorff_term']),
        (float(centroid_similarity), MASK_AGREEMENT_SCORE_WEIGHTS['centroid_term']),
    ])

    return {
        'soft_dice': float(soft_dice),
        'soft_iou': float(soft_iou),
        'ssim': float(ssim),
        'dice': float(dice),
        'iou': float(iou),
        'hausdorff_distance': float(hausdorff_distance),
        'hausdorff_similarity': float(hausdorff_similarity),
        'centroid_distance': float(centroid_distance),
        'centroid_similarity': float(centroid_similarity),
        'mae': float(mae),
        'rmse': float(rmse),
        'bce': float(bce),
        'count_agreement': float(count_agreement),
        'agreement_score': float(agreement_score),
    }


def _pairwise_model_comparisons(
    probabilities_by_model: dict[str, np.ndarray],
    masks_by_model: dict[str, np.ndarray],
    *,
    geometry_mode: GeometryMode = GeometryMode.MASK,
    model_views: dict[str, object] | None = None,
    model_structures: dict[str, dict[str, object]] | None = None,
    point_match_radius: float = 3.0,
) -> tuple[dict[str, object], ...]:
    model_ids = list(probabilities_by_model.keys())
    rows: list[dict[str, object]] = []
    current_views = model_views or {}
    mask_descriptors = _prepare_mask_pairwise_descriptors(probabilities_by_model, masks_by_model, model_structures=model_structures) if geometry_mode != GeometryMode.POINT else {}
    for index_a, model_a in enumerate(model_ids):
        for model_b in model_ids[index_a + 1:]:
            if geometry_mode == GeometryMode.POINT and model_a in current_views and model_b in current_views:
                metrics = _point_model_agreement(current_views[model_a], current_views[model_b], point_match_radius)
                rows.append({
                    'model_a': model_a,
                    'model_b': model_b,
                    'precision': float(metrics.precision_at_radius),
                    'recall': float(metrics.recall_at_radius),
                    'f1': float(metrics.f1_at_radius),
                    'mean_localization_error': float(metrics.mean_localization_error),
                    'localization_agreement': float(metrics.localization_agreement),
                    'count_agreement': float(metrics.count_agreement),
                    'matched_count': int(metrics.matched_count),
                    'tp': int(metrics.true_positive_count),
                    'fp': int(metrics.false_positive_count),
                    'fn': int(metrics.false_negative_count),
                    'point_count_a': int(metrics.point_count_a),
                    'point_count_b': int(metrics.point_count_b),
                    'agreement_score': float(metrics.agreement_score),
                })
                continue

            metrics_row = _pairwise_mask_metrics(mask_descriptors[str(model_a)], mask_descriptors[str(model_b)])
            rows.append({
                'model_a': model_a,
                'model_b': model_b,
                **metrics_row,
            })
    return tuple(rows)


def _point_feature_vector(prediction_view: object) -> dict[str, float]:
    points = tuple(getattr(prediction_view, "points", ()))
    point_count = len(points)
    image_shape = tuple(int(v) for v in getattr(prediction_view, "pred_gray").shape)
    image_area = max(1.0, float(image_shape[0] * image_shape[1]))
    mean_radius = float(np.mean([float(getattr(point, "radius", 0.0)) for point in points], dtype=np.float64)) if points else 0.0
    mean_peak_intensity = float(np.mean([float(getattr(point, "peak_intensity", 0.0)) for point in points], dtype=np.float64) / 255.0) if points else 0.0
    mean_local_snr = float(np.mean([float(getattr(point, "local_snr", 0.0)) for point in points], dtype=np.float64)) if points else 0.0
    mean_blob_score = float(np.mean([float(getattr(point, "blob_score", 0.0)) for point in points], dtype=np.float64)) if points else 0.0
    return {
        "area_fraction": float(point_count / image_area),
        "component_count": float(point_count),
        "mean_component_area": float(mean_peak_intensity),
        "skeleton_length": float(mean_radius),
        "endpoint_count": float(_normalize_ratio(mean_local_snr)),
        "branchpoint_count": float(_normalize_ratio(mean_blob_score)),
    }


def _point_diagnostic_metrics(prediction_view: object) -> PointDiagnosticMetrics:
    points = tuple(getattr(prediction_view, "points", ()))
    mean_radius = float(np.mean([float(getattr(point, "radius", 0.0)) for point in points], dtype=np.float64)) if points else 0.0
    mean_peak_intensity = float(np.mean([float(getattr(point, "peak_intensity", 0.0)) for point in points], dtype=np.float64) / 255.0) if points else 0.0
    mean_local_snr = float(np.mean([float(getattr(point, "local_snr", 0.0)) for point in points], dtype=np.float64)) if points else 0.0
    false_spot_ratio = float(_clip01(np.mean([1.0 if float(getattr(point, "local_snr", 0.0)) < 0.5 else 0.0 for point in points], dtype=np.float64))) if points else 0.0
    proxy_score = _weighted_mean([
        (_clip01(mean_peak_intensity), 0.45),
        (_clip01(_normalize_ratio(mean_local_snr)), 0.35),
        (1.0 - false_spot_ratio, 0.20),
    ])
    return PointDiagnosticMetrics(
        point_count=int(len(points)),
        mean_radius=float(mean_radius),
        mean_peak_intensity=float(mean_peak_intensity),
        false_spot_ratio=float(false_spot_ratio),
        proxy_score=float(proxy_score),
    )


def _infer_geometry_mode(prediction_view: object) -> GeometryMode:
    points = tuple(getattr(prediction_view, "points", ()))
    region_summary = getattr(prediction_view, "region_summary", None)
    area_fraction = float(getattr(region_summary, "area_fraction", 0.0)) if region_summary is not None else 0.0
    mean_area = float(getattr(region_summary, "mean_area", 0.0)) if region_summary is not None else 0.0
    if len(points) > 0 and area_fraction <= 0.08 and mean_area <= 64.0:
        return GeometryMode.POINT
    return GeometryMode.MASK


def _consensus_probability(probabilities: list[np.ndarray]) -> np.ndarray:
    if not probabilities:
        return np.zeros((1, 1), dtype=np.float32)
    stacked = np.stack(probabilities, axis=0).astype(np.float32)
    return np.mean(stacked, axis=0, dtype=np.float32)


def _entropy_map(probability: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    entropy = -(prob * np.log(prob) + (1.0 - prob) * np.log(1.0 - prob)) / np.log(2.0)
    return np.asarray(entropy, dtype=np.float32)


def _disagreement_score(probabilities: list[np.ndarray], masks: list[np.ndarray]) -> tuple[float, np.ndarray]:
    if not probabilities:
        return 0.0, np.zeros((1, 1), dtype=np.float32)
    if len(probabilities) == 1:
        variance_map = np.zeros_like(probabilities[0], dtype=np.float32)
        return 0.0, variance_map
    pairwise: list[float] = []
    for index_a in range(len(masks)):
        for index_b in range(index_a + 1, len(masks)):
            pairwise.append(1.0 - _dice(masks[index_a], masks[index_b]))
    stacked = np.stack(probabilities, axis=0).astype(np.float32)
    variance_map = np.clip(4.0 * np.var(stacked, axis=0, dtype=np.float32), 0.0, 1.0)
    score = 0.6 * float(np.mean(pairwise, dtype=np.float64)) + 0.4 * float(np.mean(variance_map, dtype=np.float64))
    return float(_clip01(score)), variance_map.astype(np.float32)


def _structural_feature_vector(consensus_mask: np.ndarray, *, structure: dict[str, object] | None = None) -> dict[str, float]:
    current_structure = structure or _mask_structure(consensus_mask)
    return {
        "area_fraction": float(current_structure["area_fraction"]),
        "component_count": float(current_structure["component_count"]),
        "mean_component_area": float(current_structure["mean_component_area"]),
        "skeleton_length": float(current_structure["skeleton_length"]),
        "endpoint_count": float(current_structure["endpoint_count"]),
        "branchpoint_count": float(current_structure["branchpoint_count"]),
    }


def _robust_feature_bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return median, mad if mad > EPS else float(np.std(array) + EPS)


def _structural_anomaly_score(vector: dict[str, float], bounds: dict[str, tuple[float, float]]) -> float:
    terms: list[float] = []
    for key, value in vector.items():
        median, scale = bounds.get(key, (0.0, 1.0))
        z_score = abs(float(value) - median) / max(EPS, scale)
        terms.append(_clip01(z_score / 3.0))
    return float(np.mean(terms, dtype=np.float64)) if terms else 0.0



def _build_model_payloads(
    record: FrameRecord,
    model_specs: tuple[ModelSpec, ...],
    *,
    analysis_max_side: int | None = None,
    geometry_mode: GeometryMode = GeometryMode.MASK,
    point_match_radius: float = 3.0,
    boundary_radius: int = 1,
    confidence_uncertainty_delta: float = MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    point_confidence_radius: int = POINT_CONFIDENCE_NEIGHBOR_RADIUS,
    polygon_confidence_summary: str = POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
    include_confidence_objects: bool = True,
    include_original_gray: bool = True,
    include_model_confidence: bool = True,
    include_structure_details: bool = True,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    np.ndarray | None,
    np.ndarray | None,
    dict[str, object] | None,
    GeometryMode,
    dict[str, object],
    object | None,
]:
    original_gray = _load_optional_gray(record.original_path, max_side=analysis_max_side) if include_original_gray else None
    gt_gray = _load_optional_gray(record.gt_path, max_side=analysis_max_side)
    gt_mask = _mask_from_gray(gt_gray) if gt_gray is not None else None
    gt_structure = _mask_structure(gt_mask, include_skeleton=include_structure_details) if gt_mask is not None else None

    probabilities: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    model_structures: dict[str, dict[str, object]] = {}
    model_diagnostics: dict[str, object] = {}
    model_metrics: dict[str, object] = {}
    model_confidence: dict[str, object] = {}
    model_views: dict[str, object] = {}

    target_shape = tuple(int(v) for v in gt_gray.shape) if gt_gray is not None else None
    if target_shape is None and original_gray is not None:
        target_shape = tuple(int(v) for v in original_gray.shape)

    loaded_rows: list[tuple[ModelSpec, np.ndarray, np.ndarray]] = []
    first_point_candidate = None
    for spec in model_specs:
        mask_gray = _load_optional_gray(record.model_mask_paths.get(spec.model_id), target_shape=target_shape, max_side=analysis_max_side)
        if mask_gray is None:
            continue
        if target_shape is None:
            target_shape = tuple(int(v) for v in mask_gray.shape)
        if original_gray is not None and tuple(int(v) for v in original_gray.shape) != target_shape:
            original_gray = _load_optional_gray(record.original_path, target_shape=target_shape, max_side=analysis_max_side)
        if gt_gray is not None and tuple(int(v) for v in gt_gray.shape) != target_shape:
            gt_gray = _load_optional_gray(record.gt_path, target_shape=target_shape, max_side=analysis_max_side)
            gt_mask = _mask_from_gray(gt_gray) if gt_gray is not None else None
            gt_structure = _mask_structure(gt_mask, include_skeleton=include_structure_details) if gt_mask is not None else None
        prob_gray = _load_optional_gray(record.model_prob_paths.get(spec.model_id), target_shape=target_shape, max_side=analysis_max_side)
        if prob_gray is None:
            prob_gray = mask_gray
        loaded_rows.append((spec, mask_gray, prob_gray))
        if geometry_mode == GeometryMode.AUTO and first_point_candidate is None:
            first_point_candidate = build_prediction_view_from_gray(spec.display_name, np.asarray(prob_gray, dtype=np.uint8), threshold=int(round(float(spec.threshold) * 255.0)))

    resolved_geometry_mode = geometry_mode
    if resolved_geometry_mode == GeometryMode.AUTO:
        resolved_geometry_mode = _infer_geometry_mode(first_point_candidate) if first_point_candidate is not None else GeometryMode.MASK

    gt_point_view = None
    if resolved_geometry_mode == GeometryMode.POINT and gt_gray is not None:
        gt_point_view = build_prediction_view_from_gray('ground_truth', np.asarray(gt_gray, dtype=np.uint8), threshold=128)

    for spec, mask_gray, prob_gray in loaded_rows:
        prob_map = _prob_from_gray(prob_gray)
        mask = _mask_from_gray(mask_gray, threshold=spec.threshold)
        probabilities[spec.model_id] = prob_map.astype(np.float32)
        masks[spec.model_id] = mask.astype(bool)

        if resolved_geometry_mode == GeometryMode.POINT:
            prediction_view = build_prediction_view_from_gray(spec.display_name, np.asarray(prob_gray, dtype=np.uint8), threshold=int(round(float(spec.threshold) * 255.0)))
            model_views[spec.model_id] = prediction_view
            diagnostics = _point_diagnostic_metrics(prediction_view)
            model_diagnostics[spec.model_id] = diagnostics
            if include_model_confidence:
                model_confidence[spec.model_id] = _point_internal_confidence(
                    prediction_view,
                    neighborhood_radius=int(point_confidence_radius),
                    include_objects=include_confidence_objects,
                )
            if gt_point_view is not None:
                model_metrics[spec.model_id] = _point_labeled_quality(prediction_view, gt_point_view, point_match_radius)
            model_structures[spec.model_id] = {
                'component_count': int(diagnostics.point_count),
                'area_fraction': float(max(0.0, min(1.0, diagnostics.point_count / max(1.0, float(prediction_view.pred_gray.size))))),
                'skeleton_length': float(diagnostics.mean_radius),
            }
            continue

        mask_structure = _mask_structure(mask, include_skeleton=include_structure_details)
        model_structures[spec.model_id] = mask_structure
        area_fraction = float(mask_structure['area_fraction'])
        proxy_score = float(_clip01(1.0 - area_fraction))
        model_diagnostics[spec.model_id] = ModelDiagnosticMetrics(
            area_fraction=area_fraction,
            component_count=int(mask_structure['component_count']),
            skeleton_length=float(mask_structure['skeleton_length']),
            proxy_score=proxy_score,
        )
        if include_model_confidence:
            model_confidence[spec.model_id] = _polygon_frame_confidence(
                prob_map,
                mask,
                uncertainty_delta=float(confidence_uncertainty_delta),
                summary_metric=str(polygon_confidence_summary),
            )
        if gt_mask is not None and gt_structure is not None:
            model_metrics[spec.model_id] = _labeled_quality(
                prob_map,
                mask,
                gt_mask,
                pred_structure=mask_structure,
                gt_structure=gt_structure,
                boundary_radius=boundary_radius,
            )

    return probabilities, masks, model_structures, model_diagnostics, model_metrics, model_confidence, original_gray, gt_mask, gt_structure, resolved_geometry_mode, model_views, gt_point_view


def _metric_requires_model_confidence(metric_key: str | None) -> bool:
    parsed = _parse_model_metric_key(str(metric_key or ''))
    if parsed is None:
        return False
    family, _model_id = parsed
    return family in {'model_confidence', 'model_uncertain_fraction', 'model_point_contrast'}


def _analyze_record_payload(
    record: FrameRecord,
    model_specs: tuple[ModelSpec, ...],
    analysis_max_side: int | None,
    geometry_mode: GeometryMode,
    point_match_radius: float,
    boundary_radius: int,
    confidence_uncertainty_delta: float,
    point_confidence_radius: int,
    polygon_confidence_summary: str,
    cache_enabled: bool,
    include_model_confidence: bool = True,
) -> dict[str, object] | None:
    timings_ms: dict[str, float] = {}
    cache_key = None
    if cache_enabled:
        cache_key = _record_payload_cache_key(
            record,
            model_specs,
            analysis_max_side,
            geometry_mode,
            point_match_radius,
            boundary_radius,
            confidence_uncertainty_delta,
            point_confidence_radius,
            polygon_confidence_summary,
        )
        cached = _load_cached_record_payload(cache_key)
        if cached is not None:
            return cached

    load_started = perf_counter()
    probabilities_by_model, masks_by_model, model_structures, model_diagnostics, model_metrics, model_confidence, _original_gray, gt_mask, _gt_structure, resolved_geometry_mode, model_views, gt_point_view = _build_model_payloads(
        record,
        model_specs,
        analysis_max_side=analysis_max_side,
        geometry_mode=geometry_mode,
        point_match_radius=point_match_radius,
        boundary_radius=boundary_radius,
        confidence_uncertainty_delta=confidence_uncertainty_delta,
        point_confidence_radius=point_confidence_radius,
        polygon_confidence_summary=polygon_confidence_summary,
        include_confidence_objects=False,
        include_original_gray=False,
        include_model_confidence=include_model_confidence,
        include_structure_details=False,
    )
    timings_ms['loading_preprocess'] = 1000.0 * (perf_counter() - load_started)
    probabilities = list(probabilities_by_model.values())
    if not probabilities:
        return None

    metrics_started = perf_counter()
    consensus_prob = _consensus_probability(probabilities)
    consensus_mask = np.asarray(consensus_prob >= 0.5, dtype=bool)
    consensus_structure = _mask_structure(consensus_mask, include_skeleton=False)

    pairwise_rows = _pairwise_model_comparisons(
        probabilities_by_model,
        masks_by_model,
        geometry_mode=resolved_geometry_mode,
        model_views=model_views,
        model_structures=model_structures,
        point_match_radius=point_match_radius,
    )
    agreement_scores = np.asarray([float(row.get('agreement_score', 0.0)) for row in pairwise_rows], dtype=np.float64)
    disagreement = float(np.mean(1.0 - agreement_scores, dtype=np.float64)) if agreement_scores.size else 0.0
    model_model_score = float(np.mean(agreement_scores, dtype=np.float64)) if agreement_scores.size else 1.0

    if resolved_geometry_mode == GeometryMode.POINT:
        reference_view = None
        if model_diagnostics:
            best_model_id = max(model_diagnostics.items(), key=lambda item: float(getattr(item[1], 'proxy_score', 0.0)))[0]
            reference_view = model_views.get(best_model_id)
        if reference_view is None and model_views:
            reference_view = next(iter(model_views.values()))
        vector = _point_feature_vector(reference_view) if reference_view is not None else {
            'area_fraction': 0.0,
            'component_count': 0.0,
            'mean_component_area': 0.0,
            'skeleton_length': 0.0,
            'endpoint_count': 0.0,
            'branchpoint_count': 0.0,
        }
    else:
        vector = _structural_feature_vector(consensus_mask, structure=consensus_structure)

    if model_metrics:
        qualities = np.asarray([float(getattr(metrics, 'quality_score', 0.0)) for metrics in model_metrics.values()], dtype=np.float64)
        labeled_best = float(np.max(qualities))
        labeled_mean = float(np.mean(qualities, dtype=np.float64))
        model_labeled_score = float(labeled_best)
    else:
        labeled_best = None
        labeled_mean = None
        model_labeled_score = None
    timings_ms['metrics'] = 1000.0 * (perf_counter() - metrics_started)

    assemble_started = perf_counter()
    payload = {
        'key': record.key,
        'geometry_mode': resolved_geometry_mode.value,
        'vector': vector,
        'disagreement': float(disagreement),
        'model_model_score': float(_clip01(model_model_score)),
        'labeled_best': labeled_best,
        'labeled_mean': labeled_mean,
        'model_labeled_score': None if model_labeled_score is None else float(_clip01(model_labeled_score)),
        'model_diagnostics': model_diagnostics,
        'model_metrics': model_metrics,
        'model_confidence': model_confidence,
        'pairwise_rows': pairwise_rows,
        'timings_ms': timings_ms,
    }
    timings_ms['payload_assembly'] = 1000.0 * (perf_counter() - assemble_started)
    if cache_enabled and cache_key is not None:
        _store_cached_record_payload(cache_key, payload)
    return payload


def _analyze_record_payload_for_executor(args: tuple[FrameRecord, tuple[ModelSpec, ...], int | None, GeometryMode, float, int, float, int, str, bool, bool]) -> tuple[str, dict[str, object] | None]:
    record, model_specs, analysis_max_side, geometry_mode, point_match_radius, boundary_radius, confidence_uncertainty_delta, point_confidence_radius, polygon_confidence_summary, cache_enabled, include_model_confidence = args
    return record.key, _analyze_record_payload(
        record,
        model_specs,
        analysis_max_side,
        geometry_mode,
        point_match_radius,
        boundary_radius,
        confidence_uncertainty_delta,
        point_confidence_radius,
        polygon_confidence_summary,
        cache_enabled,
        include_model_confidence,
    )


def _iter_record_payloads(
    records: list[FrameRecord],
    model_specs: tuple[ModelSpec, ...],
    analysis_max_side: int | None,
    max_workers: int,
    *,
    geometry_mode: GeometryMode,
    point_match_radius: float,
    boundary_radius: int,
    confidence_uncertainty_delta: float,
    point_confidence_radius: int,
    polygon_confidence_summary: str,
    cache_enabled: bool,
    include_model_confidence: bool = True,
    progress_callback=None,
    state_callback=None,
    cancel_check=None,
):
    """Yield per-record analytics payloads without retaining the whole dataset in RAM."""

    worker_count = max(1, int(max_workers))
    if os.name == 'nt' and geometry_mode != GeometryMode.POINT:
        # Mask analytics keeps several full-frame arrays alive per task; on Windows
        # too many parallel workers lead to paging and eventual OOM before any speedup.
        safe_limit = 2
        if not include_model_confidence:
            side_limit = int(analysis_max_side or 0)
            if side_limit > 0 and side_limit <= 1024:
                safe_limit = 4
            elif side_limit == 0:
                safe_limit = 3
        worker_count = min(worker_count, safe_limit)

    def run_sequential():
        for index, record in enumerate(records, start=1):
            if cancel_check is not None and cancel_check():
                raise BuildCancelledError("Build cancelled")
            if state_callback is not None:
                state_callback(record.key, "running")
            payload = _analyze_record_payload(
                record,
                model_specs,
                analysis_max_side,
                geometry_mode,
                point_match_radius,
                boundary_radius,
                confidence_uncertainty_delta,
                point_confidence_radius,
                polygon_confidence_summary,
                cache_enabled,
                include_model_confidence,
            )
            if state_callback is not None:
                state_callback(record.key, "done")
            if progress_callback is not None:
                progress_callback(index, len(records), record.key)
            yield record.key, payload

    if worker_count <= 1 or len(records) <= 1:
        yield from run_sequential()
        return

    try:
        use_thread_pool = os.name == 'nt'
        executor_cls = ThreadPoolExecutor if use_thread_pool else ProcessPoolExecutor
        work_items = [
            (
                record,
                model_specs,
                analysis_max_side,
                geometry_mode,
                point_match_radius,
                boundary_radius,
                confidence_uncertainty_delta,
                point_confidence_radius,
                polygon_confidence_summary,
                cache_enabled,
                include_model_confidence,
            )
            for record in records
        ]
        with executor_cls(max_workers=worker_count) as executor:
            try:
                completed = 0
                # Keep the queue aligned with the number of active workers so the UI
                # reflects actual parallel execution instead of pre-buffered tasks.
                max_in_flight = max(1, worker_count)
                next_index = 0
                future_to_order: dict[object, int] = {}
                ordered_results: dict[int, tuple[str, dict[str, object] | None]] = {}
                next_yield_index = 0

                def submit_work_item(order_index: int) -> None:
                    future = executor.submit(_analyze_record_payload_for_executor, work_items[order_index])
                    future_to_order[future] = order_index
                    if state_callback is not None:
                        state_callback(work_items[order_index][0].key, "running")

                while next_index < len(work_items) and len(future_to_order) < max_in_flight:
                    submit_work_item(next_index)
                    next_index += 1

                while future_to_order:
                    if cancel_check is not None and cancel_check():
                        if state_callback is not None:
                            for pending_order in future_to_order.values():
                                state_callback(work_items[pending_order][0].key, "stale")
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise BuildCancelledError("Build cancelled")
                    done, _pending = wait(tuple(future_to_order.keys()), return_when=FIRST_COMPLETED)
                    for future in done:
                        order_index = future_to_order.pop(future)
                        record_key, payload = future.result()
                        ordered_results[order_index] = (record_key, payload)
                        if state_callback is not None:
                            state_callback(record_key, "done")
                    while next_yield_index in ordered_results:
                        record_key, payload = ordered_results.pop(next_yield_index)
                        completed += 1
                        if progress_callback is not None:
                            progress_callback(completed, len(records), record_key)
                        yield record_key, payload
                        next_yield_index += 1
                        if completed % 32 == 0:
                            _clear_runtime_image_caches()
                    while next_index < len(work_items) and len(future_to_order) < max_in_flight:
                        submit_work_item(next_index)
                        next_index += 1
            except Exception:
                if state_callback is not None:
                    for pending_order in future_to_order.values():
                        state_callback(work_items[pending_order][0].key, "stale")
                executor.shutdown(wait=False, cancel_futures=True)
                raise
    except (PermissionError, OSError):
        yield from run_sequential()



def _compute_record_payloads(
    records: list[FrameRecord],
    model_specs: tuple[ModelSpec, ...],
    analysis_max_side: int | None,
    max_workers: int,
    *,
    geometry_mode: GeometryMode,
    point_match_radius: float,
    boundary_radius: int,
    confidence_uncertainty_delta: float,
    point_confidence_radius: int,
    polygon_confidence_summary: str,
    cache_enabled: bool,
    include_model_confidence: bool = True,
    progress_callback=None,
    state_callback=None,
    cancel_check=None,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for record_key, payload in _iter_record_payloads(
        records,
        model_specs,
        analysis_max_side,
        max_workers,
        geometry_mode=geometry_mode,
        point_match_radius=point_match_radius,
        boundary_radius=boundary_radius,
        confidence_uncertainty_delta=confidence_uncertainty_delta,
        point_confidence_radius=point_confidence_radius,
        polygon_confidence_summary=polygon_confidence_summary,
        cache_enabled=cache_enabled,
        include_model_confidence=include_model_confidence,
        progress_callback=progress_callback,
        state_callback=state_callback,
        cancel_check=cancel_check,
    ):
        if payload is not None:
            results[record_key] = payload
    return results



def _aggregate_model_ranking(records: list[FrameRecord], model_specs: tuple[ModelSpec, ...]) -> tuple[ModelAggregateScore, ...]:
    score_map: dict[str, list[float]] = {spec.model_id: [] for spec in model_specs}
    for record in records:
        summary = record.summary
        if summary is None:
            continue
        for model_id, metrics in (summary.model_metrics or {}).items():
            score = getattr(metrics, "quality_score", None)
            if score is None:
                continue
            score_map.setdefault(str(model_id), []).append(float(score))
    rows: list[ModelAggregateScore] = []
    for spec in model_specs:
        values = score_map.get(spec.model_id, [])
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        rows.append(ModelAggregateScore(
            model_id=spec.model_id,
            display_name=spec.display_name,
            labeled_frame_count=int(len(values)),
            mean_supervised_score=float(np.mean(array, dtype=np.float64)),
            median_supervised_score=float(np.median(array)),
            rank=0,
        ))
    rows.sort(key=lambda item: (item.mean_supervised_score, item.median_supervised_score, item.labeled_frame_count), reverse=True)
    return tuple(ModelAggregateScore(
        model_id=item.model_id,
        display_name=item.display_name,
        labeled_frame_count=item.labeled_frame_count,
        mean_supervised_score=item.mean_supervised_score,
        median_supervised_score=item.median_supervised_score,
        rank=index + 1,
    ) for index, item in enumerate(rows))


def _record_metric_value(summary: FrameAnalysisSummary | None, metric_key: str) -> float | None:
    if summary is None:
        return None
    value = summary.metric_values.get(metric_key)
    if value is not None:
        return float(value)
    if hasattr(summary, metric_key):
        raw = getattr(summary, metric_key)
        return None if raw is None else float(raw)
    return None


def _model_metric_key(metric_family: str, model_id: str) -> str:
    return f"{metric_family}::{model_id}"


def _parse_model_metric_key(metric_key: str) -> tuple[str, str] | None:
    if '::' not in str(metric_key):
        return None
    family, model_id = str(metric_key).split('::', 1)
    return (family, model_id) if family and model_id else None


def _available_metric_keys_for_models(model_specs: tuple[ModelSpec, ...]) -> tuple[str, ...]:
    keys = [
        "overall_frame_score",
        "export_priority_score",
        "model_model_score",
        "model_labeled_score",
        "disagreement_score",
        "labeled_best_quality",
        "labeled_mean_quality",
        "overall_polygon_score",
        "iou_score",
        "dice_score",
        "polygon_bce_score",
        "iou",
        "dice",
        "bce",
        "overall_point_score",
        "precision_score",
        "recall_score",
        "f1_score",
        "localization_score",
        "precision",
        "recall",
        "f1",
        "mean_localization_distance",
    ]
    for spec in model_specs:
        keys.append(_model_metric_key("model_confidence", spec.model_id))
        keys.append(_model_metric_key("model_uncertain_fraction", spec.model_id))
        keys.append(_model_metric_key("model_point_contrast", spec.model_id))
    return tuple(keys)


def _collect_frame_records_modern(
    model_specs: tuple[ModelSpec, ...],
    options: BuildOptions,
    *,
    original_folder: FolderSpec | None = None,
    gt_folder: FolderSpec | None = None,
    cancel_check=None,
) -> BuildResult:
    if not model_specs:
        raise ValueError("At least one model folder must be selected.")

    extensions = tuple(str(ext).lower() for ext in options.file_extensions)
    base_spec = model_specs[0]
    if cancel_check is not None and cancel_check():
        raise BuildCancelledError("Build cancelled")
    base_index = build_folder_index(base_spec.mask_folder, recursive=options.recursive, extensions=extensions, cancel_check=cancel_check)
    if not base_index:
        raise ValueError("Selected model folders do not contain matching image frames.")

    original_index = build_folder_index(original_folder.path, recursive=options.recursive, extensions=extensions, cancel_check=cancel_check) if original_folder is not None else {}
    gt_index = build_folder_index(gt_folder.path, recursive=options.recursive, extensions=extensions, cancel_check=cancel_check) if gt_folder is not None else {}
    fallback_model_indexes: dict[str, dict[str, Path]] = {}
    fallback_prob_indexes: dict[str, dict[str, Path]] = {}

    records: list[FrameRecord] = []
    sorted_keys = sorted(base_index.keys(), key=natural_sort_key)
    for index, key in enumerate(sorted_keys):
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError("Build cancelled")
        model_mask_paths: dict[str, str] = {base_spec.model_id: str(base_index[key])}
        model_prob_paths: dict[str, str] = {}
        if base_spec.prob_folder is not None:
            resolved_prob = _resolve_model_path_for_key(
                base_spec.prob_folder,
                key,
                extensions,
                fallback_prob_indexes,
                recursive=options.recursive,
                cancel_check=cancel_check,
            )
            model_prob_paths[base_spec.model_id] = str(resolved_prob) if resolved_prob is not None else ""
        else:
            model_prob_paths[base_spec.model_id] = ""
        missing_required_model = False
        for spec in model_specs[1:]:
            resolved = _resolve_model_path_for_key(
                spec.mask_folder,
                key,
                extensions,
                fallback_model_indexes,
                recursive=options.recursive,
                cancel_check=cancel_check,
            )
            if resolved is None:
                missing_required_model = True
                break
            model_mask_paths[spec.model_id] = str(resolved)
            if spec.prob_folder is not None:
                resolved_prob = _resolve_model_path_for_key(
                    spec.prob_folder,
                    key,
                    extensions,
                    fallback_prob_indexes,
                    recursive=options.recursive,
                    cancel_check=cancel_check,
                )
                model_prob_paths[spec.model_id] = str(resolved_prob) if resolved_prob is not None else ""
            else:
                model_prob_paths[spec.model_id] = ""
        if missing_required_model:
            continue
        original_path = _resolve_aux_path(key, original_index)
        gt_path = _resolve_aux_path(key, gt_index)
        records.append(FrameRecord(
            key=key,
            display_name=Path(key).name,
            identity=build_frame_identity(key, index),
            first_path=str(base_index[key]),
            second_path=str(model_mask_paths.get(model_specs[1].model_id, model_mask_paths[base_spec.model_id])) if len(model_specs) > 1 else str(base_index[key]),
            base_path=str(original_path) if original_path is not None else None,
            original_path=str(original_path) if original_path is not None else None,
            gt_path=str(gt_path) if gt_path is not None else None,
            model_mask_paths=model_mask_paths,
            model_prob_paths=model_prob_paths,
        ))
    return BuildResult(
        records=tuple(records),
        model_specs=model_specs,
        original_folder=original_folder,
        gt_folder=gt_folder,
        first_folder=FolderSpec(path=model_specs[0].mask_folder, label=model_specs[0].display_name) if len(model_specs) > 0 else None,
        second_folder=FolderSpec(path=model_specs[1].mask_folder, label=model_specs[1].display_name) if len(model_specs) > 1 else None,
        base_folder=original_folder,
        options=options,
        min_score=0.0,
        max_score=0.0,
        eligible_key_count=len(records),
        scores_computed=False,
        best_match_key=None,
        min_absolute_score=None,
        max_absolute_score=None,
        selected_metric_key="overall_frame_score",
        available_metric_keys=_available_metric_keys_for_models(model_specs),
    )


def collect_frame_records(
    model_specs: tuple[ModelSpec, ...] | FolderSpec,
    options: BuildOptions | FolderSpec,
    maybe_options: BuildOptions | None = None,
    *,
    original_folder: FolderSpec | None = None,
    gt_folder: FolderSpec | None = None,
    base_folder: FolderSpec | None = None,
    cancel_check=None,
) -> BuildResult:
    """Collect matched frame records in modern or legacy-lite mode."""

    if isinstance(model_specs, FolderSpec):
        first_folder = model_specs
        second_folder = options if isinstance(options, FolderSpec) else None
        build_options = maybe_options if isinstance(maybe_options, BuildOptions) else BuildOptions()
        if second_folder is None:
            raise ValueError("Legacy collect_frame_records requires two folder specs.")
        compat_model_specs = (
            ModelSpec(model_id="first", display_name=first_folder.label, mask_folder=first_folder.path, threshold=float(build_options.mask_threshold)),
            ModelSpec(model_id="second", display_name=second_folder.label, mask_folder=second_folder.path, threshold=float(build_options.mask_threshold)),
        )
        modern = _collect_frame_records_modern(
            compat_model_specs,
            build_options,
            original_folder=base_folder,
            gt_folder=None,
            cancel_check=cancel_check,
        )
        records: list[FrameRecord] = []
        for record in modern.records:
            first_path = str(record.model_mask_paths.get("first", ""))
            second_path = str(record.model_mask_paths.get("second", ""))
            records.append(
                replace(
                    record,
                    first_path=first_path,
                    second_path=second_path,
                    base_path=record.original_path,
                )
            )
        return replace(
            modern,
            records=tuple(records),
            first_folder=first_folder,
            second_folder=second_folder,
            base_folder=base_folder,
            original_folder=base_folder,
        )

    resolved_options = options if isinstance(options, BuildOptions) else maybe_options
    if not isinstance(resolved_options, BuildOptions):
        raise ValueError("collect_frame_records requires BuildOptions for modern mode.")
    return _collect_frame_records_modern(
        tuple(model_specs),
        resolved_options,
        original_folder=original_folder,
        gt_folder=gt_folder,
        cancel_check=cancel_check,
    )


def _sequence_groups(records: tuple[FrameRecord, ...]) -> dict[str, list[FrameRecord]]:
    groups: dict[str, list[FrameRecord]] = {}
    for record in records:
        sequence_id = record.identity.sequence_id if record.identity is not None and record.identity.sequence_id else "__root__"
        groups.setdefault(sequence_id, []).append(record)
    for items in groups.values():
        items.sort(key=lambda item: natural_sort_key(item.key))
    return groups


def compute_build_result_analytics(build_result: BuildResult, *, metric_key: str | None = None, progress_callback=None, state_callback=None, cancel_check=None) -> BuildResult:
    _clear_runtime_image_caches()
    records = list(build_result.records)
    if not records:
        return replace(build_result, scores_computed=True)

    updated_records: list[FrameRecord] = []
    try:
        include_model_confidence = _metric_requires_model_confidence(metric_key or build_result.selected_metric_key)
        payload_iter = _iter_record_payloads(
            records,
            build_result.model_specs,
            build_result.options.analysis_max_side,
            build_result.options.max_workers,
            geometry_mode=build_result.options.geometry_mode,
            point_match_radius=float(build_result.options.point_match_radius),
            boundary_radius=int(getattr(build_result.options, 'boundary_radius', 1) or 1),
            confidence_uncertainty_delta=float(getattr(build_result.options, 'confidence_uncertainty_delta', MODEL_CONFIDENCE_UNCERTAIN_DELTA)),
            point_confidence_radius=int(getattr(build_result.options, 'point_confidence_radius', POINT_CONFIDENCE_NEIGHBOR_RADIUS) or POINT_CONFIDENCE_NEIGHBOR_RADIUS),
            polygon_confidence_summary=str(getattr(build_result.options, 'polygon_confidence_summary', POLYGON_CONFIDENCE_SUMMARY_WEIGHTED) or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED),
            cache_enabled=bool(build_result.options.cache_enabled),
            include_model_confidence=include_model_confidence,
            progress_callback=progress_callback,
            state_callback=state_callback,
            cancel_check=cancel_check,
        )
        for record, (record_key, payload) in zip(records, payload_iter):
            if payload is None:
                continue
            if record.key != record_key:
                continue
            vector = dict(payload.get('vector') or {})
            disagreement = float(payload.get('disagreement', 0.0))
            model_model_score = float(payload.get('model_model_score', 0.0))
            model_labeled_score = payload.get('model_labeled_score')
            frame_type = 'point' if str(payload.get('geometry_mode')) == GeometryMode.POINT.value else 'polygon'
            labeled_best = payload.get('labeled_best')
            labeled_mean = payload.get('labeled_mean')
            model_diagnostics = payload.get('model_diagnostics') or {}
            model_metrics = payload.get('model_metrics') or {}
            model_confidence = payload.get('model_confidence') or {}
            pairwise_rows = tuple(payload.get('pairwise_rows', ()))
            polygon_scores = _aggregate_inter_model_polygon_scores(pairwise_rows) if frame_type != 'point' else {}
            point_scores = _aggregate_inter_model_point_scores(pairwise_rows, float(build_result.options.point_match_radius)) if frame_type == 'point' else {}

            export_priority = float(_clip01(1.0 - labeled_best)) if labeled_best is not None else float(_clip01(disagreement))
            metric_values = {
                'overall_frame_score': export_priority,
                'export_priority_score': export_priority,
                'model_model_score': model_model_score,
                'disagreement_score': disagreement,
            }
            if model_labeled_score is not None:
                metric_values['model_labeled_score'] = float(model_labeled_score)
            if labeled_best is not None:
                metric_values['labeled_best_quality'] = labeled_best
            if labeled_mean is not None:
                metric_values['labeled_mean_quality'] = labeled_mean
            metric_values.update(polygon_scores)
            metric_values.update(point_scores)
            for model_id, confidence_row in model_confidence.items():
                if hasattr(confidence_row, 'mean_object_confidence'):
                    metric_values[_model_metric_key('model_confidence', str(model_id))] = float(getattr(confidence_row, 'frame_uncertainty_score', 0.0))
                    metric_values[_model_metric_key('model_uncertain_fraction', str(model_id))] = float(getattr(confidence_row, 'uncertain_fraction', 0.0))
                elif hasattr(confidence_row, 'mean_point_confidence'):
                    metric_values[_model_metric_key('model_confidence', str(model_id))] = float(getattr(confidence_row, 'frame_uncertainty_score', 0.0))
                    metric_values[_model_metric_key('model_point_contrast', str(model_id))] = float(getattr(confidence_row, 'mean_point_contrast', 0.0))

            summary = FrameAnalysisSummary(
                is_labeled=labeled_best is not None,
                disagreement_score=float(disagreement),
                temporal_instability=0.0,
                structural_anomaly=0.0,
                labeled_best_quality=labeled_best,
                labeled_mean_quality=labeled_mean,
                export_priority_score=float(export_priority),
                metric_values=metric_values,
                model_metrics=model_metrics,
                model_confidence=model_confidence,
                model_diagnostics=model_diagnostics,
                pairwise_metrics=pairwise_rows,
                notes=(('labeled' if labeled_best is not None else 'unlabeled'),),
                frame_type=frame_type,
            )
            updated_records.append(replace(record, summary=summary))
    finally:
        _clear_runtime_image_caches()

    active_metric = metric_key or build_result.selected_metric_key or 'overall_frame_score'
    absolute_scores = [float(_record_metric_value(record.summary, active_metric) or 0.0) for record in updated_records]
    min_absolute = min(absolute_scores) if absolute_scores else 0.0
    max_absolute = max(absolute_scores) if absolute_scores else 0.0
    span = max(EPS, max_absolute - min_absolute)
    higher_is_better = metric_higher_is_better(active_metric)

    scored_records: list[FrameRecord] = []
    best_key = None
    best_value = None
    for record, absolute in zip(updated_records, absolute_scores):
        relative = 0.0 if abs(max_absolute - min_absolute) <= EPS else (absolute - min_absolute) / span
        display = relative if higher_is_better else (1.0 - relative)
        scored = replace(record, score=float(display), absolute_score=float(absolute), relative_score=float(relative), score_ready=True)
        scored_records.append(scored)
        if best_value is None or (absolute > best_value if higher_is_better else absolute < best_value):
            best_value = absolute
            best_key = record.key

    percentile_map = compute_metric_percentiles(scored_records, active_metric)
    scored_records = [replace(record, score_percentile=float(percentile_map.get(record.key, 0.0))) for record in scored_records]
    model_ranking = _aggregate_model_ranking(scored_records, build_result.model_specs)
    available_metric_keys: list[str] = []
    seen_metric_keys: set[str] = set()
    for metric_key_candidate in _available_metric_keys_for_models(build_result.model_specs):
        if metric_key_candidate not in seen_metric_keys:
            available_metric_keys.append(metric_key_candidate)
            seen_metric_keys.add(metric_key_candidate)
    for record in scored_records:
        summary = record.summary
        if summary is None:
            continue
        for metric_key_candidate in summary.metric_values.keys():
            metric_key_str = str(metric_key_candidate)
            if metric_key_str not in seen_metric_keys:
                available_metric_keys.append(metric_key_str)
                seen_metric_keys.add(metric_key_str)
    return replace(build_result, records=tuple(scored_records), min_score=min((record.score for record in scored_records), default=0.0), max_score=max((record.score for record in scored_records), default=0.0), scores_computed=True, best_match_key=best_key, min_absolute_score=min_absolute, max_absolute_score=max_absolute, selected_metric_key=active_metric, model_ranking=model_ranking, available_metric_keys=tuple(available_metric_keys))


def compute_comparison_score(first: np.ndarray, second: np.ndarray, mode: ComparisonMode) -> float:
    """Legacy-lite helper that returns only scalar score for a comparison mode."""

    _heatmap, score = compute_comparison(first, second, mode)
    return float(score)


def load_frame_layers(record: FrameRecord) -> dict[str, object]:
    """Legacy-lite helper returning first/second/base grayscale and binary layers."""

    first_path = str(record.first_path or "")
    second_path = str(record.second_path or "")
    if not first_path and record.model_mask_paths:
        first_path = str(next(iter(record.model_mask_paths.values()), ""))
    if not second_path and len(record.model_mask_paths) > 1:
        second_path = str(list(record.model_mask_paths.values())[1])
    if not second_path:
        second_path = first_path
    first_gray = load_grayscale_image(Path(first_path))
    second_gray = load_grayscale_image(Path(second_path))
    if second_gray.shape != first_gray.shape:
        second_gray = resize_grayscale_image(second_gray, tuple(int(v) for v in first_gray.shape))
    base_gray = None
    base_path = record.base_path or record.original_path
    if base_path:
        base_gray = load_grayscale_image(Path(base_path))
        if base_gray.shape != first_gray.shape:
            base_gray = resize_grayscale_image(base_gray, tuple(int(v) for v in first_gray.shape))
    return {
        "first_gray": first_gray.copy(),
        "second_gray": second_gray.copy(),
        "first_binary": np.asarray(first_gray >= 128, dtype=bool),
        "second_binary": np.asarray(second_gray >= 128, dtype=bool),
        "base_gray": None if base_gray is None else base_gray.copy(),
        "shape": tuple(int(value) for value in first_gray.shape),
    }


def compute_build_result_mismatches(
    build_result: BuildResult,
    *,
    comparison_mode: ComparisonMode | None = None,
    display_metric: str = "relative",
    progress_callback=None,
    cancel_check=None,
) -> BuildResult:
    """Legacy-lite mismatch pipeline wrapper implemented on top of analytics."""

    mode = comparison_mode or build_result.options.comparison_mode
    normalized_records: list[FrameRecord] = []
    total = max(1, len(build_result.records))
    for index, record in enumerate(build_result.records, start=1):
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError("Build cancelled")
        layers = load_frame_layers(record)
        score = compute_comparison_score(layers["first_binary"], layers["second_binary"], mode)
        normalized_records.append(
            replace(
                record,
                absolute_score=float(score),
                score=float(score),
                score_ready=True,
            )
        )
        if progress_callback is not None:
            progress_callback(index, total, record.key)
    if not normalized_records:
        return replace(build_result, options=replace(build_result.options, comparison_mode=mode), scores_computed=True)
    absolute_scores = [float(record.absolute_score or 0.0) for record in normalized_records]
    min_absolute = float(min(absolute_scores))
    max_absolute = float(max(absolute_scores))
    span = max(EPS, max_absolute - min_absolute)
    use_absolute = str(display_metric or "relative").lower() == "absolute"
    updated_records = []
    for record in normalized_records:
        absolute = float(record.absolute_score or 0.0)
        relative = 0.0 if abs(max_absolute - min_absolute) <= EPS else float((absolute - min_absolute) / span)
        display = absolute if use_absolute else relative
        updated_records.append(replace(record, score=float(display), relative_score=float(relative), score_ready=True))
    best_record = min(updated_records, key=lambda item: float(item.absolute_score or 0.0))
    return replace(
        build_result,
        records=tuple(updated_records),
        options=replace(build_result.options, comparison_mode=mode),
        min_score=min((record.score for record in updated_records), default=0.0),
        max_score=max((record.score for record in updated_records), default=0.0),
        scores_computed=True,
        best_match_key=best_record.key,
        min_absolute_score=min_absolute,
        max_absolute_score=max_absolute,
    )


def build_frame_records(
    first_folder: FolderSpec,
    second_folder: FolderSpec,
    options: BuildOptions,
    *,
    base_folder: FolderSpec | None = None,
    progress_callback=None,
    cancel_check=None,
) -> BuildResult:
    """Legacy-lite full build wrapper (index + mismatch compute)."""

    initial = collect_frame_records(first_folder, second_folder, options, base_folder=base_folder, cancel_check=cancel_check)
    return compute_build_result_mismatches(
        initial,
        comparison_mode=options.comparison_mode,
        display_metric="relative",
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )



def metric_value_for_record(record: FrameRecord, metric_key: str) -> float | None:
    return _record_metric_value(record.summary, metric_key)


def metric_higher_is_better(metric_key: str) -> bool:
    metric_key = str(metric_key or "")
    parsed = _parse_model_metric_key(metric_key)
    if parsed is not None:
        family, _model_id = parsed
        if family == "model_uncertain_fraction":
            return False
        if family == "model_confidence":
            return False
        if family == "model_point_contrast":
            return True
    if metric_key in {"bce", "mean_localization_distance"}:
        return False
    if metric_key in {
        "labeled_best_quality",
        "labeled_mean_quality",
        "model_model_score",
        "model_labeled_score",
        "iou",
        "dice",
        "iou_score",
        "dice_score",
        "polygon_bce_score",
        "overall_polygon_score",
        "precision",
        "recall",
        "f1",
        "precision_score",
        "recall_score",
        "f1_score",
        "localization_score",
        "overall_point_score",
    }:
        return True
    return False


def metric_percentile_high_is_bad(metric_key: str) -> bool:
    metric_key = str(metric_key or "")
    parsed = _parse_model_metric_key(metric_key)
    if parsed is not None:
        family, _model_id = parsed
        if family == "model_confidence":
            return True
    return False


def rank_records_by_metric(records: tuple[FrameRecord, ...] | list[FrameRecord], metric_key: str) -> list[FrameRecord]:
    higher_is_better = metric_higher_is_better(metric_key)
    scored: list[FrameRecord] = []
    for record in records:
        value = metric_value_for_record(record, metric_key)
        if value is None or not np.isfinite(float(value)):
            continue
        scored.append(record)
    return sorted(
        scored,
        key=lambda item: float(metric_value_for_record(item, metric_key) or 0.0),
        reverse=bool(higher_is_better),
    )


def compute_metric_percentiles(records: tuple[FrameRecord, ...] | list[FrameRecord], metric_key: str) -> dict[str, float]:
    ranked = rank_records_by_metric(records, metric_key)
    if not ranked:
        return {}
    if len(ranked) == 1:
        return {ranked[0].key: (0.0 if metric_percentile_high_is_bad(metric_key) else 100.0)}
    denominator = max(1, len(ranked) - 1)
    if metric_percentile_high_is_bad(metric_key):
        return {record.key: float(100.0 * index / denominator) for index, record in enumerate(ranked)}
    return {record.key: float(100.0 * (denominator - index) / denominator) for index, record in enumerate(ranked)}


def select_candidate_records(
    build_result: BuildResult,
    *,
    metric_key: str,
    selection_mode: str = EXPORT_SELECTION_MODE_COUNT,
    top_k: int = 32,
    top_percent: float = 10.0,
    percentile_threshold: float = 90.0,
) -> tuple[FrameRecord, ...]:
    ranked = rank_records_by_metric(build_result.records, metric_key)
    if not ranked:
        return tuple()
    mode = str(selection_mode or EXPORT_SELECTION_MODE_COUNT)
    if mode == EXPORT_SELECTION_MODE_PERCENT:
        count = max(1, int(math.ceil(len(ranked) * max(0.0, float(top_percent)) / 100.0)))
        return tuple(ranked[:count])
    if mode == EXPORT_SELECTION_MODE_PERCENTILE:
        percentiles = compute_metric_percentiles(ranked, metric_key)
        selected = [record for record in ranked if float(percentiles.get(record.key, 0.0)) >= float(percentile_threshold)]
        return tuple(selected or ranked[:1])
    count = max(1, int(top_k))
    return tuple(ranked[:count])



def load_frame_detail_base(
    record: FrameRecord,
    build_result: BuildResult,
    model_id: str | None = None,
    *,
    max_side: int | None = None,
) -> dict[str, object]:
    active_record = record
    active_build_result = build_result
    if not active_build_result.model_specs:
        first_path = str(active_record.first_path or "")
        second_path = str(active_record.second_path or "")
        if not first_path and active_record.model_mask_paths:
            first_path = str(next(iter(active_record.model_mask_paths.values()), ""))
        if not second_path and len(active_record.model_mask_paths) > 1:
            second_path = str(list(active_record.model_mask_paths.values())[1])
        if not second_path:
            second_path = first_path
        if first_path:
            compat_specs = (
                ModelSpec(model_id="first", display_name="1", mask_folder=Path(first_path).parent, threshold=0.5),
                ModelSpec(model_id="second", display_name="2", mask_folder=Path(second_path).parent if second_path else Path(first_path).parent, threshold=0.5),
            )
            active_record = replace(
                active_record,
                model_mask_paths={"first": first_path, "second": second_path},
                model_prob_paths={"first": "", "second": ""},
                original_path=active_record.original_path or active_record.base_path,
                base_path=active_record.base_path or active_record.original_path,
            )
            active_build_result = replace(
                active_build_result,
                model_specs=compat_specs,
                original_folder=active_build_result.original_folder or active_build_result.base_folder,
                first_folder=active_build_result.first_folder or FolderSpec(path=compat_specs[0].mask_folder, label=compat_specs[0].display_name),
                second_folder=active_build_result.second_folder or FolderSpec(path=compat_specs[1].mask_folder, label=compat_specs[1].display_name),
                base_folder=active_build_result.base_folder or active_build_result.original_folder,
                options=replace(active_build_result.options, geometry_mode=GeometryMode.MASK),
            )

    target_model_id = model_id or (active_build_result.model_specs[0].model_id if active_build_result.model_specs else None)
    cache_key = _detail_base_payload_cache_key(active_record, active_build_result, max_side)
    cached = _load_cached_detail_payload(cache_key)
    if isinstance(cached, dict):
        return _with_selected_detail_payload(cached, target_model_id)

    boundary_radius = int(getattr(active_build_result.options, 'boundary_radius', 1) or 1)
    confidence_uncertainty_delta = float(getattr(active_build_result.options, 'confidence_uncertainty_delta', MODEL_CONFIDENCE_UNCERTAIN_DELTA))
    point_confidence_radius = int(getattr(active_build_result.options, 'point_confidence_radius', POINT_CONFIDENCE_NEIGHBOR_RADIUS) or POINT_CONFIDENCE_NEIGHBOR_RADIUS)
    polygon_confidence_summary = str(getattr(active_build_result.options, 'polygon_confidence_summary', POLYGON_CONFIDENCE_SUMMARY_WEIGHTED) or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED)

    probabilities, masks, _model_structures, model_diagnostics, model_metrics, model_confidence, original_gray, gt_mask, _gt_structure, detail_geometry_mode, model_views, gt_point_view = _build_model_payloads(
        active_record,
        active_build_result.model_specs,
        analysis_max_side=max_side,
        geometry_mode=active_build_result.options.geometry_mode,
        point_match_radius=float(active_build_result.options.point_match_radius),
        boundary_radius=boundary_radius,
        confidence_uncertainty_delta=confidence_uncertainty_delta,
        point_confidence_radius=point_confidence_radius,
        polygon_confidence_summary=polygon_confidence_summary,
        include_confidence_objects=False,
        include_model_confidence=False,
        include_structure_details=True,
    )

    gt_gray = _load_optional_gray(active_record.gt_path, target_shape=tuple(int(v) for v in gt_mask.shape) if gt_mask is not None else (tuple(int(v) for v in original_gray.shape) if original_gray is not None else None), max_side=max_side)
    model_display_names = {spec.model_id: spec.display_name for spec in active_build_result.model_specs}
    selected_model_id = target_model_id if target_model_id in probabilities else (next(iter(probabilities.keys()), None))
    summary_confidence = {}
    if active_record.summary is not None and getattr(active_record.summary, 'model_confidence', None):
        summary_confidence = dict(getattr(active_record.summary, 'model_confidence', {}) or {})
    if summary_confidence:
        model_confidence = {**summary_confidence, **model_confidence}
    fallback_prob = np.zeros_like(next(iter(probabilities.values()))) if probabilities else np.zeros((1, 1), dtype=np.float32)
    selected_prob = probabilities.get(selected_model_id, fallback_prob)
    selected_mask = masks.get(selected_model_id, np.asarray(selected_prob >= 0.5, dtype=bool))
    consensus_prob = _consensus_probability(list(probabilities.values())) if probabilities else selected_prob
    consensus_mask = np.asarray(consensus_prob >= 0.5, dtype=bool)

    detail_payload = {
        "model_ids": tuple(probabilities.keys()),
        "model_display_names": model_display_names,
        "selected_model_id": selected_model_id,
        "original_gray": original_gray,
        "gt_gray": gt_gray,
        "gt_mask": gt_mask,
        "selected_prob": np.asarray(selected_prob, dtype=np.float32),
        "selected_mask": np.asarray(selected_mask, dtype=bool),
        "model_probabilities": probabilities,
        "model_masks": masks,
        "model_views": model_views,
        "gt_point_view": gt_point_view,
        "consensus_prob": consensus_prob,
        "consensus_mask": consensus_mask,
        "pairwise_model_comparisons": tuple(active_record.summary.pairwise_metrics) if active_record.summary is not None and active_record.summary.pairwise_metrics else _pairwise_model_comparisons(probabilities, masks, geometry_mode=detail_geometry_mode, model_views=model_views, point_match_radius=float(active_build_result.options.point_match_radius)),
        "frame_metrics": dict(active_record.summary.metric_values) if active_record.summary is not None else {},
        "model_metrics": model_metrics,
        "model_confidence": model_confidence,
        "model_diagnostics": model_diagnostics,
        "geometry_mode": detail_geometry_mode.value,
        "point_match_radius": float(active_build_result.options.point_match_radius),
        "boundary_radius": boundary_radius,
        "confidence_uncertainty_delta": confidence_uncertainty_delta,
        "point_confidence_radius": point_confidence_radius,
        "polygon_confidence_summary": polygon_confidence_summary,
        "original_features": extract_original_frame_features(original_gray),
    }
    _store_cached_detail_payload(cache_key, detail_payload)
    return _with_selected_detail_payload(detail_payload, target_model_id)


def load_frame_detail_model_confidence(
    record: FrameRecord,
    build_result: BuildResult,
    model_id: str | None = None,
    *,
    max_side: int | None = None,
    detail_payload: dict[str, object] | None = None,
):
    target_model_id = model_id or (build_result.model_specs[0].model_id if build_result.model_specs else None)
    if target_model_id is None and (record.first_path or record.model_mask_paths):
        target_model_id = "first" if (record.first_path or "first" in record.model_mask_paths) else next(iter(record.model_mask_paths.keys()), None)
    if not target_model_id:
        return None

    payload = detail_payload if detail_payload is not None else load_frame_detail_base(
        record,
        build_result,
        model_id=target_model_id,
        max_side=max_side,
    )
    model_confidence = payload.setdefault("model_confidence", {})
    geometry_mode = str(payload.get("geometry_mode") or GeometryMode.MASK.value)
    existing = (model_confidence or {}).get(target_model_id)
    if _detail_confidence_payload_ready(existing, geometry_mode):
        return existing

    cache_key = _detail_confidence_cache_key(record, build_result, max_side, target_model_id)
    cached = _load_cached_detail_payload(cache_key)
    if _detail_confidence_payload_ready(cached, geometry_mode):
        model_confidence[target_model_id] = cached
        return cached

    probabilities = payload.get("model_probabilities") or {}
    masks = payload.get("model_masks") or {}
    boundary_radius = int(payload.get("boundary_radius") or int(getattr(build_result.options, 'boundary_radius', 1) or 1))
    confidence_uncertainty_delta = float(payload.get("confidence_uncertainty_delta") or float(getattr(build_result.options, 'confidence_uncertainty_delta', MODEL_CONFIDENCE_UNCERTAIN_DELTA)))
    point_confidence_radius = int(payload.get("point_confidence_radius") or int(getattr(build_result.options, 'point_confidence_radius', POINT_CONFIDENCE_NEIGHBOR_RADIUS) or POINT_CONFIDENCE_NEIGHBOR_RADIUS))
    polygon_confidence_summary = str(payload.get("polygon_confidence_summary") or str(getattr(build_result.options, 'polygon_confidence_summary', POLYGON_CONFIDENCE_SUMMARY_WEIGHTED) or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED))

    if geometry_mode == GeometryMode.POINT.value:
        selected_view = (payload.get("model_views") or {}).get(target_model_id)
        if selected_view is None:
            return None
        confidence_row = _point_internal_confidence(
            selected_view,
            neighborhood_radius=point_confidence_radius,
            include_objects=True,
        )
    else:
        selected_probability = probabilities.get(target_model_id)
        selected_model_mask = masks.get(target_model_id)
        if selected_probability is None or selected_model_mask is None:
            return None
        confidence_row = _polygon_frame_confidence(
            selected_probability,
            selected_model_mask,
            uncertainty_delta=confidence_uncertainty_delta,
            summary_metric=polygon_confidence_summary,
            include_debug=True,
        )
    model_confidence[target_model_id] = confidence_row
    _store_cached_detail_payload(cache_key, confidence_row)
    return confidence_row


def load_frame_detail(
    record: FrameRecord,
    build_result: BuildResult,
    model_id: str | None = None,
    *,
    max_side: int | None = None,
    include_selected_confidence: bool = True,
) -> dict[str, object]:
    target_model_id = model_id or (build_result.model_specs[0].model_id if build_result.model_specs else None)
    detail_payload = load_frame_detail_base(
        record,
        build_result,
        model_id=target_model_id,
        max_side=max_side,
    )
    if include_selected_confidence and target_model_id is not None:
        confidence_row = load_frame_detail_model_confidence(
            record,
            build_result,
            model_id=target_model_id,
            max_side=max_side,
            detail_payload=detail_payload,
        )
        if confidence_row is not None:
            (detail_payload.setdefault("model_confidence", {}))[target_model_id] = confidence_row
    return _with_selected_detail_payload(detail_payload, target_model_id)


def export_ranked_frames(
    build_result: BuildResult,
    destination: Path | str,
    *,
    top_k: int,
    neighbor_radius: int = 1,
    metric_key: str = "export_priority_score",
    selection_mode: str = EXPORT_SELECTION_MODE_COUNT,
    top_percent: float = 10.0,
    percentile_threshold: float = 90.0,
) -> dict[str, object]:
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    records = [record for record in build_result.records if record.summary is not None]
    if not records:
        raise ValueError("Nothing to export. Compute analytics first.")
    selected = list(select_candidate_records(
        replace(build_result, records=tuple(records)),
        metric_key=metric_key,
        selection_mode=selection_mode,
        top_k=top_k,
        top_percent=top_percent,
        percentile_threshold=percentile_threshold,
    ))
    percentile_map = compute_metric_percentiles(records, metric_key)
    sequence_groups = _sequence_groups(tuple(build_result.records))
    selected_keys = {record.key for record in selected}
    supplemental_keys: set[str] = set()
    radius = max(0, int(neighbor_radius))
    if radius > 0:
        for group in sequence_groups.values():
            key_to_index = {item.key: index for index, item in enumerate(group)}
            for record in selected:
                if record.key not in key_to_index:
                    continue
                center = key_to_index[record.key]
                for offset in range(-radius, radius + 1):
                    neighbor_index = center + offset
                    if neighbor_index < 0 or neighbor_index >= len(group):
                        continue
                    candidate = group[neighbor_index]
                    if candidate.key not in selected_keys:
                        supplemental_keys.add(candidate.key)

    export_records = {record.key: record for record in build_result.records if record.key in selected_keys | supplemental_keys}
    manifest: dict[str, object] = {
        "metric_key": metric_key,
        "selection_mode": selection_mode,
        "top_k": int(top_k),
        "top_percent": float(top_percent),
        "percentile_threshold": float(percentile_threshold),
        "selected_keys": [record.key for record in selected],
        "supplemental_keys": sorted(supplemental_keys, key=natural_sort_key),
        "scores": {record.key: metric_value_for_record(record, metric_key) for record in export_records.values()},
        "score_percentiles": {record.key: float(percentile_map.get(record.key, 0.0)) for record in export_records.values()},
        "reasons": {record.key: ([metric_key, "primary"] if record.key in selected_keys else [metric_key, "neighbor"]) for record in export_records.values()},
    }

    original_dir = destination_path / "original"
    gt_dir = destination_path / "gt"
    masks_root = destination_path / "models"
    probs_root = destination_path / "probabilities"
    for root in (original_dir, gt_dir, masks_root, probs_root):
        root.mkdir(parents=True, exist_ok=True)

    for key, record in export_records.items():
        normalized_name = key.replace("/", "__")
        if record.original_path:
            shutil.copy2(record.original_path, original_dir / Path(normalized_name).name)
        if record.gt_path:
            shutil.copy2(record.gt_path, gt_dir / Path(normalized_name).name)
        for model_id, path_text in record.model_mask_paths.items():
            model_dir = masks_root / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path_text, model_dir / Path(normalized_name).name)
        for model_id, path_text in record.model_prob_paths.items():
            if not path_text:
                continue
            model_dir = probs_root / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path_text, model_dir / Path(normalized_name).name)

    manifest_path = destination_path / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "selected_count": len(selected_keys),
        "supplemental_count": len(supplemental_keys),
        "manifest_path": str(manifest_path),
        "selected_keys": manifest["selected_keys"],
        "supplemental_keys": manifest["supplemental_keys"],
    }


def compute_comparison(first: np.ndarray, second: np.ndarray, mode: ComparisonMode) -> tuple[np.ndarray, float]:
    if mode == ComparisonMode.GRAYSCALE_DIFF:
        first_gray = np.asarray(first, dtype=np.float32)
        second_gray = np.asarray(second, dtype=np.float32)
        heatmap = np.abs(first_gray - second_gray)
        if heatmap.max() > 1.0:
            heatmap /= 255.0
        return heatmap.astype(np.float32), float(np.mean(heatmap, dtype=np.float64))
    first_bool = np.asarray(first, dtype=bool)
    second_bool = np.asarray(second, dtype=bool)
    if mode == ComparisonMode.OVERLAY_ONLY:
        heatmap = np.zeros_like(first_bool, dtype=np.float32)
    elif mode in {ComparisonMode.XOR, ComparisonMode.DISAGREEMENT}:
        heatmap = np.logical_xor(first_bool, second_bool).astype(np.float32)
    elif mode == ComparisonMode.FIRST_MINUS_SECOND:
        heatmap = np.logical_and(first_bool, np.logical_not(second_bool)).astype(np.float32)
    else:
        heatmap = np.logical_and(np.logical_not(first_bool), second_bool).astype(np.float32)
    return heatmap, float(np.mean(heatmap, dtype=np.float64))


















