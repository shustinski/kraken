# ruff: noqa: E402,F401

"""Shared imports, constants, and internal value types for Karakal analytics."""

from __future__ import annotations

import hashlib

import logging

import os

import json

import math

import pickle

import shutil

import ctypes

from collections import OrderedDict

from time import perf_counter

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait

from dataclasses import dataclass, replace

from functools import lru_cache

from pathlib import Path

from typing import Sequence

import numpy as np

from PyQt6.QtCore import Qt

from PyQt6.QtGui import QImage

try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

_CKDTREE_UNSET = object()

_cached_ckdtree: object = _CKDTREE_UNSET


def _get_ckdtree():
    """Import scipy.spatial only when a spatial metric actually needs it."""

    global _cached_ckdtree
    if _cached_ckdtree is _CKDTREE_UNSET:
        try:
            from scipy.spatial import cKDTree
        except Exception:
            cKDTree = None
        _cached_ckdtree = cKDTree
    return _cached_ckdtree


try:
    import cv2
except Exception:
    cv2 = None

from .backend_constants import (  # noqa: E402
    ANALYSIS_CACHE_DIR,
    ANALYSIS_CACHE_VERSION,
    BCE_SCORE_CAP,
    DETAIL_CACHE_DIR,
    IMAGE_CACHE_SIZE,
    INTER_MODEL_POINT_SCORE_WEIGHTS,
    INTER_MODEL_POLYGON_SCORE_WEIGHTS,
    NATURAL_SPLIT_PATTERN,
    MASK_AGREEMENT_SCORE_WEIGHTS,
    MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    MODEL_RISK_TOP_UNCERTAIN_FRACTION,
    MODEL_RISK_UNCERTAINTY_THRESHOLD,
    MODEL_RISK_WEIGHT_CLUSTER,
    MODEL_RISK_WEIGHT_FRACTION,
    MODEL_RISK_WEIGHT_MEAN,
    MODEL_RISK_WEIGHT_TOP,
    POINT_AGREEMENT_SCORE_WEIGHTS,
    POINT_CONFIDENCE_NEIGHBOR_RADIUS,
    POINT_SUPPORT_THRESHOLD,
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
    INVALID_FILENAME_PATTERN,
)

from .confidence_maps import build_model_uncertainty, confidence_bad_area_intensity, normalize_algorithmic_confidence

from .cache_utils import ByteLruCache, atomic_pickle_dump, estimate_size_bytes, trim_directory_by_bytes

from .domain import (  # noqa: E402
    BuildOptions,
    BuildResult,
    ComparisonPairSelection,
    ComparisonMode,
    ComparisonTarget,
    FolderSpec,
    FrameAnalysisSummary,
    FrameIdentity,
    FrameRecord,
    GeometryMode,
    ModelDiagnosticMetrics,
    ModelOutputConfidenceMetrics,
    ModelSpec,
    PointAgreementMetrics,
    PointConfidenceMetrics,
    PointDiagnosticMetrics,
    PointObjectConfidence,
    PolygonConfidenceDebugCandidate,
    PolygonConfidenceDebugData,
    PolygonConfidenceMetrics,
    PolygonConfidencePipelineConfig,
    PolygonObjectConfidence,
)

from .grid_anomaly import GridFrameAnalysisResult, detect_grid_cell_anomalies

from .profiling import current_profiler, profile_stage

from .performance import PerformanceConfig, load_performance_config

from ..comparison import (  # noqa: E402
    EnsembleComparisonRequest,
    FrameComparisonResult,
    ModelFrameResult,
    PairwiseComparisonRequest,
    compare_ensemble,
    compare_pairwise,
)

EPS = 1e-8

ANALYTICS_MAX_BATCH_SIZE = 16

ANALYTICS_BATCH_TARGETS_PER_WORKER = 32

ANALYTICS_WAIT_TIMEOUT_SECONDS = 0.25

ANALYTICS_STALL_TIMEOUT_SECONDS = 180.0

ANALYTICS_STALL_PROGRESS_SECONDS = 5.0

WINDOWS_THREAD_ANALYTICS_MAX_WORKERS = 64

WINDOWS_THREAD_CONFIDENCE_MAX_WORKERS = 32

WINDOWS_PROCESS_ANALYTICS_MAX_WORKERS = 4

ANALYTICS_WORKER_ENV = "VALIDATION_MATRIX_ANALYTICS_WORKERS"

EXPORT_WORKER_ENV = "KARAKAL_EXPORT_WORKERS"

EXPORT_MAX_WORKERS = 32

ANALYTICS_MEMORY_FRACTION = 0.45

ANALYSIS_CACHE_MAX_FILES = 100000

DETAIL_CACHE_MAX_FILES = 20000

CACHE_TRIM_INTERVAL_SECONDS = 300.0

_CACHE_TRIM_LAST_BY_DIR: dict[str, float] = {}

_LAST_ANALYTICS_WORKER_PLAN: dict[str, object] = {}

_LOGGER = logging.getLogger(__name__)

PAIR_METRIC_OPERATIONS = frozenset({"xor", "iou", "dice"})

CONFIDENCE_PAIR_METRIC_OPERATIONS = frozenset({"mae", "rmse", "mean_delta", "correlation", "low_iou", "disagreement"})

COMBINED_PAIR_OUTPUT_WEIGHT = 0.7

COMBINED_PAIR_CONFIDENCE_WEIGHT = 0.3

CONFIDENCE_LOW_THRESHOLD = 0.5

CONFIDENCE_DIFF_THRESHOLD = 0.25


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


def _active_performance_config() -> PerformanceConfig:
    profiler = current_profiler()
    return profiler.config if profiler is not None else load_performance_config()


def _is_binary_like_probability(probability: np.ndarray, *, tolerance: float = 1e-4) -> bool:
    prob = np.asarray(probability, dtype=np.float32)
    if prob.size == 0:
        return False
    finite = prob[np.isfinite(prob)]
    if finite.size == 0:
        return False
    distance_to_binary = np.minimum(np.abs(finite), np.abs(finite - 1.0))
    return bool(np.max(distance_to_binary, initial=0.0) <= float(max(EPS, tolerance)))
