"""marker_consolidation: ridge link."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True, slots=True)
class ConsolidationEvidence:
    """Image evidence used by consolidation. No GT, ids, or filenames."""

    intensity: np.ndarray
    ridge_confidence: np.ndarray
    ridge_orientation: np.ndarray
    structure_orientation: np.ndarray
    coherence: np.ndarray
    persistent_edge: np.ndarray
    magnitude: np.ndarray
    rim_response: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray

@dataclass(frozen=True, slots=True)
class RidgeLink:
    fragment_a: int
    fragment_b: int
    endpoint_a: int
    endpoint_b: int
    point_a: tuple[int, int]
    point_b: tuple[int, int]
    score: float
    orientation_score: float
    longitudinal_score: float
    corridor_score: float
    ridge_score: float
    intensity_score: float
    boundary_score: float
    offset_penalty: float
    reason: str = ""

@dataclass(frozen=True, slots=True)
class MarkerConsolidationStats:
    raw_ridge_count: int = 0
    logical_ridge_count: int = 0
    accepted_ridge_links: int = 0
    rejected_ridge_links: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    raw_wide_count: int = 0
    logical_wide_count: int = 0
    wide_unions: int = 0
    combined_logical_count: int = 0
    band_count: int = 0
    accepted_band_groups: int = 0
    wide_kept_with_multi_ridge: int = 0
    wide_dropped_for_separator: int = 0

@dataclass(frozen=True, slots=True)
class MarkerConsolidationResult:
    ridge_raw_labels: np.ndarray
    ridge_logical_labels: np.ndarray
    wide_raw_labels: np.ndarray
    wide_logical_labels: np.ndarray
    combined_labels: np.ndarray
    accepted_links: tuple[RidgeLink, ...]
    rejected_links: tuple[RidgeLink, ...]
    stats: MarkerConsolidationStats
    debug_images: dict[str, np.ndarray]

@dataclass(frozen=True, slots=True)
class _Endpoint:
    fragment_id: int
    index: int
    row: int
    col: int
    tangent_y: float
    tangent_x: float
    width: float
    mean_intensity: float
    mean_ridge: float

@dataclass(frozen=True, slots=True)
class _Fragment:
    fragment_id: int
    area: int
    centroid_row: float
    centroid_col: float
    mean_intensity: float
    mean_ridge: float
    width: float
    endpoints: tuple[_Endpoint, ...]
