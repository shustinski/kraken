"""Typed domain models for inter-model comparison."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

ComparisonMode = Literal["pairwise", "ensemble"]


@dataclass(frozen=True, slots=True)
class ModelFrameResult:
    model_id: str
    frame_id: str
    probability_map: np.ndarray | None
    binary_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ComparisonProfile:
    profile_id: str
    metric_groups: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True, slots=True)
class PairwiseComparisonRequest:
    frame_id: str
    model_a: ModelFrameResult
    model_b: ModelFrameResult
    profile: ComparisonProfile | str = "auto"
    threshold: float = 0.5
    connectivity: int = 8
    pruning_min_length_px: int = 5
    evidence_provider_version: str = "none"
    compute_level: str = "deep"


@dataclass(frozen=True, slots=True)
class EnsembleComparisonRequest:
    frame_id: str
    models: tuple[ModelFrameResult, ...]
    profile: ComparisonProfile | str = "auto"
    threshold: float = 0.5
    consensus_threshold: float = 0.5
    connectivity: int = 8
    pruning_min_length_px: int = 5
    evidence_provider_version: str = "none"
    compute_level: str = "deep"


@dataclass(frozen=True, slots=True)
class MetricValue:
    name: str
    value: float | int | None
    group: str
    description: str
    unit: str | None = None
    valid: bool = True


@dataclass(frozen=True, slots=True)
class RasterLayer:
    layer_id: str
    title: str
    image: np.ndarray
    opacity: float
    visible: bool
    layer_group: str


@dataclass(frozen=True, slots=True)
class VectorLayer:
    layer_id: str
    title: str
    primitives: list[Any]
    opacity: float
    visible: bool
    layer_group: str


@dataclass(frozen=True, slots=True)
class ComparisonEvent:
    event_id: str
    event_type: str
    risk: float
    bbox: tuple[int, int, int, int]
    point: tuple[float, float] | None
    object_ids: list[str]
    model_ids: list[str]
    description: str
    recommended_layers: list[str]
    metrics: dict[str, float | int | str]


@dataclass(frozen=True, slots=True)
class ComparisonCacheKey:
    frame_id: str
    model_ids: tuple[str, ...]
    comparison_mode: ComparisonMode
    profile: str
    threshold: float
    consensus_threshold: float | None
    connectivity: int
    pruning_threshold: int
    evidence_provider_version: str
    algorithm_version: str


@dataclass(frozen=True, slots=True)
class FrameComparisonResult:
    frame_id: str
    mode: ComparisonMode
    model_ids: tuple[str, ...]
    profile: str
    metrics: tuple[MetricValue, ...]
    raster_layers: tuple[RasterLayer, ...]
    vector_layers: tuple[VectorLayer, ...]
    events: tuple[ComparisonEvent, ...]
    risk: dict[str, float]
    cache_key: ComparisonCacheKey
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PairwiseComparisonResult:
    frame: FrameComparisonResult


@dataclass(frozen=True, slots=True)
class EnsembleComparisonResult:
    frame: FrameComparisonResult
