"""Per-model prepared artifacts and dependency-aware cache keys."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .components import Component, label_components
from .geometry import boundary_mask
from .models import ModelFrameResult
from .skeleton import SkeletonStats, skeleton_stats
from .topology import topology_metrics

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

NORMALIZATION_VERSION = "normalize-v1"
SKELETON_ALGORITHM_VERSION = "skeleton-v1"
TOPOLOGY_ALGORITHM_VERSION = "topology-v1"


@dataclass(frozen=True, slots=True)
class ArtifactCacheKey:
    frame_id: str
    model_id: str
    threshold: float
    connectivity: int
    pruning_threshold: int
    normalization_version: str = NORMALIZATION_VERSION
    skeleton_algorithm_version: str = SKELETON_ALGORITHM_VERSION
    topology_algorithm_version: str = TOPOLOGY_ALGORITHM_VERSION


@dataclass(slots=True)
class ModelArtifacts:
    model_id: str
    frame_id: str
    binary_mask: np.ndarray
    probability_map: np.ndarray | None
    threshold: float
    connectivity: int
    pruning_threshold: int
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    labels: np.ndarray | None = None
    components: tuple[Component, ...] | None = None
    boundary: np.ndarray | None = None
    boundary_distance_transform: np.ndarray | None = None
    skeleton: np.ndarray | None = None
    skeleton_distance_transform: np.ndarray | None = None
    skeleton_stats: SkeletonStats | None = None
    topology: dict[str, float] | None = None

    def ensure_components(self) -> tuple[np.ndarray, tuple[Component, ...]]:
        if self.labels is None or self.components is None:
            started = perf_counter()
            self.labels, self.components = label_components(self.binary_mask, connectivity=self.connectivity)
            self.stage_timings_ms["component_labeling_time"] = self.stage_timings_ms.get("component_labeling_time", 0.0) + _elapsed_ms(started)
        return self.labels, self.components

    def ensure_boundary(self) -> np.ndarray:
        if self.boundary is None:
            started = perf_counter()
            self.boundary = boundary_mask(self.binary_mask)
            self.stage_timings_ms["boundary_time"] = self.stage_timings_ms.get("boundary_time", 0.0) + _elapsed_ms(started)
        return self.boundary

    def ensure_boundary_distance_transform(self) -> np.ndarray:
        if self.boundary_distance_transform is None:
            started = perf_counter()
            boundary = self.ensure_boundary()
            self.boundary_distance_transform = _distance_transform(~boundary)
            self.stage_timings_ms["distance_transform_time"] = self.stage_timings_ms.get("distance_transform_time", 0.0) + _elapsed_ms(started)
        return self.boundary_distance_transform

    def ensure_skeleton(self) -> np.ndarray:
        if self.skeleton is None or self.skeleton_stats is None:
            started = perf_counter()
            stats = skeleton_stats(self.binary_mask)
            self.skeleton_stats = stats
            self.skeleton = stats.skeleton
            self.stage_timings_ms["skeletonization_time"] = self.stage_timings_ms.get("skeletonization_time", 0.0) + _elapsed_ms(started)
            self.stage_timings_ms["branch_extraction_time"] = self.stage_timings_ms.get("branch_extraction_time", 0.0)
        return self.skeleton

    def ensure_skeleton_distance_transform(self) -> np.ndarray:
        if self.skeleton_distance_transform is None:
            started = perf_counter()
            skeleton = self.ensure_skeleton()
            self.skeleton_distance_transform = _distance_transform(~skeleton)
            self.stage_timings_ms["skeleton_distance_transform_time"] = self.stage_timings_ms.get("skeleton_distance_transform_time", 0.0) + _elapsed_ms(started)
        return self.skeleton_distance_transform

    def ensure_topology(self) -> dict[str, float]:
        if self.topology is None:
            started = perf_counter()
            self.topology = topology_metrics(self.binary_mask, connectivity=self.connectivity)
            self.stage_timings_ms["topology_time"] = self.stage_timings_ms.get("topology_time", 0.0) + _elapsed_ms(started)
        return self.topology


class PerModelArtifactCache:
    def __init__(self, max_items: int = 256) -> None:
        self.max_items = max(1, int(max_items))
        self._items: dict[ArtifactCacheKey, ModelArtifacts] = {}
        self.hits = 0
        self.misses = 0

    def get_or_build(
        self,
        *,
        frame_id: str,
        model_result: ModelFrameResult,
        threshold: float,
        connectivity: int,
        pruning_threshold: int,
    ) -> ModelArtifacts:
        key = ArtifactCacheKey(
            frame_id=str(frame_id),
            model_id=str(model_result.model_id),
            threshold=round(float(threshold), 6),
            connectivity=int(connectivity),
            pruning_threshold=int(pruning_threshold),
        )
        cached = self._items.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        artifact = prepare_model_artifacts(
            frame_id=frame_id,
            model_result=model_result,
            threshold=threshold,
            connectivity=connectivity,
            pruning_threshold=pruning_threshold,
        )
        if len(self._items) >= self.max_items:
            first_key = next(iter(self._items))
            self._items.pop(first_key, None)
        self._items[key] = artifact
        return artifact

    def clear(self) -> None:
        self._items.clear()
        self.hits = 0
        self.misses = 0


def prepare_model_artifacts(
    *,
    frame_id: str,
    model_result: ModelFrameResult,
    threshold: float,
    connectivity: int,
    pruning_threshold: int,
) -> ModelArtifacts:
    timings: dict[str, float] = {}
    started = perf_counter()
    mask = np.nan_to_num(np.asarray(model_result.binary_mask, dtype=bool), nan=False).astype(bool, copy=False)
    timings["binary_mask_time"] = _elapsed_ms(started)
    probability = None
    if model_result.probability_map is not None:
        started = perf_counter()
        probability = np.nan_to_num(np.asarray(model_result.probability_map, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        if probability.shape != mask.shape:
            raise ValueError(f"Probability map shape {probability.shape} does not match mask shape {mask.shape}")
        if probability.size > 0 and float(np.max(probability)) > 1.0:
            probability = probability / 255.0
        probability = np.clip(probability, 0.0, 1.0).astype(np.float32, copy=False)
        timings["normalization_time"] = _elapsed_ms(started)
    else:
        timings["normalization_time"] = 0.0
    return ModelArtifacts(
        model_id=str(model_result.model_id),
        frame_id=str(frame_id),
        binary_mask=mask,
        probability_map=probability,
        threshold=float(threshold),
        connectivity=int(connectivity),
        pruning_threshold=int(pruning_threshold),
        stage_timings_ms=timings,
    )


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if ndi is None or not hasattr(ndi, "distance_transform_edt"):
        return np.where(binary, 1.0, 0.0).astype(np.float32)
    return np.asarray(ndi.distance_transform_edt(binary), dtype=np.float32)


def _elapsed_ms(started: float) -> float:
    return 1000.0 * (perf_counter() - started)
