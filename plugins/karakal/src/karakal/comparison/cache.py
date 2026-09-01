"""Small LRU cache for comparison results."""
from __future__ import annotations

from .models import ComparisonCacheKey, FrameComparisonResult
from ..core.cache_utils import ByteLruCache
from ..core.performance import load_performance_config

ALGORITHM_VERSION = "karakal-comparison-v1"


class ComparisonResultCache:
    def __init__(self, max_items: int = 256, *, max_bytes: int | None = None) -> None:
        self.max_items = max(1, int(max_items))
        limit = max_bytes
        if limit is None:
            limit = int(load_performance_config().ram_cache_limit_mb) * 1024 * 1024
        self._items: ByteLruCache[ComparisonCacheKey, FrameComparisonResult] = ByteLruCache(
            limit,
            max_items=self.max_items,
        )

    def get(self, key: ComparisonCacheKey) -> FrameComparisonResult | None:
        return self._items.get(key)

    def put(self, key: ComparisonCacheKey, value: FrameComparisonResult) -> None:
        self._items.put(key, value)

    def clear(self) -> None:
        self._items.clear()


def build_cache_key(
    *,
    frame_id: str,
    model_ids: tuple[str, ...],
    comparison_mode: str,
    profile: str,
    threshold: float,
    consensus_threshold: float | None,
    connectivity: int,
    pruning_threshold: int,
    evidence_provider_version: str,
) -> ComparisonCacheKey:
    return ComparisonCacheKey(
        frame_id=str(frame_id),
        model_ids=tuple(str(model_id) for model_id in model_ids),
        comparison_mode="ensemble" if str(comparison_mode) == "ensemble" else "pairwise",
        profile=str(profile),
        threshold=round(float(threshold), 6),
        consensus_threshold=None if consensus_threshold is None else round(float(consensus_threshold), 6),
        connectivity=int(connectivity),
        pruning_threshold=int(pruning_threshold),
        evidence_provider_version=str(evidence_provider_version or "none"),
        algorithm_version=ALGORITHM_VERSION,
    )
