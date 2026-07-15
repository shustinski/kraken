"""Small LRU cache for comparison results."""
from __future__ import annotations

from collections import OrderedDict

from .models import ComparisonCacheKey, FrameComparisonResult

ALGORITHM_VERSION = "karakal-comparison-v1"


class ComparisonResultCache:
    def __init__(self, max_items: int = 256) -> None:
        self.max_items = max(1, int(max_items))
        self._items: OrderedDict[ComparisonCacheKey, FrameComparisonResult] = OrderedDict()

    def get(self, key: ComparisonCacheKey) -> FrameComparisonResult | None:
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key: ComparisonCacheKey, value: FrameComparisonResult) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

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
