"""Inter-model comparison package for Karakal."""
from __future__ import annotations

from .ensemble import compare_ensemble
from .models import (
    ComparisonCacheKey,
    ComparisonEvent,
    ComparisonProfile,
    EnsembleComparisonRequest,
    EnsembleComparisonResult,
    FrameComparisonResult,
    MetricValue,
    ModelFrameResult,
    PairwiseComparisonRequest,
    PairwiseComparisonResult,
    RasterLayer,
    VectorLayer,
)
from .pairwise import compare_pairwise
from .profiles import resolve_profile

__all__ = [
    "ComparisonCacheKey",
    "ComparisonEvent",
    "ComparisonProfile",
    "EnsembleComparisonRequest",
    "EnsembleComparisonResult",
    "FrameComparisonResult",
    "MetricValue",
    "ModelFrameResult",
    "PairwiseComparisonRequest",
    "PairwiseComparisonResult",
    "RasterLayer",
    "VectorLayer",
    "compare_ensemble",
    "compare_pairwise",
    "resolve_profile",
]
