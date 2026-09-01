"""Translate Karakal runtime results to the shared analysis-result contract."""
from __future__ import annotations

import math

import numpy as np

from kraken_core.analysis_protocol import (
    AnalysisFrameResult,
    AnalysisMetricValue,
    AnalysisOutcome,
    AnalysisProfileKind,
    AnalysisResultManifest,
    AnalysisScaleDefinition,
    AnalysisScaleMode,
)

from ..core.analysis_modes import metric_visual_ratio
from ..core.domain import BuildResult, FrameRecord
from ..core.metric_keys import metric_higher_is_better


def _frame_position(record: FrameRecord, index: int) -> tuple[int, int]:
    identity = record.identity
    if identity is not None and identity.tile_x is not None and identity.tile_y is not None:
        return int(identity.tile_x) + 1, int(identity.tile_y) + 1
    return index + 1, 1


def _goodness(record: FrameRecord, metric_key: str, build_result: BuildResult) -> float | None:
    if not bool(record.score_ready) or record.absolute_score is None:
        return None
    ratio = metric_visual_ratio(
        metric_key,
        float(record.absolute_score),
        point_match_radius=float(getattr(build_result.options, "point_match_radius", 3.0)),
        bce_score_cap=1.0,
    )
    if ratio is None:
        return max(0.0, min(1.0, float(record.score)))
    return max(0.0, min(1.0, float(ratio) if metric_higher_is_better(metric_key) else 1.0 - float(ratio)))


def _scale(metric_key: str, values: list[float], mode: AnalysisScaleMode) -> AnalysisScaleDefinition:
    if not values:
        return AnalysisScaleDefinition(metric_key=metric_key, mode=mode, low=0.0, high=1.0)
    p05, p50, p95 = (float(value) for value in np.percentile(values, (5, 50, 95)))
    if mode == AnalysisScaleMode.ABSOLUTE:
        low, high = 0.0, 1.0
    else:
        low, high = p05, p95
        if high - low < 0.01:
            midpoint = (high + low) / 2.0
            low = max(0.0, midpoint - 0.005)
            high = min(1.0, midpoint + 0.005)
            if high <= low:
                low, high = 0.0, 1.0
    return AnalysisScaleDefinition(
        metric_key=metric_key,
        mode=mode,
        low=low,
        high=high,
        p05=p05,
        p50=p50,
        p95=p95,
        clipped_low=sum(1 for value in values if value < low),
        clipped_high=sum(1 for value in values if value > high),
    )


def build_analysis_result_manifest(
    *,
    job_id: str,
    project_id: str,
    profile: AnalysisProfileKind,
    build_result: BuildResult,
    metric_key: str,
    scale_mode: AnalysisScaleMode,
    outcome: AnalysisOutcome = AnalysisOutcome.SUCCEEDED,
    message: str = "",
) -> AnalysisResultManifest:
    """Create a project-safe result containing metrics and coordinates, not storage paths."""

    frame_results: list[AnalysisFrameResult] = []
    goodness_values: list[float] = []
    for index, record in enumerate(build_result.records):
        x, y = _frame_position(record, index)
        goodness = _goodness(record, metric_key, build_result)
        metrics: tuple[AnalysisMetricValue, ...] = ()
        if goodness is not None and record.absolute_score is not None and math.isfinite(float(record.absolute_score)):
            goodness_values.append(goodness)
            metrics = (
                AnalysisMetricValue(
                    key=metric_key,
                    raw_value=float(record.absolute_score),
                    goodness=goodness,
                    percentile=None if record.score_percentile is None else float(record.score_percentile),
                    higher_is_better=metric_higher_is_better(metric_key),
                ),
            )
        frame_results.append(
            AnalysisFrameResult(
                frame_id=str(record.key),
                x=x,
                y=y,
                status="ready" if metrics else "not_computed",
                metrics=metrics,
            )
        )
    return AnalysisResultManifest(
        job_id=job_id,
        project_id=project_id,
        profile=profile,
        outcome=outcome,
        frames=tuple(frame_results),
        scales=(_scale(metric_key, goodness_values, scale_mode),),
        message=message,
    )


__all__ = ["build_analysis_result_manifest"]
