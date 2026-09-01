"""Ensemble comparison core."""
from __future__ import annotations

import itertools
from time import perf_counter

import numpy as np

from karakal.core.profiling import profile_stage

from .artifacts import PerModelArtifactCache
from .cache import build_cache_key
from .cache import ComparisonResultCache
from .layers import ensemble_raster_layers
from .models import EnsembleComparisonRequest, EnsembleComparisonResult, FrameComparisonResult, MetricValue, PairwiseComparisonRequest
from .pairwise import compare_pairwise
from .profiles import resolve_profile

EPS = 1e-6


def compare_ensemble(
    request: EnsembleComparisonRequest,
    *,
    artifact_cache: PerModelArtifactCache | None = None,
    result_cache: ComparisonResultCache | None = None,
) -> EnsembleComparisonResult:
    if len(request.models) <= 0:
        raise ValueError("Ensemble comparison requires at least one model")
    profile = resolve_profile(request.profile, request.models[0].metadata if request.models else {})
    cache_key = build_cache_key(
        frame_id=request.frame_id,
        model_ids=tuple(model.model_id for model in request.models),
        comparison_mode="ensemble",
        profile=profile.profile_id,
        threshold=request.threshold,
        consensus_threshold=request.consensus_threshold,
        connectivity=request.connectivity,
        pruning_threshold=request.pruning_min_length_px,
        evidence_provider_version=request.evidence_provider_version,
    )
    if result_cache is not None:
        cached = result_cache.get(cache_key)
        if cached is not None:
            return EnsembleComparisonResult(cached)

    timings: dict[str, float] = {}
    started = perf_counter()
    active_artifact_cache = artifact_cache or PerModelArtifactCache(max_items=max(16, len(request.models) * 2))
    with profile_stage("validation.ensemble.prepare_artifacts", frame_id=request.frame_id):
        artifacts = [
            active_artifact_cache.get_or_build(
                frame_id=request.frame_id,
                model_result=model,
                threshold=request.threshold,
                connectivity=request.connectivity,
                pruning_threshold=request.pruning_min_length_px,
            )
            for model in request.models
        ]
    timings["load_time"] = 1000.0 * (perf_counter() - started)
    for artifact in artifacts:
        for key, value in artifact.stage_timings_ms.items():
            timings[str(key)] = timings.get(str(key), 0.0) + float(value)

    masks = [artifact.binary_mask for artifact in artifacts]
    first_shape = masks[0].shape
    if any(mask.shape != first_shape for mask in masks):
        raise ValueError("All ensemble masks must have the same shape")
    started = perf_counter()
    with profile_stage("validation.ensemble.vote", frame_id=request.frame_id):
        stack = np.stack(masks, axis=0).astype(np.float32)
        vote_map = np.mean(stack, axis=0, dtype=np.float32)
        consensus_mask = vote_map >= float(request.consensus_threshold)
        uncertainty = np.asarray(1.0 - np.abs(vote_map - 0.5) * 2.0, dtype=np.float32)
        entropy = _binary_vote_entropy(vote_map)
    timings["ensemble_vote_time"] = 1000.0 * (perf_counter() - started)
    metrics: list[MetricValue] = [
        MetricValue("model_count", len(request.models), "ensemble", "Number of models in the ensemble."),
        MetricValue("consensus_area", int(np.count_nonzero(consensus_mask)), "ensemble", "Foreground area in the consensus mask.", "px"),
        MetricValue("mean_vote_fraction", float(np.mean(vote_map, dtype=np.float64)), "ensemble", "Mean foreground vote fraction."),
        MetricValue("mean_vote_entropy", float(np.mean(entropy, dtype=np.float64)), "ensemble", "Mean predictive entropy over votes."),
        MetricValue("mean_uncertainty", float(np.mean(uncertainty, dtype=np.float64)), "ensemble", "Mean disagreement uncertainty."),
    ]

    pairwise_started = perf_counter()
    pairwise_scores: list[tuple[tuple[str, str], float]] = []
    with profile_stage("validation.ensemble.pair_matrix", frame_id=request.frame_id):
        for model_a, model_b in itertools.combinations(request.models, 2):
            pair_result = compare_pairwise(
                PairwiseComparisonRequest(
                    frame_id=request.frame_id,
                    model_a=model_a,
                    model_b=model_b,
                    profile=profile,
                    threshold=request.threshold,
                    connectivity=request.connectivity,
                    pruning_min_length_px=request.pruning_min_length_px,
                    evidence_provider_version=request.evidence_provider_version,
                    compute_level="fast",
                ),
                artifact_cache=active_artifact_cache,
            ).frame
            metric_map = {metric.name: metric.value for metric in pair_result.metrics}
            pairwise_scores.append(((model_a.model_id, model_b.model_id), float(metric_map.get("foreground_disagreement_rate") or 0.0)))
    timings["pairwise_matrix_time"] = 1000.0 * (perf_counter() - pairwise_started)
    if pairwise_scores:
        metrics.append(MetricValue("pairwise_mean_foreground_disagreement", float(np.mean([score for _pair, score in pairwise_scores], dtype=np.float64)), "ensemble", "Mean pairwise foreground disagreement."))

    outlier_scores = _outlier_scores(stack, consensus_mask, tuple(model.model_id for model in request.models))
    for model_id, score in outlier_scores.items():
        metrics.append(MetricValue(f"outlier_score::{model_id}", float(score), "ensemble", "Model distance from consensus."))
    outlier_model_id = max(outlier_scores, key=outlier_scores.get) if outlier_scores else ""

    risk = {
        "ensemble_uncertainty": float(np.clip(np.mean(uncertainty, dtype=np.float64), 0.0, 1.0)),
        "pairwise": float(np.clip(np.mean([score for _pair, score in pairwise_scores], dtype=np.float64), 0.0, 1.0)) if pairwise_scores else 0.0,
        "outlier": float(np.clip(max(outlier_scores.values()), 0.0, 1.0)) if outlier_scores else 0.0,
    }
    risk["total"] = float(np.mean(list(risk.values()), dtype=np.float64))
    result = EnsembleComparisonResult(
        FrameComparisonResult(
            frame_id=request.frame_id,
            mode="ensemble",
            model_ids=tuple(model.model_id for model in request.models),
            profile=profile.profile_id,
            metrics=tuple(metrics),
            raster_layers=ensemble_raster_layers(vote_map, consensus_mask, uncertainty),
            vector_layers=(),
            events=(),
            risk=risk,
            cache_key=cache_key,
            metadata={"outlier_model_id": outlier_model_id, "pairwise_matrix": pairwise_scores, "metric_groups": profile.metric_groups, "stage_timings_ms": timings, "compute_level": str(getattr(request, "compute_level", "deep") or "deep")},
        )
    )
    if result_cache is not None:
        started = perf_counter()
        result_cache.put(cache_key, result.frame)
        result.frame.metadata.setdefault("stage_timings_ms", {})["cache_write_time"] = 1000.0 * (perf_counter() - started)
    return result


def _binary_vote_entropy(vote_map: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(vote_map, dtype=np.float32), EPS, 1.0 - EPS)
    return np.asarray(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)), dtype=np.float32)


def _outlier_scores(stack: np.ndarray, consensus_mask: np.ndarray, model_ids: tuple[str, ...]) -> dict[str, float]:
    consensus = np.asarray(consensus_mask, dtype=bool)
    result: dict[str, float] = {}
    for index in range(stack.shape[0]):
        model_mask = np.asarray(stack[index] >= 0.5, dtype=bool)
        xor = np.logical_xor(model_mask, consensus)
        union = np.logical_or(model_mask, consensus)
        model_id = str(model_ids[index]) if index < len(model_ids) else str(index)
        result[model_id] = 0.0 if not np.any(union) else float(np.count_nonzero(xor) / max(1, np.count_nonzero(union)))
    return result
