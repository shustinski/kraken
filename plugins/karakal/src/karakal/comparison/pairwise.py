"""Pairwise inter-model comparison core."""
from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from .artifacts import ModelArtifacts, PerModelArtifactCache, prepare_model_artifacts
from .cache import build_cache_key
from .cache import ComparisonResultCache
from .events import component_events_from_labels, xor_hotspot_events
from .layers import pairwise_raster_layers
from .models import (
    FrameComparisonResult,
    MetricValue,
    PairwiseComparisonRequest,
    PairwiseComparisonResult,
)
from .profiles import COMPONENTS_GROUP, GEOMETRY_GROUP, SKELETON_GROUP, SOFT_GROUP, TOPOLOGY_GROUP, resolve_profile
from .topology import topology_delta_events

EPS = 1e-8


def compare_pairwise(
    request: PairwiseComparisonRequest,
    *,
    artifact_cache: PerModelArtifactCache | None = None,
    result_cache: ComparisonResultCache | None = None,
) -> PairwiseComparisonResult:
    metadata = {**request.model_a.metadata, **request.model_b.metadata}
    profile = resolve_profile(request.profile, metadata)
    cache_key = build_cache_key(
        frame_id=request.frame_id,
        model_ids=(request.model_a.model_id, request.model_b.model_id),
        comparison_mode="pairwise",
        profile=profile.profile_id,
        threshold=request.threshold,
        consensus_threshold=None,
        connectivity=request.connectivity,
        pruning_threshold=request.pruning_min_length_px,
        evidence_provider_version=request.evidence_provider_version,
    )
    if result_cache is not None:
        cached = result_cache.get(cache_key)
        if cached is not None:
            cached.metadata.setdefault("stage_timings_ms", {})["cache_read_time"] = 0.0
            return PairwiseComparisonResult(cached)

    timings: dict[str, float] = {}
    started = perf_counter()
    if artifact_cache is None:
        artifact_a = prepare_model_artifacts(
            frame_id=request.frame_id,
            model_result=request.model_a,
            threshold=request.threshold,
            connectivity=request.connectivity,
            pruning_threshold=request.pruning_min_length_px,
        )
        artifact_b = prepare_model_artifacts(
            frame_id=request.frame_id,
            model_result=request.model_b,
            threshold=request.threshold,
            connectivity=request.connectivity,
            pruning_threshold=request.pruning_min_length_px,
        )
    else:
        artifact_a = artifact_cache.get_or_build(
            frame_id=request.frame_id,
            model_result=request.model_a,
            threshold=request.threshold,
            connectivity=request.connectivity,
            pruning_threshold=request.pruning_min_length_px,
        )
        artifact_b = artifact_cache.get_or_build(
            frame_id=request.frame_id,
            model_result=request.model_b,
            threshold=request.threshold,
            connectivity=request.connectivity,
            pruning_threshold=request.pruning_min_length_px,
        )
    timings["load_time"] = _elapsed_ms(started)
    _merge_timings(timings, artifact_a.stage_timings_ms)
    _merge_timings(timings, artifact_b.stage_timings_ms)

    mask_a = artifact_a.binary_mask
    mask_b = artifact_b.binary_mask
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Model masks must have the same shape for frame {request.frame_id}: {mask_a.shape} != {mask_b.shape}")
    prob_a = artifact_a.probability_map
    prob_b = artifact_b.probability_map

    started = perf_counter()
    common = mask_a & mask_b
    a_only = mask_a & ~mask_b
    b_only = mask_b & ~mask_a
    xor = mask_a ^ mask_b
    union = mask_a | mask_b
    area_a = int(np.count_nonzero(mask_a))
    area_b = int(np.count_nonzero(mask_b))
    common_area = int(np.count_nonzero(common))
    xor_area = int(np.count_nonzero(xor))
    union_area = int(np.count_nonzero(union))
    pixel_count = int(mask_a.size)
    timings["fast_pixel_time"] = _elapsed_ms(started)

    metrics: list[MetricValue] = [
        _metric("dice_ab", 1.0 if area_a + area_b <= 0 else (2.0 * common_area) / max(1, area_a + area_b), "pixel", "Dice overlap between A and B."),
        _metric("iou_ab", 1.0 if union_area <= 0 else common_area / max(1, union_area), "pixel", "IoU overlap between A and B."),
        _metric("disagreement_rate", xor_area / max(1, pixel_count), "pixel", "XOR pixels divided by image pixels."),
        _metric("foreground_disagreement_rate", xor_area / (union_area + EPS), "pixel", "XOR pixels divided by foreground union."),
        _metric("area_a", area_a, "pixel", "Foreground area in model A.", "px"),
        _metric("area_b", area_b, "pixel", "Foreground area in model B.", "px"),
        _metric("relative_area_delta", abs(area_a - area_b) / max(area_a, area_b, 1), "pixel", "Relative area delta between A and B."),
        _metric("a_only_area", int(np.count_nonzero(a_only)), "pixel", "Pixels present only in A.", "px"),
        _metric("b_only_area", int(np.count_nonzero(b_only)), "pixel", "Pixels present only in B.", "px"),
    ]

    compute_level = str(getattr(request, "compute_level", "deep") or "deep").lower()
    run_standard = compute_level in {"standard", "deep"}
    run_deep = compute_level == "deep"

    if SOFT_GROUP in profile.metric_groups and run_standard:
        started = perf_counter()
        metrics.extend(_soft_metrics(prob_a, prob_b, threshold=request.threshold))
        timings["soft_metrics_time"] = _elapsed_ms(started)
    if GEOMETRY_GROUP in profile.metric_groups and run_standard:
        started = perf_counter()
        metrics.extend(_geometry_metrics_prepared(artifact_a, artifact_b))
        timings["boundary_comparison_time"] = _elapsed_ms(started)
    if COMPONENTS_GROUP in profile.metric_groups and run_standard:
        started = perf_counter()
        metrics.extend(_component_metrics_prepared(artifact_a, artifact_b))
        timings["component_matching_time"] = _elapsed_ms(started)
    if SKELETON_GROUP in profile.metric_groups and run_standard:
        started = perf_counter()
        metrics.extend(_skeleton_metrics_prepared(artifact_a, artifact_b))
        timings["skeleton_comparison_time"] = _elapsed_ms(started)
    if TOPOLOGY_GROUP in profile.metric_groups and run_deep:
        started = perf_counter()
        metrics.extend(_topology_metrics_prepared(artifact_a, artifact_b))
        timings["topology_time"] = timings.get("topology_time", 0.0) + _elapsed_ms(started)

    started = perf_counter()
    prepared_boundary_a = artifact_a.boundary if run_standard else None
    prepared_boundary_b = artifact_b.boundary if run_standard else None
    prepared_skeleton_a = artifact_a.skeleton if run_standard else None
    prepared_skeleton_b = artifact_b.skeleton if run_standard else None
    layers = pairwise_raster_layers(
        mask_a=mask_a,
        mask_b=mask_b,
        probability_a=prob_a,
        probability_b=prob_b,
        threshold=request.threshold,
        include_standard_layers=run_standard,
        include_skeleton_layers=SKELETON_GROUP in profile.metric_groups,
        boundary_a=prepared_boundary_a,
        boundary_b=prepared_boundary_b,
        skeleton_a=prepared_skeleton_a,
        skeleton_b=prepared_skeleton_b,
    )
    timings["raster_layer_generation_time"] = _elapsed_ms(started)
    started = perf_counter()
    events = [
        *xor_hotspot_events(xor, frame_id=request.frame_id, model_ids=(request.model_a.model_id, request.model_b.model_id), connectivity=request.connectivity),
    ]
    if COMPONENTS_GROUP in profile.metric_groups and run_standard:
        labels_a, comps_a = artifact_a.ensure_components()
        labels_b, comps_b = artifact_b.ensure_components()
        events.extend(component_events_from_labels(labels_a, comps_a, labels_b, comps_b, frame_id=request.frame_id, model_ids=(request.model_a.model_id, request.model_b.model_id)))
    if TOPOLOGY_GROUP in profile.metric_groups and run_deep:
        events.extend(topology_delta_events(mask_a, mask_b, frame_id=request.frame_id, model_ids=(request.model_a.model_id, request.model_b.model_id), connectivity=request.connectivity))
    timings["event_detection_time"] = _elapsed_ms(started)

    risk = _risk_from_metrics(metrics)
    result = PairwiseComparisonResult(
        FrameComparisonResult(
            frame_id=request.frame_id,
            mode="pairwise",
            model_ids=(request.model_a.model_id, request.model_b.model_id),
            profile=profile.profile_id,
            metrics=tuple(metrics),
            raster_layers=layers,
            vector_layers=(),
            events=tuple(sorted(events, key=lambda item: item.risk, reverse=True)),
            risk=risk,
            cache_key=cache_key,
            metadata={"metric_groups": profile.metric_groups, "stage_timings_ms": timings, "compute_level": compute_level},
        )
    )
    if result_cache is not None:
        started = perf_counter()
        result_cache.put(cache_key, result.frame)
        result.frame.metadata.setdefault("stage_timings_ms", {})["cache_write_time"] = _elapsed_ms(started)
    return result


def _metric(name: str, value: float | int | None, group: str, description: str, unit: str | None = None, valid: bool = True) -> MetricValue:
    clean_value: float | int | None
    if value is None:
        clean_value = None
    elif isinstance(value, int):
        clean_value = int(value)
    else:
        clean = float(value)
        clean_value = None if not math.isfinite(clean) else clean
        valid = bool(valid and clean_value is not None)
    return MetricValue(name=name, value=clean_value, group=group, description=description, unit=unit, valid=valid)


def _soft_metrics(prob_a: np.ndarray | None, prob_b: np.ndarray | None, *, threshold: float) -> list[MetricValue]:
    if prob_a is None or prob_b is None:
        return [
            _metric("soft_mae_ab", None, "soft_confidence", "Mean absolute probability difference.", valid=False),
            _metric("soft_rmse_ab", None, "soft_confidence", "RMSE probability difference.", valid=False),
            _metric("soft_p95_ab", None, "soft_confidence", "P95 probability difference.", valid=False),
            _metric("soft_max_ab", None, "soft_confidence", "Max probability difference.", valid=False),
            _metric("signed_mean_delta", None, "soft_confidence", "Mean signed probability delta.", valid=False),
            _metric("threshold_crossing_rate", None, "soft_confidence", "Rate of threshold crossing disagreements.", valid=False),
        ]
    diff = np.abs(prob_a - prob_b)
    signed = prob_a - prob_b
    crossing = np.logical_xor(prob_a >= float(threshold), prob_b >= float(threshold))
    return [
        _metric("soft_mae_ab", float(np.mean(diff, dtype=np.float64)), "soft_confidence", "Mean absolute probability difference."),
        _metric("soft_rmse_ab", float(np.sqrt(np.mean(diff * diff, dtype=np.float64))), "soft_confidence", "RMSE probability difference."),
        _metric("soft_p95_ab", float(np.percentile(diff, 95.0)), "soft_confidence", "P95 probability difference."),
        _metric("soft_max_ab", float(np.max(diff)) if diff.size else 0.0, "soft_confidence", "Max probability difference."),
        _metric("signed_mean_delta", float(np.mean(signed, dtype=np.float64)), "soft_confidence", "Mean signed probability delta."),
        _metric("threshold_crossing_rate", float(np.mean(crossing, dtype=np.float64)), "soft_confidence", "Rate of threshold crossing disagreements."),
    ]


def _component_metrics_prepared(artifact_a: ModelArtifacts, artifact_b: ModelArtifacts) -> list[MetricValue]:
    _labels_a, comps_a = artifact_a.ensure_components()
    _labels_b, comps_b = artifact_b.ensure_components()
    return [
        _metric("component_count_a", len(comps_a), "components", "Connected components in A."),
        _metric("component_count_b", len(comps_b), "components", "Connected components in B."),
        _metric("component_count_delta", abs(len(comps_a) - len(comps_b)), "components", "Absolute component-count delta."),
    ]


def _skeleton_metrics_prepared(artifact_a: ModelArtifacts, artifact_b: ModelArtifacts) -> list[MetricValue]:
    stats_a = artifact_a.skeleton_stats
    stats_b = artifact_b.skeleton_stats
    if stats_a is None:
        artifact_a.ensure_skeleton()
        stats_a = artifact_a.skeleton_stats
    if stats_b is None:
        artifact_b.ensure_skeleton()
        stats_b = artifact_b.skeleton_stats
    assert stats_a is not None
    assert stats_b is not None
    skel_a = artifact_a.ensure_skeleton()
    skel_b = artifact_b.ensure_skeleton()
    total = int(np.count_nonzero(skel_a)) + int(np.count_nonzero(skel_b))
    exact = 1.0 if total <= 0 else float((2.0 * np.count_nonzero(skel_a & skel_b)) / total)
    dt_a = artifact_a.ensure_skeleton_distance_transform()
    dt_b = artifact_b.ensure_skeleton_distance_transform()
    precision = float(np.count_nonzero(skel_b & (dt_a <= 1.0)) / max(1, np.count_nonzero(skel_b)))
    recall = float(np.count_nonzero(skel_a & (dt_b <= 1.0)) / max(1, np.count_nonzero(skel_a)))
    f1 = 0.0 if precision + recall <= 0.0 else float((2.0 * precision * recall) / (precision + recall))
    mask_a = artifact_a.binary_mask
    mask_b = artifact_b.binary_mask
    tprec = float(np.count_nonzero(skel_a & mask_b) / max(1, np.count_nonzero(skel_a)))
    tsens = float(np.count_nonzero(skel_b & mask_a) / max(1, np.count_nonzero(skel_b)))
    cl = 1.0 if (not np.any(mask_a) and not np.any(mask_b)) else (0.0 if tprec + tsens <= 0.0 else float((2.0 * tprec * tsens) / (tprec + tsens)))
    return [
        _metric("skeleton_dice_ab", exact, "skeleton", "Exact skeleton Dice."),
        _metric("skeleton_f1_r1_ab", f1, "skeleton", "Skeleton F1 within 1 px."),
        _metric("cldice_ab", cl, "skeleton", "Topology-aware clDice."),
        _metric("endpoint_count_a", stats_a.endpoint_count, "skeleton", "Skeleton endpoints in A."),
        _metric("endpoint_count_b", stats_b.endpoint_count, "skeleton", "Skeleton endpoints in B."),
        _metric("junction_count_a", stats_a.junction_count, "skeleton", "Skeleton junctions in A."),
        _metric("junction_count_b", stats_b.junction_count, "skeleton", "Skeleton junctions in B."),
        _metric("skeleton_length_delta", abs(stats_a.length - stats_b.length), "skeleton", "Absolute skeleton length delta.", "px"),
    ]


def _geometry_metrics_prepared(artifact_a: ModelArtifacts, artifact_b: ModelArtifacts) -> list[MetricValue]:
    boundary_a = artifact_a.ensure_boundary()
    boundary_b = artifact_b.ensure_boundary()
    intersection = int(np.count_nonzero(boundary_a & boundary_b))
    union = int(np.count_nonzero(boundary_a | boundary_b))
    boundary_iou = 1.0 if union <= 0 else float(intersection / union)
    if not np.any(boundary_a) and not np.any(boundary_b):
        assd = 0.0
        hd95 = 0.0
    elif not np.any(boundary_a) or not np.any(boundary_b):
        max_distance = float(np.hypot(boundary_a.shape[0], boundary_a.shape[1]))
        assd = max_distance
        hd95 = max_distance
    else:
        distance_to_b = artifact_b.ensure_boundary_distance_transform()
        distance_to_a = artifact_a.ensure_boundary_distance_transform()
        distances = np.concatenate([distance_to_b[boundary_a], distance_to_a[boundary_b]]).astype(np.float32, copy=False)
        assd = float(np.mean(distances, dtype=np.float64)) if distances.size else 0.0
        hd95 = float(np.percentile(distances, 95.0)) if distances.size else 0.0
    perimeter_a = int(np.count_nonzero(boundary_a))
    perimeter_b = int(np.count_nonzero(boundary_b))
    perimeter_delta = abs(perimeter_a - perimeter_b) / max(1, max(perimeter_a, perimeter_b))
    return [
        _metric("boundary_iou_ab", boundary_iou, "geometry", "IoU of boundary pixels."),
        _metric("assd_ab", assd, "geometry", "Average symmetric surface distance.", "px"),
        _metric("hd95_ab", hd95, "geometry", "95th percentile Hausdorff distance.", "px"),
        _metric("centroid_distance_ab", _centroid_distance(artifact_a.binary_mask, artifact_b.binary_mask), "geometry", "Centroid distance.", "px"),
        _metric("perimeter_delta", perimeter_delta, "geometry", "Relative boundary-length delta."),
    ]


def _topology_metrics_prepared(artifact_a: ModelArtifacts, artifact_b: ModelArtifacts) -> list[MetricValue]:
    metrics_a = artifact_a.ensure_topology()
    metrics_b = artifact_b.ensure_topology()
    return [
        _metric("beta0_a", metrics_a["beta0"], "topology", "Connected component count for A."),
        _metric("beta0_b", metrics_b["beta0"], "topology", "Connected component count for B."),
        _metric("beta1_a", metrics_a["beta1"], "topology", "Cycle/hole count for A."),
        _metric("beta1_b", metrics_b["beta1"], "topology", "Cycle/hole count for B."),
        _metric("euler_a", metrics_a["euler"], "topology", "Euler characteristic for A."),
        _metric("euler_b", metrics_b["euler"], "topology", "Euler characteristic for B."),
    ]


def _risk_from_metrics(metrics: list[MetricValue]) -> dict[str, float]:
    values = {metric.name: metric.value for metric in metrics if metric.valid and metric.value is not None}
    pixel = float(values.get("foreground_disagreement_rate", values.get("disagreement_rate", 0.0)) or 0.0)
    geometry = float(min(1.0, (values.get("hd95_ab", 0.0) or 0.0) / 16.0)) if "hd95_ab" in values else pixel
    skeleton = 1.0 - float(values.get("skeleton_f1_r1_ab", 1.0) or 1.0)
    topology = min(1.0, float(abs((values.get("beta0_a", 0.0) or 0.0) - (values.get("beta0_b", 0.0) or 0.0))) / 4.0)
    total = float(np.mean([pixel, geometry, skeleton, topology], dtype=np.float64))
    return {
        "pixel": float(np.clip(pixel, 0.0, 1.0)),
        "geometry": float(np.clip(geometry, 0.0, 1.0)),
        "skeleton": float(np.clip(skeleton, 0.0, 1.0)),
        "topology": float(np.clip(topology, 0.0, 1.0)),
        "total": float(np.clip(total, 0.0, 1.0)),
    }


def _centroid_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    centroid_a = _centroid(mask_a)
    centroid_b = _centroid(mask_b)
    if centroid_a is None and centroid_b is None:
        return 0.0
    if centroid_a is None or centroid_b is None:
        return float(np.hypot(mask_a.shape[0], mask_a.shape[1]))
    return float(np.hypot(centroid_a[0] - centroid_b[0], centroid_a[1] - centroid_b[1]))


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size <= 0:
        return None
    return float(np.mean(xs, dtype=np.float64)), float(np.mean(ys, dtype=np.float64))


def _merge_timings(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        target[str(key)] = float(target.get(str(key), 0.0)) + float(value)


def _elapsed_ms(started: float) -> float:
    return 1000.0 * (perf_counter() - started)
