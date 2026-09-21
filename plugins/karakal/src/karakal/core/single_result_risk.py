"""Explainable risk assessment for one binary model output."""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from math import pi
from pathlib import Path
from typing import Callable

import numpy as np

from .confidence_analysis import _model_output_confidence_metrics
from .domain import BuildResult, FrameAnalysisSummary, FrameRecord
from .image_io import load_grayscale_image, resize_grayscale_image
from .mask_primitives import _binary_dilate, _boundary_mask, _label_components, _mask_structure
from .metric_keys import compute_metric_percentiles
from .repository_shared import BuildCancelledError
from .source_mask_from_gray import (
    SOURCE_MASK_ALGO_VERSION,
    build_source_mask,
    mask_dice,
    mask_iou,
)


SINGLE_RESULT_RISK_METRIC = "single_result_risk_score"
MASK_STRUCTURE_RISK_METRIC = "mask_structure_risk"
BATCH_OUTLIER_RISK_METRIC = "batch_outlier_risk"
SOURCE_ALIGNMENT_RISK_METRIC = "source_alignment_risk"
SOURCE_MASK_AGREEMENT_RISK_METRIC = "source_mask_agreement_risk"
CONFIDENCE_RISK_METRIC = "confidence_risk"
SINGLE_RESULT_RISK_METRICS = (
    SINGLE_RESULT_RISK_METRIC,
    MASK_STRUCTURE_RISK_METRIC,
    BATCH_OUTLIER_RISK_METRIC,
    SOURCE_ALIGNMENT_RISK_METRIC,
    SOURCE_MASK_AGREEMENT_RISK_METRIC,
    CONFIDENCE_RISK_METRIC,
)

SENSITIVITY_MULTIPLIERS = {"soft": 1.25, "balanced": 1.0, "strict": 0.8}
MIN_BASELINE_FRAMES = 10


@dataclass(frozen=True, slots=True)
class RiskReason:
    code: str
    severity: str
    contribution: float
    bbox: tuple[float, float, float, float] | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class SingleResultRiskSummary:
    total_risk: float
    structure_risk: float
    batch_outlier_risk: float | None
    source_alignment_risk: float | None
    source_mask_agreement_risk: float | None
    confidence_risk: float | None
    sensitivity: str
    evidence: tuple[str, ...]
    reasons: tuple[RiskReason, ...]
    component_weights: dict[str, float]
    component_contributions: dict[str, float]
    feature_values: dict[str, float]


@dataclass(slots=True)
class _FeatureRow:
    record: FrameRecord
    features: dict[str, float]
    structure_raw: float
    source_raw: float | None
    source_mask_raw: float | None
    confidence_raw: float | None
    reasons: list[RiskReason]
    error: str = ""


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0)) if np.isfinite(value) else 0.0


def _severity(value: float) -> str:
    if value < 0.50:
        return "low"
    if value < 0.75:
        return "medium"
    if value < 0.90:
        return "high"
    return "critical"


def _bbox_for_label(labels: np.ndarray, label_id: int) -> tuple[float, float, float, float] | None:
    yy, xx = np.nonzero(labels == int(label_id))
    if yy.size == 0:
        return None
    height, width = labels.shape
    x0, x1 = int(np.min(xx)), int(np.max(xx)) + 1
    y0, y1 = int(np.min(yy)), int(np.max(yy)) + 1
    return (
        x0 / max(1, width),
        y0 / max(1, height),
        (x1 - x0) / max(1, width),
        (y1 - y0) / max(1, height),
    )


def _region_reasons(mask: np.ndarray, code: str, contribution: float, *, limit: int = 4) -> list[RiskReason]:
    labels, count = _label_components(np.asarray(mask, dtype=bool))
    if count <= 0:
        return []
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    label_ids = sorted(range(1, count + 1), key=lambda item: int(areas[item]), reverse=True)[:limit]
    return [
        RiskReason(code, _severity(contribution), contribution, _bbox_for_label(labels, label_id))
        for label_id in label_ids
    ]


def _hole_count(mask: np.ndarray) -> int:
    inverse_labels, count = _label_components(~np.asarray(mask, dtype=bool))
    if count <= 0:
        return 0
    border_ids = set(np.unique(inverse_labels[0, :]).tolist())
    border_ids.update(np.unique(inverse_labels[-1, :]).tolist())
    border_ids.update(np.unique(inverse_labels[:, 0]).tolist())
    border_ids.update(np.unique(inverse_labels[:, -1]).tolist())
    return sum(1 for label_id in range(1, count + 1) if label_id not in border_ids)


def _dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if not np.any(value):
        return value
    return np.asarray(_binary_dilate(value, radius=max(0, int(round(float(radius))))), dtype=bool)


def _input_edge_strength(original: np.ndarray) -> np.ndarray:
    values = np.asarray(original, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = (float(item) for item in np.percentile(finite, (5.0, 95.0)))
    normalized = np.zeros_like(values, dtype=np.float32)
    if high > low + 1e-6:
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    grad_x = np.zeros_like(normalized)
    grad_y = np.zeros_like(normalized)
    grad_x[:, :-1] = np.abs(normalized[:, 1:] - normalized[:, :-1])
    grad_y[:-1, :] = np.abs(normalized[1:, :] - normalized[:-1, :])
    gradient = np.hypot(grad_x, grad_y).astype(np.float32)
    positive = gradient[gradient > 0.0]
    if positive.size:
        scale = max(float(np.percentile(positive, 95.0)), 1e-6)
        gradient = np.clip(gradient / scale, 0.0, 1.0)
    return gradient


def _source_alignment(mask: np.ndarray, original: np.ndarray) -> tuple[float, list[RiskReason]]:
    if original.shape != mask.shape:
        original = resize_grayscale_image(np.asarray(original, dtype=np.uint8), mask.shape)
    boundary = _boundary_mask(mask)
    if not np.any(boundary):
        return 1.0, []
    strength = _input_edge_strength(original)
    strong_edges = strength >= 0.35
    supported = _dilate(strong_edges, 2.0)
    unsupported = boundary & ~supported
    unsupported_fraction = float(np.count_nonzero(unsupported) / max(1, np.count_nonzero(boundary)))
    boundary_contrast = float(np.mean(strength[boundary], dtype=np.float64))
    values = np.asarray(original, dtype=np.float32)
    finite = values[np.isfinite(values)]
    local_side_contrast = 0.0
    if finite.size:
        low, high = (float(item) for item in np.percentile(finite, (5.0, 95.0)))
        normalized = np.zeros_like(values, dtype=np.float32)
        if high > low + 1e-6:
            normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
        boundary_band = _dilate(boundary, 2.0)
        inside_band = boundary_band & mask
        outside_band = boundary_band & ~mask
        if np.any(inside_band) and np.any(outside_band):
            local_side_contrast = abs(
                float(np.mean(normalized[inside_band], dtype=np.float64))
                - float(np.mean(normalized[outside_band], dtype=np.float64))
            )
    risk = _clip01(
        0.60 * unsupported_fraction
        + 0.20 * (1.0 - boundary_contrast)
        + 0.20 * (1.0 - local_side_contrast)
    )
    reasons = []
    if unsupported_fraction >= 0.25:
        reasons.extend(_region_reasons(unsupported, "unsupported_boundary", risk))
    return risk, reasons


def _mask_features(mask: np.ndarray) -> tuple[dict[str, float], float, list[RiskReason]]:
    structure = _mask_structure(mask, include_skeleton=True, include_component_labels=True)
    labels = np.asarray(structure["labels"], dtype=np.int32)
    count = int(structure["component_count"])
    total = max(1, int(mask.size))
    foreground = int(np.count_nonzero(mask))
    area_fraction = foreground / total
    areas = np.bincount(labels.ravel(), minlength=count + 1)[1:] if count else np.zeros(0, dtype=np.int64)
    tiny_threshold = max(4, int(round(total * 0.00001)))
    tiny_ids = np.flatnonzero(areas <= tiny_threshold) + 1
    tiny_pixels = int(np.sum(areas[tiny_ids - 1])) if tiny_ids.size else 0
    tiny_fraction = tiny_pixels / max(1, foreground)
    boundary = np.asarray(structure["boundary"], dtype=bool)
    perimeter = int(np.count_nonzero(boundary))
    compactness = (perimeter * perimeter) / max(1.0, 4.0 * pi * foreground)
    skeleton_length = float(structure["skeleton_length"])
    endpoints = float(structure["endpoint_count"])
    branchpoints = float(structure["branchpoint_count"])
    skeleton = np.asarray(structure["skeleton"], dtype=bool)
    skeleton_neighbors = np.asarray(structure["skeleton_neighbors"], dtype=np.uint8)
    short_branch_limit = max(3, int(round(np.sqrt(total) * 0.02)))
    short_branches = 0
    endpoints_y, endpoints_x = np.nonzero(skeleton & (skeleton_neighbors == 1))
    for start_y, start_x in zip(endpoints_y.tolist(), endpoints_x.tolist()):
        previous: tuple[int, int] | None = None
        current = (int(start_y), int(start_x))
        for distance in range(1, short_branch_limit + 2):
            cy, cx = current
            if distance > 1 and int(skeleton_neighbors[cy, cx]) >= 3:
                if distance - 1 <= short_branch_limit:
                    short_branches += 1
                break
            candidates: list[tuple[int, int]] = []
            for ny in range(max(0, cy - 1), min(mask.shape[0], cy + 2)):
                for nx in range(max(0, cx - 1), min(mask.shape[1], cx + 2)):
                    candidate = (ny, nx)
                    if candidate != current and candidate != previous and skeleton[ny, nx]:
                        candidates.append(candidate)
            if len(candidates) != 1:
                break
            previous, current = current, candidates[0]
    border_pixels = int(
        np.count_nonzero(mask[0, :])
        + np.count_nonzero(mask[-1, :])
        + np.count_nonzero(mask[:, 0])
        + np.count_nonzero(mask[:, -1])
    )
    border_length = max(1, 2 * mask.shape[0] + 2 * mask.shape[1])
    holes = _hole_count(mask)
    features = {
        "area_fraction": float(area_fraction),
        "component_density": float(count * 1_000_000.0 / total),
        "mean_component_area_fraction": float(np.mean(areas) / total) if areas.size else 0.0,
        "tiny_component_fraction": float(tiny_fraction),
        "boundary_fraction": float(perimeter / total),
        "compactness_log": float(np.log1p(max(0.0, compactness))),
        "hole_density": float(holes * 1_000_000.0 / total),
        "skeleton_fraction": float(skeleton_length / total),
        "endpoint_density": float(endpoints / max(1.0, skeleton_length)),
        "branchpoint_density": float(branchpoints / max(1.0, skeleton_length)),
        "short_branch_density": float(short_branches / max(1.0, skeleton_length)),
        "border_contact_fraction": float(border_pixels / border_length),
    }
    reasons: list[RiskReason] = []
    if foreground == 0:
        reasons.append(RiskReason("empty_mask", "critical", 1.0, (0.0, 0.0, 1.0, 1.0)))
        return features, 1.0, reasons
    if foreground == total:
        reasons.append(RiskReason("full_mask", "critical", 1.0, (0.0, 0.0, 1.0, 1.0)))
        return features, 1.0, reasons

    tiny_risk = _clip01(tiny_fraction / 0.15)
    fragmentation_risk = _clip01(max(0.0, count - 1.0) / 8.0)
    roughness_risk = _clip01(max(0.0, compactness - 12.0) / 48.0)
    hole_risk = _clip01(holes / max(1.0, count * 3.0))
    spur_risk = _clip01(
        (endpoints + branchpoints + 2.0 * short_branches) / max(1.0, skeleton_length * 0.18)
    )
    border_risk = _clip01(max(0.0, features["border_contact_fraction"] - 0.40) / 0.60)
    structure_risk = _clip01(
        0.25 * tiny_risk
        + 0.20 * fragmentation_risk
        + 0.15 * roughness_risk
        + 0.10 * hole_risk
        + 0.20 * spur_risk
        + 0.10 * border_risk
    )
    if tiny_risk >= 0.35:
        tiny_mask = np.isin(labels, tiny_ids)
        reasons.extend(_region_reasons(tiny_mask, "tiny_components", tiny_risk))
    for code, value in (
        ("fragmented_mask", fragmentation_risk),
        ("rough_boundary", roughness_risk),
        ("excess_holes", hole_risk),
        ("skeleton_complexity", spur_risk),
        ("edge_clipping", border_risk),
    ):
        if value >= 0.35:
            reasons.append(RiskReason(code, _severity(value), value))
    return features, structure_risk, reasons


def _extract_row(record: FrameRecord, model_id: str) -> _FeatureRow:
    path_text = str((record.model_mask_paths or {}).get(model_id) or "")
    if not path_text:
        return _FeatureRow(record, {}, 0.0, None, None, None, [], "missing_model_output")
    try:
        gray = load_grayscale_image(Path(path_text))
        unique_values = np.unique(gray)
        if unique_values.size > 2:
            return _FeatureRow(record, {}, 0.0, None, None, None, [], "non_binary_mask")
        if unique_values.size == 1:
            mask = np.asarray(gray > 0, dtype=bool)
        else:
            mask = np.asarray(gray == unique_values[-1], dtype=bool)
        features, structure_raw, reasons = _mask_features(mask)
        source_raw = None
        source_mask_raw = None
        if record.original_path:
            source_gray = load_grayscale_image(Path(record.original_path))
            source_raw, source_reasons = _source_alignment(mask, source_gray)
            reasons.extend(source_reasons)
            source_mask = build_source_mask(source_gray, target_shape=mask.shape)
            iou = mask_iou(source_mask, mask)
            dice = mask_dice(source_mask, mask)
            source_mask_raw = _clip01(1.0 - iou)
            features["source_iou"] = float(iou)
            features["source_dice"] = float(dice)
            features["source_mask_algo_version"] = float(str(SOURCE_MASK_ALGO_VERSION).lstrip("v") or 1.0)
            if source_mask_raw >= 0.25:
                mismatch = np.logical_xor(source_mask, mask)
                reasons.extend(_region_reasons(mismatch, "source_mask_mismatch", source_mask_raw))
        confidence_raw = None
        confidence_path = str((record.model_prob_paths or {}).get(model_id) or "")
        if confidence_path:
            confidence_gray = load_grayscale_image(Path(confidence_path))
            if confidence_gray.shape != mask.shape:
                confidence_gray = resize_grayscale_image(confidence_gray, mask.shape)
            confidence = np.asarray(confidence_gray, dtype=np.float32) / 255.0
            confidence_raw = float(_model_output_confidence_metrics(confidence).frame_uncertainty_score)
            uncertainty = 1.0 - np.abs(2.0 * confidence - 1.0)
            if confidence_raw >= 0.35:
                reasons.extend(_region_reasons(uncertainty >= 0.5, "low_confidence", confidence_raw))
        return _FeatureRow(record, features, structure_raw, source_raw, source_mask_raw, confidence_raw, reasons)
    except Exception:
        return _FeatureRow(record, {}, 0.0, None, None, None, [], "decode_error")


def _robust_baseline(rows: list[_FeatureRow]) -> dict[str, tuple[float, float]]:
    if len(rows) < MIN_BASELINE_FRAMES:
        return {}
    keys = sorted(set.intersection(*(set(row.features) for row in rows))) if rows else []
    baseline: dict[str, tuple[float, float]] = {}
    for key in keys:
        values = np.asarray([row.features[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < MIN_BASELINE_FRAMES:
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad <= 1e-12:
            q25, q75 = (float(value) for value in np.percentile(values, (25.0, 75.0)))
            mad = (q75 - q25) / 1.349
        if mad > 1e-12:
            baseline[key] = (median, mad)
    return baseline


def _batch_risk(row: _FeatureRow, baseline: dict[str, tuple[float, float]], multiplier: float) -> tuple[float | None, list[RiskReason]]:
    if not baseline:
        return None, [RiskReason("insufficient_baseline", "low", 0.0, detail=str(MIN_BASELINE_FRAMES))]
    contributions: list[tuple[float, str]] = []
    for key, (median, scale) in baseline.items():
        value = row.features.get(key)
        if value is None or not np.isfinite(value):
            continue
        robust_z = abs(float(value) - median) / max(scale * 1.4826, 1e-12)
        contributions.append((_clip01(robust_z / max(0.1, 3.0 * multiplier)), key))
    contributions.sort(reverse=True)
    top = contributions[:3]
    risk = float(np.mean([value for value, _key in top], dtype=np.float64)) if top else 0.0
    reasons = [
        RiskReason(f"batch_outlier.{key}", _severity(value), value)
        for value, key in contributions
        if value >= 0.35
    ]
    return risk, reasons[:4]


def _scaled(value: float | None, multiplier: float) -> float | None:
    return None if value is None else _clip01(float(value) / max(0.1, multiplier))


def _risk_extract_worker_count(max_workers: int, record_count: int) -> int:
    if record_count <= 1:
        return 1
    requested = max(1, int(max_workers))
    cpu_limit = max(1, os.cpu_count() or requested)
    return min(requested, cpu_limit, record_count)


def _extract_rows(
    source_records: list[FrameRecord],
    model_id: str,
    *,
    max_workers: int,
    total_work: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
    state_callback: Callable[[str, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[_FeatureRow]:
    """Extract per-frame features, optionally in parallel via ThreadPool."""

    if not source_records:
        return []
    worker_count = _risk_extract_worker_count(max_workers, len(source_records))
    if worker_count <= 1:
        rows: list[_FeatureRow] = []
        for index, record in enumerate(source_records, start=1):
            if cancel_check is not None and cancel_check():
                raise BuildCancelledError("Build cancelled")
            if state_callback is not None:
                state_callback(str(record.key), "processing")
            row = _extract_row(record, model_id)
            rows.append(row)
            if progress_callback is not None:
                progress_callback(index, total_work, str(record.key))
        return rows

    ordered: list[_FeatureRow | None] = [None] * len(source_records)
    completed = 0
    executor = ThreadPoolExecutor(max_workers=worker_count)
    shutdown_wait = True
    try:
        future_to_index = {}
        for index, record in enumerate(source_records):
            if cancel_check is not None and cancel_check():
                raise BuildCancelledError("Build cancelled")
            if state_callback is not None:
                state_callback(str(record.key), "processing")
            future = executor.submit(_extract_row, record, model_id)
            future_to_index[future] = index

        pending = set(future_to_index.keys())
        while pending:
            if cancel_check is not None and cancel_check():
                shutdown_wait = False
                executor.shutdown(wait=False, cancel_futures=True)
                raise BuildCancelledError("Build cancelled")
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                index = future_to_index[future]
                record = source_records[index]
                ordered[index] = future.result()
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total_work, str(record.key))
    finally:
        if shutdown_wait:
            executor.shutdown(wait=True, cancel_futures=False)

    return [row for row in ordered if row is not None]


def compute_single_result_risk(
    build_result: BuildResult,
    *,
    sensitivity: str = "balanced",
    excluded_record_keys: set[str] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    state_callback: Callable[[str, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> BuildResult:
    """Return a build result ranked by explainable single-output anomaly risk."""

    if len(build_result.model_specs) != 1:
        raise ValueError("Single-result risk requires exactly one model output")
    sensitivity_key = str(sensitivity or "balanced").strip().lower()
    multiplier = float(SENSITIVITY_MULTIPLIERS.get(sensitivity_key, 1.0))
    sensitivity_key = sensitivity_key if sensitivity_key in SENSITIVITY_MULTIPLIERS else "balanced"
    excluded = {str(key) for key in (excluded_record_keys or set())}
    model = build_result.model_specs[0]
    source_records = [record for record in build_result.records if str(record.key) not in excluded]
    total_work = max(1, len(source_records) * 2)
    max_workers = int(getattr(build_result.options, "max_workers", 0) or (os.cpu_count() or 4))
    rows = _extract_rows(
        source_records,
        str(model.model_id),
        max_workers=max_workers,
        total_work=total_work,
        progress_callback=progress_callback,
        state_callback=state_callback,
        cancel_check=cancel_check,
    )

    valid_rows = [row for row in rows if not row.error and row.features]
    baseline = _robust_baseline(valid_rows)
    updated: dict[str, FrameRecord] = {}
    for offset, row in enumerate(rows, start=1):
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError("Build cancelled")
        record = row.record
        if row.error:
            summary = FrameAnalysisSummary(0.0, 0.0, 0.0, 0.0, notes=(row.error,), frame_type="polygon")
            updated[str(record.key)] = replace(record, summary=summary, score_ready=False, absolute_score=None)
            if state_callback is not None:
                state_callback(str(record.key), "error")
            continue
        batch_raw, batch_reasons = _batch_risk(row, baseline, multiplier)
        critical_absolute = any(reason.code in {"empty_mask", "full_mask"} for reason in row.reasons)
        structure_risk = 1.0 if critical_absolute else (_scaled(row.structure_raw, multiplier) or 0.0)
        source_risk = _scaled(row.source_raw, multiplier)
        source_mask_risk = _scaled(row.source_mask_raw, multiplier)
        confidence_risk = row.confidence_raw
        component_values = {
            MASK_STRUCTURE_RISK_METRIC: structure_risk,
            BATCH_OUTLIER_RISK_METRIC: batch_raw,
            SOURCE_ALIGNMENT_RISK_METRIC: source_risk,
            SOURCE_MASK_AGREEMENT_RISK_METRIC: source_mask_risk,
            CONFIDENCE_RISK_METRIC: confidence_risk,
        }
        nominal_weights = {
            MASK_STRUCTURE_RISK_METRIC: 0.35,
            BATCH_OUTLIER_RISK_METRIC: 0.25,
            SOURCE_ALIGNMENT_RISK_METRIC: 0.15,
            SOURCE_MASK_AGREEMENT_RISK_METRIC: 0.15,
            CONFIDENCE_RISK_METRIC: 0.10,
        }
        available = [(key, value) for key, value in component_values.items() if value is not None]
        weight_sum = sum(nominal_weights[key] for key, _value in available)
        component_weights = {
            key: nominal_weights[key] / max(weight_sum, 1e-12) for key, _value in available
        }
        component_contributions = {
            key: 100.0 * component_weights[key] * float(value) for key, value in available
        }
        total_risk = sum(component_contributions.values()) / 100.0
        if critical_absolute:
            component_contributions["absolute_rule"] = max(
                0.0, 100.0 - sum(component_contributions.values())
            )
            total_risk = 1.0
        reasons = [
            replace(
                reason,
                contribution=(
                    1.0
                    if reason.code in {"empty_mask", "full_mask"}
                    else reason.contribution
                    if reason.code == "low_confidence"
                    else (_scaled(reason.contribution, multiplier) or 0.0)
                ),
                severity=(
                    "critical"
                    if reason.code in {"empty_mask", "full_mask"}
                    else _severity(reason.contribution)
                    if reason.code == "low_confidence"
                    else _severity(_scaled(reason.contribution, multiplier) or 0.0)
                ),
            )
            for reason in row.reasons
        ]
        reasons.extend(batch_reasons)
        reasons.sort(key=lambda item: item.contribution, reverse=True)
        reasons = reasons[:8]
        evidence = ["model_output"]
        if source_risk is not None or source_mask_risk is not None:
            evidence.append("original")
        if confidence_risk is not None:
            evidence.append("confidence")
        risk_summary = SingleResultRiskSummary(
            total_risk=100.0 * _clip01(total_risk),
            structure_risk=100.0 * structure_risk,
            batch_outlier_risk=None if batch_raw is None else 100.0 * batch_raw,
            source_alignment_risk=None if source_risk is None else 100.0 * source_risk,
            source_mask_agreement_risk=None if source_mask_risk is None else 100.0 * source_mask_risk,
            confidence_risk=None if confidence_risk is None else 100.0 * confidence_risk,
            sensitivity=sensitivity_key,
            evidence=tuple(evidence),
            reasons=tuple(reasons),
            component_weights=component_weights,
            component_contributions=component_contributions,
            feature_values=dict(row.features),
        )
        metric_values = {
            SINGLE_RESULT_RISK_METRIC: risk_summary.total_risk,
            MASK_STRUCTURE_RISK_METRIC: risk_summary.structure_risk,
        }
        for key, value in (
            (BATCH_OUTLIER_RISK_METRIC, risk_summary.batch_outlier_risk),
            (SOURCE_ALIGNMENT_RISK_METRIC, risk_summary.source_alignment_risk),
            (SOURCE_MASK_AGREEMENT_RISK_METRIC, risk_summary.source_mask_agreement_risk),
            (CONFIDENCE_RISK_METRIC, risk_summary.confidence_risk),
        ):
            if value is not None:
                metric_values[key] = value
        summary = FrameAnalysisSummary(
            disagreement_score=0.0,
            temporal_instability=0.0,
            structural_anomaly=structure_risk,
            export_priority_score=risk_summary.total_risk / 100.0,
            metric_values=metric_values,
            notes=tuple(reason.code for reason in reasons),
            frame_type="polygon",
            single_result_risk=risk_summary,
        )
        updated[str(record.key)] = replace(
            record,
            summary=summary,
            absolute_score=risk_summary.total_risk,
            score=1.0 - risk_summary.total_risk / 100.0,
            relative_score=risk_summary.total_risk / 100.0,
            score_ready=True,
        )
        if state_callback is not None:
            state_callback(str(record.key), "ready")
        if progress_callback is not None:
            progress_callback(len(source_records) + offset, total_work, str(record.key))

    records = tuple(updated.get(str(record.key), replace(record, score_ready=False)) for record in build_result.records)
    percentile_map = compute_metric_percentiles(records, SINGLE_RESULT_RISK_METRIC)
    records = tuple(
        replace(record, score_percentile=float(percentile_map.get(record.key, 0.0))) if record.score_ready else record
        for record in records
    )
    scores = [float(record.absolute_score) for record in records if record.score_ready and record.absolute_score is not None]
    available_metrics = [SINGLE_RESULT_RISK_METRIC, MASK_STRUCTURE_RISK_METRIC]
    for key in (
        BATCH_OUTLIER_RISK_METRIC,
        SOURCE_ALIGNMENT_RISK_METRIC,
        SOURCE_MASK_AGREEMENT_RISK_METRIC,
        CONFIDENCE_RISK_METRIC,
    ):
        if any(record.summary is not None and key in record.summary.metric_values for record in records):
            available_metrics.append(key)
    return replace(
        build_result,
        records=records,
        min_score=min((record.score for record in records if record.score_ready), default=0.0),
        max_score=max((record.score for record in records if record.score_ready), default=0.0),
        scores_computed=True,
        best_match_key=min(
            (record for record in records if record.score_ready and record.absolute_score is not None),
            key=lambda item: float(item.absolute_score),
            default=None,
        ).key
        if scores
        else None,
        min_absolute_score=min(scores) if scores else None,
        max_absolute_score=max(scores) if scores else None,
        selected_metric_key=SINGLE_RESULT_RISK_METRIC,
        available_metric_keys=tuple(available_metrics),
    )


__all__ = [
    "BATCH_OUTLIER_RISK_METRIC",
    "CONFIDENCE_RISK_METRIC",
    "MASK_STRUCTURE_RISK_METRIC",
    "MIN_BASELINE_FRAMES",
    "RiskReason",
    "SENSITIVITY_MULTIPLIERS",
    "SINGLE_RESULT_RISK_METRIC",
    "SINGLE_RESULT_RISK_METRICS",
    "SOURCE_ALIGNMENT_RISK_METRIC",
    "SOURCE_MASK_AGREEMENT_RISK_METRIC",
    "SingleResultRiskSummary",
    "compute_single_result_risk",
]
