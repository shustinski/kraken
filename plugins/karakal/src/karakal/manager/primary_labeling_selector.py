"""Hard-case mining helpers for manager-mode labeling priority overlays."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core.pipeline import discover_image_paths
from ..core.repository import load_grayscale_image

EPS = 1e-8


@dataclass(frozen=True, slots=True)
class LabelingPriorityConfig:
    target_ratio: float = 0.10
    candidate_pool_ratio: float = 0.35
    enable_diversity: bool = True
    include_normal_reference_frames: bool = False
    max_normal_reference_frames: int = 1
    max_cluster_share: float = 0.40
    diversity_alpha: float = 0.65
    diversity_beta: float = 0.35
    min_cluster_size: int = 2
    max_pattern_clusters: int = 10
    non_dominant_cluster_min_selection: int = 2


@dataclass(frozen=True, slots=True)
class LabelingPriorityFrame:
    frame_key: str
    display_name: str
    original_path: str
    priority_score: float
    category: str
    reasons: tuple[str, ...]
    recommended: bool
    rank: int
    pattern_cluster_id: int
    pattern_group: str
    artifact_score: float
    rarity_score: float
    uncertainty_score: float | None
    edge_complexity_score: float
    object_complexity_score: float
    saturation_or_noise_score: float
    metadata: dict[str, float | int | str | bool | None]


@dataclass(frozen=True, slots=True)
class LabelingPriorityResult:
    total_discovered: int
    selected_count: int
    frames: tuple[LabelingPriorityFrame, ...]


@dataclass(frozen=True, slots=True)
class _FrameRow:
    path: Path
    frame_key: str
    display_name: str
    image: np.ndarray
    prediction: np.ndarray | None


@dataclass(frozen=True, slots=True)
class PrimaryLabelingConfig:
    target_ratio: float = 0.10
    candidate_pool_ratio: float = 0.35
    enable_diversity_filter: bool = True
    include_normal_reference_frames: bool = False
    max_normal_reference_frames: int = 1
    max_group_share: float = 0.40
    alpha: float = 0.65
    beta: float = 0.35
    min_group_size: int = 2
    max_pattern_groups: int = 10
    non_dominant_cluster_min_selection: int = 2


@dataclass(frozen=True, slots=True)
class FrameFeatureSet:
    frame_key: str
    display_name: str
    mean_intensity: float
    std_intensity: float
    min_intensity: float
    max_intensity: float
    entropy: float
    edge_density: float
    sobel_mean: float
    sobel_std: float
    laplacian_variance: float
    high_frequency_score: float
    object_count: float
    area_ratio: float
    total_object_area: float
    mean_object_area: float
    object_area_std: float
    small_object_count: float
    large_object_count: float
    object_density: float
    contour_complexity: float
    compactness_score: float
    merged_object_score: float
    thin_structure_score: float
    pattern_vector: np.ndarray
    hardness_vector: np.ndarray


@dataclass(frozen=True, slots=True)
class FrameSelectionResult:
    frame_key: str
    display_name: str
    priority_score: float
    selected_for_primary_labeling: bool
    pattern_group_id: int
    pattern_group_label: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrimaryLabelingSelectionResult:
    total_discovered: int
    selected_count: int
    frames: tuple[FrameSelectionResult, ...]


_PRIMARY_LABELING_CACHE: dict[tuple[str, int, int, float, float, bool, bool, int, float], LabelingPriorityResult] = {}


def normalize_scores(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size <= 0:
        return np.zeros((0,), dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    result = np.zeros_like(array, dtype=np.float32)
    valid = array[finite]
    low = float(np.min(valid))
    high = float(np.max(valid))
    if abs(high - low) <= EPS:
        result[finite] = 0.0
        return result
    result[finite] = (valid - low) / max(EPS, high - low)
    result[~finite] = 0.0
    return result


def _spectral_peak_ratio(profile: np.ndarray) -> float:
    values = np.asarray(profile, dtype=np.float32).reshape(-1)
    if values.size <= 2:
        return 0.0
    centered = values - float(np.mean(values))
    spectrum = np.abs(np.fft.rfft(centered))
    if spectrum.size <= 1:
        return 0.0
    spectrum[0] = 0.0
    total = float(np.sum(spectrum))
    if total <= EPS:
        return 0.0
    return float(np.clip(float(np.max(spectrum)) / total, 0.0, 1.0))


def extract_pattern_features(frame: np.ndarray, *, precomputed: dict[str, float] | None = None) -> np.ndarray:
    image = _to_uint8(frame)
    normalized = image.astype(np.float32) / 255.0
    if precomputed is None:
        padded = np.pad(normalized, 1, mode="edge")
        gx = padded[1:-1, 2:] - padded[1:-1, :-2]
        gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
        grad = np.hypot(gx, gy)
        edge_density = float(np.mean(grad >= 0.16))
        grad_energy = float(np.mean(np.clip(grad, 0.0, 1.0)))
        binary = normalized > _otsu_threshold_uint8(image)
        reduced = _downsample_bool(binary, max_side=96)
        component_count, mean_area, area_var, small_ratio, merged_ratio = _component_stats(reduced)
        active_ratio = float(np.mean(binary))
        transition_h = float(np.mean(np.abs(np.diff(reduced.astype(np.int8), axis=1)))) if reduced.shape[1] > 1 else 0.0
        transition_v = float(np.mean(np.abs(np.diff(reduced.astype(np.int8), axis=0)))) if reduced.shape[0] > 1 else 0.0
        q_std = _quadrant_std(reduced)
        row_profile = normalized.mean(axis=1)
        col_profile = normalized.mean(axis=0)
        row_profile_std = float(np.std(row_profile))
        col_profile_std = float(np.std(col_profile))
        row_profile_delta = float(np.mean(np.abs(np.diff(row_profile)))) if row_profile.size > 1 else 0.0
        col_profile_delta = float(np.mean(np.abs(np.diff(col_profile)))) if col_profile.size > 1 else 0.0
        row_periodicity = _spectral_peak_ratio(row_profile)
        col_periodicity = _spectral_peak_ratio(col_profile)
        gradient_x_energy = float(np.mean(np.abs(gx)))
        gradient_y_energy = float(np.mean(np.abs(gy)))
        gradient_axis_bias = float(abs(gradient_x_energy - gradient_y_energy) / max(EPS, gradient_x_energy + gradient_y_energy))
        projection_balance = float(abs(row_profile_std - col_profile_std) / max(EPS, row_profile_std + col_profile_std))
        stripe_score = float(min(1.0, (float(np.std(np.diff(row_profile, prepend=row_profile[:1]))) + float(np.std(np.diff(col_profile, prepend=col_profile[:1])))) / max(EPS, float(np.std(normalized) + 0.01))))
        saturation_ratio = float(np.mean((image <= 5) | (image >= 250)))
        mean3 = _mean_filter_3x3(normalized)
        local_noise = float(np.mean(np.abs(normalized - mean3)))
        entropy = _entropy_0_1(image)
    else:
        edge_density = float(precomputed.get("edge_density", 0.0))
        grad_energy = float(precomputed.get("grad_energy", 0.0))
        component_count = float(precomputed.get("component_count", 0.0))
        mean_area = float(precomputed.get("mean_component_area", 0.0))
        area_var = float(precomputed.get("area_variance", 0.0))
        small_ratio = float(precomputed.get("small_component_ratio", 0.0))
        merged_ratio = float(precomputed.get("merged_ratio", 0.0))
        active_ratio = float(precomputed.get("active_ratio", 0.0))
        transition_h = float(precomputed.get("transition_h", 0.0))
        transition_v = float(precomputed.get("transition_v", 0.0))
        entropy = float(precomputed.get("entropy", 0.0))
        q_std = float(precomputed.get("q_std", 0.0))
        saturation_ratio = float(precomputed.get("saturation_ratio", 0.0))
        stripe_score = float(precomputed.get("stripe_score", 0.0))
        local_noise = float(precomputed.get("local_noise", 0.0))
        row_profile = normalized.mean(axis=1)
        col_profile = normalized.mean(axis=0)
        row_profile_std = float(precomputed.get("row_profile_std", float(np.std(row_profile))))
        col_profile_std = float(precomputed.get("col_profile_std", float(np.std(col_profile))))
        row_profile_delta = float(precomputed.get("row_profile_delta", float(np.mean(np.abs(np.diff(row_profile)))) if row_profile.size > 1 else 0.0))
        col_profile_delta = float(precomputed.get("col_profile_delta", float(np.mean(np.abs(np.diff(col_profile)))) if col_profile.size > 1 else 0.0))
        row_periodicity = float(precomputed.get("row_periodicity", _spectral_peak_ratio(row_profile)))
        col_periodicity = float(precomputed.get("col_periodicity", _spectral_peak_ratio(col_profile)))
        gradient_x_energy = float(precomputed.get("gradient_x_energy", 0.0))
        gradient_y_energy = float(precomputed.get("gradient_y_energy", 0.0))
        gradient_axis_bias = float(precomputed.get("gradient_axis_bias", 0.0))
        projection_balance = float(precomputed.get("projection_balance", 0.0))

    hist8 = _histogram_bins(image, bins=8)
    component_count_norm = float(np.clip(component_count / 24.0, 0.0, 1.0))
    return np.asarray(
        [
            *hist8.tolist(),
            edge_density,
            grad_energy,
            component_count_norm,
            mean_area,
            area_var,
            small_ratio,
            merged_ratio,
            active_ratio,
            transition_h,
            transition_v,
            entropy,
            q_std,
            saturation_ratio,
            stripe_score,
            local_noise,
            row_profile_std,
            col_profile_std,
            row_profile_delta,
            col_profile_delta,
            row_periodicity,
            col_periodicity,
            gradient_x_energy,
            gradient_y_energy,
            gradient_axis_bias,
            projection_balance,
        ],
        dtype=np.float32,
    )


def extract_frame_features(frame: np.ndarray, optional_prediction: np.ndarray | None = None) -> dict[str, float | np.ndarray]:
    grayscale = _to_uint8(frame)
    normalized = grayscale.astype(np.float32) / 255.0
    padded = np.pad(normalized, 1, mode="edge")
    mean3 = _mean_filter_3x3(normalized)
    local_noise = float(np.mean(np.abs(normalized - mean3)))

    gx = padded[1:-1, 2:] - padded[1:-1, :-2]
    gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    grad = np.hypot(gx, gy)
    edge_density = float(np.mean(grad >= 0.16))
    grad_energy = float(np.mean(np.clip(grad, 0.0, 1.0)))

    center = padded[1:-1, 1:-1]
    lap = -4.0 * center + padded[1:-1, :-2] + padded[1:-1, 2:] + padded[:-2, 1:-1] + padded[2:, 1:-1]
    lap_var = float(np.var(lap))

    clip_low = float(np.mean(grayscale <= 5))
    clip_high = float(np.mean(grayscale >= 250))
    saturation_ratio = float(clip_low + clip_high)

    row_profile = normalized.mean(axis=1)
    col_profile = normalized.mean(axis=0)
    row_stripe = float(np.std(np.diff(row_profile, prepend=row_profile[:1])))
    col_stripe = float(np.std(np.diff(col_profile, prepend=col_profile[:1])))
    stripe_score = float(min(1.0, (row_stripe + col_stripe) / max(EPS, float(np.std(normalized) + 0.01))))
    row_profile_std = float(np.std(row_profile))
    col_profile_std = float(np.std(col_profile))
    row_profile_delta = float(np.mean(np.abs(np.diff(row_profile)))) if row_profile.size > 1 else 0.0
    col_profile_delta = float(np.mean(np.abs(np.diff(col_profile)))) if col_profile.size > 1 else 0.0
    row_periodicity = _spectral_peak_ratio(row_profile)
    col_periodicity = _spectral_peak_ratio(col_profile)
    gradient_x_energy = float(np.mean(np.abs(gx)))
    gradient_y_energy = float(np.mean(np.abs(gy)))
    gradient_axis_bias = float(abs(gradient_x_energy - gradient_y_energy) / max(EPS, gradient_x_energy + gradient_y_energy))
    projection_balance = float(abs(row_profile_std - col_profile_std) / max(EPS, row_profile_std + col_profile_std))

    binary = normalized > _otsu_threshold_uint8(grayscale)
    reduced = _downsample_bool(binary, max_side=96)
    component_count, mean_component_area, area_variance, small_component_ratio, merged_ratio = _component_stats(reduced)
    component_density = float(component_count / max(1.0, reduced.size / 1024.0))
    transition_h = float(np.mean(np.abs(np.diff(reduced.astype(np.int8), axis=1)))) if reduced.shape[1] > 1 else 0.0
    transition_v = float(np.mean(np.abs(np.diff(reduced.astype(np.int8), axis=0)))) if reduced.shape[0] > 1 else 0.0
    active_ratio = float(np.mean(binary))
    q_std = _quadrant_std(reduced)
    entropy = _entropy_0_1(grayscale)

    edge_complexity = float(np.clip(0.55 * edge_density + 0.45 * grad_energy, 0.0, 1.0))
    object_complexity_raw = (
        0.35 * np.clip(component_density / 2.1, 0.0, 1.0)
        + 0.20 * np.clip(small_component_ratio * 2.0, 0.0, 1.0)
        + 0.20 * np.clip((transition_h + transition_v) * 0.9, 0.0, 1.0)
        + 0.15 * np.clip(merged_ratio * 2.0, 0.0, 1.0)
        + 0.10 * np.clip(active_ratio * 1.4, 0.0, 1.0)
    )
    artifact_raw = (
        0.26 * np.clip(local_noise * 6.0, 0.0, 1.0)
        + 0.22 * np.clip(lap_var * 2.2, 0.0, 1.0)
        + 0.18 * np.clip(saturation_ratio * 2.6, 0.0, 1.0)
        + 0.18 * np.clip(stripe_score * 2.2, 0.0, 1.0)
        + 0.16 * np.clip(abs(float(normalized.mean()) - 0.5) * 2.0, 0.0, 1.0)
    )
    saturation_or_noise = float(np.clip(0.65 * np.clip(local_noise * 6.0, 0.0, 1.0) + 0.35 * np.clip(saturation_ratio * 2.6, 0.0, 1.0), 0.0, 1.0))

    uncertainty = _prediction_uncertainty(optional_prediction, normalized, grad)
    hardness_vector = np.asarray(
        [
            local_noise,
            lap_var,
            stripe_score,
            saturation_ratio,
            edge_density,
            grad_energy,
            component_density,
            mean_component_area,
            merged_ratio,
            active_ratio,
            entropy,
        ],
        dtype=np.float32,
    )
    pattern_vector = extract_pattern_features(
        grayscale,
        precomputed={
            "edge_density": edge_density,
            "grad_energy": grad_energy,
            "component_count": component_count,
            "mean_component_area": mean_component_area,
            "area_variance": area_variance,
            "small_component_ratio": small_component_ratio,
            "merged_ratio": merged_ratio,
            "active_ratio": active_ratio,
            "transition_h": transition_h,
            "transition_v": transition_v,
            "entropy": entropy,
            "q_std": q_std,
            "saturation_ratio": saturation_ratio,
            "stripe_score": stripe_score,
            "local_noise": local_noise,
            "row_profile_std": row_profile_std,
            "col_profile_std": col_profile_std,
            "row_profile_delta": row_profile_delta,
            "col_profile_delta": col_profile_delta,
            "row_periodicity": row_periodicity,
            "col_periodicity": col_periodicity,
            "gradient_x_energy": gradient_x_energy,
            "gradient_y_energy": gradient_y_energy,
            "gradient_axis_bias": gradient_axis_bias,
            "projection_balance": projection_balance,
        },
    )

    result: dict[str, float | np.ndarray] = {
        "artifact_raw": float(np.clip(artifact_raw, 0.0, 1.0)),
        "edge_complexity_raw": float(np.clip(edge_complexity, 0.0, 1.0)),
        "object_complexity_raw": float(np.clip(object_complexity_raw, 0.0, 1.0)),
        "saturation_or_noise_raw": float(np.clip(saturation_or_noise, 0.0, 1.0)),
        "hardness_vector": hardness_vector,
        "pattern_vector": pattern_vector,
        "local_noise": local_noise,
        "stripe_score": stripe_score,
        "saturation_ratio": saturation_ratio,
        "component_count": component_count,
        "edge_density": edge_density,
        "mean_component_area": mean_component_area,
        "area_variance": area_variance,
        "active_ratio": active_ratio,
        "merged_ratio": merged_ratio,
        "entropy": entropy,
        "q_std": q_std,
        "row_profile_std": row_profile_std,
        "col_profile_std": col_profile_std,
        "row_profile_delta": row_profile_delta,
        "col_profile_delta": col_profile_delta,
        "row_periodicity": row_periodicity,
        "col_periodicity": col_periodicity,
        "gradient_x_energy": gradient_x_energy,
        "gradient_y_energy": gradient_y_energy,
        "gradient_axis_bias": gradient_axis_bias,
        "projection_balance": projection_balance,
    }
    if uncertainty is not None:
        result["uncertainty_raw"] = float(uncertainty)
    return result


def compute_labeling_priority_scores(
    frames: Iterable[np.ndarray],
    optional_predictions: Iterable[np.ndarray | None] | None = None,
    config: LabelingPriorityConfig | None = None,
) -> dict[str, np.ndarray]:
    _ = config
    frame_list = [np.asarray(item) for item in frames]
    predictions = list(optional_predictions) if optional_predictions is not None else [None] * len(frame_list)
    if len(predictions) < len(frame_list):
        predictions.extend([None] * (len(frame_list) - len(predictions)))

    extracted = [extract_frame_features(frame_list[index], predictions[index]) for index in range(len(frame_list))]
    hardness_features = np.asarray([row["hardness_vector"] for row in extracted], dtype=np.float32)
    pattern_features = np.asarray([row["pattern_vector"] for row in extracted], dtype=np.float32)

    artifact = normalize_scores([float(row["artifact_raw"]) for row in extracted])
    edge_complexity = normalize_scores([float(row["edge_complexity_raw"]) for row in extracted])
    object_complexity = normalize_scores([float(row["object_complexity_raw"]) for row in extracted])
    saturation_or_noise = normalize_scores([float(row["saturation_or_noise_raw"]) for row in extracted])
    rarity = _rarity_scores(hardness_features)

    uncertainty_values = [row.get("uncertainty_raw", np.nan) for row in extracted]
    uncertainty_array = np.asarray(uncertainty_values, dtype=np.float32)
    has_uncertainty = bool(np.isfinite(uncertainty_array).any())
    uncertainty_norm = normalize_scores(np.where(np.isfinite(uncertainty_array), uncertainty_array, 0.0)) if has_uncertainty else np.zeros((len(extracted),), dtype=np.float32)

    if has_uncertainty:
        score = (
            0.25 * artifact
            + 0.20 * rarity
            + 0.20 * uncertainty_norm
            + 0.15 * edge_complexity
            + 0.10 * object_complexity
            + 0.10 * saturation_or_noise
        )
    else:
        score = (
            0.30 * artifact
            + 0.25 * rarity
            + 0.20 * edge_complexity
            + 0.15 * object_complexity
            + 0.10 * saturation_or_noise
        )
    score = np.clip(score.astype(np.float32), 0.0, 1.0)

    return {
        "score": score,
        "artifact": artifact,
        "rarity": rarity,
        "uncertainty": uncertainty_norm,
        "has_uncertainty": np.asarray([1.0 if has_uncertainty else 0.0], dtype=np.float32),
        "edge_complexity": edge_complexity,
        "object_complexity": object_complexity,
        "saturation_or_noise": saturation_or_noise,
        "hardness_features": hardness_features,
        "pattern_features": pattern_features,
    }


def cluster_candidate_patterns(pattern_features: np.ndarray, *, max_clusters: int = 10) -> dict[str, np.ndarray]:
    features = np.asarray(pattern_features, dtype=np.float32)
    rows = int(features.shape[0])
    if rows <= 0:
        return {"cluster_ids": np.zeros((0,), dtype=np.int32), "centers": np.zeros((0, 0), dtype=np.float32)}
    normalized = _normalize_feature_matrix(features)
    if rows < 4:
        return {"cluster_ids": np.arange(rows, dtype=np.int32), "centers": normalized.copy()}
    proposed = int(np.clip(round(np.sqrt(float(rows)) * 1.45), 2, max(2, int(max_clusters))))
    cluster_count = int(max(2, min(rows, proposed, int(max_clusters))))
    cluster_ids, centers = _kmeans_assign(normalized, cluster_count, seed=42, max_iterations=30)
    return {"cluster_ids": cluster_ids.astype(np.int32), "centers": centers.astype(np.float32)}


def assign_cluster_quotas(
    cluster_ids: np.ndarray,
    scores: np.ndarray,
    *,
    target_count: int,
    max_cluster_share: float = 0.40,
    min_cluster_size: int = 2,
    structure_scores: np.ndarray | None = None,
) -> dict[int, int]:
    assignments = np.asarray(cluster_ids, dtype=np.int32).reshape(-1)
    priority = np.asarray(scores, dtype=np.float32).reshape(-1)
    structure = None if structure_scores is None else np.asarray(structure_scores, dtype=np.float32).reshape(-1)
    rows = int(assignments.shape[0])
    if rows <= 0 or target_count <= 0:
        return {}

    members: dict[int, list[int]] = {}
    for index, cluster_id in enumerate(assignments.tolist()):
        members.setdefault(int(cluster_id), []).append(int(index))
    stats: list[tuple[int, int, float]] = []
    for cluster_id, idx_list in members.items():
        idx = np.asarray(idx_list, dtype=np.int32)
        cluster_scores = priority[idx]
        structure_bonus = float(np.max(structure[idx])) if structure is not None and structure.size >= int(np.max(idx)) + 1 else 0.0
        hardness = float(np.mean(cluster_scores) * 0.45 + np.max(cluster_scores) * 0.25 + structure_bonus * 0.30)
        stats.append((int(cluster_id), int(idx.size), hardness))
    if not stats:
        return {}
    significant = [item for item in stats if item[1] >= int(max(1, min_cluster_size))] or list(stats)
    quotas: dict[int, int] = {cluster_id: 0 for cluster_id, _size, _hardness in stats}
    if target_count <= len(significant):
        ranked = sorted(significant, key=lambda row: (row[2], row[1]), reverse=True)
        for cluster_id, _size, _hardness in ranked[:target_count]:
            quotas[int(cluster_id)] = 1
        return quotas

    cap = int(max(1, np.floor(target_count * float(np.clip(max_cluster_share, 0.20, 1.00))))) if len(stats) > 1 else target_count
    for cluster_id, _size, _hardness in significant:
        quotas[int(cluster_id)] = 1
    remaining = int(target_count - len(significant))
    while remaining > 0:
        best_cluster = None
        best_score = -1.0
        for cluster_id, size, hardness in stats:
            current = int(quotas.get(cluster_id, 0))
            if current >= size:
                continue
            if len(stats) > 1 and current >= cap:
                continue
            size_norm = float(size) / max(1.0, float(rows))
            pressure = 1.0 / float((current + 1) ** 0.70)
            value = (0.65 * hardness + 0.35 * size_norm) * pressure
            if value > best_score:
                best_score = value
                best_cluster = int(cluster_id)
        if best_cluster is None:
            break
        quotas[best_cluster] = int(quotas.get(best_cluster, 0) + 1)
        remaining -= 1
    if remaining > 0:
        ranked = sorted(stats, key=lambda row: (row[2], row[1]), reverse=True)
        for cluster_id, size, _hardness in ranked:
            if remaining <= 0:
                break
            current = int(quotas.get(cluster_id, 0))
            add = int(min(max(0, size - current), remaining))
            if add <= 0:
                continue
            quotas[cluster_id] = current + add
            remaining -= add
    return quotas


def _non_dominant_cluster_minimum_quotas(
    cluster_ids: np.ndarray,
    *,
    dominant_cluster_id: int | None,
    min_selection: int,
) -> dict[int, int]:
    assignments = np.asarray(cluster_ids, dtype=np.int32).reshape(-1)
    if assignments.size <= 0:
        return {}
    minimum = int(max(1, int(min_selection)))
    quotas: dict[int, int] = {}
    for cluster_id in sorted(set(int(item) for item in assignments.tolist() if int(item) >= 0)):
        if dominant_cluster_id is not None and int(cluster_id) == int(dominant_cluster_id):
            continue
        member_count = int(np.sum(assignments == int(cluster_id)))
        if member_count <= 0:
            continue
        quotas[int(cluster_id)] = int(min(member_count, minimum))
    return quotas


def compute_diversity_gain(candidate_vector: np.ndarray, selected_vectors: np.ndarray) -> float:
    vector = np.asarray(candidate_vector, dtype=np.float32).reshape(-1)
    selected = np.asarray(selected_vectors, dtype=np.float32)
    if selected.size <= 0:
        return 1.0
    if selected.ndim == 1:
        selected = selected.reshape(1, -1)
    distances = np.linalg.norm(selected - vector[None, :], axis=1)
    min_dist = float(np.min(distances)) if distances.size else 0.0
    return float(np.clip(min_dist / np.sqrt(max(1.0, float(vector.size))), 0.0, 1.0))


def select_diverse_hard_cases(
    scores: np.ndarray,
    pattern_features: np.ndarray,
    *,
    target_ratio: float,
    candidate_pool_ratio: float,
    diversity: bool,
    max_cluster_share: float,
    alpha: float,
    beta: float,
    min_cluster_size: int,
    max_pattern_clusters: int,
    normal_reference_mask: np.ndarray | None,
    include_normal_reference_frames: bool,
    max_normal_reference_frames: int,
    non_dominant_cluster_min_selection: int = 2,
) -> dict[str, np.ndarray]:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    patterns = np.asarray(pattern_features, dtype=np.float32)
    rows = int(values.shape[0])
    if rows <= 0:
        return {
            "selected_indexes": np.zeros((0,), dtype=np.int32),
            "cluster_ids_all": np.zeros((0,), dtype=np.int32),
            "candidate_indexes": np.zeros((0,), dtype=np.int32),
            "candidate_cluster_ids": np.zeros((0,), dtype=np.int32),
        }
    ratio = float(np.clip(target_ratio, 0.05, 0.25))
    target_count = int(max(1, min(rows, ceil(rows * ratio))))
    pool_ratio = float(np.clip(candidate_pool_ratio, ratio, 0.70))
    candidate_count = int(max(target_count, min(rows, ceil(rows * pool_ratio))))
    structure_rarity = _centroid_rarity_scores(patterns)
    structural_count = int(max(target_count, min(rows, ceil(rows * 0.18))))
    ranked = np.argsort(-values).astype(np.int32)
    structure_ranked = np.argsort(-structure_rarity).astype(np.int32)
    combined_priority = np.clip(0.55 * values + 0.45 * structure_rarity, 0.0, 1.0)
    combined_ranked = np.argsort(-combined_priority).astype(np.int32)
    candidate_indexes = np.asarray(
        list(
            dict.fromkeys(
                ranked[:candidate_count].tolist()
                + structure_ranked[:structural_count].tolist()
                + combined_ranked[: max(candidate_count, structural_count)].tolist()
            )
        ),
        dtype=np.int32,
    )
    candidate_patterns = patterns[candidate_indexes]
    candidate_structure_rarity = structure_rarity[candidate_indexes]
    clustering = cluster_candidate_patterns(candidate_patterns, max_clusters=max_pattern_clusters)
    candidate_cluster_ids = np.asarray(clustering["cluster_ids"], dtype=np.int32)
    centers = np.asarray(clustering["centers"], dtype=np.float32)
    cluster_ids_all = _assign_full_cluster_ids(patterns, candidate_indexes, candidate_cluster_ids, centers)
    cluster_members_all: dict[int, list[int]] = {}
    for index, cluster_id in enumerate(cluster_ids_all.tolist()):
        cluster_members_all.setdefault(int(cluster_id), []).append(int(index))
    cluster_stats: list[tuple[int, int, float]] = []
    cluster_priority_all = np.clip(0.55 * values + 0.45 * structure_rarity, 0.0, 1.0)
    for cluster_id, member_indexes in cluster_members_all.items():
        idx = np.asarray(member_indexes, dtype=np.int32)
        cluster_priority = cluster_priority_all[idx] if idx.size > 0 else np.zeros((0,), dtype=np.float32)
        cluster_stats.append((int(cluster_id), int(idx.size), float(np.mean(cluster_priority)) if cluster_priority.size else 0.0))
    dominant_cluster_id: int | None = None
    if cluster_stats:
        dominant_cluster_id = int(max(cluster_stats, key=lambda row: (row[1], row[2]))[0])
    mandatory_quota_map = _non_dominant_cluster_minimum_quotas(
        cluster_ids_all,
        dominant_cluster_id=dominant_cluster_id,
        min_selection=int(max(1, int(non_dominant_cluster_min_selection))),
    )
    mandatory_global_indexes: list[int] = []
    for cluster_id, quota in sorted(mandatory_quota_map.items(), key=lambda row: (-row[1], row[0])):
        members = cluster_members_all.get(int(cluster_id), [])
        if not members:
            continue
        ranked_members = sorted(members, key=lambda idx: float(cluster_priority_all[idx]), reverse=True)
        mandatory_global_indexes.extend(ranked_members[: int(quota)])
    mandatory_global_indexes = list(dict.fromkeys(int(idx) for idx in mandatory_global_indexes))
    mandatory_global_set = set(mandatory_global_indexes)
    goal_count = max(int(target_count), int(len(mandatory_global_indexes)))
    candidate_target_count = max(0, int(goal_count - len(mandatory_global_indexes)))
    candidate_keep_mask = np.asarray([int(idx) not in mandatory_global_set for idx in candidate_indexes.tolist()], dtype=bool)
    candidate_indexes = candidate_indexes[candidate_keep_mask]
    candidate_patterns = candidate_patterns[candidate_keep_mask]
    candidate_cluster_ids = candidate_cluster_ids[candidate_keep_mask]
    candidate_structure_rarity = candidate_structure_rarity[candidate_keep_mask]

    if not diversity or candidate_indexes.size <= candidate_target_count:
        selected = np.asarray(candidate_indexes[:candidate_target_count], dtype=np.int32)
    else:
        candidate_scores = np.clip(0.55 * values[candidate_indexes] + 0.45 * candidate_structure_rarity, 0.0, 1.0)
        score_norm = normalize_scores(candidate_scores)
        structure_norm = normalize_scores(candidate_structure_rarity)
        norm_patterns = _normalize_feature_matrix(candidate_patterns)
        structure_weight = float(np.clip(float(beta) * 0.7, 0.15, 0.35))
        quotas = assign_cluster_quotas(
            candidate_cluster_ids,
            candidate_scores,
            target_count=candidate_target_count,
            max_cluster_share=max_cluster_share,
            min_cluster_size=min_cluster_size,
            structure_scores=candidate_structure_rarity,
        )
        cluster_members: dict[int, list[int]] = {}
        for local_idx, cluster_id in enumerate(candidate_cluster_ids.tolist()):
            cluster_members.setdefault(int(cluster_id), []).append(int(local_idx))
        selected_local: list[int] = []
        selected_vectors = np.zeros((0, norm_patterns.shape[1]), dtype=np.float32)
        for cluster_id, quota in sorted(quotas.items(), key=lambda row: row[1], reverse=True):
            if quota <= 0:
                continue
            members = sorted(cluster_members.get(int(cluster_id), []), key=lambda idx: float(candidate_scores[idx]), reverse=True)
            for _ in range(int(quota)):
                best_idx = None
                best_obj = -1.0
                for member in members:
                    if member in selected_local:
                        continue
                    gain = compute_diversity_gain(norm_patterns[member], selected_vectors)
                    obj = float(alpha) * float(score_norm[member]) + float(beta) * float(gain) + float(structure_weight) * float(structure_norm[member])
                    if _is_duplicate_local(member, selected_local, norm_patterns):
                        obj -= 0.20
                    if obj > best_obj:
                        best_obj = obj
                        best_idx = int(member)
                if best_idx is None:
                    break
                selected_local.append(int(best_idx))
                selected_vectors = np.vstack([selected_vectors, norm_patterns[best_idx][None, :]]) if selected_vectors.size else norm_patterns[best_idx][None, :]
                if len(selected_local) >= candidate_target_count:
                    break
            if len(selected_local) >= candidate_target_count:
                break
        selected_local = _greedy_fill(
            score_norm,
            structure_norm,
            norm_patterns,
            candidate_target_count,
            selected_local,
            alpha=alpha,
            beta=beta,
            structure_weight=structure_weight,
        )
        selected = candidate_indexes[np.asarray(selected_local[:candidate_target_count], dtype=np.int32)] if selected_local else candidate_indexes[:candidate_target_count]
        selected = np.asarray(selected, dtype=np.int32)
    if mandatory_global_indexes:
        selected = np.asarray(list(dict.fromkeys([*mandatory_global_indexes, *selected.tolist()])), dtype=np.int32)

    if normal_reference_mask is not None and selected.size > 0:
        normal_mask = np.asarray(normal_reference_mask, dtype=bool).reshape(-1)
        normal_selected = [int(idx) for idx in selected.tolist() if 0 <= int(idx) < normal_mask.size and bool(normal_mask[int(idx)])]
        if not include_normal_reference_frames:
            keep = [int(idx) for idx in selected.tolist() if int(idx) not in set(normal_selected)]
            if keep:
                selected = np.asarray(keep, dtype=np.int32)
        elif len(normal_selected) > int(max_normal_reference_frames):
            normal_sorted = sorted(normal_selected, key=lambda idx: float(values[idx]), reverse=True)
            keep_normal = set(normal_sorted[: int(max_normal_reference_frames)])
            keep = [int(idx) for idx in selected.tolist() if idx not in set(normal_selected) or idx in keep_normal]
            selected = np.asarray(keep, dtype=np.int32)

    if selected.size < goal_count:
        selected_set = set(int(item) for item in selected.tolist())
        for idx in ranked.tolist():
            idx_int = int(idx)
            if idx_int in selected_set:
                continue
            if normal_reference_mask is not None and not include_normal_reference_frames and bool(normal_reference_mask[idx_int]):
                continue
            selected_set.add(idx_int)
            if len(selected_set) >= goal_count:
                break
        selected = np.asarray(sorted(selected_set, key=lambda idx: float(values[idx]), reverse=True)[:goal_count], dtype=np.int32)

    return {
        "selected_indexes": selected,
        "cluster_ids_all": cluster_ids_all.astype(np.int32),
        "candidate_indexes": candidate_indexes.astype(np.int32),
        "candidate_cluster_ids": candidate_cluster_ids.astype(np.int32),
    }


def select_hard_cases(
    scores: np.ndarray,
    features: np.ndarray,
    target_ratio: float = 0.10,
    diversity: bool = True,
    config: LabelingPriorityConfig | None = None,
) -> np.ndarray:
    cfg = config or LabelingPriorityConfig()
    payload = select_diverse_hard_cases(
        np.asarray(scores, dtype=np.float32),
        np.asarray(features, dtype=np.float32),
        target_ratio=float(np.clip(target_ratio if target_ratio is not None else cfg.target_ratio, 0.05, 0.25)),
        candidate_pool_ratio=float(cfg.candidate_pool_ratio),
        diversity=bool(diversity),
        max_cluster_share=float(cfg.max_cluster_share),
        alpha=float(cfg.diversity_alpha),
        beta=float(cfg.diversity_beta),
        min_cluster_size=int(cfg.min_cluster_size),
        max_pattern_clusters=int(cfg.max_pattern_clusters),
        non_dominant_cluster_min_selection=int(cfg.non_dominant_cluster_min_selection),
        normal_reference_mask=None,
        include_normal_reference_frames=True,
        max_normal_reference_frames=int(cfg.max_normal_reference_frames),
    )
    return np.asarray(payload["selected_indexes"], dtype=np.int32)


def explain_frame_recommendation(
    *,
    artifact: float,
    rarity: float,
    uncertainty: float | None,
    edge_complexity: float,
    object_complexity: float,
    saturation_or_noise: float,
    pattern_group: str,
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    if artifact >= 0.68:
        reasons.append("artifact_risk")
    if saturation_or_noise >= 0.65:
        reasons.append("high_noise_or_saturation")
    if rarity >= 0.66:
        reasons.append("rare_frame")
    if edge_complexity >= 0.62:
        reasons.append("high_edge_density")
    if object_complexity >= 0.58:
        reasons.append("complex_objects")
    if uncertainty is not None and uncertainty >= 0.60:
        reasons.append("uncertain_prediction")
    if pattern_group and pattern_group != "mixed_pattern":
        reasons.append(f"pattern:{pattern_group}")
    if not reasons:
        reasons.append("normal_reference" if artifact < 0.45 and rarity < 0.52 else "borderline_hard_case")
    if "normal_reference" in reasons:
        category = "normal_reference"
    elif artifact >= 0.62 and rarity >= 0.55:
        category = "artifact_rare_hard_case"
    elif artifact >= 0.62:
        category = "artifact_hard_case"
    elif rarity >= 0.62:
        category = "rare_pattern_hard_case"
    elif uncertainty is not None and uncertainty >= 0.60:
        category = "uncertain_hard_case"
    else:
        category = "complex_hard_case"
    return category, tuple(reasons)


def compute_labeling_priority_for_paths(
    source_folder: Path,
    *,
    prediction_path_by_key: dict[str, str] | None = None,
    config: LabelingPriorityConfig | None = None,
) -> LabelingPriorityResult:
    cfg = config or LabelingPriorityConfig()
    paths = discover_image_paths(Path(source_folder))
    if not paths:
        return LabelingPriorityResult(total_discovered=0, selected_count=0, frames=())
    rows: list[_FrameRow] = []
    prediction_by_key = {str(key).strip(): str(value).strip() for key, value in (prediction_path_by_key or {}).items() if str(key).strip() and str(value).strip()}
    root = Path(source_folder)
    for path in paths:
        try:
            image = np.asarray(load_grayscale_image(path), dtype=np.uint8)
        except Exception:
            continue
        if image.ndim != 2 or image.size <= 0:
            continue
        frame_key = _frame_key_from_path(path, root)
        prediction = None
        prediction_path = prediction_by_key.get(frame_key)
        if prediction_path:
            try:
                prediction = np.asarray(load_grayscale_image(Path(prediction_path)))
            except Exception:
                prediction = None
        rows.append(_FrameRow(path=path, frame_key=frame_key, display_name=path.stem, image=image, prediction=prediction))
    if not rows:
        return LabelingPriorityResult(total_discovered=len(paths), selected_count=0, frames=())

    payload = compute_labeling_priority_scores([row.image for row in rows], [row.prediction for row in rows], config=cfg)
    score = np.asarray(payload["score"], dtype=np.float32)
    pattern_features = np.asarray(payload["pattern_features"], dtype=np.float32)
    artifact = np.asarray(payload["artifact"], dtype=np.float32)
    rarity = np.asarray(payload["rarity"], dtype=np.float32)
    uncertainty = np.asarray(payload["uncertainty"], dtype=np.float32) if bool(payload["has_uncertainty"][0]) else np.full((len(rows),), np.nan, dtype=np.float32)
    edge_complexity = np.asarray(payload["edge_complexity"], dtype=np.float32)
    object_complexity = np.asarray(payload["object_complexity"], dtype=np.float32)
    saturation_or_noise = np.asarray(payload["saturation_or_noise"], dtype=np.float32)

    normal_mask = np.asarray([
        _is_normal_reference(float(artifact[idx]), float(rarity[idx]), float(edge_complexity[idx]), float(object_complexity[idx]), uncertainty[idx])
        for idx in range(len(rows))
    ], dtype=bool)

    selected_payload = select_diverse_hard_cases(
        score,
        pattern_features,
        target_ratio=float(cfg.target_ratio),
        candidate_pool_ratio=float(cfg.candidate_pool_ratio),
        diversity=bool(cfg.enable_diversity),
        max_cluster_share=float(cfg.max_cluster_share),
        alpha=float(cfg.diversity_alpha),
        beta=float(cfg.diversity_beta),
        min_cluster_size=int(cfg.min_cluster_size),
        max_pattern_clusters=int(cfg.max_pattern_clusters),
        non_dominant_cluster_min_selection=int(cfg.non_dominant_cluster_min_selection),
        normal_reference_mask=normal_mask,
        include_normal_reference_frames=bool(cfg.include_normal_reference_frames),
        max_normal_reference_frames=int(cfg.max_normal_reference_frames),
    )
    selected = np.asarray(selected_payload["selected_indexes"], dtype=np.int32)
    selected_set = {int(index) for index in selected.tolist()}
    cluster_ids_all = np.asarray(selected_payload["cluster_ids_all"], dtype=np.int32)
    if cluster_ids_all.size != len(rows):
        cluster_ids_all = np.zeros((len(rows),), dtype=np.int32)
    pattern_groups = _pattern_groups_from_clusters(pattern_features, cluster_ids_all)
    rank_map = {int(index): rank + 1 for rank, index in enumerate(sorted(selected_set, key=lambda idx: float(score[idx]), reverse=True))}

    frames: list[LabelingPriorityFrame] = []
    for index, row in enumerate(rows):
        cluster_id = int(cluster_ids_all[index]) if 0 <= index < cluster_ids_all.size else 0
        pattern_group = str(pattern_groups.get(cluster_id, "mixed_pattern"))
        uncertainty_value = None if not np.isfinite(float(uncertainty[index])) else float(np.clip(float(uncertainty[index]), 0.0, 1.0))
        category, reasons = explain_frame_recommendation(
            artifact=float(artifact[index]),
            rarity=float(rarity[index]),
            uncertainty=uncertainty_value,
            edge_complexity=float(edge_complexity[index]),
            object_complexity=float(object_complexity[index]),
            saturation_or_noise=float(saturation_or_noise[index]),
            pattern_group=pattern_group,
        )
        recommended = int(index) in selected_set
        frames.append(
            LabelingPriorityFrame(
                frame_key=row.frame_key,
                display_name=row.display_name,
                original_path=str(row.path),
                priority_score=float(np.clip(score[index], 0.0, 1.0)),
                category=category,
                reasons=reasons,
                recommended=bool(recommended),
                rank=int(rank_map.get(int(index), 0)),
                pattern_cluster_id=cluster_id,
                pattern_group=pattern_group,
                artifact_score=float(artifact[index]),
                rarity_score=float(rarity[index]),
                uncertainty_score=uncertainty_value,
                edge_complexity_score=float(edge_complexity[index]),
                object_complexity_score=float(object_complexity[index]),
                saturation_or_noise_score=float(saturation_or_noise[index]),
                metadata={
                    "artifact": float(artifact[index]),
                    "rarity": float(rarity[index]),
                    "uncertainty": None if uncertainty_value is None else float(uncertainty_value),
                    "edge_complexity": float(edge_complexity[index]),
                    "object_complexity": float(object_complexity[index]),
                    "saturation_or_noise": float(saturation_or_noise[index]),
                    "pattern_cluster_id": int(cluster_id),
                    "pattern_group": pattern_group,
                },
            )
        )

    return LabelingPriorityResult(total_discovered=len(paths), selected_count=int(len(selected_set)), frames=tuple(frames))


def _to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        if array.ndim == 3:
            array = array[..., 0]
        else:
            flat = np.asarray(array).reshape((-1,))
            side = int(max(1, round(np.sqrt(flat.size))))
            array = np.resize(flat, (side, side))
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        finite = np.where(np.isfinite(array), array, 0.0)
        max_value = float(np.max(finite)) if finite.size else 0.0
        scaled = finite * (255.0 if max_value <= 1.0 + 1e-3 else 1.0)
        return np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    return np.clip(array.astype(np.float32), 0.0, 255.0).astype(np.uint8)


def _prediction_uncertainty(prediction: np.ndarray | None, frame_norm: np.ndarray, grad: np.ndarray) -> float | None:
    if prediction is None:
        return None
    pred = np.asarray(prediction)
    if pred.ndim == 3:
        pred = pred[..., 0]
    if pred.ndim != 2 or pred.size <= 0:
        return None
    if pred.shape != frame_norm.shape:
        pred = _resize_nearest(pred, frame_norm.shape)
    pred_f = pred.astype(np.float32)
    if np.issubdtype(pred.dtype, np.floating):
        max_val = float(np.max(np.where(np.isfinite(pred_f), pred_f, 0.0))) if pred_f.size else 0.0
        p = np.clip(pred_f, 0.0, 1.0) if max_val <= 1.0 + 1e-3 else np.clip(pred_f / 255.0, 0.0, 1.0)
    else:
        p = np.clip(pred_f / 255.0, 0.0, 1.0)
    uncertainty_map = 1.0 - 2.0 * np.abs(p - 0.5)
    active_mask = (p > 0.03) & (p < 0.97)
    edge_mask = grad >= np.percentile(grad, 70.0) if grad.size else np.zeros_like(active_mask)
    mask = active_mask | edge_mask
    if np.mean(mask) < 0.03:
        mask = active_mask
    if np.mean(mask) < 0.02:
        mask = np.ones_like(active_mask, dtype=bool)
    return float(np.mean(uncertainty_map[mask]))


def _resize_nearest(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    if target_h <= 0 or target_w <= 0:
        return np.zeros((1, 1), dtype=np.float32)
    src = np.asarray(image)
    src_h, src_w = src.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return np.zeros((target_h, target_w), dtype=np.float32)
    row_index = np.linspace(0, max(0, src_h - 1), num=target_h).astype(np.int32)
    col_index = np.linspace(0, max(0, src_w - 1), num=target_w).astype(np.int32)
    return src[np.ix_(row_index, col_index)]


def _downsample_bool(binary: np.ndarray, *, max_side: int) -> np.ndarray:
    h, w = binary.shape
    if max(h, w) <= int(max_side):
        return binary.astype(bool)
    step_h = max(1, int(np.ceil(h / float(max_side))))
    step_w = max(1, int(np.ceil(w / float(max_side))))
    return binary[::step_h, ::step_w].astype(bool)


def _component_stats(binary: np.ndarray) -> tuple[float, float, float, float, float]:
    if binary.size <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    grid = binary.astype(np.uint8)
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    areas: list[int] = []
    for y in range(h):
        for x in range(w):
            if grid[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or nx < 0 or ny >= h or nx >= w:
                        continue
                    if visited[ny, nx] or grid[ny, nx] == 0:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
            if area > 0:
                areas.append(int(area))
    if not areas:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    areas_np = np.asarray(areas, dtype=np.float32)
    image_area = max(1.0, float(grid.size))
    norm_areas = areas_np / image_area
    mean_area = float(np.mean(norm_areas))
    area_var = float(np.var(norm_areas))
    small_ratio = float(np.mean(areas_np <= np.percentile(areas_np, 35.0)))
    merged_ratio = float(np.mean(norm_areas >= np.percentile(norm_areas, 85.0)))
    return float(len(areas)), mean_area, area_var, small_ratio, merged_ratio


def _histogram_bins(image: np.ndarray, *, bins: int) -> np.ndarray:
    hist, _ = np.histogram(image.ravel(), bins=max(2, int(bins)), range=(0, 256))
    hist = hist.astype(np.float32)
    total = float(np.sum(hist))
    return hist / total if total > 0.0 else np.zeros((max(2, int(bins)),), dtype=np.float32)


def _entropy_0_1(image: np.ndarray) -> float:
    hist = np.bincount(np.asarray(image, dtype=np.uint8).ravel(), minlength=256).astype(np.float64)
    hist /= max(1.0, float(hist.sum()))
    nz = hist > 0.0
    value = float(-(hist[nz] * np.log2(hist[nz])).sum())
    return float(np.clip(value / 8.0, 0.0, 1.0))


def _quadrant_std(binary: np.ndarray) -> float:
    h, w = binary.shape
    if h <= 1 or w <= 1:
        return 0.0
    h_mid = max(1, h // 2)
    w_mid = max(1, w // 2)
    quads = [binary[:h_mid, :w_mid], binary[:h_mid, w_mid:], binary[h_mid:, :w_mid], binary[h_mid:, w_mid:]]
    values = np.asarray([float(np.mean(quad)) if quad.size else 0.0 for quad in quads], dtype=np.float32)
    return float(np.std(values))


def _mean_filter_3x3(image: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(image, dtype=np.float32), 1, mode="edge")
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0


def _otsu_threshold_uint8(image: np.ndarray) -> float:
    histogram = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    total = float(histogram.sum())
    if total <= 0:
        return 0.5
    prob = histogram / total
    omega = np.cumsum(prob)
    mean = np.cumsum(prob * np.arange(256, dtype=np.float64))
    mean_total = float(mean[-1])
    denom = np.where(omega * (1.0 - omega) <= EPS, np.nan, omega * (1.0 - omega))
    variance = ((mean_total * omega - mean) ** 2) / denom
    idx = int(np.nanargmax(variance)) if np.isfinite(variance).any() else 127
    return float(idx / 255.0)


def _rarity_scores(features: np.ndarray, *, neighbors: int = 5) -> np.ndarray:
    rows = int(features.shape[0])
    if rows <= 0:
        return np.zeros((0,), dtype=np.float32)
    normalized = _normalize_feature_matrix(features)
    if rows == 1:
        return np.zeros((1,), dtype=np.float32)
    k = int(max(1, min(int(neighbors), rows - 1)))
    if rows > 1200:
        centroid = normalized.mean(axis=0)
        return normalize_scores(np.linalg.norm(normalized - centroid[None, :], axis=1))
    chunk = 128
    distances = np.empty((rows,), dtype=np.float32)
    for start in range(0, rows, chunk):
        end = min(rows, start + chunk)
        block = normalized[start:end]
        dist = np.linalg.norm(block[:, None, :] - normalized[None, :, :], axis=2)
        for local in range(dist.shape[0]):
            dist[local, start + local] = np.inf
        distances[start:end] = np.mean(np.partition(dist, k, axis=1)[:, :k], axis=1)
    return normalize_scores(distances)


def _centroid_rarity_scores(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    rows = int(matrix.shape[0])
    if rows <= 0:
        return np.zeros((0,), dtype=np.float32)
    normalized = _normalize_feature_matrix(matrix)
    if rows == 1:
        return np.zeros((1,), dtype=np.float32)
    centroid = normalized.mean(axis=0)
    return normalize_scores(np.linalg.norm(normalized - centroid[None, :], axis=1))


def _normalize_feature_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    mins = matrix.min(axis=0)
    spans = np.maximum(matrix.max(axis=0) - mins, EPS)
    return (matrix - mins) / spans


def _kmeans_assign(normalized: np.ndarray, cluster_count: int, *, seed: int, max_iterations: int) -> tuple[np.ndarray, np.ndarray]:
    rows = int(normalized.shape[0])
    cols = int(normalized.shape[1]) if normalized.ndim == 2 else 1
    if rows <= 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, cols), dtype=np.float32)
    if cluster_count <= 1 or rows == 1:
        return np.zeros((rows,), dtype=np.int32), np.asarray([normalized.mean(axis=0)], dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(rows, size=min(cluster_count, rows), replace=False).astype(np.int32)
    centers = normalized[indices].copy()
    assignments = np.zeros((rows,), dtype=np.int32)
    for _ in range(max(1, int(max_iterations))):
        dist = np.linalg.norm(normalized[:, None, :] - centers[None, :, :], axis=2)
        next_assignments = np.argmin(dist, axis=1).astype(np.int32)
        if np.array_equal(assignments, next_assignments):
            break
        assignments = next_assignments
        for cluster_id in range(int(centers.shape[0])):
            members = normalized[assignments == cluster_id]
            centers[cluster_id] = members.mean(axis=0) if members.size > 0 else normalized[int(rng.integers(0, rows))]
    return assignments, centers


def _assign_full_cluster_ids(all_pattern_features: np.ndarray, candidate_indexes: np.ndarray, candidate_cluster_ids: np.ndarray, candidate_centers: np.ndarray) -> np.ndarray:
    rows = int(np.asarray(all_pattern_features).shape[0])
    full = np.full((rows,), -1, dtype=np.int32)
    if rows <= 0:
        return full
    for local_idx, global_idx in enumerate(candidate_indexes.tolist()):
        if 0 <= int(global_idx) < rows and 0 <= int(local_idx) < candidate_cluster_ids.size:
            full[int(global_idx)] = int(candidate_cluster_ids[int(local_idx)])
    if candidate_centers.size <= 0:
        full[full < 0] = 0
        return full
    normalized_all = _normalize_feature_matrix(np.asarray(all_pattern_features, dtype=np.float32))
    centers = np.asarray(candidate_centers, dtype=np.float32)
    for idx in range(rows):
        if full[idx] >= 0:
            continue
        distances = np.linalg.norm(centers - normalized_all[idx][None, :], axis=1)
        full[idx] = int(np.argmin(distances)) if distances.size else 0
    return full


def _pattern_groups_from_clusters(pattern_features: np.ndarray, cluster_ids: np.ndarray) -> dict[int, str]:
    groups: dict[int, str] = {}
    if pattern_features.size <= 0 or cluster_ids.size <= 0:
        return groups
    normalized = _normalize_feature_matrix(np.asarray(pattern_features, dtype=np.float32))
    unique_ids = sorted(set(int(item) for item in cluster_ids.tolist() if int(item) >= 0))
    for cluster_id in unique_ids:
        idx = np.where(cluster_ids == int(cluster_id))[0]
        if idx.size <= 0:
            groups[int(cluster_id)] = "mixed_pattern"
            continue
        centroid = np.mean(normalized[idx], axis=0)
        edge_density = float(centroid[8]) if centroid.size > 8 else 0.0
        component_count = float(centroid[10]) if centroid.size > 10 else 0.0
        mean_area = float(centroid[11]) if centroid.size > 11 else 0.0
        small_ratio = float(centroid[13]) if centroid.size > 13 else 0.0
        merged_ratio = float(centroid[14]) if centroid.size > 14 else 0.0
        active_ratio = float(centroid[15]) if centroid.size > 15 else 0.0
        entropy = float(centroid[18]) if centroid.size > 18 else 0.0
        q_std = float(centroid[19]) if centroid.size > 19 else 0.0
        saturation = float(centroid[20]) if centroid.size > 20 else 0.0
        stripe = float(centroid[21]) if centroid.size > 21 else 0.0
        noise = float(centroid[22]) if centroid.size > 22 else 0.0
        row_periodicity = float(centroid[-6]) if centroid.size >= 33 else 0.0
        col_periodicity = float(centroid[-5]) if centroid.size >= 33 else 0.0
        if active_ratio < 0.10 and edge_density < 0.20:
            label = "sparse_or_empty"
        elif component_count > 0.58 and small_ratio > 0.55:
            label = "many_small_objects"
        elif merged_ratio > 0.62 and component_count < 0.45:
            label = "dense_merged_objects"
        elif mean_area > 0.58 and component_count < 0.40:
            label = "large_objects"
        elif edge_density > 0.62 and q_std > 0.55:
            label = "complex_geometry"
        elif row_periodicity > 0.58 and col_periodicity < 0.42:
            label = "horizontal_layer_pattern"
        elif col_periodicity > 0.58 and row_periodicity < 0.42:
            label = "vertical_stripe_pattern"
        elif stripe > 0.62:
            label = "stripe_artifact_pattern"
        elif saturation > 0.62:
            label = "exposure_saturation_pattern"
        elif noise > 0.60 and entropy > 0.50:
            label = "noisy_texture_pattern"
        else:
            label = "mixed_pattern"
        groups[int(cluster_id)] = label
    return groups


def _is_duplicate_local(local_index: int, selected_local: list[int], features: np.ndarray) -> bool:
    if not selected_local:
        return False
    candidate = features[int(local_index)]
    for index in selected_local:
        if float(np.linalg.norm(candidate - features[int(index)])) <= 0.060:
            return True
    return False


def _greedy_fill(
    score_norm: np.ndarray,
    structure_norm: np.ndarray,
    patterns: np.ndarray,
    target_count: int,
    selected: list[int],
    *,
    alpha: float,
    beta: float,
    structure_weight: float,
) -> list[int]:
    result = list(int(item) for item in selected)
    if not result and score_norm.size > 0:
        result.append(int(np.argmax(score_norm)))
    while len(result) < int(target_count):
        vectors = patterns[np.asarray(result, dtype=np.int32)] if result else np.zeros((0, patterns.shape[1]), dtype=np.float32)
        best_idx = None
        best_value = -1.0
        for idx in range(int(score_norm.shape[0])):
            if idx in result:
                continue
            gain = compute_diversity_gain(patterns[idx], vectors)
            structure_bonus = float(structure_norm[idx]) if structure_norm.size > idx else 0.0
            value = float(alpha) * float(score_norm[idx]) + float(beta) * float(gain) + float(structure_weight) * float(structure_bonus)
            if _is_duplicate_local(idx, result, patterns):
                value -= 0.20
            if value > best_value:
                best_value = value
                best_idx = int(idx)
        if best_idx is None:
            break
        result.append(int(best_idx))
    return result[: int(target_count)]


def _is_normal_reference(artifact: float, rarity: float, edge_complexity: float, object_complexity: float, uncertainty: float | np.floating | None) -> bool:
    uncertainty_value = float(uncertainty) if uncertainty is not None and np.isfinite(float(uncertainty)) else 0.0
    return (
        float(artifact) < 0.40
        and float(rarity) < 0.46
        and float(edge_complexity) < 0.48
        and float(object_complexity) < 0.45
        and uncertainty_value < 0.50
    )


def _frame_key_from_path(path: Path, root: Path) -> str:
    path_obj = Path(path)
    try:
        relative = path_obj.resolve().relative_to(Path(root).resolve())
        return str(relative.with_suffix("")).replace("\\", "/")
    except Exception:
        return str(path_obj.with_suffix("").name)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    return _to_uint8(frame).astype(np.float32) / 255.0


def extract_object_features(frame: np.ndarray) -> dict[str, float]:
    normalized = normalize_frame(frame)
    binary = normalized > _otsu_threshold_uint8(_to_uint8(frame))
    reduced = _downsample_bool(binary, max_side=96)
    component_count, mean_area, area_var, small_ratio, merged_ratio = _component_stats(reduced)
    area_ratio = float(np.mean(binary))
    total_object_area = float(area_ratio)
    object_density = float(component_count / max(1.0, reduced.size / 1024.0))
    object_area_std = float(np.sqrt(max(0.0, area_var)))
    small_object_count = float(component_count * small_ratio)
    large_object_count = float(component_count * merged_ratio)
    contour_complexity = float(np.mean(np.abs(np.diff(reduced.astype(np.int8), axis=0)))) if reduced.shape[0] > 1 else 0.0
    compactness_score = float(np.clip(1.0 - mean_area * 4.0, 0.0, 1.0))
    thin_structure_score = float(np.clip(contour_complexity * 1.5, 0.0, 1.0))
    return {
        "object_count": float(component_count),
        "area_ratio": area_ratio,
        "total_object_area": total_object_area,
        "mean_object_area": float(mean_area),
        "object_area_std": object_area_std,
        "small_object_count": small_object_count,
        "large_object_count": large_object_count,
        "object_density": object_density,
        "contour_complexity": contour_complexity,
        "compactness_score": compactness_score,
        "merged_object_score": float(merged_ratio),
        "thin_structure_score": thin_structure_score,
    }


def compute_artifact_score(feature_row: dict[str, float | np.ndarray]) -> float:
    return float(np.clip(float(feature_row.get("artifact_raw", 0.0)), 0.0, 1.0))


def compute_rarity_score(hardness_features: np.ndarray) -> np.ndarray:
    return _rarity_scores(np.asarray(hardness_features, dtype=np.float32))


def compute_object_complexity_score(feature_row: dict[str, float | np.ndarray]) -> float:
    return float(np.clip(float(feature_row.get("object_complexity_raw", 0.0)), 0.0, 1.0))


def cluster_pattern_groups(pattern_features: np.ndarray, *, max_pattern_groups: int = 10) -> dict[str, np.ndarray]:
    return cluster_candidate_patterns(pattern_features, max_clusters=max_pattern_groups)


def explain_recommendation(
    *,
    artifact: float,
    rarity: float,
    uncertainty: float | None,
    edge_complexity: float,
    object_complexity: float,
    saturation_or_noise: float,
    pattern_group: str,
) -> tuple[str, tuple[str, ...]]:
    return explain_frame_recommendation(
        artifact=artifact,
        rarity=rarity,
        uncertainty=uncertainty,
        edge_complexity=edge_complexity,
        object_complexity=object_complexity,
        saturation_or_noise=saturation_or_noise,
        pattern_group=pattern_group,
    )


def _config_to_internal(config: PrimaryLabelingConfig | None) -> LabelingPriorityConfig:
    cfg = config or PrimaryLabelingConfig()
    return LabelingPriorityConfig(
        target_ratio=float(cfg.target_ratio),
        candidate_pool_ratio=float(cfg.candidate_pool_ratio),
        enable_diversity=bool(cfg.enable_diversity_filter),
        include_normal_reference_frames=bool(cfg.include_normal_reference_frames),
        max_normal_reference_frames=int(cfg.max_normal_reference_frames),
        max_cluster_share=float(cfg.max_group_share),
        diversity_alpha=float(cfg.alpha),
        diversity_beta=float(cfg.beta),
        min_cluster_size=int(cfg.min_group_size),
        max_pattern_clusters=int(cfg.max_pattern_groups),
        non_dominant_cluster_min_selection=int(cfg.non_dominant_cluster_min_selection),
    )


def _cache_key_for_selection(
    source_folder: Path,
    *,
    prediction_path_by_key: dict[str, str] | None,
    config: PrimaryLabelingConfig | None,
) -> tuple[str, int, int, float, float, bool, bool, int, float, int]:
    source = Path(source_folder)
    source_text = str(source.resolve()) if source.exists() else str(source)
    discovered = discover_image_paths(source)
    count = len(discovered)
    signature = 0
    for path in discovered[:2000]:
        try:
            stat = path.stat()
            signature ^= int(stat.st_mtime_ns) ^ int(stat.st_size)
        except Exception:
            continue
    pred_count = int(len(prediction_path_by_key or {}))
    cfg = config or PrimaryLabelingConfig()
    return (
        source_text,
        int(count),
        int(signature),
        float(cfg.target_ratio),
        float(cfg.candidate_pool_ratio),
        bool(cfg.enable_diversity_filter),
        bool(cfg.include_normal_reference_frames),
        int(cfg.max_normal_reference_frames),
        float(cfg.max_group_share),
        int(cfg.non_dominant_cluster_min_selection),
    )


def build_primary_labeling_selection(
    source_folder: Path,
    *,
    prediction_path_by_key: dict[str, str] | None = None,
    config: PrimaryLabelingConfig | None = None,
    use_cache: bool = True,
) -> PrimaryLabelingSelectionResult:
    result = _compute_primary_labeling_cached_result(
        source_folder,
        prediction_path_by_key=prediction_path_by_key,
        config=config,
        use_cache=use_cache,
    )
    frames = tuple(
        FrameSelectionResult(
            frame_key=item.frame_key,
            display_name=item.display_name,
            priority_score=float(item.priority_score),
            selected_for_primary_labeling=bool(item.recommended),
            pattern_group_id=int(item.pattern_cluster_id),
            pattern_group_label=str(item.pattern_group),
            reasons=tuple(item.reasons),
        )
        for item in result.frames
    )
    return PrimaryLabelingSelectionResult(
        total_discovered=int(result.total_discovered),
        selected_count=int(result.selected_count),
        frames=frames,
    )


def cached_primary_labeling_priority_result(
    source_folder: Path,
    *,
    prediction_path_by_key: dict[str, str] | None = None,
    config: PrimaryLabelingConfig | None = None,
    use_cache: bool = True,
) -> LabelingPriorityResult:
    return _compute_primary_labeling_cached_result(
        source_folder,
        prediction_path_by_key=prediction_path_by_key,
        config=config,
        use_cache=use_cache,
    )


def _compute_primary_labeling_cached_result(
    source_folder: Path,
    *,
    prediction_path_by_key: dict[str, str] | None,
    config: PrimaryLabelingConfig | None,
    use_cache: bool,
) -> LabelingPriorityResult:
    cache_key = _cache_key_for_selection(
        source_folder,
        prediction_path_by_key=prediction_path_by_key,
        config=config,
    )
    internal_cfg = _config_to_internal(config)
    if use_cache:
        cached = _PRIMARY_LABELING_CACHE.get(cache_key)
        if cached is None:
            cached = compute_labeling_priority_for_paths(
                source_folder,
                prediction_path_by_key=prediction_path_by_key,
                config=internal_cfg,
            )
            _PRIMARY_LABELING_CACHE[cache_key] = cached
        result = cached
    else:
        result = compute_labeling_priority_for_paths(
            source_folder,
            prediction_path_by_key=prediction_path_by_key,
            config=internal_cfg,
        )
    return result


__all__ = [
    "PrimaryLabelingConfig",
    "FrameFeatureSet",
    "FrameSelectionResult",
    "PrimaryLabelingSelectionResult",
    "normalize_frame",
    "extract_object_features",
    "compute_artifact_score",
    "compute_rarity_score",
    "compute_object_complexity_score",
    "cluster_pattern_groups",
    "explain_recommendation",
    "build_primary_labeling_selection",
    "cached_primary_labeling_priority_result",
    "LabelingPriorityConfig",
    "LabelingPriorityFrame",
    "LabelingPriorityResult",
    "compute_labeling_priority_for_paths",
    "extract_frame_features",
    "extract_pattern_features",
    "compute_labeling_priority_scores",
    "cluster_candidate_patterns",
    "select_hard_cases",
    "select_diverse_hard_cases",
    "assign_cluster_quotas",
    "compute_diversity_gain",
    "normalize_scores",
    "explain_frame_recommendation",
]
