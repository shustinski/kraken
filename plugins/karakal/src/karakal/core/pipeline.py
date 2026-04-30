"""Pipeline helpers for bootstrap sampling and corrective recommendations."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .domain import FrameRecord
from .repository import extract_original_frame_features, load_grayscale_image

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True, slots=True)
class PipelineTaskCandidate:
    """One frame candidate that can be converted into one management task."""

    frame_key: str
    display_name: str
    original_path: str
    priority_score: float
    reason: str
    metadata: dict[str, float | int | str | bool | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BootstrapSamplingResult:
    """Result of semi-automatic initial sampling over raw original frames."""

    total_discovered: int
    total_quality_passed: int
    total_unique: int
    selected: tuple[PipelineTaskCandidate, ...]


@dataclass(frozen=True, slots=True)
class _BootstrapFrameFeatures:
    path: Path
    frame_key: str
    display_name: str
    quality_score: float
    vector: tuple[float, ...]
    fingerprint: tuple[int, ...]


def discover_image_paths(root: Path) -> tuple[Path, ...]:
    """Find all supported image files under the provided directory."""

    root_path = Path(root)
    if root_path.is_file():
        return (root_path,) if root_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else ()
    paths = [
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    paths.sort(key=lambda item: item.as_posix().lower())
    return tuple(paths)


def generate_bootstrap_sample(
    frame_paths: Iterable[Path],
    *,
    root: Path,
    target_size: int,
    seed: int = 42,
    uniform_ratio: float = 0.70,
) -> BootstrapSamplingResult:
    """Build first-iteration labeling candidates using uniform+diversity strategy."""

    paths = tuple(Path(item) for item in frame_paths)
    total_discovered = len(paths)
    if total_discovered <= 0 or target_size <= 0:
        return BootstrapSamplingResult(
            total_discovered=total_discovered,
            total_quality_passed=0,
            total_unique=0,
            selected=(),
        )

    quality_passed = [_extract_frame_features(path, root) for path in paths]
    quality_passed = [item for item in quality_passed if item is not None]
    total_quality_passed = len(quality_passed)
    if total_quality_passed <= 0:
        return BootstrapSamplingResult(
            total_discovered=total_discovered,
            total_quality_passed=0,
            total_unique=0,
            selected=(),
        )

    unique_frames = _deduplicate_by_fingerprint(quality_passed)
    total_unique = len(unique_frames)
    if total_unique <= 0:
        return BootstrapSamplingResult(
            total_discovered=total_discovered,
            total_quality_passed=total_quality_passed,
            total_unique=0,
            selected=(),
        )

    rng = np.random.default_rng(int(seed))
    capped_target = int(max(1, min(int(target_size), total_unique)))
    uniform_count = int(round(capped_target * float(np.clip(uniform_ratio, 0.0, 1.0))))
    diversity_count = max(0, capped_target - uniform_count)

    uniform_selected = _uniform_sample(unique_frames, uniform_count)
    diversity_selected = _diversity_sample(unique_frames, diversity_count, rng)

    merged: list[_BootstrapFrameFeatures] = []
    seen_keys: set[str] = set()
    for candidate in uniform_selected + diversity_selected:
        if candidate.frame_key in seen_keys:
            continue
        seen_keys.add(candidate.frame_key)
        merged.append(candidate)
    if len(merged) < capped_target:
        remaining = [item for item in unique_frames if item.frame_key not in seen_keys]
        if remaining:
            rng.shuffle(remaining)
            for candidate in remaining[: capped_target - len(merged)]:
                merged.append(candidate)

    tasks = tuple(
        PipelineTaskCandidate(
            frame_key=item.frame_key,
            display_name=item.display_name,
            original_path=str(item.path),
            priority_score=float(item.quality_score),
            reason="bootstrap_uniform_diversity",
            metadata={
                "quality_score": float(item.quality_score),
                "brightness": float(item.vector[0]),
                "contrast": float(item.vector[1]),
                "entropy": float(item.vector[2]),
                "edge_density": float(item.vector[3]),
            },
        )
        for item in merged
    )
    return BootstrapSamplingResult(
        total_discovered=total_discovered,
        total_quality_passed=total_quality_passed,
        total_unique=total_unique,
        selected=tasks,
    )


def generate_bootstrap_clustered_sample(
    frame_paths: Iterable[Path],
    *,
    root: Path,
    target_size: int,
    seed: int = 42,
    cluster_count: int | None = None,
    max_iterations: int = 30,
    prefer_anomalies: bool = False,
    exclude_normal_clusters: bool = False,
) -> BootstrapSamplingResult:
    """Build first-iteration labeling candidates using unsupervised clustering."""

    paths = tuple(Path(item) for item in frame_paths)
    total_discovered = len(paths)
    if total_discovered <= 0 or target_size <= 0:
        return BootstrapSamplingResult(
            total_discovered=total_discovered,
            total_quality_passed=0,
            total_unique=0,
            selected=(),
        )

    quality_passed = [_extract_frame_features(path, root) for path in paths]
    quality_passed = [item for item in quality_passed if item is not None]
    total_quality_passed = len(quality_passed)
    if total_quality_passed <= 0:
        return BootstrapSamplingResult(
            total_discovered=total_discovered,
            total_quality_passed=0,
            total_unique=0,
            selected=(),
        )

    unique_frames = _deduplicate_by_fingerprint(quality_passed)
    total_unique = len(unique_frames)
    if total_unique <= 0:
        return BootstrapSamplingResult(
            total_discovered=total_discovered,
            total_quality_passed=total_quality_passed,
            total_unique=0,
            selected=(),
        )

    capped_target = int(max(1, min(int(target_size), total_unique)))
    features = np.asarray([item.vector for item in unique_frames], dtype=np.float32)
    normalized = _normalize_feature_matrix(features)

    auto_cluster_count = int(np.clip(round(np.sqrt(float(capped_target)) * 2.0), 2, capped_target))
    requested_clusters = int(cluster_count) if cluster_count is not None else auto_cluster_count
    cluster_total = int(max(1, min(requested_clusters, capped_target, total_unique)))
    assignments, centers = _kmeans_assign(normalized, cluster_total, seed=int(seed), max_iterations=max_iterations)

    quality_values = np.asarray([item.quality_score for item in unique_frames], dtype=np.float32)
    if quality_values.size > 0:
        min_quality = float(np.min(quality_values))
        max_quality = float(np.max(quality_values))
        quality_span = max(max_quality - min_quality, 1e-8)
        quality_norm = (quality_values - min_quality) / quality_span
    else:
        quality_norm = np.zeros((len(unique_frames),), dtype=np.float32)
    anomaly_norm = np.asarray(
        [
            _bootstrap_anomaly_score_from_vector(item.vector)
            for item in unique_frames
        ],
        dtype=np.float32,
    )

    cluster_members: list[list[int]] = [[] for _ in range(cluster_total)]
    for index, cluster_id in enumerate(assignments.tolist()):
        cluster_members[int(cluster_id)].append(int(index))

    cluster_labels: dict[int, str] = {}
    for cluster_id, members in enumerate(cluster_members):
        if not members:
            cluster_labels[int(cluster_id)] = "normal_clean"
            continue
        cluster_anomaly = float(np.mean(anomaly_norm[np.asarray(members, dtype=np.int32)]))
        mean_vector = np.mean(
            np.asarray([unique_frames[int(index)].vector for index in members], dtype=np.float32),
            axis=0,
        )
        cluster_labels[int(cluster_id)] = _cluster_label_from_profile(mean_vector, cluster_anomaly)
    allowed_clusters = {
        int(cluster_id)
        for cluster_id in range(cluster_total)
        if not exclude_normal_clusters or cluster_labels.get(int(cluster_id), "normal_clean") != "normal_clean"
    }
    if not allowed_clusters:
        allowed_clusters = set(range(cluster_total))
    frame_fingerprints = [item.fingerprint for item in unique_frames]
    min_same_cluster_feature_distance = 0.18 if prefer_anomalies else 0.12
    min_global_feature_distance = 0.06 if prefer_anomalies else 0.05
    max_same_cluster_fingerprint_similarity = 0.94
    max_global_fingerprint_similarity = 0.98

    selected_indexes: list[int] = []
    for cluster_id, members in enumerate(cluster_members):
        if int(cluster_id) not in allowed_clusters:
            continue
        if not members:
            continue
        center = centers[cluster_id]
        members_array = np.asarray(members, dtype=np.int32)
        member_features = normalized[members_array]
        distances = np.linalg.norm(member_features - center, axis=1)
        center_closeness = 1.0 / (1.0 + distances)
        if prefer_anomalies:
            combined = 0.70 * anomaly_norm[members_array] + 0.20 * quality_norm[members_array] + 0.10 * center_closeness
        else:
            combined = 0.75 * quality_norm[members_array] + 0.25 * center_closeness
        best_local_index = int(members_array[int(np.argmax(combined))])
        selected_indexes.append(best_local_index)
        if len(selected_indexes) >= capped_target:
            break

    selected_set = set(selected_indexes)
    if len(selected_indexes) < capped_target:
        per_cluster_ranked: list[list[int]] = []
        for cluster_id, members in enumerate(cluster_members):
            if int(cluster_id) not in allowed_clusters:
                per_cluster_ranked.append([])
                continue
            if not members:
                per_cluster_ranked.append([])
                continue
            center = centers[cluster_id]
            ranked = sorted(
                members,
                key=lambda idx: (
                    (
                        float(anomaly_norm[int(idx)]) * 0.70
                        + float(quality_norm[int(idx)]) * 0.20
                        + float(1.0 / (1.0 + np.linalg.norm(normalized[int(idx)] - center))) * 0.10
                    )
                    if prefer_anomalies
                    else (
                        float(quality_norm[int(idx)]) * 0.75
                        + float(1.0 / (1.0 + np.linalg.norm(normalized[int(idx)] - center))) * 0.25
                    )
                ),
                reverse=True,
            )
            per_cluster_ranked.append([int(idx) for idx in ranked])
        selected_set = set(selected_indexes)
        round_index = 0
        while len(selected_indexes) < capped_target:
            progressed = False
            for ranked in per_cluster_ranked:
                if round_index >= len(ranked):
                    continue
                candidate_index = int(ranked[round_index])
                if candidate_index in selected_set:
                    continue
                if _is_near_duplicate_candidate(
                    candidate_index=candidate_index,
                    selected_indexes=selected_indexes,
                    normalized=normalized,
                    fingerprints=frame_fingerprints,
                    assignments=assignments,
                    same_cluster_only=True,
                    min_feature_distance=min_same_cluster_feature_distance,
                    max_fingerprint_similarity=max_same_cluster_fingerprint_similarity,
                ):
                    continue
                if _is_near_duplicate_candidate(
                    candidate_index=candidate_index,
                    selected_indexes=selected_indexes,
                    normalized=normalized,
                    fingerprints=frame_fingerprints,
                    assignments=assignments,
                    same_cluster_only=False,
                    min_feature_distance=min_global_feature_distance,
                    max_fingerprint_similarity=max_global_fingerprint_similarity,
                ):
                    continue
                selected_indexes.append(candidate_index)
                selected_set.add(candidate_index)
                progressed = True
                if len(selected_indexes) >= capped_target:
                    break
            if not progressed:
                break
            round_index += 1

    selected_indexes = selected_indexes[:capped_target]
    selected_set = set(selected_indexes)
    if len(selected_indexes) < capped_target:
        fallback_score = anomaly_norm if prefer_anomalies else quality_norm
        for index in np.argsort(-fallback_score).tolist():
            index_int = int(index)
            cluster_id = int(assignments[index_int]) if assignments.size > index_int else 0
            if int(cluster_id) not in allowed_clusters:
                continue
            if index_int in selected_set:
                continue
            if _is_near_duplicate_candidate(
                candidate_index=index_int,
                selected_indexes=selected_indexes,
                normalized=normalized,
                fingerprints=frame_fingerprints,
                assignments=assignments,
                same_cluster_only=True,
                min_feature_distance=min_same_cluster_feature_distance,
                max_fingerprint_similarity=max_same_cluster_fingerprint_similarity,
            ):
                continue
            if _is_near_duplicate_candidate(
                candidate_index=index_int,
                selected_indexes=selected_indexes,
                normalized=normalized,
                fingerprints=frame_fingerprints,
                assignments=assignments,
                same_cluster_only=False,
                min_feature_distance=min_global_feature_distance,
                max_fingerprint_similarity=max_global_fingerprint_similarity,
            ):
                continue
            selected_indexes.append(index_int)
            selected_set.add(index_int)
            if len(selected_indexes) >= capped_target:
                break
    if len(selected_indexes) < capped_target:
        fallback_score = anomaly_norm if prefer_anomalies else quality_norm
        for index in np.argsort(-fallback_score).tolist():
            index_int = int(index)
            if index_int in selected_set:
                continue
            if _is_near_duplicate_candidate(
                candidate_index=index_int,
                selected_indexes=selected_indexes,
                normalized=normalized,
                fingerprints=frame_fingerprints,
                assignments=assignments,
                same_cluster_only=False,
                min_feature_distance=min_global_feature_distance * 0.6,
                max_fingerprint_similarity=max_global_fingerprint_similarity,
            ):
                continue
            selected_indexes.append(index_int)
            selected_set.add(index_int)
            if len(selected_indexes) >= capped_target:
                break

    cluster_sizes = [len(members) for members in cluster_members]
    tasks: list[PipelineTaskCandidate] = []
    for rank, frame_index in enumerate(selected_indexes):
        frame = unique_frames[int(frame_index)]
        cluster_id = int(assignments[int(frame_index)])
        anomaly_score = float(anomaly_norm[int(frame_index)]) if anomaly_norm.size > int(frame_index) else 0.0
        cluster_label = str(cluster_labels.get(cluster_id, "normal_clean"))
        priority_score = (
            float(0.75 * anomaly_score + 0.25 * float(quality_norm[int(frame_index)]))
            if prefer_anomalies
            else float(frame.quality_score)
        )
        tasks.append(
            PipelineTaskCandidate(
                frame_key=frame.frame_key,
                display_name=frame.display_name,
                original_path=str(frame.path),
                priority_score=priority_score,
                reason="bootstrap_clustered_anomaly" if prefer_anomalies else "bootstrap_clustered_diversity",
                metadata={
                    "quality_score": float(frame.quality_score),
                    "anomaly_score": float(anomaly_score),
                    "cluster_id": int(cluster_id),
                    "cluster_label": cluster_label,
                    "cluster_size": int(cluster_sizes[cluster_id]) if 0 <= cluster_id < len(cluster_sizes) else 0,
                    "selection_rank": int(rank + 1),
                    "cluster_count": int(cluster_total),
                    "selected_from_non_normal_cluster": bool(cluster_label != "normal_clean"),
                    "brightness": float(frame.vector[0]),
                    "contrast": float(frame.vector[1]),
                    "entropy": float(frame.vector[2]),
                    "edge_density": float(frame.vector[3]),
                },
            )
        )

    return BootstrapSamplingResult(
        total_discovered=total_discovered,
        total_quality_passed=total_quality_passed,
        total_unique=total_unique,
        selected=tuple(tasks),
    )


def recommend_corrective_candidates(
    records: Iterable[FrameRecord],
    *,
    top_k: int,
) -> tuple[PipelineTaskCandidate, ...]:
    """Select high-risk frames for corrective labeling from validation output."""

    rows: list[tuple[float, FrameRecord, float, float, float]] = []
    for record in records:
        summary = record.summary
        if summary is None:
            continue
        export_priority = float(summary.export_priority_score or 0.0)
        disagreement = float(summary.disagreement_score or 0.0)
        matrix_score = float(record.score) if bool(getattr(record, "score_ready", False)) else 0.0
        matrix_score = float(max(0.0, min(1.0, matrix_score)))
        matrix_badness = float(1.0 - matrix_score)
        # Primary signal: visual badness on the matrix (red frames are prioritized).
        # Secondary signal: export priority from validation summary.
        risk_score = float(0.85 * matrix_badness + 0.15 * max(0.0, min(1.0, export_priority)))
        rows.append((risk_score, record, export_priority, disagreement, matrix_badness))
    rows.sort(key=lambda item: item[0], reverse=True)
    capped = max(0, min(int(top_k), len(rows)))
    selected = rows[:capped]

    candidates: list[PipelineTaskCandidate] = []
    for risk_score, record, export_priority, disagreement, matrix_badness in selected:
        original_path = str(record.original_path or record.base_path or "")
        candidates.append(
            PipelineTaskCandidate(
                frame_key=str(record.key),
                display_name=str(record.display_name or record.key),
                original_path=original_path,
                priority_score=float(risk_score),
                reason="corrective_from_validation_metrics",
                metadata={
                    "risk_score": float(risk_score),
                    "matrix_badness": float(matrix_badness),
                    "export_priority_score": float(export_priority),
                    "disagreement_score": float(disagreement),
                },
            )
        )
    return tuple(candidates)


def _extract_frame_features(path: Path, root: Path) -> _BootstrapFrameFeatures | None:
    try:
        image = np.asarray(load_grayscale_image(path), dtype=np.uint8)
    except Exception:
        return None
    if image.ndim != 2 or image.size <= 0:
        return None
    if min(image.shape[0], image.shape[1]) < 24:
        return None

    raw = extract_original_frame_features(image)
    if raw is None:
        return None
    contrast = float(getattr(raw, "contrast", 0.0))
    dynamic_range = float(getattr(raw, "dynamic_range", 0.0))
    if contrast < 0.01 or dynamic_range < 0.05:
        return None

    mean_brightness = float(getattr(raw, "mean_brightness", 0.0))
    entropy = float(getattr(raw, "entropy", 0.0))
    edge_density = float(getattr(raw, "edge_density", 0.0))
    blur_score = float(getattr(raw, "blur_score", 0.0))
    noise_score = float(getattr(raw, "noise_score", 0.0))
    saturation_ratio = float(getattr(raw, "saturation_ratio", 0.0))
    quality_score = (
        0.20 * contrast
        + 0.20 * edge_density
        + 0.15 * entropy
        + 0.15 * np.clip(dynamic_range, 0.0, 1.0)
        + 0.15 * np.clip(noise_score, 0.0, 1.0)
        + 0.15 * (1.0 - np.clip(saturation_ratio, 0.0, 1.0))
    )
    frame_key = _frame_key_from_path(path, root)
    return _BootstrapFrameFeatures(
        path=path,
        frame_key=frame_key,
        display_name=path.stem,
        quality_score=float(quality_score),
        vector=(mean_brightness, contrast, entropy, edge_density, blur_score, noise_score, saturation_ratio),
        fingerprint=_image_fingerprint(image),
    )


def _frame_key_from_path(path: Path, root: Path) -> str:
    path_obj = Path(path)
    try:
        relative = path_obj.resolve().relative_to(Path(root).resolve())
        return str(relative.with_suffix("")).replace("\\", "/")
    except Exception:
        return str(path_obj.with_suffix("").name)


def _image_fingerprint(image: np.ndarray) -> tuple[int, ...]:
    height, width = image.shape
    rows = np.linspace(0, max(0, height - 1), num=16, dtype=np.int32)
    cols = np.linspace(0, max(0, width - 1), num=16, dtype=np.int32)
    sampled = image[np.ix_(rows, cols)].astype(np.float32)
    threshold = float(sampled.mean())
    binary = (sampled >= threshold).astype(np.uint8).flatten()
    packed = np.packbits(binary)
    return tuple(int(item) for item in packed.tolist())


def _deduplicate_by_fingerprint(items: Iterable[_BootstrapFrameFeatures]) -> list[_BootstrapFrameFeatures]:
    best_by_fp: dict[tuple[int, ...], _BootstrapFrameFeatures] = {}
    for item in items:
        existing = best_by_fp.get(item.fingerprint)
        if existing is None or item.quality_score > existing.quality_score:
            best_by_fp[item.fingerprint] = item
    result = list(best_by_fp.values())
    result.sort(key=lambda row: row.path.as_posix().lower())
    return result


def _uniform_sample(items: list[_BootstrapFrameFeatures], target_count: int) -> list[_BootstrapFrameFeatures]:
    if target_count <= 0 or not items:
        return []
    count = min(target_count, len(items))
    if count == len(items):
        return list(items)
    indexes = np.linspace(0, len(items) - 1, num=count, dtype=np.int32)
    return [items[int(index)] for index in indexes.tolist()]


def _diversity_sample(
    items: list[_BootstrapFrameFeatures],
    target_count: int,
    rng: np.random.Generator,
) -> list[_BootstrapFrameFeatures]:
    if target_count <= 0 or not items:
        return []
    count = min(target_count, len(items))
    matrix = np.asarray([item.vector for item in items], dtype=np.float32)
    mins = matrix.min(axis=0)
    spans = np.maximum(matrix.max(axis=0) - mins, 1e-8)
    normalized = (matrix - mins) / spans

    start_index = int(rng.integers(0, len(items)))
    selected = [start_index]
    selected_mask = np.zeros((len(items),), dtype=bool)
    selected_mask[start_index] = True
    min_distances = np.linalg.norm(normalized - normalized[start_index], axis=1)
    while len(selected) < count:
        candidate_index = int(np.argmax(min_distances))
        if selected_mask[candidate_index]:
            break
        selected.append(candidate_index)
        selected_mask[candidate_index] = True
        distances = np.linalg.norm(normalized - normalized[candidate_index], axis=1)
        min_distances = np.minimum(min_distances, distances)
    return [items[index] for index in selected]


def _normalize_feature_matrix(matrix: np.ndarray) -> np.ndarray:
    mins = matrix.min(axis=0)
    spans = np.maximum(matrix.max(axis=0) - mins, 1e-8)
    return (matrix - mins) / spans


def _kmeans_assign(
    normalized: np.ndarray,
    cluster_count: int,
    *,
    seed: int,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
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
        distances = np.linalg.norm(normalized[:, None, :] - centers[None, :, :], axis=2)
        next_assignments = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(assignments, next_assignments):
            break
        assignments = next_assignments
        for cluster_id in range(int(centers.shape[0])):
            members = normalized[assignments == cluster_id]
            if members.size <= 0:
                replacement = int(rng.integers(0, rows))
                centers[cluster_id] = normalized[replacement]
            else:
                centers[cluster_id] = members.mean(axis=0)
    return assignments, centers


def _bootstrap_anomaly_score_from_vector(vector: tuple[float, ...]) -> float:
    brightness = float(vector[0]) if len(vector) > 0 else 0.0
    contrast = float(vector[1]) if len(vector) > 1 else 0.0
    entropy = float(vector[2]) if len(vector) > 2 else 0.0
    edge_density = float(vector[3]) if len(vector) > 3 else 0.0
    blur_score = float(vector[4]) if len(vector) > 4 else 0.0
    noise_score = float(vector[5]) if len(vector) > 5 else 0.0
    saturation_ratio = float(vector[6]) if len(vector) > 6 else 0.0

    blur_penalty = max(0.0, min(1.0, (0.40 - blur_score) / 0.40))
    low_contrast_penalty = max(0.0, min(1.0, (0.14 - contrast) / 0.14))
    low_edge_penalty = max(0.0, min(1.0, (0.08 - edge_density) / 0.08))
    high_noise_penalty = max(0.0, min(1.0, (noise_score - 0.58) / 0.42))
    over_saturation_penalty = max(0.0, min(1.0, (saturation_ratio - 0.72) / 0.28))
    under_entropy_penalty = max(0.0, min(1.0, (0.22 - entropy) / 0.22))
    exposure_penalty = max(0.0, min(1.0, (abs(brightness - 0.5) - 0.25) / 0.25))

    score = (
        0.30 * blur_penalty
        + 0.20 * low_contrast_penalty
        + 0.15 * low_edge_penalty
        + 0.15 * high_noise_penalty
        + 0.08 * over_saturation_penalty
        + 0.07 * under_entropy_penalty
        + 0.05 * exposure_penalty
    )
    return float(max(0.0, min(1.0, score)))


def _cluster_label_from_profile(mean_vector: np.ndarray, anomaly_score: float) -> str:
    brightness = float(mean_vector[0]) if mean_vector.size > 0 else 0.0
    contrast = float(mean_vector[1]) if mean_vector.size > 1 else 0.0
    entropy = float(mean_vector[2]) if mean_vector.size > 2 else 0.0
    edge_density = float(mean_vector[3]) if mean_vector.size > 3 else 0.0
    blur_score = float(mean_vector[4]) if mean_vector.size > 4 else 0.0
    noise_score = float(mean_vector[5]) if mean_vector.size > 5 else 0.0
    saturation_ratio = float(mean_vector[6]) if mean_vector.size > 6 else 0.0

    if anomaly_score < 0.20 and contrast >= 0.14 and edge_density >= 0.08 and blur_score >= 0.36:
        return "normal_clean"
    if blur_score < 0.34 or (contrast < 0.12 and edge_density < 0.07):
        return "blur_or_low_detail"
    if noise_score > 0.58:
        return "noisy_artifacts"
    if saturation_ratio > 0.72 or abs(brightness - 0.5) > 0.27:
        return "exposure_or_saturation_issue"
    if entropy < 0.22 or contrast < 0.10:
        return "low_information"
    return "mixed_anomaly"


def _is_near_duplicate_candidate(
    *,
    candidate_index: int,
    selected_indexes: list[int],
    normalized: np.ndarray,
    fingerprints: list[tuple[int, ...]],
    assignments: np.ndarray,
    same_cluster_only: bool,
    min_feature_distance: float,
    max_fingerprint_similarity: float,
) -> bool:
    if not selected_indexes:
        return False
    candidate_vector = normalized[int(candidate_index)]
    candidate_cluster = int(assignments[int(candidate_index)]) if assignments.size > int(candidate_index) else 0
    candidate_fp = fingerprints[int(candidate_index)] if int(candidate_index) < len(fingerprints) else tuple()
    for selected_index in selected_indexes:
        selected_int = int(selected_index)
        if same_cluster_only:
            selected_cluster = int(assignments[selected_int]) if assignments.size > selected_int else 0
            if selected_cluster != candidate_cluster:
                continue
        selected_vector = normalized[selected_int]
        feature_distance = float(np.linalg.norm(candidate_vector - selected_vector))
        if feature_distance <= float(min_feature_distance):
            return True
        selected_fp = fingerprints[selected_int] if selected_int < len(fingerprints) else tuple()
        similarity = _fingerprint_similarity(candidate_fp, selected_fp)
        if similarity >= float(max_fingerprint_similarity) and feature_distance <= float(min_feature_distance) * 1.8:
            return True
    return False


def _fingerprint_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    if size <= 0:
        return 0.0
    total_bits = int(size * 8)
    if total_bits <= 0:
        return 0.0
    different_bits = 0
    for left_value, right_value in zip(left[:size], right[:size]):
        different_bits += int((int(left_value) ^ int(right_value)).bit_count())
    similarity = 1.0 - (float(different_bits) / float(total_bits))
    return float(max(0.0, min(1.0, similarity)))
