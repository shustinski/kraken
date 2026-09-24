"""Grid calibration: reference, examples, and the shared cell decision. Qt-free and picklable."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

_LOGGER = logging.getLogger(__name__)

REFERENCE_KEYS = (
    "width",
    "height",
    "area",
    "fill",
    "interior_fill",
    "center_fill",
    "aspect",
    "solidity",
    "extent",
)


def truncated_median(values: Sequence[float], trim: float = 0.20) -> float:
    """Median after dropping both tails, so a minority of outliers does not move the reference."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) < 5:
        return float(np.median(ordered))
    cut = int(len(ordered) * trim)
    core = ordered[cut : len(ordered) - cut] or ordered
    return float(np.median(core))


def _feature(item: Any, *names: str) -> float | None:
    source = item.get("features", item) if isinstance(item, Mapping) else item
    for name in names:
        if isinstance(source, Mapping) and name in source:
            return float(source[name])
        if hasattr(source, name):
            return float(getattr(source, name))
    return None


def _sample_record(item: Any) -> dict[str, float] | None:
    record: dict[str, float] = {}
    mapping = (
        ("width", ("width", "median_width")),
        ("height", ("height", "median_height")),
        ("area", ("area", "median_area")),
        ("fill", ("fill", "fill_ratio", "median_fill")),
        ("interior_fill", ("interior_fill", "interior_fill_ratio", "median_interior_fill")),
        ("center_fill", ("center_fill", "center_fill_ratio", "median_center_fill")),
        ("aspect", ("aspect", "aspect_ratio", "median_aspect")),
        ("solidity", ("solidity", "median_solidity")),
        ("extent", ("extent", "median_extent")),
    )
    for key, names in mapping:
        value = _feature(item, *names)
        if value is not None:
            record[key] = value
    if "interior_fill" not in record:
        return None
    if "width" not in record and hasattr(item, "bbox"):
        bbox = getattr(item, "bbox")
        record["width"] = float(bbox[2])
        record["height"] = float(bbox[3])
    return record


def reference_from_normal_examples(samples: Sequence[Any]) -> dict[str, float] | None:
    """Median feature reference from operator examples labeled normal."""

    records: list[dict[str, float]] = []
    for sample in samples:
        label = "normal"
        if isinstance(sample, Mapping):
            label = str(sample.get("label", "normal") or "normal")
        if label not in {"normal", "good"}:
            continue
        record = _sample_record(sample)
        if record is not None:
            records.append(record)
    if not records:
        return None
    reference: dict[str, float] = {}
    for key in REFERENCE_KEYS:
        values = [item[key] for item in records if key in item]
        if values:
            reference[key] = truncated_median(values)
    return reference or None


def modal_size_cluster(items: Sequence[Any]) -> list[Any]:
    """Largest group of similar contour areas. This is the frame's typical cell, not every contour."""

    sized = []
    for item in items:
        area = _feature(item, "area")
        if area is not None and area >= 40.0:
            sized.append(item)
    if len(sized) < 4:
        return []
    logs = np.log(np.maximum([float(_feature(item, "area") or 1.0) for item in sized], 1.0))
    bins = np.floor(logs / 0.25).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    groups = []
    for bin_id, count in zip(values, counts):
        if int(count) < 4:
            continue
        group = [item for item, item_bin in zip(sized, bins) if int(item_bin) == int(bin_id)]
        interior = [
            value
            for item in group
            if (value := _feature(item, "interior_fill", "interior_fill_ratio")) is not None
        ]
        groups.append((float(np.median(interior)) if interior else 1.0, -int(count), group))
    if not groups:
        mode = int(values[int(np.argmax(counts))])
        chosen = [item for item, bin_id in zip(sized, bins) if int(bin_id) == mode]
    else:
        groups.sort(key=lambda item: (item[0], item[1]))
        chosen = groups[0][2]
    hollow = []
    for item in chosen:
        interior = _feature(item, "interior_fill", "interior_fill_ratio")
        if interior is not None and interior < 0.45:
            hollow.append(item)
    if len(hollow) >= 4 and len(hollow) < len(chosen):
        return hollow
    return chosen


def robust_frame_reference(items: Sequence[Any]) -> dict[str, float] | None:
    """Truncated medians of the modal size cluster. Used when no normal examples exist."""

    cluster = modal_size_cluster(items)
    if len(cluster) < 4:
        return None
    records = [record for item in cluster if (record := _sample_record(item)) is not None]
    if len(records) < 4:
        return None
    reference = {key: truncated_median([item[key] for item in records if key in item]) for key in REFERENCE_KEYS}
    reference = {key: value for key, value in reference.items() if any(key in item for item in records)}
    if float(reference.get("area", 0.0)) < 36.0:
        return None
    return reference


def local_neighbor_medians(
    candidates: Sequence[Any], *, neighbors: int = 8
) -> tuple[list[dict[str, float]], list[tuple[int, ...]]]:
    """Median size and appearance of the k nearest cells, excluding the cell itself."""

    count = len(candidates)
    if count == 0:
        return [], []
    points = np.array(
        [[float(getattr(item, "centroid")[0]), float(getattr(item, "centroid")[1])] for item in candidates],
        dtype=np.float64,
    )
    features = np.array(
        [
            [
                float(item.bbox[2]),
                float(item.bbox[3]),
                float(item.area),
                float(item.interior_fill_ratio),
                float(item.solidity),
                float(item.extent),
            ]
            for item in candidates
        ],
        dtype=np.float64,
    )
    query_k = min(count, neighbors + 1)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        _distances, indexes = tree.query(points, k=query_k)
        if query_k == 1:
            indexes = np.asarray(indexes).reshape(count, 1)
    except Exception:
        indexes = np.argsort(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2), axis=1)[:, :query_k]
    indexes = np.asarray(indexes, dtype=np.int64)
    self_ids = np.arange(count, dtype=np.int64)[:, None]
    masked = np.where(indexes == self_ids, -1, indexes)
    neighbor_lists: list[tuple[int, ...]] = []
    chosen = np.empty((count, neighbors), dtype=np.int64)
    for index in range(count):
        picked = [int(value) for value in masked[index].tolist() if value >= 0][:neighbors]
        if not picked:
            picked = [index]
        while len(picked) < neighbors:
            picked.append(picked[-1])
        chosen[index] = picked[:neighbors]
        neighbor_lists.append(tuple(dict.fromkeys(picked[:neighbors])))
    gathered = features[chosen]
    values = np.median(gathered, axis=1)
    medians = [
        {
            "width": max(1.0, float(row[0])),
            "height": max(1.0, float(row[1])),
            "area": max(1.0, float(row[2])),
            "interior_fill": float(row[3]),
            "solidity": float(row[4]),
            "extent": float(row[5]),
        }
        for row in values
    ]
    return medians, neighbor_lists


def select_diverse_calibration_keys(
    records: Sequence[Any],
    *,
    limit: int = 8,
) -> tuple[str, ...]:
    """Pick start/middle/end frames and fill the rest evenly."""

    items = [item for item in records if getattr(item, "key", None)]
    if not items:
        return ()
    if len(items) <= limit:
        return tuple(str(item.key) for item in items)
    anchors = [0, len(items) // 2, len(items) - 1]
    chosen: list[str] = []
    for index in anchors:
        key = str(items[index].key)
        if key not in chosen:
            chosen.append(key)
    step = max(1, len(items) // max(1, limit))
    for index in range(0, len(items), step):
        key = str(items[index].key)
        if key not in chosen:
            chosen.append(key)
        if len(chosen) >= limit:
            break
    # Keep anchors even if the even fill pushed them out of the first slots.
    ordered = list(dict.fromkeys([*(str(items[index].key) for index in anchors), *chosen]))
    return tuple(ordered[:limit])


def reference_pairs(reference: Mapping[str, float] | None) -> tuple[tuple[str, float], ...]:
    if not reference:
        return ()
    return tuple(
        (key, float(reference[key]))
        for key in REFERENCE_KEYS
        if key in reference
    )


SLIDER_KEYS = (
    "fill_sensitivity",
    "debris_sensitivity",
    "geometry_sensitivity",
    "merge_sensitivity",
    "mismatch_sensitivity",
    "disagreement_sensitivity",
)
_DEFECT_LABELS = ("filled_cell", "partial_filled_cell", "broken_geometry", "merged_contour", "small_artifact")


_DISTANCE_KEYS = REFERENCE_KEYS + ("width_ratio", "height_ratio", "area_ratio")


def example_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Scale-free distance between two cell feature records."""

    deltas = []
    for key in _DISTANCE_KEYS:
        if key not in left or key not in right:
            continue
        scale = max(abs(float(left[key])), abs(float(right[key])), 1.0)
        deltas.append((float(left[key]) - float(right[key])) / scale)
    if not deltas:
        return 1.0e9
    return float(np.linalg.norm(deltas))


def influence_radius(example_influence: float) -> tuple[float, float]:
    """Return (radius, score margin) for one influence setting in 0..1."""

    unit = max(0.0, min(1.0, float(example_influence)))
    return 0.25 + 1.35 * unit, 0.05 + 0.20 * unit


def example_distance_matrix(
    candidates: Sequence[Mapping[str, float]],
    examples: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Distances from each candidate to each example, scaled by the candidates' median and IQR."""

    usable = [item for item in examples if isinstance(item.get("features"), Mapping)]
    if not candidates or not usable:
        return np.zeros((len(candidates), 0))
    sample_keys = set(candidates[0]) | set(usable[0]["features"])
    keys = [key for key in _DISTANCE_KEYS if key in sample_keys]
    if not keys:
        return np.full((len(candidates), len(usable)), 1.0e9)
    candidate_matrix = np.empty((len(candidates), len(keys)), dtype=np.float64)
    for row_index, row in enumerate(candidates):
        for key_index, key in enumerate(keys):
            candidate_matrix[row_index, key_index] = float(row[key]) if key in row else np.nan
    center = np.nanmedian(candidate_matrix, axis=0)
    q75 = np.nanpercentile(candidate_matrix, 75, axis=0)
    q25 = np.nanpercentile(candidate_matrix, 25, axis=0)
    span = np.maximum(np.where(np.isfinite(q75 - q25), q75 - q25, 1.0), 0.05)
    center = np.where(np.isfinite(center), center, 0.0)
    filled = np.where(np.isfinite(candidate_matrix), candidate_matrix, center)
    example_matrix = np.empty((len(usable), len(keys)), dtype=np.float64)
    for row_index, item in enumerate(usable):
        features = item["features"]
        for key_index, key in enumerate(keys):
            example_matrix[row_index, key_index] = float(features[key]) if key in features else float(center[key_index])
    scaled_candidates = (filled - center) / span
    scaled_examples = (example_matrix - center) / span
    return np.linalg.norm(scaled_candidates[:, None, :] - scaled_examples[None, :, :], axis=2)


def apply_example_correction(
    reasons: tuple[str, ...],
    scores: Mapping[str, float],
    thresholds: Mapping[str, float],
    features: Mapping[str, float] | None,
    examples: Sequence[Mapping[str, Any]],
    *,
    example_influence: float,
    distance_row: Sequence[float] | None = None,
) -> tuple[str, ...]:
    """Suppress or add a defect when a labeled example is clearly nearer in feature space."""

    if features is None or not examples:
        return tuple(reasons)
    radius, margin = influence_radius(example_influence)
    if radius <= 0.0:
        return tuple(reasons)
    usable = [item for item in examples if isinstance(item.get("features"), Mapping)]
    if not usable:
        return tuple(reasons)
    if distance_row is None:
        distances = [example_distance(features, item["features"]) for item in usable]
    else:
        distances = [float(value) for value in distance_row]
    labels = [str(item.get("label") or "") for item in usable]
    normal_distance = 1.0e9
    type_nearest = {reason: 1.0e9 for reason in _DEFECT_LABELS}
    for distance, label in zip(distances, labels):
        if label in {"normal", "good", "ignore"}:
            normal_distance = min(normal_distance, distance)
        elif label in type_nearest:
            type_nearest[label] = min(type_nearest[label], distance)

    def nearest(labels: set[str]) -> float:
        if labels <= {"normal", "good", "ignore"}:
            return normal_distance
        return min((type_nearest.get(label, 1.0e9) for label in labels), default=1.0e9)

    found = [reason for reason in reasons if reason in _DEFECT_LABELS]
    kept: list[str] = [reason for reason in reasons if reason not in _DEFECT_LABELS]
    for reason in found:
        type_distance = nearest({reason})
        if normal_distance <= radius and normal_distance <= 0.7 * type_distance:
            continue
        kept.append(reason)
    for reason, score_key, threshold_key in (
        ("filled_cell", "fill", "fill"),
        ("broken_geometry", "geometry", "geometry"),
        ("merged_contour", "merge", "merge"),
        ("small_artifact", "debris", "debris"),
    ):
        if reason in kept:
            continue
        threshold = float(thresholds.get(threshold_key, 1.0))
        if float(scores.get(score_key, 0.0)) < threshold - margin:
            continue
        type_distance = nearest({reason})
        if type_distance <= radius and type_distance <= 0.7 * normal_distance:
            kept.append(reason)
    return tuple(dict.fromkeys(kept))


def decide_reasons_batch(
    score_rows: Sequence[Mapping[str, float]],
    thresholds: Mapping[str, float],
    *,
    feature_rows: Sequence[Mapping[str, float]] | None = None,
    examples: Sequence[Mapping[str, Any]] = (),
    example_influence: float = 0.5,
    edge_enabled: bool = False,
) -> list[tuple[str, ...]]:
    """Recompute reasons for many cells. Uses one distance matrix for examples."""

    from .grid_scoring import NORMAL_SCORE_CEILING, PARTIAL_FILL_GAP

    if not score_rows:
        return []
    fill_t = max(float(thresholds.get("fill", 1.0)), NORMAL_SCORE_CEILING)
    partial_t = max(fill_t - PARTIAL_FILL_GAP, NORMAL_SCORE_CEILING)
    geometry_t = max(float(thresholds.get("geometry", 1.0)), NORMAL_SCORE_CEILING)
    merge_t = max(float(thresholds.get("merge", 1.0)), NORMAL_SCORE_CEILING)
    debris_t = max(float(thresholds.get("debris", 1.0)), NORMAL_SCORE_CEILING)
    edge_t = float(thresholds.get("edge", 1.0))
    fills = np.array([float(row.get("fill", 0.0)) for row in score_rows], dtype=np.float64)
    geometries = np.array([float(row.get("geometry", 0.0)) for row in score_rows], dtype=np.float64)
    merges = np.array([float(row.get("merge", 0.0)) for row in score_rows], dtype=np.float64)
    debris = np.array([float(row.get("debris", 0.0)) for row in score_rows], dtype=np.float64)
    edges = np.array([float(row.get("edge", 0.0)) for row in score_rows], dtype=np.float64)
    results: list[list[str]] = [[] for _ in score_rows]
    for index, fill in enumerate(fills):
        if fill > NORMAL_SCORE_CEILING and fill >= fill_t:
            results[index].append("filled_cell")
        elif fill > NORMAL_SCORE_CEILING and partial_t < fill_t and fill >= partial_t:
            results[index].append("partial_filled_cell")
        if geometries[index] > NORMAL_SCORE_CEILING and geometries[index] >= geometry_t:
            results[index].append("broken_geometry")
        if merges[index] > NORMAL_SCORE_CEILING and merges[index] >= merge_t:
            results[index].append("merged_contour")
        if (
            debris[index] > NORMAL_SCORE_CEILING
            and debris[index] >= debris_t
            and "merged_contour" not in results[index]
        ):
            results[index].append("small_artifact")
        if edge_enabled and edges[index] >= edge_t and "broken_geometry" in results[index]:
            results[index] = [reason for reason in results[index] if reason != "broken_geometry"]
            results[index].append("edge_clipped_cell")
    usable = [item for item in examples if isinstance(item.get("features"), Mapping)]
    if not usable or not feature_rows:
        return [tuple(row) for row in results]
    radius, margin = influence_radius(example_influence)
    if radius <= 0.0:
        return [tuple(row) for row in results]
    distances = example_distance_matrix(feature_rows, usable)
    labels = [str(item.get("label") or "") for item in usable]
    normal_mask = np.array([label in {"normal", "good", "ignore"} for label in labels], dtype=bool)
    type_masks = {reason: np.array([label == reason for label in labels], dtype=bool) for reason in _DEFECT_LABELS}
    for index, row in enumerate(results):
        row_distances = distances[index]
        normal_distance = float(row_distances[normal_mask].min()) if normal_mask.any() else 1.0e9
        kept = [reason for reason in row if reason not in _DEFECT_LABELS]
        for reason in [item for item in row if item in _DEFECT_LABELS]:
            mask = type_masks[reason]
            type_distance = float(row_distances[mask].min()) if mask.any() else 1.0e9
            if normal_distance <= radius and normal_distance <= 0.7 * type_distance:
                continue
            kept.append(reason)
        scores = score_rows[index]
        for reason, score_key, threshold_key in (
            ("filled_cell", "fill", "fill"),
            ("broken_geometry", "geometry", "geometry"),
            ("merged_contour", "merge", "merge"),
            ("small_artifact", "debris", "debris"),
        ):
            if reason in kept:
                continue
            threshold = float(thresholds.get(threshold_key, 1.0))
            if float(scores.get(score_key, 0.0)) < threshold - margin:
                continue
            mask = type_masks[reason]
            type_distance = float(row_distances[mask].min()) if mask.any() else 1.0e9
            if type_distance <= radius and type_distance <= 0.7 * normal_distance:
                kept.append(reason)
        results[index] = list(dict.fromkeys(kept))
    return [tuple(row) for row in results]


def decide_reasons(
    scores: Mapping[str, float],
    thresholds: Mapping[str, float],
    *,
    features: Mapping[str, float] | None = None,
    examples: Sequence[Mapping[str, Any]] = (),
    example_influence: float = 0.5,
    edge_enabled: bool = False,
    distance_row: Sequence[float] | None = None,
) -> tuple[str, ...]:
    """Thresholds plus example correction. No image IO."""

    from .grid_scoring import CalibratedScores

    scored = CalibratedScores(
        fill=float(scores.get("fill", 0.0)),
        geometry=float(scores.get("geometry", 0.0)),
        merge=float(scores.get("merge", 0.0)),
        debris=float(scores.get("debris", 0.0)),
        edge=float(scores.get("edge", 0.0)),
    )
    reasons = scored.reasons(
        fill=float(thresholds.get("fill", 1.0)),
        geometry=float(thresholds.get("geometry", 1.0)),
        merge=float(thresholds.get("merge", 1.0)),
        debris=float(thresholds.get("debris", 1.0)),
        edge=float(thresholds.get("edge", 1.0)),
        edge_enabled=edge_enabled,
    )
    return apply_example_correction(
        reasons,
        scores,
        thresholds,
        features,
        examples,
        example_influence=example_influence,
        distance_row=distance_row,
    )


def fit_slider(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> int | None:
    """Pick a 0..100 slider that best separates defect examples from normal ones."""

    if len(positive_scores) < 2 or len(negative_scores) < 2:
        return None
    best_f1 = -1.0
    best_threshold = 0.5
    for step in range(1, 20):
        threshold = step / 20.0
        true_positive = sum(score >= threshold for score in positive_scores)
        false_positive = sum(score >= threshold for score in negative_scores)
        false_negative = len(positive_scores) - true_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if f1 > best_f1 or (f1 == best_f1 and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_f1 = f1
            best_threshold = threshold
    return int(round((1.0 - best_threshold) * 100.0))


def example_detector_scores(features: Mapping[str, float]) -> dict[str, float]:
    """Same score_* the detector applies, from relative features."""

    from .grid_scoring import score_debris, score_fill, score_geometry, score_merge

    width_ratio = float(features.get("width_ratio", 1.0))
    height_ratio = float(features.get("height_ratio", 1.0))
    area_ratio = float(features.get("area_ratio", 1.0))
    axis = max(width_ratio, height_ratio)
    return {
        "fill": score_fill(float(features.get("interior_fill", 0.0)), float(features.get("reference_interior", 0.0))),
        "geometry": score_geometry(
            float(features.get("solidity", 1.0)),
            float(features.get("extent", 1.0)),
            float(features.get("reference_solidity", 1.0)),
            float(features.get("reference_extent", 1.0)),
        ),
        "merge": score_merge(axis, area_ratio),
        "debris": score_debris(area_ratio, axis),
    }


def fit_sliders_from_examples(examples: Sequence[Any]) -> dict[str, int]:
    """Fit each slider that has at least two typed examples and two normal ones."""

    groups = {"fill": [], "geometry": [], "merge": [], "debris": []}
    normals = {"fill": [], "geometry": [], "merge": [], "debris": []}
    label_key = {
        "filled_cell": "fill",
        "partial_filled_cell": "fill",
        "broken_geometry": "geometry",
        "merged_contour": "merge",
        "small_artifact": "debris",
    }
    slider_name = {
        "fill": "fill_sensitivity",
        "geometry": "geometry_sensitivity",
        "merge": "merge_sensitivity",
        "debris": "debris_sensitivity",
    }
    for example in examples:
        features = example.get("features") if isinstance(example, Mapping) else getattr(example, "feature_map", lambda: None)()
        if not isinstance(features, Mapping) or not features:
            continue
        scores = example_detector_scores(features)
        label = str(example.get("label") if isinstance(example, Mapping) else getattr(example, "label", ""))
        if label in {"normal", "good"}:
            for key, value in scores.items():
                normals[key].append(float(value))
            continue
        key = label_key.get(label)
        if key is not None:
            groups[key].append(float(scores[key]))
    fitted: dict[str, int] = {}
    for key, positives in groups.items():
        slider = fit_slider(positives, normals[key])
        if slider is not None:
            fitted[slider_name[key]] = int(slider)
    return fitted


def calibration_summary(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-frame before/after counts and example agreement."""

    rows = []
    matched = 0
    total = 0
    for frame in frames:
        before = int(frame.get("before", 0))
        after = int(frame.get("after", 0))
        ok = int(frame.get("examples_ok", 0))
        count = int(frame.get("examples_total", 0))
        matched += ok
        total += count
        rows.append(
            {
                "key": str(frame.get("key") or ""),
                "before": before,
                "after": after,
                "examples_ok": ok,
                "examples_total": count,
                "disputed": int(frame.get("disputed", 0)),
            }
        )
    return {
        "frames": rows,
        "examples_ok": matched,
        "examples_total": total,
        "before": sum(int(row["before"]) for row in rows),
        "after": sum(int(row["after"]) for row in rows),
    }


@dataclass(frozen=True, slots=True)
class GridCellExample:
    label: str
    features: tuple[tuple[str, float], ...] | None = None
    frame_key: str = ""
    file_signature: str = ""
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    thumbnail_png_b64: str = ""
    created_at: str = ""

    def feature_map(self) -> dict[str, float] | None:
        if not self.features:
            return None
        return {str(key): float(value) for key, value in self.features}


@dataclass(frozen=True, slots=True)
class GridCalibration:
    """Working or confirmed calibration shared by preview and matrix analysis."""

    schema: str = "karakal.grid-calibration.v1"
    preset: str = "balanced"
    sliders: dict[str, int] | None = None
    enabled_reason_types: tuple[str, ...] = ()
    examples: tuple[GridCellExample, ...] = ()
    calibration_frame_keys: tuple[str, ...] = ()
    reference: dict[str, float] | None = None
    example_influence: float = 0.5
    confirmed_at: str | None = None

    def with_normal_examples(self, samples: Sequence[Any]) -> "GridCalibration":
        return replace(self, reference=reference_from_normal_examples(samples))

    def example_records(self) -> tuple[dict[str, Any], ...]:
        records = []
        for example in self.examples[:500]:
            features = example.feature_map()
            if features is None:
                continue
            records.append({"label": example.label, "features": features})
        return tuple(records)

    def fingerprint(self) -> str:
        payload = {
            "schema": self.schema,
            "preset": self.preset,
            "sliders": {key: int((self.sliders or {}).get(key, 0)) for key in SLIDER_KEYS},
            "enabled_reason_types": list(self.enabled_reason_types),
            "examples": [
                {"label": example.label, "features": list(example.features or ())}
                for example in self.examples
            ],
            "reference": self.reference,
            "example_influence": float(self.example_influence),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "preset": self.preset,
            "sliders": dict(self.sliders or {}),
            "enabled_reason_types": list(self.enabled_reason_types),
            "examples": [
                {
                    "label": example.label,
                    "features": list(example.features) if example.features is not None else None,
                    "frame_key": example.frame_key,
                    "file_signature": example.file_signature,
                    "bbox": list(example.bbox),
                    "thumbnail_png_b64": example.thumbnail_png_b64,
                    "created_at": example.created_at,
                }
                for example in self.examples[:500]
            ],
            "calibration_frame_keys": list(self.calibration_frame_keys),
            "reference": self.reference,
            "example_influence": float(self.example_influence),
            "confirmed_at": self.confirmed_at,
        }

    @staticmethod
    def from_payload(payload: object) -> "GridCalibration":
        if not isinstance(payload, dict) or str(payload.get("schema") or "") != "karakal.grid-calibration.v1":
            return GridCalibration()
        try:
            return GridCalibration._from_payload_fields(payload)
        except Exception:
            _LOGGER.warning("Ignoring unreadable grid calibration", exc_info=True)
            return GridCalibration()

    @staticmethod
    def _from_payload_fields(payload: dict) -> "GridCalibration":
        examples: list[GridCellExample] = []
        raw_examples = payload.get("examples") or ()
        if isinstance(raw_examples, list):
            for item in raw_examples:
                try:
                    if not isinstance(item, dict):
                        raise TypeError("example is not an object")
                    raw_features = item.get("features")
                    features = None
                    if isinstance(raw_features, list):
                        features = tuple((str(key), float(value)) for key, value in raw_features)
                    bbox_raw = item.get("bbox") or (0, 0, 0, 0)
                    bbox = tuple(int(value) for value in list(bbox_raw)[:4])
                    if len(bbox) < 4:
                        bbox = (0, 0, 0, 0)
                    examples.append(
                        GridCellExample(
                            label=str(item.get("label") or "normal"),
                            features=features,
                            frame_key=str(item.get("frame_key") or ""),
                            file_signature=str(item.get("file_signature") or ""),
                            bbox=bbox,  # type: ignore[arg-type]
                            thumbnail_png_b64=str(item.get("thumbnail_png_b64") or ""),
                            created_at=str(item.get("created_at") or ""),
                        )
                    )
                except (TypeError, ValueError):
                    _LOGGER.warning("Skipping unreadable grid example", exc_info=True)
        sliders: dict[str, int] = {}
        sliders_raw = payload.get("sliders")
        if isinstance(sliders_raw, dict):
            for key, value in sliders_raw.items():
                try:
                    sliders[str(key)] = int(value)
                except (TypeError, ValueError):
                    _LOGGER.warning("Skipping unreadable grid slider %s", key)
        reference_raw = payload.get("reference")
        reference = None
        if isinstance(reference_raw, dict):
            parsed: dict[str, float] = {}
            for key, value in reference_raw.items():
                try:
                    parsed[str(key)] = float(value)
                except (TypeError, ValueError):
                    _LOGGER.warning("Skipping unreadable reference field %s", key)
            reference = parsed or None
        influence = 0.5
        if "example_influence" in payload and payload.get("example_influence") is not None:
            try:
                influence = float(payload.get("example_influence"))
            except (TypeError, ValueError):
                _LOGGER.warning("Ignoring unreadable example influence")
        return GridCalibration(
            preset=str(payload.get("preset") or "balanced"),
            sliders=sliders,
            enabled_reason_types=tuple(str(item) for item in payload.get("enabled_reason_types") or ()),
            examples=tuple(examples[:500]),
            calibration_frame_keys=tuple(str(item) for item in payload.get("calibration_frame_keys") or ())
            if isinstance(payload.get("calibration_frame_keys"), list)
            else (),
            reference=reference,
            example_influence=influence,
            confirmed_at=str(payload["confirmed_at"]) if payload.get("confirmed_at") else None,
        )

    def confirmed_copy(self) -> "GridCalibration":
        return replace(self, confirmed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
