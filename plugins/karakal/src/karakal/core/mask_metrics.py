"""Mask, point, structural, pairwise, and comparison metrics."""

from __future__ import annotations

from .mask_primitives import (
    _boundary_mask,
    _clip01,
    _distance_transform,
    _has_distance_transform_backend,
    _mask_structure,
    _normalize_ratio,
    _weighted_mean,
)

from .metric_keys import (
    _normalized_comparison_pairs,
    combined_pair_metric_key,
    confidence_pair_metric_key,
    pair_metric_key,
)

from .repository_shared import (
    BCE_SCORE_CAP,
    COMBINED_PAIR_CONFIDENCE_WEIGHT,
    COMBINED_PAIR_OUTPUT_WEIGHT,
    CONFIDENCE_DIFF_THRESHOLD,
    CONFIDENCE_LOW_THRESHOLD,
    ComparisonMode,
    ComparisonPairSelection,
    EPS,
    GeometryMode,
    INTER_MODEL_POINT_SCORE_WEIGHTS,
    INTER_MODEL_POLYGON_SCORE_WEIGHTS,
    MASK_AGREEMENT_SCORE_WEIGHTS,
    POINT_AGREEMENT_SCORE_WEIGHTS,
    PointAgreementMetrics,
    PointDiagnosticMetrics,
    _get_ckdtree,
    _is_binary_like_probability,
    math,
    np,
)


def _dice(first: np.ndarray, second: np.ndarray) -> float:
    first_bool = np.asarray(first, dtype=bool)
    second_bool = np.asarray(second, dtype=bool)
    intersection = int(np.count_nonzero(first_bool & second_bool))
    total = int(np.count_nonzero(first_bool)) + int(np.count_nonzero(second_bool))
    if total == 0:
        return 1.0
    return float((2.0 * intersection + EPS) / (total + EPS))


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    first_bool = np.asarray(first, dtype=bool)
    second_bool = np.asarray(second, dtype=bool)
    intersection = int(np.count_nonzero(first_bool & second_bool))
    union = int(np.count_nonzero(first_bool | second_bool))
    if union == 0:
        return 1.0
    return float((intersection + EPS) / (union + EPS))


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size <= 0:
        return None
    return float(cols.mean(dtype=np.float64)), float(rows.mean(dtype=np.float64))


def _frame_diagonal(shape: tuple[int, int]) -> float:
    return float(max(EPS, math.hypot(float(shape[0]), float(shape[1]))))


def _centroid_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_center = _mask_centroid(first)
    second_center = _mask_centroid(second)
    diagonal = _frame_diagonal(tuple(int(v) for v in np.asarray(first).shape))
    if first_center is None and second_center is None:
        return 0.0
    if first_center is None or second_center is None:
        return diagonal
    return float(math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1]))


def _distance_similarity(distance: float, shape: tuple[int, int]) -> float:
    return float(1.0 - min(1.0, float(distance) / _frame_diagonal(shape)))


def _nearest_distances_between_coordinate_sets(
    coords_a: np.ndarray, coords_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coords_a = np.asarray(coords_a, dtype=np.float32)
    coords_b = np.asarray(coords_b, dtype=np.float32)
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return np.zeros(coords_a.shape[0], dtype=np.float64), np.zeros(coords_b.shape[0], dtype=np.float64)

    def _chunked_nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        pair_budget = 1_000_000
        chunk_size = max(1, min(int(source.shape[0]), int(max(1, pair_budget // max(1, int(target.shape[0]))))))
        nearest = np.empty(source.shape[0], dtype=np.float64)
        for start in range(0, source.shape[0], chunk_size):
            stop = min(source.shape[0], start + chunk_size)
            diff = np.asarray(source[start:stop, None, :] - target[None, :, :], dtype=np.float32)
            squared = np.sum(np.square(diff, dtype=np.float32), axis=2, dtype=np.float64)
            nearest[start:stop] = np.sqrt(np.min(squared, axis=1, initial=np.inf))
        return nearest

    cKDTree = _get_ckdtree()
    if cKDTree is not None:
        tree_a = cKDTree(coords_a)
        tree_b = cKDTree(coords_b)
        nearest_a = np.asarray(tree_b.query(coords_a, k=1)[0], dtype=np.float64)
        nearest_b = np.asarray(tree_a.query(coords_b, k=1)[0], dtype=np.float64)
        return nearest_a, nearest_b
    return _chunked_nearest_distances(coords_a, coords_b), _chunked_nearest_distances(coords_b, coords_a)


def _hausdorff_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_structure: dict[str, object] | None = None,
    second_structure: dict[str, object] | None = None,
) -> float:
    first_boundary = (
        np.asarray((first_structure or {}).get("boundary"), dtype=bool)
        if (first_structure or {}).get("boundary") is not None
        else _boundary_mask(first)
    )
    second_boundary = (
        np.asarray((second_structure or {}).get("boundary"), dtype=bool)
        if (second_structure or {}).get("boundary") is not None
        else _boundary_mask(second)
    )
    shape = tuple(int(v) for v in np.asarray(first_boundary).shape)
    diagonal = _frame_diagonal(shape)
    if not np.any(first_boundary) and not np.any(second_boundary):
        return 0.0
    if not np.any(first_boundary) or not np.any(second_boundary):
        return diagonal
    if _has_distance_transform_backend():
        dist_to_second = (
            np.asarray((second_structure or {}).get("boundary_dist"), dtype=np.float32)
            if (second_structure or {}).get("boundary_dist") is not None
            else _distance_transform(~second_boundary)
        )
        dist_to_first = (
            np.asarray((first_structure or {}).get("boundary_dist"), dtype=np.float32)
            if (first_structure or {}).get("boundary_dist") is not None
            else _distance_transform(~first_boundary)
        )
        directed_first = float(np.max(dist_to_second[first_boundary])) if np.any(first_boundary) else 0.0
        directed_second = float(np.max(dist_to_first[second_boundary])) if np.any(second_boundary) else 0.0
        return float(max(directed_first, directed_second))
    first_points = np.argwhere(first_boundary)
    second_points = np.argwhere(second_boundary)
    if first_points.size == 0 and second_points.size == 0:
        return 0.0
    if first_points.size == 0 or second_points.size == 0:
        return diagonal
    nearest_first, nearest_second = _nearest_distances_between_coordinate_sets(first_points, second_points)
    directed_first = float(np.max(nearest_first, initial=0.0))
    directed_second = float(np.max(nearest_second, initial=0.0))
    return float(max(directed_first, directed_second))


def _point_coordinates(points: tuple[object, ...]) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(
        [[float(getattr(point, "x", 0.0)), float(getattr(point, "y", 0.0))] for point in points], dtype=np.float32
    )


def _point_match_threshold(point_a: object, point_b: object, base_radius: float) -> float:
    _ = point_a, point_b
    return float(max(0.0, float(base_radius)))


def _match_point_sets(
    points_a: tuple[object, ...], points_b: tuple[object, ...], base_radius: float
) -> tuple[list[float], set[int], set[int]]:
    candidate_pairs: list[tuple[float, int, int]] = []
    cKDTree = _get_ckdtree()
    if cKDTree is not None and points_a and points_b:
        coords_a = _point_coordinates(points_a)
        coords_b = _point_coordinates(points_b)
        radius = max(0.0, float(base_radius))
        if coords_a.shape[0] > 0 and coords_b.shape[0] > 0 and radius >= 0.0:
            tree_b = cKDTree(coords_b)
            neighbors = tree_b.query_ball_point(coords_a, r=radius)
            for index_a, neighbor_indices in enumerate(neighbors):
                point_a = points_a[index_a]
                for index_b in neighbor_indices:
                    point_b = points_b[int(index_b)]
                    distance = float(
                        np.hypot(
                            float(getattr(point_a, "x", 0.0)) - float(getattr(point_b, "x", 0.0)),
                            float(getattr(point_a, "y", 0.0)) - float(getattr(point_b, "y", 0.0)),
                        )
                    )
                    if distance <= _point_match_threshold(point_a, point_b, base_radius):
                        candidate_pairs.append((distance, index_a, int(index_b)))
    else:
        for index_a, point_a in enumerate(points_a):
            for index_b, point_b in enumerate(points_b):
                distance = float(
                    np.hypot(
                        float(getattr(point_a, "x", 0.0)) - float(getattr(point_b, "x", 0.0)),
                        float(getattr(point_a, "y", 0.0)) - float(getattr(point_b, "y", 0.0)),
                    )
                )
                if distance <= _point_match_threshold(point_a, point_b, base_radius):
                    candidate_pairs.append((distance, index_a, index_b))
    candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    matched_distances: list[float] = []
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    for distance, index_a, index_b in candidate_pairs:
        if index_a in matched_a or index_b in matched_b:
            continue
        matched_a.add(index_a)
        matched_b.add(index_b)
        matched_distances.append(float(distance))
    return matched_distances, matched_a, matched_b


def _point_distance_scores(
    points_a: tuple[object, ...], points_b: tuple[object, ...], frame_shape: tuple[int, int]
) -> tuple[float, float]:
    coords_a = _point_coordinates(points_a)
    coords_b = _point_coordinates(points_b)
    if coords_a.shape[0] == 0 and coords_b.shape[0] == 0:
        return 0.0, 0.0
    diagonal = float(max(EPS, math.hypot(float(frame_shape[0]), float(frame_shape[1]))))
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return diagonal, diagonal
    nearest_a, nearest_b = _nearest_distances_between_coordinate_sets(coords_a, coords_b)
    chamfer = float((nearest_a.mean(dtype=np.float64) + nearest_b.mean(dtype=np.float64)) / 2.0)
    hausdorff = float(max(float(nearest_a.max(initial=0.0)), float(nearest_b.max(initial=0.0))))
    return chamfer, hausdorff


def _point_count_error(predicted_count: int, target_count: int) -> float:
    return float(_clip01(abs(int(predicted_count) - int(target_count)) / max(1.0, float(target_count))))


def _point_count_agreement(first_count: int, second_count: int) -> float:
    return float(
        1.0 - _clip01(abs(int(first_count) - int(second_count)) / max(1.0, float(max(first_count, second_count, 1))))
    )


def _mask_count_agreement(first_count: int, second_count: int) -> float:
    return float(
        1.0 - _clip01(abs(int(first_count) - int(second_count)) / max(1.0, float(max(first_count, second_count, 1))))
    )


def _symmetric_binary_cross_entropy(first_prob: np.ndarray, second_prob: np.ndarray) -> float:
    """Compute symmetric BCE between two probability maps."""

    first = np.clip(np.asarray(first_prob, dtype=np.float64), EPS, 1.0 - EPS)
    second = np.clip(np.asarray(second_prob, dtype=np.float64), EPS, 1.0 - EPS)
    forward = -(first * np.log(second) + (1.0 - first) * np.log(1.0 - second))
    backward = -(second * np.log(first) + (1.0 - second) * np.log(1.0 - first))
    return float(np.mean((forward + backward) * 0.5, dtype=np.float64)) if forward.size else 0.0


def _binary_like_bce_from_mismatch(mismatch_count: int, pixel_count: float) -> float:
    count = max(1.0, float(pixel_count))
    clip_eps = float(max(EPS, float(np.finfo(np.float32).eps)))
    mismatch_fraction = float(max(0, int(mismatch_count)) / count)
    match_fraction = float(max(0.0, 1.0 - mismatch_fraction))
    return float(mismatch_fraction * (-math.log(clip_eps)) + match_fraction * (-math.log1p(-clip_eps)))


def _dot_sum_float64(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.einsum("i,i->", np.ravel(first), np.ravel(second), dtype=np.float64))


def _polygon_bce_score(bce_value: float) -> float:
    capped = min(float(bce_value) / max(EPS, float(BCE_SCORE_CAP)), 1.0)
    return float(max(0.0, 100.0 * (1.0 - capped)))


def _aggregate_inter_model_polygon_scores(pairwise_rows: tuple[dict[str, object], ...]) -> dict[str, float]:
    """Aggregate pairwise polygon comparison metrics into one frame-level score bundle."""

    if not pairwise_rows:
        return {}
    iou = float(np.mean(np.asarray([float(row.get("iou", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    dice = float(np.mean(np.asarray([float(row.get("dice", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    bce = float(np.mean(np.asarray([float(row.get("bce", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    iou_score = float(max(0.0, min(iou, 1.0)) * 100.0)
    dice_score = float(max(0.0, min(dice, 1.0)) * 100.0)
    bce_score = _polygon_bce_score(bce)
    weight_sum = float(sum(INTER_MODEL_POLYGON_SCORE_WEIGHTS.values()))
    overall = (
        iou_score * float(INTER_MODEL_POLYGON_SCORE_WEIGHTS["iou"])
        + dice_score * float(INTER_MODEL_POLYGON_SCORE_WEIGHTS["dice"])
        + bce_score * float(INTER_MODEL_POLYGON_SCORE_WEIGHTS["bce"])
    ) / max(EPS, weight_sum)
    return {
        "iou": iou,
        "dice": dice,
        "bce": bce,
        "iou_score": iou_score,
        "dice_score": dice_score,
        "polygon_bce_score": bce_score,
        "overall_polygon_score": float(overall),
    }


def _aggregate_inter_model_point_scores(
    pairwise_rows: tuple[dict[str, object], ...], point_match_radius: float
) -> dict[str, float]:
    """Aggregate pairwise point comparison metrics into one frame-level score bundle."""

    if not pairwise_rows:
        return {}
    precision = float(
        np.mean(np.asarray([float(row.get("precision", 0.0)) for row in pairwise_rows], dtype=np.float64))
    )
    recall = float(np.mean(np.asarray([float(row.get("recall", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    f1 = float(np.mean(np.asarray([float(row.get("f1", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    mean_localization_distance = float(
        np.mean(np.asarray([float(row.get("mean_localization_error", 0.0)) for row in pairwise_rows], dtype=np.float64))
    )
    tp = float(np.mean(np.asarray([float(row.get("matched_count", 0.0)) for row in pairwise_rows], dtype=np.float64)))
    fp = float(
        np.mean(
            np.asarray(
                [
                    max(0.0, float(row.get("point_count_a", 0.0)) - float(row.get("matched_count", 0.0)))
                    for row in pairwise_rows
                ],
                dtype=np.float64,
            )
        )
    )
    fn = float(
        np.mean(
            np.asarray(
                [
                    max(0.0, float(row.get("point_count_b", 0.0)) - float(row.get("matched_count", 0.0)))
                    for row in pairwise_rows
                ],
                dtype=np.float64,
            )
        )
    )
    precision_score = float(max(0.0, min(precision, 1.0)) * 100.0)
    recall_score = float(max(0.0, min(recall, 1.0)) * 100.0)
    f1_score = float(max(0.0, min(f1, 1.0)) * 100.0)
    localization_score = float(
        max(0.0, 100.0 * (1.0 - min(mean_localization_distance / max(EPS, float(point_match_radius)), 1.0)))
    )
    weight_sum = float(sum(INTER_MODEL_POINT_SCORE_WEIGHTS.values()))
    overall = (
        precision_score * float(INTER_MODEL_POINT_SCORE_WEIGHTS["precision"])
        + recall_score * float(INTER_MODEL_POINT_SCORE_WEIGHTS["recall"])
        + f1_score * float(INTER_MODEL_POINT_SCORE_WEIGHTS["f1"])
        + localization_score * float(INTER_MODEL_POINT_SCORE_WEIGHTS["localization"])
    ) / max(EPS, weight_sum)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_localization_distance": mean_localization_distance,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
        "localization_score": localization_score,
        "overall_point_score": float(overall),
    }


def _paint_disk(target: np.ndarray, center_x: float, center_y: float, radius: float, value: float) -> None:
    if target.ndim != 2 or target.size == 0:
        return
    effective_radius = max(1.0, float(radius))
    x = int(round(float(center_x)))
    y = int(round(float(center_y)))
    y0 = max(0, int(math.floor(y - effective_radius)))
    y1 = min(target.shape[0], int(math.ceil(y + effective_radius + 1.0)))
    x0 = max(0, int(math.floor(x - effective_radius)))
    x1 = min(target.shape[1], int(math.ceil(x + effective_radius + 1.0)))
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = np.sqrt((yy - float(center_y)) ** 2 + (xx - float(center_x)) ** 2, dtype=np.float64)
    disk = distance <= effective_radius
    target[y0:y1, x0:x1] = np.maximum(target[y0:y1, x0:x1], np.asarray(disk, dtype=np.float32) * float(value))


def _point_map_from_view(
    prediction_view: object, shape: tuple[int, int] | None = None, *, scale: float = 1.0
) -> np.ndarray:
    base_shape = tuple(int(v) for v in (shape or getattr(prediction_view, "pred_gray").shape))
    result = np.zeros(base_shape, dtype=np.float32)
    for point in tuple(getattr(prediction_view, "points", ())):
        radius = max(1.0, float(getattr(point, "radius", 0.0)))
        score = float(getattr(point, "score", 1.0) or 1.0)
        _paint_disk(
            result,
            float(getattr(point, "x", 0.0)),
            float(getattr(point, "y", 0.0)),
            radius + 1.0,
            float(scale) * float(max(0.25, min(1.0, score))),
        )
    return np.clip(result, 0.0, 1.0)


def _point_model_agreement(first_view: object, second_view: object, point_match_radius: float) -> PointAgreementMetrics:
    points_a = tuple(getattr(first_view, "points", ()))
    points_b = tuple(getattr(second_view, "points", ()))
    matched_distances, _matched_a, _matched_b = _match_point_sets(points_a, points_b, point_match_radius)
    count_a = len(points_a)
    count_b = len(points_b)
    matched_count = len(matched_distances)
    false_positive_count = max(0, int(count_a - matched_count))
    false_negative_count = max(0, int(count_b - matched_count))
    if count_a == 0 and count_b == 0:
        precision = 1.0
        recall = 1.0
    else:
        precision = float(matched_count / max(1, count_a))
        recall = float(matched_count / max(1, count_b))
    f1 = 0.0 if precision + recall <= EPS else float((2.0 * precision * recall) / (precision + recall))
    mean_localization_error = (
        float(np.mean(np.asarray(matched_distances, dtype=np.float64))) if matched_count > 0 else 0.0
    )
    if matched_count <= 0:
        localization_agreement = 1.0 if count_a == 0 and count_b == 0 else 0.0
    else:
        localization_agreement = float(1.0 - min(1.0, mean_localization_error / max(EPS, float(point_match_radius))))
    count_agreement = _point_count_agreement(count_a, count_b)
    agreement_score = _weighted_mean(
        [
            (f1, POINT_AGREEMENT_SCORE_WEIGHTS["f1"]),
            (localization_agreement, POINT_AGREEMENT_SCORE_WEIGHTS["localization"]),
            (count_agreement, POINT_AGREEMENT_SCORE_WEIGHTS["count_agreement"]),
        ]
    )
    return PointAgreementMetrics(
        precision_at_radius=float(precision),
        recall_at_radius=float(recall),
        f1_at_radius=float(f1),
        mean_localization_error=float(mean_localization_error),
        localization_agreement=float(localization_agreement),
        count_agreement=float(count_agreement),
        matched_count=int(matched_count),
        true_positive_count=int(matched_count),
        false_positive_count=int(false_positive_count),
        false_negative_count=int(false_negative_count),
        point_count_a=int(count_a),
        point_count_b=int(count_b),
        agreement_score=float(agreement_score),
    )


def _prepare_mask_pairwise_descriptors(
    probabilities_by_model: dict[str, np.ndarray],
    masks_by_model: dict[str, np.ndarray],
    model_structures: dict[str, dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    """Precompute reusable per-model structures for symmetric mask agreement metrics."""

    descriptors: dict[str, dict[str, object]] = {}
    for model_id, probability in probabilities_by_model.items():
        prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
        binary_like = _is_binary_like_probability(prob)
        prob_binary_mask = np.asarray(prob >= 0.5, dtype=bool) if binary_like else None
        prob_clipped = None
        log_prob = None
        log_inv_prob = None
        log_prob_sum = 0.0
        log_inv_prob_sum = 0.0
        if not binary_like:
            clip_eps = np.float32(max(EPS, float(np.finfo(np.float32).eps)))
            prob_clipped = np.clip(prob, clip_eps, np.float32(1.0) - clip_eps)
            log_prob = np.log(prob_clipped)
            log_inv_prob = np.log1p(-prob_clipped)
            log_prob_sum = float(np.sum(log_prob, dtype=np.float64))
            log_inv_prob_sum = float(np.sum(log_inv_prob, dtype=np.float64))
        prob_sum = float(np.sum(prob, dtype=np.float64))
        prob_sq_sum = prob_sum if binary_like else float(np.sum(np.square(prob, dtype=np.float32), dtype=np.float64))
        pixel_count = max(1.0, float(prob.size))
        mask = np.asarray(masks_by_model.get(model_id), dtype=bool)
        current_structure = (model_structures or {}).get(str(model_id)) or _mask_structure(
            mask, include_boundary_distance=True
        )
        boundary = (
            np.asarray(current_structure.get("boundary"), dtype=bool)
            if current_structure.get("boundary") is not None
            else _boundary_mask(mask)
        )
        dist_to_boundary = current_structure.get("boundary_dist")
        if dist_to_boundary is not None:
            dist_to_boundary = np.asarray(dist_to_boundary, dtype=np.float32)
        elif np.any(boundary) and _has_distance_transform_backend():
            dist_to_boundary = _distance_transform(~boundary)
        descriptors[str(model_id)] = {
            "prob": prob,
            "prob_sum": prob_sum,
            "prob_sq_sum": prob_sq_sum,
            "prob_mean": float(prob_sum / pixel_count),
            "prob_var": float(max(0.0, (prob_sq_sum / pixel_count) - (prob_sum / pixel_count) ** 2)),
            "prob_binary_like": bool(binary_like),
            "prob_binary_mask": prob_binary_mask,
            "log_prob": log_prob,
            "log_inv_prob": log_inv_prob,
            "log_prob_sum": log_prob_sum,
            "log_inv_prob_sum": log_inv_prob_sum,
            "mask": mask,
            "mask_area": int(np.count_nonzero(mask)),
            "boundary": boundary,
            "boundary_dist": dist_to_boundary,
            "centroid": _mask_centroid(mask),
            "structure": current_structure,
            "shape": tuple(int(v) for v in mask.shape),
        }
    return descriptors


def _pairwise_mask_metrics(
    first: dict[str, object],
    second: dict[str, object],
) -> dict[str, float]:
    """Compute all symmetric mask agreement metrics from precomputed per-model descriptors."""

    first_prob = np.asarray(first["prob"], dtype=np.float32)
    second_prob = np.asarray(second["prob"], dtype=np.float32)
    first_mask = np.asarray(first["mask"], dtype=bool)
    second_mask = np.asarray(second["mask"], dtype=bool)
    shape = tuple(int(v) for v in first["shape"])

    intersection_mask = int(np.count_nonzero(first_mask & second_mask))
    area_first = int(first["mask_area"])
    area_second = int(second["mask_area"])
    union_mask = int(area_first + area_second - intersection_mask)

    first_binary_like = bool(first.get("prob_binary_like", False))
    second_binary_like = bool(second.get("prob_binary_like", False))
    both_binary_like = first_binary_like and second_binary_like
    prob_binary_intersection = None
    prob_binary_area_first = None
    prob_binary_area_second = None
    if both_binary_like:
        first_prob_mask = np.asarray(first.get("prob_binary_mask"), dtype=bool)
        second_prob_mask = np.asarray(second.get("prob_binary_mask"), dtype=bool)
        prob_binary_intersection = int(np.count_nonzero(first_prob_mask & second_prob_mask))
        prob_binary_area_first = int(np.count_nonzero(first_prob_mask))
        prob_binary_area_second = int(np.count_nonzero(second_prob_mask))
        prob_intersection = float(prob_binary_intersection)
    else:
        prob_intersection = _dot_sum_float64(first_prob, second_prob)
    prob_sum_first = float(first["prob_sum"])
    prob_sum_second = float(second["prob_sum"])
    prob_union = float(prob_sum_first + prob_sum_second - prob_intersection)
    pixel_count = max(1.0, float(first_prob.size))

    soft_dice = (
        1.0
        if (prob_sum_first + prob_sum_second) <= EPS
        else float((2.0 * prob_intersection + EPS) / (prob_sum_first + prob_sum_second + EPS))
    )
    soft_iou = 1.0 if prob_union <= EPS else float((prob_intersection + EPS) / (prob_union + EPS))

    mu_first = float(first.get("prob_mean", 0.0))
    mu_second = float(second.get("prob_mean", 0.0))
    sigma_first = float(max(0.0, float(first.get("prob_var", 0.0))))
    sigma_second = float(max(0.0, float(second.get("prob_var", 0.0))))
    sigma_cross = float((prob_intersection / pixel_count) - (mu_first * mu_second))
    c1 = 0.01**2
    c2 = 0.03**2
    denominator = (mu_first**2 + mu_second**2 + c1) * (sigma_first + sigma_second + c2)
    ssim = (
        1.0
        if denominator <= EPS and np.allclose(first_prob, second_prob)
        else (
            0.0
            if denominator <= EPS
            else float(_clip01(((2.0 * mu_first * mu_second + c1) * (2.0 * sigma_cross + c2)) / denominator))
        )
    )

    dice = (
        1.0
        if (area_first + area_second) == 0
        else float((2.0 * intersection_mask + EPS) / (area_first + area_second + EPS))
    )
    iou = 1.0 if union_mask == 0 else float((intersection_mask + EPS) / (union_mask + EPS))

    diagonal = _frame_diagonal(shape)
    first_center = first["centroid"]
    second_center = second["centroid"]
    if first_center is None and second_center is None:
        centroid_distance = 0.0
    elif first_center is None or second_center is None:
        centroid_distance = diagonal
    else:
        centroid_distance = float(math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1]))
    centroid_similarity = _distance_similarity(centroid_distance, shape)

    first_boundary = np.asarray(first["boundary"], dtype=bool)
    second_boundary = np.asarray(second["boundary"], dtype=bool)
    if not np.any(first_boundary) and not np.any(second_boundary):
        hausdorff_distance = 0.0
    elif not np.any(first_boundary) or not np.any(second_boundary):
        hausdorff_distance = diagonal
    elif first.get("boundary_dist") is not None and second.get("boundary_dist") is not None:
        directed_first = (
            float(np.max(np.asarray(second["boundary_dist"], dtype=np.float32)[first_boundary]))
            if np.any(first_boundary)
            else 0.0
        )
        directed_second = (
            float(np.max(np.asarray(first["boundary_dist"], dtype=np.float32)[second_boundary]))
            if np.any(second_boundary)
            else 0.0
        )
        hausdorff_distance = float(max(directed_first, directed_second))
    else:
        hausdorff_distance = _hausdorff_distance(first_mask, second_mask)
    hausdorff_similarity = _distance_similarity(hausdorff_distance, shape)

    if both_binary_like:
        mismatch_count = int(
            (prob_binary_area_first or 0) + (prob_binary_area_second or 0) - 2 * (prob_binary_intersection or 0)
        )
        mismatch_fraction = float(max(0, mismatch_count) / pixel_count)
        mae = mismatch_fraction
        rmse = float(math.sqrt(mismatch_fraction))
        bce = _binary_like_bce_from_mismatch(mismatch_count, pixel_count)
    else:
        diff = first_prob - second_prob
        mae = float(np.mean(np.abs(diff), dtype=np.float64)) if diff.size else 0.0
        rmse = float(np.sqrt(np.mean(np.square(diff, dtype=np.float32), dtype=np.float64))) if diff.size else 0.0
        first_log = np.asarray(first.get("log_prob"), dtype=np.float32)
        first_log_inv = np.asarray(first.get("log_inv_prob"), dtype=np.float32)
        second_log = np.asarray(second.get("log_prob"), dtype=np.float32)
        second_log_inv = np.asarray(second.get("log_inv_prob"), dtype=np.float32)
        forward_sum = (
            _dot_sum_float64(first_prob, second_log)
            + float(second.get("log_inv_prob_sum", 0.0))
            - _dot_sum_float64(first_prob, second_log_inv)
        )
        backward_sum = (
            _dot_sum_float64(second_prob, first_log)
            + float(first.get("log_inv_prob_sum", 0.0))
            - _dot_sum_float64(second_prob, first_log_inv)
        )
        bce = float(-0.5 * (forward_sum + backward_sum) / pixel_count) if first_prob.size else 0.0

    count_agreement = _mask_count_agreement(
        int(
            np.asarray(first["structure"]["component_count"]).item()
            if hasattr(first["structure"]["component_count"], "item")
            else first["structure"]["component_count"]
        ),
        int(
            np.asarray(second["structure"]["component_count"]).item()
            if hasattr(second["structure"]["component_count"], "item")
            else second["structure"]["component_count"]
        ),
    )
    agreement_score = _weighted_mean(
        [
            (float(soft_dice), MASK_AGREEMENT_SCORE_WEIGHTS["soft_dice"]),
            (float(soft_iou), MASK_AGREEMENT_SCORE_WEIGHTS["soft_iou"]),
            (float(ssim), MASK_AGREEMENT_SCORE_WEIGHTS["ssim"]),
            (float(dice), MASK_AGREEMENT_SCORE_WEIGHTS["dice"]),
            (float(iou), MASK_AGREEMENT_SCORE_WEIGHTS["iou"]),
            (float(hausdorff_similarity), MASK_AGREEMENT_SCORE_WEIGHTS["hausdorff_term"]),
            (float(centroid_similarity), MASK_AGREEMENT_SCORE_WEIGHTS["centroid_term"]),
        ]
    )

    return {
        "soft_dice": float(soft_dice),
        "soft_iou": float(soft_iou),
        "ssim": float(ssim),
        "dice": float(dice),
        "iou": float(iou),
        "hausdorff_distance": float(hausdorff_distance),
        "hausdorff_similarity": float(hausdorff_similarity),
        "centroid_distance": float(centroid_distance),
        "centroid_similarity": float(centroid_similarity),
        "mae": float(mae),
        "rmse": float(rmse),
        "bce": float(bce),
        "count_agreement": float(count_agreement),
        "agreement_score": float(agreement_score),
    }


def _pairwise_model_comparisons(
    probabilities_by_model: dict[str, np.ndarray],
    masks_by_model: dict[str, np.ndarray],
    *,
    geometry_mode: GeometryMode = GeometryMode.MASK,
    model_views: dict[str, object] | None = None,
    model_structures: dict[str, dict[str, object]] | None = None,
    point_match_radius: float = 3.0,
) -> tuple[dict[str, object], ...]:
    model_ids = list(probabilities_by_model.keys())
    rows: list[dict[str, object]] = []
    current_views = model_views or {}
    mask_descriptors = (
        _prepare_mask_pairwise_descriptors(probabilities_by_model, masks_by_model, model_structures=model_structures)
        if geometry_mode != GeometryMode.POINT
        else {}
    )
    for index_a, model_a in enumerate(model_ids):
        for model_b in model_ids[index_a + 1 :]:
            if geometry_mode == GeometryMode.POINT and model_a in current_views and model_b in current_views:
                metrics = _point_model_agreement(current_views[model_a], current_views[model_b], point_match_radius)
                rows.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "precision": float(metrics.precision_at_radius),
                        "recall": float(metrics.recall_at_radius),
                        "f1": float(metrics.f1_at_radius),
                        "mean_localization_error": float(metrics.mean_localization_error),
                        "localization_agreement": float(metrics.localization_agreement),
                        "count_agreement": float(metrics.count_agreement),
                        "matched_count": int(metrics.matched_count),
                        "tp": int(metrics.true_positive_count),
                        "fp": int(metrics.false_positive_count),
                        "fn": int(metrics.false_negative_count),
                        "point_count_a": int(metrics.point_count_a),
                        "point_count_b": int(metrics.point_count_b),
                        "agreement_score": float(metrics.agreement_score),
                    }
                )
                continue

            metrics_row = _pairwise_mask_metrics(mask_descriptors[str(model_a)], mask_descriptors[str(model_b)])
            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    **metrics_row,
                }
            )
    return tuple(rows)


def _pairwise_rows_by_ordered_pair(
    pairwise_rows: tuple[dict[str, object], ...],
) -> dict[tuple[str, str], dict[str, object]]:
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for row in pairwise_rows:
        model_a = str(row.get("model_a") or "")
        model_b = str(row.get("model_b") or "")
        if not model_a or not model_b:
            continue
        lookup[(model_a, model_b)] = row
        lookup[(model_b, model_a)] = row
    return lookup


def _configured_pair_metric_values(
    pairs: tuple[ComparisonPairSelection, ...],
    pairwise_rows: tuple[dict[str, object], ...],
    masks_by_model: dict[str, np.ndarray],
) -> dict[str, float]:
    values: dict[str, float] = {}
    if not pairs:
        return values
    rows_by_pair = _pairwise_rows_by_ordered_pair(pairwise_rows)
    for pair in _normalized_comparison_pairs(pairs):
        model_a = str(pair.model_a_id)
        model_b = str(pair.model_b_id)
        row = rows_by_pair.get((model_a, model_b), {})
        for operation in pair.operations:
            key = pair_metric_key(model_a, model_b, operation)
            if operation == "xor":
                mask_a = masks_by_model.get(model_a)
                mask_b = masks_by_model.get(model_b)
                if mask_a is not None and mask_b is not None:
                    values[key] = float(
                        np.mean(
                            np.logical_xor(np.asarray(mask_a, dtype=bool), np.asarray(mask_b, dtype=bool)),
                            dtype=np.float64,
                        )
                    )
                    continue
                agreement = row.get("agreement_score")
                if agreement is not None:
                    values[key] = float(_clip01(1.0 - float(agreement)))
                continue
            metric_value = row.get(operation)
            if metric_value is not None and math.isfinite(float(metric_value)):
                values[key] = float(metric_value)
    return values


def _confidence_pair_metrics(confidence_a: np.ndarray, confidence_b: np.ndarray) -> dict[str, float]:
    first_raw = np.asarray(confidence_a, dtype=np.float32)
    second_raw = np.asarray(confidence_b, dtype=np.float32)
    first = np.clip(first_raw, 0.0, 1.0)
    second = np.clip(second_raw, 0.0, 1.0)
    if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape or first.size == 0:
        return {}
    valid = np.isfinite(first_raw) & np.isfinite(second_raw)
    if not np.any(valid):
        return {}
    first = np.clip(first_raw[valid], 0.0, 1.0)
    second = np.clip(second_raw[valid], 0.0, 1.0)
    delta = first - second
    abs_delta = np.abs(delta)
    low_a = first < CONFIDENCE_LOW_THRESHOLD
    low_b = second < CONFIDENCE_LOW_THRESHOLD
    union = np.logical_or(low_a, low_b)
    intersection = np.logical_and(low_a, low_b)
    low_iou = 0.0 if not np.any(union) else float(np.mean(intersection[union], dtype=np.float64))
    flat_a = first.reshape(-1)
    flat_b = second.reshape(-1)
    if flat_a.size < 2 or float(np.std(flat_a)) <= EPS or float(np.std(flat_b)) <= EPS:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(flat_a, flat_b)[0, 1])
    return {
        "mae": float(np.mean(abs_delta, dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(delta.astype(np.float64, copy=False)), dtype=np.float64))),
        "mean_delta": float(np.mean(delta, dtype=np.float64)),
        "correlation": correlation,
        "low_iou": low_iou,
        "disagreement": float(np.mean(abs_delta > CONFIDENCE_DIFF_THRESHOLD, dtype=np.float64)),
    }


def _configured_confidence_pair_metric_values(
    pairs: tuple[ComparisonPairSelection, ...],
    output_probabilities_by_model: dict[str, np.ndarray],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for pair in _normalized_comparison_pairs(pairs):
        model_a = str(pair.model_a_id)
        model_b = str(pair.model_b_id)
        confidence_a = output_probabilities_by_model.get(model_a)
        confidence_b = output_probabilities_by_model.get(model_b)
        if confidence_a is None or confidence_b is None:
            continue
        metrics = _confidence_pair_metrics(confidence_a, confidence_b)
        for operation, value in metrics.items():
            if math.isfinite(float(value)):
                values[confidence_pair_metric_key(model_a, model_b, operation)] = float(value)
    return values


def _configured_combined_pair_metric_values(
    pairs: tuple[ComparisonPairSelection, ...],
    pair_metric_values: dict[str, float],
    confidence_pair_metric_values: dict[str, float],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for pair in _normalized_comparison_pairs(pairs):
        model_a = str(pair.model_a_id)
        model_b = str(pair.model_b_id)
        output_value = pair_metric_values.get(pair_metric_key(model_a, model_b, "xor"))
        if output_value is None:
            output_value = pair_metric_values.get(pair_metric_key(model_a, model_b, "iou"))
            if output_value is not None:
                output_value = 1.0 - float(output_value)
        if output_value is None:
            output_value = pair_metric_values.get(pair_metric_key(model_a, model_b, "dice"))
            if output_value is not None:
                output_value = 1.0 - float(output_value)
        if output_value is None:
            continue
        confidence_value = confidence_pair_metric_values.get(
            confidence_pair_metric_key(model_a, model_b, "disagreement")
        )
        if confidence_value is None:
            continue
        values[combined_pair_metric_key(model_a, model_b)] = float(
            _clip01(
                COMBINED_PAIR_OUTPUT_WEIGHT * float(output_value)
                + COMBINED_PAIR_CONFIDENCE_WEIGHT * float(confidence_value)
            )
        )
    return values


def _point_feature_vector(prediction_view: object) -> dict[str, float]:
    points = tuple(getattr(prediction_view, "points", ()))
    point_count = len(points)
    image_shape = tuple(int(v) for v in getattr(prediction_view, "pred_gray").shape)
    image_area = max(1.0, float(image_shape[0] * image_shape[1]))
    mean_radius = (
        float(np.mean([float(getattr(point, "radius", 0.0)) for point in points], dtype=np.float64)) if points else 0.0
    )
    mean_peak_intensity = (
        float(np.mean([float(getattr(point, "peak_intensity", 0.0)) for point in points], dtype=np.float64) / 255.0)
        if points
        else 0.0
    )
    mean_local_snr = (
        float(np.mean([float(getattr(point, "local_snr", 0.0)) for point in points], dtype=np.float64))
        if points
        else 0.0
    )
    mean_blob_score = (
        float(np.mean([float(getattr(point, "blob_score", 0.0)) for point in points], dtype=np.float64))
        if points
        else 0.0
    )
    return {
        "area_fraction": float(point_count / image_area),
        "component_count": float(point_count),
        "mean_component_area": float(mean_peak_intensity),
        "skeleton_length": float(mean_radius),
        "endpoint_count": float(_normalize_ratio(mean_local_snr)),
        "branchpoint_count": float(_normalize_ratio(mean_blob_score)),
    }


def _point_diagnostic_metrics(prediction_view: object) -> PointDiagnosticMetrics:
    points = tuple(getattr(prediction_view, "points", ()))
    mean_radius = (
        float(np.mean([float(getattr(point, "radius", 0.0)) for point in points], dtype=np.float64)) if points else 0.0
    )
    mean_peak_intensity = (
        float(np.mean([float(getattr(point, "peak_intensity", 0.0)) for point in points], dtype=np.float64) / 255.0)
        if points
        else 0.0
    )
    mean_local_snr = (
        float(np.mean([float(getattr(point, "local_snr", 0.0)) for point in points], dtype=np.float64))
        if points
        else 0.0
    )
    false_spot_ratio = (
        float(
            _clip01(
                np.mean(
                    [1.0 if float(getattr(point, "local_snr", 0.0)) < 0.5 else 0.0 for point in points],
                    dtype=np.float64,
                )
            )
        )
        if points
        else 0.0
    )
    proxy_score = _weighted_mean(
        [
            (_clip01(mean_peak_intensity), 0.45),
            (_clip01(_normalize_ratio(mean_local_snr)), 0.35),
            (1.0 - false_spot_ratio, 0.20),
        ]
    )
    return PointDiagnosticMetrics(
        point_count=int(len(points)),
        mean_radius=float(mean_radius),
        mean_peak_intensity=float(mean_peak_intensity),
        false_spot_ratio=float(false_spot_ratio),
        proxy_score=float(proxy_score),
    )


def _infer_geometry_mode(prediction_view: object) -> GeometryMode:
    points = tuple(getattr(prediction_view, "points", ()))
    region_summary = getattr(prediction_view, "region_summary", None)
    area_fraction = float(getattr(region_summary, "area_fraction", 0.0)) if region_summary is not None else 0.0
    mean_area = float(getattr(region_summary, "mean_area", 0.0)) if region_summary is not None else 0.0
    if len(points) > 0 and area_fraction <= 0.08 and mean_area <= 64.0:
        return GeometryMode.POINT
    return GeometryMode.MASK


def _consensus_probability(probabilities: list[np.ndarray]) -> np.ndarray:
    if not probabilities:
        return np.zeros((1, 1), dtype=np.float32)
    stacked = np.stack(probabilities, axis=0).astype(np.float32)
    return np.mean(stacked, axis=0, dtype=np.float32)


def _entropy_map(probability: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    entropy = -(prob * np.log(prob) + (1.0 - prob) * np.log(1.0 - prob)) / np.log(2.0)
    return np.asarray(entropy, dtype=np.float32)


def _disagreement_score(probabilities: list[np.ndarray], masks: list[np.ndarray]) -> tuple[float, np.ndarray]:
    if not probabilities:
        return 0.0, np.zeros((1, 1), dtype=np.float32)
    if len(probabilities) == 1:
        variance_map = np.zeros_like(probabilities[0], dtype=np.float32)
        return 0.0, variance_map
    pairwise: list[float] = []
    for index_a in range(len(masks)):
        for index_b in range(index_a + 1, len(masks)):
            pairwise.append(1.0 - _dice(masks[index_a], masks[index_b]))
    stacked = np.stack(probabilities, axis=0).astype(np.float32)
    variance_map = np.clip(4.0 * np.var(stacked, axis=0, dtype=np.float32), 0.0, 1.0)
    score = 0.6 * float(np.mean(pairwise, dtype=np.float64)) + 0.4 * float(np.mean(variance_map, dtype=np.float64))
    return float(_clip01(score)), variance_map.astype(np.float32)


def _structural_feature_vector(
    consensus_mask: np.ndarray, *, structure: dict[str, object] | None = None
) -> dict[str, float]:
    current_structure = structure or _mask_structure(consensus_mask)
    return {
        "area_fraction": float(current_structure["area_fraction"]),
        "component_count": float(current_structure["component_count"]),
        "mean_component_area": float(current_structure["mean_component_area"]),
        "skeleton_length": float(current_structure["skeleton_length"]),
        "endpoint_count": float(current_structure["endpoint_count"]),
        "branchpoint_count": float(current_structure["branchpoint_count"]),
    }


def _robust_feature_bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return median, mad if mad > EPS else float(np.std(array) + EPS)


def _structural_anomaly_score(vector: dict[str, float], bounds: dict[str, tuple[float, float]]) -> float:
    terms: list[float] = []
    for key, value in vector.items():
        median, scale = bounds.get(key, (0.0, 1.0))
        z_score = abs(float(value) - median) / max(EPS, scale)
        terms.append(_clip01(z_score / 3.0))
    return float(np.mean(terms, dtype=np.float64)) if terms else 0.0


def compute_comparison_score(first: np.ndarray, second: np.ndarray, mode: ComparisonMode) -> float:
    """Legacy-lite helper that returns only scalar score for a comparison mode."""

    if mode == ComparisonMode.GRAYSCALE_DIFF:
        first_gray = np.asarray(first, dtype=np.float32)
        second_gray = np.asarray(second, dtype=np.float32)
        if first_gray.size == 0 or second_gray.size == 0:
            return 0.0
        diff = np.abs(first_gray - second_gray)
        scale = 255.0 if max(float(np.max(first_gray)), float(np.max(second_gray))) > 1.0 else 1.0
        return float(np.mean(diff, dtype=np.float64) / scale)
    first_bool = np.asarray(first, dtype=bool)
    second_bool = np.asarray(second, dtype=bool)
    if first_bool.size == 0 or second_bool.size == 0 or mode == ComparisonMode.OVERLAY_ONLY:
        return 0.0
    if mode in {ComparisonMode.XOR, ComparisonMode.DISAGREEMENT}:
        return float(np.mean(np.not_equal(first_bool, second_bool), dtype=np.float64))
    if mode == ComparisonMode.FIRST_MINUS_SECOND:
        return float(np.mean(np.logical_and(first_bool, np.logical_not(second_bool)), dtype=np.float64))
    if mode == ComparisonMode.IOU:
        intersection = float(np.count_nonzero(np.logical_and(first_bool, second_bool)))
        union = float(np.count_nonzero(np.logical_or(first_bool, second_bool)))
        return 1.0 if union <= EPS else float(intersection / union)
    if mode == ComparisonMode.DICE:
        intersection = float(np.count_nonzero(np.logical_and(first_bool, second_bool)))
        total = float(np.count_nonzero(first_bool) + np.count_nonzero(second_bool))
        return 1.0 if total <= EPS else float((2.0 * intersection) / total)
    if mode == ComparisonMode.BCE:
        eps = 1e-6
        first_float = np.asarray(first, dtype=np.float32)
        second_float = np.asarray(second, dtype=np.float32)
        if first_float.size > 0 and float(np.nanmax(first_float)) > 1.0:
            first_float = first_float / 255.0
        if second_float.size > 0 and float(np.nanmax(second_float)) > 1.0:
            second_float = second_float / 255.0
        first_float = np.clip(first_float, eps, 1.0 - eps)
        second_float = np.clip(second_float, eps, 1.0 - eps)
        forward = -(first_float * np.log(second_float) + (1.0 - first_float) * np.log(1.0 - second_float))
        backward = -(second_float * np.log(first_float) + (1.0 - second_float) * np.log(1.0 - first_float))
        return float(np.mean((forward + backward) * 0.5, dtype=np.float64)) if forward.size else 0.0
    return float(np.mean(np.logical_and(np.logical_not(first_bool), second_bool), dtype=np.float64))


def compute_comparison(first: np.ndarray, second: np.ndarray, mode: ComparisonMode) -> tuple[np.ndarray, float]:
    if mode == ComparisonMode.GRAYSCALE_DIFF:
        first_gray = np.asarray(first, dtype=np.float32)
        second_gray = np.asarray(second, dtype=np.float32)
        heatmap = np.abs(first_gray - second_gray)
        if heatmap.size == 0:
            return heatmap.astype(np.float32), 0.0
        scale = 255.0 if max(float(np.max(first_gray)), float(np.max(second_gray))) > 1.0 else 1.0
        if scale > 1.0:
            heatmap = heatmap / scale
        return heatmap.astype(np.float32, copy=False), float(np.mean(heatmap, dtype=np.float64))
    first_bool = np.asarray(first, dtype=bool)
    second_bool = np.asarray(second, dtype=bool)
    if mode == ComparisonMode.IOU:
        intersection = np.logical_and(first_bool, second_bool)
        union = np.logical_or(first_bool, second_bool)
        heatmap = np.where(union, np.where(intersection, 1.0, 0.6), 0.0).astype(np.float32)
        return heatmap, compute_comparison_score(first_bool, second_bool, mode)
    if mode == ComparisonMode.DICE:
        intersection = np.logical_and(first_bool, second_bool)
        union = np.logical_or(first_bool, second_bool)
        heatmap = np.where(union, np.where(intersection, 1.0, 0.35), 0.0).astype(np.float32)
        return heatmap, compute_comparison_score(first_bool, second_bool, mode)
    if mode == ComparisonMode.BCE:
        eps = 1e-6
        first_float = np.asarray(first, dtype=np.float32)
        second_float = np.asarray(second, dtype=np.float32)
        if first_float.size > 0 and float(np.nanmax(first_float)) > 1.0:
            first_float = first_float / 255.0
        if second_float.size > 0 and float(np.nanmax(second_float)) > 1.0:
            second_float = second_float / 255.0
        first_float = np.clip(first_float, eps, 1.0 - eps)
        second_float = np.clip(second_float, eps, 1.0 - eps)
        forward = -(first_float * np.log(second_float) + (1.0 - first_float) * np.log(1.0 - second_float))
        backward = -(second_float * np.log(first_float) + (1.0 - second_float) * np.log(1.0 - first_float))
        heatmap = np.asarray((forward + backward) * 0.5, dtype=np.float32)
        display = np.clip(1.0 - np.exp(-heatmap), 0.0, 1.0)
        return display.astype(np.float32, copy=False), float(
            np.mean(heatmap, dtype=np.float64)
        ) if heatmap.size else 0.0
    if mode == ComparisonMode.OVERLAY_ONLY:
        return np.zeros_like(first_bool, dtype=np.float32), 0.0
    elif mode in {ComparisonMode.XOR, ComparisonMode.DISAGREEMENT}:
        mask = np.logical_xor(first_bool, second_bool)
    elif mode == ComparisonMode.FIRST_MINUS_SECOND:
        mask = np.logical_and(first_bool, np.logical_not(second_bool))
    else:
        mask = np.logical_and(np.logical_not(first_bool), second_bool)
    return mask.astype(np.float32), float(np.mean(mask, dtype=np.float64))
