"""Polygon confidence candidate extraction and refinement."""

from __future__ import annotations

from .confidence_analysis import (
    _confidence_map_from_probability,
    _frame_uncertainty_components_from_probability,
    _internal_confidence_probability_map,
    _polygon_confidence_config,
    _uncertainty_map_from_probability,
)

from .mask_primitives import (
    _binary_dilate,
    _binary_dilate_rect,
    _binary_erode,
    _boundary_mask,
    _clip01,
    _completion_radii_for_mask,
    _label_components,
    _thin_bridge_map,
    _weighted_mean,
    skeletonize,
)

from .repository_shared import (
    EPS,
    MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    POLYGON_CONFIDENCE_COMPLETION_BRIDGE_RADIUS,
    POLYGON_CONFIDENCE_COMPLETION_LOW_RATIO,
    POLYGON_CONFIDENCE_COMPLETION_MAJOR_SCALE,
    POLYGON_CONFIDENCE_COMPLETION_WEAK_RATIO,
    POLYGON_CONFIDENCE_HYSTERESIS_FLOOR,
    POLYGON_CONFIDENCE_SPILL_TRIM_DELTA,
    POLYGON_CONFIDENCE_SUMMARY_CORE,
    POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
    POLYGON_SUPPORT_THRESHOLD,
    PolygonConfidenceDebugCandidate,
    PolygonConfidenceDebugData,
    PolygonConfidenceMetrics,
    PolygonConfidencePipelineConfig,
    PolygonObjectConfidence,
    Sequence,
    dataclass,
    ndi,
    lru_cache,
    np,
    perf_counter,
)


@dataclass(slots=True)
class _PolygonConfidenceCandidate:
    """Internal candidate mask proposed by one confidence extraction branch."""

    mask: np.ndarray
    source_branches: tuple[str, ...]
    area: int
    bbox: tuple[int, int, int, int]
    aspect_ratio: float
    elongation: float
    extent: float
    peak_probability: float
    mean_probability: float


def _polygon_confidence_weak_threshold(
    strong_threshold: float, config: PolygonConfidencePipelineConfig | None = None
) -> float:
    cfg = config or _polygon_confidence_config()
    strong = float(max(0.0, min(1.0, strong_threshold)))
    weak = max(float(cfg.hysteresis_low_floor), strong * float(cfg.hysteresis_low_ratio))
    return float(min(strong, weak))


def _polygon_confidence_completion_threshold(strong_threshold: float, weak_threshold: float) -> float:
    strong = float(max(0.0, min(1.0, strong_threshold)))
    weak = float(max(0.0, min(1.0, weak_threshold)))
    completion = max(
        POLYGON_CONFIDENCE_HYSTERESIS_FLOOR,
        min(weak * POLYGON_CONFIDENCE_COMPLETION_WEAK_RATIO, strong * POLYGON_CONFIDENCE_COMPLETION_LOW_RATIO),
    )
    return float(min(strong, completion))


def _normalize_probability_map(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0).astype(np.float32)


def _smooth_probability_map(probability: np.ndarray, config: PolygonConfidencePipelineConfig) -> np.ndarray:
    prob = _normalize_probability_map(probability)
    if ndi is None:
        return prob
    result = prob
    median_radius = max(0, int(config.median_radius))
    if median_radius > 0:
        size = 2 * median_radius + 1
        result = np.asarray(ndi.median_filter(result, size=size, mode="nearest"), dtype=np.float32)
    sigma = max(0.0, float(config.gaussian_sigma))
    if sigma > EPS:
        result = np.asarray(ndi.gaussian_filter(result, sigma=sigma, mode="nearest"), dtype=np.float32)
    return _normalize_probability_map(result)


def _locally_normalized_probability_map(probability: np.ndarray, config: PolygonConfidencePipelineConfig) -> np.ndarray:
    prob = _normalize_probability_map(probability)
    radius = max(1, int(config.local_normalization_radius))
    mean_map = _local_mean_map(prob, radius=radius)
    sq_mean_map = _local_mean_map(np.square(prob, dtype=np.float32), radius=radius)
    variance_map = np.clip(sq_mean_map - np.square(mean_map, dtype=np.float32), 0.0, None)
    std_map = np.sqrt(variance_map + EPS, dtype=np.float32)
    normalized = np.clip(0.5 + 0.25 * ((prob - mean_map) / np.maximum(std_map, 0.05)), 0.0, 1.0)
    strength = float(max(0.0, min(1.0, config.local_normalization_strength)))
    enhanced = np.clip(prob + strength * (normalized - 0.5), 0.0, 1.0)
    return np.maximum(prob, np.asarray(enhanced, dtype=np.float32)).astype(np.float32)


def _preprocess_polygon_probability(
    probability: np.ndarray, config: PolygonConfidencePipelineConfig
) -> tuple[np.ndarray, np.ndarray]:
    prob = _normalize_probability_map(probability)
    smoothed = np.maximum(prob, _smooth_probability_map(prob, config))
    locally_normalized = _locally_normalized_probability_map(prob, config)
    preprocessed = np.maximum(smoothed, locally_normalized).astype(np.float32)
    return _normalize_probability_map(preprocessed), _normalize_probability_map(locally_normalized)


def _binary_erode_rect(mask: np.ndarray, radius_y: int = 1, radius_x: int = 1) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    ry = max(0, int(radius_y))
    rx = max(0, int(radius_x))
    if ry <= 0 and rx <= 0:
        return mask_bool.copy()
    if ndi is not None:
        structure = np.ones((2 * ry + 1, 2 * rx + 1), dtype=bool)
        return np.asarray(ndi.binary_erosion(mask_bool, structure=structure), dtype=bool)
    padded = np.pad(mask_bool, ((ry, ry), (rx, rx)), mode="constant", constant_values=False)
    result = np.ones_like(mask_bool, dtype=bool)
    for row_offset in range(2 * ry + 1):
        for column_offset in range(2 * rx + 1):
            result &= padded[
                row_offset : row_offset + mask_bool.shape[0], column_offset : column_offset + mask_bool.shape[1]
            ]
    return result


def _binary_close_rect(mask: np.ndarray, radius_y: int = 1, radius_x: int = 1) -> np.ndarray:
    dilated = _binary_dilate_rect(mask, radius_y, radius_x)
    return _binary_erode_rect(dilated, radius_y, radius_x)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size == 0 or xs.size == 0:
        return 0, 0, 0, 0
    y0 = int(np.min(ys))
    y1 = int(np.max(ys)) + 1
    x0 = int(np.min(xs))
    x1 = int(np.max(xs)) + 1
    return x0, y0, x1 - x0, y1 - y0


def _mask_geometry(mask: np.ndarray) -> tuple[int, tuple[int, int, int, int], float, float, float]:
    mask_bool = np.asarray(mask, dtype=bool)
    area = int(np.count_nonzero(mask_bool))
    bbox = _mask_bbox(mask_bool)
    width = max(1, int(bbox[2]))
    height = max(1, int(bbox[3]))
    aspect_ratio = float(max(width, height) / max(1, min(width, height)))
    extent = float(area / max(1, width * height))
    elongation = aspect_ratio
    ys, xs = np.nonzero(mask_bool)
    if ys.size >= 3:
        coords = np.column_stack((ys.astype(np.float32), xs.astype(np.float32)))
        centered = coords - np.mean(coords, axis=0, keepdims=True)
        covariance = np.matmul(centered.T, centered) / max(1.0, float(coords.shape[0] - 1))
        eigvals = np.linalg.eigvalsh(covariance)
        if eigvals.size >= 2:
            major = float(max(eigvals[-1], EPS))
            minor = float(max(eigvals[0], EPS))
            elongation = float(np.sqrt(major / minor))
    return area, bbox, aspect_ratio, elongation, extent


def _make_polygon_candidate(
    probability: np.ndarray,
    mask: np.ndarray,
    source_branches: tuple[str, ...],
    *,
    roi_bbox: tuple[int, int, int, int] | None = None,
) -> _PolygonConfidenceCandidate | None:
    """Build one candidate while optionally restricting geometry extraction to a known ROI."""

    mask_bool = np.asarray(mask, dtype=bool)
    if roi_bbox is not None:
        roi_x0, roi_y0, roi_width, roi_height = roi_bbox
        roi_x0 = max(0, int(roi_x0))
        roi_y0 = max(0, int(roi_y0))
        roi_x1 = min(mask_bool.shape[1], roi_x0 + max(0, int(roi_width)))
        roi_y1 = min(mask_bool.shape[0], roi_y0 + max(0, int(roi_height)))
        roi_mask = mask_bool[roi_y0:roi_y1, roi_x0:roi_x1]
    else:
        roi_x0 = 0
        roi_y0 = 0
        roi_mask = mask_bool

    area = int(np.count_nonzero(roi_mask))
    if area <= 0:
        return None

    ys, xs = np.nonzero(roi_mask)
    if ys.size == 0 or xs.size == 0:
        return None
    bbox_x0 = roi_x0 + int(np.min(xs))
    bbox_y0 = roi_y0 + int(np.min(ys))
    bbox_x1 = roi_x0 + int(np.max(xs)) + 1
    bbox_y1 = roi_y0 + int(np.max(ys)) + 1
    bbox = (bbox_x0, bbox_y0, bbox_x1 - bbox_x0, bbox_y1 - bbox_y0)
    width = max(1, int(bbox[2]))
    height = max(1, int(bbox[3]))
    aspect_ratio = float(max(width, height) / max(1, min(width, height)))
    extent = float(area / max(1, width * height))
    elongation = aspect_ratio
    if ys.size >= 3:
        coords = np.column_stack((ys.astype(np.float32), xs.astype(np.float32)))
        centered = coords - np.mean(coords, axis=0, keepdims=True)
        covariance = np.matmul(centered.T, centered) / max(1.0, float(coords.shape[0] - 1))
        eigvals = np.linalg.eigvalsh(covariance)
        if eigvals.size >= 2:
            major = float(max(eigvals[-1], EPS))
            minor = float(max(eigvals[0], EPS))
            elongation = float(np.sqrt(major / minor))

    object_prob = np.asarray(probability[bbox_y0:bbox_y1, bbox_x0:bbox_x1], dtype=np.float32)[
        mask_bool[bbox_y0:bbox_y1, bbox_x0:bbox_x1]
    ]
    if object_prob.size == 0:
        return None
    return _PolygonConfidenceCandidate(
        mask=mask_bool,
        source_branches=tuple(sorted({str(branch) for branch in source_branches if branch})),
        area=area,
        bbox=bbox,
        aspect_ratio=float(aspect_ratio),
        elongation=float(elongation),
        extent=float(extent),
        peak_probability=float(np.max(object_prob)),
        mean_probability=float(np.mean(object_prob, dtype=np.float64)),
    )


def _candidate_object_labels(
    candidates: Sequence[_PolygonConfidenceCandidate],
    shape: tuple[int, int],
) -> np.ndarray:
    """Build a label map that preserves final-candidate identity even when masks touch."""

    labels = np.zeros(shape, dtype=np.int32)
    for object_id, candidate in enumerate(candidates, start=1):
        candidate_mask = np.asarray(candidate.mask, dtype=bool)
        if candidate_mask.shape != shape or not np.any(candidate_mask):
            continue
        assignable = candidate_mask & (labels == 0)
        if np.any(assignable):
            labels[assignable] = int(object_id)
    return labels


def _append_debug_candidate(
    rows: list[PolygonConfidenceDebugCandidate],
    candidate_id: int,
    branch: str,
    candidate: _PolygonConfidenceCandidate,
    *,
    accepted: bool,
    notes: tuple[str, ...] = (),
) -> None:
    rows.append(
        PolygonConfidenceDebugCandidate(
            object_id=int(candidate_id),
            branch=str(branch),
            source_branches=tuple(candidate.source_branches),
            accepted=bool(accepted),
            area=int(candidate.area),
            bbox_x=int(candidate.bbox[0]),
            bbox_y=int(candidate.bbox[1]),
            bbox_width=int(candidate.bbox[2]),
            bbox_height=int(candidate.bbox[3]),
            aspect_ratio=float(candidate.aspect_ratio),
            elongation=float(candidate.elongation),
            peak_probability=float(candidate.peak_probability),
            mean_probability=float(candidate.mean_probability),
            extent=float(candidate.extent),
            notes=tuple(str(note) for note in notes),
        )
    )


def _extract_branch_candidates(
    probability: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
    *,
    branch: str,
    min_area: int = 1,
    max_area: int | None = None,
    require_high_core: bool = True,
    acceptance_fn=None,
    start_candidate_id: int = 1,
) -> tuple[list[_PolygonConfidenceCandidate], np.ndarray, tuple[PolygonConfidenceDebugCandidate, ...], int]:
    accepted: list[_PolygonConfidenceCandidate] = []
    debug_rows: list[PolygonConfidenceDebugCandidate] = []
    accepted_mask = np.zeros_like(np.asarray(low_mask, dtype=bool), dtype=bool)
    labels, count = _label_components(low_mask)
    candidate_id = int(start_candidate_id)
    for label_id in range(1, int(count) + 1):
        component_mask = labels == label_id
        candidate = _make_polygon_candidate(probability, component_mask, (branch,))
        if candidate is None:
            continue
        has_high_core = bool(np.any(np.asarray(high_mask, dtype=bool) & component_mask))
        notes: list[str] = []
        accepted_flag = candidate.area >= max(1, int(min_area))
        if max_area is not None:
            accepted_flag = accepted_flag and candidate.area <= max(1, int(max_area))
            if candidate.area > max(1, int(max_area)):
                notes.append("area_above_max")
        if candidate.area < max(1, int(min_area)):
            notes.append("area_below_min")
        if require_high_core and not has_high_core:
            accepted_flag = False
            notes.append("missing_high_core")
        if acceptance_fn is not None:
            branch_ok, branch_notes = acceptance_fn(candidate, has_high_core)
            accepted_flag = accepted_flag and bool(branch_ok)
            notes.extend(tuple(branch_notes))
        _append_debug_candidate(debug_rows, candidate_id, branch, candidate, accepted=accepted_flag, notes=tuple(notes))
        if accepted_flag:
            accepted.append(candidate)
            accepted_mask |= candidate.mask
        candidate_id += 1
    return accepted, accepted_mask, tuple(debug_rows), candidate_id


def _candidate_iou(first: _PolygonConfidenceCandidate, second: _PolygonConfidenceCandidate) -> float:
    x0_a, y0_a, w_a, h_a = first.bbox
    x0_b, y0_b, w_b, h_b = second.bbox
    x1_a = x0_a + w_a
    y1_a = y0_a + h_a
    x1_b = x0_b + w_b
    y1_b = y0_b + h_b
    ix0 = max(x0_a, x0_b)
    iy0 = max(y0_a, y0_b)
    ix1 = min(x1_a, x1_b)
    iy1 = min(y1_a, y1_b)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = int(np.count_nonzero(first.mask[iy0:iy1, ix0:ix1] & second.mask[iy0:iy1, ix0:ix1]))
    if intersection <= 0:
        return 0.0
    union = int(first.area + second.area - intersection)
    return float(intersection / max(1, union))


def _bbox_gap(first_bbox: tuple[int, int, int, int], second_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x0_a, y0_a, w_a, h_a = first_bbox
    x0_b, y0_b, w_b, h_b = second_bbox
    x1_a = x0_a + w_a
    y1_a = y0_a + h_a
    x1_b = x0_b + w_b
    y1_b = y0_b + h_b
    gap_x = max(0, max(x0_b - x1_a, x0_a - x1_b))
    gap_y = max(0, max(y0_b - y1_a, y0_a - y1_b))
    return int(gap_y), int(gap_x)


def _touches_within_distance(
    first: _PolygonConfidenceCandidate, second: _PolygonConfidenceCandidate, distance: int
) -> bool:
    gap = max(0, int(distance))
    if gap <= 0:
        iy, ix = _bbox_gap(first.bbox, second.bbox)
        if iy > 0 or ix > 0:
            return False
        x0 = max(int(first.bbox[0]), int(second.bbox[0]))
        y0 = max(int(first.bbox[1]), int(second.bbox[1]))
        x1 = min(int(first.bbox[0] + first.bbox[2]), int(second.bbox[0] + second.bbox[2]))
        y1 = min(int(first.bbox[1] + first.bbox[3]), int(second.bbox[1] + second.bbox[3]))
        if x1 <= x0 or y1 <= y0:
            return False
        return bool(np.any(first.mask[y0:y1, x0:x1] & second.mask[y0:y1, x0:x1]))
    gap_y, gap_x = _bbox_gap(first.bbox, second.bbox)
    if gap_y > gap or gap_x > gap:
        return False
    x0 = max(0, min(int(first.bbox[0]), int(second.bbox[0])) - gap)
    y0 = max(0, min(int(first.bbox[1]), int(second.bbox[1])) - gap)
    x1 = min(first.mask.shape[1], max(int(first.bbox[0] + first.bbox[2]), int(second.bbox[0] + second.bbox[2])) + gap)
    y1 = min(first.mask.shape[0], max(int(first.bbox[1] + first.bbox[3]), int(second.bbox[1] + second.bbox[3])) + gap)
    roi_first = np.asarray(first.mask[y0:y1, x0:x1], dtype=bool)
    roi_second = np.asarray(second.mask[y0:y1, x0:x1], dtype=bool)
    return bool(np.any(_binary_dilate(roi_first, gap) & roi_second))


def _axis_gap_close(
    first: _PolygonConfidenceCandidate, second: _PolygonConfidenceCandidate, config: PolygonConfidencePipelineConfig
) -> bool:
    x0_a, y0_a, w_a, h_a = first.bbox
    x0_b, y0_b, w_b, h_b = second.bbox
    x1_a = x0_a + w_a
    x1_b = x0_b + w_b
    y1_a = y0_a + h_a
    y1_b = y0_b + h_b
    overlap_x = min(x1_a, x1_b) - max(x0_a, x0_b)
    overlap_y = min(y1_a, y1_b) - max(y0_a, y0_b)
    vertical_gap = max(0, max(y0_b - y1_a, y0_a - y1_b))
    horizontal_gap = max(0, max(x0_b - x1_a, x0_a - x1_b))
    major_gap = max(1, int(config.merge_distance) * max(1, int(POLYGON_CONFIDENCE_COMPLETION_MAJOR_SCALE)))

    def _dominant_axis(candidate: _PolygonConfidenceCandidate) -> str:
        bbox_width = max(1, int(candidate.bbox[2]))
        bbox_height = max(1, int(candidate.bbox[3]))
        if bbox_width == bbox_height:
            return "compact"
        return "horizontal" if bbox_width >= bbox_height else "vertical"

    first_axis = _dominant_axis(first)
    second_axis = _dominant_axis(second)
    if first_axis == "compact" and second_axis != "compact":
        first_axis = second_axis
    elif second_axis == "compact" and first_axis != "compact":
        second_axis = first_axis
    if first_axis != second_axis:
        return False
    if first_axis == "vertical":
        return overlap_x > 0 and vertical_gap <= major_gap
    if first_axis == "horizontal":
        return overlap_y > 0 and horizontal_gap <= major_gap
    return False


def _filter_bridge_region(
    probability: np.ndarray,
    bridge_mask: np.ndarray,
    first_support: np.ndarray,
    second_support: np.ndarray,
    *,
    floor_threshold: float,
    trim_delta: float,
) -> np.ndarray:
    """Keep only bridge pixels that have enough support to justify connecting two regions."""

    bridge_bool = np.asarray(bridge_mask, dtype=bool)
    if not np.any(bridge_bool):
        return bridge_bool
    first_bool = np.asarray(first_support, dtype=bool)
    second_bool = np.asarray(second_support, dtype=bool)
    prob = np.asarray(probability, dtype=np.float32)
    first_values = np.asarray(prob[first_bool], dtype=np.float32)
    second_values = np.asarray(prob[second_bool], dtype=np.float32)
    if first_values.size == 0 or second_values.size == 0:
        return np.zeros_like(bridge_bool, dtype=bool)
    reference_mean = min(
        float(np.mean(first_values, dtype=np.float64)),
        float(np.mean(second_values, dtype=np.float64)),
    )
    bridge_threshold = max(
        float(floor_threshold),
        min(reference_mean - float(trim_delta), reference_mean * 0.92),
    )
    keep = bridge_bool & (prob >= bridge_threshold)
    if not np.any(keep):
        return np.zeros_like(bridge_bool, dtype=bool)
    kept_fraction = float(np.count_nonzero(keep) / max(1, np.count_nonzero(bridge_bool)))
    if kept_fraction < 0.35:
        return np.zeros_like(bridge_bool, dtype=bool)
    keep_touch = np.asarray(_binary_dilate(keep, 1), dtype=bool)
    if not (np.any(keep_touch & first_bool) and np.any(keep_touch & second_bool)):
        return np.zeros_like(bridge_bool, dtype=bool)
    return np.asarray(keep, dtype=bool)


def _merge_candidate_pair(
    probability: np.ndarray,
    first: _PolygonConfidenceCandidate,
    second: _PolygonConfidenceCandidate,
    *,
    bridge_radius: int = 0,
    floor_threshold: float = POLYGON_CONFIDENCE_HYSTERESIS_FLOOR,
    trim_delta: float = POLYGON_CONFIDENCE_SPILL_TRIM_DELTA,
) -> _PolygonConfidenceCandidate | None:
    first_mask = np.asarray(first.mask, dtype=bool)
    second_mask = np.asarray(second.mask, dtype=bool)
    merged_mask = first_mask | second_mask
    gap_radius = max(0, int(bridge_radius))
    if gap_radius > 0 and not np.any(first_mask & second_mask):
        x0 = max(0, min(int(first.bbox[0]), int(second.bbox[0])) - gap_radius)
        y0 = max(0, min(int(first.bbox[1]), int(second.bbox[1])) - gap_radius)
        x1 = min(
            first_mask.shape[1],
            max(int(first.bbox[0] + first.bbox[2]), int(second.bbox[0] + second.bbox[2])) + gap_radius,
        )
        y1 = min(
            first_mask.shape[0],
            max(int(first.bbox[1] + first.bbox[3]), int(second.bbox[1] + second.bbox[3])) + gap_radius,
        )
        first_roi = first_mask[y0:y1, x0:x1]
        second_roi = second_mask[y0:y1, x0:x1]
        bridge_roi = _binary_dilate(first_roi, gap_radius) & _binary_dilate(second_roi, gap_radius)
        if np.any(bridge_roi):
            bridge_roi = _filter_bridge_region(
                np.asarray(probability[y0:y1, x0:x1], dtype=np.float32),
                bridge_roi,
                first_roi,
                second_roi,
                floor_threshold=float(floor_threshold),
                trim_delta=float(trim_delta),
            )
        if np.any(bridge_roi):
            merged_mask = merged_mask.copy()
            merged_mask[y0:y1, x0:x1] |= bridge_roi
    merged_bbox = (
        max(0, min(int(first.bbox[0]), int(second.bbox[0])) - gap_radius),
        max(0, min(int(first.bbox[1]), int(second.bbox[1])) - gap_radius),
        min(
            first_mask.shape[1],
            max(int(first.bbox[0] + first.bbox[2]), int(second.bbox[0] + second.bbox[2])) + gap_radius,
        )
        - max(0, min(int(first.bbox[0]), int(second.bbox[0])) - gap_radius),
        min(
            first_mask.shape[0],
            max(int(first.bbox[1] + first.bbox[3]), int(second.bbox[1] + second.bbox[3])) + gap_radius,
        )
        - max(0, min(int(first.bbox[1]), int(second.bbox[1])) - gap_radius),
    )
    return _make_polygon_candidate(
        probability,
        merged_mask,
        tuple(sorted(set(first.source_branches) | set(second.source_branches))),
        roi_bbox=merged_bbox,
    )


def _merge_search_margin(config: PolygonConfidencePipelineConfig) -> int:
    """Return the maximum bbox margin needed for merge neighborhood search."""

    base_gap = max(0, int(config.merge_distance))
    axis_gap = max(1, base_gap * max(1, int(POLYGON_CONFIDENCE_COMPLETION_MAJOR_SCALE)))
    return max(base_gap, axis_gap)


@lru_cache(maxsize=64)
def _merge_tile_size(search_margin: int) -> int:
    """Return a stable tile size for spatial candidate indexing."""

    return max(16, 2 * max(1, int(search_margin)) + 8)


def _candidate_neighbor_pairs(
    candidates: list[_PolygonConfidenceCandidate],
    config: PolygonConfidencePipelineConfig,
) -> list[tuple[int, int]]:
    """Return unique candidate pairs that are spatially close enough to merit exact merge checks."""

    if len(candidates) <= 1:
        return []
    search_margin = _merge_search_margin(config)
    tile_size = _merge_tile_size(search_margin)
    bins: dict[tuple[int, int], list[int]] = {}
    pairs: set[tuple[int, int]] = set()
    for index, candidate in enumerate(candidates):
        x0, y0, width, height = candidate.bbox
        x1 = x0 + width
        y1 = y0 + height
        gx0 = (max(0, int(x0) - search_margin)) // tile_size
        gy0 = (max(0, int(y0) - search_margin)) // tile_size
        gx1 = (max(0, int(x1) + search_margin - 1)) // tile_size
        gy1 = (max(0, int(y1) + search_margin - 1)) // tile_size
        for grid_y in range(gy0, gy1 + 1):
            for grid_x in range(gx0, gx1 + 1):
                cell = (grid_x, grid_y)
                for other_index in bins.get(cell, ()):
                    if other_index == index:
                        continue
                    pair = (other_index, index) if other_index < index else (index, other_index)
                    pairs.add(pair)
                bins.setdefault(cell, []).append(index)
    return sorted(pairs)


def _merge_connected_candidate_groups(
    probability: np.ndarray,
    candidates: list[_PolygonConfidenceCandidate],
    pair_indices: list[tuple[int, int]],
    config: PolygonConfidencePipelineConfig,
) -> list[_PolygonConfidenceCandidate]:
    """Merge all candidate groups connected by exact merge predicates for one round."""

    if not pair_indices:
        return list(candidates)

    parent = list(range(len(candidates)))

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _union(left: int, right: int) -> None:
        root_left = _find(left)
        root_right = _find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    merge_threshold = float(config.merge_iou_threshold)
    merge_distance = int(config.merge_distance)
    for index_a, index_b in pair_indices:
        first = candidates[index_a]
        second = candidates[index_b]
        overlap = _candidate_iou(first, second)
        if overlap >= merge_threshold:
            _union(index_a, index_b)
            continue
        if _touches_within_distance(first, second, merge_distance) or _axis_gap_close(first, second, config):
            _union(index_a, index_b)

    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(_find(index), []).append(index)

    if all(len(indices) == 1 for indices in groups.values()):
        return list(candidates)

    next_candidates: list[_PolygonConfidenceCandidate] = []
    for indices in groups.values():
        ordered = sorted(indices)
        if len(ordered) == 1:
            next_candidates.append(candidates[ordered[0]])
            continue
        merged_candidate = candidates[ordered[0]]
        for current_index in ordered[1:]:
            merged_candidate = _merge_candidate_pair(
                probability,
                merged_candidate,
                candidates[current_index],
                bridge_radius=max(1, merge_distance),
                floor_threshold=float(config.hysteresis_low_floor),
                trim_delta=float(config.spill_trim_delta),
            )
            if merged_candidate is None:
                merged_candidate = candidates[ordered[0]]
                break
        next_candidates.append(merged_candidate)
    return next_candidates


def _merge_polygon_candidates(
    probability: np.ndarray,
    candidates: list[_PolygonConfidenceCandidate],
    config: PolygonConfidencePipelineConfig,
) -> list[_PolygonConfidenceCandidate]:
    """Merge polygon candidates using spatial shortlists instead of repeated all-pairs scans."""

    merged = list(candidates)
    while True:
        pair_indices = _candidate_neighbor_pairs(merged, config)
        if not pair_indices:
            return merged
        next_candidates = _merge_connected_candidate_groups(probability, merged, pair_indices, config)
        if len(next_candidates) == len(merged):
            return merged
        merged = next_candidates


def _complete_polygon_candidates(
    probability: np.ndarray,
    candidates: list[_PolygonConfidenceCandidate],
    *,
    strong_threshold: float,
    weak_threshold: float,
    high_seed_mask: np.ndarray | None = None,
) -> list[_PolygonConfidenceCandidate]:
    if not candidates:
        return []
    completion_threshold = _polygon_confidence_completion_threshold(float(strong_threshold), float(weak_threshold))
    completion_mask = np.asarray(probability >= completion_threshold, dtype=bool)
    if not np.any(completion_mask):
        return list(candidates)
    completion_labels, completion_count = _label_components(completion_mask)
    if completion_count <= 0:
        return list(candidates)
    high_labels = None
    if high_seed_mask is not None:
        high_labels, _high_count = _label_components(np.asarray(high_seed_mask, dtype=bool) & completion_mask)
    bridge_radius = max(0, int(POLYGON_CONFIDENCE_COMPLETION_BRIDGE_RADIUS))
    completed: list[_PolygonConfidenceCandidate] = []
    for candidate in candidates:
        seed_mask = np.asarray(candidate.mask, dtype=bool)
        reach_radius_y, reach_radius_x = _completion_radii_for_mask(seed_mask, bridge_radius)
        reach_mask = (
            _binary_dilate_rect(seed_mask, reach_radius_y, reach_radius_x)
            if (reach_radius_y > 0 or reach_radius_x > 0)
            else seed_mask
        )
        grown_mask = np.asarray(seed_mask, dtype=bool)
        completion_ids = [
            int(candidate_id) for candidate_id in np.unique(completion_labels[reach_mask]) if int(candidate_id) > 0
        ]
        for completion_id in completion_ids:
            component_mask = completion_labels == completion_id
            component_add = component_mask
            if high_labels is not None:
                component_high_ids = np.unique(high_labels[component_mask])
                component_high_ids = component_high_ids[component_high_ids > 0]
                if component_high_ids.size > 1:
                    current_high_ids = np.unique(high_labels[component_mask & seed_mask])
                    current_high_ids = current_high_ids[current_high_ids > 0]
                    if current_high_ids.size > 0:
                        valid_seed_mask = np.isin(high_labels, component_high_ids, assume_unique=True) & component_mask
                        if np.any(valid_seed_mask):
                            _distances, nearest_indices = ndi.distance_transform_edt(
                                ~valid_seed_mask, return_indices=True
                            )
                            nearest_seed_labels = high_labels[tuple(nearest_indices)]
                            component_add = component_mask & np.isin(
                                nearest_seed_labels, current_high_ids, assume_unique=True
                            )
                            component_add &= (
                                np.asarray(probability, dtype=np.float32)
                                >= max(
                                    float(POLYGON_CONFIDENCE_HYSTERESIS_FLOOR),
                                    float(weak_threshold) + 0.02,
                                )
                            ) | seed_mask
            grown_mask |= component_add
            if (reach_radius_y > 0 or reach_radius_x > 0) and not np.any(component_mask & seed_mask):
                bridge_path = _binary_dilate_rect(seed_mask, reach_radius_y, reach_radius_x) & _binary_dilate_rect(
                    component_mask, reach_radius_y, reach_radius_x
                )
                bridge_path = _filter_bridge_region(
                    probability,
                    bridge_path,
                    seed_mask,
                    component_mask,
                    floor_threshold=max(float(weak_threshold), float(POLYGON_CONFIDENCE_HYSTERESIS_FLOOR)),
                    trim_delta=float(POLYGON_CONFIDENCE_SPILL_TRIM_DELTA),
                )
                if np.any(bridge_path):
                    grown_mask |= bridge_path
        completed_candidate = _make_polygon_candidate(probability, grown_mask, candidate.source_branches)
        completed.append(completed_candidate if completed_candidate is not None else candidate)
    return completed


def _split_polygon_candidate_by_seeds(
    probability: np.ndarray,
    candidate: _PolygonConfidenceCandidate,
    high_mask: np.ndarray,
    config: PolygonConfidencePipelineConfig,
) -> tuple[_PolygonConfidenceCandidate, ...]:
    """Backward-compatible wrapper around the barrier-aware seed separation stage."""

    split_candidates, _debug_payload = _split_polygon_candidate_by_barriers(
        probability,
        candidate,
        high_mask,
        low_threshold=_polygon_confidence_weak_threshold(
            float(np.max(np.asarray(probability[candidate.mask], dtype=np.float32))) if np.any(candidate.mask) else 0.5,
            config,
        ),
        strong_threshold=float(np.max(np.asarray(probability[candidate.mask], dtype=np.float32)))
        if np.any(candidate.mask)
        else 0.5,
        config=config,
        include_debug=False,
    )
    return split_candidates


def _candidate_boundary_separation(probability: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return 0.0
    boundary = _boundary_mask(mask_bool)
    outer_ring = _binary_dilate(mask_bool, 1) & ~mask_bool
    if not np.any(boundary) or not np.any(outer_ring):
        return 0.0
    inner_mean = float(np.mean(np.asarray(probability[boundary], dtype=np.float32), dtype=np.float64))
    outer_mean = float(np.mean(np.asarray(probability[outer_ring], dtype=np.float32), dtype=np.float64))
    return float(_clip01(inner_mean - outer_mean))


def _candidate_axis_coverage(mask: np.ndarray) -> tuple[float, float]:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return 0.0, 0.0
    height, width = mask_bool.shape
    row_coverage = np.count_nonzero(mask_bool, axis=1).astype(np.float32) / max(1, width)
    col_coverage = np.count_nonzero(mask_bool, axis=0).astype(np.float32) / max(1, height)
    active_row = row_coverage[row_coverage > 0.0]
    active_col = col_coverage[col_coverage > 0.0]
    row_p90 = float(np.quantile(active_row, 0.9)) if active_row.size else 0.0
    col_p90 = float(np.quantile(active_col, 0.9)) if active_col.size else 0.0
    return row_p90, col_p90


def _candidate_border_span_features(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    *,
    cross_axis_max: float,
) -> dict[str, float | bool]:
    """Return border-touch and span metrics shared by spill rejection and final trimming."""

    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    x0, y0, width, height = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    x1 = x0 + width
    y1 = y0 + height
    touches_lr = x0 <= 0 and x1 >= image_width
    touches_tb = y0 <= 0 and y1 >= image_height
    span_x = float(width / max(1, image_width))
    span_y = float(height / max(1, image_height))
    thin_cross_axis = (touches_lr and span_y <= float(cross_axis_max)) or (
        touches_tb and span_x <= float(cross_axis_max)
    )
    row_coverage_p90, col_coverage_p90 = _candidate_axis_coverage(mask)
    axis_coverage = 0.0
    if touches_lr:
        axis_coverage = max(axis_coverage, row_coverage_p90)
    if touches_tb:
        axis_coverage = max(axis_coverage, col_coverage_p90)
    return {
        "touches_lr": bool(touches_lr),
        "touches_tb": bool(touches_tb),
        "span_x": float(span_x),
        "span_y": float(span_y),
        "thin_cross_axis": bool(thin_cross_axis),
        "row_coverage_p90": float(row_coverage_p90),
        "col_coverage_p90": float(col_coverage_p90),
        "axis_coverage": float(axis_coverage),
    }


def _candidate_axis_support_strength(
    candidate_mask: np.ndarray,
    support_mask: np.ndarray,
    *,
    touches_lr: bool,
    touches_tb: bool,
) -> tuple[float, float]:
    """Return major-axis support coverage and support area fraction for a candidate."""

    candidate_bool = np.asarray(candidate_mask, dtype=bool)
    support_bool = np.asarray(support_mask, dtype=bool) & candidate_bool
    candidate_area = int(np.count_nonzero(candidate_bool))
    if candidate_area <= 0 or not np.any(support_bool):
        return 0.0, 0.0
    row_coverage_p90, col_coverage_p90 = _candidate_axis_coverage(support_bool)
    axis_coverage = 0.0
    if touches_lr:
        axis_coverage = max(axis_coverage, row_coverage_p90)
    if touches_tb:
        axis_coverage = max(axis_coverage, col_coverage_p90)
    support_area_fraction = float(np.count_nonzero(support_bool) / max(1, candidate_area))
    return float(axis_coverage), float(support_area_fraction)


def _candidate_has_compact_core(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    core_threshold: float,
    max_aspect_ratio: float,
) -> bool:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return False
    core_mask = mask_bool & (np.asarray(probability, dtype=np.float32) >= float(core_threshold))
    if not np.any(core_mask):
        return False
    labels, count = _label_components(core_mask)
    for label_id in range(1, int(count) + 1):
        component_mask = labels == label_id
        area = int(np.count_nonzero(component_mask))
        if area < 4:
            continue
        ys, xs = np.nonzero(component_mask)
        if ys.size == 0 or xs.size == 0:
            continue
        width = int(np.max(xs) - np.min(xs) + 1)
        height = int(np.max(ys) - np.min(ys) + 1)
        aspect_ratio = float(max(width, height) / max(1, min(width, height)))
        if aspect_ratio <= float(max_aspect_ratio):
            return True
    return False


def _compact_core_mask(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    core_threshold: float,
    max_aspect_ratio: float,
    min_area: int = 4,
) -> np.ndarray:
    """Return compact high-probability cores inside a candidate."""

    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return np.zeros_like(mask_bool, dtype=bool)
    core_mask = mask_bool & (np.asarray(probability, dtype=np.float32) >= float(core_threshold))
    if not np.any(core_mask):
        return np.zeros_like(mask_bool, dtype=bool)
    labels, count = _label_components(core_mask)
    compact = np.zeros_like(mask_bool, dtype=bool)
    for label_id in range(1, int(count) + 1):
        component_mask = labels == label_id
        area = int(np.count_nonzero(component_mask))
        if area < max(1, int(min_area)):
            continue
        ys, xs = np.nonzero(component_mask)
        if ys.size == 0 or xs.size == 0:
            continue
        width = int(np.max(xs) - np.min(xs) + 1)
        height = int(np.max(ys) - np.min(ys) + 1)
        aspect_ratio = float(max(width, height) / max(1, min(width, height)))
        if aspect_ratio <= float(max_aspect_ratio):
            compact |= component_mask
    return compact


def _local_prominence_map(probability: np.ndarray, radius: int) -> np.ndarray:
    """Estimate local prominence over a broader neighborhood."""

    prob = np.asarray(probability, dtype=np.float32)
    local_mean = _local_mean_map(prob, radius=max(1, int(radius)))
    prominence = np.clip(prob - local_mean, 0.0, 1.0)
    return np.asarray(prominence, dtype=np.float32)


def _retain_core_supported_region(
    probability: np.ndarray,
    prominence_map: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    core_threshold: float,
    grow_threshold: float,
    prominence_threshold: float,
    max_core_aspect_ratio: float,
) -> np.ndarray:
    """Trim a spill-like candidate to regions supported by compact high-prominence cores."""

    mask_bool = np.asarray(candidate_mask, dtype=bool)
    compact_core = _compact_core_mask(
        probability,
        mask_bool,
        core_threshold=core_threshold,
        max_aspect_ratio=max_core_aspect_ratio,
    )
    if not np.any(compact_core):
        return np.zeros_like(mask_bool, dtype=bool)
    support_mask = mask_bool & (
        (np.asarray(probability, dtype=np.float32) >= float(grow_threshold))
        | (np.asarray(prominence_map, dtype=np.float32) >= float(prominence_threshold))
    )
    if not np.any(support_mask):
        return compact_core
    core_labels, core_count = _label_components(compact_core)
    retained = np.zeros_like(mask_bool, dtype=bool)
    for core_id in range(1, int(core_count) + 1):
        core_component = core_labels == core_id
        if not np.any(core_component):
            continue
        core_bbox = _mask_bbox(core_component)
        core_width = max(1, int(core_bbox[2]))
        core_height = max(1, int(core_bbox[3]))
        if core_width >= core_height:
            core_window = _binary_dilate_rect(
                core_component,
                radius_y=max(1, min(4, core_height)),
                radius_x=max(2, min(4, core_width // 3 + 1)),
            )
        else:
            core_window = _binary_dilate_rect(
                core_component,
                radius_y=max(2, min(4, core_height // 3 + 1)),
                radius_x=max(1, min(4, core_width)),
            )
        local_support = support_mask & core_window
        if not np.any(local_support):
            retained |= core_component
            continue
        labels, count = _label_components(local_support)
        retained_core = False
        for label_id in range(1, int(count) + 1):
            component_mask = labels == label_id
            if np.any(component_mask & core_component):
                retained |= component_mask
                retained_core = True
        if not retained_core:
            retained |= core_component
    return retained if np.any(retained) else compact_core


def _connected_support_from_seed(support_mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    """Keep only support components connected to the provided seed mask."""

    support_bool = np.asarray(support_mask, dtype=bool)
    seed_bool = np.asarray(seed_mask, dtype=bool) & support_bool
    if not np.any(support_bool) or not np.any(seed_bool):
        return np.zeros_like(support_bool, dtype=bool)
    labels, count = _label_components(support_bool)
    if count <= 0:
        return np.zeros_like(support_bool, dtype=bool)
    seed_labels = np.unique(labels[seed_bool])
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size <= 0:
        return np.zeros_like(support_bool, dtype=bool)
    return np.isin(labels, seed_labels, assume_unique=True)


def _separation_core_mask(
    candidate_mask: np.ndarray,
    high_mask: np.ndarray,
    *,
    min_area: int,
) -> np.ndarray:
    """Return stable seed cores for one candidate, removing tiny fragments."""

    candidate_bool = np.asarray(candidate_mask, dtype=bool)
    core_mask = candidate_bool & np.asarray(high_mask, dtype=bool)
    if not np.any(core_mask):
        return np.zeros_like(candidate_bool, dtype=bool)
    labels, count = _label_components(core_mask)
    if count <= 0:
        return np.zeros_like(candidate_bool, dtype=bool)
    area_counts = np.bincount(labels.ravel(), minlength=count + 1)
    valid_ids = np.flatnonzero(area_counts >= max(1, int(min_area)))
    valid_ids = valid_ids[valid_ids > 0]
    if valid_ids.size <= 0:
        return np.zeros_like(candidate_bool, dtype=bool)
    return np.isin(labels, valid_ids, assume_unique=True)


def _max_filter_map(array: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    local_radius = max(1, int(radius))
    if ndi is not None:
        size = 2 * local_radius + 1
        return np.asarray(ndi.maximum_filter(values, size=size, mode="nearest"), dtype=np.float32)
    padded = np.pad(values, local_radius, mode="edge")
    result = np.empty_like(values, dtype=np.float32)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            patch = padded[y : y + 2 * local_radius + 1, x : x + 2 * local_radius + 1]
            result[y, x] = float(np.max(patch))
    return result


def _boundary_cue_map(
    probability: np.ndarray,
    candidate_region: np.ndarray,
    *,
    low_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> np.ndarray:
    """Build boundary cues that combine valleys, local contrast, and transition uncertainty."""

    candidate_bool = np.asarray(candidate_region, dtype=bool)
    if not np.any(candidate_bool):
        return np.zeros_like(candidate_bool, dtype=np.float32)
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    local_contrast = _local_contrast_map(prob, radius=1)
    _width_map, _inverse_local_contrast, transition_uncertainty = _polygon_transition_uncertainty_maps(
        prob,
        candidate_bool,
        contrast_radius=1,
    )
    candidate_values = np.asarray(prob[candidate_bool], dtype=np.float32)
    candidate_low = float(np.quantile(candidate_values, 0.20)) if candidate_values.size else float(low_threshold)
    valley_reference = max(float(low_threshold), candidate_low)
    valley_scale = max(0.04, 1.0 - valley_reference)
    low_score = np.clip((valley_reference - prob) / valley_scale, 0.0, 1.0).astype(np.float32)
    cue = np.clip(
        float(config.separation_boundary_low_weight) * low_score
        + float(config.separation_boundary_contrast_weight) * local_contrast
        + float(config.separation_boundary_uncertainty_weight) * transition_uncertainty,
        0.0,
        1.0,
    ).astype(np.float32)
    cue[~candidate_bool] = 0.0
    return cue


def _thin_barrier_map_from_cues(
    boundary_cues: np.ndarray,
    candidate_region: np.ndarray,
    *,
    threshold: float,
    dilate_radius: int,
) -> np.ndarray:
    """Convert wide boundary cues into a thin separating barrier."""

    candidate_bool = np.asarray(candidate_region, dtype=bool)
    cue = np.asarray(boundary_cues, dtype=np.float32)
    if not np.any(candidate_bool):
        return np.zeros_like(candidate_bool, dtype=bool)
    ridge_zone = candidate_bool & (cue >= float(threshold))
    if not np.any(ridge_zone):
        return np.zeros_like(candidate_bool, dtype=bool)
    local_max = _max_filter_map(cue, radius=1)
    ridge_center = ridge_zone & (cue >= (local_max - 1e-6))
    skeleton = skeletonize(ridge_zone)
    thin = (ridge_center | skeleton) & candidate_bool
    radius = max(0, int(dilate_radius))
    if radius > 0:
        thin = _binary_dilate(thin, radius) & candidate_bool
    return np.asarray(thin, dtype=bool)


def _bridge_cut_mask(
    probability: np.ndarray,
    candidate_region: np.ndarray,
    boundary_cues: np.ndarray,
    seed_mask: np.ndarray,
    *,
    config: PolygonConfidencePipelineConfig,
) -> np.ndarray:
    """Break weak narrow bridges that connect multiple nearby seed regions."""

    candidate_bool = np.asarray(candidate_region, dtype=bool)
    seed_bool = np.asarray(seed_mask, dtype=bool) & candidate_bool
    if not np.any(candidate_bool) or not np.any(seed_bool):
        return np.zeros_like(candidate_bool, dtype=bool)
    seed_labels, seed_count = _label_components(seed_bool)
    if seed_count <= 1:
        return np.zeros_like(candidate_bool, dtype=bool)
    thin_bridge = np.asarray(_thin_bridge_map(candidate_bool) > 0.0, dtype=bool)
    if not np.any(thin_bridge):
        return np.zeros_like(candidate_bool, dtype=bool)
    bridge_mask = (
        thin_bridge
        & (np.asarray(probability, dtype=np.float32) <= float(config.separation_bridge_probability_max))
        & (np.asarray(boundary_cues, dtype=np.float32) >= float(config.separation_bridge_barrier_threshold))
    )
    if not np.any(bridge_mask):
        return np.zeros_like(candidate_bool, dtype=bool)
    labels, count = _label_components(bridge_mask)
    if count <= 0:
        return np.zeros_like(candidate_bool, dtype=bool)
    result = np.zeros_like(candidate_bool, dtype=bool)
    for label_id in range(1, int(count) + 1):
        component = labels == label_id
        if not np.any(component):
            continue
        expanded = _binary_dilate(component, 1) & candidate_bool
        touched_seeds = np.unique(seed_labels[expanded])
        touched_seeds = touched_seeds[touched_seeds > 0]
        if touched_seeds.size >= 2:
            result |= component
    return result


def _assign_candidate_region_to_seeds(
    candidate_region: np.ndarray,
    seed_labels: np.ndarray,
    blocked_mask: np.ndarray,
) -> np.ndarray:
    """Assign candidate pixels to seed instances while respecting blocked barriers."""

    candidate_bool = np.asarray(candidate_region, dtype=bool)
    labels = np.asarray(seed_labels, dtype=np.int32)
    blocked = np.asarray(blocked_mask, dtype=bool) & candidate_bool
    if not np.any(candidate_bool) or not np.any(labels > 0):
        return np.zeros_like(labels, dtype=np.int32)
    available = candidate_bool & ~blocked
    available |= labels > 0
    if not np.any(available):
        return np.zeros_like(labels, dtype=np.int32)
    component_labels, component_count = _label_components(available)
    if component_count <= 0:
        return np.zeros_like(labels, dtype=np.int32)
    assigned = np.zeros_like(labels, dtype=np.int32)
    for component_id in range(1, int(component_count) + 1):
        component = component_labels == component_id
        if not np.any(component):
            continue
        touched = np.unique(labels[component])
        touched = touched[touched > 0]
        if touched.size == 1:
            assigned[component] = int(touched[0])
            continue
        if touched.size <= 1:
            continue
        valid_seed_mask = np.isin(labels, touched, assume_unique=True)
        if not np.any(valid_seed_mask):
            continue
        _distances, nearest_indices = ndi.distance_transform_edt(~valid_seed_mask, return_indices=True)
        nearest_seed_labels = labels[tuple(nearest_indices)]
        assigned_component = nearest_seed_labels[component]
        positive = assigned_component > 0
        if np.any(positive):
            assigned_values = assigned_component[positive]
            component_indices = np.argwhere(component)
            assigned[component_indices[positive, 0], component_indices[positive, 1]] = assigned_values
    return assigned


def _split_polygon_candidate_by_barriers(
    probability: np.ndarray,
    candidate: _PolygonConfidenceCandidate,
    high_mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
    include_debug: bool = False,
) -> tuple[tuple[_PolygonConfidenceCandidate, ...], dict[str, np.ndarray] | None]:
    """Split one candidate via thin barriers and bridge cuts, then grow from stable seed cores."""

    if not bool(config.enable_watershed) or ndi is None:
        return (candidate,), None

    candidate_mask = np.asarray(candidate.mask, dtype=bool)
    if not np.any(candidate_mask):
        return (candidate,), None
    pad = max(1, int(config.separation_roi_padding))
    x0, y0, width, height = candidate.bbox
    roi_x0 = max(0, x0 - pad)
    roi_y0 = max(0, y0 - pad)
    roi_x1 = min(candidate_mask.shape[1], x0 + width + pad)
    roi_y1 = min(candidate_mask.shape[0], y0 + height + pad)
    candidate_roi = np.asarray(candidate_mask[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    if not np.any(candidate_roi):
        return (candidate,), None
    probability_roi = np.asarray(probability[roi_y0:roi_y1, roi_x0:roi_x1], dtype=np.float32)
    high_roi = np.asarray(high_mask[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    core_roi = _separation_core_mask(
        candidate_roi,
        high_roi,
        min_area=max(1, int(config.separation_core_min_area)),
    )
    seed_labels, seed_count = _label_components(core_roi)
    debug_payload = None
    if include_debug:
        debug_payload = {
            "candidate_region": np.zeros_like(candidate_mask, dtype=bool),
            "core_seeds": np.zeros_like(candidate_mask, dtype=bool),
            "thin_barrier": np.zeros_like(candidate_mask, dtype=bool),
            "bridge_cuts": np.zeros_like(candidate_mask, dtype=bool),
            "barrier_blocked": np.zeros_like(candidate_mask, dtype=bool),
            "boundary_cues": np.zeros_like(probability, dtype=np.float32),
        }
        debug_payload["candidate_region"][roi_y0:roi_y1, roi_x0:roi_x1] = candidate_roi
        debug_payload["core_seeds"][roi_y0:roi_y1, roi_x0:roi_x1] = core_roi
    if seed_count <= 1:
        return (candidate,), debug_payload

    boundary_cues_roi = _boundary_cue_map(
        probability_roi,
        candidate_roi,
        low_threshold=low_threshold,
        config=config,
    )
    thin_barrier_roi = _thin_barrier_map_from_cues(
        boundary_cues_roi,
        candidate_roi,
        threshold=float(config.separation_barrier_threshold),
        dilate_radius=int(config.separation_barrier_dilate_radius),
    )
    bridge_cut_roi = _bridge_cut_mask(
        probability_roi,
        candidate_roi,
        boundary_cues_roi,
        core_roi,
        config=config,
    )
    if not np.any(thin_barrier_roi) and not np.any(bridge_cut_roi):
        return (candidate,), debug_payload
    blocked_roi = (thin_barrier_roi | bridge_cut_roi) & ~core_roi
    assigned_labels_roi = _assign_candidate_region_to_seeds(candidate_roi, seed_labels, blocked_roi)
    split_candidates: list[_PolygonConfidenceCandidate] = []
    for seed_id in range(1, int(seed_count) + 1):
        component_roi = assigned_labels_roi == seed_id
        if not np.any(component_roi):
            continue
        component_mask = np.zeros_like(candidate_mask, dtype=bool)
        component_mask[roi_y0:roi_y1, roi_x0:roi_x1] = component_roi
        component_candidate = _make_polygon_candidate(
            probability,
            component_mask,
            candidate.source_branches,
            roi_bbox=(roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0),
        )
        if component_candidate is not None:
            split_candidates.append(component_candidate)
    if include_debug and debug_payload is not None:
        debug_payload["thin_barrier"][roi_y0:roi_y1, roi_x0:roi_x1] = thin_barrier_roi
        debug_payload["bridge_cuts"][roi_y0:roi_y1, roi_x0:roi_x1] = bridge_cut_roi
        debug_payload["barrier_blocked"][roi_y0:roi_y1, roi_x0:roi_x1] = blocked_roi
        debug_payload["boundary_cues"][roi_y0:roi_y1, roi_x0:roi_x1] = boundary_cues_roi
    return tuple(split_candidates) if split_candidates else (candidate,), debug_payload


def _tighten_candidate_with_barrier_support(
    probability: np.ndarray,
    mask: np.ndarray,
    high_mask: np.ndarray,
    *,
    low_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> np.ndarray:
    """Tighten inflated small/medium candidate boundaries using thin barrier evidence."""

    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return mask_bool
    candidate = _make_polygon_candidate(probability, mask_bool, ("refine",))
    if candidate is None or candidate.area < 6:
        return mask_bool
    if float(max(candidate.aspect_ratio, candidate.elongation)) >= float(config.boundary_snap_min_aspect):
        return mask_bool
    pad = max(1, int(config.separation_roi_padding))
    x0, y0, width, height = candidate.bbox
    roi_x0 = max(0, x0 - pad)
    roi_y0 = max(0, y0 - pad)
    roi_x1 = min(mask_bool.shape[1], x0 + width + pad)
    roi_y1 = min(mask_bool.shape[0], y0 + height + pad)
    candidate_roi = np.asarray(mask_bool[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    high_roi = np.asarray(high_mask[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    core_roi = _separation_core_mask(
        candidate_roi,
        high_roi,
        min_area=max(1, int(config.separation_core_min_area)),
    )
    if not np.any(core_roi):
        return mask_bool
    boundary_cues_roi = _boundary_cue_map(
        np.asarray(probability[roi_y0:roi_y1, roi_x0:roi_x1], dtype=np.float32),
        candidate_roi,
        low_threshold=low_threshold,
        config=config,
    )
    thin_barrier_roi = _thin_barrier_map_from_cues(
        boundary_cues_roi,
        candidate_roi,
        threshold=max(float(config.separation_barrier_threshold), 0.56),
        dilate_radius=0,
    )
    if not np.any(thin_barrier_roi):
        return mask_bool
    blocked_roi = thin_barrier_roi & ~_binary_dilate(core_roi, 1)
    if not np.any(blocked_roi):
        return mask_bool
    tightened_roi = _connected_support_from_seed(candidate_roi & ~blocked_roi, core_roi)
    if not np.any(tightened_roi):
        return mask_bool
    retained_fraction = float(np.count_nonzero(tightened_roi) / max(1, np.count_nonzero(candidate_roi)))
    if retained_fraction < 0.68:
        return mask_bool
    tightened_mask = mask_bool.copy()
    tightened_mask[roi_y0:roi_y1, roi_x0:roi_x1] = tightened_roi
    return tightened_mask


def _spanning_barrier_mask(
    score_map: np.ndarray,
    band_mask: np.ndarray,
    *,
    threshold: float,
    horizontal: bool,
    coverage_min: float,
) -> np.ndarray:
    """Find low-score separator components that span the minor axis inside a local band."""

    band_bool = np.asarray(band_mask, dtype=bool)
    if not np.any(band_bool):
        return np.zeros_like(band_bool, dtype=bool)
    valley_mask = band_bool & (np.asarray(score_map, dtype=np.float32) <= float(threshold))
    if not np.any(valley_mask):
        return np.zeros_like(band_bool, dtype=bool)
    labels, count = _label_components(valley_mask)
    if count <= 0:
        return np.zeros_like(band_bool, dtype=bool)
    active_minor = np.any(band_bool, axis=1 if horizontal else 0)
    minor_span = max(1, int(np.count_nonzero(active_minor)))
    spanning = np.zeros_like(band_bool, dtype=bool)
    for label_id in range(1, int(count) + 1):
        component = labels == label_id
        if not np.any(component):
            continue
        ys, xs = np.nonzero(component)
        if ys.size == 0 or xs.size == 0:
            continue
        component_minor_span = int(np.max(ys) - np.min(ys) + 1) if horizontal else int(np.max(xs) - np.min(xs) + 1)
        if float(component_minor_span / max(1, minor_span)) >= float(coverage_min):
            spanning |= component
    return spanning


def _reconstruct_large_polygon_candidate(
    probability: np.ndarray,
    local_normalized_probability: np.ndarray,
    seed_mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> _PolygonConfidenceCandidate | None:
    """Reconstruct one large polygon from a strong seed while keeping internal texture separate."""

    seed_bool = np.asarray(seed_mask, dtype=bool)
    if not np.any(seed_bool):
        return None
    seed_bbox = _mask_bbox(seed_bool)
    if seed_bbox[2] <= 0 or seed_bbox[3] <= 0:
        return None

    score_map = np.maximum(
        np.asarray(probability, dtype=np.float32),
        np.asarray(local_normalized_probability, dtype=np.float32),
    ).astype(np.float32)
    image_height, image_width = score_map.shape
    seed_x0, seed_y0, seed_w, seed_h = seed_bbox
    band_expand = max(
        int(config.large_polygon_band_expand),
        min(max(seed_w, seed_h), max(seed_w if seed_w < seed_h else seed_h, 1)),
    )
    roi_padding = max(1, int(config.large_polygon_roi_padding))
    base_support_threshold = max(
        float(config.hysteresis_low_floor),
        float(low_threshold) * float(config.large_polygon_low_scale),
    )
    seed_support_threshold = max(
        float(config.hysteresis_low_floor),
        min(
            float(base_support_threshold),
            float(low_threshold) * float(config.large_polygon_seed_low_scale),
            float(strong_threshold) * 0.40,
        ),
    )
    barrier_threshold = max(
        float(config.hysteresis_low_floor),
        float(base_support_threshold) - float(config.large_polygon_barrier_delta),
    )

    def _candidate_for_orientation(horizontal: bool) -> _PolygonConfidenceCandidate | None:
        envelope_support = np.asarray(score_map >= float(seed_support_threshold), dtype=bool)
        envelope_support |= seed_bool
        connected_envelope = _connected_support_from_seed(envelope_support, seed_bool)
        envelope_bbox = _mask_bbox(connected_envelope) if np.any(connected_envelope) else seed_bbox
        env_x0, env_y0, env_w, env_h = envelope_bbox
        env_x1 = env_x0 + env_w
        env_y1 = env_y0 + env_h
        if horizontal:
            roi_y0 = max(0, min(seed_y0, env_y0) - band_expand)
            roi_y1 = min(image_height, max(seed_y0 + seed_h, env_y1) + band_expand)
            roi_x0 = max(0, env_x0 - roi_padding)
            roi_x1 = min(image_width, env_x1 + roi_padding)
        else:
            roi_x0 = max(0, min(seed_x0, env_x0) - band_expand)
            roi_x1 = min(image_width, max(seed_x0 + seed_w, env_x1) + band_expand)
            roi_y0 = max(0, env_y0 - roi_padding)
            roi_y1 = min(image_height, env_y1 + roi_padding)
        if roi_x1 <= roi_x0 or roi_y1 <= roi_y0:
            return None
        score_roi = np.asarray(score_map[roi_y0:roi_y1, roi_x0:roi_x1], dtype=np.float32)
        seed_roi = np.asarray(seed_bool[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
        if not np.any(seed_roi):
            return None

        support_roi = np.asarray(score_roi >= float(base_support_threshold), dtype=bool)
        if horizontal:
            support_roi = _binary_close_rect(
                support_roi,
                radius_y=max(0, int(config.large_polygon_minor_close_radius)),
                radius_x=max(1, int(config.large_polygon_major_close_radius)),
            )
        else:
            support_roi = _binary_close_rect(
                support_roi,
                radius_y=max(1, int(config.large_polygon_major_close_radius)),
                radius_x=max(0, int(config.large_polygon_minor_close_radius)),
            )
        support_roi |= seed_roi

        barrier_roi = _spanning_barrier_mask(
            score_roi,
            support_roi,
            threshold=barrier_threshold,
            horizontal=horizontal,
            coverage_min=float(config.large_polygon_barrier_coverage_min),
        )
        if np.any(barrier_roi):
            if horizontal:
                barrier_roi = _binary_dilate_rect(barrier_roi, radius_y=0, radius_x=1)
            else:
                barrier_roi = _binary_dilate_rect(barrier_roi, radius_y=1, radius_x=0)
            support_roi &= ~barrier_roi
            support_roi |= seed_roi

        connected_roi = _connected_support_from_seed(support_roi, seed_roi)
        if not np.any(connected_roi):
            return None
        candidate_mask = np.zeros_like(seed_bool, dtype=bool)
        candidate_mask[roi_y0:roi_y1, roi_x0:roi_x1] = connected_roi
        return _make_polygon_candidate(
            probability, candidate_mask, ("large_polygon",), roi_bbox=(roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0)
        )

    candidates = tuple(
        candidate
        for candidate in (
            _candidate_for_orientation(horizontal=True),
            _candidate_for_orientation(horizontal=False),
        )
        if candidate is not None
    )
    if not candidates:
        return None

    def _rank(candidate: _PolygonConfidenceCandidate) -> tuple[float, float, float, float]:
        bbox_width = max(1, int(candidate.bbox[2]))
        bbox_height = max(1, int(candidate.bbox[3]))
        major_span = float(max(bbox_width, bbox_height))
        return (
            major_span,
            float(candidate.extent),
            float(max(candidate.aspect_ratio, candidate.elongation)),
            float(candidate.mean_probability),
        )

    return max(candidates, key=_rank)


def _extract_large_polygon_candidates(
    probability: np.ndarray,
    local_normalized_probability: np.ndarray,
    high_mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
    start_candidate_id: int = 1,
) -> tuple[list[_PolygonConfidenceCandidate], np.ndarray, tuple[PolygonConfidenceDebugCandidate, ...], int]:
    """Extract large polygon candidates from strong seeds before small-detail processing."""

    accepted: list[_PolygonConfidenceCandidate] = []
    debug_rows: list[PolygonConfidenceDebugCandidate] = []
    accepted_mask = np.zeros_like(np.asarray(high_mask, dtype=bool), dtype=bool)
    high_labels, high_count = _label_components(high_mask)
    candidate_id = int(start_candidate_id)
    for label_id in range(1, int(high_count) + 1):
        seed_component = high_labels == label_id
        seed_candidate = _make_polygon_candidate(probability, seed_component, ("large_polygon_seed",))
        if seed_candidate is None:
            continue
        reconstructed = _reconstruct_large_polygon_candidate(
            probability,
            local_normalized_probability,
            seed_component,
            low_threshold=low_threshold,
            strong_threshold=strong_threshold,
            config=config,
        )
        candidate = reconstructed or seed_candidate
        bbox_width = max(1, int(candidate.bbox[2]))
        bbox_height = max(1, int(candidate.bbox[3]))
        major_span = max(bbox_width, bbox_height)
        geometry_ok = (
            candidate.area >= max(1, int(config.large_polygon_min_area))
            and major_span >= max(1, int(config.large_polygon_min_major_span))
            and candidate.extent >= float(config.large_polygon_min_extent)
            and max(candidate.aspect_ratio, candidate.elongation) >= float(config.large_polygon_min_aspect_ratio)
        )
        notes: list[str] = []
        if candidate.area < max(1, int(config.large_polygon_min_area)):
            notes.append("large_area_below_min")
        if major_span < max(1, int(config.large_polygon_min_major_span)):
            notes.append("large_major_span_too_small")
        if candidate.extent < float(config.large_polygon_min_extent):
            notes.append("large_extent_too_small")
        if max(candidate.aspect_ratio, candidate.elongation) < float(config.large_polygon_min_aspect_ratio):
            notes.append("large_ratio_too_small")
        has_high_core = bool(np.any(np.asarray(high_mask, dtype=bool) & np.asarray(candidate.mask, dtype=bool)))
        accepted_flag = bool(geometry_ok and has_high_core)
        if not has_high_core:
            notes.append("large_missing_high_core")
        _append_debug_candidate(
            debug_rows, candidate_id, "large_polygon", candidate, accepted=accepted_flag, notes=tuple(notes)
        )
        if accepted_flag:
            accepted.append(candidate)
            accepted_mask |= np.asarray(candidate.mask, dtype=bool)
        candidate_id += 1
    return accepted, accepted_mask, tuple(debug_rows), candidate_id


def _should_reject_branch_spill(
    probability: np.ndarray,
    contrast_map: np.ndarray,
    candidate: _PolygonConfidenceCandidate,
    *,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> tuple[bool, tuple[str, ...]]:
    image_height, image_width = probability.shape
    border_features = _candidate_border_span_features(
        candidate.mask,
        candidate.bbox,
        probability.shape,
        cross_axis_max=float(config.spill_cross_axis_max),
    )
    touches_lr = bool(border_features["touches_lr"])
    touches_tb = bool(border_features["touches_tb"])
    if not (touches_lr or touches_tb):
        return False, ()
    area_fraction = float(candidate.area / max(1, probability.size))
    thin_cross_axis = bool(border_features["thin_cross_axis"])
    if area_fraction < float(config.spill_large_area_fraction) and not thin_cross_axis:
        return False, ()
    if float(candidate.extent) < float(config.spill_large_extent):
        return False, ()
    if float(max(candidate.aspect_ratio, candidate.elongation)) < float(config.spill_ribbon_aspect_min):
        return False, ()
    axis_coverage = float(border_features["axis_coverage"])
    if axis_coverage < float(config.spill_border_coverage_min):
        return False, ()
    mask_bool = np.asarray(candidate.mask, dtype=bool)
    boundary_separation = _candidate_boundary_separation(probability, mask_bool)
    if boundary_separation >= float(config.spill_boundary_separation_max):
        return False, ()
    interior_mask = _binary_erode(mask_bool, 1) & mask_bool
    texture_region = interior_mask if np.any(interior_mask) else mask_bool
    texture_mean = (
        float(np.mean(np.asarray(contrast_map[texture_region], dtype=np.float32), dtype=np.float64))
        if np.any(texture_region)
        else 0.0
    )
    peak_margin = float(max(0.0, float(candidate.peak_probability) - float(candidate.mean_probability)))
    core_threshold = max(
        float(strong_threshold),
        float(candidate.mean_probability)
        + max(
            float(config.spill_trim_delta),
            0.35 * peak_margin,
        ),
    )
    has_compact_core = _candidate_has_compact_core(
        probability,
        mask_bool,
        core_threshold=core_threshold,
        max_aspect_ratio=max(1.6, float(config.spill_ribbon_aspect_min) * 0.55),
    )
    reject = (
        peak_margin <= float(config.spill_peak_margin_max)
        and (
            texture_mean <= float(config.spill_low_texture_max)
            or float(candidate.mean_probability) <= float(config.spill_mean_probability_max)
        )
        and not has_compact_core
    )
    if not reject:
        return False, ()
    return True, ("branch_spill_reject",)


def _carve_enclosed_low_probability_holes(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    low_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return mask_bool
    hole_threshold = min(
        float(config.hole_probability_max),
        max(float(config.hysteresis_low_floor), float(low_threshold) * float(config.hole_probability_scale)),
    )
    low_inside = mask_bool & (np.asarray(probability, dtype=np.float32) <= hole_threshold)
    if not np.any(low_inside):
        return mask_bool
    hole_labels, hole_count = _label_components(low_inside)
    if hole_count <= 0:
        return mask_bool
    object_boundary = _boundary_mask(mask_bool)
    carved = mask_bool.copy()
    min_hole_area = max(1, int(config.hole_min_area))
    for label_id in range(1, int(hole_count) + 1):
        hole_mask = hole_labels == label_id
        hole_area = int(np.count_nonzero(hole_mask))
        if hole_area < min_hole_area:
            continue
        if np.any(hole_mask & object_boundary):
            continue
        surround = (_binary_dilate(hole_mask, 1) & mask_bool) & ~hole_mask
        if not np.any(surround):
            continue
        hole_mean = float(np.mean(np.asarray(probability[hole_mask], dtype=np.float32), dtype=np.float64))
        surround_mean = float(np.mean(np.asarray(probability[surround], dtype=np.float32), dtype=np.float64))
        if surround_mean <= hole_mean + 0.05:
            continue
        carved[hole_mask] = False
    return carved


def _split_candidate_by_spanning_valleys(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
    source_branches: tuple[str, ...],
) -> tuple[_PolygonConfidenceCandidate, ...]:
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return ()
    candidate = _make_polygon_candidate(probability, mask_bool, source_branches)
    if candidate is None:
        return ()
    x0, y0, width, height = candidate.bbox
    if width < 6 and height < 6:
        return (candidate,)
    roi_mask = np.asarray(mask_bool[y0 : y0 + height, x0 : x0 + width], dtype=bool)
    roi_prob = np.asarray(probability[y0 : y0 + height, x0 : x0 + width], dtype=np.float32)
    horizontal = width >= height
    elongated_like = float(max(candidate.aspect_ratio, candidate.elongation)) >= float(config.boundary_snap_min_aspect)
    peak_margin = float(max(0.0, float(candidate.peak_probability) - float(candidate.mean_probability)))
    valley_threshold = max(
        float(config.hysteresis_low_floor),
        min(
            float(strong_threshold) - 0.08,
            float(low_threshold) + max(0.05, float(config.spill_trim_delta) + 0.02),
        ),
    )
    if "large_polygon" in set(source_branches):
        valley_threshold = max(
            float(config.hysteresis_low_floor),
            min(
                float(strong_threshold) - 0.08,
                float(candidate.mean_probability) - max(0.04, 0.50 * peak_margin),
            ),
        )
    valley_mask = roi_mask & (roi_prob <= valley_threshold)
    if not np.any(valley_mask):
        return (candidate,)
    valley_labels, valley_count = _label_components(valley_mask)
    if valley_count <= 0:
        return (candidate,)
    carved_roi = roi_mask.copy()
    coverage_min = float(config.spill_border_coverage_min)
    minor_coverage_min = float(config.valley_minor_coverage_min)
    removed_any = False
    for label_id in range(1, int(valley_count) + 1):
        component = valley_labels == label_id
        if not np.any(component):
            continue
        ys, xs = np.nonzero(component)
        if ys.size == 0 or xs.size == 0:
            continue
        comp_width = int(np.max(xs) - np.min(xs) + 1)
        comp_height = int(np.max(ys) - np.min(ys) + 1)
        cover_x = float(comp_width / max(1, width))
        cover_y = float(comp_height / max(1, height))
        is_horizontal_separator = cover_x >= coverage_min and cover_y <= 0.35
        is_vertical_separator = cover_y >= coverage_min and cover_x <= 0.35
        if elongated_like and horizontal:
            is_vertical_separator = is_vertical_separator or (cover_y >= minor_coverage_min and cover_x <= 0.25)
        elif elongated_like and not horizontal:
            is_horizontal_separator = is_horizontal_separator or (cover_x >= minor_coverage_min and cover_y <= 0.25)
        if not (is_horizontal_separator or is_vertical_separator):
            continue
        # Slight dilation makes the separator robust to weak single-pixel bridges.
        if horizontal and is_vertical_separator:
            carved_roi &= ~_binary_dilate_rect(component, 1, 0)
        elif (not horizontal) and is_horizontal_separator:
            carved_roi &= ~_binary_dilate_rect(component, 0, 1)
        else:
            carved_roi &= ~_binary_dilate(component, 1)
        removed_any = True
    if not removed_any or not np.any(carved_roi):
        return (candidate,)
    split_labels, split_count = _label_components(carved_roi)
    if split_count <= 1:
        carved_mask = mask_bool.copy()
        carved_mask[y0 : y0 + height, x0 : x0 + width] = carved_roi
        split_candidate = _make_polygon_candidate(probability, carved_mask, source_branches)
        return (split_candidate,) if split_candidate is not None else (candidate,)
    split_candidates: list[_PolygonConfidenceCandidate] = []
    strong_mask = np.asarray(probability, dtype=np.float32) >= float(strong_threshold)
    for label_id in range(1, int(split_count) + 1):
        component_roi = split_labels == label_id
        if not np.any(component_roi):
            continue
        component_mask = np.zeros_like(mask_bool, dtype=bool)
        component_mask[y0 : y0 + height, x0 : x0 + width] = component_roi
        if not np.any(component_mask & strong_mask):
            continue
        component_candidate = _make_polygon_candidate(probability, component_mask, source_branches)
        if component_candidate is not None:
            split_candidates.append(component_candidate)
    return tuple(split_candidates) if split_candidates else (candidate,)


def _tighten_candidate_with_boundary_barriers(
    probability: np.ndarray,
    mask: np.ndarray,
    high_mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
    source_branches: tuple[str, ...],
) -> np.ndarray:
    """Trim local boundary overgrowth while preserving core-supported geometry."""

    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return mask_bool
    candidate = _make_polygon_candidate(probability, mask_bool, source_branches)
    if candidate is None:
        return mask_bool
    if candidate.area < max(6, int(config.separation_core_min_area) * 3):
        return mask_bool

    pad = max(1, int(config.separation_roi_padding))
    x0, y0, width, height = candidate.bbox
    roi_x0 = max(0, x0 - pad)
    roi_y0 = max(0, y0 - pad)
    roi_x1 = min(mask_bool.shape[1], x0 + width + pad)
    roi_y1 = min(mask_bool.shape[0], y0 + height + pad)

    candidate_roi = np.asarray(mask_bool[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    if not np.any(candidate_roi):
        return mask_bool
    probability_roi = np.asarray(probability[roi_y0:roi_y1, roi_x0:roi_x1], dtype=np.float32)
    high_roi = np.asarray(high_mask[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    core_roi = _separation_core_mask(
        candidate_roi,
        high_roi,
        min_area=max(1, int(config.separation_core_min_area)),
    )
    if not np.any(core_roi):
        return mask_bool
    _seed_labels, seed_count = _label_components(core_roi)

    boundary_cues_roi = _boundary_cue_map(
        probability_roi,
        candidate_roi,
        low_threshold=low_threshold,
        config=config,
    )
    cue_values = np.asarray(boundary_cues_roi[candidate_roi], dtype=np.float32)
    if cue_values.size <= 0:
        return mask_bool
    tighten_threshold = min(
        0.95,
        max(
            float(config.separation_barrier_threshold) + 0.08,
            float(np.quantile(cue_values, 0.72)),
        ),
    )
    thin_barrier_roi = _thin_barrier_map_from_cues(
        boundary_cues_roi,
        candidate_roi,
        threshold=tighten_threshold,
        dilate_radius=0,
    )
    if not np.any(thin_barrier_roi):
        return mask_bool

    boundary_seed = thin_barrier_roi & _binary_dilate(_boundary_mask(candidate_roi), 1)
    edge_barrier_roi = (
        _connected_support_from_seed(thin_barrier_roi, boundary_seed)
        if np.any(boundary_seed)
        else np.zeros_like(candidate_roi, dtype=bool)
    )
    bridge_cut_roi = _bridge_cut_mask(
        probability_roi,
        candidate_roi,
        boundary_cues_roi,
        core_roi,
        config=config,
    )
    thin_bridge_roi = np.asarray(_thin_bridge_map(candidate_roi) > 0.0, dtype=bool)
    internal_conflict_roi = (
        _binary_dilate(thin_barrier_roi & _binary_dilate(thin_bridge_roi, 1), 1) & candidate_roi
        if seed_count > 1 and np.any(thin_bridge_roi)
        else np.zeros_like(candidate_roi, dtype=bool)
    )
    protected_roi = core_roi
    blocked_roi = (edge_barrier_roi | internal_conflict_roi | bridge_cut_roi) & ~protected_roi
    if not np.any(blocked_roi):
        return mask_bool

    tightened_roi = _connected_support_from_seed(candidate_roi & ~blocked_roi, core_roi)
    if not np.any(tightened_roi):
        return mask_bool
    original_area = int(np.count_nonzero(candidate_roi))
    tightened_area = int(np.count_nonzero(tightened_roi))
    retained_fraction = float(tightened_area / max(1, original_area))
    min_retained_fraction = max(0.72, float(config.boundary_snap_min_retained_fraction))
    if retained_fraction < min_retained_fraction:
        return mask_bool
    if tightened_area >= original_area:
        return mask_bool

    refined_mask = mask_bool.copy()
    refined_mask[roi_y0:roi_y1, roi_x0:roi_x1] = tightened_roi
    return refined_mask


def _snap_elongated_candidate_boundaries(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    low_threshold: float,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
    source_branches: tuple[str, ...],
) -> np.ndarray:
    """Refine large elongated candidates by snapping the minor-axis boundaries to local profiles."""

    def _smooth_trace(values: np.ndarray, valid_mask: np.ndarray, window: int) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32).copy()
        valid = np.asarray(valid_mask, dtype=bool)
        if not np.any(valid):
            return result
        coords = np.flatnonzero(valid)
        samples = np.asarray(result[coords], dtype=np.float32)
        if samples.size >= 3:
            kernel = max(3, int(window) | 1)
            if ndi is not None:
                samples = np.asarray(ndi.median_filter(samples, size=kernel, mode="nearest"), dtype=np.float32)
            else:
                radius = kernel // 2
                padded = np.pad(samples, (radius, radius), mode="edge")
                smoothed = np.empty_like(samples)
                for index in range(samples.size):
                    smoothed[index] = float(np.median(padded[index : index + kernel]))
                samples = smoothed
        result[coords] = samples
        return result

    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return mask_bool
    candidate = _make_polygon_candidate(probability, mask_bool, source_branches)
    if candidate is None:
        return mask_bool
    if float(max(candidate.aspect_ratio, candidate.elongation)) < float(config.boundary_snap_min_aspect):
        return mask_bool
    x0, y0, width, height = candidate.bbox
    if width < 6 or height < 6:
        return mask_bool

    pad_y = max(2, min(6, height))
    pad_x = max(2, min(6, width))
    roi_x0 = max(0, x0 - pad_x)
    roi_y0 = max(0, y0 - pad_y)
    roi_x1 = min(mask_bool.shape[1], x0 + width + pad_x)
    roi_y1 = min(mask_bool.shape[0], y0 + height + pad_y)

    roi_mask = np.asarray(mask_bool[roi_y0:roi_y1, roi_x0:roi_x1], dtype=bool)
    roi_prob = np.asarray(probability[roi_y0:roi_y1, roi_x0:roi_x1], dtype=np.float32)
    inner_x0 = x0 - roi_x0
    inner_y0 = y0 - roi_y0
    inner_x1 = inner_x0 + width
    inner_y1 = inner_y0 + height
    inner_mask = np.asarray(roi_mask[inner_y0:inner_y1, inner_x0:inner_x1], dtype=bool)
    object_prob = np.asarray(roi_prob[roi_mask], dtype=np.float32)
    if object_prob.size == 0:
        return mask_bool

    refined_roi = np.zeros_like(roi_mask, dtype=bool)
    horizontal = width >= height
    if horizontal:
        active_cols = np.any(inner_mask, axis=0)
        if not np.any(active_cols):
            return mask_bool
        center_profile = np.mean(roi_prob[:, inner_x0:inner_x1][:, active_cols], axis=1, dtype=np.float32)
        active_rows = np.any(inner_mask, axis=1)
        if np.count_nonzero(active_rows) <= 2:
            return mask_bool
        active_row_indices = inner_y0 + np.flatnonzero(active_rows)
        edge_drop = float(np.max(center_profile[active_row_indices])) - float(
            max(center_profile[active_row_indices[0]], center_profile[active_row_indices[-1]])
        )
        if edge_drop < float(config.boundary_snap_min_drop):
            return mask_bool
        center_row = int(np.argmax(center_profile))
        top_trace = np.full(width, np.nan, dtype=np.float32)
        bottom_trace = np.full(width, np.nan, dtype=np.float32)
        valid_trace = np.zeros(width, dtype=bool)
        for local_col in np.flatnonzero(active_cols):
            col = inner_x0 + int(local_col)
            col_profile = np.asarray(roi_prob[:, col], dtype=np.float32)
            baseline = float(np.quantile(col_profile, 0.2))
            peak_row = int(np.argmax(col_profile))
            peak_value = float(col_profile[peak_row])
            profile_threshold = max(
                float(low_threshold),
                min(
                    float(strong_threshold),
                    baseline + 0.45 * max(0.0, peak_value - baseline),
                ),
            )
            keep_rows = col_profile >= profile_threshold
            if not np.any(keep_rows):
                continue
            row_labels, row_count = _label_components(keep_rows[:, None])
            selected_rows = None
            preferred_row = peak_row if keep_rows[peak_row] else center_row
            for label_id in range(1, int(row_count) + 1):
                component = row_labels[:, 0] == label_id
                if component[preferred_row]:
                    selected_rows = component
                    break
            if selected_rows is None:
                continue
            selected_indices = np.flatnonzero(selected_rows & roi_mask[:, col])
            if selected_indices.size == 0:
                continue
            top_trace[int(local_col)] = float(selected_indices[0])
            bottom_trace[int(local_col)] = float(selected_indices[-1])
            valid_trace[int(local_col)] = True
        if not np.any(valid_trace):
            return mask_bool
        active_col_indices = np.flatnonzero(active_cols)
        valid_col_indices = np.flatnonzero(valid_trace)
        top_trace = np.interp(active_col_indices, valid_col_indices, top_trace[valid_col_indices]).astype(np.float32)
        bottom_trace = np.interp(active_col_indices, valid_col_indices, bottom_trace[valid_col_indices]).astype(
            np.float32
        )
        smooth_window = max(3, min(11, width // 6 if width >= 6 else 3))
        top_smoothed = _smooth_trace(top_trace, np.ones_like(top_trace, dtype=bool), smooth_window)
        bottom_smoothed = _smooth_trace(bottom_trace, np.ones_like(bottom_trace, dtype=bool), smooth_window)
        for position, local_col in enumerate(active_col_indices.tolist()):
            col = inner_x0 + int(local_col)
            top = int(np.clip(round(float(top_smoothed[position])), 0, roi_mask.shape[0] - 1))
            bottom = int(np.clip(round(float(bottom_smoothed[position])), top, roi_mask.shape[0] - 1))
            refined_roi[top : bottom + 1, col] = roi_mask[top : bottom + 1, col]
    else:
        active_rows = np.any(inner_mask, axis=1)
        if not np.any(active_rows):
            return mask_bool
        center_profile = np.mean(roi_prob[inner_y0:inner_y1, :][active_rows, :], axis=0, dtype=np.float32)
        active_cols = np.any(inner_mask, axis=0)
        if np.count_nonzero(active_cols) <= 2:
            return mask_bool
        active_col_indices = inner_x0 + np.flatnonzero(active_cols)
        edge_drop = float(np.max(center_profile[active_col_indices])) - float(
            max(center_profile[active_col_indices[0]], center_profile[active_col_indices[-1]])
        )
        if edge_drop < float(config.boundary_snap_min_drop):
            return mask_bool
        center_col = int(np.argmax(center_profile))
        left_trace = np.full(height, np.nan, dtype=np.float32)
        right_trace = np.full(height, np.nan, dtype=np.float32)
        valid_trace = np.zeros(height, dtype=bool)
        for local_row in np.flatnonzero(active_rows):
            row = inner_y0 + int(local_row)
            row_profile = np.asarray(roi_prob[row, :], dtype=np.float32)
            baseline = float(np.quantile(row_profile, 0.2))
            peak_col = int(np.argmax(row_profile))
            peak_value = float(row_profile[peak_col])
            profile_threshold = max(
                float(low_threshold),
                min(
                    float(strong_threshold),
                    baseline + 0.45 * max(0.0, peak_value - baseline),
                ),
            )
            keep_cols = row_profile >= profile_threshold
            if not np.any(keep_cols):
                continue
            col_labels, col_count = _label_components(keep_cols[None, :])
            selected_cols = None
            preferred_col = peak_col if keep_cols[peak_col] else center_col
            for label_id in range(1, int(col_count) + 1):
                component = col_labels[0, :] == label_id
                if component[preferred_col]:
                    selected_cols = component
                    break
            if selected_cols is None:
                continue
            selected_indices = np.flatnonzero(selected_cols & roi_mask[row, :])
            if selected_indices.size == 0:
                continue
            left_trace[int(local_row)] = float(selected_indices[0])
            right_trace[int(local_row)] = float(selected_indices[-1])
            valid_trace[int(local_row)] = True
        if not np.any(valid_trace):
            return mask_bool
        active_row_indices = np.flatnonzero(active_rows)
        valid_row_indices = np.flatnonzero(valid_trace)
        left_trace = np.interp(active_row_indices, valid_row_indices, left_trace[valid_row_indices]).astype(np.float32)
        right_trace = np.interp(active_row_indices, valid_row_indices, right_trace[valid_row_indices]).astype(
            np.float32
        )
        smooth_window = max(3, min(11, height // 6 if height >= 6 else 3))
        left_smoothed = _smooth_trace(left_trace, np.ones_like(left_trace, dtype=bool), smooth_window)
        right_smoothed = _smooth_trace(right_trace, np.ones_like(right_trace, dtype=bool), smooth_window)
        for position, local_row in enumerate(active_row_indices.tolist()):
            row = inner_y0 + int(local_row)
            left = int(np.clip(round(float(left_smoothed[position])), 0, roi_mask.shape[1] - 1))
            right = int(np.clip(round(float(right_smoothed[position])), left, roi_mask.shape[1] - 1))
            refined_roi[row, left : right + 1] = roi_mask[row, left : right + 1]

    if not np.any(refined_roi):
        return mask_bool
    original_area = int(np.count_nonzero(roi_mask))
    refined_area = int(np.count_nonzero(refined_roi))
    if refined_area <= 0:
        return mask_bool
    retained_fraction = float(refined_area / max(1, original_area))
    if retained_fraction < float(config.boundary_snap_min_retained_fraction):
        return mask_bool
    refined_mask = mask_bool.copy()
    refined_mask[roi_y0:roi_y1, roi_x0:roi_x1] = refined_roi
    return refined_mask


def _refine_final_polygon_candidates(
    probability: np.ndarray,
    candidates: list[_PolygonConfidenceCandidate],
    *,
    strong_mask: np.ndarray,
    strong_threshold: float,
    config: PolygonConfidencePipelineConfig,
) -> list[_PolygonConfidenceCandidate]:
    if not candidates:
        return []
    low_threshold = _polygon_confidence_weak_threshold(float(strong_threshold), config)
    contrast_map = _local_contrast_map(probability, radius=1)
    prominence_map = _local_prominence_map(probability, radius=max(2, int(config.local_normalization_radius)))
    image_height, image_width = probability.shape
    refined_candidates: list[_PolygonConfidenceCandidate] = []
    for candidate in candidates:
        source_branch_set = set(candidate.source_branches)
        if "large_polygon" in source_branch_set:
            # Large polygons may contain broad low-confidence texture that should remain
            # inside the polygon as uncertainty, not be carved into geometric holes.
            refined_mask = np.asarray(candidate.mask, dtype=bool)
        else:
            refined_mask = _carve_enclosed_low_probability_holes(
                probability,
                candidate.mask,
                low_threshold=low_threshold,
                config=config,
            )
            refined_mask = _snap_elongated_candidate_boundaries(
                probability,
                refined_mask,
                low_threshold=low_threshold,
                strong_threshold=strong_threshold,
                config=config,
                source_branches=candidate.source_branches,
            )
            refined_mask = _tighten_candidate_with_boundary_barriers(
                probability,
                refined_mask,
                strong_mask,
                low_threshold=low_threshold,
                strong_threshold=strong_threshold,
                config=config,
                source_branches=candidate.source_branches,
            )
        split_after_valleys = _split_candidate_by_spanning_valleys(
            probability,
            refined_mask,
            low_threshold=low_threshold,
            strong_threshold=strong_threshold,
            config=config,
            source_branches=candidate.source_branches,
        )
        for split_candidate in split_after_valleys:
            candidate_mask = np.asarray(split_candidate.mask, dtype=bool)
            candidate_after_holes = _make_polygon_candidate(
                probability, candidate_mask, split_candidate.source_branches
            )
            if candidate_after_holes is None:
                continue
            source_branch_set = set(split_candidate.source_branches)
            if "large_polygon" in source_branch_set:
                # Large-polygon geometry is already reconstructed from a seed-connected local ROI
                # and optionally split by spanning valleys. Re-applying generic spill trimming here
                # collapses weak but valid wide polygons to bright core fragments.
                refined_candidates.append(candidate_after_holes)
                continue
            border_features = _candidate_border_span_features(
                candidate_mask,
                candidate_after_holes.bbox,
                probability.shape,
                cross_axis_max=float(config.spill_cross_axis_max),
            )
            touches_lr = bool(border_features["touches_lr"])
            touches_tb = bool(border_features["touches_tb"])
            touches_opposite_borders = touches_lr or touches_tb
            area_fraction = float(candidate_after_holes.area / max(1, probability.size))
            thin_cross_axis = bool(border_features["thin_cross_axis"])
            if (
                touches_opposite_borders
                and (area_fraction >= float(config.spill_large_area_fraction) or thin_cross_axis)
                and float(candidate_after_holes.extent) >= float(config.spill_large_extent)
            ):
                interior_mask = _binary_erode(candidate_mask, 1) & candidate_mask
                texture_region = interior_mask if np.any(interior_mask) else candidate_mask
                texture_mean = (
                    float(np.mean(np.asarray(contrast_map[texture_region], dtype=np.float32), dtype=np.float64))
                    if np.any(texture_region)
                    else 0.0
                )
                peak_margin = float(
                    max(
                        0.0,
                        float(candidate_after_holes.peak_probability) - float(candidate_after_holes.mean_probability),
                    )
                )
                boundary_separation = _candidate_boundary_separation(probability, candidate_mask)
                is_ribbon = float(max(candidate_after_holes.aspect_ratio, candidate_after_holes.elongation)) >= float(
                    config.spill_ribbon_aspect_min
                )
                axis_coverage = float(border_features["axis_coverage"])
                object_prob = np.asarray(probability[candidate_mask], dtype=np.float32)
                local_quantile = float(np.quantile(object_prob, 0.80)) if object_prob.size else float(strong_threshold)
                core_threshold = max(
                    float(strong_threshold),
                    float(candidate_after_holes.mean_probability)
                    + max(
                        float(config.spill_trim_delta),
                        0.35 * peak_margin,
                    ),
                    local_quantile,
                )
                strong_axis_coverage, strong_area_fraction = _candidate_axis_support_strength(
                    candidate_mask,
                    strong_mask,
                    touches_lr=touches_lr,
                    touches_tb=touches_tb,
                )
                fallback_only = source_branch_set == {"strong_mask"}
                preserve_compact_core = _candidate_has_compact_core(
                    probability,
                    candidate_mask,
                    core_threshold=core_threshold,
                    max_aspect_ratio=max(2.5, float(config.spill_ribbon_aspect_min)),
                )
                has_strong_axis_support = (
                    not fallback_only
                    and preserve_compact_core
                    and peak_margin > float(config.spill_peak_margin_max)
                    and (
                        strong_axis_coverage >= float(config.spill_strong_axis_coverage_min)
                        or strong_area_fraction >= float(config.spill_strong_area_fraction_min)
                    )
                )
                spill_like = (
                    is_ribbon
                    and axis_coverage >= float(config.spill_border_coverage_min)
                    and (peak_margin <= float(config.spill_peak_margin_max) or thin_cross_axis)
                    and (
                        texture_mean <= float(config.spill_low_texture_max)
                        or float(candidate_after_holes.mean_probability) <= float(config.spill_mean_probability_max)
                    )
                    and boundary_separation <= float(config.spill_boundary_separation_max)
                    and not has_strong_axis_support
                )
                if spill_like:
                    grow_threshold = max(
                        float(strong_threshold),
                        float(candidate_after_holes.mean_probability) + 0.5 * float(config.spill_trim_delta),
                    )
                    trimmed_mask = _retain_core_supported_region(
                        probability,
                        prominence_map,
                        candidate_mask,
                        core_threshold=core_threshold,
                        grow_threshold=grow_threshold,
                        prominence_threshold=float(config.spill_prominence_min),
                        max_core_aspect_ratio=max(2.5, float(config.spill_ribbon_aspect_min)),
                    )
                    trimmed_candidate = _make_polygon_candidate(
                        probability, trimmed_mask, split_candidate.source_branches
                    )
                    if trimmed_candidate is not None:
                        trimmed_border_features = _candidate_border_span_features(
                            trimmed_mask,
                            trimmed_candidate.bbox,
                            probability.shape,
                            cross_axis_max=float(config.spill_cross_axis_max),
                        )
                        trimmed_touches_opposite_borders = bool(trimmed_border_features["touches_lr"]) or bool(
                            trimmed_border_features["touches_tb"]
                        )
                        trimmed_axis_coverage = float(trimmed_border_features["axis_coverage"])
                        trimmed_peak_margin = float(
                            max(
                                0.0,
                                float(trimmed_candidate.peak_probability) - float(trimmed_candidate.mean_probability),
                            )
                        )
                        trimmed_boundary_separation = _candidate_boundary_separation(probability, trimmed_mask)
                        trimmed_is_ribbon = float(
                            max(trimmed_candidate.aspect_ratio, trimmed_candidate.elongation)
                        ) >= float(config.spill_ribbon_aspect_min)
                        trimmed_spill_like = (
                            trimmed_is_ribbon
                            and trimmed_touches_opposite_borders
                            and trimmed_axis_coverage >= float(config.spill_border_coverage_min)
                            and trimmed_peak_margin <= float(config.spill_peak_margin_max)
                            and trimmed_boundary_separation <= float(config.spill_boundary_separation_max)
                        )
                        if trimmed_spill_like:
                            continue
                        candidate_after_holes = trimmed_candidate
                    else:
                        continue
            refined_candidates.append(candidate_after_holes)
    return refined_candidates


def _polygon_branch_debug_mask(
    shape: tuple[int, int], candidates: tuple[_PolygonConfidenceCandidate, ...]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        mask |= np.asarray(candidate.mask, dtype=bool)
    return mask


def _polygon_confidence_pipeline(
    probability: np.ndarray,
    strong_mask: np.ndarray,
    *,
    strong_threshold: float = 0.5,
    config: PolygonConfidencePipelineConfig | None = None,
    include_debug: bool = False,
) -> tuple[np.ndarray, tuple[_PolygonConfidenceCandidate, ...], PolygonConfidenceDebugData | None]:
    cfg = config or _polygon_confidence_config()
    stage_started = perf_counter()
    stage_timings_ms: dict[str, float] = {}

    raw_prob = _normalize_probability_map(probability)
    preprocessed_prob, local_normalized_prob = _preprocess_polygon_probability(raw_prob, cfg)
    high_threshold = float(max(0.0, min(1.0, strong_threshold)))
    low_threshold = _polygon_confidence_weak_threshold(high_threshold, cfg)
    stage_timings_ms["preprocess"] = 1000.0 * (perf_counter() - stage_started)

    branch_started = perf_counter()
    global_low_mask = np.asarray(preprocessed_prob >= low_threshold, dtype=bool)
    global_high_mask = np.asarray(preprocessed_prob >= high_threshold, dtype=bool) | np.asarray(strong_mask, dtype=bool)
    branch_contrast_map = _local_contrast_map(preprocessed_prob, radius=1)

    debug_rows: list[PolygonConfidenceDebugCandidate] = []
    next_debug_id = 1

    dominant_high_threshold = max(high_threshold, float(cfg.dominant_min_mean_probability))
    dominant_high_mask = np.asarray(preprocessed_prob >= dominant_high_threshold, dtype=bool) | np.asarray(
        strong_mask, dtype=bool
    )

    large_polygon_candidates, large_polygon_mask, large_polygon_debug, next_debug_id = (
        _extract_large_polygon_candidates(
            preprocessed_prob,
            local_normalized_prob,
            dominant_high_mask,
            low_threshold=low_threshold,
            strong_threshold=high_threshold,
            config=cfg,
            start_candidate_id=next_debug_id,
        )
    )
    debug_rows.extend(large_polygon_debug)
    dominant_mask = np.asarray(large_polygon_mask, dtype=bool)
    dominant_lock_mask = (
        _binary_dilate(dominant_mask, max(0, int(cfg.dominant_lock_radius)))
        if np.any(dominant_mask)
        else np.zeros_like(global_low_mask, dtype=bool)
    )

    def _global_accept(candidate: _PolygonConfidenceCandidate, has_high_core: bool) -> tuple[bool, tuple[str, ...]]:
        reject, reject_notes = _should_reject_branch_spill(
            preprocessed_prob,
            branch_contrast_map,
            candidate,
            strong_threshold=high_threshold,
            config=cfg,
        )
        return (False, reject_notes) if reject else (True, ())

    global_candidates, global_mask, global_debug, next_debug_id = _extract_branch_candidates(
        preprocessed_prob,
        global_low_mask & ~dominant_lock_mask,
        global_high_mask & ~dominant_lock_mask,
        branch="global_hysteresis",
        min_area=1,
        require_high_core=True,
        acceptance_fn=_global_accept,
        start_candidate_id=next_debug_id,
    )
    debug_rows.extend(global_debug)

    elongated_low_mask = _binary_close_rect(
        global_low_mask, int(cfg.elongated_vertical_radius), int(cfg.elongated_horizontal_radius)
    )
    elongated_low_mask |= _binary_close_rect(
        global_low_mask, int(cfg.elongated_horizontal_radius), int(cfg.elongated_vertical_radius)
    )

    def _elongated_accept(candidate: _PolygonConfidenceCandidate, has_high_core: bool) -> tuple[bool, tuple[str, ...]]:
        notes: list[str] = []
        if candidate.area < max(1, int(cfg.elongated_min_area)):
            notes.append("elongated_area_below_min")
            return False, tuple(notes)
        if max(candidate.aspect_ratio, candidate.elongation) < float(cfg.elongated_min_aspect_ratio):
            notes.append("elongated_ratio_too_small")
            return False, tuple(notes)
        if not has_high_core:
            notes.append("missing_high_core")
            return False, tuple(notes)
        reject, reject_notes = _should_reject_branch_spill(
            preprocessed_prob,
            branch_contrast_map,
            candidate,
            strong_threshold=high_threshold,
            config=cfg,
        )
        if reject:
            notes.extend(reject_notes)
            return False, tuple(notes)
        return True, tuple(notes)

    elongated_candidates, elongated_mask, elongated_debug, next_debug_id = _extract_branch_candidates(
        preprocessed_prob,
        elongated_low_mask & ~dominant_lock_mask,
        global_high_mask & ~dominant_lock_mask,
        branch="elongated",
        min_area=int(cfg.elongated_min_area),
        require_high_core=True,
        acceptance_fn=_elongated_accept,
        start_candidate_id=next_debug_id,
    )
    debug_rows.extend(elongated_debug)

    small_low_threshold = max(float(cfg.small_mean_floor), low_threshold * float(cfg.small_low_scale))
    small_high_threshold = max(float(cfg.small_peak_floor), high_threshold * float(cfg.small_high_scale))
    small_low_mask = np.asarray(preprocessed_prob >= small_low_threshold, dtype=bool)
    small_high_mask = np.asarray(preprocessed_prob >= small_high_threshold, dtype=bool)

    def _small_accept(candidate: _PolygonConfidenceCandidate, has_high_core: bool) -> tuple[bool, tuple[str, ...]]:
        notes: list[str] = []
        peak_ok = candidate.peak_probability >= max(float(cfg.small_peak_floor), small_high_threshold)
        mean_ok = candidate.mean_probability >= float(cfg.small_mean_floor)
        if not (has_high_core or peak_ok):
            notes.append("small_missing_peak")
            return False, tuple(notes)
        if not mean_ok:
            notes.append("small_mean_too_low")
            return False, tuple(notes)
        return True, tuple(notes)

    small_candidates, small_mask, small_debug, next_debug_id = _extract_branch_candidates(
        preprocessed_prob,
        small_low_mask & ~dominant_lock_mask,
        small_high_mask & ~dominant_lock_mask,
        branch="small_weak",
        min_area=max(1, int(cfg.small_min_area)),
        max_area=max(1, int(cfg.small_max_area)),
        require_high_core=False,
        acceptance_fn=_small_accept,
        start_candidate_id=next_debug_id,
    )
    debug_rows.extend(small_debug)

    adaptive_background = _local_mean_map(preprocessed_prob, radius=max(1, int(cfg.adaptive_radius)))
    adaptive_response = np.asarray(preprocessed_prob - adaptive_background, dtype=np.float32)
    adaptive_low_mask = np.asarray(adaptive_response >= float(cfg.adaptive_low_offset), dtype=bool)
    adaptive_high_mask = np.asarray(adaptive_response >= float(cfg.adaptive_high_offset), dtype=bool)

    def _adaptive_accept(candidate: _PolygonConfidenceCandidate, has_high_core: bool) -> tuple[bool, tuple[str, ...]]:
        notes: list[str] = []
        if not has_high_core:
            notes.append("adaptive_missing_high_core")
            return False, tuple(notes)
        if candidate.mean_probability < max(
            float(cfg.small_mean_floor) * 0.9, 0.12
        ) and candidate.peak_probability < max(float(cfg.small_peak_floor), 0.18):
            notes.append("adaptive_signal_too_low")
            return False, tuple(notes)
        reject, reject_notes = _should_reject_branch_spill(
            preprocessed_prob,
            branch_contrast_map,
            candidate,
            strong_threshold=high_threshold,
            config=cfg,
        )
        if reject:
            notes.extend(reject_notes)
            return False, tuple(notes)
        return True, tuple(notes)

    adaptive_candidates, adaptive_mask, adaptive_debug, next_debug_id = _extract_branch_candidates(
        preprocessed_prob,
        adaptive_low_mask & ~dominant_lock_mask,
        adaptive_high_mask & ~dominant_lock_mask,
        branch="adaptive_local",
        min_area=max(1, int(cfg.small_min_area)),
        require_high_core=True,
        acceptance_fn=_adaptive_accept,
        start_candidate_id=next_debug_id,
    )
    debug_rows.extend(adaptive_debug)

    candidates = (
        list(large_polygon_candidates)
        + list(global_candidates)
        + list(elongated_candidates)
        + list(small_candidates)
        + list(adaptive_candidates)
    )
    if not candidates and np.any(np.asarray(strong_mask, dtype=bool)):
        fallback_candidate = _make_polygon_candidate(raw_prob, np.asarray(strong_mask, dtype=bool), ("strong_mask",))
        if fallback_candidate is not None:
            candidates = [fallback_candidate]
    stage_timings_ms["branch_extract"] = 1000.0 * (perf_counter() - branch_started)

    merge_started = perf_counter()
    merged_candidates = _merge_polygon_candidates(raw_prob, candidates, cfg)
    stage_timings_ms["initial_merge"] = 1000.0 * (perf_counter() - merge_started)

    completion_started = perf_counter()
    seed_mask = global_high_mask | small_high_mask | adaptive_high_mask | np.asarray(strong_mask, dtype=bool)
    completed_candidates = _complete_polygon_candidates(
        raw_prob,
        merged_candidates,
        strong_threshold=high_threshold,
        weak_threshold=low_threshold,
        high_seed_mask=seed_mask,
    )
    stage_timings_ms["completion"] = 1000.0 * (perf_counter() - completion_started)

    split_started = perf_counter()
    split_candidates: list[_PolygonConfidenceCandidate] = []
    separation_boundary_cues = np.zeros_like(preprocessed_prob, dtype=np.float32)
    separation_core_mask = np.zeros_like(global_low_mask, dtype=bool)
    separation_candidate_region = np.zeros_like(global_low_mask, dtype=bool)
    separation_thin_barrier = np.zeros_like(global_low_mask, dtype=bool)
    separation_bridge_cuts = np.zeros_like(global_low_mask, dtype=bool)
    separation_barrier_stops = np.zeros_like(global_low_mask, dtype=bool)
    for candidate in completed_candidates:
        if "large_polygon" in set(candidate.source_branches):
            split_candidates.append(candidate)
            continue
        separated, separation_debug = _split_polygon_candidate_by_barriers(
            raw_prob,
            candidate,
            seed_mask,
            low_threshold=low_threshold,
            strong_threshold=high_threshold,
            config=cfg,
            include_debug=include_debug,
        )
        split_candidates.extend(separated)
        if include_debug and separation_debug is not None:
            separation_boundary_cues = np.maximum(
                separation_boundary_cues,
                np.asarray(separation_debug.get("boundary_cues"), dtype=np.float32),
            )
            separation_core_mask |= np.asarray(separation_debug.get("core_seeds"), dtype=bool)
            separation_candidate_region |= np.asarray(separation_debug.get("candidate_region"), dtype=bool)
            separation_thin_barrier |= np.asarray(separation_debug.get("thin_barrier"), dtype=bool)
            separation_bridge_cuts |= np.asarray(separation_debug.get("bridge_cuts"), dtype=bool)
            separation_barrier_stops |= np.asarray(separation_debug.get("barrier_blocked"), dtype=bool)
    stage_timings_ms["split_refine"] = 1000.0 * (perf_counter() - split_started)

    final_merge_started = perf_counter()
    final_candidates = _merge_polygon_candidates(raw_prob, split_candidates or completed_candidates, cfg)
    final_candidates = _refine_final_polygon_candidates(
        raw_prob,
        final_candidates,
        strong_mask=np.asarray(strong_mask, dtype=bool),
        strong_threshold=high_threshold,
        config=cfg,
    )
    stage_timings_ms["final_merge"] = 1000.0 * (perf_counter() - final_merge_started)

    final_mask = np.zeros_like(np.asarray(strong_mask, dtype=bool), dtype=bool)
    for candidate in final_candidates:
        final_mask |= np.asarray(candidate.mask, dtype=bool)

    if not np.any(final_mask) and np.any(np.asarray(strong_mask, dtype=bool)) and not candidates:
        final_mask = np.asarray(strong_mask, dtype=bool).copy()
        fallback_candidate = _make_polygon_candidate(raw_prob, final_mask, ("strong_mask",))
        final_candidates = [fallback_candidate] if fallback_candidate is not None else []

    object_labels = (
        _candidate_object_labels(final_candidates, final_mask.shape)
        if final_candidates
        else np.zeros(final_mask.shape, dtype=np.int32)
    )

    debug_data = None
    if include_debug:
        final_debug_rows = list(debug_rows)
        for object_index, candidate in enumerate(final_candidates, start=1):
            final_debug_rows.append(
                PolygonConfidenceDebugCandidate(
                    object_id=int(object_index),
                    branch="merged",
                    source_branches=tuple(candidate.source_branches),
                    accepted=True,
                    area=int(candidate.area),
                    bbox_x=int(candidate.bbox[0]),
                    bbox_y=int(candidate.bbox[1]),
                    bbox_width=int(candidate.bbox[2]),
                    bbox_height=int(candidate.bbox[3]),
                    aspect_ratio=float(candidate.aspect_ratio),
                    elongation=float(candidate.elongation),
                    peak_probability=float(candidate.peak_probability),
                    mean_probability=float(candidate.mean_probability),
                    extent=float(candidate.extent),
                    notes=("final_object",),
                )
            )
        debug_data = PolygonConfidenceDebugData(
            preprocessed_probability=np.asarray(preprocessed_prob, dtype=np.float32),
            locally_normalized_probability=np.asarray(local_normalized_prob, dtype=np.float32),
            boundary_cues=np.asarray(separation_boundary_cues, dtype=np.float32),
            low_mask=np.asarray(global_low_mask, dtype=bool),
            high_mask=np.asarray(global_high_mask, dtype=bool),
            adaptive_low_mask=np.asarray(adaptive_low_mask, dtype=bool),
            adaptive_high_mask=np.asarray(adaptive_high_mask, dtype=bool),
            core_seed_mask=np.asarray(separation_core_mask, dtype=bool),
            candidate_region_mask=np.asarray(separation_candidate_region, dtype=bool),
            thin_barrier_map=np.asarray(separation_thin_barrier, dtype=bool),
            bridge_cut_mask=np.asarray(separation_bridge_cuts, dtype=bool),
            barrier_stop_mask=np.asarray(separation_barrier_stops, dtype=bool),
            branch_masks={
                "large_polygon": np.asarray(large_polygon_mask, dtype=bool),
                "dominant_clear": np.asarray(dominant_mask, dtype=bool),
                "global_hysteresis": np.asarray(global_mask, dtype=bool),
                "elongated": np.asarray(elongated_mask, dtype=bool),
                "small_weak": np.asarray(small_mask, dtype=bool),
                "adaptive_local": np.asarray(adaptive_mask, dtype=bool),
                "core_seeds": np.asarray(separation_core_mask, dtype=bool),
                "candidate_region": np.asarray(separation_candidate_region, dtype=bool),
                "thin_barrier": np.asarray(separation_thin_barrier, dtype=bool),
                "bridge_cuts": np.asarray(separation_bridge_cuts, dtype=bool),
                "barrier_stops": np.asarray(separation_barrier_stops, dtype=bool),
            },
            merged_mask=np.asarray(final_mask, dtype=bool),
            object_labels=np.asarray(object_labels, dtype=np.int32),
            candidate_rows=tuple(final_debug_rows),
            timings_ms={name: float(value) for name, value in stage_timings_ms.items()},
        )

    return np.asarray(final_mask, dtype=bool), tuple(final_candidates), debug_data


def _polygon_confidence_support_mask(
    probability: np.ndarray,
    strong_mask: np.ndarray,
    *,
    strong_threshold: float = 0.5,
    config: PolygonConfidencePipelineConfig | None = None,
) -> np.ndarray:
    final_mask, _candidates, _debug = _polygon_confidence_pipeline(
        probability,
        strong_mask,
        strong_threshold=float(strong_threshold),
        config=config,
        include_debug=False,
    )
    return np.asarray(final_mask, dtype=bool)


def _local_contrast_map(probability: np.ndarray, radius: int = 1) -> np.ndarray:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    local_radius = max(1, int(radius))
    if ndi is not None:
        size = 2 * local_radius + 1
        local_max = np.asarray(ndi.maximum_filter(prob, size=size, mode="nearest"), dtype=np.float32)
        local_min = np.asarray(ndi.minimum_filter(prob, size=size, mode="nearest"), dtype=np.float32)
        return np.clip(local_max - local_min, 0.0, 1.0).astype(np.float32)
    padded = np.pad(prob, local_radius, mode="edge")
    local_max = np.empty_like(prob, dtype=np.float32)
    local_min = np.empty_like(prob, dtype=np.float32)
    for y in range(prob.shape[0]):
        for x in range(prob.shape[1]):
            patch = padded[y : y + 2 * local_radius + 1, x : x + 2 * local_radius + 1]
            local_max[y, x] = float(np.max(patch))
            local_min[y, x] = float(np.min(patch))
    return np.clip(local_max - local_min, 0.0, 1.0).astype(np.float32)


def _local_mean_map(array: np.ndarray, radius: int = 1) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    local_radius = max(1, int(radius))
    if ndi is not None:
        size = 2 * local_radius + 1
        return np.asarray(ndi.uniform_filter(values, size=size, mode="nearest"), dtype=np.float32)
    padded = np.pad(values, local_radius, mode="edge")
    result = np.empty_like(values, dtype=np.float32)
    patch_size = float((2 * local_radius + 1) ** 2)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            patch = padded[y : y + 2 * local_radius + 1, x : x + 2 * local_radius + 1]
            result[y, x] = float(np.sum(patch, dtype=np.float64) / patch_size)
    return result


def _polygon_transition_uncertainty_maps(
    probability: np.ndarray, support_mask: np.ndarray, *, contrast_radius: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    support = np.asarray(support_mask, dtype=bool)
    local_radius = max(1, int(contrast_radius))
    uncertainty_map = _uncertainty_map_from_probability(prob)
    local_contrast = _local_contrast_map(prob, radius=local_radius)
    width_raw = _local_mean_map(uncertainty_map, radius=local_radius)
    sample_mask = (
        np.asarray(_binary_dilate(support, radius=local_radius), dtype=bool)
        if np.any(support)
        else np.ones_like(support, dtype=bool)
    )

    def _scale(values: np.ndarray) -> float:
        valid = np.asarray(values, dtype=np.float32)
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            return 1.0
        scale = float(np.percentile(valid, 95.0))
        if scale <= EPS:
            scale = float(np.max(valid)) if valid.size else 1.0
        return max(scale, EPS)

    width_scale = _scale(width_raw[sample_mask])
    contrast_scale = _scale(local_contrast[sample_mask])
    transition_width_map = np.clip(width_raw / width_scale, 0.0, 1.0).astype(np.float32)
    inverse_local_contrast = (1.0 - np.clip(local_contrast / contrast_scale, 0.0, 1.0)).astype(np.float32)
    transition_uncertainty_map = np.clip(
        uncertainty_map * transition_width_map * (0.35 + 0.65 * inverse_local_contrast),
        0.0,
        1.0,
    ).astype(np.float32)
    return transition_width_map, inverse_local_contrast, transition_uncertainty_map


def _polygon_internal_confidence(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    boundary_radius: int = 1,
    uncertainty_delta: float = MODEL_CONFIDENCE_UNCERTAIN_DELTA,
    summary_metric: str = POLYGON_CONFIDENCE_SUMMARY_WEIGHTED,
    strong_threshold: float = 0.5,
    include_objects: bool = True,
    include_debug: bool = False,
    config: PolygonConfidencePipelineConfig | None = None,
) -> PolygonConfidenceMetrics:
    prob = _internal_confidence_probability_map(probability, support_mask=np.asarray(mask, dtype=bool))
    pipeline_config = config or _polygon_confidence_config()
    mask_bool, final_candidates, debug_data = _polygon_confidence_pipeline(
        prob,
        np.asarray(mask, dtype=bool),
        strong_threshold=float(strong_threshold),
        config=pipeline_config,
        include_debug=include_debug,
    )
    confidence_map = _confidence_map_from_probability(prob)
    object_count = int(np.count_nonzero(mask_bool))
    area_fraction = float(object_count / max(1, mask_bool.size))
    frame_uncertainty_score, mean_uncertainty, low_conf_fraction, worst_tail_uncertainty, largest_low_conf_component = (
        _frame_uncertainty_components_from_probability(
            prob,
            support_threshold=POLYGON_SUPPORT_THRESHOLD,
        )
    )
    normalized_summary_metric = str(summary_metric or POLYGON_CONFIDENCE_SUMMARY_WEIGHTED).strip().lower()
    if normalized_summary_metric != POLYGON_CONFIDENCE_SUMMARY_CORE:
        normalized_summary_metric = POLYGON_CONFIDENCE_SUMMARY_WEIGHTED
    if object_count <= 0:
        return PolygonConfidenceMetrics(
            frame_uncertainty_score=0.0,
            mean_uncertainty=0.0,
            uncertain_support_fraction=0.0,
            top_uncertainty_mean=0.0,
            largest_uncertain_region_fraction=0.0,
            mean_object_confidence=0.0,
            mean_core_confidence=0.0,
            mean_boundary_uncertainty=0.0,
            mean_weighted_confidence=0.0,
            mean_object_probability=0.0,
            uncertain_fraction=0.0,
            mean_transition_width=0.0,
            object_area_fraction=area_fraction,
            polygon_count=0,
            summary_metric=normalized_summary_metric,
            low_conf_fraction=0.0,
            worst_tail_uncertainty=0.0,
            largest_low_conf_component=0.0,
            objects=(),
            debug_data=debug_data if include_debug else None,
        )

    object_rows: list[PolygonObjectConfidence] = []
    aggregate_summary: list[tuple[float, float]] = []
    aggregate_core: list[tuple[float, float]] = []
    aggregate_boundary: list[tuple[float, float]] = []
    aggregate_weighted: list[tuple[float, float]] = []
    aggregate_probability: list[tuple[float, float]] = []
    aggregate_uncertain: list[tuple[float, float]] = []
    aggregate_transition_width: list[tuple[float, float]] = []
    object_total = 0
    delta = float(max(EPS, uncertainty_delta))
    erosion_radius = max(0, int(boundary_radius))
    transition_width_map, _inverse_local_contrast, transition_uncertainty_map = _polygon_transition_uncertainty_maps(
        prob,
        mask_bool,
        contrast_radius=max(1, erosion_radius or 1),
    )

    candidate_entries: list[tuple[int, np.ndarray, tuple[str, ...], tuple[int, int, int, int], float, float]] = []
    if final_candidates:
        for object_id, candidate in enumerate(final_candidates, start=1):
            object_mask = np.asarray(candidate.mask, dtype=bool) & mask_bool
            if not np.any(object_mask):
                continue
            candidate_entries.append(
                (
                    int(object_id),
                    object_mask,
                    tuple(candidate.source_branches) or ("merged",),
                    tuple(candidate.bbox),
                    float(candidate.aspect_ratio),
                    float(candidate.elongation),
                )
            )
    else:
        labels, label_count = _label_components(mask_bool)
        for label_id in range(1, int(label_count) + 1):
            object_mask = labels == label_id
            if not np.any(object_mask):
                continue
            bbox = _mask_bbox(object_mask)
            _area, _bbox, aspect_ratio, elongation, _extent = _mask_geometry(object_mask)
            candidate_entries.append(
                (
                    int(label_id),
                    object_mask,
                    ("merged",),
                    tuple(bbox),
                    float(aspect_ratio),
                    float(elongation),
                )
            )

    for object_id, object_mask, source_branches, bbox, aspect_ratio, elongation in candidate_entries:
        area = int(np.count_nonzero(object_mask))
        if area <= 0:
            continue
        object_total += 1
        morphological_interior = (
            _binary_erode(object_mask, erosion_radius) & object_mask if erosion_radius > 0 else object_mask.copy()
        )

        object_conf = np.asarray(confidence_map[object_mask], dtype=np.float32)
        object_prob = np.asarray(prob[object_mask], dtype=np.float32)
        object_peak = float(np.max(object_prob)) if object_prob.size > 0 else 0.0
        object_mean = float(np.mean(object_prob, dtype=np.float64)) if object_prob.size > 0 else 0.0
        plateau_threshold = max(0.5, object_mean + 0.25 * max(0.0, object_peak - object_mean))
        plateau_mask = object_mask & (prob >= plateau_threshold)
        if np.any(morphological_interior & plateau_mask):
            interior = morphological_interior & plateau_mask
        elif np.any(plateau_mask):
            interior = plateau_mask
        elif np.any(morphological_interior):
            interior = morphological_interior
        else:
            interior = object_mask.copy()

        boundary_band = object_mask & np.logical_not(interior)
        if not np.any(boundary_band):
            boundary_band = object_mask.copy()

        core_confidence = float(np.mean(confidence_map[interior], dtype=np.float64))
        mean_confidence = float(np.mean(object_conf, dtype=np.float64)) if object_conf.size > 0 else 0.0
        median_confidence = float(np.median(object_conf)) if object_conf.size > 0 else 0.0
        min_confidence = float(np.min(object_conf)) if object_conf.size > 0 else 0.0
        max_confidence = float(np.max(object_conf)) if object_conf.size > 0 else 0.0
        low_percentile_confidence = float(np.percentile(object_conf, 25.0)) if object_conf.size > 0 else 0.0
        boundary_transition_uncertainty = np.asarray(transition_uncertainty_map[boundary_band], dtype=np.float32)
        boundary_uncertainty = float(np.mean(boundary_transition_uncertainty, dtype=np.float64))
        boundary_transition_width = np.asarray(transition_width_map[boundary_band], dtype=np.float32)
        if boundary_transition_width.size > 0 and boundary_transition_uncertainty.size > 0:
            transition_width_mean = float(
                np.mean(boundary_transition_width * boundary_transition_uncertainty, dtype=np.float64)
            )
        else:
            transition_width_mean = 0.0
        if ndi is not None:
            weights = np.asarray(ndi.distance_transform_edt(object_mask), dtype=np.float32)[object_mask]
        else:
            weights = np.ones(area, dtype=np.float32)
        if weights.size == 0 or float(np.max(weights)) <= EPS:
            normalized_weights = np.ones(area, dtype=np.float32)
        else:
            normalized_weights = np.clip(weights / float(np.max(weights)), EPS, 1.0).astype(np.float32)
        weighted_confidence = float(np.average(object_conf, weights=normalized_weights))
        mean_probability = float(np.mean(object_prob, dtype=np.float64))
        uncertain_fraction = float(np.mean(np.abs(object_prob - 0.5) < delta, dtype=np.float64))
        ys, xs = np.nonzero(object_mask)
        centroid_x = float(np.mean(xs, dtype=np.float64)) if xs.size else 0.0
        centroid_y = float(np.mean(ys, dtype=np.float64)) if ys.size else 0.0
        source_branch = source_branches[0] if len(source_branches) == 1 else "merged"
        summary_confidence = (
            weighted_confidence if normalized_summary_metric == POLYGON_CONFIDENCE_SUMMARY_WEIGHTED else core_confidence
        )
        if include_objects:
            object_rows.append(
                PolygonObjectConfidence(
                    object_id=int(object_id),
                    area=area,
                    area_fraction=float(area / max(1, mask_bool.size)),
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    core_confidence=core_confidence,
                    boundary_uncertainty=boundary_uncertainty,
                    weighted_confidence=weighted_confidence,
                    summary_confidence=summary_confidence,
                    mean_probability=mean_probability,
                    mean_confidence=mean_confidence,
                    median_confidence=median_confidence,
                    min_confidence=min_confidence,
                    max_confidence=max_confidence,
                    low_percentile_confidence=low_percentile_confidence,
                    uncertain_fraction=uncertain_fraction,
                    transition_width_mean=transition_width_mean,
                    bbox_x=int(bbox[0]),
                    bbox_y=int(bbox[1]),
                    bbox_width=int(bbox[2]),
                    bbox_height=int(bbox[3]),
                    aspect_ratio=float(aspect_ratio),
                    elongation=float(elongation),
                    source_branch=source_branch,
                    source_branches=tuple(source_branches),
                )
            )
        weight = float(area)
        aggregate_summary.append((summary_confidence, weight))
        aggregate_core.append((core_confidence, weight))
        aggregate_boundary.append((boundary_uncertainty, weight))
        aggregate_weighted.append((weighted_confidence, weight))
        aggregate_probability.append((mean_probability, weight))
        aggregate_uncertain.append((uncertain_fraction, weight))
        aggregate_transition_width.append((transition_width_mean, weight))

    return PolygonConfidenceMetrics(
        frame_uncertainty_score=frame_uncertainty_score,
        mean_uncertainty=mean_uncertainty,
        uncertain_support_fraction=low_conf_fraction,
        top_uncertainty_mean=worst_tail_uncertainty,
        largest_uncertain_region_fraction=largest_low_conf_component,
        mean_object_confidence=_weighted_mean(aggregate_summary),
        mean_core_confidence=_weighted_mean(aggregate_core),
        mean_boundary_uncertainty=_weighted_mean(aggregate_boundary),
        mean_weighted_confidence=_weighted_mean(aggregate_weighted),
        mean_object_probability=_weighted_mean(aggregate_probability),
        uncertain_fraction=_weighted_mean(aggregate_uncertain),
        mean_transition_width=_weighted_mean(aggregate_transition_width),
        object_area_fraction=area_fraction,
        polygon_count=int(object_total),
        summary_metric=normalized_summary_metric,
        low_conf_fraction=low_conf_fraction,
        worst_tail_uncertainty=worst_tail_uncertainty,
        largest_low_conf_component=largest_low_conf_component,
        objects=tuple(object_rows),
        debug_data=debug_data if include_debug else None,
    )
