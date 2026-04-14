from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ...contour_extractor import extract_polygons
from ...domain import PolygonData
from ...pipeline import PreprocessingPipeline
from ...utils import ensure_binary_mask, ensure_uint8
from ..processing import ContourExtractionSettings


@dataclass(frozen=True, slots=True)
class AutoTuneResult:
    pipeline_config: dict[str, Any]
    contour_settings: ContourExtractionSettings
    score: float
    mask_score: float
    roi_bbox: tuple[int, int, int, int]
    evaluations: int


def auto_tune_pipeline(
    source_image: Any,
    reference_polygons: list[PolygonData],
) -> AutoTuneResult:
    image = ensure_uint8(np.asarray(source_image))
    if image.ndim not in (2, 3):
        raise ValueError("Auto tune expects a grayscale-compatible image.")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("Auto tune expects a grayscale-compatible image.")
    if not reference_polygons:
        raise ValueError("Auto tune requires at least one reference polygon.")

    roi_bbox = _reference_roi_bbox(reference_polygons, image.shape)
    crop = _crop_image(image, roi_bbox)
    local_reference_polygons = _shift_polygons(reference_polygons, -roi_bbox[0], -roi_bbox[1])
    target_mask = _render_polygon_mask(crop.shape, local_reference_polygons)
    if not np.any(target_mask):
        raise ValueError("Reference polygons do not cover any pixels.")

    stats = _build_reference_stats(crop, target_mask)
    pipeline_candidates = _build_pipeline_candidates(crop.shape, stats)
    best_mask_candidates, mask_evaluations = _rank_mask_candidates(crop, target_mask, pipeline_candidates)
    contour_candidates = _build_contour_candidates(local_reference_polygons, crop.shape)

    reference_topology = _mask_topology(target_mask)
    reference_count = max(1, sum(1 for polygon in local_reference_polygons if not polygon.is_hole))

    best_result: AutoTuneResult | None = None
    evaluations = 0

    for mask_candidate in best_mask_candidates:
        mask = mask_candidate["mask"]
        mask_score = float(mask_candidate["score"])
        for contour_settings in contour_candidates:
            polygons = extract_polygons(mask, contour_settings)
            polygon_mask = _render_polygon_mask(crop.shape, polygons)
            polygon_score = _mask_quality_score(target_mask, polygon_mask, reference_topology)
            predicted_count = max(1, sum(1 for polygon in polygons if not polygon.is_hole))
            count_penalty = 1.0 / (1.0 + abs(predicted_count - reference_count) * 0.25)
            final_score = (polygon_score * 0.85 + mask_score * 0.15) * count_penalty
            evaluations += 1
            if best_result is None or final_score > best_result.score:
                best_result = AutoTuneResult(
                    pipeline_config=dict(mask_candidate["config"]),
                    contour_settings=ContourExtractionSettings.from_dict(contour_settings.to_dict()),
                    score=float(final_score),
                    mask_score=mask_score,
                    roi_bbox=roi_bbox,
                    evaluations=mask_evaluations + evaluations,
                )

    if best_result is None:
        raise RuntimeError("Auto tune could not find a matching configuration.")
    return AutoTuneResult(
        pipeline_config=best_result.pipeline_config,
        contour_settings=best_result.contour_settings,
        score=best_result.score,
        mask_score=best_result.mask_score,
        roi_bbox=best_result.roi_bbox,
        evaluations=mask_evaluations + evaluations,
    )


def _build_reference_stats(image: np.ndarray, target_mask: np.ndarray) -> dict[str, Any]:
    gray = _to_gray(image)
    target_pixels = gray[target_mask > 0]
    background_pixels = gray[target_mask == 0]
    if target_pixels.size == 0:
        raise ValueError("Reference mask is empty.")
    if background_pixels.size == 0:
        background_pixels = gray.reshape(-1)

    inside_mean = float(np.mean(target_pixels))
    outside_mean = float(np.mean(background_pixels))
    object_is_brighter = inside_mean >= outside_mean

    threshold_values = {
        _clamp_uint8(round((inside_mean + outside_mean) / 2.0)),
        _clamp_uint8(round(np.percentile(target_pixels, 35))),
        _clamp_uint8(round(np.percentile(target_pixels, 65))),
        _clamp_uint8(round(np.percentile(background_pixels, 35))),
        _clamp_uint8(round(np.percentile(background_pixels, 65))),
        _clamp_uint8(round((np.percentile(target_pixels, 25) + np.percentile(background_pixels, 75)) / 2.0)),
        _clamp_uint8(round((np.percentile(target_pixels, 75) + np.percentile(background_pixels, 25)) / 2.0)),
    }

    component_count, component_areas, component_perimeters = _component_stats(target_mask)
    topology = _mask_topology(target_mask)

    selected_colors, color_deltas, color_distance = _color_candidates(image, target_mask)

    return {
        "object_is_brighter": object_is_brighter,
        "threshold_values": _select_threshold_values(sorted(threshold_values)),
        "component_count": component_count,
        "component_areas": component_areas,
        "component_perimeters": component_perimeters,
        "has_holes": topology["holes"] > 0,
        "selected_colors": selected_colors,
        "color_deltas": color_deltas,
        "color_distance": color_distance,
    }


def _build_pipeline_candidates(image_shape: tuple[int, ...], stats: dict[str, Any]) -> list[dict[str, Any]]:
    preferred_threshold_type = "binary" if stats["object_is_brighter"] else "binary_inv"
    fallback_threshold_type = "binary_inv" if preferred_threshold_type == "binary" else "binary"
    threshold_types = [preferred_threshold_type, fallback_threshold_type]
    scales = [1.0, 1.5, 2.0]

    min_component_area = max(1.0, min(stats["component_areas"], default=0.0) * 0.35)
    min_component_perimeter = max(1.0, min(stats["component_perimeters"], default=0.0) * 0.4)
    block_sizes = _candidate_block_sizes(image_shape)

    filter_candidates = [
        [],
        [_step("clahe", clip_limit=2.0, tile_grid_size=8)],
        [_step("gaussian_blur", kernel_size=3, sigma_x=0.0)],
        [_step("gaussian_blur", kernel_size=5, sigma_x=0.0)],
        [_step("median_blur", kernel_size=3)],
        [_step("bilateral_filter", diameter=7, sigma_color=60.0, sigma_space=60.0)],
        [_step("clahe", clip_limit=2.0, tile_grid_size=8), _step("gaussian_blur", kernel_size=3, sigma_x=0.0)],
        [_step("sharpen", amount=1.2, sigma=1.0)],
    ]

    postprocess_candidates = [
        [],
        [_step("morph_open", kernel_size=3, iterations=1, shape="ellipse")],
        [_step("morph_close", kernel_size=3, iterations=1, shape="ellipse")],
        [_step("binary_fill_holes")],
        [_step("morph_open", kernel_size=3, iterations=1, shape="ellipse"), _step("binary_fill_holes")],
        [_step("morph_close", kernel_size=3, iterations=1, shape="ellipse"), _step("binary_fill_holes")],
        [_step("binary_fill_holes"), _step("binary_filter_area", min_component_area=min_component_area, max_component_area=0.0)],
        [_step("binary_fill_holes"), _step("binary_filter_perimeter", min_component_perimeter=min_component_perimeter, max_component_perimeter=0.0)],
    ]
    if stats["component_count"] > 1:
        postprocess_candidates.extend(
            [
                [_step("binary_fill_holes"), _step("watershed_split", distance_ratio=0.35, min_peak_area=2, kernel_size=3, shape="ellipse", background_iterations=1)],
                [_step("morph_close", kernel_size=3, iterations=1, shape="ellipse"), _step("binary_fill_holes"), _step("watershed_split", distance_ratio=0.3, min_peak_area=2, kernel_size=3, shape="ellipse", background_iterations=1)],
            ]
        )

    region_binarizers: list[list] = []
    for threshold_type in threshold_types:
        region_binarizers.append([_step("otsu_threshold", threshold_type=threshold_type)])
        for threshold_value in stats["threshold_values"]:
            region_binarizers.append(
                [_step("threshold", threshold=threshold_value, max_value=255.0, threshold_type=threshold_type)]
            )
    for block_size in block_sizes:
        for c_value in (-4.0, 0.0, 4.0):
            region_binarizers.append(
                [
                    _step(
                        "adaptive_threshold",
                        max_value=255.0,
                        adaptive_method="gaussian",
                        threshold_type=preferred_threshold_type,
                        block_size=block_size,
                        c_value=c_value,
                    )
                ]
            )

    edge_candidates = [
        [_step("canny", threshold1=25.0, threshold2=80.0, aperture_size=3, l2gradient=False), _step("dilate", kernel_size=3, iterations=1, shape="ellipse"), _step("binary_fill_holes")],
        [_step("canny", threshold1=40.0, threshold2=120.0, aperture_size=3, l2gradient=False), _step("morph_close", kernel_size=3, iterations=1, shape="ellipse"), _step("binary_fill_holes")],
        [_step("canny", threshold1=60.0, threshold2=160.0, aperture_size=3, l2gradient=True), _step("morph_close", kernel_size=5, iterations=1, shape="ellipse"), _step("binary_fill_holes")],
    ]

    unique_candidates: dict[str, dict[str, Any]] = {}

    for scale in scales:
        for filter_steps in filter_candidates:
            for binarizer_steps in region_binarizers:
                for postprocess_steps in postprocess_candidates:
                    steps = _scaled_steps(scale, filter_steps + binarizer_steps + postprocess_steps)
                    config = PreprocessingPipeline(steps).to_dict()
                    unique_candidates.setdefault(_pipeline_signature(config), config)

        edge_filters = [[], [_step("gaussian_blur", kernel_size=3, sigma_x=0.0)], [_step("bilateral_filter", diameter=7, sigma_color=60.0, sigma_space=60.0)], [_step("sharpen", amount=1.2, sigma=1.0)]]
        for filter_steps in edge_filters:
            for edge_steps in edge_candidates:
                steps = _scaled_steps(scale, filter_steps + edge_steps)
                config = PreprocessingPipeline(steps).to_dict()
                unique_candidates.setdefault(_pipeline_signature(config), config)

        if stats["selected_colors"] and stats["color_distance"] >= 18.0:
            color_post = [
                [],
                [_step("binary_fill_holes")],
                [_step("binary_fill_holes"), _step("binary_filter_area", min_component_area=min_component_area, max_component_area=0.0)],
            ]
            if stats["component_count"] > 1:
                color_post.append(
                    [_step("binary_fill_holes"), _step("watershed_split", distance_ratio=0.35, min_peak_area=2, kernel_size=3, shape="ellipse", background_iterations=1)]
                )
            for delta in stats["color_deltas"]:
                for postprocess_steps in color_post:
                    steps = _scaled_steps(
                        scale,
                        [
                            _step("color_binarize", delta=delta, selected_colors=stats["selected_colors"]),
                            *postprocess_steps,
                        ],
                    )
                    config = PreprocessingPipeline(steps).to_dict()
                    unique_candidates.setdefault(_pipeline_signature(config), config)

    return list(unique_candidates.values())


def _rank_mask_candidates(
    crop: np.ndarray,
    target_mask: np.ndarray,
    pipeline_candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    reference_topology = _mask_topology(target_mask)
    ranked: list[dict[str, Any]] = []
    for config in pipeline_candidates:
        mask = ensure_binary_mask(PreprocessingPipeline.from_dict(config).apply(crop))
        mask = _ensure_mask_shape(mask, target_mask.shape)
        score = _mask_quality_score(target_mask, mask, reference_topology)
        operations = {str(step.get("operation", "")) for step in config.get("steps", [])}
        if crop.ndim == 3 and "color_binarize" in operations:
            score += 0.02
        if reference_topology["components"] > 1 and "watershed_split" in operations:
            score += 0.02
        ranked.append({"config": config, "mask": mask, "score": score})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:12], len(ranked)


def _build_contour_candidates(
    reference_polygons: list[PolygonData],
    image_shape: tuple[int, ...],
) -> list[ContourExtractionSettings]:
    areas = [polygon.area for polygon in reference_polygons if not polygon.is_hole and polygon.area > 0.0]
    perimeters = [polygon.perimeter for polygon in reference_polygons if not polygon.is_hole and polygon.perimeter > 0.0]
    object_type = _infer_object_type(reference_polygons)
    reference_touches_border = _reference_touches_border(reference_polygons, image_shape)

    min_area_candidates = [0.0]
    min_perimeter_candidates = [0.0]
    if areas:
        minimum_area = min(areas)
        min_area_candidates.extend([max(1.0, minimum_area * 0.15), max(1.0, minimum_area * 0.4)])
    if perimeters:
        minimum_perimeter = min(perimeters)
        min_perimeter_candidates.extend([max(1.0, minimum_perimeter * 0.2)])

    retrieval_modes = ["RETR_EXTERNAL", "RETR_CCOMP"]
    if any(polygon.is_hole for polygon in reference_polygons):
        retrieval_modes.append("RETR_TREE")

    epsilon_modes = [
        (0.0, False),
        (0.75, False),
        (1.5, False),
        (2.5, False),
        (0.0015, True),
        (0.003, True),
    ]

    unique_candidates: dict[str, ContourExtractionSettings] = {}
    for retrieval_mode in retrieval_modes:
        for approximation_mode in ("CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_NONE"):
            for epsilon, epsilon_relative in epsilon_modes:
                for min_area in min_area_candidates:
                    for min_perimeter in min_perimeter_candidates:
                        candidate = ContourExtractionSettings(
                            retrieval_mode=retrieval_mode,
                            approximation_mode=approximation_mode,
                            epsilon=epsilon,
                            epsilon_relative=epsilon_relative,
                            preserve_corners=True,
                            min_area=min_area,
                            max_area=None,
                            min_perimeter=min_perimeter,
                            min_points=3,
                            object_type=object_type,
                            output_mode="box" if object_type == "via" else "polygon",
                            exclude_border_touching=not reference_touches_border,
                        )
                        signature = json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True)
                        unique_candidates.setdefault(signature, candidate)
    return list(unique_candidates.values())


def _mask_quality_score(
    target_mask: np.ndarray,
    candidate_mask: np.ndarray,
    reference_topology: dict[str, int] | None = None,
) -> float:
    reference_topology = reference_topology or _mask_topology(target_mask)
    iou = _mask_iou(target_mask, candidate_mask)
    boundary = _boundary_f1(target_mask, candidate_mask)
    topology = _topology_score(reference_topology, _mask_topology(candidate_mask))
    return float(iou * 0.55 + boundary * 0.30 + topology * 0.15)


def _boundary_f1(first_mask: np.ndarray, second_mask: np.ndarray, tolerance: int = 2) -> float:
    first_boundary = _boundary_map(first_mask)
    second_boundary = _boundary_map(second_mask)
    first_count = int(cv2.countNonZero(first_boundary))
    second_count = int(cv2.countNonZero(second_boundary))
    if first_count == 0 and second_count == 0:
        return 1.0
    if first_count == 0 or second_count == 0:
        return 0.0

    kernel_size = max(1, tolerance * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    first_dilated = cv2.dilate(first_boundary, kernel, iterations=1)
    second_dilated = cv2.dilate(second_boundary, kernel, iterations=1)

    precision = float(cv2.countNonZero(cv2.bitwise_and(second_boundary, first_dilated)) / second_count)
    recall = float(cv2.countNonZero(cv2.bitwise_and(first_boundary, second_dilated)) / first_count)
    if precision + recall == 0.0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def _boundary_map(mask: np.ndarray) -> np.ndarray:
    mask = ensure_binary_mask(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)


def _mask_topology(mask: np.ndarray) -> dict[str, int]:
    binary = ensure_binary_mask(mask)
    contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"components": 0, "holes": 0}
    hierarchy_array = hierarchy[0] if hierarchy is not None else np.full((len(contours), 4), -1, dtype=np.int32)
    components = 0
    holes = 0
    for index, contour in enumerate(contours):
        if contour is None or len(contour) < 3:
            continue
        area = abs(float(cv2.contourArea(contour)))
        if area <= 0.0:
            continue
        if int(hierarchy_array[index][3]) == -1:
            components += 1
        else:
            holes += 1
    return {"components": components, "holes": holes}


def _topology_score(reference: dict[str, int], predicted: dict[str, int]) -> float:
    component_gap = abs(int(reference["components"]) - int(predicted["components"]))
    hole_gap = abs(int(reference["holes"]) - int(predicted["holes"]))
    return float(1.0 / (1.0 + component_gap * 0.6 + hole_gap * 0.8))


def _component_stats(mask: np.ndarray) -> tuple[int, list[float], list[float]]:
    binary = ensure_binary_mask(mask)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    areas: list[float] = []
    perimeters: list[float] = []
    for label_index in range(1, count):
        area = float(stats[label_index, cv2.CC_STAT_AREA])
        if area <= 0.0:
            continue
        component_mask = np.where(labels == label_index, 255, 0).astype(np.uint8)
        contours, _hierarchy = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        areas.append(area)
        perimeters.append(perimeter)
    return len(areas), areas, perimeters


def _color_candidates(image: np.ndarray, target_mask: np.ndarray) -> tuple[list[dict[str, Any]], list[int], float]:
    if image.ndim != 3:
        return [], [], 0.0

    rgb = _to_rgb(image)
    target_pixels = rgb[target_mask > 0].reshape((-1, 3))
    background_pixels = rgb[target_mask == 0].reshape((-1, 3))
    if target_pixels.size == 0:
        return [], [], 0.0
    if background_pixels.size == 0:
        background_pixels = rgb.reshape((-1, 3))

    foreground = _sample_rows(target_pixels.astype(np.float32), 4096)
    background = _sample_rows(background_pixels.astype(np.float32), 4096)
    color_distance = float(np.linalg.norm(np.mean(foreground, axis=0) - np.mean(background, axis=0)))

    channel_std = float(np.mean(np.std(foreground, axis=0)))
    cluster_count = 1 if channel_std < 10.0 or len(foreground) < 32 else 2 if channel_std < 22.0 else 3

    if cluster_count == 1:
        centers = np.mean(foreground, axis=0, keepdims=True)
        labels = np.zeros((len(foreground), 1), dtype=np.int32)
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        _compactness, labels, centers = cv2.kmeans(
            foreground.astype(np.float32),
            cluster_count,
            None,
            criteria,
            4,
            cv2.KMEANS_PP_CENTERS,
        )

    centers = np.clip(np.round(centers), 0, 255).astype(np.uint8)
    label_ids, counts = np.unique(labels.reshape(-1), return_counts=True)
    ordered_centers = [centers[int(label_id)] for label_id in label_ids[np.argsort(counts)[::-1]]]
    selected_colors = [{"rgb": [int(channel) for channel in center.tolist()], "enabled": True} for center in ordered_centers]

    distances = []
    foreground_centers = np.asarray(ordered_centers, dtype=np.float32)
    for pixel in foreground.astype(np.float32):
        deltas = foreground_centers - pixel
        distances.append(float(np.min(np.linalg.norm(deltas, axis=1))))
    if not distances:
        return selected_colors, [12], color_distance
    base_delta = int(max(8, min(48, round(np.percentile(distances, 75)))))
    deltas = sorted({base_delta, min(56, base_delta + 8), min(64, base_delta + 16)})
    return selected_colors, deltas, color_distance


def _scaled_steps(scale: float, steps: list) -> list:
    result = []
    if abs(scale - 1.0) > 1e-6:
        result.append(_step("scale_resize", scale=scale, interpolation="linear"))
    result.extend(step.clone() for step in steps)
    if abs(scale - 1.0) > 1e-6:
        result.append(_step("scale_resize", scale=(1.0 / scale), interpolation="nearest"))
    return result


def _pipeline_signature(config: dict[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=False, sort_keys=True)


def _infer_object_type(reference_polygons: list[PolygonData]) -> str:
    via_votes = sum(1 for polygon in reference_polygons if polygon.category == "via" or polygon.shape_hint == "box")
    return "via" if via_votes > len(reference_polygons) / 2.0 else "conductor"


def _reference_touches_border(reference_polygons: list[PolygonData], image_shape: tuple[int, ...]) -> bool:
    height, width = image_shape[:2]
    for polygon in reference_polygons:
        x_coord, y_coord, box_width, box_height = polygon.bbox
        if x_coord <= 0 or y_coord <= 0 or x_coord + box_width >= width or y_coord + box_height >= height:
            return True
    return False


def _reference_roi_bbox(reference_polygons: list[PolygonData], image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    x_min = min(polygon.bbox[0] for polygon in reference_polygons)
    y_min = min(polygon.bbox[1] for polygon in reference_polygons)
    x_max = max(polygon.bbox[0] + polygon.bbox[2] for polygon in reference_polygons)
    y_max = max(polygon.bbox[1] + polygon.bbox[3] for polygon in reference_polygons)

    padding_x = max(12, int(round((x_max - x_min) * 0.18)))
    padding_y = max(12, int(round((y_max - y_min) * 0.18)))
    height, width = image_shape[:2]

    left = max(0, x_min - padding_x)
    top = max(0, y_min - padding_y)
    right = min(width, x_max + padding_x)
    bottom = min(height, y_max + padding_y)
    return left, top, max(1, right - left), max(1, bottom - top)


def _crop_image(image: np.ndarray, roi_bbox: tuple[int, int, int, int]) -> np.ndarray:
    x_coord, y_coord, width, height = roi_bbox
    return image[y_coord : y_coord + height, x_coord : x_coord + width].copy()


def _shift_polygons(polygons: list[PolygonData], dx: int, dy: int) -> list[PolygonData]:
    shifted: list[PolygonData] = []
    for polygon in polygons:
        clone = polygon.clone()
        clone.points = [(x_coord + dx, y_coord + dy) for x_coord, y_coord in clone.points]
        bbox_x, bbox_y, bbox_width, bbox_height = clone.bbox
        clone.bbox = (bbox_x + dx, bbox_y + dy, bbox_width, bbox_height)
        shifted.append(clone)
    return shifted


def _render_polygon_mask(image_shape: tuple[int, ...], polygons: list[PolygonData]) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in sorted(polygons, key=lambda item: item.is_hole):
        if len(polygon.points) < 3:
            continue
        points = np.asarray(
            [[int(round(x_coord)), int(round(y_coord))] for x_coord, y_coord in polygon.points],
            dtype=np.int32,
        )
        fill_value = 0 if polygon.is_hole else 255
        cv2.fillPoly(mask, [points.reshape((-1, 1, 2))], fill_value)
    return mask


def _candidate_block_sizes(image_shape: tuple[int, ...]) -> list[int]:
    min_dimension = max(3, min(image_shape[:2]))
    max_block_size = min_dimension if min_dimension % 2 == 1 else min_dimension - 1
    preferred = [11, 21, 31]
    valid = [size for size in preferred if 3 <= size <= max_block_size]
    return valid or [max(3, max_block_size)]


def _select_threshold_values(values: list[int], limit: int = 3) -> list[int]:
    if len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, num=limit, dtype=int)
    return [int(values[index]) for index in indices]


def _ensure_mask_shape(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    height, width = shape[:2]
    if mask.shape[:2] == (height, width):
        return mask
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def _mask_iou(first_mask: np.ndarray, second_mask: np.ndarray) -> float:
    first_binary = first_mask > 0
    second_binary = second_mask > 0
    union = np.logical_or(first_binary, second_binary).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(first_binary, second_binary).sum()
    return float(intersection / union)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError("RGB conversion expects a color image.")
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _sample_rows(values: np.ndarray, limit: int) -> np.ndarray:
    if len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, num=limit, dtype=int)
    return values[indices]


def _clamp_uint8(value: int) -> int:
    return int(max(0, min(255, value)))


def _step(operation_name: str, **parameters: Any):
    step = PreprocessingPipeline.create_step(operation_name)
    step.parameters.update(parameters)
    return step
