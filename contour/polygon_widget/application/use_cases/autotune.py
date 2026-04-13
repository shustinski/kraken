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
    if image.ndim != 2:
        raise ValueError("Auto tune expects a grayscale image.")
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
    contour_candidates = _build_contour_candidates(local_reference_polygons)

    best_result: AutoTuneResult | None = None
    evaluations = 0
    reference_count = max(1, sum(1 for polygon in local_reference_polygons if not polygon.is_hole))

    for mask_candidate in best_mask_candidates:
        mask = mask_candidate["mask"]
        mask_score = float(mask_candidate["score"])
        for contour_settings in contour_candidates:
            polygons = extract_polygons(mask, contour_settings)
            polygon_mask = _render_polygon_mask(crop.shape, polygons)
            polygon_score = _mask_iou(target_mask, polygon_mask)
            predicted_count = max(1, sum(1 for polygon in polygons if not polygon.is_hole))
            count_penalty = 1.0 / (1.0 + abs(predicted_count - reference_count) * 0.3)
            final_score = (polygon_score * 0.8 + mask_score * 0.2) * count_penalty
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
    target_pixels = image[target_mask > 0]
    background_pixels = image[target_mask == 0]
    if target_pixels.size == 0:
        raise ValueError("Reference mask is empty.")
    if background_pixels.size == 0:
        background_pixels = image.reshape(-1)

    inside_mean = float(np.mean(target_pixels))
    outside_mean = float(np.mean(background_pixels))
    object_is_brighter = inside_mean >= outside_mean

    threshold_values = {
        _clamp_uint8(round((inside_mean + outside_mean) / 2.0)),
        _clamp_uint8(round(np.percentile(target_pixels, 50))),
        _clamp_uint8(round(np.percentile(background_pixels, 50))),
        _clamp_uint8(round((np.percentile(target_pixels, 25) + np.percentile(background_pixels, 75)) / 2.0)),
        _clamp_uint8(round((np.percentile(target_pixels, 75) + np.percentile(background_pixels, 25)) / 2.0)),
    }

    return {
        "object_is_brighter": object_is_brighter,
        "threshold_values": sorted(threshold_values),
    }


def _build_pipeline_candidates(image_shape: tuple[int, int], stats: dict[str, Any]) -> list[dict[str, Any]]:
    preferred_threshold_type = "binary" if stats["object_is_brighter"] else "binary_inv"
    threshold_types = [preferred_threshold_type]
    fallback_threshold_type = "binary_inv" if preferred_threshold_type == "binary" else "binary"
    threshold_types.append(fallback_threshold_type)

    filter_candidates = [
        [],
        [_step("gaussian_blur", kernel_size=3, sigma_x=0.0)],
        [_step("gaussian_blur", kernel_size=5, sigma_x=0.0)],
        [_step("median_blur", kernel_size=3)],
        [_step("median_blur", kernel_size=5)],
        [_step("clahe", clip_limit=1.5, tile_grid_size=8)],
        [_step("clahe", clip_limit=3.0, tile_grid_size=8)],
        [_step("gamma_correction", gamma=0.8)],
        [_step("gamma_correction", gamma=1.2)],
        [_step("brightness_contrast", alpha=0.85, beta=0.0)],
        [_step("brightness_contrast", alpha=1.15, beta=0.0)],
    ]

    block_sizes = _candidate_block_sizes(image_shape)
    binarizer_candidates: list[list] = []
    for threshold_type in threshold_types:
        binarizer_candidates.append([_step("otsu_threshold", threshold_type=threshold_type)])
        for threshold_value in stats["threshold_values"]:
            binarizer_candidates.append(
                [_step("threshold", threshold=threshold_value, max_value=255.0, threshold_type=threshold_type)]
            )
        for block_size in block_sizes:
            for method in ("gaussian", "mean"):
                for c_value in (-4.0, 0.0, 4.0):
                    binarizer_candidates.append(
                        [
                            _step(
                                "adaptive_threshold",
                                max_value=255.0,
                                adaptive_method=method,
                                threshold_type=threshold_type,
                                block_size=block_size,
                                c_value=c_value,
                            )
                        ]
                    )

    morph_candidates = [
        [],
        [_step("morph_open", kernel_size=3, iterations=1, shape="rect")],
        [_step("morph_close", kernel_size=3, iterations=1, shape="rect")],
        [_step("morph_open", kernel_size=5, iterations=1, shape="ellipse")],
        [_step("morph_close", kernel_size=5, iterations=1, shape="ellipse")],
        [_step("erode", kernel_size=3, iterations=1, shape="rect")],
        [_step("dilate", kernel_size=3, iterations=1, shape="rect")],
        [
            _step("morph_open", kernel_size=3, iterations=1, shape="rect"),
            _step("morph_close", kernel_size=3, iterations=1, shape="rect"),
        ],
    ]

    unique_candidates: dict[str, dict[str, Any]] = {}
    for filter_steps in filter_candidates:
        for binarizer_steps in binarizer_candidates:
            base_steps = [step.clone() for step in filter_steps + binarizer_steps]
            for morph_steps in morph_candidates:
                steps = [step.clone() for step in base_steps]
                steps.extend(step.clone() for step in morph_steps)
                config = PreprocessingPipeline(steps).to_dict()
                signature = json.dumps(config, ensure_ascii=False, sort_keys=True)
                unique_candidates.setdefault(signature, config)
    return list(unique_candidates.values())


def _rank_mask_candidates(
    crop: np.ndarray,
    target_mask: np.ndarray,
    pipeline_candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    ranked: list[dict[str, Any]] = []
    for config in pipeline_candidates:
        mask = ensure_binary_mask(PreprocessingPipeline.from_dict(config).apply(crop))
        score = _mask_iou(target_mask, mask)
        ranked.append({"config": config, "mask": mask, "score": score})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:8], len(ranked)


def _build_contour_candidates(reference_polygons: list[PolygonData]) -> list[ContourExtractionSettings]:
    areas = [polygon.area for polygon in reference_polygons if not polygon.is_hole and polygon.area > 0.0]
    perimeters = [polygon.perimeter for polygon in reference_polygons if not polygon.is_hole and polygon.perimeter > 0.0]
    min_area_candidates = [0.0]
    min_perimeter_candidates = [0.0]
    if areas:
        min_area_candidates.append(max(1.0, min(areas) * 0.2))
    if perimeters:
        min_perimeter_candidates.append(max(1.0, min(perimeters) * 0.2))

    retrieval_modes = ["RETR_EXTERNAL", "RETR_CCOMP"]
    if any(polygon.is_hole for polygon in reference_polygons):
        retrieval_modes.append("RETR_TREE")

    unique_candidates: dict[str, ContourExtractionSettings] = {}
    for retrieval_mode in retrieval_modes:
        for approximation_mode in ("CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_NONE"):
            for epsilon in (0.0, 1.0, 2.0, 4.0):
                for min_area in min_area_candidates:
                    for min_perimeter in min_perimeter_candidates:
                        candidate = ContourExtractionSettings(
                            retrieval_mode=retrieval_mode,
                            approximation_mode=approximation_mode,
                            epsilon=epsilon,
                            epsilon_relative=False,
                            min_area=min_area,
                            max_area=None,
                            min_perimeter=min_perimeter,
                            min_points=3,
                        )
                        signature = json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True)
                        unique_candidates.setdefault(signature, candidate)
            for epsilon in (0.001, 0.003):
                candidate = ContourExtractionSettings(
                    retrieval_mode=retrieval_mode,
                    approximation_mode=approximation_mode,
                    epsilon=epsilon,
                    epsilon_relative=True,
                    min_area=min_area_candidates[-1],
                    max_area=None,
                    min_perimeter=min_perimeter_candidates[-1],
                    min_points=3,
                )
                signature = json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True)
                unique_candidates.setdefault(signature, candidate)
    return list(unique_candidates.values())


def _reference_roi_bbox(reference_polygons: list[PolygonData], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x_min = min(polygon.bbox[0] for polygon in reference_polygons)
    y_min = min(polygon.bbox[1] for polygon in reference_polygons)
    x_max = max(polygon.bbox[0] + polygon.bbox[2] for polygon in reference_polygons)
    y_max = max(polygon.bbox[1] + polygon.bbox[3] for polygon in reference_polygons)

    padding_x = max(12, int(round((x_max - x_min) * 0.15)))
    padding_y = max(12, int(round((y_max - y_min) * 0.15)))
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


def _render_polygon_mask(image_shape: tuple[int, int], polygons: list[PolygonData]) -> np.ndarray:
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


def _candidate_block_sizes(image_shape: tuple[int, int]) -> list[int]:
    min_dimension = max(3, min(image_shape[:2]))
    max_block_size = min_dimension if min_dimension % 2 == 1 else min_dimension - 1
    preferred = [11, 21, 31]
    valid = [size for size in preferred if 3 <= size <= max_block_size]
    if valid:
        return valid
    return [max(3, max_block_size)]


def _mask_iou(first_mask: np.ndarray, second_mask: np.ndarray) -> float:
    first_binary = first_mask > 0
    second_binary = second_mask > 0
    union = np.logical_or(first_binary, second_binary).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(first_binary, second_binary).sum()
    return float(intersection / union)


def _clamp_uint8(value: int) -> int:
    return int(max(0, min(255, value)))


def _step(operation_name: str, **parameters: Any):
    step = PreprocessingPipeline.create_step(operation_name)
    step.parameters.update(parameters)
    return step
