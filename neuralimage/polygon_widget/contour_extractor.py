from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import ContourExtractionSettings, PolygonData
from .utils import compute_polygon_metrics, ensure_binary_mask


RETRIEVAL_MODE_MAP = {
    "RETR_EXTERNAL": cv2.RETR_EXTERNAL,
    "RETR_CCOMP": cv2.RETR_CCOMP,
    "RETR_TREE": cv2.RETR_TREE,
}


APPROXIMATION_MODE_MAP = {
    "CHAIN_APPROX_SIMPLE": cv2.CHAIN_APPROX_SIMPLE,
    "CHAIN_APPROX_NONE": cv2.CHAIN_APPROX_NONE,
}


@dataclass(slots=True)
class _IntermediateContour:
    contour_index: int
    parent_contour_index: int | None
    points: list[tuple[float, float]]
    is_hole: bool
    area: float
    perimeter: float
    bbox: tuple[int, int, int, int]


def _depth(index: int, hierarchy: np.ndarray, cache: dict[int, int]) -> int:
    if index in cache:
        return cache[index]
    parent_index = int(hierarchy[index][3])
    if parent_index == -1:
        cache[index] = 0
        return 0
    cache[index] = 1 + _depth(parent_index, hierarchy, cache)
    return cache[index]


def extract_polygons(mask: np.ndarray, settings: ContourExtractionSettings | None = None) -> list[PolygonData]:
    config = settings or ContourExtractionSettings()
    binary_mask = ensure_binary_mask(mask)
    contours, hierarchy = cv2.findContours(
        binary_mask.copy(),
        RETRIEVAL_MODE_MAP.get(config.retrieval_mode, cv2.RETR_EXTERNAL),
        APPROXIMATION_MODE_MAP.get(config.approximation_mode, cv2.CHAIN_APPROX_SIMPLE),
    )
    if not contours:
        return []

    hierarchy_array = hierarchy[0] if hierarchy is not None else np.full((len(contours), 4), -1, dtype=np.int32)
    depth_cache: dict[int, int] = {}
    kept: list[_IntermediateContour] = []

    for contour_index, contour in enumerate(contours):
        if contour is None or len(contour) < 3:
            continue
        epsilon = float(config.epsilon)
        if config.epsilon_relative:
            epsilon *= cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True) if epsilon > 0 else contour
        points = [(float(point[0][0]), float(point[0][1])) for point in approx]
        if len(points) < max(3, config.min_points):
            continue

        area, perimeter, bbox = compute_polygon_metrics(points)
        if area <= 0.0 or perimeter <= 0.0:
            continue
        if area < config.min_area:
            continue
        if config.max_area is not None and area > config.max_area:
            continue
        if perimeter < config.min_perimeter:
            continue

        parent_index = int(hierarchy_array[contour_index][3])
        depth = _depth(contour_index, hierarchy_array, depth_cache)
        kept.append(
            _IntermediateContour(
                contour_index=contour_index,
                parent_contour_index=None if parent_index == -1 else parent_index,
                points=points,
                is_hole=bool(depth % 2),
                area=area,
                perimeter=perimeter,
                bbox=bbox,
            )
        )

    contour_id_to_polygon_id: dict[int, int] = {}
    polygons: list[PolygonData] = []
    for polygon_id, intermediate in enumerate(kept, start=1):
        contour_id_to_polygon_id[intermediate.contour_index] = polygon_id
        polygons.append(
            PolygonData(
                id=polygon_id,
                points=intermediate.points,
                is_hole=intermediate.is_hole,
                parent_id=None,
                area=intermediate.area,
                perimeter=intermediate.perimeter,
                bbox=intermediate.bbox,
            )
        )

    for polygon, intermediate in zip(polygons, kept, strict=False):
        if intermediate.parent_contour_index is not None:
            polygon.parent_id = contour_id_to_polygon_id.get(intermediate.parent_contour_index)

    return polygons
