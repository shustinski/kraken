from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, pi

import cv2
import numpy as np

from .application.processing import ContourDebugCandidate, ContourExtractionSettings
from .domain import PolygonData, compute_polygon_metrics
from .utils import ensure_binary_mask


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
    depth: int
    area: float
    perimeter: float
    bbox: tuple[int, int, int, int]
    solidity: float
    extent: float
    output_bbox: tuple[int, int, int, int] | None = None


def _depth(index: int, hierarchy: np.ndarray, cache: dict[int, int]) -> int:
    if index in cache:
        return cache[index]
    parent_index = int(hierarchy[index][3])
    if parent_index == -1:
        cache[index] = 0
        return 0
    cache[index] = 1 + _depth(parent_index, hierarchy, cache)
    return cache[index]


def _bbox_box_points(bbox: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    x_coord, y_coord, width, height = bbox
    left = float(x_coord)
    top = float(y_coord)
    right = float(x_coord + max(1, width))
    bottom = float(y_coord + max(1, height))
    return [
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
    ]


def _bboxes_overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    first_right = first[0] + first[2]
    first_bottom = first[1] + first[3]
    second_right = second[0] + second[2]
    second_bottom = second[1] + second[3]
    return first[0] < second_right and first_right > second[0] and first[1] < second_bottom and first_bottom > second[1]


def _suppress_overlapping_vias(polygons: list[PolygonData]) -> list[PolygonData]:
    kept: list[PolygonData] = []
    for polygon in sorted(polygons, key=lambda item: (item.area, item.perimeter, item.bbox[0], item.bbox[1])):
        if any(_bboxes_overlap(polygon.bbox, accepted.bbox) for accepted in kept):
            continue
        kept.append(polygon)
    reindexed: list[PolygonData] = []
    for index, polygon in enumerate(kept, start=1):
        clone = polygon.clone()
        clone.id = index
        clone.parent_id = None
        reindexed.append(clone)
    return reindexed


def _centered_bbox(
    bbox: tuple[int, int, int, int],
    target_width: int,
    target_height: int,
) -> tuple[int, int, int, int]:
    x_coord, y_coord, width, height = bbox
    center_x = x_coord + width / 2.0
    center_y = y_coord + height / 2.0
    left = int(round(center_x - target_width / 2.0))
    top = int(round(center_y - target_height / 2.0))
    return (left, top, max(1, int(target_width)), max(1, int(target_height)))


def _size_matches_with_tolerance(actual: int, target: int) -> bool:
    tolerance = max(2, int(round(target * 0.25)))
    return abs(actual - target) <= tolerance


def _aspect_matches_with_tolerance(
    bbox_width: int,
    bbox_height: int,
    target_width: int,
    target_height: int,
) -> bool:
    actual_aspect = bbox_width / max(1, bbox_height)
    target_aspect = target_width / max(1, target_height)
    return abs(actual_aspect - target_aspect) / max(target_aspect, 1e-6) <= 0.15


def _matched_fixed_via_bbox(
    bbox: tuple[int, int, int, int],
    config: ContourExtractionSettings,
) -> tuple[int, int, int, int] | None:
    size_pairs = list(zip(config.fixed_via_widths, config.fixed_via_heights, strict=False))
    if not size_pairs:
        return bbox if not config.fixed_via_widths and not config.fixed_via_heights else None

    bbox_width = int(bbox[2])
    bbox_height = int(bbox[3])
    best_match: tuple[float, int, int] | None = None
    for target_width, target_height in size_pairs:
        if (
            _size_matches_with_tolerance(bbox_width, target_width)
            and _size_matches_with_tolerance(bbox_height, target_height)
            and _aspect_matches_with_tolerance(bbox_width, bbox_height, target_width, target_height)
        ):
            score = (abs(bbox_width - target_width) / max(1, target_width)) + (
                abs(bbox_height - target_height) / max(1, target_height)
            )
            if best_match is None or score < best_match[0]:
                best_match = (score, target_width, target_height)
    if best_match is None:
        return None
    return _centered_bbox(bbox, best_match[1], best_match[2])


def _matches_via_size_constraints(bbox_width: int, bbox_height: int, config: ContourExtractionSettings) -> bool:
    if config.via_size_mode == "fixed":
        return _matched_fixed_via_bbox((0, 0, bbox_width, bbox_height), config) is not None

    if bbox_width < config.min_via_width:
        return False
    if config.max_via_width is not None and bbox_width > config.max_via_width:
        return False
    if bbox_height < config.min_via_height:
        return False
    if config.max_via_height is not None and bbox_height > config.max_via_height:
        return False
    return True


def _via_roundness_score(contour: np.ndarray, bbox_width: int, bbox_height: int) -> float:
    contour_area = abs(float(cv2.contourArea(contour)))
    contour_perimeter = float(cv2.arcLength(contour, True))
    if contour_area <= 0.0 or contour_perimeter <= 0.0:
        return 0.0
    circularity = 100.0 * (4.0 * pi * contour_area / max(1e-6, contour_perimeter * contour_perimeter))
    aspect_roundness = 100.0 * min(bbox_width, bbox_height) / max(1, max(bbox_width, bbox_height))
    return max(0.0, min(100.0, circularity, aspect_roundness))


def _bbox_center_inside(inner_bbox: tuple[int, int, int, int], outer_bbox: tuple[int, int, int, int]) -> bool:
    center_x = inner_bbox[0] + inner_bbox[2] / 2.0
    center_y = inner_bbox[1] + inner_bbox[3] / 2.0
    return (
        outer_bbox[0] <= center_x <= outer_bbox[0] + outer_bbox[2]
        and outer_bbox[1] <= center_y <= outer_bbox[1] + outer_bbox[3]
    )


def _debug_candidates_for_mask(mask: np.ndarray, config: ContourExtractionSettings, polygons: list[PolygonData]) -> list[ContourDebugCandidate]:
    binary_mask = ensure_binary_mask(mask)
    image_height, image_width = binary_mask.shape[:2]
    contours, hierarchy = cv2.findContours(
        binary_mask.copy(),
        RETRIEVAL_MODE_MAP.get(config.retrieval_mode, cv2.RETR_EXTERNAL),
        APPROXIMATION_MODE_MAP.get(config.approximation_mode, cv2.CHAIN_APPROX_SIMPLE),
    )
    if not contours:
        return []

    hierarchy_array = hierarchy[0] if hierarchy is not None else np.full((len(contours), 4), -1, dtype=np.int32)
    depth_cache: dict[int, int] = {}
    contour_areas: dict[int, float] = {
        index: abs(float(cv2.contourArea(contour))) for index, contour in enumerate(contours) if contour is not None
    }
    accepted_bboxes = [polygon.bbox for polygon in polygons]
    candidates: list[ContourDebugCandidate] = []

    for contour_index, contour in enumerate(contours):
        if contour is None or len(contour) < 3:
            continue
        epsilon = float(config.epsilon)
        if config.epsilon_relative:
            epsilon *= cv2.arcLength(contour, True)
        approx = _adaptive_approximate_contour(contour, epsilon, config.preserve_corners) if epsilon > 0 else contour
        points = [(float(point[0][0]), float(point[0][1])) for point in approx]
        if len(points) < 3:
            continue

        area, perimeter, bbox = compute_polygon_metrics(points)
        bbox_width = int(bbox[2])
        bbox_height = int(bbox[3])
        roundness = _via_roundness_score(contour, bbox_width, bbox_height) if config.object_type == "via" or config.output_mode == "box" else 0.0
        candidate = ContourDebugCandidate(
            contour_index=contour_index,
            bbox=bbox,
            area=area,
            perimeter=perimeter,
            roundness=roundness,
        )

        reason = ""
        if len(points) < max(3, config.min_points):
            reason = "min_points"
        elif area <= 0.0 or perimeter <= 0.0:
            reason = "empty_geometry"
        elif area < config.min_area:
            reason = "min_area"
        elif config.max_area is not None and area > config.max_area:
            reason = "max_area"
        elif perimeter < config.min_perimeter:
            reason = "min_perimeter"
        elif config.max_perimeter is not None and perimeter > config.max_perimeter:
            reason = "max_perimeter"
        elif bbox_width < config.min_bbox_width:
            reason = "min_bbox_width"
        elif config.max_bbox_width is not None and bbox_width > config.max_bbox_width:
            reason = "max_bbox_width"
        elif bbox_height < config.min_bbox_height:
            reason = "min_bbox_height"
        elif config.max_bbox_height is not None and bbox_height > config.max_bbox_height:
            reason = "max_bbox_height"
        elif (config.object_type == "via" or config.output_mode == "box") and not _matches_via_size_constraints(bbox_width, bbox_height, config):
            reason = "via_size"
        elif (config.object_type == "via" or config.output_mode == "box") and roundness < config.via_min_roundness:
            reason = "roundness"
        else:
            aspect_ratio = float(bbox_width / max(1, bbox_height))
            if aspect_ratio < config.min_aspect_ratio:
                reason = "min_aspect_ratio"
            elif config.max_aspect_ratio is not None and aspect_ratio > config.max_aspect_ratio:
                reason = "max_aspect_ratio"
            elif config.exclude_border_touching and (
                bbox[0] <= 0
                or bbox[1] <= 0
                or bbox[0] + bbox_width >= image_width
                or bbox[1] + bbox_height >= image_height
            ):
                reason = "border"
            else:
                parent_index = int(hierarchy_array[contour_index][3])
                depth = _depth(contour_index, hierarchy_array, depth_cache)
                if depth < config.min_hierarchy_depth:
                    reason = "min_hierarchy_depth"
                elif config.max_hierarchy_depth is not None and depth > config.max_hierarchy_depth:
                    reason = "max_hierarchy_depth"
                else:
                    hull = cv2.convexHull(approx)
                    hull_area = abs(float(cv2.contourArea(hull))) if hull is not None and len(hull) >= 3 else 0.0
                    solidity = float(area / hull_area) if hull_area > 0.0 else 0.0
                    bbox_area = float(max(1, bbox_width * bbox_height))
                    extent = float(area / bbox_area)
                    is_hole = bool(depth % 2)
                    if solidity < config.min_solidity:
                        reason = "min_solidity"
                    elif extent < config.min_extent:
                        reason = "min_extent"
                    elif config.max_hole_area_ratio is not None and is_hole and parent_index != -1:
                        parent_area = contour_areas.get(parent_index, 0.0)
                        if parent_area > 0.0 and (area / parent_area) > config.max_hole_area_ratio:
                            reason = "max_hole_area_ratio"

        if reason:
            candidate.accepted = False
            candidate.reason = reason
        else:
            candidate.accepted = any(_bboxes_overlap(candidate.bbox, bbox) or _bbox_center_inside(candidate.bbox, bbox) for bbox in accepted_bboxes)
            candidate.reason = "accepted" if candidate.accepted else "overlap_suppressed"
        candidates.append(candidate)

    return candidates


def _nearest_contour_index(points: np.ndarray, target: np.ndarray) -> int:
    deltas = points - target
    distances = np.sum(deltas * deltas, axis=1)
    return int(np.argmin(distances))


def _corner_indices(points: np.ndarray, *, step: int = 2, max_angle: float = 145.0) -> set[int]:
    if len(points) < 5:
        return set()
    result: set[int] = set()
    total = len(points)
    for index in range(total):
        prev_point = points[(index - step) % total]
        current_point = points[index]
        next_point = points[(index + step) % total]
        first = prev_point - current_point
        second = next_point - current_point
        first_norm = float(np.linalg.norm(first))
        second_norm = float(np.linalg.norm(second))
        if first_norm < 1e-6 or second_norm < 1e-6:
            continue
        cosine = float(np.dot(first, second) / (first_norm * second_norm))
        cosine = max(-1.0, min(1.0, cosine))
        angle = degrees(acos(cosine))
        if angle <= max_angle:
            result.add(index)
    return result


def _adaptive_approximate_contour(contour: np.ndarray, epsilon: float, preserve_corners: bool) -> np.ndarray:
    if epsilon <= 0.0 or contour is None or len(contour) < 3:
        return contour

    simplified = cv2.approxPolyDP(contour, epsilon, True)
    if simplified is None or len(simplified) < 3:
        return contour
    if not preserve_corners or len(contour) < 5:
        return simplified

    contour_points = contour.reshape((-1, 2)).astype(np.float32, copy=False)
    simplified_points = simplified.reshape((-1, 2)).astype(np.float32, copy=False)
    selected_indices = {_nearest_contour_index(contour_points, point) for point in simplified_points}

    angle_limit = 150.0 if epsilon <= 1.5 else 140.0 if epsilon <= 3.0 else 130.0
    for corner_index in _corner_indices(contour_points, max_angle=angle_limit):
        corner_point = contour_points[corner_index]
        if any(np.linalg.norm(corner_point - simplified_point) <= max(1.5, epsilon * 1.2) for simplified_point in simplified_points):
            continue
        selected_indices.add(corner_index)

    ordered_indices = sorted(selected_indices)
    if len(ordered_indices) < 3 or len(ordered_indices) <= len(simplified):
        return simplified
    return contour[ordered_indices]


def extract_polygons(mask: np.ndarray, settings: ContourExtractionSettings | None = None) -> list[PolygonData]:
    config = settings or ContourExtractionSettings()
    binary_mask = ensure_binary_mask(mask)
    image_height, image_width = binary_mask.shape[:2]
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
    contour_areas: dict[int, float] = {}

    for contour_index, contour in enumerate(contours):
        if contour is None or len(contour) < 3:
            continue
        contour_areas[contour_index] = abs(float(cv2.contourArea(contour)))
        epsilon = float(config.epsilon)
        if config.epsilon_relative:
            epsilon *= cv2.arcLength(contour, True)
        approx = _adaptive_approximate_contour(contour, epsilon, config.preserve_corners) if epsilon > 0 else contour
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
        if config.max_perimeter is not None and perimeter > config.max_perimeter:
            continue

        bbox_width = int(bbox[2])
        bbox_height = int(bbox[3])
        if bbox_width < config.min_bbox_width:
            continue
        if config.max_bbox_width is not None and bbox_width > config.max_bbox_width:
            continue
        if bbox_height < config.min_bbox_height:
            continue
        if config.max_bbox_height is not None and bbox_height > config.max_bbox_height:
            continue

        output_bbox: tuple[int, int, int, int] | None = None
        if config.object_type == "via" or config.output_mode == "box":
            if not _matches_via_size_constraints(bbox_width, bbox_height, config):
                continue
            if _via_roundness_score(contour, bbox_width, bbox_height) < config.via_min_roundness:
                continue
            if config.via_size_mode == "fixed":
                output_bbox = _matched_fixed_via_bbox(bbox, config)
                if output_bbox is None:
                    continue

        aspect_ratio = float(bbox_width / max(1, bbox_height))
        if aspect_ratio < config.min_aspect_ratio:
            continue
        if config.max_aspect_ratio is not None and aspect_ratio > config.max_aspect_ratio:
            continue

        if config.exclude_border_touching:
            touches_border = (
                bbox[0] <= 0
                or bbox[1] <= 0
                or bbox[0] + bbox_width >= image_width
                or bbox[1] + bbox_height >= image_height
            )
            if touches_border:
                continue

        parent_index = int(hierarchy_array[contour_index][3])
        depth = _depth(contour_index, hierarchy_array, depth_cache)
        if depth < config.min_hierarchy_depth:
            continue
        if config.max_hierarchy_depth is not None and depth > config.max_hierarchy_depth:
            continue

        hull = cv2.convexHull(approx)
        hull_area = abs(float(cv2.contourArea(hull))) if hull is not None and len(hull) >= 3 else 0.0
        solidity = float(area / hull_area) if hull_area > 0.0 else 0.0
        if solidity < config.min_solidity:
            continue

        bbox_area = float(max(1, bbox_width * bbox_height))
        extent = float(area / bbox_area)
        if extent < config.min_extent:
            continue

        is_hole = bool(depth % 2)
        if config.max_hole_area_ratio is not None and is_hole and parent_index != -1:
            parent_area = contour_areas.get(parent_index, 0.0)
            if parent_area > 0.0 and (area / parent_area) > config.max_hole_area_ratio:
                continue

        kept.append(
            _IntermediateContour(
                contour_index=contour_index,
                parent_contour_index=None if parent_index == -1 else parent_index,
                points=points,
                is_hole=is_hole,
                depth=depth,
                area=area,
                perimeter=perimeter,
                bbox=bbox,
                solidity=solidity,
                extent=extent,
                output_bbox=output_bbox,
            )
        )

    contour_id_to_polygon_id: dict[int, int] = {}
    polygons: list[PolygonData] = []
    for polygon_id, intermediate in enumerate(kept, start=1):
        contour_id_to_polygon_id[intermediate.contour_index] = polygon_id
        polygon_points = intermediate.points
        shape_hint = "polygon"
        category = "conductor"
        is_hole = intermediate.is_hole
        parent_id: int | None = None
        polygon_area = intermediate.area
        polygon_perimeter = intermediate.perimeter
        polygon_bbox = intermediate.bbox
        if config.object_type == "via" or config.output_mode == "box":
            polygon_points = _bbox_box_points(intermediate.output_bbox or intermediate.bbox)
            polygon_area, polygon_perimeter, polygon_bbox = compute_polygon_metrics(polygon_points)
            shape_hint = "box"
            category = "via"
            is_hole = False
        polygons.append(
            PolygonData(
                id=polygon_id,
                points=polygon_points,
                is_hole=is_hole,
                parent_id=parent_id,
                category=category,
                shape_hint=shape_hint,
                area=polygon_area,
                perimeter=polygon_perimeter,
                bbox=polygon_bbox,
            )
        )

    for polygon, intermediate in zip(polygons, kept, strict=False):
        if polygon.category != "via" and intermediate.parent_contour_index is not None:
            polygon.parent_id = contour_id_to_polygon_id.get(intermediate.parent_contour_index)

    if config.object_type == "via" or config.output_mode == "box":
        polygons = _suppress_overlapping_vias(polygons)

    return polygons


def extract_polygons_with_debug(
    mask: np.ndarray,
    settings: ContourExtractionSettings | None = None,
) -> tuple[list[PolygonData], list[ContourDebugCandidate]]:
    config = settings or ContourExtractionSettings()
    polygons = extract_polygons(mask, config)
    return polygons, _debug_candidates_for_mask(mask, config, polygons)
