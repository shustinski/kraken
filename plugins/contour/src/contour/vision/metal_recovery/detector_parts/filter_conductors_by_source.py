"""detector: filter conductors by source."""
from __future__ import annotations
import cv2
import numpy as np
from contour.domain import PolygonData
from contour.utils import ensure_uint8

def _filter_conductors_by_source_contrast(
    polygons: list[PolygonData],
    source_image: np.ndarray | None,
    *,
    min_contrast: float,
    min_rim_contrast: float,
    min_rim_area_fraction: float,
) -> list[PolygonData]:
    """Reject dark shadow islands that watershed can grow beside a conductor."""
    if source_image is None or min_contrast <= 0.0 or not polygons:
        return polygons
    if source_image.ndim == 3:
        source_gray = cv2.cvtColor(ensure_uint8(source_image), cv2.COLOR_BGR2GRAY)
    else:
        source_gray = ensure_uint8(source_image)

    by_parent: dict[int, list[PolygonData]] = {}
    for polygon in polygons:
        if polygon.is_hole and polygon.parent_id is not None:
            by_parent.setdefault(int(polygon.parent_id), []).append(polygon)

    kept_parent_ids: set[int] = set()
    ring_radius = 10
    min_rim_backed_area_fraction = max(0.000001, min(1.0, float(min_rim_area_fraction)))
    required_rim_contrast = max(0.0, float(min_rim_contrast))
    rim_percentile = 90.0
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * ring_radius + 1, 2 * ring_radius + 1),
    )
    for polygon in polygons:
        if polygon.is_hole:
            continue
        points = np.asarray(polygon.points, dtype=np.int32).reshape(-1, 2)
        x_min = max(0, int(points[:, 0].min()) - ring_radius)
        x_max = min(source_gray.shape[1] - 1, int(points[:, 0].max()) + ring_radius)
        y_min = max(0, int(points[:, 1].min()) - ring_radius)
        y_max = min(source_gray.shape[0] - 1, int(points[:, 1].max()) + ring_radius)
        local_source = source_gray[y_min : y_max + 1, x_min : x_max + 1]

        filled = np.zeros(local_source.shape, dtype=np.uint8)
        local_polygon_points = points.copy()
        local_polygon_points[:, 0] -= x_min
        local_polygon_points[:, 1] -= y_min
        cv2.fillPoly(filled, [local_polygon_points.reshape(-1, 1, 2)], 255)
        interior = filled.copy()
        for hole in by_parent.get(int(polygon.id), ()):
            local_hole_points = np.asarray(hole.points, dtype=np.int32).reshape(-1, 2).copy()
            local_hole_points[:, 0] -= x_min
            local_hole_points[:, 1] -= y_min
            cv2.fillPoly(interior, [local_hole_points.reshape(-1, 1, 2)], 0)
        inside_values = local_source[interior > 0]
        ring = (cv2.dilate(filled, ring_kernel) > 0) & (filled == 0)
        outside_values = local_source[ring]
        if inside_values.size == 0 or outside_values.size == 0:
            kept_parent_ids.add(int(polygon.id))
            continue
        inside_median = float(np.median(inside_values))
        outside_median = float(np.median(outside_values))
        interior_contrast = inside_median - outside_median
        area_fraction = float(inside_values.size / max(1, source_gray.size))
        inner_edge = (interior > 0) & (
            cv2.erode(interior, np.ones((3, 3), dtype=np.uint8)) == 0
        )
        near_ring = (cv2.dilate(filled, np.ones((3, 3), dtype=np.uint8)) > 0) & (
            filled == 0
        )
        edge_values = local_source[inner_edge]
        near_values = local_source[near_ring]
        rim_contrast = (
            float(np.percentile(edge_values, rim_percentile) - np.median(near_values))
            if edge_values.size > 0 and near_values.size > 0
            else 0.0
        )

        if interior_contrast >= float(min_contrast):
            is_sizeable_component = area_fraction >= min_rim_backed_area_fraction
            has_strong_interior = interior_contrast >= 2.0 * float(min_contrast)
            has_strong_rim = rim_contrast >= required_rim_contrast
            if is_sizeable_component or has_strong_interior or has_strong_rim:
                kept_parent_ids.add(int(polygon.id))
            continue

        # Wide SEM conductors can have a substrate-like interior while their
        # bright material rim remains unambiguous.  Use that evidence only for
        # dominant candidates so textured background islands are not promoted.
        is_large_rim_candidate = (
            area_fraction >= 6.0 * min_rim_backed_area_fraction
            and interior_contrast >= 0.75 * float(min_contrast)
            and rim_contrast >= 0.75 * required_rim_contrast
        )
        is_dominant_component = area_fraction >= 50.0 * min_rim_backed_area_fraction
        if is_large_rim_candidate or (
            rim_contrast >= (2.0 / 3.0) * required_rim_contrast and is_dominant_component
        ):
            kept_parent_ids.add(int(polygon.id))

    return [
        polygon
        for polygon in polygons
        if (
            int(polygon.parent_id) in kept_parent_ids
            if polygon.is_hole and polygon.parent_id is not None
            else not polygon.is_hole and int(polygon.id) in kept_parent_ids
        )
    ]
