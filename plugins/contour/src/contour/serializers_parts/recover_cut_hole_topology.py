"""serializers: recover cut hole topology."""
from __future__ import annotations
import cv2
import numpy as np
from contour.domain import PolygonData, compute_polygon_metrics, integer_points

from contour.serializers_parts.boundary_vertices_by_distance import (
    _dedupe_closed_points,
    _has_duplicate_points,
    _to_local_int_points,
)

from contour.serializers_parts.cif_paint_mask_ring import (
    _legacy_cut_hole_recovery_enabled,
    _nested_outer_punch_on_load_enabled,
)

from contour.serializers_parts.contour_points_polygon import _contour_points_to_polygon

from contour.serializers_parts.normalize_cif_polygon_families import _normalize_cif_polygon_families

from contour.serializers_parts.punch_nested_outer_covers import _punch_nested_outer_covers

from contour.serializers_parts.split_linked_polygon_rings import _split_linked_polygon_rings

def _recover_cut_hole_topology(
    polygons: list[PolygonData],
    image_size: tuple[int, int] | None,
    *,
    cutout_display: bool = False,
) -> list[PolygonData]:
    from contour.serializers_parts.cif_box_polygon import _clone_polygon_with_id
    from ..application.fix_internal_contours import should_use_cutout_display_for_keyhole_family

    if image_size is None:
        return polygons
    image_width, image_height = image_size
    del image_height
    recovered: list[PolygonData] = []
    next_id = 1
    recover_legacy_cut_holes = _legacy_cut_hole_recovery_enabled()
    for polygon in polygons:
        if polygon.shape_hint == "box" or polygon.category == "via" or len(polygon.points) < 3:
            recovered.append(_clone_polygon_with_id(polygon, next_id))
            next_id += 1
            continue
        open_points = _dedupe_closed_points(polygon.points)
        if not _has_duplicate_points(open_points):
            recovered.append(_clone_polygon_with_id(polygon, next_id))
            next_id += 1
            continue

        families = _normalize_cif_polygon_families(polygon.points)
        linked_outer, linked_holes = _split_linked_polygon_rings(polygon.points)
        # Prefer make_valid when it yields one outer with holes (KLayout-like fill).
        # Fall back to keyhole split when make_valid shatters a linked bridge into
        # separate solids and loses hole topology (right-edge multi-hole CIF).
        if families and len(families) == 1 and families[0][1]:
            pass
        elif linked_holes:
            families = [(linked_outer, linked_holes)]
        elif not families:
            families = []

        if families:
            authored_paint_ring = integer_points(polygon.points)
            for outer_points, hole_rings in families:
                use_cutout_display = bool(cutout_display) or should_use_cutout_display_for_keyhole_family(
                    authored_paint_ring,
                    outer_points,
                    hole_rings,
                    image_size,
                )
                paint_ring = integer_points(outer_points) if use_cutout_display else authored_paint_ring
                area, perimeter, bbox = compute_polygon_metrics(outer_points)
                parent_id = next_id
                recovered.append(
                    PolygonData(
                        id=parent_id,
                        points=outer_points,
                        is_hole=False,
                        parent_id=None,
                        category=polygon.category,
                        shape_hint=polygon.shape_hint,
                        area=area,
                        perimeter=perimeter,
                        bbox=bbox,
                        cif_paint_ring=[(float(x_coord), float(y_coord)) for x_coord, y_coord in paint_ring],
                    )
                )
                next_id += 1
                for hole_points in hole_rings:
                    area, perimeter, bbox = compute_polygon_metrics(hole_points)
                    recovered.append(
                        PolygonData(
                            id=next_id,
                            points=hole_points,
                            is_hole=True,
                            parent_id=parent_id,
                            category=polygon.category,
                            shape_hint=polygon.shape_hint,
                            area=area,
                            perimeter=perimeter,
                            bbox=bbox,
                        )
                    )
                    next_id += 1
            continue
        if not recover_legacy_cut_holes:
            recovered.append(_clone_polygon_with_id(polygon, next_id))
            next_id += 1
            continue
        left = max(0, int(np.floor(min(point[0] for point in polygon.points))) - 3)
        top = max(0, int(np.floor(min(point[1] for point in polygon.points))) - 3)
        right = min(image_width - 1, int(np.ceil(max(point[0] for point in polygon.points))) + 3)
        bottom = int(np.ceil(max(point[1] for point in polygon.points))) + 3
        local_width = max(1, right - left + 1)
        local_height = max(1, bottom - top + 1)
        raw_mask = np.zeros((local_height, local_width), dtype=np.uint8)
        local_points = _to_local_int_points(polygon.points, left, top)
        cv2.fillPoly(raw_mask, [local_points], 255)
        _contours_raw, hierarchy_raw = cv2.findContours(raw_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        raw_hole_count = 0
        if hierarchy_raw is not None and len(hierarchy_raw) > 0:
            raw_hole_count = int(sum(1 for item in hierarchy_raw[0] if int(item[3]) >= 0))
        best_contours: list[np.ndarray] = []
        best_hierarchy: np.ndarray | None = None
        best_hole_count = raw_hole_count
        for kernel_size in (3, 5):
            closed = cv2.morphologyEx(
                raw_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
                iterations=1,
            )
            contours_closed, hierarchy_closed = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            closed_hole_count = 0
            if hierarchy_closed is not None and len(hierarchy_closed) > 0:
                closed_hole_count = int(sum(1 for item in hierarchy_closed[0] if int(item[3]) >= 0))
            if closed_hole_count > best_hole_count:
                best_hole_count = closed_hole_count
                best_contours = [np.asarray(contour) for contour in contours_closed]
                best_hierarchy = hierarchy_closed
        if best_hole_count <= raw_hole_count or best_hierarchy is None or not best_contours or len(best_hierarchy) == 0:
            recovered.append(_clone_polygon_with_id(polygon, next_id))
            next_id += 1
            continue
        hierarchy = best_hierarchy[0]
        contour_to_parent_id: dict[int, int] = {}
        for index, contour in enumerate(best_contours):
            parent_index = int(hierarchy[index][3])
            if parent_index >= 0:
                continue
            parent_poly = _contour_points_to_polygon(
                contour,
                left=left,
                top=top,
                polygon_id=next_id,
                is_hole=False,
                parent_id=None,
            )
            recovered.append(parent_poly)
            contour_to_parent_id[index] = next_id
            next_id += 1
        for index, contour in enumerate(best_contours):
            parent_index = int(hierarchy[index][3])
            if parent_index < 0:
                continue
            recovered_parent_id = contour_to_parent_id.get(parent_index)
            if recovered_parent_id is None:
                continue
            hole_poly = _contour_points_to_polygon(
                contour,
                left=left,
                top=top,
                polygon_id=next_id,
                is_hole=True,
                parent_id=recovered_parent_id,
            )
            recovered.append(hole_poly)
            next_id += 1
        # fall through: punch handling below uses full recovered list after loop
        continue
    # Assumption: KLayout renders each CIF polygon independently with OR/overlap
    # paint and does not punch nested outers out of covering frames. Keep that
    # fidelity on load; opt in only when empty nested windows are needed.
    if _nested_outer_punch_on_load_enabled():
        return _punch_nested_outer_covers(recovered)
    return recovered
