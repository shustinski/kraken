"""editor_scene: clip authored polygons image."""
from __future__ import annotations
from PyQt6.QtCore import QPointF, QRectF
from shapely import make_valid, unary_union
from shapely.geometry import box as shapely_box
from contour.domain import PolygonData
from contour.graphics.brush_vector import (
    QUAD_SEGS_BRUSH_DEFAULT,
    region_geometry,
    shapely_to_polygon_data_list,
    tool_geometry,
)


class ClipAuthoredPolygonsImageMixin:
    def _clip_authored_polygon_to_image(
        self,
        polygon: PolygonData,
        *,
        allocate_ids: bool = True,
    ) -> list[PolygonData]:
        image_rect = QRectF(self._image_rect).normalized()
        if image_rect.width() <= 1.0 or image_rect.height() <= 1.0:
            out = [polygon.clone()]
        else:
            try:
                polygon_geom = tool_geometry(polygon.points, None, quad_segs=QUAD_SEGS_BRUSH_DEFAULT)
                image_geom = shapely_box(
                    float(image_rect.left()),
                    float(image_rect.top()),
                    float(image_rect.right()),
                    float(image_rect.bottom()),
                )
                clipped_geom = unary_union(make_valid(polygon_geom.intersection(image_geom)))
                out = shapely_to_polygon_data_list(clipped_geom)
            except Exception:
                out = [polygon.clone()]
        if not out:
            return []
        id_map: dict[int, int] = {}
        for index, clipped in enumerate(out):
            clipped.category = polygon.category
            clipped.shape_hint = polygon.shape_hint
            if allocate_ids:
                if index == 0:
                    new_id = polygon.id
                    self._next_polygon_id = max(self._next_polygon_id, int(new_id) + 1)
                else:
                    new_id = self._next_polygon_id
                    self._next_polygon_id += 1
            else:
                new_id = polygon.id if index == 0 else self._next_polygon_id
                if index > 0:
                    self._next_polygon_id += 1
            id_map[clipped.id] = new_id
            clipped.id = new_id
        for clipped in out:
            clipped.parent_id = None if clipped.parent_id is None else id_map.get(clipped.parent_id)
        return out

    def _clip_authored_polygons_to_image(self, polygons: list[PolygonData]) -> list[PolygonData]:
        image_rect = QRectF(self._image_rect).normalized()
        if image_rect.width() <= 1.0 or image_rect.height() <= 1.0:
            return [polygon.clone() for polygon in polygons]
        if all(
            image_rect.contains(QPointF(float(x_coord), float(y_coord)))
            for polygon in polygons
            for x_coord, y_coord in polygon.points
        ):
            # Preserve the complete root/hole family when clipping is a no-op.
            # Clipping rings independently turns every hole into an outer ring.
            return [polygon.clone() for polygon in polygons]
        try:
            by_id = {polygon.id: polygon.clone() for polygon in polygons}
            family_geom = region_geometry(by_id, list(by_id))
            image_geom = shapely_box(
                float(image_rect.left()),
                float(image_rect.top()),
                float(image_rect.right()),
                float(image_rect.bottom()),
            )
            clipped_family = shapely_to_polygon_data_list(
                unary_union(make_valid(family_geom.intersection(image_geom)))
            )
            if not clipped_family:
                return []
            reference = next((polygon for polygon in polygons if not polygon.is_hole), polygons[0])
            for candidate in clipped_family:
                candidate.category = reference.category
                candidate.shape_hint = reference.shape_hint
            return self._assign_polygon_ids(clipped_family, [polygon.id for polygon in polygons])
        except Exception:
            # Fall back to the older per-ring path only for malformed legacy
            # families that cannot be represented by GEOS.
            pass
        clipped: list[PolygonData] = []
        used_ids: set[int] = set()
        for polygon in polygons:
            for candidate in self._clip_authored_polygon_to_image(polygon, allocate_ids=False):
                if candidate.id in used_ids:
                    candidate.id = self._next_polygon_id
                    self._next_polygon_id += 1
                used_ids.add(candidate.id)
                clipped.append(candidate)
        return clipped
