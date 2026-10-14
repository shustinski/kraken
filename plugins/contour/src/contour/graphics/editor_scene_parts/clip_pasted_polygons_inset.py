"""editor_scene: clip pasted polygons inset."""
from __future__ import annotations
from PyQt6.QtCore import QRectF
from shapely import make_valid, unary_union
from shapely.geometry import box as shapely_box
from contour.domain import PolygonData
from contour.graphics.brush_vector import (
    QUAD_SEGS_BRUSH_DEFAULT,
    region_geometry,
    shapely_to_polygon_data_list,
    tool_geometry,
)
from contour.graphics.geometry import (
    _polygon_data_rect,
)
from contour.graphics.tool_mode_logic import (
    is_via_polygon as _is_via_polygon,
)


class ClipPastedPolygonsInsetMixin:
    def _clip_pasted_polygons_to_inset(
        self,
        polygons: list[PolygonData],
        *,
        inset: float,
    ) -> list[PolygonData]:
        image_rect = QRectF(self._image_rect).normalized()
        if image_rect.width() <= 1.0 or image_rect.height() <= 1.0:
            return [polygon.clone() for polygon in polygons]
        clip_rect = image_rect.adjusted(inset, inset, -inset, -inset)
        if clip_rect.width() <= 0.0 or clip_rect.height() <= 0.0:
            return []
        polygon_rects = [
            (polygon, _polygon_data_rect(polygon))
            for polygon in polygons
        ]
        if all(clip_rect.contains(polygon_rect) for _polygon, polygon_rect in polygon_rects):
            return [polygon.clone() for polygon in polygons]
        clip_geom = shapely_box(
            float(clip_rect.left()),
            float(clip_rect.top()),
            float(clip_rect.right()),
            float(clip_rect.bottom()),
        )
        if all(_is_via_polygon(polygon) for polygon in polygons):
            clipped_contacts: list[PolygonData] = []
            for polygon, polygon_rect in polygon_rects:
                if not clip_rect.intersects(polygon_rect):
                    continue
                if clip_rect.contains(polygon_rect):
                    clipped_contacts.append(polygon.clone())
                    continue
                try:
                    polygon_geom = tool_geometry(
                        polygon.points,
                        None,
                        quad_segs=QUAD_SEGS_BRUSH_DEFAULT,
                    )
                    clipped = shapely_to_polygon_data_list(
                        unary_union(make_valid(polygon_geom.intersection(clip_geom)))
                    )
                except Exception:
                    clipped = []
                for candidate in clipped:
                    candidate.category = polygon.category
                    candidate.shape_hint = polygon.shape_hint
                    clipped_contacts.append(candidate)
            return clipped_contacts
        try:
            polygon_map = {polygon.id: polygon.clone() for polygon in polygons}
            pasted_region = region_geometry(polygon_map, list(polygon_map))
            clipped = shapely_to_polygon_data_list(
                unary_union(make_valid(pasted_region.intersection(clip_geom)))
            )
        except Exception:
            return []
        reference = next((polygon for polygon in polygons if not polygon.is_hole), polygons[0])
        for candidate in clipped:
            candidate.category = reference.category
            candidate.shape_hint = reference.shape_hint
        return clipped
