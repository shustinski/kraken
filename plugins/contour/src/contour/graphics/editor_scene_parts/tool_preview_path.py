"""editor_scene: tool preview path."""
from __future__ import annotations
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import (
    QPainterPath,
)
from shapely import make_valid
from shapely.geometry import Polygon as ShapelyPolygon
from contour.domain import PolygonData
from contour.graphics.brush_vector import (
    QUAD_SEGS_BRUSH_DEFAULT,
    extract_polygonal_union,
    tool_geometry,
)

from contour.graphics.editor_scene_parts.module_state import (
    _HOLE_POLYGON_ITEM_Z,
    _OUTER_PICK_Z_SPAN,
    _OUTER_POLYGON_ITEM_Z,
)


class ToolPreviewPathMixin:
    def _tool_preview_path(self, points: list[tuple[float, float]], thickness: float) -> QPainterPath:
        path = QPainterPath()
        try:
            geom = extract_polygonal_union(
                make_valid(tool_geometry(points, float(thickness), quad_segs=QUAD_SEGS_BRUSH_DEFAULT))
            )
        except Exception:
            return path
        if geom.is_empty:
            return path

        def add_ring(coords: object) -> None:
            ring_points = list(coords)
            if len(ring_points) < 2:
                return
            ring = QPainterPath()
            start_x, start_y = float(ring_points[0][0]), float(ring_points[0][1])
            ring.moveTo(start_x, start_y)
            for x_coord, y_coord in ring_points[1:]:
                ring.lineTo(float(x_coord), float(y_coord))
            ring.closeSubpath()
            path.addPath(ring)

        def add_polygon(poly: ShapelyPolygon) -> None:
            add_ring(poly.exterior.coords)
            for interior in poly.interiors:
                add_ring(interior.coords)

        geom_type = getattr(geom, "geom_type", "")
        if geom_type == "Polygon":
            add_polygon(geom)
        elif geom_type == "MultiPolygon":
            for part in geom.geoms:
                add_polygon(part)
        path.setFillRule(Qt.FillRule.WindingFill)
        return path

    def refresh_polygon_items(self) -> None:
        self._refresh_all_items()

    def _rebuild_outer_pick_z_ranks(self) -> None:
        outer_ids = sorted(
            polygon_id for polygon_id, polygon in self._polygons.items() if not polygon.is_hole
        )
        outer_ids.sort(key=lambda polygon_id: float(self._polygons[polygon_id].area))
        self._outer_pick_z_rank = {polygon_id: rank for rank, polygon_id in enumerate(outer_ids)}

    def _pick_z_value_for_polygon(self, polygon: PolygonData) -> float:
        if polygon.is_hole:
            return _HOLE_POLYGON_ITEM_Z
        rank = self._outer_pick_z_rank.get(polygon.id, 0)
        count = len(self._outer_pick_z_rank)
        if count <= 1:
            return _OUTER_POLYGON_ITEM_Z + _OUTER_PICK_Z_SPAN
        return _OUTER_POLYGON_ITEM_Z + _OUTER_PICK_Z_SPAN * (1.0 - rank / (count - 1))

    def _refresh_all_items(
        self,
        *,
        rebuild_repair_cache: bool = True,
        defer_hit_paths: bool = False,
        defer_object_colors: bool = False,
        defer_overlap_repair: bool = False,
    ) -> None:
        if rebuild_repair_cache:
            self._rebuild_polygons_needing_repair_cache(defer_overlap=defer_overlap_repair)
        if defer_object_colors:
            self._queue_deferred_object_colors()
        else:
            self._rebuild_object_colors()
        self._rebuild_outer_pick_z_ranks()
        editable_vertex_ids = self._editable_vertex_polygon_ids()
        bulk_load = defer_hit_paths
        for polygon_id, item in self._polygon_items.items():
            self._refresh_polygon_item(
                polygon_id,
                item,
                editable_vertex_ids=editable_vertex_ids,
                defer_hit_path=defer_hit_paths,
                bulk_load=bulk_load,
            )

    def _queue_deferred_object_colors(self) -> None:
        if not self._random_object_colors_enabled:
            return
        self._object_colors_refresh_pending = True

    def flush_deferred_object_colors(self) -> None:
        if not self._object_colors_refresh_pending:
            return
        self._object_colors_refresh_pending = False
        self._apply_deferred_object_colors()

    def _schedule_deferred_object_colors(self) -> None:
        if not self._random_object_colors_enabled:
            return
        timer = getattr(self, "_deferred_object_colors_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._apply_deferred_object_colors)
            self._deferred_object_colors_timer = timer
        timer.start(0)

    def _object_colors_signature(self) -> tuple[object, ...]:
        conductors = tuple(
            (polygon_id, polygon.bbox)
            for polygon_id, polygon in sorted(self._polygons.items())
            if not polygon.is_hole and str(getattr(polygon, "category", "") or "") == "conductor"
        )
        return (
            round(float(self._image_rect.width()), 3),
            round(float(self._image_rect.height()), 3),
            conductors,
        )

    def _apply_deferred_object_colors(self) -> None:
        if not self._random_object_colors_enabled or not self._polygon_items:
            return
        changed_ids = self._rebuild_object_colors()
        if not changed_ids:
            return
        editable_vertex_ids = self._editable_vertex_polygon_ids()
        for polygon_id in changed_ids:
            item = self._polygon_items.get(polygon_id)
            if item is not None:
                self._refresh_polygon_appearance(polygon_id, item, editable_vertex_ids=editable_vertex_ids)
