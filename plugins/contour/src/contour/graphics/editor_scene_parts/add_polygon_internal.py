"""editor_scene: add polygon internal."""
from __future__ import annotations
import numpy as np
from contour.domain import PolygonData
from contour.graphics_items import (
    EditablePolygonItem,
)


class AddPolygonInternalMixin:
    def _add_polygon_internal(
        self,
        polygon: PolygonData,
        emit_signal: bool = True,
        refresh: bool = True,
        *,
        paint: bool | None = None,
        defer_pick_z_rebuild: bool = False,
    ) -> None:
        if polygon.id in self._polygon_items:
            self._remove_polygon_internal(polygon.id, emit_signal=False, refresh=False)
        self._polygons[polygon.id] = polygon.clone()
        self._polygon_spatial_index.upsert(self._polygons[polygon.id])
        self._polygon_hover_contours[polygon.id] = np.asarray(
            self._polygons[polygon.id].points,
            dtype=np.float32,
        ).reshape((-1, 1, 2))
        self._index_polygon_relationship(self._polygons[polygon.id])
        self._next_polygon_id = max(self._next_polygon_id, polygon.id + 1)
        should_paint = True if paint is None else bool(paint)
        custom_color = self._via_score_color(self._polygons[polygon.id]) or self._object_color_for(
            polygon.id
        )
        if self._recycled_polygon_items:
            item = self._recycled_polygon_items.pop()
            if should_paint:
                item.update_from_polygon(
                    self._polygons[polygon.id],
                    self._display_settings,
                    selected=False,
                    cutout_polygons=self._cutout_polygons_for(polygon.id) if refresh else None,
                    custom_color=custom_color,
                    needs_repair=self.polygon_needs_repair(polygon.id),
                )
            else:
                item.bind_polygon_data(self._polygons[polygon.id])
        else:
            item = EditablePolygonItem(
                self._polygons[polygon.id],
                self._display_settings,
                custom_color=custom_color,
                paint=should_paint,
            )
            self.addItem(item)
        if not refresh and not defer_pick_z_rebuild:
            self._rebuild_outer_pick_z_ranks()
            item.setZValue(self._pick_z_value_for_polygon(self._polygons[polygon.id]))
        category = str(getattr(polygon, "category", "") or "")
        item.setVisible(
            self._polygon_category_visible.get(category, True)
            and self._polygon_overlays_visible
        )
        self._polygon_items[polygon.id] = item
        if refresh:
            self._refresh_all_items()
        elif not defer_pick_z_rebuild:
            changed_color_ids = self._rebuild_object_colors()
            self._refresh_polygon_items_by_id(*(changed_color_ids | {polygon.id}))
        if emit_signal:
            self.polygonsChanged.emit()
