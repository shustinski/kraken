"""editor_scene: refresh polygon item."""
from __future__ import annotations
from contour.graphics_items import (
    EditablePolygonItem,
    _hole_display_hidden,
)
from contour.i18n import tr


class RefreshPolygonItemMixin:
    def _refresh_polygon_items_by_id(self, *polygon_ids: int | None) -> None:
        editable_vertex_ids = self._editable_vertex_polygon_ids()
        for polygon_id in {polygon_id for polygon_id in polygon_ids if polygon_id is not None}:
            item = self._polygon_items.get(polygon_id)
            if item is not None:
                self._refresh_polygon_item(polygon_id, item, editable_vertex_ids=editable_vertex_ids)

    def _refresh_polygon_selection_item(
        self,
        polygon_id: int,
        item: EditablePolygonItem,
        *,
        editable_vertex_ids: set[int],
    ) -> None:
        polygon = self._polygons.get(polygon_id)
        if polygon is None:
            return
        item.update_selection_appearance(
            self._display_settings,
            selected=polygon_id in self._selected_polygon_ids,
            custom_color=self._via_score_color(polygon) or self._object_color_for(polygon_id),
            needs_repair=self.polygon_needs_repair(polygon_id),
        )
        item.update_move_target_highlight(
            self._display_settings,
            preview_vertices=polygon_id in editable_vertex_ids,
            selected=polygon_id in self._selected_polygon_ids,
            highlight_vertex_index=None,
            highlight_edge_index=None,
        )

    def _refresh_polygon_item(
        self,
        polygon_id: int,
        item: EditablePolygonItem | None = None,
        *,
        editable_vertex_ids: set[int] | None = None,
        defer_hit_path: bool = False,
        bulk_load: bool = False,
    ) -> None:
        polygon = self._polygons.get(polygon_id)
        if polygon is None:
            return
        if item is None:
            item = self._polygon_items.get(polygon_id)
        if item is None:
            return
        conductor_hover_highlight = self._polygon_has_conductor_hover_highlight(
            polygon_id
        ) or (
            polygon_id in self._delete_area_highlight_ids
            and polygon_id not in self._selected_polygon_ids
        )
        item.update_from_polygon(
            polygon,
            self._display_settings,
            selected=polygon_id in self._selected_polygon_ids,
            cutout_polygons=self._cutout_polygons_for(polygon_id),
            custom_color=self._via_score_color(polygon) or self._object_color_for(polygon_id),
            conductor_hover_highlight=conductor_hover_highlight,
            preview_vertices=polygon_id
            in (
                editable_vertex_ids
                if editable_vertex_ids is not None
                else self._editable_vertex_polygon_ids()
            ),
            highlight_vertex_index=(
                self._move_target_preview[2]
                if self._move_target_preview is not None
                and self._move_target_preview[0] == "vertex"
                and self._move_target_preview[1] == polygon_id
                else None
            ),
            highlight_edge_index=(
                self._move_target_preview[2]
                if self._move_target_preview is not None
                and self._move_target_preview[0] == "edge"
                and self._move_target_preview[1] == polygon_id
                else None
            ),
            needs_repair=self.polygon_needs_repair(polygon_id),
            defer_hit_path=defer_hit_path,
            bulk_load=bulk_load,
        )
        item.setZValue(self._pick_z_value_for_polygon(polygon))
        cat = str(getattr(polygon, "category", "") or "")
        vis = self._polygon_category_visible.get(cat, True)
        item.setVisible(
            bool(vis)
            and self._polygon_overlays_visible
            and not _hole_display_hidden(polygon, self._polygons)
        )
        if self.polygon_needs_repair(polygon_id):
            item.setToolTip(tr("invalid_polygon_description_tooltip", language=self._ui_language))
        else:
            item.setToolTip("")
