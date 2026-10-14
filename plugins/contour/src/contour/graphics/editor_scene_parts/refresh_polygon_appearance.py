"""editor_scene: refresh polygon appearance."""
from __future__ import annotations
from PyQt6.QtGui import (
    QColor,
)
from contour.domain import PolygonData
from contour.graphics_items import (
    EditablePolygonItem,
)
from contour.graphics.geometry import (
    _contrasting_object_colors,
)
from contour.graphics.tool_mode_logic import (
    is_via_polygon as _is_via_polygon,
)


class RefreshPolygonAppearanceMixin:
    @staticmethod
    def _via_score_color(polygon: PolygonData) -> str | None:
        if not _is_via_polygon(polygon):
            return None
        score = getattr(polygon, "recognition_score", None)
        if score is None:
            return None
        normalized = max(0.0, min(1.0, float(score) / 100.0))
        # HSL hue 0° is red, 60° yellow and 120° green.
        return QColor.fromHslF(normalized / 3.0, 0.82, 0.50).name()

    def set_polygon_category_visible(self, category: str, visible: bool) -> None:
        self._polygon_category_visible[str(category)] = bool(visible)
        for polygon_id, polygon in self._polygons.items():
            if str(getattr(polygon, "category", "") or "") != str(category):
                continue
            item = self._polygon_items.get(polygon_id)
            if item is not None:
                item.setVisible(
                    bool(visible)
                    and self._polygon_overlays_visible
                    and polygon_id not in self._zoom_hidden_contact_ids
                )
        if str(category) == "via":
            for item in self._zoom_contact_composite_items:
                item.setVisible(bool(visible) and self._polygon_overlays_visible)

    def _refresh_polygon_appearance(
        self,
        polygon_id: int,
        item: EditablePolygonItem,
        *,
        editable_vertex_ids: set[int] | None = None,
    ) -> None:
        polygon = self._polygons.get(polygon_id)
        if polygon is None:
            return
        conductor_hover_highlight = self._polygon_has_conductor_hover_highlight(polygon_id)
        if conductor_hover_highlight:
            self._refresh_polygon_item(
                polygon_id,
                item,
                editable_vertex_ids=editable_vertex_ids,
            )
            return
        item.update_selection_appearance(
            self._display_settings,
            selected=polygon_id in self._selected_polygon_ids,
            custom_color=self._via_score_color(polygon) or self._object_color_for(polygon_id),
            needs_repair=self.polygon_needs_repair(polygon_id),
        )

    def _object_color_for(self, polygon_id: int) -> str | None:
        if not self._random_object_colors_enabled:
            return None
        return self._object_colors.get(polygon_id)

    def _rebuild_object_colors(self) -> set[int]:
        previous = self._object_colors
        if self._random_object_colors_enabled:
            signature = self._object_colors_signature()
            if signature == self._object_colors_cache_signature:
                current = dict(self._object_colors_cache)
            else:
                current = _contrasting_object_colors(
                    self._polygons.values(),
                    (self._image_rect.width(), self._image_rect.height()),
                )
                self._object_colors_cache_signature = signature
                self._object_colors_cache = dict(current)
        else:
            current = {}
        self._object_colors = current
        return {
            polygon_id
            for polygon_id in previous.keys() | current.keys()
            if previous.get(polygon_id) != current.get(polygon_id)
        }

    def _cutout_polygons_for(self, polygon_id: int) -> list[PolygonData]:
        if not self._hole_children_by_parent:
            return []
        polygon = self._polygons.get(polygon_id)
        if polygon is None or polygon.is_hole:
            return []
        return self._hole_children_by_parent.get(polygon_id, [])

    def _rebuild_polygon_relationship_cache(self) -> None:
        self._hole_children_by_parent.clear()
        self._polygon_child_ids_by_parent.clear()
        for polygon in self._polygons.values():
            self._index_polygon_relationship(polygon)

    def _index_polygon_relationship(self, polygon: PolygonData) -> None:
        if polygon.parent_id is None:
            return
        self._polygon_child_ids_by_parent.setdefault(
            polygon.parent_id,
            set(),
        ).add(polygon.id)
        if polygon.is_hole:
            self._hole_children_by_parent.setdefault(polygon.parent_id, []).append(polygon)

    def _unindex_polygon_relationship(self, polygon: PolygonData | None) -> None:
        if polygon is None or polygon.parent_id is None:
            return
        child_ids = self._polygon_child_ids_by_parent.get(polygon.parent_id)
        if child_ids is not None:
            child_ids.discard(polygon.id)
            if not child_ids:
                self._polygon_child_ids_by_parent.pop(polygon.parent_id, None)
        if not polygon.is_hole:
            return
        children = self._hole_children_by_parent.get(polygon.parent_id)
        if not children:
            return
        self._hole_children_by_parent[polygon.parent_id] = [child for child in children if child.id != polygon.id]
        if not self._hole_children_by_parent[polygon.parent_id]:
            self._hole_children_by_parent.pop(polygon.parent_id, None)

    def _render_polygon_ids(self, overlapping_ids: list[int]) -> list[int]:
        render_ids: list[int] = []
        for polygon_id in overlapping_ids:
            for family_id in self._polygon_family_ids(polygon_id):
                if family_id not in render_ids:
                    render_ids.append(family_id)
        return sorted(render_ids)

    def _editable_vertex_polygon_ids(self, extra_ids: set[int | None] | None = None) -> set[int]:
        if self._show_all_editable_vertices:
            return {
                polygon_id
                for polygon_id, polygon in self._polygons.items()
                if not _is_via_polygon(polygon)
            }
        active_ids: set[int] = {
            polygon_id for polygon_id in self._selected_polygon_ids if polygon_id in self._polygons
        }
        if self._vertex_preview_polygon_id in self._polygons:
            active_ids.add(self._vertex_preview_polygon_id)
        for polygon_id in extra_ids or set():
            if polygon_id in self._polygons:
                active_ids.add(polygon_id)
        editable_ids: set[int] = set()
        for polygon_id in active_ids:
            if _is_via_polygon(self._polygons[polygon_id]):
                continue
            editable_ids.update(self._polygon_edit_family_ids(polygon_id))
        return editable_ids

    def _polygon_edit_family_ids(self, polygon_id: int) -> list[int]:
        root_id = self._polygon_root_id(polygon_id)
        if root_id is None:
            return []
        return self._polygon_family_ids(root_id)

    def _polygon_root_id(self, polygon_id: int) -> int | None:
        current_id: int | None = polygon_id
        seen: set[int] = set()
        while current_id is not None and current_id in self._polygons and current_id not in seen:
            seen.add(current_id)
            polygon = self._polygons[current_id]
            if polygon.parent_id is None or polygon.parent_id not in self._polygons:
                return current_id
            current_id = polygon.parent_id
        return polygon_id if polygon_id in self._polygons else None
