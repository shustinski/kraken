"""editor_scene: refresh move target highlight."""
from __future__ import annotations
from math import ceil, floor
from PyQt6.QtCore import QRectF
from contour.domain import PolygonData
from contour.graphics.geometry import (
    _polygon_data_rect,
)
from contour.graphics.tool_mode_logic import (
    is_via_polygon as _is_via_polygon,
)


class RefreshMoveTargetHighlightMixin:
    def _set_move_target_preview(
        self,
        target: tuple[str, int, int] | None,
        *,
        refresh: bool = True,
    ) -> None:
        if target is not None:
            kind, polygon_id, _index = target
            if polygon_id not in self._polygons or kind not in {"vertex", "edge"}:
                target = None
        if target == self._move_target_preview:
            return
        refresh_ids: set[int] = set()
        if self._move_target_preview is not None:
            refresh_ids.add(self._move_target_preview[1])
        if target is not None:
            refresh_ids.add(target[1])
        self._move_target_preview = target
        if refresh and refresh_ids:
            self._refresh_move_target_highlight_for_polygons(refresh_ids)

    def _refresh_move_target_highlight_for_polygons(self, polygon_ids: set[int]) -> None:
        editable_vertex_ids = self._editable_vertex_polygon_ids()
        for polygon_id in polygon_ids:
            item = self._polygon_items.get(polygon_id)
            polygon = self._polygons.get(polygon_id)
            if item is None or polygon is None:
                continue
            conductor_hover_highlight = self._polygon_has_conductor_hover_highlight(
                polygon_id
            ) or (
                polygon_id in self._delete_area_highlight_ids
                and polygon_id not in self._selected_polygon_ids
            )
            item.update_move_target_highlight(
                self._display_settings,
                preview_vertices=polygon_id in editable_vertex_ids,
                selected=polygon_id in self._selected_polygon_ids,
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
                conductor_hover_highlight=conductor_hover_highlight,
            )

    def _polygon_has_conductor_hover_highlight(self, polygon_id: int) -> bool:
        return (
            self._hover_conductor_polygon_id is not None
            and polygon_id == self._hover_conductor_polygon_id
            and polygon_id not in self._selected_polygon_ids
        )

    def _set_hover_conductor_polygon_id(self, conductor_id: int | None) -> None:
        if conductor_id is not None and conductor_id not in self._polygons:
            conductor_id = None
        if conductor_id == self._hover_conductor_polygon_id:
            return
        previous_id = self._hover_conductor_polygon_id
        self._hover_conductor_polygon_id = conductor_id
        changed_ids = {
            polygon_id
            for polygon_id in (previous_id, conductor_id)
            if polygon_id is not None
        }
        editable_vertex_ids = self._editable_vertex_polygon_ids()
        for polygon_id in changed_ids:
            item = self._polygon_items.get(polygon_id)
            if item is not None:
                self._refresh_polygon_item(
                    polygon_id,
                    item,
                    editable_vertex_ids=editable_vertex_ids,
                )
        self._refresh_move_target_highlight_for_polygons(changed_ids)

    def _set_vertex_preview_polygon_id(self, polygon_id: int | None) -> None:
        if polygon_id is not None and polygon_id not in self._polygons:
            polygon_id = None
        if polygon_id == self._vertex_preview_polygon_id:
            return
        previous_ids = self._editable_vertex_polygon_ids(extra_ids={self._vertex_preview_polygon_id})
        self._vertex_preview_polygon_id = polygon_id
        next_ids = self._editable_vertex_polygon_ids()
        self._refresh_polygon_items_by_id(*(previous_ids | next_ids))

    def selected_polygon_id(self) -> int | None:
        return self._selected_polygon_id

    def select_polygon(self, polygon_id: int | None, *, additive: bool = False) -> None:
        if polygon_id is not None and polygon_id not in self._polygons:
            polygon_id = None
        refresh_ids = set(self._selected_polygon_ids)
        if self._selected_polygon_ids or self._vertex_preview_polygon_id is not None:
            refresh_ids.update(self._editable_vertex_polygon_ids())
        if polygon_id is None:
            if not additive:
                self._selected_polygon_ids.clear()
            self._selected_polygon_id = None
        elif additive:
            if polygon_id in self._selected_polygon_ids:
                self._selected_polygon_ids.remove(polygon_id)
                self._selected_polygon_id = next(iter(sorted(self._selected_polygon_ids)), None)
            else:
                self._selected_polygon_ids.add(polygon_id)
                self._selected_polygon_id = polygon_id
        else:
            self._selected_polygon_ids = {polygon_id}
            self._selected_polygon_id = polygon_id
        editable_vertex_ids = self._editable_vertex_polygon_ids()
        refresh_ids.update(self._selected_polygon_ids)
        refresh_ids.update(editable_vertex_ids)
        for refresh_id in refresh_ids:
            item = self._polygon_items.get(refresh_id)
            if item is not None:
                self._refresh_polygon_selection_item(
                    refresh_id,
                    item,
                    editable_vertex_ids=editable_vertex_ids,
                )
        self.activePolygonChanged.emit(self._selected_polygon_id)

    def select_polygons(self, polygon_ids: list[int]) -> None:
        refresh_ids = set(self._selected_polygon_ids)
        if self._selected_polygon_ids or self._vertex_preview_polygon_id is not None:
            refresh_ids.update(self._editable_vertex_polygon_ids())
        selected_ids = {polygon_id for polygon_id in polygon_ids if polygon_id in self._polygons}
        self._selected_polygon_ids = selected_ids
        self._selected_polygon_id = min(selected_ids) if selected_ids else None
        editable_vertex_ids = self._editable_vertex_polygon_ids()
        refresh_ids.update(selected_ids)
        refresh_ids.update(editable_vertex_ids)
        for refresh_id in refresh_ids:
            item = self._polygon_items.get(refresh_id)
            if item is not None:
                self._refresh_polygon_selection_item(
                    refresh_id,
                    item,
                    editable_vertex_ids=editable_vertex_ids,
                )
        self.activePolygonChanged.emit(self._selected_polygon_id)

    def select_polygons_in_rect(self, rect: QRectF, *, additive: bool = False) -> None:
        normalized = rect.normalized()
        if normalized.width() <= 0.0 or normalized.height() <= 0.0:
            if not additive:
                self.select_polygon(None)
            return
        candidate_ids = self._polygon_spatial_index.query_bbox(
            (
                floor(normalized.left()),
                floor(normalized.top()),
                max(1, ceil(normalized.width())),
                max(1, ceil(normalized.height())),
            )
        )
        selected_ids = {
            polygon_id
            for polygon_id in candidate_ids
            if (polygon := self._polygons.get(polygon_id)) is not None
            if _polygon_data_rect(polygon).intersects(normalized)
        }
        if additive:
            selected_ids.update(self._selected_polygon_ids)
        self.select_polygons(sorted(selected_ids))

    def polygon_snapshot(self, polygon_id: int | None) -> PolygonData | None:
        if polygon_id is None or polygon_id not in self._polygons:
            return None
        return self._polygons[polygon_id].clone()

    def polygon_is_contact(self, polygon_id: int | None) -> bool:
        return (
            polygon_id is not None
            and polygon_id in self._polygons
            and _is_via_polygon(self._polygons[polygon_id])
        )
