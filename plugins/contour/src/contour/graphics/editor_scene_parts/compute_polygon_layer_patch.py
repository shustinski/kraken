"""editor_scene: compute polygon layer patch."""
from __future__ import annotations
from contour.commands import (
    MovePolygonCommand,
    ReplacePolygonSetCommand,
    ReplacePolygonsPatchCommand,
)
from contour.domain import PolygonData


class ComputePolygonLayerPatchMixin:
    def _push_known_polygon_patch(
        self,
        *,
        removed_polygons: list[PolygonData],
        added_polygons: list[PolygonData],
        description: str,
        select_polygon_id: int | None = None,
        sync_overlap_repair: bool = False,
    ) -> None:
        removed_ids = frozenset(polygon.id for polygon in removed_polygons)
        self._pending_workspace_layer_patch = (removed_ids, list(added_polygons))
        if select_polygon_id is not None:
            self._last_geometry_changed_polygon_id = select_polygon_id
        elif added_polygons:
            self._last_geometry_changed_polygon_id = added_polygons[0].id
        self.undo_stack.push(
            ReplacePolygonsPatchCommand(
                self,
                removed_polygons,
                added_polygons,
                description,
                select_polygon_id=select_polygon_id,
                sync_overlap_repair=sync_overlap_repair,
            )
        )

    def take_last_workspace_layer_patch(self) -> tuple[frozenset[int], list[PolygonData]] | None:
        patch = self._pending_workspace_layer_patch
        self._pending_workspace_layer_patch = None
        return patch

    @staticmethod
    def _compute_polygon_layer_patch(
        before: list[PolygonData],
        after: list[PolygonData],
    ) -> tuple[list[PolygonData], list[PolygonData]]:
        before_by_id = {polygon.id: polygon for polygon in before}
        after_by_id = {polygon.id: polygon for polygon in after}
        removed: list[PolygonData] = []
        added: list[PolygonData] = []

        def _same_geometry(left: PolygonData, right: PolygonData) -> bool:
            return (
                left.points == right.points
                and left.parent_id == right.parent_id
                and left.is_hole == right.is_hole
            )

        for polygon_id, before_polygon in before_by_id.items():
            after_polygon = after_by_id.get(polygon_id)
            if after_polygon is None:
                removed.append(before_polygon)
            elif not _same_geometry(before_polygon, after_polygon):
                removed.append(before_polygon)
                added.append(after_polygon)

        for polygon_id, after_polygon in after_by_id.items():
            if polygon_id not in before_by_id:
                added.append(after_polygon)

        return removed, added

    @staticmethod
    def _should_use_incremental_polygon_patch(
        before: list[PolygonData],
        removed: list[PolygonData],
    ) -> bool:
        if not before or not removed:
            return False
        removed_ids = {polygon.id for polygon in removed}
        unchanged_count = sum(1 for polygon in before if polygon.id not in removed_ids)
        return unchanged_count > 0 and len(removed) < len(before)

    def _push_polygon_set_change(
        self,
        before: list[PolygonData],
        after: list[PolygonData],
        description: str,
        *,
        select_polygon_id: int | None = None,
        sync_overlap_repair: bool = False,
    ) -> None:
        """Commit a geometry pass with an incremental patch when most polygons are unchanged."""

        removed, added = self._compute_polygon_layer_patch(before, after)
        if self._should_use_incremental_polygon_patch(before, removed):
            self.undo_stack.push(
                ReplacePolygonsPatchCommand(
                    self,
                    removed,
                    added,
                    description,
                    select_polygon_id=select_polygon_id,
                    sync_overlap_repair=sync_overlap_repair,
                ),
            )
            return
        self.undo_stack.push(ReplacePolygonSetCommand(self, before, after, description))
        if select_polygon_id is not None:
            self.select_polygon(select_polygon_id)

    def _try_commit_single_polygon_points_change(
        self,
        before: list[PolygonData],
        after: list[PolygonData],
        *,
        description: str,
        select_polygon_id: int | None = None,
    ) -> bool:
        removed, added = self._compute_polygon_layer_patch(before, after)
        if len(removed) != 1 or len(added) != 1 or removed[0].id != added[0].id:
            return False
        polygon_id = added[0].id
        self._last_geometry_changed_polygon_id = polygon_id
        self.undo_stack.push(
            MovePolygonCommand(
                self,
                polygon_id,
                removed[0].points,
                added[0].points,
                description=description,
            )
        )
        if select_polygon_id is not None:
            self.select_polygon(select_polygon_id)
        return True

    def take_last_geometry_changed_polygon_id(self) -> int | None:
        polygon_id = self._last_geometry_changed_polygon_id
        self._last_geometry_changed_polygon_id = None
        return polygon_id

    def _discard_authored_cif_paint_rings(self, polygon_ids: set[int]) -> None:
        for polygon_id in polygon_ids:
            polygon = self._polygons.get(polygon_id)
            if polygon is None:
                continue
            if polygon.cif_paint_ring:
                polygon.cif_paint_ring = []
            root_id = self._polygon_root_id(polygon_id)
            if root_id is not None:
                root = self._polygons.get(root_id)
                if root is not None and root.cif_paint_ring:
                    root.cif_paint_ring = []
