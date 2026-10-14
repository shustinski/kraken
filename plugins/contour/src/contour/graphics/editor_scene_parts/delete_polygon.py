"""editor_scene: delete polygon."""
from __future__ import annotations
from ...application.vector_geometry_postprocess import (
    union_after_removing_polygon_ids,
)


class DeletePolygonMixin:
    def delete_polygon(self, polygon_id: int | None = None) -> bool:
        target_ids = [polygon_id] if polygon_id is not None else sorted(self._selected_polygon_ids)
        if not target_ids and self._selected_polygon_id is not None:
            target_ids = [self._selected_polygon_id]
        target_ids = [
            target_id
            for target_id in target_ids
            if self.polygon_is_deletable(self._polygons.get(target_id))
        ]
        delete_ids = self._delete_polygon_ids_with_descendants(target_ids)
        delete_ids = [
            target_id
            for target_id in delete_ids
            if self.polygon_is_deletable(self._polygons.get(target_id))
        ]
        target_polygons = [self._polygons[target_id].clone() for target_id in delete_ids if target_id in self._polygons]
        if not target_polygons:
            return False
        before = self.get_polygons()
        removed_ids = set(delete_ids)
        removed_snapshots = [
            polygon.clone()
            for polygon in before
            if polygon.id in removed_ids
        ]
        remaining = [polygon.clone() for polygon in before if polygon.id not in removed_ids]
        remaining = union_after_removing_polygon_ids(
            remaining,
            removed_ids,
            removed_polygons=removed_snapshots,
        )
        self._push_polygon_set_change(before, remaining, "Delete polygon")
        return True

    def _delete_polygon_ids_with_descendants(self, target_ids: list[int | None]) -> list[int]:
        delete_ids: set[int] = set()
        pending = [
            target_id
            for target_id in target_ids
            if target_id in self._polygons and not self._polygons[target_id].is_hole
        ]
        while pending:
            current_id = pending.pop()
            if current_id is None or current_id in delete_ids or current_id not in self._polygons:
                continue
            delete_ids.add(current_id)
            pending.extend(
                child_id
                for child_id in self._polygon_child_ids_by_parent.get(current_id, ())
                if child_id not in delete_ids
            )
        for target_id in target_ids:
            if target_id in self._polygons and self._polygons[target_id].is_hole:
                delete_ids.add(target_id)
        return sorted(delete_ids)
