from __future__ import annotations

from typing import Any

from PyQt6.QtGui import QUndoCommand

from .domain import PolygonData


class AddPolygonCommand(QUndoCommand):
    def __init__(self, scene: Any, polygon: PolygonData, *, select_after_redo: bool = False) -> None:
        super().__init__("Add polygon")
        self._scene = scene
        self._polygon = polygon.clone()
        self._select_after_redo = bool(select_after_redo)

    def redo(self) -> None:
        polygon = self._polygon.clone()
        if self._select_after_redo:
            self._scene._add_polygon_internal(polygon, emit_signal=True, refresh=False)
            self._scene.select_polygon(polygon.id)
            return
        self._scene._add_polygon_internal(polygon, emit_signal=True, refresh=True)

    def undo(self) -> None:
        self._scene._remove_polygon_internal(self._polygon.id)


class DeletePolygonCommand(QUndoCommand):
    def __init__(self, scene: Any, polygon: PolygonData) -> None:
        super().__init__("Delete polygon")
        self._scene = scene
        self._polygon = polygon.clone()

    def redo(self) -> None:
        self._scene._remove_polygon_internal(self._polygon.id)

    def undo(self) -> None:
        self._scene._add_polygon_internal(self._polygon.clone())


class MoveVertexCommand(QUndoCommand):
    def __init__(
        self,
        scene: Any,
        polygon_id: int,
        vertex_index: int,
        old_point: tuple[float, float],
        new_point: tuple[float, float],
    ) -> None:
        super().__init__("Move vertex")
        self._scene = scene
        self._polygon_id = polygon_id
        self._vertex_index = vertex_index
        self._old_point = old_point
        self._new_point = new_point

    def redo(self) -> None:
        self._scene._set_vertex_internal(self._polygon_id, self._vertex_index, self._new_point)

    def undo(self) -> None:
        self._scene._set_vertex_internal(self._polygon_id, self._vertex_index, self._old_point)


class AddVertexCommand(QUndoCommand):
    def __init__(self, scene: Any, polygon_id: int, insert_index: int, point: tuple[float, float]) -> None:
        super().__init__("Add vertex")
        self._scene = scene
        self._polygon_id = polygon_id
        self._insert_index = insert_index
        self._point = point

    def redo(self) -> None:
        self._scene._insert_vertex_internal(self._polygon_id, self._insert_index, self._point)

    def undo(self) -> None:
        self._scene._remove_vertex_internal(self._polygon_id, self._insert_index)


class DeleteVertexCommand(QUndoCommand):
    def __init__(self, scene: Any, polygon_id: int, vertex_index: int, point: tuple[float, float]) -> None:
        super().__init__("Delete vertex")
        self._scene = scene
        self._polygon_id = polygon_id
        self._vertex_index = vertex_index
        self._point = point

    def redo(self) -> None:
        self._scene._remove_vertex_internal(self._polygon_id, self._vertex_index)

    def undo(self) -> None:
        self._scene._insert_vertex_internal(self._polygon_id, self._vertex_index, self._point)


class MovePolygonCommand(QUndoCommand):
    def __init__(
        self,
        scene: Any,
        polygon_id: int,
        old_points: list[tuple[float, float]],
        new_points: list[tuple[float, float]],
    ) -> None:
        super().__init__("Move polygon")
        self._scene = scene
        self._polygon_id = polygon_id
        self._old_points = [(float(x), float(y)) for x, y in old_points]
        self._new_points = [(float(x), float(y)) for x, y in new_points]

    def redo(self) -> None:
        self._scene._replace_polygon_points_internal(self._polygon_id, self._new_points)

    def undo(self) -> None:
        self._scene._replace_polygon_points_internal(self._polygon_id, self._old_points)


class ReplacePolygonSetCommand(QUndoCommand):
    """Replace the whole polygon layer (used for geometry post-processing with correct undo)."""

    def __init__(
        self,
        scene: Any,
        polygons_before: list[PolygonData],
        polygons_after: list[PolygonData],
        description: str = "Vector geometry cleanup",
    ) -> None:
        super().__init__(description)
        self._scene = scene
        self._before = [polygon.clone() for polygon in polygons_before]
        self._after = [polygon.clone() for polygon in polygons_after]

    def redo(self) -> None:
        self._scene._bulk_restore_polygons(self._after, emit_signal=True)

    def undo(self) -> None:
        self._scene._bulk_restore_polygons(self._before, emit_signal=True)
