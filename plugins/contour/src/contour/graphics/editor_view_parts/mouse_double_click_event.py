"""editor_view: mouse double click event."""
from __future__ import annotations
from PyQt6.QtCore import (
    Qt,
)
from PyQt6.QtGui import (
    QContextMenuEvent,
    QMouseEvent,
    QTabletEvent,
)
from PyQt6.QtWidgets import QMenu
from contour.graphics.geometry import (
    _snap_to_45,
)
from contour.graphics.tools import BrushMode, EditorTool, PolygonCreateMode


class MouseDoubleClickEventMixin:
    def tabletEvent(self, event: QTabletEvent | None) -> None:
        if event is None:
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        self._last_pointer_scene_pos = scene_pos
        self._update_tool_cursors()
        if self._tool != EditorTool.BRUSH:
            super().tabletEvent(event)
            return
        if event.type() == event.Type.TabletPress:
            self._start_brush_drag(scene_pos, erase=False)
            event.accept()
            return
        if event.type() == event.Type.TabletMove and self._drag_kind == "brush":
            if self._brush_mode == BrushMode.ANGLED and self._drag_start_scene_pos is not None:
                self._editor_scene.update_pending_cursor(_snap_to_45(self._drag_start_scene_pos, scene_pos))
            else:
                self._append_brush_point(scene_pos)
            event.accept()
            return
        if event.type() == event.Type.TabletRelease and self._drag_kind == "brush":
            self._commit_brush_drag(scene_pos)
            # Keep cleanup symmetrical with mouse release branch.
            self._drag_kind = None
            self._drag_polygon_id = None
            self._drag_vertex_index = None
            self._drag_edge_index = None
            self._drag_origin_points = None
            self._drag_start_scene_pos = None
            self._drag_polygons_snapshot = None
            self._drag_erases = False
            self._update_tool_cursors()
            event.accept()
            return
        super().tabletEvent(event)

    def contextMenuEvent(self, event: QContextMenuEvent | None) -> None:
        if event is None:
            return
        scene_pos = self.mapToScene(event.pos())
        polygon_id = self._editor_scene.polygon_at(scene_pos)
        polygon = self._editor_scene.polygon_snapshot(polygon_id) if polygon_id is not None else None
        if polygon is None or (polygon.category != "via" and polygon.shape_hint != "box"):
            event.ignore()
            return
        if self.vector_edits_locked() or not self._editor_scene.polygon_is_deletable(polygon):
            event.accept()
            return
        language = getattr(self._editor_scene, "_ui_language", "en")
        delete_label = (
            "Удалить переходное отверстие" if language == "ru" else "Delete transition hole"
        )
        menu = QMenu(self)
        delete_action = menu.addAction(delete_label)
        chosen = menu.exec(event.globalPos())
        if chosen == delete_action:
            self.contactDeletionStarted.emit(1)
            deleted = self._editor_scene.delete_via_at(scene_pos)
            self.contactDeletionFinished.emit(1 if deleted else 0)
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            neighbor_path = self._editor_scene.neighbor_frame_path_at(self.mapToScene(event.position().toPoint()))
            if neighbor_path:
                if self._confirm_frame_navigation():
                    self.neighborFrameActivated.emit(neighbor_path)
                event.accept()
                return
        if (
            self._tool == EditorTool.ADD_POLYGON
            and self._effective_polygon_create_mode() == PolygonCreateMode.RECTANGLE
            and self._drag_kind == "rect_polygon"
            and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)
        ):
            started_with_right = self._drag_erases
            clicked_right = event.button() == Qt.MouseButton.RightButton
            if clicked_right == started_with_right:
                self._commit_rectangle_polygon(self.mapToScene(event.position().toPoint()))
            else:
                self._cancel_rectangle_polygon()
            event.accept()
            return
        if (
            self._tool in (EditorTool.ADD_POLYGON, EditorTool.TRACE_PEN)
            and self._editor_scene.has_pending_polygon()
            and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)
        ):
            if self._tool == EditorTool.TRACE_PEN:
                self._finish_pending_trace()
            else:
                self._finish_pending_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)
