"""editor_view: configure opengl viewport."""
from __future__ import annotations
from collections.abc import Callable
from PyQt6.QtGui import (
    QGuiApplication,
    QUndoStack,
)
from PyQt6.QtWidgets import QWidget
from contour.graphics.tools import EditorTool

from contour.graphics.editor_view_parts.module_state import (
    _OPENGL_DISABLED_PLATFORMS,
    _OPENGL_VIEWPORT_ENABLED,
)


class ConfigureOpenglViewportMixin:
    def _activate_tool_shortcut(self, tool: EditorTool) -> None:
        if tool not in self._available_tools:
            return
        if tool == EditorTool.ADD_VIA:
            self.contactPlacementHotkeyPressed.emit()
        self.set_tool(tool)

    def _configure_opengl_viewport(self) -> bool:
        if not _OPENGL_VIEWPORT_ENABLED:
            return False
        app = QGuiApplication.instance()
        platform = str(app.platformName()).lower() if app is not None else ""
        if platform in _OPENGL_DISABLED_PLATFORMS:
            return False
        try:
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget
        except Exception:
            return False
        try:
            viewport = QOpenGLWidget(self)
            viewport.setUpdateBehavior(QOpenGLWidget.UpdateBehavior.PartialUpdate)
            self.setViewport(viewport)
        except Exception:
            return False
        return True

    def _require_viewport(self) -> QWidget:
        viewport = self.viewport()
        if viewport is None:
            raise RuntimeError("Graphics view has no viewport")
        return viewport

    @property
    def undo_stack(self) -> QUndoStack:
        return self._editor_scene.undo_stack

    @property
    def current_tool(self) -> EditorTool:
        return self._tool

    def set_frame_navigation_guard(self, guard: Callable[[], bool] | None) -> None:
        self._frame_navigation_guard = guard

    def _confirm_frame_navigation(self) -> bool:
        if self._frame_navigation_guard is None:
            return True
        return bool(self._frame_navigation_guard())
