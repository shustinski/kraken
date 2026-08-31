"""Viewport overlay widget for the polygon editor minimap."""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPaintEvent, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

from .minimap_geometry import (
    fitted_minimap_size,
    has_usable_image_size,
    image_rect_in_minimap,
    minimap_point_to_scene,
    viewport_frame_in_minimap,
)

_BACKGROUND = QColor("#1F2937")
_BORDER = QColor("#4B5563")
_FRAME_FILL = QColor(248, 113, 113, 55)
_FRAME_PEN = QColor("#FECACA")


class MinimapWidget(QWidget):
    scenePointRequested = pyqtSignal(QPointF)
    sceneDeltaRequested = pyqtSignal(QPointF)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("editorMinimap")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(False)
        self._thumbnail = QPixmap()
        self._image_scene_rect = (0.0, 0.0, 0.0, 0.0)
        self._minimap_image_rect = (0.0, 0.0, 0.0, 0.0)
        self._viewport_frame: tuple[float, float, float, float] | None = None
        self._drag_active = False
        self._last_scene_point: tuple[float, float] | None = None
        self.hide()

    def has_image(self) -> bool:
        return not self._thumbnail.isNull()

    def set_mouse_interactive(self, interactive: bool) -> None:
        transparent = not interactive
        if self.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents) == transparent:
            return
        if transparent:
            self._drag_active = False
            self._last_scene_point = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, transparent)

    def clear(self) -> None:
        self._thumbnail = QPixmap()
        self._image_scene_rect = (0.0, 0.0, 0.0, 0.0)
        self._minimap_image_rect = (0.0, 0.0, 0.0, 0.0)
        self._viewport_frame = None
        self._drag_active = False
        self._last_scene_point = None
        self.hide()

    def set_image(self, pixmap: QPixmap, image_scene_rect: QRectF) -> None:
        if pixmap.isNull() or not has_usable_image_size(image_scene_rect.width(), image_scene_rect.height()):
            self.clear()
            return
        self._image_scene_rect = (
            float(image_scene_rect.x()),
            float(image_scene_rect.y()),
            float(image_scene_rect.width()),
            float(image_scene_rect.height()),
        )
        fitted_width, fitted_height = fitted_minimap_size(
            self._image_scene_rect[2],
            self._image_scene_rect[3],
        )
        self.setFixedSize(max(1, round(fitted_width)), max(1, round(fitted_height)))
        self._minimap_image_rect = image_rect_in_minimap(
            (self._image_scene_rect[2], self._image_scene_rect[3]),
            (float(self.width()), float(self.height())),
        )
        _offset_x, _offset_y, drawn_width, drawn_height = self._minimap_image_rect
        self._thumbnail = pixmap.scaled(
            max(1, round(drawn_width)),
            max(1, round(drawn_height)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.update()

    def set_viewport_scene_rect(self, viewport_scene_rect: QRectF) -> None:
        if not self.has_image():
            self._viewport_frame = None
            return
        frame = viewport_frame_in_minimap(
            self._image_scene_rect,
            (
                float(viewport_scene_rect.x()),
                float(viewport_scene_rect.y()),
                float(viewport_scene_rect.width()),
                float(viewport_scene_rect.height()),
            ),
            self._minimap_image_rect,
        )
        self._viewport_frame = frame
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), _BACKGROUND)
        if not self._thumbnail.isNull():
            offset_x, offset_y, drawn_width, drawn_height = self._minimap_image_rect
            target = QRect(round(offset_x), round(offset_y), round(drawn_width), round(drawn_height))
            painter.drawPixmap(target, self._thumbnail)
        painter.setPen(QPen(_BORDER, 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if self._viewport_frame is not None:
            frame_x, frame_y, frame_width, frame_height = self._viewport_frame
            painter.setPen(QPen(_FRAME_PEN, 1.25))
            painter.setBrush(QBrush(_FRAME_FILL))
            painter.drawRect(QRectF(frame_x, frame_y, frame_width, frame_height))

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None or event.button() != Qt.MouseButton.LeftButton:
            return
        scene_point = self._scene_point_at(event.position())
        if scene_point is None:
            event.accept()
            return
        self._drag_active = True
        self._last_scene_point = scene_point
        self.scenePointRequested.emit(QPointF(scene_point[0], scene_point[1]))
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None or not self._drag_active or self._last_scene_point is None:
            return
        scene_point = self._scene_point_at(event.position())
        if scene_point is None:
            event.accept()
            return
        delta_x = scene_point[0] - self._last_scene_point[0]
        delta_y = scene_point[1] - self._last_scene_point[1]
        self._last_scene_point = scene_point
        self.sceneDeltaRequested.emit(QPointF(delta_x, delta_y))
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        self._drag_active = False
        self._last_scene_point = None
        event.accept()

    def _scene_point_at(self, widget_pos: QPointF) -> tuple[float, float] | None:
        if not self.has_image():
            return None
        offset_x, offset_y, drawn_width, drawn_height = self._minimap_image_rect
        if drawn_width <= 0.0 or drawn_height <= 0.0:
            return None
        clamped_x = min(max(widget_pos.x(), offset_x), offset_x + drawn_width)
        clamped_y = min(max(widget_pos.y(), offset_y), offset_y + drawn_height)
        return minimap_point_to_scene(
            (clamped_x, clamped_y),
            self._image_scene_rect,
            self._minimap_image_rect,
        )
