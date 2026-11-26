"""qt: frame matrix minimap."""

from __future__ import annotations
from typing import TYPE_CHECKING
import weakref
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QWidget,
)


class FrameMatrixMinimap(QWidget):
    """Compact matrix overview with viewport indication and click navigation."""

    WIDTH = 180
    HEIGHT = 120
    EDGE_MARGIN = 12
    CONTENT_MARGIN = 8

    def __init__(self, matrix_view: FrameMatrixView) -> None:
        super().__init__(matrix_view.viewport())
        self._matrix_view = weakref.proxy(matrix_view)
        self.setObjectName("matrixMinimap")
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Нажмите или перетащите, чтобы перейти к нужной области")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        matrix_view.viewportChanged.connect(self._viewport_changed)
        self._content_update_timer = QTimer(self)
        self._content_update_timer.setSingleShot(True)
        self._content_update_timer.setInterval(100)
        self._content_update_timer.timeout.connect(self.update)
        matrix_view.cellsChanged.connect(self._schedule_content_update)
        self._reposition()
        self.show()
        self.raise_()

    def apply_kraken_scene_theme(self) -> None:
        self.update()

    def _schedule_content_update(self) -> None:
        if not self._content_update_timer.isActive():
            self._content_update_timer.start()

    def _viewport_changed(self, _visible_rect: QRectF) -> None:
        self._reposition()
        self.update()
        self.raise_()

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.move(
            max(0, parent.width() - self.width() - self.EDGE_MARGIN),
            max(0, parent.height() - self.height() - self.EDGE_MARGIN),
        )
        self.raise_()

    def _matrix_rect(self) -> QRectF:
        scene = self._matrix_view.sceneRect()
        content = QRectF(self.rect()).adjusted(
            self.CONTENT_MARGIN,
            self.CONTENT_MARGIN,
            -self.CONTENT_MARGIN,
            -self.CONTENT_MARGIN,
        )
        if scene.isEmpty():
            return content
        scale = min(content.width() / scene.width(), content.height() / scene.height())
        width = scene.width() * scale
        height = scene.height() * scale
        return QRectF(
            content.center().x() - width / 2,
            content.center().y() - height / 2,
            width,
            height,
        )

    def _map_scene_rect(self, scene_rect: QRectF) -> QRectF:
        scene = self._matrix_view.sceneRect()
        matrix = self._matrix_rect()
        if scene.isEmpty() or matrix.isEmpty():
            return QRectF()
        return QRectF(
            matrix.left() + (scene_rect.left() - scene.left()) / scene.width() * matrix.width(),
            matrix.top() + (scene_rect.top() - scene.top()) / scene.height() * matrix.height(),
            scene_rect.width() / scene.width() * matrix.width(),
            scene_rect.height() / scene.height() * matrix.height(),
        ).intersected(matrix)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        from kraken_core.theme import scene_color

        panel = QColor(scene_color("minimap"))
        panel.setAlpha(225)
        painter.setPen(QPen(QColor(scene_color("minimap_border")), 1.0))
        painter.setBrush(panel)
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)

        matrix = self._matrix_rect()
        painter.setPen(QPen(QColor(scene_color("minimap_border")), 1.0))
        painter.setBrush(QColor(scene_color("minimap_matrix")))
        painter.drawRect(matrix)

        scene = self._matrix_view.sceneRect()
        if not scene.isEmpty():
            painter.save()
            painter.setClipRect(matrix)
            for cell in self._matrix_view._cells.values():
                thumbnail = cell.thumbnail
                if thumbnail is None or thumbnail.isNull():
                    continue
                frame = self._matrix_view.scene_rect_for_frame(cell.x, cell.y)
                target = self._map_scene_rect(frame)
                if not target.isEmpty():
                    painter.drawPixmap(target, thumbnail, QRectF(thumbnail.rect()))
            for scene_rect, pixmap in self._layer_tiles():
                target = self._map_scene_rect(scene_rect)
                if not target.isEmpty():
                    painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))
            painter.restore()

        visible = self._map_scene_rect(self._matrix_view.visible_scene_rect())
        if not visible.isEmpty():
            painter.setPen(QPen(QColor("#60a5fa"), 1.5))
            painter.setBrush(QColor(37, 99, 235, 75))
            painter.drawRect(visible)
        painter.end()

    def _layer_tiles(self) -> tuple[tuple[QRectF, QPixmap], ...]:
        """Loaded mosaic tiles, coarse first, so the whole layer stays visible."""

        view = self._matrix_view
        images = getattr(view, "_mosaic_images", None)
        if not images:
            return ()
        orientation = view.orientation().value
        revision = getattr(view, "_mosaic_revision", None)
        chosen: dict[tuple[int, int, int], QPixmap] = {}
        for key, pixmap in images.items():
            if pixmap is None or pixmap.isNull() or len(key) < 5 or key[4] != orientation:
                continue
            slot = (int(key[0]), int(key[1]), int(key[2]))
            if slot not in chosen or key[3] == revision:
                chosen[slot] = pixmap
        tiles: list[tuple[QRectF, QPixmap]] = []
        pitch_x, pitch_y = view.cell_pitch()
        for tile_x, tile_y, cells in sorted(chosen, key=lambda slot: slot[2], reverse=True):
            left = tile_x * cells * pitch_x
            top = tile_y * cells * pitch_y
            columns = min(cells, view.matrix_width - tile_x * cells)
            rows = min(cells, view.matrix_height - tile_y * cells)
            if columns <= 0 or rows <= 0:
                continue
            tiles.append((QRectF(left, top, columns * pitch_x, rows * pitch_y), chosen[(tile_x, tile_y, cells)]))
        return tuple(tiles)

    def _navigate(self, position: QPointF) -> None:
        matrix = self._matrix_rect()
        scene = self._matrix_view.sceneRect()
        if matrix.isEmpty() or scene.isEmpty():
            return
        x = min(matrix.right(), max(matrix.left(), position.x()))
        y = min(matrix.bottom(), max(matrix.top(), position.y()))
        scene_x = scene.left() + (x - matrix.left()) / matrix.width() * scene.width()
        scene_y = scene.top() + (y - matrix.top()) / matrix.height() * scene.height()
        self._matrix_view.centerOn(scene_x, scene_y)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._navigate(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._navigate(event.position())
            event.accept()
            return
        super().mouseMoveEvent(event)


if TYPE_CHECKING:
    from kraken_core.frame_matrix.qt import FrameMatrixView
