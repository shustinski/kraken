"""editor_scene: debug candidates."""
from __future__ import annotations
from math import hypot
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QGraphicsPathItem,
)
from ...adapters.qt.image_conversion import cv_to_qimage
from contour.domain import PolygonData
from contour.graphics_items import (
    VectorPolygonDisplayItem,
    _display_path_for_polygon,
)

from contour.graphics.editor_scene_parts.gradient_arrows_path import _gradient_arrows_path


class DebugCandidatesMixin:
    def neighbor_frame_path_at(self, scene_pos: QPointF) -> str | None:
        x_coord = float(scene_pos.x())
        y_coord = float(scene_pos.y())
        if self._neighbor_frame_bounds:
            main_rect = QRectF(self._image_rect)
            candidates: list[tuple[str, QRectF]] = []
            for image_path, frame_rect in self._neighbor_frame_bounds:
                if frame_rect.contains(x_coord, y_coord):
                    candidates.append((image_path, frame_rect))
            if not candidates:
                return None
            if len(candidates) == 1:
                return candidates[0][0]

            def _distance_to_center(item: tuple[str, QRectF]) -> float:
                center = item[1].center()
                return hypot(x_coord - center.x(), y_coord - center.y())

            if main_rect.contains(x_coord, y_coord):
                return min(candidates, key=_distance_to_center)[0]
            return min(candidates, key=_distance_to_center)[0]

        for item in self.items(scene_pos):
            if item in self._neighbor_frame_paths:
                return self._neighbor_frame_paths[item]
        return None

    def _neighbor_vector_item(
        self,
        polygon: PolygonData,
        image_path: str,
        opacity: float,
        cutout_polygons: list[PolygonData] | None = None,
    ) -> QGraphicsPathItem:
        path_item = VectorPolygonDisplayItem(
            polygon,
            self._display_settings,
            cutout_polygons=cutout_polygons,
            opacity=opacity,
            tooltip=image_path,
        )
        path_item.setZValue(-19)
        return path_item

    def set_debug_candidates(self, candidates: list[object]) -> None:
        self._recycled_debug_cleanup_timer.stop()
        for item in self._debug_candidate_items:
            item.setVisible(False)
            if isinstance(item, QGraphicsPathItem):
                self._recycled_debug_candidate_items.append(item)
            else:
                self.removeItem(item)
        self._debug_candidate_items.clear()
        for candidate in candidates:
            if bool(getattr(candidate, "accepted", False)):
                continue
            bbox = getattr(candidate, "bbox", (0, 0, 0, 0))
            if not isinstance(bbox, (tuple, list)) or len(bbox) < 4:
                continue
            x_coord, y_coord, width, height = (float(value) for value in bbox[:4])
            if width <= 0.0 or height <= 0.0:
                continue

            reason = str(getattr(candidate, "reason", "") or "")
            score = max(0.0, min(100.0, float(getattr(candidate, "score", 0.0) or 0.0)))
            below_threshold = "below_threshold" in reason
            color = QColor("#F59E0B" if below_threshold else "#EF4444")
            rect = QRectF(x_coord, y_coord, width, height)
            path = QPainterPath()
            path.addEllipse(rect)
            if self._recycled_debug_candidate_items:
                path_item = self._recycled_debug_candidate_items.pop()
                path_item.setPath(path)
            else:
                path_item = QGraphicsPathItem(path)
                self.addItem(path_item)
            path_item.setZValue(4.5)
            pen = QPen(color, 2.0, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            path_item.setPen(pen)
            fill = QColor(color)
            fill.setAlpha(36)
            path_item.setBrush(QBrush(fill))
            status = "Ниже порога" if below_threshold else "Отклонён"
            if self._ui_language != "ru":
                status = "Below threshold" if below_threshold else "Rejected"
            path_item.setToolTip(f"{status}\n{reason or '-'}\nScore: {score:.1f}")
            path_item.setVisible(True)
            self._debug_candidate_items.append(path_item)
        if self._recycled_debug_candidate_items:
            self._recycled_debug_cleanup_timer.start()

    def _drain_recycled_debug_candidate_items(self) -> None:
        batch_size = min(32, len(self._recycled_debug_candidate_items))
        for _index in range(batch_size):
            self.removeItem(self._recycled_debug_candidate_items.pop())
        if not self._recycled_debug_candidate_items:
            self._recycled_debug_cleanup_timer.stop()

    def set_metal_overlays(
        self,
        layers: dict[str, list[PolygonData]],
        visibility: dict[str, bool],
    ) -> None:
        for item in self._metal_overlay_items:
            self.removeItem(item)
        self._metal_overlay_items.clear()
        layer_styles: list[tuple[str, str, bool]] = [
            ("rejected", "#EF4444", False),
            ("suspicious", "#EAB308", False),
            ("border", "#3B82F6", False),
            ("wide_pairs_suspicious", "#EAB308", True),
            ("wide_pairs_rejected", "#DC2626", True),
        ]
        z = 2.2
        _ru = self._ui_language == "ru"
        _layer_tip = {
            "rejected": "Отклонён" if _ru else "Rejected",
            "suspicious": "Сомнительный" if _ru else "Suspicious",
            "border": "У границы кадра" if _ru else "Border touch",
            "wide_pairs_suspicious": "Широкий проводник (сомнительно)" if _ru else "Wide trace (suspicious)",
            "wide_pairs_rejected": "Широкий проводник (отклонён)" if _ru else "Wide trace (rejected)",
        }
        for key, color_hex, dashed in layer_styles:
            if not visibility.get(key, False):
                continue
            for poly in layers.get(key) or []:
                path_item = QGraphicsPathItem(_display_path_for_polygon(poly, self._display_settings))
                path_item.setZValue(z)
                c = QColor(color_hex)
                pen = QPen(c, 1.75)
                pen.setCosmetic(True)
                if dashed:
                    pen.setStyle(Qt.PenStyle.DashLine)
                path_item.setPen(pen)
                path_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                reason = str(getattr(poly, "reject_reason", "") or "")
                cap = _layer_tip.get(key, key)
                tip_lines = [cap]
                if reason.strip():
                    tip_lines.append(reason)
                path_item.setToolTip("\n".join(tip_lines))
                path_item.setData(int(Qt.ItemDataRole.UserRole), key)
                path_item.setData(int(Qt.ItemDataRole.UserRole) + 1, reason)
                self.addItem(path_item)
                self._metal_overlay_items.append(path_item)

    def metal_overlay_pick(self, scene_pos: QPointF) -> tuple[str, str] | None:
        """Return ``(layer_key, reject_reason)`` for the topmost metal overlay under ``scene_pos``."""

        for item in self.items(scene_pos):
            if item in self._metal_overlay_items:
                layer = item.data(int(Qt.ItemDataRole.UserRole))
                if layer is None:
                    continue
                reason = item.data(int(Qt.ItemDataRole.UserRole) + 1)
                return (str(layer), str(reason or ""))
        return None

    def set_gradient_overlay(self, image, opacity: float = 0.45) -> None:
        self.clear_gradient_field_arrows()
        if image is None:
            self._gradient_overlay_item.setPixmap(QPixmap())
            self._gradient_overlay_item.hide()
            return
        pixmap = QPixmap.fromImage(cv_to_qimage(image))
        if pixmap.isNull():
            self._gradient_overlay_item.setPixmap(QPixmap())
            self._gradient_overlay_item.hide()
            return
        self._gradient_overlay_item.setPixmap(pixmap)
        self._gradient_overlay_item.setOpacity(max(0.0, min(1.0, float(opacity))))
        self._gradient_overlay_item.setPos(0.0, 0.0)
        self._sync_gradient_overlay_visibility()

    def clear_gradient_overlay(self) -> None:
        self._gradient_overlay_item.setPixmap(QPixmap())
        self._gradient_overlay_item.hide()
        self.clear_gradient_field_arrows()

    def set_gradient_field_arrows(self, arrows: list[tuple[float, float, float, float]]) -> None:
        path = _gradient_arrows_path(arrows)
        self._gradient_arrows_item.setPath(path)
        self._gradient_arrows_has_content = bool(arrows) and not path.isEmpty()
        self._sync_gradient_overlay_visibility()
