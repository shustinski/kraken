"""editor_scene: extra layers."""
from __future__ import annotations
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import (
    QPainterPath,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem,
)
from ...application.processing import DisplaySettings
from contour.i18n import active_language, tr
from contour.graphics.polygon_creation import (
    POLYGON_COMMIT_TOO_FEW_VERTICES,
    POLYGON_COMMIT_TOO_SMALL_AREA,
)


class ExtraLayersMixin:
    def clear_gradient_field_arrows(self) -> None:
        if self._gradient_arrows_has_content or not self._gradient_arrows_item.path().isEmpty():
            self._gradient_arrows_item.setPath(QPainterPath())
        self._gradient_arrows_has_content = False
        self._gradient_arrows_item.hide()

    def set_gradient_overlay_opacity(self, opacity: float) -> None:
        self._gradient_overlay_item.setOpacity(max(0.0, min(1.0, float(opacity))))

    def set_gradient_overlay_visible(self, visible: bool) -> None:
        self._gradient_overlay_user_visible = bool(visible)
        self._sync_gradient_overlay_visibility()

    def gradient_overlay_user_visible(self) -> bool:
        return self._gradient_overlay_user_visible

    def _sync_gradient_overlay_visibility(self) -> None:
        has_content = not self._gradient_overlay_item.pixmap().isNull()
        self._gradient_overlay_item.setVisible(self._gradient_overlay_user_visible and has_content)
        self._gradient_arrows_item.setVisible(self._gradient_overlay_user_visible and self._gradient_arrows_has_content)

    def _update_main_frame(self) -> None:
        path = QPainterPath()
        path.addRect(self._image_rect)
        self._main_frame_item.setPath(path)

    def _update_scene_rect(self) -> None:
        rect = QRectF(self._image_rect)
        if self._neighbor_grid_bounds is not None:
            rect = rect.united(self._neighbor_grid_bounds)
        for item in self._neighbor_frame_items:
            rect = rect.united(item.sceneBoundingRect())
        self.setSceneRect(rect)

    def set_ui_language(self, language: str | None) -> None:
        self._ui_language = active_language(language)

    def warn_invalid_polygon_geometry(self) -> None:
        self.logRequested.emit(tr("polygon_invalid_geometry_log", language=self._ui_language))

    def _log_polygon_commit_rejection(self, reason: str | None) -> None:
        if reason == POLYGON_COMMIT_TOO_SMALL_AREA:
            self.logRequested.emit(tr("polygon_too_small_area_commit_log", language=self._ui_language))
            return
        if reason == POLYGON_COMMIT_TOO_FEW_VERTICES:
            self.logRequested.emit(tr("polygon_need_min_vertices_finish_log", language=self._ui_language))

    def set_display_settings(self, settings: DisplaySettings) -> None:
        self._display_settings = settings
        self._refresh_all_items()

    def set_random_object_colors_enabled(self, enabled: bool) -> None:
        self._random_object_colors_enabled = bool(enabled)
        self._refresh_all_items()

    def set_extra_layers(self, layers: list[dict[str, object]]) -> None:
        for item in self._extra_layer_items:
            self.removeItem(item)
        self._extra_layer_items.clear()
        for index, layer in enumerate(layers):
            if not bool(layer.get("visible", True)):
                continue
            pixmap = layer.get("pixmap")
            if not isinstance(pixmap, QPixmap) or pixmap.isNull():
                continue
            dx = float(layer.get("dx", 0.0) or 0.0)
            dy = float(layer.get("dy", 0.0) or 0.0)
            try:
                opacity = float(layer.get("opacity", 1.0))
            except (TypeError, ValueError):
                opacity = 1.0
            opacity = max(0.0, min(1.0, opacity))
            item = QGraphicsPixmapItem(pixmap)
            item.setZValue(0.8 + index * 0.001)
            item.setOpacity(opacity)
            item.setPos(dx, dy)
            item.setToolTip(str(layer.get("name", "")))
            self.addItem(item)
            self._extra_layer_items.append(item)

    def set_polygon_overlays_visible(self, visible: bool) -> None:
        self._polygon_overlays_visible = bool(visible)
        for polygon_id, item in self._polygon_items.items():
            poly = self._polygons[polygon_id]
            cat = str(getattr(poly, "category", "") or "")
            vis = self._polygon_category_visible.get(cat, True)
            item.setVisible(
                self._polygon_overlays_visible
                and vis
                and polygon_id not in self._zoom_hidden_contact_ids
            )
        zoom_contacts_visible = (
            self._polygon_overlays_visible
            and self._polygon_category_visible.get("via", True)
        )
        for item in self._zoom_contact_composite_items:
            item.setVisible(zoom_contacts_visible)
        self._pending_path_item.setVisible(self._polygon_overlays_visible)
        self._preview_rect_item.setVisible(self._polygon_overlays_visible)
