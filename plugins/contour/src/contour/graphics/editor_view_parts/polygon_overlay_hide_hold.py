"""editor_view: polygon overlay hide hold."""
from __future__ import annotations
from PyQt6.QtCore import (
    QRectF,
    Qt,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import QGraphicsPathItem
from contour.graphics.geometry import (
    _polygons_center,
)
from contour.graphics.tool_mode_logic import (
    is_via_polygon,
)
from contour.graphics.viewport_navigation import (
    DEFAULT_ZOOM_STEP_FACTOR,
)


class PolygonOverlayHideHoldMixin:
    def set_image_region_selection_mode(self, enabled: bool) -> None:
        self._image_region_selection_mode = bool(enabled)
        if not enabled:
            self._editor_scene.clear_preview_rect()

    def set_ctrl_image_region_selection_enabled(self, enabled: bool) -> None:
        self._ctrl_image_region_selection_enabled = bool(enabled)

    def fit_to_view(self) -> None:
        rect = self._editor_scene.main_image_rect()
        if rect.width() > 0 and rect.height() > 0:
            self._stop_zoom_animation()
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._clamp_current_zoom_at_viewport_pixel(self._require_viewport().rect().center())
            self._update_navigation_scene_rect()
            self.zoomChanged.emit(self.zoom_factor())

    def _set_polygon_overlay_hide_hold(
        self,
        source: str,
        active: bool,
        *,
        hide_gradient_overlay: bool = True,
    ) -> None:
        if active:
            if source in self._polygon_overlay_hide_holds:
                return
            if not self._polygon_overlay_hide_holds:
                self._polygon_overlays_visible_before_holds = self._editor_scene.polygon_overlays_visible()
            self._polygon_overlay_hide_holds.add(source)
            self._editor_scene.set_polygon_overlays_visible(False)
            if not hide_gradient_overlay:
                return
            if not self._gradient_overlay_hide_holds:
                self._gradient_overlay_visible_before_holds = (
                    self._editor_scene.gradient_overlay_user_visible()
                )
            self._gradient_overlay_hide_holds.add(source)
            self._editor_scene.set_gradient_overlay_visible(False)
            return
        if source not in self._polygon_overlay_hide_holds:
            return
        self._polygon_overlay_hide_holds.discard(source)
        if not self._polygon_overlay_hide_holds:
            if self._polygon_overlays_visible_before_holds is not None:
                self._editor_scene.set_polygon_overlays_visible(self._polygon_overlays_visible_before_holds)
            self._polygon_overlays_visible_before_holds = None
        self._gradient_overlay_hide_holds.discard(source)
        if not self._gradient_overlay_hide_holds:
            if self._gradient_overlay_visible_before_holds is not None:
                self._editor_scene.set_gradient_overlay_visible(
                    self._gradient_overlay_visible_before_holds
                )
            self._gradient_overlay_visible_before_holds = None

    def center_main_image(self) -> None:
        rect = self._editor_scene.main_image_rect()
        if rect.width() > 0 and rect.height() > 0:
            self.centerOn(rect.center())
            self._update_navigation_scene_rect()

    def main_image_visible_fraction(self) -> float:
        rect = self._editor_scene.main_image_rect()
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return 1.0
        viewport_scene = self.mapToScene(self._require_viewport().rect()).boundingRect()
        intersection = rect.intersected(viewport_scene)
        if intersection.isEmpty():
            return 0.0
        image_area = float(rect.width()) * float(rect.height())
        visible_area = float(intersection.width()) * float(intersection.height())
        return max(0.0, min(1.0, visible_area / image_area))

    def should_auto_reposition_view(self, *, force: bool = False) -> bool:
        if force:
            return True
        return self.main_image_visible_fraction() < 0.5

    def zoom_in(self) -> None:
        self._start_zoom_animation(self._zoom_focus_viewport_pixel(), DEFAULT_ZOOM_STEP_FACTOR)

    def zoom_out(self) -> None:
        self._start_zoom_animation(self._zoom_focus_viewport_pixel(), 1.0 / DEFAULT_ZOOM_STEP_FACTOR)

    def undo(self) -> None:
        if self.vector_edits_locked():
            return
        can_undo = self.undo_stack.canUndo()
        contacts_before = self._editor_scene.contact_count()
        self.contactUndoStarted.emit()
        self.undo_stack.undo()
        contacts_changed = abs(self._editor_scene.contact_count() - contacts_before)
        self.contactUndoFinished.emit(can_undo, contacts_changed)

    def redo(self) -> None:
        if self.vector_edits_locked():
            return
        can_redo = self.undo_stack.canRedo()
        contacts_before = self._editor_scene.contact_count()
        self.contactRedoStarted.emit()
        self.undo_stack.redo()
        contacts_changed = abs(self._editor_scene.contact_count() - contacts_before)
        self.contactRedoFinished.emit(can_redo, contacts_changed)

    def copy_selected(self) -> None:
        self.contactCopyStarted.emit()
        polygons = self._editor_scene.selected_polygons()
        if not polygons:
            self.contactCopyFinished.emit(0)
            return
        self._clipboard_polygons = [polygon.clone() for polygon in polygons]
        self._clipboard_anchor = _polygons_center(self._clipboard_polygons)
        self.contactCopyFinished.emit(len(polygons))
        self.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.start_paste_mode()
        if self._paste_mode:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def cut_selected(self) -> None:
        if self.vector_edits_locked():
            return
        polygons = self._editor_scene.selected_deletable_polygons()
        if polygons:
            contacts = [polygon for polygon in polygons if is_via_polygon(polygon)]
            if contacts:
                self.contactDeletionStarted.emit(len(contacts))
            self._clipboard_polygons = [polygon.clone() for polygon in polygons]
            self._clipboard_anchor = _polygons_center(self._clipboard_polygons)
            deleted = self._editor_scene.delete_polygon()
            if contacts:
                self.contactDeletionFinished.emit(len(contacts) if deleted else 0)

    def start_paste_mode(self) -> None:
        if self.vector_edits_locked():
            self.contactPasteFinished.emit(0)
            return
        if self._paste_mode:
            self._update_paste_preview(
                self._last_pointer_scene_pos or self.mapToScene(self._require_viewport().rect().center())
            )
            return
        self.contactPasteStarted.emit(len(self._clipboard_polygons))
        if not self._clipboard_polygons or not self._editor_scene.can_add_polygon_set(self._clipboard_polygons):
            self.contactPasteFinished.emit(0)
            return
        self._paste_mode = True
        self._rebuild_paste_preview()
        self._update_paste_preview(
            self._last_pointer_scene_pos or self.mapToScene(self._require_viewport().rect().center())
        )

    def _clear_paste_preview(self) -> None:
        for item in self._paste_preview_items:
            if item.scene() is not None:
                self._editor_scene.removeItem(item)
        self._paste_preview_items.clear()

    def _exit_paste_mode(self) -> None:
        self._paste_mode = False
        self._clear_paste_preview()
        self.unsetCursor()

    def _rebuild_paste_preview(self) -> None:
        self._clear_paste_preview()
        if not self._paste_mode:
            return
        pen = QPen(QColor("#38BDF8"), 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        brush = QColor("#38BDF8")
        brush.setAlpha(42)
        for polygon in self._clipboard_polygons:
            path = QPainterPath()
            if polygon.shape_hint == "box" or polygon.category == "via":
                x_coord, y_coord, width, height = polygon.bbox
                path.addEllipse(
                    QRectF(
                        float(x_coord),
                        float(y_coord),
                        float(max(0, width - 1)),
                        float(max(0, height - 1)),
                    )
                )
            else:
                if polygon.points:
                    path.moveTo(polygon.points[0][0], polygon.points[0][1])
                    for x_coord, y_coord in polygon.points[1:]:
                        path.lineTo(x_coord, y_coord)
                    path.closeSubpath()
            item = QGraphicsPathItem(path)
            item.setZValue(40)
            item.setPen(pen)
            item.setBrush(QBrush(brush))
            self._editor_scene.addItem(item)
            self._paste_preview_items.append(item)
