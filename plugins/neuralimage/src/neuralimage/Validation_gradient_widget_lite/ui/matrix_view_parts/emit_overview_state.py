"""matrix_view: emit overview state."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QImage, QPixmap, QTransform
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem,
)
from ...core.domain import FrameRecord
from neuralimage.Validation_gradient_widget_lite.ui.ui_constants import (
    MATRIX_BACKGROUND_ALT,
    NORMALIZATION_EPSILON,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.module_state import (
    LOW_ZOOM_OVERVIEW_MAX_ZOOM,
    LOW_ZOOM_OVERVIEW_RECORD_THRESHOLD,
)

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.matrix_cell_item import _MatrixCellItem


class EmitOverviewStateMixin:
    def _build_overview_image(self, placements: list[tuple[FrameRecord, int, int]]) -> QImage:
        width = max(1, int(self._columns))
        height = max(1, int(self._rows))
        background = MATRIX_BACKGROUND_ALT
        pixels = np.empty((height, width, 3), dtype=np.uint8)
        pixels[:, :, 0] = int(background.red())
        pixels[:, :, 1] = int(background.green())
        pixels[:, :, 2] = int(background.blue())
        for record, row, column in placements:
            if 0 <= int(row) < height and 0 <= int(column) < width:
                color = self._background_color_for_record(record)
                pixels[int(row), int(column)] = (int(color.red()), int(color.green()), int(color.blue()))
        return QImage(pixels.data, width, height, int(pixels.strides[0]), QImage.Format.Format_RGB888).copy()

    def _matrix_scene_rect(self) -> QRectF:
        matrix_width = self._columns * (self._cell_size + self._gap)
        matrix_height = self._rows * (self._cell_size + self._gap)
        return QRectF(self._scene_padding, self._scene_padding, matrix_width, matrix_height)

    def _refresh_overview_layer_pixmap(self) -> None:
        if self._overview_image is None or self._overview_image.isNull() or self._columns <= 0 or self._rows <= 0:
            self._overview_layer_item = None
            return
        matrix_rect = self._matrix_scene_rect()
        pixmap = QPixmap.fromImage(self._overview_image)
        if self._overview_layer_item is None:
            item = QGraphicsPixmapItem()
            item.setTransformationMode(Qt.TransformationMode.FastTransformation)
            item.setShapeMode(QGraphicsPixmapItem.ShapeMode.BoundingRectShape)
            item.setZValue(-10.0)
            self._scene.addItem(item)
            self._overview_layer_item = item
        self._overview_layer_item.setPixmap(pixmap)
        self._overview_layer_item.setPos(matrix_rect.left(), matrix_rect.top())
        self._overview_layer_item.setTransform(
            QTransform.fromScale(
                matrix_rect.width() / max(1.0, float(pixmap.width())),
                matrix_rect.height() / max(1.0, float(pixmap.height())),
            )
        )
        self._overview_layer_item.setVisible(self._overview_layer_visible)

    def _overview_overlay_keys(self) -> set[str]:
        keys: set[str] = set(self._processing_keys)
        if self._selected_item is not None:
            keys.add(str(self._selected_item.record.key))
        if self._hovered_item is not None:
            keys.add(str(self._hovered_item.record.key))
        if self._reference_key:
            keys.add(str(self._reference_key))
        return keys

    def _overview_layer_should_be_active(self) -> bool:
        if self._overview_image is None or self._overview_image.isNull():
            return False
        if len(self._records) < LOW_ZOOM_OVERVIEW_RECORD_THRESHOLD:
            return False
        zoom_level = max(0.01, abs(float(self.transform().m11())))
        return zoom_level <= LOW_ZOOM_OVERVIEW_MAX_ZOOM

    def _sync_overview_layer_visibility(self, *, force: bool = False) -> bool:
        use_overview = self._overview_layer_should_be_active()
        previous = self._overview_layer_visible
        self._overview_layer_visible = use_overview
        if self._overview_layer_item is not None and (force or use_overview != previous or self._overview_layer_item.isVisible() != use_overview):
            self._overview_layer_item.setVisible(use_overview)
        if force or use_overview != previous:
            overlay_keys = self._overview_overlay_keys() if use_overview else set()
            for key, item in self._item_by_key.items():
                item.setVisible((not use_overview) or (key in overlay_keys))
        return use_overview != previous

    def _sync_low_zoom_visibility_for_keys(self, keys) -> None:
        if not self._overview_layer_visible:
            return
        overlay_keys = self._overview_overlay_keys()
        normalized_keys = {str(key) for key in keys if key}
        if not normalized_keys:
            return
        for key in normalized_keys:
            item = self._ensure_item_for_key(key) if key in overlay_keys else self._item_by_key.get(key)
            if item is not None:
                item.setVisible(key in overlay_keys)

    def _item_from_scene_pos(self, scene_pos) -> _MatrixCellItem | None:
        x = float(scene_pos.x()) - float(self._scene_padding)
        y = float(scene_pos.y()) - float(self._scene_padding)
        if x < 0.0 or y < 0.0:
            return None
        span = float(self._cell_size + self._gap)
        if span <= 0.0:
            return None
        column = int(x // span)
        row = int(y // span)
        if row < 0 or column < 0 or row >= self._rows or column >= self._columns:
            return None
        local_x = x - float(column) * span
        local_y = y - float(row) * span
        if local_x > float(self._cell_size) or local_y > float(self._cell_size):
            return None
        record = self._record_by_position.get((row, column))
        if record is None:
            return None
        return self._ensure_item_for_key(str(record.key))

    def _item_for_view_pos(self, view_pos) -> _MatrixCellItem | None:
        item = self.itemAt(view_pos)
        if isinstance(item, _MatrixCellItem):
            return item
        return self._item_from_scene_pos(self.mapToScene(view_pos))

    def _emit_overview_state(self) -> None:
        if self._overview_image is None or self._columns <= 0 or self._rows <= 0:
            self.overviewChanged.emit(None, QRectF(), None, False, tuple(), None)
            return
        scene_rect = self._scene.sceneRect()
        visible_scene_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        selected_position = None if self._selected_item is None else (self._selected_item.row, self._selected_item.column)
        processing_positions = tuple(self._record_positions[key] for key in self._processing_keys if key in self._record_positions)
        reference_position = self._record_positions.get(self._reference_key) if self._reference_key in self._record_positions else None
        if scene_rect.width() <= 0 or scene_rect.height() <= 0:
            self.overviewChanged.emit(self._overview_image, QRectF(), selected_position, self._selection_blink_on, processing_positions, reference_position)
            return
        left_padding = self._scene_padding / max(1.0, scene_rect.width())
        top_padding = self._scene_padding / max(1.0, scene_rect.height())
        width_padding = (self._scene_padding * 2) / max(1.0, scene_rect.width())
        height_padding = (self._scene_padding * 2) / max(1.0, scene_rect.height())
        normalized = QRectF(
            max(0.0, (visible_scene_rect.left() / scene_rect.width() - left_padding) / max(NORMALIZATION_EPSILON, 1.0 - width_padding)),
            max(0.0, (visible_scene_rect.top() / scene_rect.height() - top_padding) / max(NORMALIZATION_EPSILON, 1.0 - height_padding)),
            min(1.0, (visible_scene_rect.width() / scene_rect.width()) / max(NORMALIZATION_EPSILON, 1.0 - width_padding)),
            min(1.0, (visible_scene_rect.height() / scene_rect.height()) / max(NORMALIZATION_EPSILON, 1.0 - height_padding)),
        )
        self.overviewChanged.emit(self._overview_image, normalized, selected_position, self._selection_blink_on, processing_positions, reference_position)

    def _select_item(self, item: _MatrixCellItem) -> None:
        if self._selected_item is item:
            return
        previous = self._selected_item
        self._selected_item = item
        self._selection_blink_on = True
        if previous is not None:
            self._apply_item_style(previous)
        self._apply_item_style(item)
        self._sync_low_zoom_visibility_for_keys({None if previous is None else previous.record.key, item.record.key})
        self._emit_overview_state()

    def _clear_tile_selection(self) -> None:
        if self._selected_subpixel_selection is not None:
            previous = self._selected_subpixel_selection
            self._selected_subpixel_selection = None
            self._sync_tile_state_for_keys({previous.record.key})
        if self._hovered_subpixel_selection is not None:
            previous_hover = self._hovered_subpixel_selection
            self._hovered_subpixel_selection = None
            self._sync_tile_state_for_keys({previous_hover.record.key})

    def _select_tile_selection(self, selection: MatrixTileSelection) -> None:
        previous = self._selected_subpixel_selection
        self._selected_subpixel_selection = selection
        keys = {selection.record.key}
        if previous is not None:
            keys.add(previous.record.key)
        self._select_item(self._item_by_key[selection.record.key])
        self._sync_tile_state_for_keys(keys)
        self._emit_overview_state()
        self.tileSelected.emit(selection)

    def _set_hover_tile_selection(self, selection: MatrixTileSelection | None) -> None:
        previous = self._hovered_subpixel_selection
        if previous is selection:
            return
        self._hovered_subpixel_selection = selection
        keys = set()
        if previous is not None:
            keys.add(previous.record.key)
        if selection is not None:
            keys.add(selection.record.key)
        if keys:
            self._sync_tile_state_for_keys(keys)

    def _tile_hover_text(self, selection: MatrixTileSelection) -> str:
        payload = selection
        return (
            f"{payload.record.display_name} | parent r{payload.matrix_row + 1}, c{payload.matrix_column + 1}"
            f" | subpixel r{payload.sub_row + 1}, c{payload.sub_column + 1}"
            f" | value {payload.subpixel_value:.4f}"
            f" | parent {payload.parent_value:.4f}"
        )
