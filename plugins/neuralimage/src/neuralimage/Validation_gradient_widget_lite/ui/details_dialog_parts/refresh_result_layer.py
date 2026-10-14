"""details_dialog: refresh result layer."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from ...core.tile_grid import TileGridPlan, plan_tile_grid, tile_bounds_for_index


class RefreshResultLayerMixin:
    def _refresh_result_layer(self) -> None:
        if self._payload_loading and not self._payload:
            self._comparison_score = None
            self.result_item.setPixmap(QPixmap())
            self._update_layer_states()
            return
        _preset_key, first_key, second_key = self._current_comparison_tuple()
        mode = self._current_operation_mode()
        kind = self._selected_result_kind()
        confidence_model_id = self._confidence_model_id()
        if kind == "confidence":
            self._comparison_score = None
            self.result_item.setPixmap(self._confidence_overlay_pixmap(confidence_model_id))
        elif kind == "confidence_bad_areas":
            self._comparison_score = None
            self.result_item.setPixmap(self._confidence_bad_areas_pixmap(confidence_model_id))
        elif kind == "confidence_mix":
            self._comparison_score = None
            self.result_item.setPixmap(self._confidence_mix_pixmap(confidence_model_id))
        elif kind == "boundary":
            pixmap, score = self._boundary_overlay_result(first_key, second_key, self._named_color("boundary_mask"))
            self._comparison_score = float(score)
            self.result_item.setPixmap(pixmap)
        elif kind == "point_matches":
            self._comparison_score = None
            self.result_item.setPixmap(self._point_matches_pixmap(first_key, second_key))
        else:
            pixmap, score = self._comparison_overlay_result(first_key, second_key, mode, self._named_color("difference_mask"))
            self._comparison_score = float(score)
            self.result_item.setPixmap(pixmap)
        self._update_layer_states()

    def _tile_plan_for_view(self) -> TileGridPlan | None:
        if str(getattr(self._build_result.options, "tile_mode", "pixel") or "pixel") != "tile":
            return None
        base = self._full_base_array()
        if base.ndim != 2 or base.size == 0:
            return None
        return plan_tile_grid(
            base.shape,
            int(getattr(self._build_result.options, "tile_width", 256)),
            int(getattr(self._build_result.options, "tile_height", 256)),
            int(getattr(self._build_result.options, "tile_overlap", 0)),
        )

    def _tile_rect_for_index(self, row: int, column: int, plan: TileGridPlan) -> QRectF:
        left, top, width, height = tile_bounds_for_index(plan, row, column)
        return QRectF(float(left), float(top), float(width), float(height))

    def _tile_mosaic_pixmap(self, base_array: np.ndarray, plan: TileGridPlan) -> QPixmap:
        cache_key = (
            "tile_mosaic",
            id(base_array),
            int(plan.tile_width),
            int(plan.tile_height),
            int(plan.overlap),
            int(plan.rows),
            int(plan.columns),
        )
        cached = self._overlay_cache.get(cache_key)
        if isinstance(cached, QPixmap):
            return cached
        mosaic = np.ascontiguousarray(np.asarray(base_array, dtype=np.uint8)).copy()
        for row in range(int(plan.rows)):
            for column in range(int(plan.columns)):
                left, top, width, height = tile_bounds_for_index(plan, row, column)
                if width <= 0 or height <= 0:
                    continue
                tile = mosaic[top:top + height, left:left + width]
                if tile.size == 0:
                    continue
                mean_value = int(np.clip(np.rint(float(np.mean(tile, dtype=np.float64))), 0, 255))
                mosaic[top:top + height, left:left + width] = mean_value
        pixmap = self._grayscale_to_pixmap(mosaic)
        self._overlay_cache[cache_key] = pixmap
        return pixmap

    def _tile_grid_pixmap(self, plan: TileGridPlan) -> QPixmap:
        cache_key = (
            "tile_grid",
            int(plan.source_shape[0]),
            int(plan.source_shape[1]),
            int(plan.tile_width),
            int(plan.tile_height),
            int(plan.overlap),
            int(plan.rows),
            int(plan.columns),
        )
        cached = self._overlay_cache.get(cache_key)
        if isinstance(cached, QPixmap):
            return cached
        source_height, source_width = (int(plan.source_shape[0]), int(plan.source_shape[1]))
        image = QImage(max(1, source_width), max(1, source_height), QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setPen(QPen(QColor(92, 196, 255, 180), 1.0))
        for row in range(int(plan.rows)):
            for column in range(int(plan.columns)):
                rect = self._tile_rect_for_index(row, column, plan)
                if rect.isNull():
                    continue
                painter.drawRect(rect)
        painter.end()
        pixmap = QPixmap.fromImage(image)
        self._overlay_cache[cache_key] = pixmap
        return pixmap
