"""details_dialog: point object confidence overlay."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from ...core.backend_constants import POINT_SUPPORT_THRESHOLD
from ...core.repository import (
    _boundary_mask,
)


class PointObjectConfidenceOverlayMixin:
    def _confidence_color(self, value: float) -> tuple[int, int, int]:
        score = float(max(0.0, min(1.0, value)))
        if score <= 0.5:
            factor = score / 0.5
            return 255, int(round(200.0 * factor + 40.0)), 48
        factor = (score - 0.5) / 0.5
        return int(round(255.0 * (1.0 - factor))), 220, 70

    def _confidence_color_arrays(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        score = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
        red = np.empty(score.shape, dtype=np.uint8)
        green = np.empty(score.shape, dtype=np.uint8)
        blue = np.empty(score.shape, dtype=np.uint8)
        low_mask = score <= 0.5
        if np.any(low_mask):
            factor = np.clip(score[low_mask] / 0.5, 0.0, 1.0)
            red[low_mask] = 255
            green[low_mask] = np.clip(np.round(200.0 * factor + 40.0), 0.0, 255.0).astype(np.uint8)
            blue[low_mask] = 48
        high_mask = ~low_mask
        if np.any(high_mask):
            factor = np.clip((score[high_mask] - 0.5) / 0.5, 0.0, 1.0)
            red[high_mask] = np.clip(np.round(255.0 * (1.0 - factor)), 0.0, 255.0).astype(np.uint8)
            green[high_mask] = 220
            blue[high_mask] = 70
        return red, green, blue

    def _point_object_confidence_overlay_pixmap(self, model_id: str | None) -> QPixmap:
        cache_key = ('point_confidence', model_id, self._selection_cache_key())
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        if model_id is None:
            return QPixmap()
        confidence_row = self._confidence_metrics_for_model(model_id)
        if confidence_row is None:
            return QPixmap()
        objects = tuple(getattr(confidence_row, 'objects', ()))
        tile_rect = self._selection_crop_rect()
        x_offset = 0.0
        y_offset = 0.0
        if tile_rect is not None:
            x0 = float(tile_rect.left())
            y0 = float(tile_rect.top())
            x1 = float(tile_rect.right())
            y1 = float(tile_rect.bottom())
            x_offset = x0
            y_offset = y0
            objects = tuple(
                point_row for point_row in objects
                if x0 <= float(getattr(point_row, 'x', 0.0)) <= x1 and y0 <= float(getattr(point_row, 'y', 0.0)) <= y1
            )
        if not objects:
            return QPixmap()
        base = self._base_array()
        height, width = base.shape
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for point_row in objects:
            score = float(getattr(point_row, 'local_confidence', getattr(point_row, 'center_confidence', 0.0)))
            red, green, blue = self._confidence_color(score)
            point_probability = float(getattr(point_row, 'point_probability', 0.0))
            support_weight = float(max(0.0, min(1.0, (point_probability - float(POINT_SUPPORT_THRESHOLD)) / max(1e-8, 1.0 - float(POINT_SUPPORT_THRESHOLD))))) if point_probability >= float(POINT_SUPPORT_THRESHOLD) else 0.0
            color = QColor(red, green, blue, int(round(80.0 + 175.0 * support_weight)))
            radius = max(2, int(round(float(getattr(point_row, 'radius', 1.0)) + 1.0 + 1.5 * score)))
            painter.setPen(QPen(color, 1.4))
            painter.setBrush(color)
            x = int(round(float(getattr(point_row, 'x', 0.0)) - x_offset))
            y = int(round(float(getattr(point_row, 'y', 0.0)) - y_offset))
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
        painter.end()
        self._overlay_cache[cache_key] = pixmap
        return pixmap

    def _polygon_object_confidence_overlay_pixmap(self, model_id: str | None) -> QPixmap:
        tile_rect = self._selection_crop_rect()
        cache_key = (
            'polygon_confidence_frame',
            model_id,
            int(self._payload.get('boundary_radius') or 1),
            self._selection_cache_key(),
        )
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        render_data = self._polygon_confidence_render_data(model_id)
        if render_data is None:
            return QPixmap()
        confidence_map, support_weights, support_mask = render_data
        if tile_rect is not None:
            confidence_map = np.asarray(self._crop_ndarray(np.asarray(confidence_map, dtype=np.float32), tile_rect), dtype=np.float32)
            support_weights = np.asarray(self._crop_ndarray(np.asarray(support_weights, dtype=np.float32), tile_rect), dtype=np.float32)
            support_mask = np.asarray(self._crop_ndarray(np.asarray(support_mask, dtype=bool), tile_rect), dtype=bool)
        if not np.any(support_mask):
            return QPixmap()
        height, width = confidence_map.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        red, green, blue = self._confidence_color_arrays(confidence_map)
        final_alpha = np.clip(np.round(235.0 * support_weights), 0.0, 255.0).astype(np.uint8)
        rgba[..., 0][support_mask] = red[support_mask]
        rgba[..., 1][support_mask] = green[support_mask]
        rgba[..., 2][support_mask] = blue[support_mask]
        rgba[..., 3][support_mask] = final_alpha[support_mask]
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(image)
        self._overlay_cache[cache_key] = pixmap
        return pixmap

    def _confidence_debug_mask_pixmap(self, model_id: str | None, key: str, color: QColor) -> QPixmap:
        cache_key = (
            'confidence_debug_mask',
            model_id,
            key,
            color.red(),
            color.green(),
            color.blue(),
            color.alpha(),
            self._selection_cache_key(),
        )
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        mask = self._confidence_debug_mask(model_id, key)
        tile_rect = self._selection_crop_rect()
        if mask is not None and tile_rect is not None:
            mask = np.asarray(self._crop_ndarray(np.asarray(mask, dtype=bool), tile_rect), dtype=bool)
        pixmap = self._mask_to_pixmap(mask, color) if mask is not None else QPixmap()
        self._overlay_cache[cache_key] = pixmap
        return pixmap

    def _confidence_object_contours_pixmap(self, model_id: str | None) -> QPixmap:
        cache_key = ('confidence_contours', model_id, self._selection_cache_key())
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        object_labels = self._confidence_object_labels(model_id)
        if object_labels is not None:
            contour_mask = self._label_boundary_mask(object_labels)
        else:
            final_mask = self._confidence_debug_mask(model_id, 'final_mask')
            if final_mask is None:
                return QPixmap()
            contour_mask = _boundary_mask(final_mask)
        if contour_mask is None:
            return QPixmap()
        tile_rect = self._selection_crop_rect()
        if tile_rect is not None:
            contour_mask = np.asarray(self._crop_ndarray(np.asarray(contour_mask, dtype=bool), tile_rect), dtype=bool)
        pixmap = self._mask_to_pixmap(contour_mask, self._named_color("boundary_mask"))
        self._overlay_cache[cache_key] = pixmap
        return pixmap
