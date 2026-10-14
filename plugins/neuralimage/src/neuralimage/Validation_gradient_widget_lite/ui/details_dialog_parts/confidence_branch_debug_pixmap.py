"""details_dialog: confidence branch debug pixmap."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


class ConfidenceBranchDebugPixmapMixin:
    def _confidence_branch_debug_pixmap(self, model_id: str | None) -> QPixmap:
        cache_key = ('confidence_branch_debug', model_id, self._selection_cache_key())
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        branch_masks = {
            'large_polygon': self._confidence_debug_mask(model_id, 'large_polygon'),
            'global_hysteresis': self._confidence_debug_mask(model_id, 'global_hysteresis'),
            'elongated': self._confidence_debug_mask(model_id, 'elongated'),
            'small_weak': self._confidence_debug_mask(model_id, 'small_weak'),
            'adaptive_local': self._confidence_debug_mask(model_id, 'adaptive_local'),
        }
        base = self._base_array()
        height, width = base.shape
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        tile_rect = self._selection_crop_rect()
        color_map = {
            'large_polygon': (255, 225, 70),
            'global_hysteresis': (80, 210, 255),
            'elongated': (255, 170, 0),
            'small_weak': (255, 0, 140),
            'adaptive_local': (100, 255, 120),
        }
        for branch_key, rgb in color_map.items():
            branch_mask = branch_masks.get(branch_key)
            if branch_mask is None:
                continue
            branch_bool = np.asarray(branch_mask, dtype=bool)
            if tile_rect is not None:
                branch_bool = np.asarray(self._crop_ndarray(branch_bool, tile_rect), dtype=bool)
            rgba[..., 0][branch_bool] = rgb[0]
            rgba[..., 1][branch_bool] = rgb[1]
            rgba[..., 2][branch_bool] = rgb[2]
            rgba[..., 3][branch_bool] = np.maximum(rgba[..., 3][branch_bool], 155)
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(image)
        self._overlay_cache[cache_key] = pixmap
        return pixmap

    def _confidence_overlay_pixmap(self, model_id: str | None) -> QPixmap:
        tile_rect = self._selection_crop_rect()
        cache_key = ('confidence_frame_overlay', model_id, self._selection_cache_key())
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        if model_id is None:
            return QPixmap()
        if self._is_point_geometry():
            confidence_pixmap = self._point_object_confidence_overlay_pixmap(model_id)
            if not confidence_pixmap.isNull():
                self._overlay_cache[cache_key] = confidence_pixmap
                return confidence_pixmap
        render_data = self._polygon_confidence_render_data(model_id)
        if render_data is None:
            return QPixmap()
        confidence_map, support_weights, support_mask = render_data
        if tile_rect is not None:
            confidence_map = np.asarray(self._crop_ndarray(np.asarray(confidence_map, dtype=np.float32), tile_rect), dtype=np.float32)
            support_weights = np.asarray(self._crop_ndarray(np.asarray(support_weights, dtype=np.float32), tile_rect), dtype=np.float32)
            support_mask = np.asarray(self._crop_ndarray(np.asarray(support_mask, dtype=bool), tile_rect), dtype=bool)
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

    def _comparison_model_ids(self) -> list[str]:
        _preset_key, first_key, second_key = self._current_comparison_tuple()
        result: list[str] = []
        for source_key in (first_key, second_key):
            if isinstance(source_key, str) and source_key.startswith("model:"):
                model_id = source_key.split(":", 1)[1]
                if model_id not in result:
                    result.append(model_id)
        return result

    def _pairwise_comparison_entry(self, model_a_id: str | None, model_b_id: str | None) -> dict[str, float | str] | None:
        if not model_a_id or not model_b_id:
            return None
        pairwise = self._payload.get("pairwise_model_comparisons") or ()
        for row in pairwise:
            left = str(row.get("model_a", ""))
            right = str(row.get("model_b", ""))
            if {left, right} == {str(model_a_id), str(model_b_id)}:
                return row
        return None
