"""details_dialog: algorithmic uncertainty map."""
from __future__ import annotations
import numpy as np
from PyQt6.QtGui import QColor, QImage, QPixmap
from ...core.backend_constants import MODEL_CONFIDENCE_UNCERTAIN_DELTA
from ...core.confidence_maps import (
    DEFAULT_CONFIDENCE_BAD_AREA_THRESHOLD,
    build_algorithmic_uncertainty,
    build_model_uncertainty,
    confidence_bad_area_intensity,
)


class AlgorithmicUncertaintyMapMixin:
    def _confidence_bad_area_threshold(self) -> float:
        delta = float(self._payload.get("confidence_uncertainty_delta") or MODEL_CONFIDENCE_UNCERTAIN_DELTA)
        if not np.isfinite(delta):
            return float(DEFAULT_CONFIDENCE_BAD_AREA_THRESHOLD)
        return float(np.clip(1.0 - 2.0 * max(0.0, delta), 0.0, 0.999))

    def _model_output_uncertainty_map(self, model_id: str | None) -> np.ndarray:
        if model_id is None:
            return np.zeros_like(self._base_array(), dtype=np.float32)
        if not self._model_confidence_output_available(model_id):
            return np.zeros_like(self._base_array(), dtype=np.float32)
        probabilities = self._payload.get("model_probabilities") or {}
        confidence_map = probabilities.get(model_id)
        if confidence_map is None:
            return np.zeros_like(self._base_array(), dtype=np.float32)
        values = np.asarray(confidence_map, dtype=np.float32)
        tile_rect = self._selection_crop_rect()
        if tile_rect is not None:
            values = np.asarray(self._crop_ndarray(values, tile_rect), dtype=np.float32)
        return build_model_uncertainty(values)

    def _algorithmic_uncertainty_map(self, model_id: str | None) -> np.ndarray:
        if model_id is None:
            return np.zeros_like(self._base_array(), dtype=np.float32)
        if self._is_point_geometry():
            point_map = self._source_point_map(f"model:{model_id}")
            source = np.asarray(point_map, dtype=np.float32) if point_map is not None else None
        else:
            masks = self._payload.get("model_masks") or {}
            source = masks.get(model_id)
        if source is None:
            return np.zeros_like(self._base_array(), dtype=np.float32)
        values = np.asarray(source, dtype=np.float32)
        tile_rect = self._selection_crop_rect()
        if tile_rect is not None:
            values = np.asarray(self._crop_ndarray(values, tile_rect), dtype=np.float32)
        return build_algorithmic_uncertainty(values, boundary_radius=float(self._payload.get("boundary_radius") or 1.0) + 2.0)

    def _bad_area_intensity_pixmap(self, intensity: np.ndarray, color: QColor, *, alpha_scale: float = 245.0) -> QPixmap:
        alpha = np.clip(np.asarray(intensity, dtype=np.float32), 0.0, 1.0)
        if alpha.ndim != 2 or alpha.size == 0:
            return QPixmap()
        height, width = alpha.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = color.red()
        rgba[..., 1] = color.green()
        rgba[..., 2] = color.blue()
        rgba[..., 3] = np.clip(np.round(alpha * float(alpha_scale)), 0.0, 255.0).astype(np.uint8)
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(image)

    def _confidence_bad_areas_pixmap(self, model_id: str | None) -> QPixmap:
        threshold = self._confidence_bad_area_threshold()
        cache_key = (
            "confidence_bad_areas",
            model_id,
            round(float(threshold), 4),
            self._selection_cache_key(),
        )
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        uncertainty = self._model_output_uncertainty_map(model_id)
        intensity = confidence_bad_area_intensity(uncertainty, threshold=threshold)
        pixmap = self._bad_area_intensity_pixmap(intensity, QColor(255, 48, 80, 255), alpha_scale=250.0)
        self._overlay_cache[cache_key] = pixmap
        return pixmap
