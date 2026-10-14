"""details_dialog: confidence mix pixmap."""
from __future__ import annotations
import numpy as np
from PyQt6.QtGui import QImage, QPixmap
from ...core.confidence_maps import (
    combine_uncertainty_maps,
)


class ConfidenceMixPixmapMixin:
    def _confidence_mix_pixmap(self, model_id: str | None) -> QPixmap:
        threshold = self._confidence_bad_area_threshold()
        cache_key = (
            "confidence_mix",
            model_id,
            round(float(threshold), 4),
            self._selection_cache_key(),
        )
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        algorithmic_uncertainty = self._algorithmic_uncertainty_map(model_id)
        model_uncertainty = self._model_output_uncertainty_map(model_id)
        combined = combine_uncertainty_maps(
            algorithmic_uncertainty,
            model_uncertainty,
            algorithmic_threshold=threshold,
            model_threshold=threshold,
        )
        agreement = np.asarray(combined.agreement, dtype=np.float32)
        algorithmic_only = np.asarray(combined.algorithmic_only, dtype=np.float32)
        model_only = np.asarray(combined.model_only, dtype=np.float32)
        if agreement.ndim != 2 or agreement.size == 0:
            return QPixmap()
        height, width = agreement.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        both_color = np.asarray([255.0, 216.0, 32.0], dtype=np.float32)
        algorithmic_color = np.asarray([0.0, 210.0, 255.0], dtype=np.float32)
        model_color = np.asarray([255.0, 0.0, 160.0], dtype=np.float32)
        both_weight = 1.45 * agreement
        algorithmic_weight = 0.85 * algorithmic_only
        model_weight = 0.95 * model_only
        total = both_weight + algorithmic_weight + model_weight
        active = total > 1e-6
        red = np.zeros_like(total, dtype=np.float32)
        green = np.zeros_like(total, dtype=np.float32)
        blue = np.zeros_like(total, dtype=np.float32)
        red[active] = (
            both_weight[active] * both_color[0]
            + algorithmic_weight[active] * algorithmic_color[0]
            + model_weight[active] * model_color[0]
        ) / total[active]
        green[active] = (
            both_weight[active] * both_color[1]
            + algorithmic_weight[active] * algorithmic_color[1]
            + model_weight[active] * model_color[1]
        ) / total[active]
        blue[active] = (
            both_weight[active] * both_color[2]
            + algorithmic_weight[active] * algorithmic_color[2]
            + model_weight[active] * model_color[2]
        ) / total[active]
        alpha = np.clip(0.95 * agreement + 0.62 * algorithmic_only + 0.68 * model_only, 0.0, 1.0)
        rgba[..., 0] = np.clip(np.round(red), 0.0, 255.0).astype(np.uint8)
        rgba[..., 1] = np.clip(np.round(green), 0.0, 255.0).astype(np.uint8)
        rgba[..., 2] = np.clip(np.round(blue), 0.0, 255.0).astype(np.uint8)
        rgba[..., 3] = np.clip(np.round(alpha * 255.0), 0.0, 255.0).astype(np.uint8)
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(image)
        self._overlay_cache[cache_key] = pixmap
        return pixmap
