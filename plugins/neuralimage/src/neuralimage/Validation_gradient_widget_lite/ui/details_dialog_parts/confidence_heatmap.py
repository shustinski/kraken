"""details_dialog: confidence heatmap."""
from __future__ import annotations
import numpy as np
from ...core.backend_constants import POLYGON_SUPPORT_THRESHOLD
from ...core.repository import (
    _boundary_mask,
    _confidence_display_map_from_probability,
    _paint_disk,
    _support_weights_from_probability,
)


class ConfidenceHeatmapMixin:
    def _confidence_metrics_for_model(self, model_id: str | None):
        if model_id is None:
            return None
        return (self._payload.get("model_confidence") or {}).get(model_id)

    def _model_confidence_output_available(self, model_id: str | None) -> bool:
        if model_id is None:
            return False
        availability = self._payload.get("model_confidence_output_available")
        if isinstance(availability, dict) and model_id in availability:
            return bool(availability.get(model_id))
        return bool((getattr(self._record, "model_prob_paths", {}) or {}).get(model_id))

    def _confidence_debug_for_model(self, model_id: str | None):
        confidence_row = self._confidence_metrics_for_model(model_id)
        if confidence_row is None:
            return None
        return getattr(confidence_row, 'debug_data', None)

    def _confidence_debug_mask(self, model_id: str | None, key: str) -> np.ndarray | None:
        debug_data = self._confidence_debug_for_model(model_id)
        if debug_data is None:
            return None
        if key == 'final_mask':
            value = getattr(debug_data, 'merged_mask', None)
        elif key == 'low_mask':
            value = getattr(debug_data, 'low_mask', None)
        elif key == 'high_mask':
            value = getattr(debug_data, 'high_mask', None)
        elif key == 'adaptive_low_mask':
            value = getattr(debug_data, 'adaptive_low_mask', None)
        elif key == 'adaptive_high_mask':
            value = getattr(debug_data, 'adaptive_high_mask', None)
        elif key == 'core_seeds':
            value = getattr(debug_data, 'core_seed_mask', None)
        elif key == 'candidate_region':
            value = getattr(debug_data, 'candidate_region_mask', None)
        elif key == 'thin_barrier':
            value = getattr(debug_data, 'thin_barrier_map', None)
        elif key == 'bridge_cuts':
            value = getattr(debug_data, 'bridge_cut_mask', None)
        elif key == 'barrier_stops':
            value = getattr(debug_data, 'barrier_stop_mask', None)
        else:
            value = (getattr(debug_data, 'branch_masks', {}) or {}).get(key)
        if value is None:
            return None
        return np.asarray(value, dtype=bool)

    def _confidence_debug_float_map(self, model_id: str | None, key: str) -> np.ndarray | None:
        debug_data = self._confidence_debug_for_model(model_id)
        if debug_data is None:
            return None
        value = getattr(debug_data, key, None)
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32)

    def _confidence_object_labels(self, model_id: str | None) -> np.ndarray | None:
        debug_data = self._confidence_debug_for_model(model_id)
        if debug_data is None:
            return None
        value = getattr(debug_data, 'object_labels', None)
        if value is None:
            return None
        labels = np.asarray(value)
        if labels.ndim != 2:
            return None
        return labels.astype(np.int32, copy=False)

    @staticmethod
    def _label_boundary_mask(labels: np.ndarray) -> np.ndarray:
        labels_int = np.asarray(labels, dtype=np.int32)
        positive = labels_int > 0
        if not np.any(positive):
            return np.zeros_like(labels_int, dtype=bool)
        boundary = np.asarray(_boundary_mask(positive), dtype=bool)
        horizontal_diff = (
            (labels_int[:, 1:] != labels_int[:, :-1])
            & (labels_int[:, 1:] > 0)
            & (labels_int[:, :-1] > 0)
        )
        vertical_diff = (
            (labels_int[1:, :] != labels_int[:-1, :])
            & (labels_int[1:, :] > 0)
            & (labels_int[:-1, :] > 0)
        )
        boundary[:, 1:] |= horizontal_diff
        boundary[1:, :] |= vertical_diff
        return boundary

    def _model_threshold(self, model_id: str | None) -> float:
        if model_id is None:
            return 0.5
        for spec in self._build_result.model_specs:
            if spec.model_id == model_id:
                return float(spec.threshold)
        return 0.5

    def _polygon_confidence_render_data(self, model_id: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        cache_key = ("polygon_confidence_render", model_id)
        cached = self._derived_cache.get(cache_key)
        if isinstance(cached, tuple) and len(cached) == 3:
            return cached
        if model_id is None:
            return None
        model_probabilities = self._payload.get("model_probabilities") or {}
        model_probability = model_probabilities.get(model_id)
        if model_probability is None:
            return None
        probability = np.clip(np.asarray(model_probability, dtype=np.float32), 0.0, 1.0)
        if probability.ndim != 2 or probability.size == 0:
            return None
        source_mask = None
        model_masks = self._payload.get("model_masks") or {}
        model_mask = model_masks.get(model_id)
        if model_mask is not None:
            source_mask = np.asarray(model_mask, dtype=bool)
        if source_mask is None:
            source_mask = np.asarray(probability >= self._model_threshold(model_id), dtype=bool)
        strong_mask = np.asarray(source_mask, dtype=bool)
        if strong_mask.shape != probability.shape:
            strong_mask = np.zeros_like(probability, dtype=bool)
        support_weights = _support_weights_from_probability(probability, POLYGON_SUPPORT_THRESHOLD)
        support_mask = np.asarray(support_weights > 0.0, dtype=bool)
        if not np.any(support_mask):
            support_mask = strong_mask
        display_confidence = _confidence_display_map_from_probability(
            probability,
            support_mask=support_mask,
            support_threshold=POLYGON_SUPPORT_THRESHOLD,
        )
        result = (
            np.asarray(display_confidence, dtype=np.float32),
            np.asarray(support_weights, dtype=np.float32),
            np.asarray(support_mask, dtype=bool),
        )
        self._derived_cache[cache_key] = result
        return result

    def _confidence_heatmap(self, model_id: str | None, *, uncertainty: bool) -> np.ndarray:
        if model_id is None:
            return np.zeros_like(self._base_array(), dtype=np.float32)
        probabilities = self._payload.get("model_probabilities") or {}
        model_prob = probabilities.get(model_id)
        if model_prob is None:
            return np.zeros_like(self._base_array(), dtype=np.float32)
        prob = np.asarray(model_prob, dtype=np.float32)
        tile_rect = self._selection_crop_rect()
        if tile_rect is not None:
            prob = self._crop_ndarray(prob, tile_rect)
        if self._is_point_geometry():
            model_view = (self._payload.get("model_views") or {}).get(model_id)
            result = np.zeros_like(prob, dtype=np.float32)
            if model_view is None:
                return result
            points = tuple(getattr(model_view, 'points', ()))
            x_offset = 0.0
            y_offset = 0.0
            if tile_rect is not None:
                x0 = float(tile_rect.left())
                y0 = float(tile_rect.top())
                x1 = float(tile_rect.right())
                y1 = float(tile_rect.bottom())
                x_offset = x0
                y_offset = y0
                points = tuple(
                    point for point in points
                    if x0 <= float(getattr(point, 'x', 0.0)) <= x1 and y0 <= float(getattr(point, 'y', 0.0)) <= y1
                )
            for point in points:
                x = float(getattr(point, 'x', 0.0)) - x_offset
                y = float(getattr(point, 'y', 0.0)) - y_offset
                px = int(round(x))
                py = int(round(y))
                point_prob = float(prob[py, px]) if 0 <= py < prob.shape[0] and 0 <= px < prob.shape[1] else 0.0
                point_conf = float(2.0 * abs(point_prob - 0.5))
                value = float(1.0 - point_conf) if uncertainty else point_conf
                _paint_disk(result, x, y, max(1.0, float(getattr(point, 'radius', 0.0))) + 1.0, value)
            return np.clip(result, 0.0, 1.0).astype(np.float32)
        render_data = self._polygon_confidence_render_data(model_id)
        if render_data is None:
            return np.zeros_like(prob, dtype=np.float32)
        display_confidence, _support_weights, _support_mask = render_data
        display_confidence = np.asarray(display_confidence, dtype=np.float32)
        return np.clip(1.0 - display_confidence if uncertainty else display_confidence, 0.0, 1.0).astype(np.float32, copy=False)
