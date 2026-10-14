"""details_dialog: quick confidence score."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QPixmap
from ...core.backend_constants import MODEL_CONFIDENCE_UNCERTAIN_DELTA, POLYGON_SUPPORT_THRESHOLD
from ...core.repository import (
    _confidence_map_from_probability,
    _frame_uncertainty_components_from_probability,
    _support_weights_from_probability,
)

from neuralimage.Validation_gradient_widget_lite.ui.details_dialog_parts.module_state import DETAIL_PIXEL_VIEW_THRESHOLD


class QuickConfidenceScoreMixin:
    def _on_view_clicked(self, scene_pos: QPointF) -> None:
        plan = self._tile_plan
        if plan is None or not self.original_item.isVisible() or self._tile_selection is not None:
            return
        scale = max(0.01, float(self.overlay_view.transform().m11()))
        if min(int(plan.tile_width), int(plan.tile_height)) * scale < DETAIL_PIXEL_VIEW_THRESHOLD:
            return
        position = QPointF(float(scene_pos.x()), float(scene_pos.y()))
        if position.x() < 0.0 or position.y() < 0.0:
            return
        row_guess = min(max(int(position.y() // max(1, int(plan.stride_y))), 0), max(0, int(plan.rows) - 1))
        column_guess = min(max(int(position.x() // max(1, int(plan.stride_x))), 0), max(0, int(plan.columns) - 1))
        candidates: list[tuple[float, int, int]] = []
        for row in range(max(0, row_guess - 1), min(int(plan.rows), row_guess + 2)):
            for column in range(max(0, column_guess - 1), min(int(plan.columns), column_guess + 2)):
                rect = self._tile_rect_for_index(row, column, plan)
                if rect.contains(position):
                    center = rect.center()
                    distance = abs(center.x() - position.x()) + abs(center.y() - position.y())
                    candidates.append((distance, row, column))
        if not candidates:
            return
        _distance, row, column = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        self._tile_selection = (int(row), int(column))
        self._update_tile_lod()

    def _refresh_tile_overlays(self, base_array: np.ndarray) -> None:
        if self._subpixel_selection is not None:
            self._tile_plan = None
            self.tile_preview_item.setPixmap(QPixmap())
            self.tile_grid_item.setPixmap(QPixmap())
            self._update_tile_lod()
            return
        self._tile_plan = self._tile_plan_for_view()
        plan = self._tile_plan
        if plan is None:
            self._tile_selection = None
            self.tile_preview_item.setPixmap(QPixmap())
            self.tile_grid_item.setPixmap(QPixmap())
            self.tile_selection_item.setVisible(False)
            self._update_tile_hint()
            return
        if self._tile_selection is not None:
            row, column = self._tile_selection
            if row < 0 or column < 0 or row >= int(plan.rows) or column >= int(plan.columns):
                self._tile_selection = None
        if self._tile_selection is not None:
            self.tile_preview_item.setPixmap(QPixmap())
            self.tile_grid_item.setPixmap(QPixmap())
            self.tile_selection_item.setVisible(False)
            self._update_tile_hint()
            return
        self.tile_preview_item.setPixmap(self._tile_mosaic_pixmap(base_array, plan))
        self.tile_grid_item.setPixmap(self._tile_grid_pixmap(plan))
        self._update_tile_lod()

    def _quick_confidence_score(self, model_id: str | None) -> dict[str, float] | None:
        if model_id is None:
            return None
        model_probabilities = self._payload.get("model_probabilities") or {}
        model_probability = model_probabilities.get(model_id)
        if model_probability is None:
            return None
        probability = np.clip(np.asarray(model_probability, dtype=np.float32), 0.0, 1.0)
        if probability.ndim != 2 or probability.size == 0:
            return None
        model_mask = self._source_mask(f"model:{model_id}")
        if model_mask is None:
            model_mask = np.asarray(probability >= self._model_threshold(model_id), dtype=bool)
        confidence = _confidence_map_from_probability(probability)
        support_weights = _support_weights_from_probability(probability, POLYGON_SUPPORT_THRESHOLD)
        support_mask = support_weights > 0.0
        weight_total = float(np.sum(np.asarray(support_weights, dtype=np.float64), dtype=np.float64))
        delta = float(self._payload.get("confidence_uncertainty_delta") or MODEL_CONFIDENCE_UNCERTAIN_DELTA)
        uncertain = np.abs(probability - 0.5) <= max(0.0, delta)
        frame_uncertainty_score, mean_uncertainty, low_conf_fraction, worst_tail_uncertainty, largest_low_conf_component = _frame_uncertainty_components_from_probability(
            probability,
            support_threshold=POLYGON_SUPPORT_THRESHOLD,
        )
        return {
            "mean_confidence": float(np.sum(np.asarray(support_weights * confidence, dtype=np.float64), dtype=np.float64) / max(1e-8, weight_total)) if weight_total > 0.0 else 0.0,
            "mean_probability": float(np.sum(np.asarray(support_weights * probability, dtype=np.float64), dtype=np.float64) / max(1e-8, weight_total)) if weight_total > 0.0 else 0.0,
            "uncertain_fraction": float(np.mean(uncertain[support_mask], dtype=np.float64)) if np.any(support_mask) else 0.0,
            "frame_uncertainty_score": float(frame_uncertainty_score),
            "mean_uncertainty": float(mean_uncertainty),
            "low_conf_fraction": float(low_conf_fraction),
            "worst_tail_uncertainty": float(worst_tail_uncertainty),
            "largest_low_conf_component": float(largest_low_conf_component),
            "uncertain_support_fraction": float(low_conf_fraction),
            "top_uncertainty_mean": float(worst_tail_uncertainty),
            "largest_uncertain_region_fraction": float(largest_low_conf_component),
            "focus_fraction": float(np.count_nonzero(support_mask) / max(1, support_mask.size)),
        }
