"""details_dialog: refresh result kind options."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import Qt
from ...core.repository import (
    _boundary_mask,
    _point_map_from_view,
)


class RefreshResultKindOptionsMixin:
    def _source_render_float_map(self, source_key: str | None) -> np.ndarray | None:
        if source_key == "original":
            original = self._payload.get("original_gray")
            value = np.asarray(original, dtype=np.float32) / 255.0 if original is not None else None
            return self._crop_ndarray(value, self._selection_crop_rect()) if value is not None else None
        if source_key == "gt":
            gt_mask = self._payload.get("gt_mask")
            value = np.asarray(gt_mask, dtype=np.float32) if gt_mask is not None else None
            return self._crop_ndarray(value, self._selection_crop_rect()) if value is not None else None
        if isinstance(source_key, str) and source_key.startswith("model:"):
            model_id = source_key.split(":", 1)[1]
            probabilities = self._payload.get("model_probabilities") or {}
            model_prob = probabilities.get(model_id)
            value = np.asarray(model_prob, dtype=np.float32) if model_prob is not None else None
            return self._crop_ndarray(value, self._selection_crop_rect()) if value is not None else None
        return self._source_float_map(source_key)

    def _refresh_result_kind_options(self, preferred_kind: str | None = None) -> None:
        current_kind = str(
            preferred_kind
            or self._sticky_result_kind
            or self.result_kind_combo.currentData()
            or ""
        )
        if not current_kind:
            if self._preferred_confidence_model_id() is not None:
                current_kind = "confidence"
            elif self._is_point_geometry():
                current_kind = "point_matches"
            else:
                current_kind = "diff"
        confidence_model_id = self._confidence_model_id()
        confidence_output_available = self._model_confidence_output_available(confidence_model_id)
        if not confidence_output_available and current_kind in {"confidence_bad_areas", "confidence_mix"}:
            current_kind = "point_matches" if self._is_point_geometry() else "diff"
        items: list[tuple[str, str]] = [(self._t("details.comparison_difference"), "diff")]
        if self._is_point_geometry():
            items.append((self._t("details.point_matches"), "point_matches"))
        else:
            items.append((self._t("details.boundary_difference"), "boundary"))
        items.append((self._t("details.model_confidence_map"), "confidence"))
        if confidence_output_available:
            items.append((self._t("details.confidence_bad_areas"), "confidence_bad_areas"))
            items.append((self._t("details.confidence_mix"), "confidence_mix"))
        self.result_kind_combo.blockSignals(True)
        self.result_kind_combo.clear()
        for label, key in items:
            self.result_kind_combo.addItem(label, key)
            combo_index = self.result_kind_combo.count() - 1
            self.result_kind_combo.setItemData(combo_index, self._result_kind_hint(str(key)) or "", Qt.ItemDataRole.ToolTipRole)
        index = self.result_kind_combo.findData(current_kind)
        if index < 0 and self._sticky_result_kind is not None:
            index = self.result_kind_combo.findData(self._sticky_result_kind)
        self.result_kind_combo.setCurrentIndex(index if index >= 0 else 0)
        self.result_kind_combo.blockSignals(False)
        self._sticky_result_kind = self._selected_result_kind()
        self.result_kind_combo.setToolTip(self._result_kind_hint(self._selected_result_kind()) or '')

    def _selected_result_kind(self) -> str:
        return str(self.result_kind_combo.currentData() or "diff")

    def _is_point_geometry(self) -> bool:
        return str(self._payload.get("geometry_mode") or "mask") == "point"

    def _source_point_map(self, source_key: str | None) -> np.ndarray | None:
        cache_key = ("source_point_map", source_key)
        cached = self._derived_cache.get(cache_key)
        if isinstance(cached, np.ndarray):
            return cached
        if source_key == "gt":
            gt_view = self._payload.get("gt_point_view")
            point_map = _point_map_from_view(gt_view) if gt_view is not None else None
            if point_map is not None:
                self._derived_cache[cache_key] = point_map
            return point_map
        if isinstance(source_key, str) and source_key.startswith("model:"):
            model_id = source_key.split(":", 1)[1]
            model_views = self._payload.get("model_views") or {}
            view = model_views.get(model_id)
            point_map = _point_map_from_view(view) if view is not None else None
            if point_map is not None:
                self._derived_cache[cache_key] = point_map
            return point_map
        return None

    def _source_point_view(self, source_key: str | None):
        if source_key == 'gt':
            return self._payload.get('gt_point_view')
        if isinstance(source_key, str) and source_key.startswith('model:'):
            model_id = source_key.split(':', 1)[1]
            return (self._payload.get('model_views') or {}).get(model_id)
        return None

    def _boundary_input(self, source_key: str | None) -> np.ndarray:
        mask = self._source_mask(source_key)
        if mask is None:
            return np.zeros_like(self._base_array(), dtype=bool)
        return np.asarray(_boundary_mask(mask), dtype=bool)

    def _point_match_pairs(self, first_points: tuple[object, ...], second_points: tuple[object, ...], radius: float) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
        threshold = max(0.0, float(radius))
        candidate_pairs: list[tuple[float, int, int]] = []
        for index_a, point_a in enumerate(first_points):
            for index_b, point_b in enumerate(second_points):
                distance = float(np.hypot(float(getattr(point_a, 'x', 0.0)) - float(getattr(point_b, 'x', 0.0)), float(getattr(point_a, 'y', 0.0)) - float(getattr(point_b, 'y', 0.0))))
                if distance <= threshold:
                    candidate_pairs.append((distance, index_a, index_b))
        candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
        pairs: list[tuple[int, int, float]] = []
        matched_a: set[int] = set()
        matched_b: set[int] = set()
        for distance, index_a, index_b in candidate_pairs:
            if index_a in matched_a or index_b in matched_b:
                continue
            matched_a.add(index_a)
            matched_b.add(index_b)
            pairs.append((index_a, index_b, distance))
        return pairs, matched_a, matched_b
