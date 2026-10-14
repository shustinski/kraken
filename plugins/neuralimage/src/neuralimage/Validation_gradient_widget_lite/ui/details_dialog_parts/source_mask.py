"""details_dialog: source mask."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import QRectF
from ...core.subpixel_grid import (
    subpixel_bounds_for_index,
)
from ...core.tile_grid import TileGridPlan


class SourceMaskMixin:
    def _selected_subpixel_score_info(self) -> tuple[str, float | None, list[str]]:
        if self._subpixel_selection is None:
            return self._t("details.subpixel_score"), None, []
        selection = self._subpixel_selection
        spec = selection.spec or self._default_subpixel_spec()
        lines = [
            f"{self._t('details.subpixel_selection')}: r{int(selection.sub_row) + 1}, c{int(selection.sub_column) + 1}",
            f"{self._t('details.parent_frame')}: r{int(selection.parent_row) + 1}, c{int(selection.parent_column) + 1}",
            f"{int(spec.rows)} x {int(spec.columns)}",
        ]
        return self._t("details.subpixel_score"), float(selection.subpixel_value), lines

    def _on_model_a_changed(self, *_args) -> None:
        self._refresh_scene(reset_view=False)
        self._refresh_info()
        self._ensure_confidence_ready()
        self._store_view_settings()

    def _on_model_pair_changed(self, *_args) -> None:
        self._apply_selected_model(self._selected_model_for_current_comparison())
        self._refresh_scene(reset_view=False)
        self._refresh_info()
        self._ensure_confidence_ready()
        self._store_view_settings()

    def _refresh_scene_from_controls(self, *_args) -> None:
        self._refresh_scene(reset_view=False)
        self._refresh_info()
        if self._result_kind_requires_confidence():
            self._ensure_confidence_ready()
        else:
            self._set_confidence_loading_state(False)
        source_mode = self._selected_layer_view() == "source"
        self.first_color.setEnabled(not source_mode)
        self.second_color.setEnabled(not source_mode)
        self._store_view_settings()

    def _set_confidence_loading_state(self, active: bool, model_id: str | None = None) -> None:
        if not active:
            self.confidence_loading_label.setVisible(False)
            self.confidence_loading_bar.setVisible(False)
            return
        name = self._model_display_name(model_id) if model_id is not None else "model"
        self.confidence_loading_label.setText(f"Loading confidence: {name}")
        self.confidence_loading_label.setVisible(True)
        self.confidence_loading_bar.setVisible(True)

    def _selected_layer_view(self) -> str:
        return str(self.layer_view_combo.currentData() or "binary")

    def _tile_selection_rect(self, plan: TileGridPlan | None = None) -> QRectF | None:
        if self._tile_selection is None:
            return None
        resolved_plan = plan or self._tile_plan_for_view()
        if resolved_plan is None:
            return None
        row, column = self._tile_selection
        if row < 0 or column < 0 or row >= int(resolved_plan.rows) or column >= int(resolved_plan.columns):
            return None
        return self._tile_rect_for_index(int(row), int(column), resolved_plan)

    def _selection_crop_rect(self) -> QRectF | None:
        if self._subpixel_selection is not None:
            return self._subpixel_selection_rect()
        return self._tile_selection_rect()

    def _selection_cache_key(self) -> tuple[int, int, int, int] | None:
        rect = self._selection_crop_rect()
        if rect is None:
            return None
        return (
            max(0, int(np.floor(float(rect.left())))),
            max(0, int(np.floor(float(rect.top())))),
            max(0, int(np.ceil(float(rect.right())))),
            max(0, int(np.ceil(float(rect.bottom())))),
        )

    def _subpixel_selection_rect(self) -> QRectF | None:
        if self._subpixel_selection is None:
            return None
        spec = self._subpixel_selection.spec
        if spec is None:
            spec = self._default_subpixel_spec()
        full = self._full_base_array()
        left, top, width, height = subpixel_bounds_for_index(
            float(full.shape[1]),
            float(full.shape[0]),
            int(self._subpixel_selection.sub_row),
            int(self._subpixel_selection.sub_column),
            spec,
        )
        if width <= 0 or height <= 0:
            return None
        return QRectF(float(left), float(top), float(width), float(height))

    @staticmethod
    def _crop_ndarray(array: np.ndarray, rect: QRectF | None) -> np.ndarray:
        if rect is None:
            return array
        if not isinstance(array, np.ndarray) or array.ndim < 2:
            return array
        left = max(0, int(np.floor(float(rect.left()))))
        top = max(0, int(np.floor(float(rect.top()))))
        right = min(int(array.shape[1]), int(np.ceil(float(rect.right()))))
        bottom = min(int(array.shape[0]), int(np.ceil(float(rect.bottom()))))
        if right <= left or bottom <= top:
            return array[:0, :0]
        if array.ndim == 2:
            return array[top:bottom, left:right]
        return array[top:bottom, left:right, ...]

    def _full_base_array(self) -> np.ndarray:
        cache_key = ("base_array", str(self._payload.get("selected_model_id") or ""))
        cached = self._derived_cache.get(cache_key)
        if isinstance(cached, np.ndarray):
            return cached
        original = self._payload.get("original_gray")
        if original is not None:
            base_array = np.asarray(original, dtype=np.uint8)
            if base_array.size > 0:
                self._derived_cache[cache_key] = base_array
                return base_array
        selected_prob = np.asarray(self._payload.get("selected_prob"), dtype=np.float32)
        if selected_prob.ndim == 2 and selected_prob.size > 0:
            base_array = np.clip(np.rint(selected_prob * 255.0), 0, 255).astype(np.uint8)
            self._derived_cache[cache_key] = base_array
            return base_array
        fallback = np.zeros((1, 1), dtype=np.uint8)
        self._derived_cache[cache_key] = fallback
        return fallback

    def _base_array(self) -> np.ndarray:
        base = self._full_base_array()
        rect = self._subpixel_selection_rect()
        if rect is None:
            return base
        return self._crop_ndarray(base, rect)

    def _source_mask(self, source_key: str | None) -> np.ndarray | None:
        cache_key = ("source_mask", source_key, bool(self._is_point_geometry()))
        cached = self._derived_cache.get(cache_key)
        if isinstance(cached, np.ndarray):
            return self._crop_ndarray(cached, self._selection_crop_rect())
        if self._is_point_geometry():
            point_map = self._source_point_map(source_key)
            mask = np.asarray(point_map > 0.1, dtype=bool) if point_map is not None else None
            if mask is not None:
                self._derived_cache[cache_key] = mask
            return self._crop_ndarray(mask, self._selection_crop_rect()) if mask is not None else None
        if source_key == "gt":
            gt_mask = self._payload.get("gt_mask")
            mask = np.asarray(gt_mask, dtype=bool) if gt_mask is not None else None
            if mask is not None:
                self._derived_cache[cache_key] = mask
            return self._crop_ndarray(mask, self._selection_crop_rect()) if mask is not None else None
        if isinstance(source_key, str) and source_key.startswith("model:"):
            model_id = source_key.split(":", 1)[1]
            masks = self._payload.get("model_masks") or {}
            model_mask = masks.get(model_id)
            mask = np.asarray(model_mask, dtype=bool) if model_mask is not None else None
            if mask is not None:
                self._derived_cache[cache_key] = mask
            return self._crop_ndarray(mask, self._selection_crop_rect()) if mask is not None else None
        return None

    def _source_float_map(self, source_key: str | None) -> np.ndarray | None:
        if self._is_point_geometry():
            point_map = self._source_point_map(source_key)
            value = np.asarray(point_map, dtype=np.float32) if point_map is not None else None
            return self._crop_ndarray(value, self._selection_crop_rect()) if value is not None else None
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
        return None
