"""details_dialog: source pixmap."""
from __future__ import annotations
import numpy as np
from PyQt6.QtGui import QColor, QPixmap
from ...core.domain import ComparisonMode
from ...core.repository import (
    compute_comparison,
)


class SourcePixmapMixin:
    def _current_difference_mode_label(self) -> str:
        return self._t("details.grayscale_difference") if self.grayscale_diff_checkbox.isChecked() else self._t("details.difference")

    def _current_operation_mode(self) -> ComparisonMode:
        return ComparisonMode.GRAYSCALE_DIFF if self.grayscale_diff_checkbox.isChecked() else ComparisonMode.DISAGREEMENT

    def _update_result_controls(self) -> None:
        show_difference_controls = self._selected_result_kind() == "diff"
        self.operation_label.setVisible(show_difference_controls)
        self.grayscale_diff_checkbox.setVisible(show_difference_controls)
        show_mask_colors = self._selected_layer_view() == "binary"
        self.first_mask_color_button.setVisible(show_mask_colors)
        self.second_mask_color_button.setVisible(show_mask_colors)
        show_result_color = show_mask_colors and self._selected_result_kind() in {"diff", "boundary"}
        self.result_mask_color_button.setVisible(show_result_color)
        self.result_mask_color_button.setEnabled(show_result_color)
        self._update_color_button_styles()

    def _result_overlay_title(self) -> str:
        kind = self._selected_result_kind()
        if kind == "boundary":
            return self._t("details.boundary_difference")
        if kind == "point_matches":
            return self._t("details.point_matches")
        if kind == "confidence":
            return self._t("details.model_confidence_map")
        if kind == "confidence_bad_areas":
            return self._t("details.confidence_bad_areas")
        if kind == "confidence_mix":
            return self._t("details.confidence_mix")
        return self._t("details.difference_heatmap")

    def _refresh_dynamic_labels(self) -> None:
        _preset_key, first_key, second_key = self._current_comparison_tuple()
        self.first_source_layer_title.setText(self._source_display_name(first_key))
        self.second_source_layer_title.setText(self._source_display_name(second_key))
        self.result_layer_title.setText(self._result_overlay_title())
        self._update_color_button_styles()

    def _source_display_name(self, source_key: str | None) -> str:
        if source_key == "original":
            return self._t("details.original")
        if source_key == "gt":
            return self._t("details.ground_truth")
        if isinstance(source_key, str) and source_key.startswith("model:"):
            return self._model_display_name(source_key.split(":", 1)[1])
        return str(source_key or "-")

    def _source_pixmap(self, source_key: str | None, color: QColor, *, prefer_grayscale: bool) -> QPixmap:
        cache_key = (
            "source_pixmap",
            source_key,
            self._selected_layer_view(),
            bool(prefer_grayscale),
            bool(self._is_point_geometry()),
            self._selection_cache_key(),
            color.name(QColor.NameFormat.HexArgb),
        )
        cached = self._overlay_cache.get(cache_key)
        if cached is not None:
            return cached
        view_mode = self._selected_layer_view()
        render_float_map = self._source_render_float_map(source_key)
        float_map = self._source_float_map(source_key)
        if source_key == "original" and render_float_map is not None:
            pixmap = self._grayscale_to_pixmap(np.clip(np.rint(render_float_map * 255.0), 0, 255).astype(np.uint8))
            self._overlay_cache[cache_key] = pixmap
            return pixmap
        if view_mode == "source" and render_float_map is not None:
            pixmap = self._grayscale_to_pixmap(np.clip(np.rint(render_float_map * 255.0), 0, 255).astype(np.uint8))
            self._overlay_cache[cache_key] = pixmap
            return pixmap
        if self._is_point_geometry() and float_map is not None and source_key != "original":
            pixmap = self._intensity_to_pixmap(np.asarray(float_map, dtype=np.float32), color)
            self._overlay_cache[cache_key] = pixmap
            return pixmap
        if self._is_point_geometry() and source_key == "original" and float_map is not None:
            pixmap = self._intensity_to_pixmap(np.asarray(float_map, dtype=np.float32), color)
            self._overlay_cache[cache_key] = pixmap
            return pixmap
        if prefer_grayscale and render_float_map is not None:
            pixmap = self._grayscale_to_pixmap(np.clip(np.rint(render_float_map * 255.0), 0, 255).astype(np.uint8))
            self._overlay_cache[cache_key] = pixmap
            return pixmap
        mask = self._source_mask(source_key)
        if mask is not None:
            pixmap = self._mask_to_pixmap(mask, color)
            self._overlay_cache[cache_key] = pixmap
            return pixmap
        if render_float_map is not None:
            pixmap = self._grayscale_to_pixmap(np.clip(np.rint(render_float_map * 255.0), 0, 255).astype(np.uint8))
            self._overlay_cache[cache_key] = pixmap
            return pixmap
        empty = QPixmap()
        self._overlay_cache[cache_key] = empty
        return empty

    def _comparison_input(self, source_key: str | None, mode: ComparisonMode) -> np.ndarray:
        if mode == ComparisonMode.GRAYSCALE_DIFF:
            value = self._source_float_map(source_key)
            if value is not None:
                return np.asarray(value, dtype=np.float32)
            reference = self._base_array()
            return np.zeros_like(reference, dtype=np.float32)
        value = self._source_mask(source_key)
        if value is not None:
            return np.asarray(value, dtype=bool)
        reference = self._base_array()
        return np.zeros_like(reference, dtype=bool)

    def _pairwise_comparison_inputs(self, first_key: str | None, second_key: str | None, mode: ComparisonMode) -> tuple[np.ndarray, np.ndarray]:
        return self._comparison_input(first_key, mode), self._comparison_input(second_key, mode)

    def _comparison_overlay_result(
        self,
        first_key: str | None,
        second_key: str | None,
        mode: ComparisonMode,
        color: QColor,
    ) -> tuple[QPixmap, float]:
        cache_key = (
            "comparison_overlay",
            first_key,
            second_key,
            str(mode.value),
            self._selection_cache_key(),
            color.name(QColor.NameFormat.HexArgb),
        )
        cached = self._derived_cache.get(cache_key)
        if isinstance(cached, tuple) and len(cached) == 2:
            pixmap, score = cached
            if isinstance(pixmap, QPixmap):
                return pixmap, float(score)
        first, second = self._pairwise_comparison_inputs(first_key, second_key, mode)
        heatmap, score = compute_comparison(first, second, mode)
        pixmap = self._intensity_to_pixmap(np.asarray(heatmap, dtype=np.float32), color)
        result = (pixmap, float(score))
        self._derived_cache[cache_key] = result
        return result

    def _boundary_overlay_result(self, first_key: str | None, second_key: str | None, color: QColor) -> tuple[QPixmap, float]:
        cache_key = (
            "boundary_overlay",
            first_key,
            second_key,
            self._selection_cache_key(),
            color.name(QColor.NameFormat.HexArgb),
        )
        cached = self._derived_cache.get(cache_key)
        if isinstance(cached, tuple) and len(cached) == 2:
            pixmap, score = cached
            if isinstance(pixmap, QPixmap):
                return pixmap, float(score)
        first = self._boundary_input(first_key)
        second = self._boundary_input(second_key)
        heatmap, score = compute_comparison(first, second, ComparisonMode.DISAGREEMENT)
        pixmap = self._intensity_to_pixmap(np.asarray(heatmap, dtype=np.float32), color)
        result = (pixmap, float(score))
        self._derived_cache[cache_key] = result
        return result
