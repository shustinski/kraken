"""details_dialog: build view settings payload."""
from __future__ import annotations
import numpy as np
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QColorDialog,
)


class BuildViewSettingsPayloadMixin:
    def _update_layer_states(self) -> None:
        if self._legacy_base_hold_active:
            show_base = self._hold_preview_mode != "confidence_source"
            self.original_item.setVisible(show_base)
            self.original_item.setOpacity(self.original_opacity.value() / 100.0)
            self.first_source_item.setVisible(False)
            self.second_source_item.setVisible(False)
            self.result_item.setVisible(False)
            if self._hold_preview_mode == "confidence_source" and not self._hold_preview_pixmap.isNull():
                self.hold_preview_item.setPixmap(self._hold_preview_pixmap)
                self.hold_preview_item.setVisible(True)
                self.hold_preview_item.setOpacity(1.0)
            else:
                self.hold_preview_item.setVisible(False)
            self._update_tile_lod()
            return
        self.hold_preview_item.setVisible(False)
        self.original_item.setVisible(self.original_visible.isChecked())
        self.original_item.setOpacity(self.original_opacity.value() / 100.0)
        self.first_source_item.setVisible(self.first_source_visible.isChecked())
        self.first_source_item.setOpacity(self.first_source_opacity.value() / 100.0)
        self.second_source_item.setVisible(self.second_source_visible.isChecked())
        self.second_source_item.setOpacity(self.second_source_opacity.value() / 100.0)
        self.result_item.setVisible(self.result_visible.isChecked())
        self.result_item.setOpacity(self.result_opacity.value() / 100.0)
        self._update_tile_lod()

    def _grayscale_to_pixmap(self, array: np.ndarray) -> QPixmap:
        contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.uint8))
        height, width = contiguous.shape
        image = QImage(contiguous.data, width, height, int(contiguous.strides[0]), QImage.Format.Format_Grayscale8).copy()
        return QPixmap.fromImage(image.convertToFormat(QImage.Format.Format_RGB888))

    def _mask_to_pixmap(self, mask: np.ndarray, color: QColor) -> QPixmap:
        mask_uint8 = np.ascontiguousarray(np.asarray(mask, dtype=bool).astype(np.uint8))
        height, width = mask_uint8.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = color.red()
        rgba[..., 1] = color.green()
        rgba[..., 2] = color.blue()
        rgba[..., 3] = mask_uint8 * 255
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(image)

    def _intensity_to_pixmap(self, intensity: np.ndarray, color: QColor) -> QPixmap:
        alpha = np.clip(np.asarray(intensity, dtype=np.float32), 0.0, 1.0)
        if alpha.ndim != 2:
            alpha = np.zeros((1, 1), dtype=np.float32)
        height, width = alpha.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = color.red()
        rgba[..., 1] = color.green()
        rgba[..., 2] = color.blue()
        rgba[..., 3] = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(image)

    def _dual_intensity_to_pixmap(self, confident: np.ndarray, uncertain: np.ndarray, confident_color: QColor, uncertain_color: QColor) -> QPixmap:
        confident_alpha = np.clip(np.asarray(confident, dtype=np.float32), 0.0, 1.0)
        uncertain_alpha = np.clip(np.asarray(uncertain, dtype=np.float32), 0.0, 1.0)
        if confident_alpha.ndim != 2:
            confident_alpha = np.zeros((1, 1), dtype=np.float32)
        if uncertain_alpha.ndim != 2:
            uncertain_alpha = np.zeros_like(confident_alpha)
        if confident_alpha.shape != uncertain_alpha.shape:
            uncertain_alpha = np.zeros_like(confident_alpha)
        height, width = confident_alpha.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        red = uncertain_alpha * float(uncertain_color.red()) + confident_alpha * float(confident_color.red())
        green = uncertain_alpha * float(uncertain_color.green()) + confident_alpha * float(confident_color.green())
        blue = uncertain_alpha * float(uncertain_color.blue()) + confident_alpha * float(confident_color.blue())
        alpha = np.maximum(confident_alpha, uncertain_alpha)
        rgba[..., 0] = np.clip(red, 0.0, 255.0).astype(np.uint8)
        rgba[..., 1] = np.clip(green, 0.0, 255.0).astype(np.uint8)
        rgba[..., 2] = np.clip(blue, 0.0, 255.0).astype(np.uint8)
        rgba[..., 3] = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
        image = QImage(rgba.data, width, height, int(rgba.strides[0]), QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(image)

    def _named_color(self, key: str) -> QColor:
        defaults = {
            "first_mask": QColor(80, 210, 255, 255),
            "second_mask": QColor(255, 0, 140, 255),
            "difference_mask": QColor(255, 196, 0, 255),
            "boundary_mask": QColor(255, 210, 0, 235),
        }
        value = (self._session_view_state.get("colors") or {}).get(key)
        if isinstance(value, str):
            color = QColor(value)
            if color.isValid():
                return color
        return QColor(defaults[key])

    def _set_named_color(self, key: str, color: QColor) -> None:
        if not color.isValid():
            return
        colors = dict(self._session_view_state.get("colors") or {})
        colors[key] = color.name(QColor.NameFormat.HexArgb)
        self._session_view_state["colors"] = colors
        self._overlay_cache.clear()
        self._derived_cache.clear()
        self._update_color_button_styles()
        self._refresh_scene(reset_view=False)
        self._refresh_info()
        self._store_view_settings()

    def _choose_named_color(self, key: str) -> None:
        initial = self._named_color(key)
        color = QColorDialog.getColor(initial, self, self._t("details.color_dialog_title"))
        if color.isValid():
            self._set_named_color(key, color)

    def _active_result_color_key(self) -> str | None:
        kind = self._selected_result_kind()
        if kind == "boundary":
            return "boundary_mask"
        if kind == "diff":
            return "difference_mask"
        return None

    def _choose_active_result_color(self) -> None:
        key = self._active_result_color_key()
        if key is not None:
            self._choose_named_color(key)

    def _update_color_button_styles(self) -> None:
        mapping = {
            self.first_mask_color_button: "first_mask",
            self.second_mask_color_button: "second_mask",
            self.result_mask_color_button: self._active_result_color_key() or "difference_mask",
        }
        for button, key in mapping.items():
            color = self._named_color(key)
            button.setStyleSheet(
                f"padding: 0px; border-radius: 4px; border: 1px solid #30445a; background-color: {color.name(QColor.NameFormat.HexRgb)};"
            )

    def _build_view_settings_payload(self) -> dict[str, object]:
        payload = {
            "layer_view": self._selected_layer_view(),
            "result_kind": self._selected_result_kind(),
            "grayscale_diff": bool(self.grayscale_diff_checkbox.isChecked()),
            "original_visible": bool(self.original_visible.isChecked()),
            "first_visible": bool(self.first_source_visible.isChecked()),
            "second_visible": bool(self.second_source_visible.isChecked()),
            "result_visible": bool(self.result_visible.isChecked()),
            "original_opacity": int(self.original_opacity.value()),
            "first_opacity": int(self.first_source_opacity.value()),
            "second_opacity": int(self.second_source_opacity.value()),
            "result_opacity": int(self.result_opacity.value()),
            "colors": dict(self._session_view_state.get("colors") or {}),
        }
        if self._tile_selection is not None:
            payload["tile_selection"] = {
                "row": int(self._tile_selection[0]),
                "column": int(self._tile_selection[1]),
            }
        if self._subpixel_selection is not None:
            payload["subpixel_selection"] = {
                "parent_row": int(self._subpixel_selection.parent_row),
                "parent_column": int(self._subpixel_selection.parent_column),
                "sub_row": int(self._subpixel_selection.sub_row),
                "sub_column": int(self._subpixel_selection.sub_column),
                "parent_value": float(self._subpixel_selection.parent_value),
                "subpixel_value": float(self._subpixel_selection.subpixel_value),
                "subpixel_confidence": None if self._subpixel_selection.subpixel_confidence is None else float(self._subpixel_selection.subpixel_confidence),
                "aggregation": str(self._subpixel_selection.aggregation),
                "metric_key": str(self._subpixel_selection.metric_key),
                "spec_rows": int(self._subpixel_selection.spec.rows if self._subpixel_selection.spec is not None else 0),
                "spec_columns": int(self._subpixel_selection.spec.columns if self._subpixel_selection.spec is not None else 0),
                "spec_mode": str(self._subpixel_selection.spec.mode if self._subpixel_selection.spec is not None else "grid"),
                "spec_tile_width": int(self._subpixel_selection.spec.tile_width if self._subpixel_selection.spec is not None else 0),
                "spec_tile_height": int(self._subpixel_selection.spec.tile_height if self._subpixel_selection.spec is not None else 0),
                "spec_overlap": int(self._subpixel_selection.spec.overlap if self._subpixel_selection.spec is not None else 0),
            }
        return payload
