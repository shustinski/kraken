"""details_dialog: restore view settings."""
from __future__ import annotations
from PyQt6.QtGui import QPixmap


class RestoreViewSettingsMixin:
    def _restore_view_settings(self) -> None:
        payload = dict(self._session_view_state or {})
        layer_view = payload.get("layer_view")
        if layer_view in {"binary", "source"}:
            index = self.layer_view_combo.findData(layer_view)
            if index >= 0:
                self.layer_view_combo.setCurrentIndex(index)
        self._restored_result_kind = str(payload.get("result_kind") or "") or None
        self._sticky_result_kind = self._restored_result_kind
        self.grayscale_diff_checkbox.setChecked(bool(payload.get("grayscale_diff", False)))
        self.original_visible.setChecked(bool(payload.get("original_visible", self.original_visible.isChecked())))
        self.first_source_visible.setChecked(bool(payload.get("first_visible", self.first_source_visible.isChecked())))
        self.second_source_visible.setChecked(bool(payload.get("second_visible", self.second_source_visible.isChecked())))
        self.result_visible.setChecked(bool(payload.get("result_visible", self.result_visible.isChecked())))
        self.original_opacity.setValue(int(payload.get("original_opacity", self.original_opacity.value())))
        self.first_source_opacity.setValue(int(payload.get("first_opacity", self.first_source_opacity.value())))
        self.second_source_opacity.setValue(int(payload.get("second_opacity", self.second_source_opacity.value())))
        self.result_opacity.setValue(int(payload.get("result_opacity", self.result_opacity.value())))
        self._update_color_button_styles()

    def _store_view_settings(self, *_args) -> None:
        payload = self._build_view_settings_payload()
        self._session_view_state.clear()
        self._session_view_state.update(payload)
        self._sticky_result_kind = str(payload.get("result_kind") or "") or self._sticky_result_kind
        callback = self._on_view_state_changed
        if callable(callback):
            callback(dict(self._session_view_state))

    def _activate_context_hold(self) -> None:
        if self._result_kind_requires_confidence():
            model_id = self._confidence_model_id()
            source_key = f"model:{model_id}" if model_id else None
            pixmap = self._source_pixmap(source_key, self._named_color("first_mask"), prefer_grayscale=True) if source_key else QPixmap()
            if not pixmap.isNull():
                self._legacy_base_hold_active = True
                self._hold_preview_mode = "confidence_source"
                self._hold_preview_pixmap = pixmap
                self._update_layer_states()
                return
        self._hold_preview_mode = "base"
        self._activate_base_hold()
