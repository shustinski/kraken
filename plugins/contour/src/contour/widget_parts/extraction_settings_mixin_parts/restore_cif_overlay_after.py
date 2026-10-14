"""extraction_settings_mixin: restore cif overlay after."""
from __future__ import annotations
from contour.widget_parts._imports import (
    RECOGNITION_SCENE_FRAME_STYLE,
    Path,
    QSignalBlocker,
    metal_preset_table,
    offset_conductor_polygons,
)


class RestoreCifOverlayAfterMixin:
    def _sync_recognition_scene_frame(self) -> None:
        if not hasattr(self, "editor_scene_frame"):
            return
        enabled = False
        if hasattr(self, "recognition_mode_combo"):
            enabled = str(self.recognition_mode_combo.currentData() or "") != "disabled"
        self.editor_scene_frame.setStyleSheet(RECOGNITION_SCENE_FRAME_STYLE if enabled else "")

    def _restore_cif_overlay_after_recognition(self) -> None:
        current = self._workspace.current_image_path
        if not current or not hasattr(self, "polygon_editor"):
            return
        if not self._find_matching_cif_path(current):
            return
        self._workspace.forget_cleared_vectors([str(Path(current))])
        polygons = self._load_cif_overlay_polygons(current)
        if Path(current).stem.lower() in self._cif_load_failure_stems:
            return
        clones = [polygon.clone() for polygon in polygons]
        state = self._workspace.current_state
        if state is not None and state.image_path == current:
            state.reference_polygons = [polygon.clone() for polygon in clones]
            self._workspace.update_current_polygons(clones)
            state.polygons_dirty = False
        self._updating_views = True
        try:
            self.polygon_editor.set_polygons([polygon.clone() for polygon in clones], emit_signal=False)
            if hasattr(self, "_polygons_editor_signature"):
                self._editor_polygons_signature = self._polygons_editor_signature(current, clones)
        finally:
            self._updating_views = False
        if hasattr(self, "_update_frame_item_status"):
            self._update_frame_item_status(current)
        if hasattr(self, "_update_vector_edit_status_label"):
            self._update_vector_edit_status_label()
        if hasattr(self, "_schedule_invalid_polygon_description_offer"):
            self._schedule_invalid_polygon_description_offer(current)

    def _on_via_brightness_range_changed(self, *_args) -> None:
        if (
            hasattr(self, "via_white_range_checkbox")
            and hasattr(self, "bright_via_bright_center_score_spin")
            and self.via_white_range_checkbox.isChecked()
        ):
            with QSignalBlocker(self.bright_via_bright_center_score_spin):
                self.bright_via_bright_center_score_spin.setValue(float(self.via_white_range_min_spin.value()))
        if hasattr(self, "_update_via_brightness_range_controls_state"):
            self._update_via_brightness_range_controls_state()
        self._on_extraction_settings_changed()

    def _on_via_display_settings_changed(self, *_args) -> None:
        if hasattr(self, "polygon_editor") and hasattr(self, "via_show_detected_checkbox"):
            self.polygon_editor.set_polygon_category_visible("via", self.via_show_detected_checkbox.isChecked())
        self._on_extraction_settings_changed()

    def _current_metal_preset_key(self) -> str:
        if not hasattr(self, "metal_preset_combo"):
            return "standard"
        data = self.metal_preset_combo.currentData()
        if isinstance(data, tuple) and len(data) == 2:
            preset_type, preset_name = str(data[0]), str(data[1])
            if preset_type == "builtin":
                payload = self._built_in_metal_presets().get(preset_name, {})
                return str(payload.get("metal_preset", "standard") or "standard")
        return "standard"

    def _on_conductor_size_offset_changed(self, value: int) -> None:
        if hasattr(self, "metal_conductor_size_offset_value_label"):
            self.metal_conductor_size_offset_value_label.setText(str(int(value)))
        if not hasattr(self, "_workspace"):
            return
        state = self._workspace.current_state
        image_path = self._workspace.current_image_path
        if state is None or not image_path:
            return
        if not self._state_has_recognition_result(state):
            return
        base = list(getattr(state, "recognition_base_polygons", None) or [])
        if not base:
            base = [polygon.clone() for polygon in state.polygons]
            state.recognition_base_polygons = [polygon.clone() for polygon in base]
        buffered = offset_conductor_polygons(base, float(value))
        self._workspace.update_current_polygons(buffered)
        if hasattr(self, "_sync_polygons_to_editor"):
            self._sync_polygons_to_editor(image_path, buffered)
        if hasattr(self, "_update_frame_item_status"):
            self._update_frame_item_status(image_path)

    def _on_metal_overlay_opacity_changed(self, value: float) -> None:
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_gradient_overlay_opacity(float(value))
        self._on_metal_display_changed()

    def _on_metal_display_changed(self, *_args) -> None:
        self._apply_conductor_display_visibility(self._workspace.current_state)
        if hasattr(self, "_refresh_gradient_overlay"):
            self._refresh_gradient_overlay()

    def _metal_preset_table(self) -> dict[str, dict[str, object]]:
        return metal_preset_table()

    def _on_metal_preset_changed(self, *_args) -> None:
        if not hasattr(self, "metal_preset_combo"):
            return
        data = self.metal_preset_combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2 or data[0] != "builtin":
            self._on_extraction_settings_changed()
            return
        payload = self._built_in_metal_presets().get(str(data[1]))
        if payload:
            self._apply_metal_preset_payload(payload)

    def _preview_metal_mask(self, *_args) -> None:
        if hasattr(self, "metal_debug_visual_combo"):
            idx = self.metal_debug_visual_combo.findData("metal_binary_mask")
            if idx >= 0:
                self.metal_debug_visual_combo.setCurrentIndex(idx)
        if hasattr(self, "metal_show_mask_checkbox"):
            self.metal_show_mask_checkbox.setChecked(True)
        self._refresh_gradient_overlay()
