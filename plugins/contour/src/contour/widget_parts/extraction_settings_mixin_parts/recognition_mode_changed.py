"""extraction_settings_mixin: recognition mode changed."""
from __future__ import annotations


class RecognitionModeChangedMixin:
    def _on_recognition_mode_changed(self, *_args) -> None:
        if not hasattr(self, "recognition_mode_combo") or not hasattr(self, "recognition_stack"):
            return
        data = str(self.recognition_mode_combo.currentData() or "conductors")
        if data == "disabled":
            self._active_extraction_profile = "conductors"
            self._sync_recognition_stack_visibility()
        elif data == "conductors":
            self._active_extraction_profile = "conductors"
            self._sync_recognition_stack_visibility()
            self._set_extraction_settings(self._contour_settings_profiles["conductors"])
        else:
            self._active_extraction_profile = "vias"
            self.recognition_stack.setVisible(False)
            self._set_extraction_settings(self._contour_settings_profiles["vias"])
        if hasattr(self, "via_group"):
            self.via_group.setVisible(False)
        if hasattr(self, "polygon_editor"):
            # "No extraction" disables recognition only; existing imported or
            # manually edited vectors remain part of the editor overlay.
            self.polygon_editor.set_polygon_overlays_visible(True)
            self.polygon_editor.set_via_debug_inspection_enabled(self._via_debug_inspection_enabled())
        self.polygon_editor.set_debug_candidates([])
        if hasattr(self, "_update_extraction_profile_controls_state"):
            self._update_extraction_profile_controls_state()
        self._sync_recognition_scene_frame()
        if data == "disabled":
            current = self._workspace.current_image_path
            has_cif = bool(current and self._find_matching_cif_path(current))
            self._restore_cif_overlay_after_recognition()
            if not has_cif and hasattr(self, "_apply_editor_vectors_for_frame"):
                state = self._workspace.current_state
                if current and state is not None:
                    self._apply_editor_vectors_for_frame(
                        current,
                        state,
                        list(state.polygons),
                        defer_heavy_overlays=False,
                    )
        elif hasattr(self, "_apply_editor_vectors_for_frame"):
            current = self._workspace.current_image_path
            state = self._workspace.current_state
            if current and state is not None:
                self._apply_editor_vectors_for_frame(
                    current,
                    state,
                    list(state.polygons),
                    defer_heavy_overlays=False,
                )
        if hasattr(self, "_apply_conductor_display_visibility"):
            self._apply_conductor_display_visibility(self._workspace.current_state)
        if hasattr(self, "_refresh_gradient_overlay"):
            self._refresh_gradient_overlay()
        self._on_extraction_settings_changed()
