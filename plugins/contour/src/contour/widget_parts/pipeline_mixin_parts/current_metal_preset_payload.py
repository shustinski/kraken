"""pipeline_mixin: current metal preset payload."""
from __future__ import annotations
from contour.value_conversion import to_float
from contour.widget_parts._imports import (
    ContourExtractionSettings,
    Path,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    built_in_metal_presets,
    save_pipeline_config_to_path,
)


class CurrentMetalPresetPayloadMixin:
    def _apply_selected_via_preset(self) -> None:
        data = self.via_preset_combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2:
            return
        preset_type, preset_name = data
        payload = (
            self._built_in_via_presets().get(str(preset_name))
            if preset_type == "builtin"
            else self._user_via_presets.get(str(preset_name))
        )
        if payload:
            self._apply_via_preset_payload(payload)

    def _save_current_via_preset(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "Сохранить пресет" if self._ui_language == "ru" else "Save preset",
            "Имя пресета:" if self._ui_language == "ru" else "Preset name:",
        )
        name = str(name).strip()
        if not ok or not name:
            return
        expert_payload = self._current_via_preset_payload()
        safe_name = "".join(
            character if character not in '<>:"/\\|?*' else "_"
            for character in name
        ).strip(" .") or "via_expert_preset"
        start_directory = self._dialog_start_directory_from_line_edit(self.output_dir_edit)
        suggested_path = str(Path(start_directory) / f"{safe_name}.json")
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить пресет контактов" if self._ui_language == "ru" else "Save contact preset",
            suggested_path,
            self._tr("json_file_filter"),
        )
        if not path:
            return
        save_pipeline_config_to_path(
            path,
            {
                "format": "contour-via-expert-preset",
                "version": 1,
                "name": name,
                "expert_settings": expert_payload,
                # Kept as a complete snapshot for audit/reproducibility. Loading
                # this preset applies expert_settings only.
                "recognition_settings": self._current_contour_settings().to_dict(),
            },
        )
        self._user_via_presets[name] = expert_payload
        self._save_user_via_presets()
        self._refresh_via_preset_combo()
        index = self.via_preset_combo.findText(name)
        if index >= 0:
            self.via_preset_combo.setCurrentIndex(index)

    def _delete_selected_via_preset(self) -> None:
        data = self.via_preset_combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2 or data[0] != "user":
            return
        self._user_via_presets.pop(str(data[1]), None)
        self._save_user_via_presets()
        self._refresh_via_preset_combo()

    def _built_in_metal_presets(self) -> dict[str, dict[str, object]]:
        return built_in_metal_presets(self._ui_language)

    def _load_user_metal_presets(self) -> dict[str, dict[str, object]]:
        return self._metal_preset_settings_store.load()

    def _save_user_metal_presets(self) -> None:
        self._metal_preset_settings_store.save(self._user_metal_presets)

    def _refresh_metal_preset_combo(self) -> None:
        if not hasattr(self, "metal_preset_combo"):
            return
        current_name = self.metal_preset_combo.currentText()
        self.metal_preset_combo.clear()
        for name in self._built_in_metal_presets():
            self.metal_preset_combo.addItem(name, ("builtin", name))
        for name in sorted(self._user_metal_presets):
            self.metal_preset_combo.addItem(name, ("user", name))
        index = self.metal_preset_combo.findText(current_name)
        if index >= 0:
            self.metal_preset_combo.setCurrentIndex(index)

    def _current_metal_preset_payload(self) -> dict[str, object]:
        payload = self._current_contour_settings().to_dict()
        keys = (
            "metal_preset",
            "metal_noise_suppression",
            "metal_min_contrast",
            "metal_min_object_source_contrast",
            "metal_min_object_rim_contrast",
            "metal_min_object_rim_area_fraction",
            "metal_min_hole_source_contrast",
            "metal_min_hole_source_contrast_fraction",
            "metal_segmentation_strategy",
            "metal_strategy_parameters",
            "metal_auto_contrast_step",
            "metal_auto_source_contrast_step",
            "metal_auto_directional_gap_bridge_px",
            "metal_auto_directional_gap_min_source_intensity",
            "metal_gap_bridge_px",
            "metal_speckle_removal_px",
            "metal_contour_smooth_px",
            "metal_min_trace_width_px",
            "metal_conductor_size_offset_px",
            "metal_max_trace_width_px",
            "metal_min_trace_length_px",
            "metal_min_straightness",
            "metal_min_area",
            "metal_min_perimeter",
            "metal_use_wide_conductor_gradient",
            "metal_watershed_smoothing_sigma",
            "metal_watershed_core_margin",
            "metal_watershed_groove_margin",
            "metal_watershed_rim_probe_px",
            "metal_watershed_seed_speckle_px",
            "metal_watershed_valley_span_px",
            "metal_watershed_valley_depth",
            "metal_adaptive_block_size",
            "metal_adaptive_c",
            "metal_adaptive_method",
            "metal_random_walker_beta",
            "metal_random_walker_iterations",
            "metal_graph_cut_iterations",
            "metal_reconstruction_erode_px",
            "metal_boundary_relief",
            "metal_boundary_background_sigma",
            "metal_structural_variant",
            "metal_allowed_angles",
            "metal_angle_tolerance_deg",
            "epsilon",
        )
        return {key: payload[key] for key in keys if key in payload}

    def _apply_metal_preset_payload(self, payload: dict[str, object]) -> None:
        normalized_payload = dict(payload)
        if "metal_min_contrast" not in normalized_payload and "metal_contrast_bias" in normalized_payload:
            normalized_payload["metal_min_contrast"] = max(
                1.0,
                to_float(normalized_payload.get("metal_contrast_bias", 0.0)),
            )
        merged = ContourExtractionSettings.from_dict(
            self._current_contour_settings().to_dict() | normalized_payload
        )
        self._set_extraction_settings(merged)
        self._on_extraction_settings_changed()

    def _apply_selected_metal_preset(self) -> None:
        data = self.metal_preset_combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2:
            return
        preset_type, preset_name = data
        payload = (
            self._built_in_metal_presets().get(str(preset_name))
            if preset_type == "builtin"
            else self._user_metal_presets.get(str(preset_name))
        )
        if payload:
            self._apply_metal_preset_payload(payload)

    def _save_current_metal_preset(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "Сохранить пресет" if self._ui_language == "ru" else "Save preset",
            "Имя пресета:" if self._ui_language == "ru" else "Preset name:",
        )
        name = str(name).strip()
        if not ok or not name:
            return
        if name in self._built_in_metal_presets():
            QMessageBox.warning(
                self,
                "Пресет" if self._ui_language == "ru" else "Preset",
                (
                    "Нельзя перезаписать встроенный пресет. Выберите другое имя."
                    if self._ui_language == "ru"
                    else "Built-in presets cannot be overwritten. Choose another name."
                ),
            )
            return
        self._user_metal_presets[name] = self._current_metal_preset_payload()
        self._save_user_metal_presets()
        self._refresh_metal_preset_combo()
        index = self.metal_preset_combo.findText(name)
        if index >= 0:
            self.metal_preset_combo.setCurrentIndex(index)
