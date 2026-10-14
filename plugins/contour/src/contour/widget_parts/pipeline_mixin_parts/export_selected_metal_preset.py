"""pipeline_mixin: export selected metal preset."""
from __future__ import annotations
from contour.widget_parts._imports import (
    Path,
    QFileDialog,
    save_pipeline_config_to_path,
)


class ExportSelectedMetalPresetMixin:
    def _delete_selected_metal_preset(self) -> None:
        data = self.metal_preset_combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2 or data[0] != "user":
            return
        self._user_metal_presets.pop(str(data[1]), None)
        self._save_user_metal_presets()
        self._refresh_metal_preset_combo()

    def _selected_metal_preset_payload(self) -> tuple[str, dict[str, object]] | None:
        data = self.metal_preset_combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2:
            return None
        preset_type, preset_name = str(data[0]), str(data[1])
        payload = (
            self._built_in_metal_presets().get(preset_name)
            if preset_type == "builtin"
            else self._user_metal_presets.get(preset_name)
        )
        if not isinstance(payload, dict):
            return None
        return preset_name, dict(payload)

    def _export_selected_metal_preset(self) -> None:
        selected = self._selected_metal_preset_payload()
        if selected is None:
            payload = self._current_metal_preset_payload()
            default_name = "metal_preset"
        else:
            default_name, payload = selected
        safe_name = "".join(
            character if character not in '<>:"/\\|?*' else "_"
            for character in default_name
        ).strip(" .") or "metal_preset"
        start_directory = self._dialog_start_directory_from_line_edit(self.output_dir_edit)
        suggested_path = str(Path(start_directory) / f"{safe_name}.json")
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Выгрузить пресет распознавания" if self._ui_language == "ru" else "Export recognition preset",
            suggested_path,
            self._tr("json_file_filter"),
        )
        if not path:
            return
        save_pipeline_config_to_path(
            path,
            {
                "format": "contour-metal-preset",
                "version": 1,
                "name": default_name,
                "settings": payload,
            },
        )
        self._append_log(
            f"Пресет распознавания выгружен: {path}"
            if self._ui_language == "ru"
            else f"Recognition preset exported: {path}"
        )
