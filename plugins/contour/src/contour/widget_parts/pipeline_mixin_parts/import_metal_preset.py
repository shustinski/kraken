"""pipeline_mixin: import metal preset."""
from __future__ import annotations
from contour.widget_parts._imports import (
    Path,
    QFileDialog,
    QMessageBox,
    QSignalBlocker,
    load_pipeline_config_from_path,
)


class ImportMetalPresetMixin:
    def _import_metal_preset(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Загрузить пресет распознавания" if self._ui_language == "ru" else "Import recognition preset",
            self._dialog_start_directory_from_line_edit(self.input_dir_edit),
            self._tr("json_file_filter"),
        )
        if not path:
            return
        file_payload = load_pipeline_config_from_path(path)
        if str(file_payload.get("format") or "") != "contour-metal-preset":
            QMessageBox.warning(
                self,
                "Пресет" if self._ui_language == "ru" else "Preset",
                (
                    "Файл не является пресетом распознавания проводников."
                    if self._ui_language == "ru"
                    else "File is not a conductor recognition preset."
                ),
            )
            return
        settings_payload = file_payload.get("settings")
        if not isinstance(settings_payload, dict):
            QMessageBox.warning(
                self,
                "Пресет" if self._ui_language == "ru" else "Preset",
                (
                    "В файле пресета отсутствуют настройки."
                    if self._ui_language == "ru"
                    else "Preset file has no settings."
                ),
            )
            return
        name = str(file_payload.get("name") or Path(path).stem).strip() or Path(path).stem
        if name in self._built_in_metal_presets():
            name = f"{name} (user)" if self._ui_language != "ru" else f"{name} (пользовательский)"
        self._user_metal_presets[name] = {
            str(key): value for key, value in settings_payload.items()
        }
        self._save_user_metal_presets()
        self._refresh_metal_preset_combo()
        self._apply_metal_preset_payload(dict(settings_payload))
        with QSignalBlocker(self.metal_preset_combo):
            index = self.metal_preset_combo.findText(name)
            if index >= 0:
                self.metal_preset_combo.setCurrentIndex(index)
        self._append_log(
            f"Пресет распознавания загружен: {path}"
            if self._ui_language == "ru"
            else f"Recognition preset imported: {path}"
        )
