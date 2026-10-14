"""pipeline_mixin: editor image region selected."""
from __future__ import annotations
from contour.widget_parts._imports import (
    blurred_via_preset_payload,
    built_in_via_presets,
    cv2,
    noisy_traces_via_preset_payload,
    np,
)


class EditorImageRegionSelectedMixin:
    def _ensure_via_template_metadata(self) -> None:
        count = len(self._via_template_images)
        while len(self._via_template_min_scores) < count:
            fallback_score = self._via_template_min_scores[-1] if self._via_template_min_scores else 0.6
            self._via_template_min_scores.append(float(fallback_score))
        while len(self._via_template_diameters) < count:
            fallback_diameter = self._via_template_diameters[-1] if self._via_template_diameters else 8
            self._via_template_diameters.append(max(1, int(fallback_diameter)))
        del self._via_template_min_scores[count:]
        del self._via_template_diameters[count:]

    def _move_via_template(self, source_row: int, target_row: int) -> None:
        count = len(self._via_template_images)
        if source_row < 0 or source_row >= count:
            return
        target_row = max(0, min(count - 1, int(target_row)))
        if source_row == target_row:
            return
        self._ensure_via_template_metadata()
        self._via_template_images.insert(target_row, self._via_template_images.pop(source_row))
        self._via_template_min_scores.insert(target_row, self._via_template_min_scores.pop(source_row))
        self._via_template_diameters.insert(target_row, self._via_template_diameters.pop(source_row))
        self._refresh_via_template_list()
        self._on_extraction_settings_changed()

    def _set_via_template_similarity(self, row: int, value: float) -> None:
        if 0 <= row < len(self._via_template_min_scores):
            self._via_template_min_scores[row] = max(0.0, min(1.0, float(value)))
            self._on_extraction_settings_changed()

    def _set_via_template_diameter(self, row: int, value: int) -> None:
        if 0 <= row < len(self._via_template_diameters):
            self._via_template_diameters[row] = max(1, int(value))
            self._on_extraction_settings_changed()

    def _normalize_via_template_images(self, payload: list[object]) -> list[np.ndarray]:
        templates: list[np.ndarray] = []
        for item in payload:
            try:
                image: np.ndarray = np.asarray(item, dtype=np.uint8)
            except (TypeError, ValueError):
                continue
            if image.ndim == 3:
                if image.shape[2] >= 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    image = image[:, :, 0]
            if image.ndim != 2 or image.shape[0] < 2 or image.shape[1] < 2:
                continue
            templates.append(image.copy())
        return templates

    def _on_editor_image_region_selected(self, x_coord: float, y_coord: float, width: float, height: float) -> None:
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_image_region_selection_mode(False)
        image = self._workspace.current_display_image()
        if image is None:
            return
        data = np.asarray(image)
        if data.size == 0:
            return
        left = max(0, int(np.floor(x_coord)))
        top = max(0, int(np.floor(y_coord)))
        right = min(data.shape[1], int(np.ceil(x_coord + width)))
        bottom = min(data.shape[0], int(np.ceil(y_coord + height)))
        if right - left < 2 or bottom - top < 2:
            return
        template = data[top:bottom, left:right].copy()
        if template.ndim == 3:
            if template.shape[2] >= 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                template = template[:, :, 0]
        self._ensure_via_template_metadata()
        similarity = self._via_template_min_scores[-1] if self._via_template_min_scores else 0.6
        diameter = self._via_template_diameters[-1] if self._via_template_diameters else 8
        self._via_template_images.append(template.astype(np.uint8, copy=False))
        self._via_template_min_scores.append(float(similarity))
        self._via_template_diameters.append(max(1, int(diameter)))
        self._refresh_via_template_list()
        self._on_extraction_settings_changed()
        self._append_log(
            self._tr(
                "via_template_added_log",
                "Добавлен шаблон контакта {width}x{height}. Всего шаблонов: {count}."
                if self._ui_language == "ru"
                else "Added contact template {width}x{height}. Total templates: {count}.",
                width=right - left,
                height=bottom - top,
                count=len(self._via_template_images),
            )
        )

    def _clear_via_templates(self, *_args) -> None:
        self._via_template_images.clear()
        self._via_template_min_scores.clear()
        self._via_template_diameters.clear()
        self._refresh_via_template_list()
        self._on_extraction_settings_changed()

    def _remove_via_template(self, row: int) -> None:
        if row < 0 or row >= len(self._via_template_images):
            return
        self._ensure_via_template_metadata()
        self._via_template_images.pop(row)
        self._via_template_min_scores.pop(row)
        self._via_template_diameters.pop(row)
        self._refresh_via_template_list()
        self._on_extraction_settings_changed()

    def _remove_selected_via_template(self, *_args) -> None:
        row = self.via_template_table.currentRow() if hasattr(self, "via_template_table") else -1
        self._remove_via_template(row)

    def _built_in_via_presets(self) -> dict[str, dict[str, object]]:
        return built_in_via_presets(self._ui_language)

    def _noisy_traces_via_preset_payload(self) -> dict[str, object]:
        return noisy_traces_via_preset_payload()

    def _blurred_via_preset_payload(self) -> dict[str, object]:
        return blurred_via_preset_payload()

    def _load_user_via_presets(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                str(key): value
                for key, value in payload.items()
                if str(key).startswith("heuristic_")
            }
            for name, payload in self._via_preset_settings_store.load().items()
        }

    def _save_user_via_presets(self) -> None:
        self._via_preset_settings_store.save(self._user_via_presets)

    def _refresh_via_preset_combo(self) -> None:
        if not hasattr(self, "via_preset_combo"):
            return
        current_name = self.via_preset_combo.currentText()
        self.via_preset_combo.clear()
        for name in self._built_in_via_presets():
            self.via_preset_combo.addItem(name, ("builtin", name))
        for name in sorted(self._user_via_presets):
            self.via_preset_combo.addItem(name, ("user", name))
        index = self.via_preset_combo.findText(current_name)
        if index >= 0:
            self.via_preset_combo.setCurrentIndex(index)

    def _current_via_preset_payload(self) -> dict[str, object]:
        payload = self._current_contour_settings().to_dict()
        return {
            key: value
            for key, value in payload.items()
            if key.startswith("heuristic_")
        }
