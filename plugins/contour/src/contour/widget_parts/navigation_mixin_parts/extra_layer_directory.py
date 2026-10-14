"""navigation_mixin: extra layer directory."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    Path,
    QAbstractSpinBox,
    QFileDialog,
    QMessageBox,
    QPixmap,
    QSpinBox,
    Qt,
    build_additional_layer_frame_map,
    scan_image_files,
)


class ExtraLayerDirectoryMixin:
    @staticmethod
    def _configure_extra_layer_spinbox(spinbox: QSpinBox, width: int) -> None:
        spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spinbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spinbox.setFixedWidth(width)

    @staticmethod
    def _compact_spinbox_width(spinbox: QSpinBox, sample_text: str) -> int:
        text_width = spinbox.fontMetrics().horizontalAdvance(sample_text)
        style = spinbox.style()
        assert style is not None
        arrow_width = max(24, style.pixelMetric(style.PixelMetric.PM_ScrollBarExtent))
        frame = 16
        return text_width + arrow_width + frame

    def _set_extra_layer_field(self: Any, layer_id: int, key: str, value: object) -> None:
        for layer in self._extra_layers:
            if int(layer.get("id", -1)) == layer_id:
                layer[key] = value
                self._sync_extra_layers()
                return

    def _remove_extra_layer_by_id(self: Any, layer_id: int) -> None:
        self._extra_layers = [layer for layer in self._extra_layers if int(layer.get("id", -1)) != layer_id]
        self._refresh_extra_layers_list()
        self._sync_extra_layers()

    def _clear_extra_layers(self: Any) -> None:
        self._extra_layers.clear()
        self._refresh_extra_layers_list()
        self._sync_extra_layers()

    def _update_extra_layers_enabled_state(self: Any) -> None:
        has_base = bool(self._workspace.image_paths)
        if hasattr(self, "add_extra_layers_button"):
            self.add_extra_layers_button.setEnabled(has_base)

    def _sync_extra_layers(self: Any) -> None:
        if not hasattr(self, "polygon_editor"):
            return
        current_path = self._workspace.current_image_path
        current_number = self._base_frame_number_by_path.get(str(Path(current_path))) if current_path else None
        active_layers: list[dict[str, object]] = []
        if current_number is not None:
            for layer in self._extra_layers:
                if not bool(layer.get("visible", True)):
                    continue
                frame_map = layer.get("frame_map")
                if not isinstance(frame_map, dict):
                    continue
                layer_image_path = frame_map.get(current_number)
                if not isinstance(layer_image_path, str):
                    continue
                pixmap_cache = layer.setdefault("_pixmap_cache", {})
                pixmap = pixmap_cache.get(layer_image_path) if isinstance(pixmap_cache, dict) else None
                if not isinstance(pixmap, QPixmap):
                    pixmap = QPixmap(layer_image_path)
                    if isinstance(pixmap_cache, dict) and not pixmap.isNull():
                        pixmap_cache[layer_image_path] = pixmap
                if pixmap.isNull():
                    continue
                active_layers.append(
                    {
                        "name": layer.get("name", ""),
                        "visible": bool(layer.get("visible", True)),
                        "opacity": max(0.0, min(1.0, float(layer.get("opacity", 100)) / 100.0)),
                        "dx": float(layer.get("dx", 0) or 0),
                        "dy": float(layer.get("dy", 0) or 0),
                        "pixmap": pixmap,
                    }
                )
        self.polygon_editor.set_extra_layers(active_layers)

    def _load_extra_layers(self: Any) -> None:
        if not self._workspace.image_paths:
            QMessageBox.information(self, "Contour", "Сначала загрузите базовый слой")
            return
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку дополнительного слоя"
            if self._ui_language == "ru"
            else "Select additional layer directory",
            self._dialog_start_directory_from_line_edit(self.input_dir_edit),
        )
        if not folder_path:
            return
        layer = self._extra_layer_from_directory(folder_path)
        if layer is None:
            return
        self._extra_layers.append(layer)
        self._refresh_extra_layers_list()
        self._sync_extra_layers()

    def _extra_layer_from_directory(self: Any, directory_path: str) -> dict[str, object] | None:
        path = Path(directory_path.strip().strip("\"'")).expanduser()
        if not path.is_dir():
            self._append_log(
                self._tr(
                    "extra_layer_missing_file_log",
                    "Папка слоя не найдена: {path}" if self._ui_language == "ru" else "Layer folder not found: {path}",
                    path=directory_path,
                )
            )
            return None
        layer_image_paths = scan_image_files(path)
        frame_map, warnings = build_additional_layer_frame_map(
            layer_image_paths,
            base_frame_numbers=set(self._base_frame_numbers),
        )
        for message in warnings:
            self._append_log(message)
        if not frame_map:
            self._append_log(
                self._tr(
                    "extra_layer_load_failed_log",
                    "В папке слоя нет кадров, совпадающих с базовым слоем: {path}"
                    if self._ui_language == "ru"
                    else "Layer folder has no frames matching base layer: {path}",
                    path=str(path),
                )
            )
            return None
        layer_id = self._next_extra_layer_id
        self._next_extra_layer_id += 1
        return {
            "id": layer_id,
            "name": path.name,
            "folder_path": str(path),
            "frame_map": frame_map,
            "visible": True,
            "opacity": 100,
            "dx": 0,
            "dy": 0,
        }

    def _on_extra_layers_rows_moved(self: Any, *_args) -> None:
        if not hasattr(self, "extra_layers_list"):
            return
        order: list[int] = []
        for row in range(self.extra_layers_list.count()):
            item = self.extra_layers_list.item(row)
            if item is None:
                continue
            try:
                order.append(int(item.data(Qt.ItemDataRole.UserRole)))
            except (TypeError, ValueError):
                continue
        if not order:
            return
        by_id = {int(layer.get("id", -1)): layer for layer in self._extra_layers}
        self._extra_layers = [by_id[layer_id] for layer_id in order if layer_id in by_id]
        self._sync_extra_layers()
