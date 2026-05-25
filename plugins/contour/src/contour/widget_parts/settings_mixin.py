from __future__ import annotations

from typing import Any

from ._imports import *  # noqa: F403


class WidgetSettingsMixin:
    @staticmethod
    def _apply_contour_application_icon() -> None:
        app = QApplication.instance()
        if not isinstance(app, QApplication) or not app.windowIcon().isNull():
            return
        plugin_root = Path(__file__).resolve().parents[2]
        for icon_path in (
            plugin_root / "resources" / "icons" / "contour.png",
            plugin_root / "resources" / "icons" / "contour.ico",
        ):
            if icon_path.exists():
                app.setWindowIcon(QIcon(str(icon_path)))
                return

    def _build_ui(self: Any) -> None:
        return build_ui(self)

    def _apply_compact_ui_style(self: Any) -> None:
        self.setStyleSheet(COMPACT_UI_STYLE)

    def _build_path_panel(self: Any) -> QWidget:
        return build_path_panel(self)

    def _build_paths_tab(self: Any) -> QWidget:
        return build_paths_tab(self)

    def _build_tabs(self: Any) -> QWidget:
        return build_tabs(self)

    def _restore_persisted_paths(self: Any) -> None:
        paths = self._path_settings.load()

        if paths.output_directory:
            self.output_dir_edit.setText(paths.output_directory)
        if paths.dataset_directory:
            self.dataset_dir_edit.setText(paths.dataset_directory)
        if paths.cif_directory:
            self.cif_dir_edit.setText(paths.cif_directory)
        if paths.input_directory:
            self.input_dir_edit.setText(paths.input_directory)

    def _save_persisted_paths(self: Any) -> None:
        self._path_settings.save(
            PersistedPaths(
                input_directory=self.input_dir_edit.text().strip(),
                cif_directory=self.cif_dir_edit.text().strip(),
                output_directory=self.output_dir_edit.text().strip(),
                dataset_directory=self.dataset_dir_edit.text().strip(),
            )
        )

    def _restore_persisted_display_settings(self: Any) -> None:
        payload = self._display_settings_store.load()
        self._display_settings = DisplaySettings.from_dict(payload)
        if not hasattr(self, "line_width_spin"):
            return

        blockers = [
            QSignalBlocker(self.line_width_spin),
            QSignalBlocker(self.vertex_size_spin),
            QSignalBlocker(self.fill_opacity_spin),
            QSignalBlocker(self.show_vertices_checkbox),
            QSignalBlocker(self.show_labels_checkbox),
            QSignalBlocker(self.random_object_colors_checkbox),
            QSignalBlocker(self.show_frame_matrix_checkbox),
            QSignalBlocker(self.show_frame_matrix_thumbnails_checkbox),
            QSignalBlocker(self.show_neighbor_frames_checkbox),
            QSignalBlocker(self.show_neighbor_vectors_checkbox),
            QSignalBlocker(self.neighbor_columns_spin),
            QSignalBlocker(self.neighbor_max_grid_spin),
            QSignalBlocker(self.neighbor_opacity_spin),
            QSignalBlocker(self.neighbor_overlap_spin),
            QSignalBlocker(self.autosave_on_frame_transition_checkbox),
        ]
        if hasattr(self, "vector_geom_clip_checkbox"):
            blockers.extend(
                [
                    QSignalBlocker(self.vector_geom_clip_checkbox),
                    QSignalBlocker(self.vector_geom_min_outer_spin),
                    QSignalBlocker(self.vector_geom_min_hole_spin),
                    QSignalBlocker(self.vector_geom_merge_checkbox),
                    QSignalBlocker(self.vector_geom_spike_angle_spin),
                    QSignalBlocker(self.vector_geom_drop_triangle_checkbox),
                ]
            )
        self._restoring_display_settings = True
        try:
            self._update_color_button(self.external_color_button, self._display_settings.external_color)
            self._update_color_button(self.hole_color_button, self._display_settings.hole_color)
            self._update_color_button(self.selected_color_button, self._display_settings.selected_color)
            self._update_color_button(
                self.conductor_hover_highlight_color_button, self._display_settings.conductor_hover_highlight_color
            )
            self._update_color_button(self.vertex_color_button, self._display_settings.vertex_color)
            self.line_width_spin.setValue(float(self._display_settings.line_width))
            self.vertex_size_spin.setValue(float(self._display_settings.vertex_size))
            self.fill_opacity_spin.setValue(float(self._display_settings.fill_opacity))
            self.show_vertices_checkbox.setChecked(bool(self._display_settings.show_vertices))
            self.show_labels_checkbox.setChecked(bool(self._display_settings.show_labels))
            self.random_object_colors_checkbox.setChecked(bool(payload.get("random_object_colors", False)))
            self.show_frame_matrix_checkbox.setChecked(bool(payload.get("show_frame_matrix", True)))
            self.show_frame_matrix_thumbnails_checkbox.setChecked(
                bool(payload.get("show_frame_matrix_thumbnails", True))
            )
            self.show_neighbor_frames_checkbox.setChecked(bool(payload.get("show_neighbor_frames", False)))
            self.show_neighbor_vectors_checkbox.setChecked(bool(payload.get("show_neighbor_vectors", False)))
            self.neighbor_columns_spin.setValue(max(1, int(payload.get("neighbor_columns", 3))))
            self.neighbor_max_grid_spin.setValue(self._odd_neighbor_grid_size(int(payload.get("neighbor_max_grid", 7))))
            self.neighbor_opacity_spin.setValue(float(payload.get("neighbor_opacity", 0.35)))
            self.neighbor_overlap_spin.setValue(max(0, int(payload.get("neighbor_overlap_pixels", 0))))
            self.autosave_on_frame_transition_checkbox.setChecked(
                bool(payload.get("autosave_on_frame_transition", False))
            )
            self._restore_main_splitter_sizes(payload.get("main_splitter_sizes"))
            if hasattr(self, "vector_geom_clip_checkbox"):
                self.vector_geom_clip_checkbox.setChecked(bool(payload.get("vector_geom_clip_on_sync", True)))
                self.vector_geom_min_outer_spin.setValue(float(payload.get("vector_geom_min_outer_area", 9.0)))
                self.vector_geom_min_hole_spin.setValue(float(payload.get("vector_geom_min_hole_area", 0.0)))
                self.vector_geom_merge_checkbox.setChecked(bool(payload.get("vector_geom_merge_on_edit", True)))
                self.vector_geom_spike_angle_spin.setValue(float(payload.get("vector_geom_spike_angle_deg", 30.0)))
                self.vector_geom_drop_triangle_checkbox.setChecked(bool(payload.get("vector_geom_drop_triangles", True)))
        finally:
            self._restoring_display_settings = False
            del blockers
        self._sync_neighbor_frames()
        self._sync_frame_matrix_controls()
        if self._frame_matrix_enabled():
            self._schedule_thumbnail_grid_rebuild(force=True)
        else:
            self._disable_frame_matrix_runtime()
        self._apply_vector_geometry_editor_config()

    def _current_display_settings_payload(self: Any) -> dict[str, object]:
        payload_out: dict[str, object] = {
            **self._display_settings.to_dict(),
            "random_object_colors": bool(self.random_object_colors_checkbox.isChecked()),
            "show_frame_matrix": bool(self.show_frame_matrix_checkbox.isChecked()),
            "show_frame_matrix_thumbnails": bool(self.show_frame_matrix_thumbnails_checkbox.isChecked()),
            "show_neighbor_frames": bool(self.show_neighbor_frames_checkbox.isChecked()),
            "show_neighbor_vectors": bool(self.show_neighbor_vectors_checkbox.isChecked()),
            "neighbor_columns": int(self.neighbor_columns_spin.value()),
            "neighbor_max_grid": int(self.neighbor_max_grid_spin.value()),
            "neighbor_opacity": float(self.neighbor_opacity_spin.value()),
            "neighbor_overlap_pixels": int(self.neighbor_overlap_spin.value()),
            "autosave_on_frame_transition": bool(self.autosave_on_frame_transition_checkbox.isChecked()),
            "main_splitter_sizes": self.main_splitter.sizes() if hasattr(self, "main_splitter") else [],
        }
        if hasattr(self, "vector_geom_clip_checkbox"):
            payload_out.update(
                {
                    "vector_geom_clip_on_sync": bool(self.vector_geom_clip_checkbox.isChecked()),
                    "vector_geom_min_outer_area": float(self.vector_geom_min_outer_spin.value()),
                    "vector_geom_min_hole_area": float(self.vector_geom_min_hole_spin.value()),
                    "vector_geom_merge_on_edit": bool(self.vector_geom_merge_checkbox.isChecked()),
                    "vector_geom_spike_angle_deg": float(self.vector_geom_spike_angle_spin.value()),
                    "vector_geom_drop_triangles": bool(self.vector_geom_drop_triangle_checkbox.isChecked()),
                }
            )
        return payload_out

    def _save_persisted_display_settings(self: Any) -> None:
        if self._restoring_display_settings or not hasattr(self, "line_width_spin"):
            return
        self._display_settings_store.save(self._current_display_settings_payload())

    def _save_persisted_current_image_path(self: Any, image_path: str | Path | None) -> None:
        if hasattr(self, "_session_settings_store"):
            self._session_settings_store.save_current_image_path(image_path)

    def _persist_session_state(self: Any) -> None:
        self._save_persisted_display_settings()
        current_path = self._workspace.current_image_path if hasattr(self, "_workspace") else None
        self._save_persisted_current_image_path(current_path)

    def _vector_geometry_settings_from_widgets(self: Any) -> VectorGeometrySettings:
        if not hasattr(self, "vector_geom_clip_checkbox"):
            return VectorGeometrySettings()
        return VectorGeometrySettings(
            clip_to_frame_on_sync=bool(self.vector_geom_clip_checkbox.isChecked()),
            min_outer_area_px2=float(self.vector_geom_min_outer_spin.value()),
            min_hole_area_to_remove_px2=float(self.vector_geom_min_hole_spin.value()),
            merge_overlapping_on_edit=bool(self.vector_geom_merge_checkbox.isChecked()),
            min_spike_interior_angle_deg=float(self.vector_geom_spike_angle_spin.value()),
            drop_three_vertex_triangle_artifacts=bool(self.vector_geom_drop_triangle_checkbox.isChecked()),
        )

    def _apply_vector_geometry_editor_config(self: Any) -> None:
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_vector_geometry_settings(self._vector_geometry_settings_from_widgets())

    def _on_min_inner_hole_area_changed(self: Any, *_args) -> None:
        self._on_extraction_settings_changed()

    def _set_default_extraction_disabled(self: Any) -> None:
        if not hasattr(self, "recognition_mode_combo"):
            return
        idx = self.recognition_mode_combo.findData("disabled")
        if idx < 0:
            return
        with QSignalBlocker(self.recognition_mode_combo):
            self.recognition_mode_combo.setCurrentIndex(idx)
        self._active_extraction_profile = "conductors"
        self._sync_recognition_stack_visibility()
        if hasattr(self, "_set_recognition_status"):
            self._set_recognition_status("disabled")

    def _on_vector_geom_control_changed(self: Any, *_args) -> None:
        self._apply_vector_geometry_editor_config()
        self._save_persisted_display_settings()

    def _show_manual_tool_postprocess_dialog(self: Any) -> None:
        existing = getattr(self, "_manual_tool_postprocess_dialog", None)
        if isinstance(existing, QDialog):
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return
        dialog = QDialog(self)
        self._manual_tool_postprocess_dialog = dialog
        dialog.setObjectName("manualToolPostprocessDialog")
        dialog.setWindowTitle("Постобработка ручных инструментов")
        dialog.resize(460, 320)
        dialog.setMinimumSize(420, 280)

        root = QVBoxLayout(dialog)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll, 1)

        container = QWidget()
        form = QFormLayout(container)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        scroll.setWidget(container)

        form.addRow(self.vector_geom_clip_checkbox)
        form.addRow("Минимальная площадь внешнего объекта, px²", self.vector_geom_min_outer_spin)
        self.vector_geom_min_outer_label_widget = form.labelForField(self.vector_geom_min_outer_spin)
        form.addRow("Минимальная площадь внутренней области для заливки, px²", self.vector_geom_min_hole_spin)
        self.vector_geom_min_hole_label_widget = form.labelForField(self.vector_geom_min_hole_spin)
        form.addRow(self.vector_geom_merge_checkbox)
        form.addRow("Минимальный угол острого выброса, °", self.vector_geom_spike_angle_spin)
        self.vector_geom_spike_angle_label_widget = form.labelForField(self.vector_geom_spike_angle_spin)
        form.addRow(self.vector_geom_drop_triangle_checkbox)

        close_button = QPushButton("Закрыть" if self._ui_language == "ru" else "Close")
        close_button.clicked.connect(dialog.accept)
        root.addWidget(close_button, 0, Qt.AlignmentFlag.AlignRight)
        dialog.exec()

    def _display_image_dimensions_for_vectors(self: Any) -> tuple[int, int]:
        frame = self._display_image_for_current_state()
        if frame is None:
            return (0, 0)
        shape = getattr(frame, "shape", None)
        if isinstance(shape, tuple) and len(shape) >= 2:
            return (int(shape[1]), int(shape[0]))
        return (0, 0)

    def _restore_main_splitter_sizes(self: Any, raw_sizes: object) -> None:
        if not hasattr(self, "main_splitter"):
            return
        if not isinstance(raw_sizes, (list, tuple)):
            return
        try:
            sizes = [max(1, int(value)) for value in raw_sizes]
        except (TypeError, ValueError):
            return
        if len(sizes) != self.main_splitter.count() or sum(sizes) <= 0:
            return
        self.main_splitter.setSizes(sizes)

    def _on_main_splitter_moved(self: Any, *_args) -> None:
        self._save_persisted_display_settings()


