"""presenter: start build."""
from __future__ import annotations
from pathlib import Path
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from ...core.analysis_modes import (
    geometry_mode_for_object_type,
)
from ...core.domain import BuildOptions, FolderSpec
from ...core.workers import FrameIndexWorker
from ...ui.ui_constants import (
    DEFAULT_CELL_SIZE,
    DEFAULT_POINT_EXTRACTION_MODE,
    DEFAULT_POLYGON_CONFIDENCE_SUMMARY,
)


class StartBuildMixin:
    def _overall_score_style(self, value: float | None) -> str:
        if value is None:
            background = "#2f3844"
            foreground = "#edf3fb"
        elif value < 0.33:
            background = "#1f5f3b"
            foreground = "#e9fff1"
        elif value < 0.66:
            background = "#8a6a12"
            foreground = "#fff7da"
        else:
            background = "#8c2f39"
            foreground = "#ffe9ec"
        return f"padding: 6px 10px; border-radius: 8px; background-color: {background}; color: {foreground}; font-weight: 700;"

    def _overall_score_text(self, value: float | None) -> str:
        if value is None:
            return "-"
        if value < 0.33:
            level = "LOW"
        elif value < 0.66:
            level = "MEDIUM"
        else:
            level = "HIGH"
        return f"{level} {value:.4f}"

    def _show_progress_bar(self, *, visible: bool, current: int = 0, total: int = 0, key: str = "", format_text: str | None = None) -> None:
        if not visible:
            self.build_progress.hide()
            self.build_progress.setRange(0, 1)
            self.build_progress.setValue(0)
            return
        if total > 0:
            self.build_progress.setRange(0, total)
            self.build_progress.setValue(min(current, total))
            self.build_progress.setFormat(format_text or f"{current}/{total}")
        else:
            self.build_progress.setRange(0, 0)
            self.build_progress.setFormat(format_text or "Working...")
        self.build_progress.setToolTip(key)
        self.build_progress.show()

    def _compact_folder_label(self, folder: FolderSpec | None) -> tuple[str, str]:
        if folder is None:
            return "not set", ""
        path = folder.path
        tail = path.name
        parent = path.parent.name if path.parent != path else ""
        short = f"{parent}/{tail}" if parent else tail
        return short, str(path)

    @staticmethod
    def _compact_path_text(path_text: str | None) -> str:
        if not path_text:
            return ""
        path = Path(str(path_text))
        tail = path.name or str(path)
        parent = path.parent.name if path.parent != path else ""
        return f"{parent}/{tail}" if parent else tail

    def _update_source_labels(self) -> None:
        original_text, original_tooltip = self._compact_folder_label(self._original_folder)
        gt_text, gt_tooltip = self._compact_folder_label(self._gt_folder)
        self.original_folder_value.setText(original_text)
        self.original_folder_value.setToolTip(original_tooltip)
        self.gt_folder_value.setText(gt_text)
        self.gt_folder_value.setToolTip(gt_tooltip)

    def _add_folder(self) -> None:
        if self._worker_thread is not None:
            return
        folder = QFileDialog.getExistingDirectory(self._view, self._t("dialog.select_model_folder"))
        if not folder:
            return
        folder_path = Path(folder)
        if not self._folder_has_supported_images(folder_path):
            QMessageBox.warning(
                self._view,
                self._t("dialog.warning_title"),
                f"Folder has no supported images: {folder_path}",
            )
            return
        item = self._append_folder_item(folder_path, checked=True)
        self.folder_list.setCurrentItem(item)
        self._refresh_folder_rows()
        self._sync_action_buttons()

    def _clear_folders(self) -> None:
        self.folder_list.clear()
        self._refresh_folder_rows()
        self._sync_action_buttons()

    def _set_original_folder(self) -> None:
        if self._worker_thread is not None:
            return
        folder = QFileDialog.getExistingDirectory(self._view, self._t("dialog.select_original_folder"))
        if not folder:
            return
        path = Path(folder)
        if not self._folder_has_supported_images(path):
            QMessageBox.warning(
                self._view,
                self._t("dialog.warning_title"),
                f"Base folder has no supported images: {path}",
            )
            return
        self._original_folder = FolderSpec(path=path, label=path.name)
        self._update_source_labels()
        self._sync_action_buttons()

    def _clear_original_folder(self) -> None:
        self._original_folder = None
        self._update_source_labels()
        self._sync_action_buttons()

    def _set_gt_folder(self) -> None:
        if self._worker_thread is not None:
            return
        folder = QFileDialog.getExistingDirectory(self._view, self._t("dialog.select_gt_folder"))
        if not folder:
            return
        path = Path(folder)
        if not self._folder_has_supported_images(path):
            QMessageBox.warning(
                self._view,
                self._t("dialog.warning_title"),
                f"Labeled folder has no supported images: {path}",
            )
            return
        self._gt_folder = FolderSpec(path=path, label=path.name)
        self._update_source_labels()
        self._sync_action_buttons()

    def _clear_gt_folder(self) -> None:
        self._gt_folder = None
        self._update_source_labels()
        self._sync_action_buttons()

    def _start_build(self) -> None:
        model_specs = self._checked_model_specs()
        if not model_specs:
            QMessageBox.warning(self._view, self._t("dialog.warning_title"), self._t("errors.active_model_required"))
            return
        self._close_all_details_dialogs()
        geometry_mode = geometry_mode_for_object_type(self._selected_object_type())
        mask_threshold, boundary_radius = self._selected_polygon_compare_values()
        options = BuildOptions(
            thumbnail_size=int(DEFAULT_CELL_SIZE),
            recursive=True,
            tile_mode=self._selected_tile_mode(),
            tile_width=int(self.tile_width_spin.value()),
            tile_height=int(self.tile_height_spin.value()),
            tile_overlap_mode=self._selected_tile_overlap_mode(),
            tile_overlap=int(self.tile_overlap_spin.value()),
            subpixel_view_mode=self._selected_subpixel_view_mode(),
            subpixel_rows=self._selected_subpixel_rows(),
            subpixel_columns=self._selected_subpixel_columns(),
            subpixel_aggregation=self._selected_subpixel_aggregation(),
            geometry_mode=geometry_mode,
            mask_threshold=float(mask_threshold),
            boundary_radius=int(boundary_radius),
            confidence_uncertainty_delta=self._selected_confidence_uncertainty_delta(),
            point_match_radius=float(self.point_match_radius_spin.value()),
            point_confidence_radius=int(self.point_confidence_radius_spin.value()),
            point_extraction_mode=str(self.point_extraction_mode_combo.currentData() or DEFAULT_POINT_EXTRACTION_MODE),
            polygon_confidence_summary=str(self.polygon_confidence_summary_combo.currentData() or DEFAULT_POLYGON_CONFIDENCE_SUMMARY),
        )
        self._pending_build_snapshot = self._capture_view_snapshot()
        self._worker_kind = "build"
        self._worker_thread = QThread(self._view)
        self._worker = FrameIndexWorker(model_specs, options, self._original_folder, self._gt_folder)
        generation = self._begin_worker_request(state=None)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda current, total, key, g=generation: self._on_build_progress(current, total, key, generation=g))
        self._worker.finished.connect(lambda result, g=generation: self._on_build_finished(result, generation=g))
        self._worker.failed.connect(lambda message, g=generation: self._on_worker_failed(message, generation=g))
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)
        self._worker_thread.start()
        self._show_progress_bar(visible=True, format_text="Indexing frames...")
        self._sync_action_buttons()
