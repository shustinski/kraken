"""presenter: refresh folder rows."""
from __future__ import annotations
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QListWidgetItem, QMessageBox
from ...core.analysis_modes import (
    POINT_OBJECT_TYPE,
    POLYGON_OBJECT_TYPE,
)
from ...core.domain import FrameRecord
from ...ui.matrix_view import MatrixLayoutConfig
from ...ui.ui_components import FolderRowWidget
from ...ui.ui_constants import (
    DEFAULT_CELL_SIZE,
    DEFAULT_GEOMETRY_MODE,
    DEFAULT_MATRIX_LAYOUT_MODE,
    DEFAULT_MATRIX_SCORE_VIEW_MODE,
    DEFAULT_MATRIX_METRIC_KEY,
    DEFAULT_POINT_EXTRACTION_MODE,
    DEFAULT_POLYGON_CONFIDENCE_SUMMARY,
    FOLDER_CHECKED_ROLE,
    FOLDER_CONFIDENCE_ROLE,
    FOLDER_LABEL_ROLE,
    FOLDER_ROW_MIN_HEIGHT,
    PERCENTILE_BAND_BOUNDS,
)
from neuralimage.Validation_gradient_widget_lite.app.state import ExtendMatrixTabState


class RefreshFolderRowsMixin:
    @staticmethod
    def _folder_has_supported_images(folder_path: Path) -> bool:
        if not folder_path.exists() or not folder_path.is_dir():
            return False
        allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        try:
            for image_path in folder_path.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in allowed:
                    return True
        except Exception:
            return False
        return False

    def _refresh_folder_rows(self) -> None:
        for row in range(self.folder_list.count()):
            item = self.folder_list.item(row)
            path_text = str(item.data(Qt.ItemDataRole.UserRole))
            display_text = str(item.data(FOLDER_LABEL_ROLE) or (Path(path_text).name or path_text))
            confidence_path_text = str(item.data(FOLDER_CONFIDENCE_ROLE) or "")
            confidence_display_text = self._compact_path_text(confidence_path_text)
            row_widget = FolderRowWidget(
                self.folder_list,
                path_text=path_text,
                display_text=display_text,
                checked=bool(item.data(FOLDER_CHECKED_ROLE)),
                confidence_display_text=confidence_display_text,
                confidence_path_text=confidence_path_text,
                can_move_up=row > 0,
                can_move_down=row < self.folder_list.count() - 1,
                on_checked_changed=lambda checked, item=item: self._set_folder_item_checked(item, checked),
                on_label_changed=lambda text, item=item: self._set_folder_item_label(item, text),
                on_confidence_folder=lambda _checked=False, item=item: self._set_folder_item_confidence_folder(item),
                on_clear_confidence_folder=lambda _checked=False, item=item: self._clear_folder_item_confidence_folder(item),
                on_remove=lambda _checked=False, item=item: self._remove_folder_item(item),
                on_move_up=lambda _checked=False, item=item: self._move_folder_item(item, -1),
                on_move_down=lambda _checked=False, item=item: self._move_folder_item(item, 1),
                checkbox_tooltip="Use model in analytics",
                confidence_placeholder=self._t("folders.confidence_not_set"),
                confidence_tooltip=confidence_path_text,
                confidence_select_tooltip=self._t("folders.select_confidence"),
                confidence_clear_tooltip=self._t("folders.clear_confidence"),
                remove_tooltip="Remove model folder",
                move_up_tooltip="Move up",
                move_down_tooltip="Move down",
            )
            row_widget.setMinimumHeight(FOLDER_ROW_MIN_HEIGHT)
            item.setSizeHint(row_widget.sizeHint())
            self.folder_list.setItemWidget(item, row_widget)

    def _set_folder_item_checked(self, item: QListWidgetItem, checked: bool) -> None:
        self._folder_check_guard = True
        item.setData(FOLDER_CHECKED_ROLE, bool(checked))
        self._folder_check_guard = False
        self._refresh_folder_rows()
        self._sync_action_buttons()

    def _set_folder_item_label(self, item: QListWidgetItem, text: str) -> None:
        folder_path = Path(item.data(Qt.ItemDataRole.UserRole))
        item.setData(FOLDER_LABEL_ROLE, text or folder_path.name)
        self._refresh_folder_rows()

    def _set_folder_item_confidence_folder(self, item: QListWidgetItem) -> None:
        if self._worker_thread is not None:
            return
        folder = QFileDialog.getExistingDirectory(self._view, self._t("dialog.select_model_confidence_folder"))
        if not folder:
            return
        folder_path = Path(folder)
        if not self._folder_has_supported_images(folder_path):
            QMessageBox.warning(
                self._view,
                self._t("dialog.warning_title"),
                f"Confidence folder has no supported images: {folder_path}",
            )
            return
        item.setData(FOLDER_CONFIDENCE_ROLE, str(folder_path))
        self._refresh_folder_rows()
        self._sync_action_buttons()

    def _clear_folder_item_confidence_folder(self, item: QListWidgetItem) -> None:
        item.setData(FOLDER_CONFIDENCE_ROLE, "")
        self._refresh_folder_rows()
        self._sync_action_buttons()

    def _remove_folder_item(self, item: QListWidgetItem) -> None:
        row = self.folder_list.row(item)
        if row < 0:
            return
        self.folder_list.takeItem(row)
        self._refresh_folder_rows()
        self._sync_action_buttons()

    def _move_folder_item(self, item: QListWidgetItem, delta: int) -> None:
        row = self.folder_list.row(item)
        target_row = row + int(delta)
        if row < 0 or target_row < 0 or target_row >= self.folder_list.count():
            return
        moved_item = self.folder_list.takeItem(row)
        self.folder_list.insertItem(target_row, moved_item)
        self.folder_list.setCurrentRow(target_row)
        self._refresh_folder_rows()

    def _build_layout_config(self) -> MatrixLayoutConfig:
        return MatrixLayoutConfig(
            mode=str(self.layout_mode_combo.currentData() or DEFAULT_MATRIX_LAYOUT_MODE),
            total_frames=int(self.total_frames_spin.value()),
            frames_per_row=int(self.frames_per_row_spin.value()),
            rows=int(self.matrix_rows_spin.value()),
            columns=int(self.matrix_columns_spin.value()),
        )

    def _percentile_bin_bounds(self, bin_index: int) -> tuple[float, float]:
        normalized = max(0, min(int(bin_index), len(PERCENTILE_BAND_BOUNDS) - 1))
        low_bound, high_bound = PERCENTILE_BAND_BOUNDS[normalized]
        return float(low_bound), float(high_bound)

    def _records_in_percentile_bin(self, records: tuple[FrameRecord, ...] | list[FrameRecord], percentile_map: dict[str, float], bin_index: int) -> tuple[FrameRecord, ...]:
        low_bound, high_bound = self._percentile_bin_bounds(bin_index)
        selected: list[FrameRecord] = []
        for record in records:
            percentile = float(percentile_map.get(record.key, 0.0))
            if bin_index >= len(PERCENTILE_BAND_BOUNDS) - 1:
                matches = low_bound <= percentile <= high_bound
            else:
                matches = low_bound <= percentile < high_bound
            if matches:
                selected.append(record)
        return tuple(selected)

    def _base_records_for_state(self, state: ExtendMatrixTabState) -> tuple[FrameRecord, ...]:
        cache_key = (str(getattr(state, 'object_type', POLYGON_OBJECT_TYPE) or POLYGON_OBJECT_TYPE), id(state.build_result.records))
        cached = state.base_records_cache.get(cache_key)
        if cached is not None:
            return tuple(cached)
        records: tuple[FrameRecord, ...] | list[FrameRecord] = state.build_result.records
        object_type = str(getattr(state, 'object_type', POLYGON_OBJECT_TYPE) or POLYGON_OBJECT_TYPE)
        if object_type in {POLYGON_OBJECT_TYPE, POINT_OBJECT_TYPE}:
            records = tuple(
                record for record in records
                if record.summary is None or str(getattr(record.summary, 'frame_type', POLYGON_OBJECT_TYPE)) == object_type
            )
        resolved = tuple(records)
        state.base_records_cache[cache_key] = resolved
        return resolved

    def _display_records_for_state(self, state: ExtendMatrixTabState) -> tuple[FrameRecord, ...]:
        records: tuple[FrameRecord, ...] | list[FrameRecord] = self._base_records_for_state(state)
        available_keys = set(self._percentile_basis_keys_for_state(state, state.build_result))
        if state.percentile_filter_metric_key not in available_keys:
            state.percentile_filter_metric_key = None
            state.percentile_filter_bin_index = None
        if state.percentile_filter_metric_key is not None and state.percentile_filter_bin_index is not None:
            percentile_map = self._percentile_map_for_metric(state, state.percentile_filter_metric_key)
            records = self._records_in_percentile_bin(records, percentile_map, state.percentile_filter_bin_index)
        if state.correlation_filter_band in {'bad', 'good'}:
            allowed_keys = {record.key for record, _count, _avg, _labels in self._repeated_percentile_entries(state, band=str(state.correlation_filter_band))[:25]}
            records = tuple(record for record in records if record.key in allowed_keys)
        return tuple(records)

    def _capture_view_snapshot(self) -> dict[str, object]:
        confidence_model_id = self._selected_confidence_model_id(None)
        mask_threshold, boundary_radius = self._selected_polygon_compare_values()
        return {
            "cell_size": int(DEFAULT_CELL_SIZE),
            "layout_config": self._build_layout_config(),
            "matrix_score_view_mode": str(self.matrix_score_view_combo.currentData() or DEFAULT_MATRIX_SCORE_VIEW_MODE),
            "analysis_mode": self._selected_analysis_mode(),
            "object_type": self._selected_object_type(),
            "geometry_mode": str(self.geometry_mode_combo.currentData() or DEFAULT_GEOMETRY_MODE),
            "polygon_compare_profile": self._selected_polygon_compare_profile(),
            "mask_threshold": float(mask_threshold),
            "boundary_radius": int(boundary_radius),
            "confidence_uncertainty_profile": self._selected_confidence_uncertainty_profile(),
            "confidence_uncertainty_delta": self._selected_confidence_uncertainty_delta(),
            "point_match_radius": float(self.point_match_radius_spin.value()),
            "point_confidence_radius": int(self.point_confidence_radius_spin.value()),
            "point_extraction_mode": str(self.point_extraction_mode_combo.currentData() or DEFAULT_POINT_EXTRACTION_MODE),
            "polygon_confidence_summary": str(self.polygon_confidence_summary_combo.currentData() or DEFAULT_POLYGON_CONFIDENCE_SUMMARY),
            "tile_mode": self._selected_tile_mode(),
            "tile_width": int(self.tile_width_spin.value()),
            "tile_height": int(self.tile_height_spin.value()),
            "tile_overlap_mode": self._selected_tile_overlap_mode(),
            "tile_overlap": int(self.tile_overlap_spin.value()),
            "subpixel_view_mode": self._selected_subpixel_view_mode(),
            "subpixel_rows": self._selected_subpixel_rows(),
            "subpixel_columns": self._selected_subpixel_columns(),
            "subpixel_aggregation": self._selected_subpixel_aggregation(),
            "metric_key": str(self.metric_combo.currentData() or DEFAULT_MATRIX_METRIC_KEY),
            "metric_scope": str(confidence_model_id or ""),
            "confidence_model_id": confidence_model_id,
            "frame_type_filter": str(self.frame_type_filter_combo.currentData() or 'all'),
        }
