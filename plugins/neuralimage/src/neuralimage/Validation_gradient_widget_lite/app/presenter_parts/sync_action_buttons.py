"""presenter: sync action buttons."""
from __future__ import annotations
from pathlib import Path
from PyQt6.QtCore import Qt
from ...core.domain import FolderSpec
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
)


class SyncActionButtonsMixin:
    def _sync_action_buttons(self) -> None:
        current_state = self._current_tab_state()
        active_model_count = len(self._checked_model_specs())
        is_busy = self._worker_thread is not None
        self.btn_clear_folders.setEnabled(self.folder_list.count() > 0 and not is_busy)
        self.btn_set_original.setEnabled(not is_busy)
        self.btn_clear_original.setEnabled(self._original_folder is not None and not is_busy)
        self.btn_set_gt.setEnabled(not is_busy)
        self.btn_clear_gt.setEnabled(self._gt_folder is not None and not is_busy)
        self.btn_build.setEnabled(active_model_count > 0 and not is_busy)
        self.btn_compute.setEnabled(current_state is not None and not is_busy)
        self.btn_cancel.setEnabled(is_busy)
        if hasattr(self._view, "set_workflow_summary"):
            original_state = self._t("workflow.state.ready") if self._original_folder is not None else self._t("workflow.state.pending")
            gt_state = self._t("workflow.state.ready") if self._gt_folder is not None else self._t("workflow.state.optional")
            sources_tone = "ready" if self._original_folder is not None else "warn"
            models_status = self._t("workflow.state.ready") if active_model_count > 0 else self._t("workflow.state.pending")
            models_tone = "ready" if active_model_count > 0 else "warn"
            if is_busy:
                analysis_status = self._t("workflow.state.running")
                analysis_detail = self.build_progress.format() or self._t("workflow.analysis_running")
                analysis_tone = "busy"
            elif current_state is None:
                analysis_status = self._t("workflow.state.pending")
                analysis_detail = self._t("workflow.analysis_pending")
                analysis_tone = "idle"
            elif bool(getattr(current_state.build_result, "scores_computed", False)):
                analysis_status = self._t("workflow.state.computed")
                analysis_detail = self._t("workflow.analysis_computed")
                analysis_tone = "active"
            else:
                analysis_status = self._t("workflow.state.built")
                analysis_detail = self._t("workflow.analysis_built")
                analysis_tone = "ready"
            self._view.set_workflow_summary({
                "sources": (
                    self._t("workflow.state.partial") if self._original_folder is None and self._gt_folder is not None else original_state,
                    self._t("workflow.sources_detail", original=original_state, gt=gt_state),
                    sources_tone,
                ),
                "models": (
                    models_status,
                    self._t("workflow.models_detail", count=active_model_count),
                    models_tone,
                ),
                "analysis": (analysis_status, analysis_detail, analysis_tone),
            })

    def _build_folder_manager_payload(self) -> dict:
        return {
            "folders": [
                {
                    "path": str(self.folder_list.item(row).data(Qt.ItemDataRole.UserRole)),
                    "checked": bool(self.folder_list.item(row).data(FOLDER_CHECKED_ROLE)),
                    "label": str(self.folder_list.item(row).data(FOLDER_LABEL_ROLE) or ""),
                    "confidence_path": str(self.folder_list.item(row).data(FOLDER_CONFIDENCE_ROLE) or ""),
                }
                for row in range(self.folder_list.count())
            ],
            "original_folder": str(self._original_folder.path) if self._original_folder is not None else None,
            "gt_folder": str(self._gt_folder.path) if self._gt_folder is not None else None,
        }

    def _restore_persisted_state(self) -> None:
        self._restore_folder_manager_state()
        self._restore_build_settings()
        self._update_source_labels()

    def _restore_folder_manager_state(self) -> None:
        payload = self._settings_service.load_folder_manager_payload()
        if not payload:
            return
        self.folder_list.blockSignals(True)
        try:
            for folder_entry in payload.get("folders", []):
                path = folder_entry.get("path")
                if not path:
                    continue
                folder_path = Path(path)
                if not folder_path.exists():
                    continue
                item = self._append_folder_item(folder_path, checked=bool(folder_entry.get("checked", False)))
                item.setData(FOLDER_LABEL_ROLE, str(folder_entry.get("label") or folder_path.name))
                confidence_path = folder_entry.get("confidence_path")
                if confidence_path and Path(confidence_path).exists():
                    item.setData(FOLDER_CONFIDENCE_ROLE, str(confidence_path))
            original_folder = payload.get("original_folder")
            gt_folder = payload.get("gt_folder")
            if original_folder and Path(original_folder).exists():
                path = Path(original_folder)
                self._original_folder = FolderSpec(path=path, label=path.name)
            if gt_folder and Path(gt_folder).exists():
                path = Path(gt_folder)
                self._gt_folder = FolderSpec(path=path, label=path.name)
        finally:
            self.folder_list.blockSignals(False)

    def _build_build_settings_payload(self) -> dict:
        mask_threshold, boundary_radius = self._selected_polygon_compare_values()
        return {
            "thumbnail_size": int(DEFAULT_CELL_SIZE),
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
            "layout_mode": str(self.layout_mode_combo.currentData() or DEFAULT_MATRIX_LAYOUT_MODE),
            "total_frames": int(self.total_frames_spin.value()),
            "frames_per_row": int(self.frames_per_row_spin.value()),
            "rows": int(self.matrix_rows_spin.value()),
            "columns": int(self.matrix_columns_spin.value()),
            "metric_key": str(self.metric_combo.currentData() or DEFAULT_MATRIX_METRIC_KEY),
            "metric_scope": str(self.metric_scope_combo.currentData() or ""),
            "confidence_model_id": str(self.metric_scope_combo.currentData() or ""),
            "frame_type_filter": str(self.frame_type_filter_combo.currentData() or 'all'),
        }
