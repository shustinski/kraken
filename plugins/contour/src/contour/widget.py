from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import (
    QEvent,
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QSettings,
    QSignalBlocker,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QBrush, QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .adapters.qt.directory_scan import ScanInputDirectoryRunnable, ScanInputDirectorySignals
from .adapters.qt.image_conversion import cv_to_qimage
from .adapters.qt.preview import AutoTuneRunnable, PreparedImageRunnable, PreviewProcessingRunnable
from .application.dto import PersistedPaths
from .application.frame_asset_sync import (
    background_hex_image_paint_status,
    background_hex_vector_status,
    build_image_cif_matching_report,
    classify_image_side_paint_status,
    classify_vector_side_status,
    foreground_hex_image_has_vector_overlay,
    index_cif_file_paths,
)
from .application.processing import (
    VIA_SEARCH_MODE_HEURISTIC,
    VIA_SEARCH_MODE_TEMPLATE,
    VIA_SIZE_MODE_FIXED,
    ContourExtractionSettings,
    DisplaySettings,
    ImageProcessingState,
    SaveOptions,
    _normalize_bright_via_metal_constraint_mode,
    normalize_algorithm_backend,
    normalize_recognition_mode,
    normalize_via_search_mode,
    normalize_via_size_mode,
)
from .application.services import (
    BatchController,
    BatchStartRequest,
    WorkspaceSession,
    export_frame_to_dataset,
    load_pipeline_config_from_path,
    save_pipeline_config_to_path,
)
from .application.transition_save_guard import (
    TransitionPromptChoice,
    navigation_allowed_after_autosave_attempt,
    navigation_allowed_after_prompt,
)
from .application.use_cases import (
    AutoTuneResult,
    PreparedImageRequest,
    PreviewProcessingRequest,
    build_prepared_image_signature,
    build_preview_request_signature,
    index_cif_directory,
)
from .application.vector_geometry_postprocess import VectorGeometrySettings
from .batch_processor import BatchProcessor
from .domain import PolygonData
from .graphics.editor_hotkeys import (
    append_shortcut_to_tooltip,
    build_editor_hotkeys_plain_text,
    tool_shortcut_native_text,
)
from .graphics_view import EditorTool, PolygonCreateMode
from .i18n import active_language, tr
from .infrastructure import WidgetDisplaySettingsStore, WidgetPathSettingsStore
from .pipeline import (
    PreprocessingPipeline,
    available_operations,
    get_choice_display_label,
    get_operation_descriptor,
    get_operation_display_name,
    get_parameter_display_label,
)
from .serializers import load_polygons_cif, save_polygons_cif, save_result_bundle
from .ui.builders import (
    build_display_tab,
    build_editor_toolbar,
    build_extraction_tab,
    build_files_tab,
    build_help_tab,
    build_path_panel,
    build_paths_tab,
    build_pipeline_tab,
    build_tabs,
    build_ui,
    build_visual_panel,
)
from .ui.editor_icons import (
    TOOLBAR_BUTTON_SIZE_PX,
    TOOLBAR_ICON_CANVAS_SIZE_PX,
    TOOLBAR_ICON_SIZE_PX,
    create_editor_action_icon,
    create_editor_tool_icon,
)
from .ui.i18n_content import (
    EDITOR_ACTION_TOOLTIPS,
    EDITOR_TOOL_TOOLTIPS,
    EXTRACTION_HELP_TEXTS,
    GENERAL_CONTROL_TOOLTIPS,
    PIPELINE_CONTROL_TOOLTIPS,
    PIPELINE_OPERATION_GROUPS,
    PIPELINE_OPERATION_HELP_TEXTS,
    PIPELINE_PARAMETER_HELP_TEXTS,
    LocalizedTextMap,
    _localized_text,
)
from .ui.pipeline_presets import built_in_pipeline_presets
from .ui.retranslate import retranslate_ui
from .ui.styles import COMPACT_UI_STYLE
from .ui.via_presets import (
    blurred_via_preset_payload,
    built_in_via_presets,
    noisy_traces_via_preset_payload,
)
from .utils import is_image_path, load_image_color

__all__ = [
    "EDITOR_ACTION_TOOLTIPS",
    "EDITOR_TOOL_TOOLTIPS",
    "EXTRACTION_HELP_TEXTS",
    "GENERAL_CONTROL_TOOLTIPS",
    "PIPELINE_CONTROL_TOOLTIPS",
    "PIPELINE_OPERATION_GROUPS",
    "PIPELINE_OPERATION_HELP_TEXTS",
    "PIPELINE_PARAMETER_HELP_TEXTS",
    "LocalizedTextMap",
    "PolygonExtractionWidget",
    "_localized_text",
]


FRAME_STATUS_ROLE = int(Qt.ItemDataRole.UserRole) + 1
VIA_PRESETS_SETTINGS_KEY = "via_search/user_presets"


class ThumbnailLoadSignals(QObject):
    result = pyqtSignal(int, str, object)
    finished = pyqtSignal(int, str)


class ThumbnailLoadRunnable(QRunnable):
    def __init__(self, generation: int, path: str, width: int, height: int) -> None:
        super().__init__()
        self.generation = int(generation)
        self.path = str(path)
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.signals = ThumbnailLoadSignals()

    def run(self) -> None:
        image = None
        try:
            source = cv2.imread(self.path, cv2.IMREAD_COLOR)
            if source is not None:
                h, w = source.shape[:2]
                if w > 0 and h > 0:
                    scale = min(self.width / float(w), self.height / float(h), 1.0)
                    target_w = max(1, round(w * scale))
                    target_h = max(1, round(h * scale))
                    if target_w != w or target_h != h:
                        source = cv2.resize(source, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    image = source
        except Exception:
            image = None
        self.signals.result.emit(self.generation, self.path, image)
        self.signals.finished.emit(self.generation, self.path)


class PolygonExtractionWidget(QWidget):
    imageProcessed = pyqtSignal(str, list)
    batchProgress = pyqtSignal(int, int)
    batchFinished = pyqtSignal()
    polygonsEdited = pyqtSignal()
    logMessage = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("polygonExtractionWidget")
        self._ui_language = active_language()
        self._path_settings_store = WidgetPathSettingsStore()
        self._display_settings_store = WidgetDisplaySettingsStore()
        self._workspace = WorkspaceSession()
        self._pipeline = PreprocessingPipeline()
        self._display_settings = DisplaySettings()
        self._contour_settings_profiles = {
            "conductors": ContourExtractionSettings(
                algorithm_backend="legacy",
                sem_noise_level="medium",
                extraction_profile="conductors",
                object_type="conductor",
                output_mode="polygon",
                min_polygon_angle=30.0,
                retrieval_mode="RETR_TREE",
                epsilon=2.0,
                min_area=70.0,
                min_perimeter=32.0,
                min_points=4,
                min_polygon_width_px=4.0,
                metal_structural_pipeline=True,
            ),
            "vias": ContourExtractionSettings(
                algorithm_backend="sem",
                sem_noise_level="medium",
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_search_mode="heuristic",
                min_solidity=0.6,
                min_extent=0.5,
                min_aspect_ratio=0.5,
                max_aspect_ratio=2.0,
            ),
        }
        self._active_extraction_profile = "conductors"
        self._ignore_extraction_profile_change = False
        self._ignore_pipeline_item_change = False
        self._suspend_fixed_via_updates = False
        self._restoring_display_settings = False
        self._fixed_via_rows: list[dict[str, QWidget]] = []
        self._parameter_widgets: dict[str, QWidget] = {}
        self._updating_views = False
        self._batch_progress_enabled = False
        self._progress_status_key = "idle_status"
        self._progress_status_kwargs: dict[str, object] = {}
        self._preview_thread_pool = QThreadPool(self)
        self._preview_thread_pool.setMaxThreadCount(1)
        self._preview_thread_pool.setExpiryTimeout(-1)
        self._preview_update_timer = QTimer(self)
        self._preview_update_timer.setSingleShot(True)
        self._preview_update_timer.setInterval(50)
        self._preview_update_timer.timeout.connect(self._start_pending_preview_processing)
        self._preview_request_serial = 0
        self._preview_running_request_id: int | None = None
        self._preview_pending_request: PreviewProcessingRequest | None = None
        self._preview_running_signature: tuple[str, str, str] | None = None
        self._preview_pending_signature: tuple[str, str, str] | None = None
        self._preview_run_cancel: threading.Event | None = None
        self._help_menu: QMenu | None = None
        self._color_pick_pipeline_row: int | None = None
        self._via_template_images: list[np.ndarray] = []
        self._viewed_image_paths: set[str] = set()
        self._user_via_presets: dict[str, dict[str, object]] = self._load_user_via_presets()
        self._extra_layers: list[dict[str, object]] = []
        self._prepared_image_thread_pool = QThreadPool(self)
        self._prepared_image_thread_pool.setMaxThreadCount(1)
        self._prepared_image_thread_pool.setExpiryTimeout(-1)
        self._prepared_image_request_serial = 0
        self._prepared_image_running_request_id: int | None = None
        self._prepared_image_pending_request: PreparedImageRequest | None = None
        self._prepared_image_running_signature: tuple[str, str] | None = None
        self._prepared_image_pending_signature: tuple[str, str] | None = None
        self._prepared_image_run_cancel: threading.Event | None = None
        self._auto_tune_thread_pool = QThreadPool(self)
        self._auto_tune_thread_pool.setMaxThreadCount(1)
        self._auto_tune_thread_pool.setExpiryTimeout(-1)
        self._auto_tune_request_serial = 0
        self._auto_tune_running_request_id: int | None = None
        self._neighbor_image_cache: dict[str, object] = {}
        self._thumbnail_thread_pool = QThreadPool(self)
        self._thumbnail_thread_pool.setMaxThreadCount(2)
        self._thumbnail_thread_pool.setExpiryTimeout(30000)
        self._thumbnail_generation = 0
        self._thumbnail_icon_size = QSize(64, 48)
        self._thumbnail_placeholder_icon = QIcon()
        self._show_source_while_middle_held = False

        self._persisted_highlight_paths: set[str] = set()
        self._cif_load_failure_stems: set[str] = set()
        self._directory_scan_busy = False
        self._directory_scan_pending_directory: str | None = None
        self._directory_scan_signals = ScanInputDirectorySignals(self)
        self._directory_scan_signals.finished.connect(self._on_input_directory_scan_finished)
        self._directory_scan_signals.failed.connect(self._on_input_directory_scan_failed)
        self._scan_generation = 0
        self._vectors_list_ignore_navigate_until: float = 0.0
        self._scan_thread_pool = QThreadPool(self)
        self._scan_thread_pool.setMaxThreadCount(1)
        self._scan_thread_pool.setExpiryTimeout(-1)

        self._batch_processor = BatchProcessor(self)
        self._batch_processor.set_ui_language(self._ui_language)
        self._batch_processor.resultReady.connect(self._on_batch_result)
        self._batch_processor.progressChanged.connect(self._on_batch_progress)
        self._batch_processor.finished.connect(self._on_batch_finished)
        self._batch_processor.errorOccurred.connect(self._on_batch_error)
        self._batch_processor.logMessage.connect(self._append_log)
        self._batch_controller = BatchController(self._batch_processor)

        self._build_ui()
        self._apply_compact_ui_style()
        self._disable_spinbox_wheel_changes()
        self._restore_persisted_paths()
        self._restore_persisted_display_settings()
        self._populate_pipeline_operations()
        self._populate_pipeline_list()
        self._apply_display_settings()
        self._set_extraction_settings(self._contour_settings_profiles[self._active_extraction_profile])
        self._set_default_extraction_disabled()
        self.set_ui_language(self._ui_language)

    def _build_ui(self) -> None:
        return build_ui(self)

    def _apply_compact_ui_style(self) -> None:
        self.setStyleSheet(COMPACT_UI_STYLE)

    def _build_path_panel(self) -> QWidget:
        return build_path_panel(self)

    def _build_paths_tab(self) -> QWidget:
        return build_paths_tab(self)

    def _build_tabs(self) -> QWidget:
        return build_tabs(self)

    def _restore_persisted_paths(self) -> None:
        paths = self._path_settings_store.load()

        if paths.output_directory:
            self.set_output_directory(paths.output_directory)
        if paths.dataset_directory:
            self.set_dataset_directory(paths.dataset_directory)
        if paths.cif_directory:
            self.set_cif_directory(paths.cif_directory)
        if paths.input_directory:
            self.set_input_directory(paths.input_directory)

    def _save_persisted_paths(self) -> None:
        self._path_settings_store.save(
            PersistedPaths(
                input_directory=self.input_dir_edit.text().strip(),
                cif_directory=self.cif_dir_edit.text().strip(),
                output_directory=self.output_dir_edit.text().strip(),
                dataset_directory=self.dataset_dir_edit.text().strip(),
            )
        )

    def _restore_persisted_display_settings(self) -> None:
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
            QSignalBlocker(self.show_neighbor_frames_checkbox),
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
            self.show_neighbor_frames_checkbox.setChecked(bool(payload.get("show_neighbor_frames", False)))
            self.neighbor_columns_spin.setValue(max(1, int(payload.get("neighbor_columns", 3))))
            self.neighbor_max_grid_spin.setValue(self._odd_neighbor_grid_size(int(payload.get("neighbor_max_grid", 7))))
            self.neighbor_opacity_spin.setValue(float(payload.get("neighbor_opacity", 0.35)))
            self.neighbor_overlap_spin.setValue(max(0, int(payload.get("neighbor_overlap_pixels", 0))))
            self.autosave_on_frame_transition_checkbox.setChecked(False)
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
        self._apply_vector_geometry_editor_config()

    def _current_display_settings_payload(self) -> dict[str, object]:
        payload_out: dict[str, object] = {
            **self._display_settings.to_dict(),
            "random_object_colors": bool(self.random_object_colors_checkbox.isChecked()),
            "show_neighbor_frames": bool(self.show_neighbor_frames_checkbox.isChecked()),
            "neighbor_columns": int(self.neighbor_columns_spin.value()),
            "neighbor_max_grid": int(self.neighbor_max_grid_spin.value()),
            "neighbor_opacity": float(self.neighbor_opacity_spin.value()),
            "neighbor_overlap_pixels": int(self.neighbor_overlap_spin.value()),
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

    def _save_persisted_display_settings(self) -> None:
        if self._restoring_display_settings or not hasattr(self, "line_width_spin"):
            return
        self._display_settings_store.save(self._current_display_settings_payload())

    def _vector_geometry_settings_from_widgets(self) -> VectorGeometrySettings:
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

    def _apply_vector_geometry_editor_config(self) -> None:
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_vector_geometry_settings(self._vector_geometry_settings_from_widgets())

    def _set_default_extraction_disabled(self) -> None:
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

    def _on_vector_geom_control_changed(self, *_args) -> None:
        self._apply_vector_geometry_editor_config()
        self._save_persisted_display_settings()

    def _show_manual_tool_postprocess_dialog(self) -> None:
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
        form.addRow("Минимальная площадь отверстия для заливки, px²", self.vector_geom_min_hole_spin)
        self.vector_geom_min_hole_label_widget = form.labelForField(self.vector_geom_min_hole_spin)
        form.addRow(self.vector_geom_merge_checkbox)
        form.addRow("Минимальный угол острого выброса, °", self.vector_geom_spike_angle_spin)
        self.vector_geom_spike_angle_label_widget = form.labelForField(self.vector_geom_spike_angle_spin)
        form.addRow(self.vector_geom_drop_triangle_checkbox)

        close_button = QPushButton("Закрыть" if self._ui_language == "ru" else "Close")
        close_button.clicked.connect(dialog.accept)
        root.addWidget(close_button, 0, Qt.AlignmentFlag.AlignRight)
        dialog.exec()

    def _display_image_dimensions_for_vectors(self) -> tuple[int, int]:
        frame = self._display_image_for_current_state()
        if frame is None:
            return (0, 0)
        shape = getattr(frame, "shape", None)
        if isinstance(shape, tuple) and len(shape) >= 2:
            return (int(shape[1]), int(shape[0]))
        return (0, 0)

    def _restore_main_splitter_sizes(self, raw_sizes: object) -> None:
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

    def _on_main_splitter_moved(self, *_args) -> None:
        self._save_persisted_display_settings()

    def _build_files_tab(self) -> QWidget:
        return build_files_tab(self)

    def _build_pipeline_tab(self) -> QWidget:
        return build_pipeline_tab(self)

    def _build_extraction_tab(self) -> QWidget:
        return build_extraction_tab(self)

    def _build_display_tab(self) -> QWidget:
        return build_display_tab(self)

    def _build_help_tab(self) -> QWidget:
        return build_help_tab(self)

    def _clear_layout_widgets(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout_widgets(child_layout)

    @staticmethod
    def _build_help_sample_image() -> np.ndarray:
        image = np.full((180, 260), 38, dtype=np.uint8)
        cv2.rectangle(image, (18, 18), (110, 90), 190, thickness=-1)
        cv2.circle(image, (176, 60), 26, 230, thickness=-1)
        cv2.circle(image, (176, 60), 10, 70, thickness=-1)
        cv2.line(image, (20, 136), (236, 120), 160, thickness=6)
        cv2.line(image, (22, 154), (236, 154), 210, thickness=4)
        cv2.putText(image, "A1", (126, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 240, 2, cv2.LINE_AA)
        noise = np.random.default_rng(42).normal(0, 12, image.shape).astype(np.int16)
        return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def _operation_help_entry(self, operation_name: str) -> tuple[str, str]:
        entry = PIPELINE_OPERATION_HELP_TEXTS.get(operation_name, {})
        summary_pair = entry.get("summary", ("", ""))
        use_pair = entry.get("use", ("", ""))
        summary = summary_pair[0] if self._ui_language == "ru" else summary_pair[1]
        use_case = use_pair[0] if self._ui_language == "ru" else use_pair[1]
        if not summary:
            summary = (
                "Преобразование обрабатывает изображение перед извлечением контуров."
                if self._ui_language == "ru"
                else "This transformation preprocesses the image before contour extraction."
            )
        if not use_case:
            use_case = (
                "Используйте, когда этот эффект приближает изображение к удобной бинарной маске."
                if self._ui_language == "ru"
                else "Use it when the effect moves the image toward a cleaner binary mask."
            )
        return summary, use_case

    def _pipeline_parameter_tooltip(self, operation_name: str, parameter_name: str) -> str:
        del operation_name
        return _localized_text(PIPELINE_PARAMETER_HELP_TEXTS, parameter_name, self._ui_language)

    def _pixmap_for_help_image(self, image: np.ndarray) -> QPixmap:
        return QPixmap.fromImage(cv_to_qimage(image)).scaled(
            190,
            132,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _rebuild_help_cards(self) -> None:
        if not hasattr(self, "help_layout"):
            return
        self._clear_layout_widgets(self.help_layout)
        intro = QLabel(
            "Ниже показано, как каждое преобразование меняет один и тот же тестовый кадр. Это помогает понять, когда шаг уместен в pipeline."
            if self._ui_language == "ru"
            else "Below, each transformation is applied to the same synthetic sample image so you can see what it changes and when to use it."
        )
        intro.setWordWrap(True)
        self.help_layout.addWidget(intro)
        sample_image = self._build_help_sample_image()
        before_pixmap = self._pixmap_for_help_image(sample_image)
        for descriptor in available_operations():
            card = QGroupBox(get_operation_display_name(descriptor.type_name, self._ui_language))
            card_layout = QVBoxLayout(card)
            summary, use_case = self._operation_help_entry(descriptor.type_name)
            summary_label = QLabel(summary)
            summary_label.setWordWrap(True)
            use_label = QLabel(("Когда использовать: " if self._ui_language == "ru" else "When to use: ") + use_case)
            use_label.setWordWrap(True)
            images_row = QWidget()
            images_layout = QHBoxLayout(images_row)
            images_layout.setContentsMargins(0, 0, 0, 0)
            before_box = QVBoxLayout()
            before_title = QLabel("До" if self._ui_language == "ru" else "Before")
            before_image = QLabel()
            before_image.setPixmap(before_pixmap)
            before_box.addWidget(before_title)
            before_box.addWidget(before_image)
            after_box = QVBoxLayout()
            after_title = QLabel("После" if self._ui_language == "ru" else "After")
            after_image = QLabel()
            try:
                processed = descriptor.handler(sample_image.copy(), descriptor.default_parameters())
            except Exception:
                processed = sample_image
            after_image.setPixmap(self._pixmap_for_help_image(processed))
            after_box.addWidget(after_title)
            after_box.addWidget(after_image)
            images_layout.addLayout(before_box)
            images_layout.addLayout(after_box)
            card_layout.addWidget(summary_label)
            card_layout.addWidget(use_label)
            card_layout.addWidget(images_row)
            self.help_layout.addWidget(card)
        self.help_layout.addStretch(1)

    def help_menu_title(self) -> str:
        return self._tr("tab_help")

    def attach_help_menu(self, menu: QMenu) -> None:
        self._help_menu = menu
        self._refresh_help_menu()

    def _refresh_help_menu(self) -> None:
        if self._help_menu is None:
            return
        self._help_menu.clear()
        postprocess_action = self._help_menu.addAction(
            "Постобработка ручных инструментов"
            if self._ui_language == "ru"
            else "Manual tool post-processing"
        )
        postprocess_action.setObjectName("manualToolPostprocessAction")
        postprocess_action.triggered.connect(lambda _checked=False: self._show_manual_tool_postprocess_dialog())
        self._help_menu.addSeparator()
        overview_action = self._help_menu.addAction(
            self._tr(
                "help_all_filters_action", "Все преобразования" if self._ui_language == "ru" else "All transformations"
            )
        )
        overview_action.triggered.connect(lambda _checked=False: self._show_help_dialog())
        hotkeys_action = self._help_menu.addAction(
            self._tr(
                "help_editor_hotkeys_action",
                "Горячие клавиши редактора" if self._ui_language == "ru" else "Editor hotkeys",
            )
        )
        hotkeys_action.triggered.connect(lambda _checked=False: self._show_editor_hotkeys_dialog())
        self._help_menu.addSeparator()
        for group_key, labels, operations in PIPELINE_OPERATION_GROUPS:
            submenu = self._help_menu.addMenu(labels[0] if self._ui_language == "ru" else labels[1])
            submenu.setObjectName(f"helpMenu_{group_key}")
            for operation_name in operations:
                action = submenu.addAction(get_operation_display_name(operation_name, self._ui_language))
                action.triggered.connect(lambda _checked=False, op=operation_name: self._show_help_dialog(op))

    def _show_help_dialog(self, operation_name: str | None = None) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(
            self._tr("tab_help")
            if operation_name is None
            else get_operation_display_name(operation_name, self._ui_language)
        )
        dialog.resize(960, 720)
        layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        help_layout = QVBoxLayout(container)
        help_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)
        self._populate_help_cards(
            help_layout,
            [operation_name] if operation_name is not None else self._all_operation_names(),
        )
        dialog.exec()

    def _show_editor_hotkeys_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(
            self._tr(
                "help_editor_hotkeys_action",
                "Горячие клавиши редактора" if self._ui_language == "ru" else "Editor hotkeys",
            )
        )
        dialog.resize(520, 560)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(build_editor_hotkeys_plain_text(ru=self._ui_language == "ru"))
        layout.addWidget(text, 1)
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_button = QPushButton("Закрыть" if self._ui_language == "ru" else "Close")
        close_button.clicked.connect(dialog.accept)
        close_row.addWidget(close_button)
        layout.addLayout(close_row)
        dialog.exec()

    def _populate_help_cards(self, layout: QVBoxLayout, operation_names: list[str]) -> None:
        self._clear_layout_widgets(layout)
        intro = QLabel(
            self._tr(
                "help_intro_text",
                "Ниже показано, как каждое преобразование меняет один и тот же тестовый кадр. Это помогает понять, когда шаг уместен в pipeline."
                if self._ui_language == "ru"
                else "Each transformation below is applied to the same sample image so you can see its effect and when to use it.",
            )
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        sample_image = self._build_help_sample_image()
        before_pixmap = self._pixmap_for_help_image(sample_image)
        for operation_name in operation_names:
            descriptor = get_operation_descriptor(operation_name)
            card = QGroupBox(get_operation_display_name(descriptor.type_name, self._ui_language))
            card_layout = QVBoxLayout(card)
            summary, use_case = self._operation_help_entry(descriptor.type_name)
            summary_label = QLabel(summary)
            summary_label.setWordWrap(True)
            use_label = QLabel(("Когда использовать: " if self._ui_language == "ru" else "When to use: ") + use_case)
            use_label.setWordWrap(True)
            images_row = QWidget()
            images_layout = QHBoxLayout(images_row)
            images_layout.setContentsMargins(0, 0, 0, 0)
            before_box = QVBoxLayout()
            before_title = QLabel("До" if self._ui_language == "ru" else "Before")
            before_image = QLabel()
            before_image.setPixmap(before_pixmap)
            before_box.addWidget(before_title)
            before_box.addWidget(before_image)
            after_box = QVBoxLayout()
            after_title = QLabel("После" if self._ui_language == "ru" else "After")
            after_image = QLabel()
            try:
                processed = descriptor.handler(sample_image.copy(), descriptor.default_parameters())
            except Exception:
                processed = sample_image
            after_image.setPixmap(self._pixmap_for_help_image(processed))
            after_box.addWidget(after_title)
            after_box.addWidget(after_image)
            images_layout.addLayout(before_box)
            images_layout.addLayout(after_box)
            card_layout.addWidget(summary_label)
            card_layout.addWidget(use_label)
            card_layout.addWidget(images_row)
            layout.addWidget(card)
        layout.addStretch(1)

    def _all_operation_names(self) -> list[str]:
        return [descriptor.type_name for descriptor in available_operations()]

    def _selected_available_operation_name(self) -> str | None:
        if not hasattr(self, "operation_tree"):
            return None
        item = self.operation_tree.currentItem()
        if item is None:
            return None
        operation_name = item.data(0, Qt.ItemDataRole.UserRole)
        return str(operation_name) if operation_name else None

    def _find_operation_tree_item(self, operation_name: str) -> QTreeWidgetItem | None:
        if not hasattr(self, "operation_tree"):
            return None
        for index in range(self.operation_tree.topLevelItemCount()):
            group_item = self.operation_tree.topLevelItem(index)
            for child_index in range(group_item.childCount()):
                child_item = group_item.child(child_index)
                if child_item.data(0, Qt.ItemDataRole.UserRole) == operation_name:
                    return child_item
        return None

    def _update_pipeline_help_preview(self, operation_name: str | None) -> None:
        if not hasattr(self, "pipeline_help_title"):
            return
        if not operation_name:
            self.pipeline_help_title.clear()
            self.pipeline_help_summary.clear()
            self.pipeline_help_use.clear()
            self.pipeline_help_before_image.clear()
            self.pipeline_help_after_image.clear()
            return
        descriptor = get_operation_descriptor(operation_name)
        summary, use_case = self._operation_help_entry(operation_name)
        sample_image = self._build_help_sample_image()
        self.pipeline_help_title.setText(get_operation_display_name(operation_name, self._ui_language))
        self.pipeline_help_summary.setText(summary)
        self.pipeline_help_use.setText(
            ("Когда использовать: " if self._ui_language == "ru" else "When to use: ") + use_case
        )
        self.pipeline_help_before_image.setPixmap(self._pixmap_for_help_image(sample_image))
        try:
            processed = descriptor.handler(sample_image.copy(), descriptor.default_parameters())
        except Exception:
            processed = sample_image
        self.pipeline_help_after_image.setPixmap(self._pixmap_for_help_image(processed))

    def _on_available_operation_selected(
        self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None
    ) -> None:
        operation_name = current.data(0, Qt.ItemDataRole.UserRole) if current is not None else None
        self._update_pipeline_help_preview(str(operation_name) if operation_name else None)

    def _on_available_operation_activated(self, item: QTreeWidgetItem, _column: int) -> None:
        if item.data(0, Qt.ItemDataRole.UserRole):
            self._add_pipeline_step()

    def _set_field_tooltip(self, label_widget: QLabel | None, field_widget: QWidget, help_key: str) -> None:
        tooltip = _localized_text(EXTRACTION_HELP_TEXTS, help_key, self._ui_language)
        if label_widget is not None:
            label_widget.setToolTip(tooltip)
        field_widget.setToolTip(tooltip)

    def _renumber_fixed_via_rows(self) -> None:
        for index, row in enumerate(self._fixed_via_rows, start=1):
            label = row["label"]
            if isinstance(label, QLabel):
                label.setText(f"via{index}")

    def _clear_fixed_via_rows(self) -> None:
        while self._fixed_via_rows:
            row = self._fixed_via_rows.pop()
            widget = row["widget"]
            if isinstance(widget, QWidget):
                self.fixed_via_rows_layout.removeWidget(widget)
                widget.deleteLater()

    def _fixed_via_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for row in self._fixed_via_rows:
            width_spin = row["width_spin"]
            height_spin = row["height_spin"]
            if isinstance(width_spin, QSpinBox) and isinstance(height_spin, QSpinBox):
                pairs.append((int(width_spin.value()), int(height_spin.value())))
        return pairs

    def _delete_fixed_via_row(self, row_widget: QWidget) -> None:
        for index, row in enumerate(self._fixed_via_rows):
            if row["widget"] is row_widget:
                self._fixed_via_rows.pop(index)
                self.fixed_via_rows_layout.removeWidget(row_widget)
                row_widget.deleteLater()
                self._renumber_fixed_via_rows()
                if not self._suspend_fixed_via_updates:
                    self._on_extraction_settings_changed()
                return

    def _add_fixed_via_row(self, *_args, width: int = 1, height: int = 1) -> None:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        via_label = QLabel("")
        via_label.setMinimumWidth(44)
        width_spin = QSpinBox()
        width_spin.setRange(1, 100_000)
        width_spin.setValue(max(1, int(width)))
        width_spin.setPrefix("X ")
        height_spin = QSpinBox()
        height_spin.setRange(1, 100_000)
        height_spin.setValue(max(1, int(height)))
        height_spin.setPrefix("Y ")
        remove_button = QPushButton("-")
        remove_button.setFixedWidth(36)
        remove_button.setMinimumHeight(30)
        remove_button.setStyleSheet(
            "QPushButton { background-color: #d64545; color: white; font-size: 18px; font-weight: 700; border-radius: 6px; }"
            "QPushButton:hover { background-color: #bf3838; }"
            "QPushButton:pressed { background-color: #a93030; }"
        )

        width_spin.valueChanged.connect(self._on_extraction_settings_changed)
        height_spin.valueChanged.connect(self._on_extraction_settings_changed)
        remove_button.clicked.connect(lambda _checked=False, widget=row_widget: self._delete_fixed_via_row(widget))

        self._fixed_via_rows.append(
            {
                "widget": row_widget,
                "label": via_label,
                "width_spin": width_spin,
                "height_spin": height_spin,
                "remove_button": remove_button,
            }
        )

        row_layout.addWidget(via_label)
        row_layout.addWidget(width_spin, 1)
        row_layout.addWidget(height_spin, 1)
        row_layout.addWidget(remove_button)
        self.fixed_via_rows_layout.addWidget(row_widget)
        self._renumber_fixed_via_rows()

        width_spin.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "fixed_via_widths", self._ui_language))
        height_spin.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "fixed_via_heights", self._ui_language))
        remove_button.setToolTip(
            "Удаляет эту строку с допустимым размером via из списка."
            if self._ui_language == "ru"
            else "Removes this allowed via-size row from the list."
        )

        if not self._suspend_fixed_via_updates:
            self._on_extraction_settings_changed()

    def _apply_extraction_tooltips(self) -> None:
        self._set_field_tooltip(self.retrieval_mode_label_widget, self.retrieval_mode_combo, "retrieval_mode")
        self._set_field_tooltip(
            self.approximation_mode_label_widget, self.approximation_mode_combo, "approximation_mode"
        )
        self._set_field_tooltip(
            self.epsilon_label_widget,
            self.epsilon_row_widget if hasattr(self, "epsilon_row_widget") else self.epsilon_spin,
            "epsilon",
        )
        self._set_field_tooltip(self.epsilon_mode_label_widget, self.epsilon_relative_checkbox, "epsilon_mode")
        self._set_field_tooltip(self.min_area_label_widget, self.min_area_spin, "min_area")
        self._set_field_tooltip(self.max_area_label_widget, self.max_area_spin, "max_area")
        self._set_field_tooltip(self.min_perimeter_label_widget, self.min_perimeter_spin, "min_perimeter")
        self._set_field_tooltip(self.max_perimeter_label_widget, self.max_perimeter_spin, "max_perimeter")
        self._set_field_tooltip(self.min_point_count_label_widget, self.min_points_spin, "min_points")
        self._set_field_tooltip(
            self.min_polygon_width_label_widget, self.min_polygon_width_spin, "min_polygon_width"
        )
        self._set_field_tooltip(self.min_bbox_width_label_widget, self.min_bbox_width_spin, "min_bbox_width")
        self._set_field_tooltip(self.max_bbox_width_label_widget, self.max_bbox_width_spin, "max_bbox_width")
        self._set_field_tooltip(self.min_bbox_height_label_widget, self.min_bbox_height_spin, "min_bbox_height")
        self._set_field_tooltip(self.max_bbox_height_label_widget, self.max_bbox_height_spin, "max_bbox_height")
        self._set_field_tooltip(self.min_aspect_ratio_label_widget, self.min_aspect_ratio_spin, "min_aspect_ratio")
        self._set_field_tooltip(self.max_aspect_ratio_label_widget, self.max_aspect_ratio_spin, "max_aspect_ratio")
        self._set_field_tooltip(
            self.border_handling_label_widget, self.exclude_border_touching_checkbox, "exclude_border_touching"
        )
        self._set_field_tooltip(self.min_solidity_label_widget, self.min_solidity_spin, "min_solidity")
        self._set_field_tooltip(self.min_extent_label_widget, self.min_extent_spin, "min_extent")
        self._set_field_tooltip(self.via_size_mode_label_widget, self.via_size_mode_combo, "via_size_mode")
        if getattr(self, "via_search_mode_label_widget", None) is not None:
            self._set_field_tooltip(self.via_search_mode_label_widget, self.via_search_mode_combo, "via_search_mode")
        if hasattr(self, "bright_via_viamode_label_widget"):
            self._set_field_tooltip(self.bright_via_viamode_label_widget, self.via_search_mode_combo, "via_search_mode")
        self._set_field_tooltip(self.via_white_range_label_widget, self.via_white_range_widget, "via_white_range")
        self._set_field_tooltip(self.via_black_range_label_widget, self.via_black_range_widget, "via_black_range")
        self._set_field_tooltip(self.via_min_score_label_widget, self.via_min_score_spin, "via_min_score")
        self._set_field_tooltip(self.via_min_contrast_label_widget, self.via_min_contrast_spin, "via_min_contrast")
        self._set_field_tooltip(
            self.via_min_edge_coverage_label_widget,
            self.via_min_edge_coverage_spin,
            "via_min_edge_coverage",
        )
        self._set_field_tooltip(
            self.via_spot_line_suppression_label_widget,
            self.via_spot_line_suppression_spin,
            "via_spot_line_suppression",
        )
        self._set_field_tooltip(
            self.via_template_min_score_label_widget, self.via_template_min_score_spin, "via_template_min_score"
        )
        self._set_field_tooltip(self.via_templates_label_widget, self.via_templates_widget, "via_templates")
        self._set_field_tooltip(self.via_preset_label_widget, self.via_preset_widget, "via_preset_selector")
        if getattr(self, "noisy_traces_via_preset_label_widget", None) is not None:
            self._set_field_tooltip(
                self.noisy_traces_via_preset_label_widget,
                self.noisy_traces_via_preset_button,
                "via_noisy_traces_preset",
            )
        else:
            self.noisy_traces_via_preset_button.setToolTip(
                _localized_text(EXTRACTION_HELP_TEXTS, "via_noisy_traces_preset", self._ui_language)
            )
        if getattr(self, "blurred_via_preset_label_widget", None) is not None:
            self._set_field_tooltip(
                self.blurred_via_preset_label_widget,
                self.blurred_via_preset_button,
                "via_blurred_preset",
            )
        else:
            self.blurred_via_preset_button.setToolTip(
                _localized_text(EXTRACTION_HELP_TEXTS, "via_blurred_preset", self._ui_language)
            )
        self._set_field_tooltip(self.reset_via_search_label_widget, self.reset_via_search_button, "reset_via_search")
        self.add_via_template_button.setToolTip(
            _localized_text(EXTRACTION_HELP_TEXTS, "via_templates", self._ui_language)
        )
        self.remove_via_template_button.setToolTip(
            "Удаляет выбранный шаблон via из списка."
            if self._ui_language == "ru"
            else "Removes the selected via template from the list."
        )
        self.clear_via_templates_button.setToolTip(
            "Удаляет все сохраненные шаблоны via из списка."
            if self._ui_language == "ru"
            else "Removes all saved via templates from the list."
        )
        for checkbox, tooltip_key in (
            (self.via_white_range_checkbox, "via_white_range"),
            (self.via_black_range_checkbox, "via_black_range"),
        ):
            detector_tooltip = _localized_text(EXTRACTION_HELP_TEXTS, tooltip_key, self._ui_language)
            checkbox.setToolTip(detector_tooltip)
            checkbox.setStatusTip(detector_tooltip)
        self._set_field_tooltip(self.debug_candidates_label_widget, self.debug_candidates_checkbox, "debug_candidates")
        self._set_field_tooltip(self.via_roundness_label_widget, self.via_roundness_spin, "via_min_roundness")
        self._set_field_tooltip(self.min_via_width_label_widget, self.min_via_width_spin, "min_via_width")
        self._set_field_tooltip(self.max_via_width_label_widget, self.max_via_width_spin, "max_via_width")
        self._set_field_tooltip(self.min_via_height_label_widget, self.min_via_height_spin, "min_via_height")
        self._set_field_tooltip(self.max_via_height_label_widget, self.max_via_height_spin, "max_via_height")
        self._set_field_tooltip(self.fixed_vias_label_widget, self.fixed_vias_widget, "fixed_via_widths")
        self.fixed_via_add_button.setToolTip(
            "Добавляет еще одну допустимую пару ширины и высоты via."
            if self._ui_language == "ru"
            else "Adds another allowed via width and height pair."
        )
        for row in self._fixed_via_rows:
            width_spin = row["width_spin"]
            height_spin = row["height_spin"]
            remove_button = row["remove_button"]
            if isinstance(width_spin, QSpinBox):
                width_spin.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "fixed_via_widths", self._ui_language))
            if isinstance(height_spin, QSpinBox):
                height_spin.setToolTip(_localized_text(EXTRACTION_HELP_TEXTS, "fixed_via_heights", self._ui_language))
            if isinstance(remove_button, QPushButton):
                remove_button.setToolTip(
                    "Удаляет эту строку с допустимым размером via из списка."
                    if self._ui_language == "ru"
                    else "Removes this allowed via-size row from the list."
                )
        self._set_field_tooltip(
            self.min_hierarchy_depth_label_widget, self.min_hierarchy_depth_spin, "min_hierarchy_depth"
        )
        self._set_field_tooltip(
            self.max_hierarchy_depth_label_widget, self.max_hierarchy_depth_spin, "max_hierarchy_depth"
        )
        self._set_field_tooltip(
            self.max_hole_area_ratio_label_widget, self.max_hole_area_ratio_spin, "max_hole_area_ratio"
        )
        self._apply_bright_via_tooltips()

    def _apply_bright_via_tooltips(self) -> None:
        if not hasattr(self, "bright_via_diameter_min_spin"):
            return
        ru = self._ui_language == "ru"

        def tt(ru_text: str, en_text: str) -> str:
            return ru_text if ru else en_text

        self.bright_via_diameter_min_spin.setToolTip(
            tt(
                "Минимальный допустимый размер переходного отверстия в пикселях.\n"
                "Если значение слишком большое — маленькие via будут пропущены.\n"
                "Если слишком маленькое — появится больше ложных срабатываний на шуме.\n"
                "Обычно: 5–8 px.",
                "Minimum via diameter in pixels (typ. 5–8).",
            )
        )
        self.bright_via_diameter_max_spin.setToolTip(
            tt(
                "Максимальный допустимый размер via.\n"
                "Если слишком маленькое — крупные via будут пропущены.\n"
                "Если слишком большое — алгоритм начнёт принимать яркие фрагменты дорожек.\n"
                "Обычно: 8–14 px.",
                "Maximum via diameter in pixels (typ. 8–14).",
            )
        )
        self.bright_via_clahe_clip_spin.setToolTip(
            tt(
                "Предел усиления локального контраста (CLAHE).\n"
                "Больше значение — сильнее вытягиваются слабые детали, но растёт шум.\n"
                "Меньше — картинка ровнее, но слабые via могут стать незаметнее.\n"
                "Типично 1.5–3.5.",
                "CLAHE clip limit; higher emphasizes weak details and noise.",
            )
        )
        self.bright_via_clahe_tile_spin.setToolTip(
            tt(
                "Размер ячейки сетки CLAHE в пикселях.\n"
                "Меньше — контраст подстраивается локальнее (мелкие объекты), больше шума на мелкой текстуре.\n"
                "Больше — более глобально, меньше артефактов на зерне, но слабее локальный контраст.\n"
                "Часто 6–12.",
                "CLAHE tile size; smaller = more local adaptation.",
            )
        )
        self.bright_via_median_kernel_spin.setToolTip(
            tt(
                "Размер медианного фильтра (нечётное число; 1 = отключено по смыслу).\n"
                "Больше — сильнее подавление шума SEM, но мягче края via.\n"
                "Меньше — лучше сохраняются острые via, выше риск ложных точек.\n"
                "Типично 3.",
                "Median blur kernel (odd); larger removes more noise and softens edges.",
            )
        )
        self.bright_via_tophat_kernel_spin.setToolTip(
            tt(
                "Размер структурного элемента для белого top-hat (нечётное).\n"
                "Больше — подчёркиваются более крупные яркие вкрапления, фон на большей шкале.\n"
                "Меньше — чувствительнее к мелким пятнам и зерну.\n"
                "Сопоставляйте с ожидаемым диаметром via.",
                "White top-hat structuring size; match expected via scale.",
            )
        )
        self.bright_via_dog_small_spin.setToolTip(
            tt(
                "Меньшая сигма Гаусса в разности гауссов (DoG).\n"
                "Вместе с большой сигмой задаёт масштаб выделяемых ярких деталей.\n"
                "Слишком большая малая сигма — больше отклика на мелкий шум.\n"
                "Должна быть строго меньше «большой сигмы».",
                "DoG small sigma; must be < large sigma.",
            )
        )
        self.bright_via_dog_large_spin.setToolTip(
            tt(
                "Большая сигма Гаусса в DoG.\n"
                "Больше значение — сильнее сглаживание «крупного» масштаба, иначе выделяется фон.\n"
                "Меньше — остаётся больше мелких деталей в отклике.\n"
                "Подбирайте пару с малой сигмой под размер via.",
                "DoG large sigma; tune with small sigma for via size.",
            )
        )
        self.bright_via_threshold_percentile_spin.setToolTip(
            tt(
                "Определяет, насколько ярким должен быть пиксель, чтобы попасть в маску отклика.\n"
                "Большее значение → меньше ложных срабатываний, но больше пропусков.\n"
                "Меньшее значение → выше полнота поиска, но больше шума.\n"
                "Обычно: 97.5–99.2.",
                "Response percentile threshold (typ. 97.5–99.2).",
            )
        )
        self.bright_via_mask_combine_combo.setToolTip(
            tt(
                "ИЛИ — высокая полнота поиска, больше кандидатов.\n"
                "И — строгий режим, меньше ложных срабатываний, но больше пропусков.\n"
                "Обычно рекомендуется начинать с режима ИЛИ.",
                "OR = high recall; AND = stricter overlap of top-hat and DoG masks.",
            )
        )
        self.bright_via_min_area_factor_spin.setToolTip(
            tt(
                "Нижняя граница площади кандидата относительно площади идеального круга минимального диаметра.\n"
                "Больше — отсекаются слишком маленькие пятна (часто шум).\n"
                "Меньше — допускаются более мелкие объекты.\n"
                "Меняйте, если стабильно теряются мелкие via или наоборот много «крошек».",
                "Min area as a factor of π·(d_min/2)².",
            )
        )
        self.bright_via_max_area_factor_spin.setToolTip(
            tt(
                "Верхняя граница площади кандидата относительно площади круга максимального диаметра.\n"
                "Меньше — жёстче отсекаются крупные пятна (часто куски дорожек).\n"
                "Больше — допускаются более крупные отклики.\n"
                "Согласуйте с реальным размером via на SEM.",
                "Max area factor relative to max diameter.",
            )
        )
        self.bright_via_min_circularity_spin.setToolTip(
            tt(
                "Ожидаемая «круглость» контура (4π·area/perimeter²).\n"
                "Низкие значения допускают вытянутые пятна (часто артефакты дорожек).\n"
                "Высокие — ближе к диску, но реальные размытые via могут получать меньший балл.\n"
                "Обычно 0.15–0.45 в зависимости от качества изображения.",
                "Circularity expectation for blob shape (0–1).",
            )
        )
        self.bright_via_min_aspect_spin.setToolTip(
            tt(
                "Минимальное отношение ширины bounding box к высоте.\n"
                "Слишком большое — отсекаются слегка вытянутые via.\n"
                "Слишком маленькое — пропускаются сильно вытянутые ложные объекты реже.\n"
                "Для via обычно около 0.4–0.6.",
                "Min aspect ratio w/h of bbox.",
            )
        )
        self.bright_via_max_aspect_spin.setToolTip(
            tt(
                "Максимальное отношение сторон bbox.\n"
                "Меньше — строже к вытянутым контурам (меньше дорожных «колбас»).\n"
                "Больше — допускаются более вытянутые кандидаты.\n"
                "Слишком большое — растут ложные на границах дорожек.",
                "Max aspect ratio w/h of bbox.",
            )
        )
        self.bright_via_bright_center_score_spin.setToolTip(
            tt(
                "Центр via должен быть ярче окружающей области (разница средних по диску и кольцу).\n"
                "Увеличение значения уменьшает ложные срабатывания на слабом шуме,\n"
                "но может пропускать слабые или размытые via.\n"
                "Это жёсткий порог: ниже — кандидат отбрасывается сразу.",
                "Hard minimum center-vs-ring brightness delta.",
            )
        )
        self.bright_via_max_radial_asymmetry_spin.setToolTip(
            tt(
                "Проверяет симметричность яркости вокруг via (СКО по 8 направлениям).\n"
                "Настоящее via обычно симметрично, край дорожки — нет.\n"
                "Порог задаёт, насколько большой разброс ещё считается «похожим на via» в мягком режиме.\n"
                "Меньше значение в мягком режиме сильнее снижает итоговую оценку при асимметрии.\n"
                "Слишком жёсткий ручной отбор (если включить жёсткий режим) ведёт к пропускам на шуме.",
                "Reference level for radial brightness asymmetry (std).",
            )
        )
        self.bright_via_max_edge_likeness_spin.setToolTip(
            tt(
                "Ограничивает срабатывания на краях металлизации.\n"
                "Меньше значение — сильнее штраф в мягком режиме за «краевой» профиль.\n"
                "Больше — терпимее к via у границы дорожки.\n"
                "С жёстким режимом (если включён) пары с метрикой выше порога отбрасываются сразу.",
                "Edge-likeness cap / soft scale.",
            )
        )
        self.bright_via_max_line_likeness_spin.setToolTip(
            tt(
                "Отсекает объекты, похожие на куски дорожек (анизотропия градиентов в окне).\n"
                "Большее значение — мягче к вытянутым откликам, выше риск ложных срабатываний на трассы.\n"
                "Меньшее — жёстче к линиям, но больше риск пропуска via, слитых с трассой.\n"
                "В мягком режиме влияет на итоговый балл; в жёстком — и на немедленный отказ.",
                "Line-likeness (structure tensor) cap / scale.",
            )
        )
        self.bright_via_metal_constraint_combo.setToolTip(
            tt(
                "Определяет, использовать ли информацию о металлизации (Otsu+морфология).\n"
                "Отключено — не учитывать металл.\n"
                "Мягкая оценка — металл влияет только на итоговую оценку (бонус к баллу).\n"
                "Жёсткий фильтр — кандидаты вне металла с низкой долей покрытия отбрасываются.\n"
                "Если металл плохо виден, используйте «Отключено» или «Мягкая оценка».",
                "Metal mask: disabled / soft score / strict reject.",
            )
        )
        self.bright_via_metal_fraction_spin.setToolTip(
            tt(
                "Минимальная доля пикселей металла в окне вокруг кандидата для режима «Жёсткий фильтр».\n"
                "Выше — принимаются только via, лежащие на металлизации по маске.\n"
                "Ниже — больше кандидатов проходят, но растут ложные вне металла.\n"
                "В мягком режиме на порог ориентироваться не обязательно: используется непрерывный бонус.",
                "Min metal fraction for strict mode (0–1).",
            )
        )
        self.bright_via_min_final_score_spin.setToolTip(
            tt(
                "Главный параметр отбора итоговых via по суммарной оценке 0…100 (форма + локальные метрики).\n"
                "Увеличение → меньше ложных срабатываний, но больше пропусков.\n"
                "Уменьшение → больше найденных via, но больше кандидатов ниже порога (жёлтые на отладке).\n"
                "Обычно это один из самых важных параметров настройки.",
                "Minimum composite score (0–100) to accept a via.",
            )
        )
        self.bright_via_nms_distance_spin.setToolTip(
            tt(
                "Минимальное расстояние между двумя кандидатами после этапа слияния и подавления дублей.\n"
                "Если слишком маленькое — одно via может быть найдено несколько раз с разных откликов.\n"
                "Если слишком большое — соседние реальные via могут сливаться.\n"
                "Связывайте с ожидаемым шагом растра via.",
                "Non-maximum suppression distance in pixels.",
            )
        )
        self.bright_via_show_rejected_checkbox.setToolTip(
            tt(
                "Если включено, на итоговом наложении в отладке рисуются и отклонённые кандидаты: "
                "жёлтые — ниже порога итоговой оценки, красные — жёстко отброшенные по геометрии/контрасту/металлу.\n"
                "Если выключено — видны только принятые (зелёные).",
                "Show soft/hard rejected candidates on the debug overlay.",
            )
        )
        self.bright_via_hard_asym_checkbox.setToolTip(
            tt(
                "Если включено: при превышении «максимальной радиальной асимметрии» кандидат сразу отбрасывается.\n"
                "По умолчанию (выкл.) асимметрия влияет на балл, а не на мгновенный отказ.\n"
                "Включайте только если уверенно настроили порог по этой метрике.",
                "Hard-reject on radial asymmetry vs threshold.",
            )
        )
        self.bright_via_hard_edge_checkbox.setToolTip(
            tt(
                "Если включено: при слишком высокой «похожести на край» кандидат сразу отбрасывается.\n"
                "По умолчанию метрика только снижает итоговый балл.\n"
                "Полезно, если остаются устойчивые ложные на кромках металла после настройки мягкого скоринга.",
                "Hard-reject when edge-likeness exceeds cap.",
            )
        )
        self.bright_via_hard_line_checkbox.setToolTip(
            tt(
                "Если включено: при слишком высокой линейности (анизотропии градиентов) — мгновенный отказ.\n"
                "По умолчанию влияет на балл, чтобы не терять слабые круги на фоне трасс.\n"
                "Включайте при массовых ложных вдоль дорожек.",
                "Hard-reject when line-likeness exceeds cap.",
            )
        )
        self.preview_bright_via_mask_button.setToolTip(
            tt(
                "Переключает профиль на поиск via, режим «яркий top-hat/DoG», "
                "включает отладочные слои и открывает окно с картами (исходник, top-hat, DoG, маски, итог).",
                "Switch to bright via mode and open debug map window.",
            )
        )
        self.reset_bright_via_button.setToolTip(
            tt(
                "Сбрасывает параметры детектора к заводским значениям и запускает пересчёт (как при изменении настроек).",
                "Reset bright via parameters to defaults and re-run.",
            )
        )
        for w in (self.bright_via_diameter_range_widget,):
            w.setToolTip(
                tt(
                    "Пара min/max: см. подсказки у полей минимума и максимума диаметра.",
                    "Diameter range: see min and max tooltips.",
                )
            )
        if hasattr(self, "recognition_mode_combo"):
            self.recognition_mode_combo.setToolTip(
                tt(
                    "Выбор режима извлечения. По умолчанию включено «Без извлечения»; обработка запускается только после явного выбора режима.\n"
                    "Параметры на панели меняются в зависимости от режима.",
                    "Extraction mode. Defaults to No extraction; processing runs only after an explicit mode choice.",
                )
            )
        if hasattr(self, "via_search_sensitivity_combo"):
            self.via_search_sensitivity_combo.setToolTip(
                tt(
                    "Общий уровень агрессии поиска: «Низкая» — меньше ложных, больше пропусков; "
                    "«Средняя» — баланс; «Высокая» — больше срабатываний и кандидатов.\n"
                    "Меняет пороги и фильтры; в «Дополнительно» значения можно подправить вручную.",
                    "Coarse sensitivity for via search; adjust advanced fields manually if needed.",
                )
            )
        if hasattr(self, "via_show_detected_checkbox"):
            self.via_show_detected_checkbox.setToolTip(
                tt(
                    "Показывать на изображении полигоны via, найденные автоматически.",
                    "Show auto-detected via polygons on the image.",
                )
            )
        if hasattr(self, "via_debug_gradient_map_checkbox"):
            self.via_debug_gradient_map_checkbox.setToolTip(
                tt(
                    "Сохранять и показывать отладочные карты (градиент, маски) в окне «карта градиента» и при клике по отладке.",
                    "Enable extra debug image maps in the gradient / inspect views.",
                )
            )
        if getattr(self, "metal_preset_combo", None) is not None:
            self.metal_preset_combo.setToolTip(
                tt(
                    "Готовый набор порогов и морфологии под тип слоя.\n"
                    "«Стандартный» — универсальный баланс; «Плотная металлизация» — чуть агрессивнее к шуму; "
                    "«Тонкие дорожки» — ниже минимальная ширина; «Шумное SEM» — жёстче отсев; "
                    "«Консервативный» — меньше ложных, выше пороги длины/прямолинейности.",
                    "Preset bundle for metal recovery.",
                )
            )
        if getattr(self, "metal_sensitivity_slider", None) is not None:
            self.metal_sensitivity_slider.setToolTip(
                tt(
                    "Единый регулятор чувствительности 0–100: увеличение добавляет пиксели в маску и чаще оставляет слабые дорожки, "
                    "но усиливает ложные срабатывания на зерне и артефактах; уменьшение убирает шум, но может проглотить тусклые реальные проводники.\n"
                    "Типичный диапазон 35–65; при «Шумном SEM» чаще 30–45, при контрастных кадрах 55–70.",
                    "Unified sensitivity 0–100 for internal thresholds.",
                )
            )
        if getattr(self, "metal_min_width_spin", None) is not None:
            self.metal_min_width_spin.setToolTip(
                tt(
                    "Оценка эффективной ширины по маске (медиальное ядро): объекты уже порога отбрасываются как шумовые царапины.\n"
                    "Увеличение убирает тонкие ложные сегменты, но может отрезать реальные узкие дорожки; уменьшение спасает тонкие линии, но пропускает больше мусора.\n"
                    "Стартуйте с 6–10 px для тонких технологий и 10–14 px для грубого SEM.",
                    "Minimum conductor width in pixels.",
                )
            )
        if getattr(self, "metal_max_width_spin", None) is not None:
            self.metal_max_width_spin.setToolTip(
                tt(
                    "Верхняя граница ширины: отсекает широкие заливки, контактные площадки и яркие «пятна», не являющиеся трассами.\n"
                    "0 или пусто — без ограничения. Уменьшайте максимум, если в результат попадают крупные артефакты; увеличивайте, если режет широкие шины.\n"
                    "Часто 40–120 px в зависимости от масштаба кадра.",
                    "Maximum trace width; 0 = unlimited.",
                )
            )
        if getattr(self, "metal_min_length_spin", None) is not None:
            self.metal_min_length_spin.setToolTip(
                tt(
                    "Минимальная длина по ограничивающему прямоугольнику: короткие фрагменты травления и одиночные засветы отсекаются.\n"
                    "Увеличение сильнее чистит шум; уменьшение сохраняет короткие, но реальные сегменты (перемычки, стабы).\n"
                    "Рабочий диапазон обычно 18–40 px.",
                    "Minimum trace length.",
                )
            )
        if getattr(self, "metal_use_wide_gradient_checkbox", None) is not None:
            self.metal_use_wide_gradient_checkbox.setToolTip(
                tt(
                    "Включает дополнительное восстановление широких проводников по ярким краям. Полезно для SEM, где ярко видны только границы проводника, "
                    "а центр похож на фон. Может находить широкие дорожки, которые пропускает обычная бинаризация, но при слишком шумном изображении "
                    "может добавить ложные срабатывания.",
                    "Wide conductor recovery from bright edges (SEM).",
                )
            )
        if getattr(self, "metal_wide_grad_radius_spin", None) is not None:
            self.metal_wide_grad_radius_spin.setToolTip(
                tt(
                    "Сколько пикселей по обе стороны от яркого края используется для анализа профиля яркости. Увеличение помогает для широких и размытых "
                    "проводников, но может захватывать соседние объекты.",
                    "Gradient profile half-width in pixels.",
                )
            )
        if getattr(self, "metal_wide_grad_conf_spin", None) is not None:
            self.metal_wide_grad_conf_spin.setToolTip(
                tt(
                    "Насколько явно одна сторона края похожа на фон, а другая — на внутреннюю часть проводника. Увеличение делает режим строже и уменьшает "
                    "ложные пары краёв, но может пропустить слабые проводники.",
                    "Minimum direction confidence.",
                )
            )
        if getattr(self, "metal_wide_grad_pair_len_spin", None) is not None:
            self.metal_wide_grad_pair_len_spin.setToolTip(
                tt(
                    "Минимальная длина двух параллельных границ, чтобы они считались сторонами широкого проводника. Увеличение отсекает короткие шумовые линии, "
                    "уменьшение помогает находить короткие проводники.",
                    "Minimum parallel edge length for pairing.",
                )
            )
        if getattr(self, "metal_wide_grad_parallel_spin", None) is not None:
            self.metal_wide_grad_parallel_spin.setToolTip(
                tt(
                    "Максимальное отличие углов двух границ. Меньшее значение требует почти параллельных краёв, большее допускает искажённые SEM-границы.",
                    "Parallelism tolerance in degrees.",
                )
            )
        if getattr(self, "metal_wide_grad_gap_spin", None) is not None:
            self.metal_wide_grad_gap_spin.setToolTip(
                tt(
                    "Позволяет соединять прерывистые яркие края. Увеличение помогает на шумных изображениях, но может ошибочно соединять разные объекты.",
                    "Max gap for Hough line linking.",
                )
            )
        if getattr(self, "metal_wide_grad_overlap_spin", None) is not None:
            self.metal_wide_grad_overlap_spin.setToolTip(
                tt(
                    "Минимальная доля перекрытия двух границ по длине. Увеличение делает поиск пар строже, уменьшение допускает частично видимые края.",
                    "Minimum overlap ratio of paired edges.",
                )
            )
        if getattr(self, "metal_segmentation_method_combo", None) is not None:
            self.metal_segmentation_method_combo.setToolTip(
                tt(
                    "По умолчанию — без глобальной пороговой сегментации: контуры строятся по границам на grayscale (Canny + локальная морфология), что лучше сохраняет топологию тонких проводников на SEM.\n"
                    "Otsu / адаптивная — классическая бинаризация яркости (опционально). Гибрид — объединение граничной маски и Otsu, если нужны и островки по яркости.",
                    "Default: grayscale edge-based mask (topology-first). Otsu/Adaptive: optional intensity thresholding. Hybrid: edges OR Otsu.",
                )
            )
        if getattr(self, "metal_sensitivity_combo", None) is not None:
            self.metal_sensitivity_combo.setToolTip(
                tt(
                    "Грубый уровень вместе со слайдером 0–100: «Низкая» — эрозия/порог жёстче, меньше ложных; «Высокая» — больше пикселей в маске.\n"
                    "Используйте как быстрый сдвиг до тонкой подстройки слайдером.",
                    "Coarse low/medium/high bias paired with slider.",
                )
            )
        if getattr(self, "metal_show_conductors_checkbox", None) is not None:
            self.metal_show_conductors_checkbox.setToolTip(
                tt("Показывать принятые полигоны проводников на сцене редактора.", "Show accepted conductor polygons.")
            )
        if getattr(self, "metal_show_rejected_checkbox", None) is not None:
            self.metal_show_rejected_checkbox.setToolTip(
                tt(
                    "Красным контуром показать отклонённые компоненты (после фильтров). Полезно понять, что алгоритм отбрасывает.",
                    "Draw rejected candidates in red.",
                )
            )
        if getattr(self, "metal_show_suspicious_checkbox", None) is not None:
            self.metal_show_suspicious_checkbox.setToolTip(
                tt(
                    "Жёлтым — объекты, прошедшие фильтр, но с пограничными углами или прямолинейностью; проверьте вручную.",
                    "Highlight borderline accepted traces in yellow.",
                )
            )
        if getattr(self, "metal_show_border_checkbox", None) is not None:
            self.metal_show_border_checkbox.setToolTip(
                tt(
                    "Синим — проводники, касающиеся края кадра (часто обрезаны SEM). Не ошибка, но требует осторожности при метриках.",
                    "Highlight border-touching traces in blue.",
                )
            )
        if getattr(self, "metal_show_mask_checkbox", None) is not None:
            self.metal_show_mask_checkbox.setToolTip(
                tt(
                    "Включить цветное наложение поверх изображения по выбранному режиму отладки (маска, контуры, фильтр и т.д.).",
                    "Enable debug / mask overlay on the image.",
                )
            )
        if getattr(self, "metal_debug_visual_combo", None) is not None:
            self.metal_debug_visual_combo.setToolTip(
                tt(
                    "Что именно рисуется в оверлее: итоговая смесь, сырая маска, контуры или этапы фильтрации.",
                    "Which debug channel is shown in the overlay.",
                )
            )
        if getattr(self, "metal_overlay_opacity_spin", None) is not None:
            self.metal_overlay_opacity_spin.setToolTip(
                tt("Прозрачность оверлея отладки/маски (0.05–1.0).", "Overlay opacity.")
            )
        if getattr(self, "metal_min_area_spin", None) is not None:
            self.metal_min_area_spin.setToolTip(
                tt(
                    "Минимальная площадь компонента в px² после бинаризации; отсекает мелкие засветы.\n"
                    "Увеличение — меньше шумовых островков; уменьшение — спасает тонкие, но короткие фрагменты.\n"
                    "Часто 40–120.",
                    "Minimum area filter.",
                )
            )
        if getattr(self, "metal_max_area_spin", None) is not None:
            self.metal_max_area_spin.setToolTip(
                tt("Максимальная площадь (0 = нет лимита); режет крупные заливки.", "Maximum area, 0 = off.")
            )
        if getattr(self, "metal_min_perimeter_spin", None) is not None:
            self.metal_min_perimeter_spin.setToolTip(
                tt("Минимальный периметр контура; дополнительный отсев «крошки» вокруг реальных трасс.", "Minimum perimeter.")
            )
        if getattr(self, "metal_max_perimeter_spin", None) is not None:
            self.metal_max_perimeter_spin.setToolTip(
                tt("Максимальный периметр (0 = нет); для отсечения огромных некорректных компонентов.", "Maximum perimeter.")
            )
        if getattr(self, "metal_epsilon_spin", None) is not None:
            self.metal_epsilon_spin.setToolTip(
                tt(
                    "Epsilon для Douglas–Peucker при упрощении цепочки контура перед проверками углов и топологии.\n"
                    "Больше — меньше вершин, устойчивее к зубцам; меньше — точнее геометрия, но шумнее углы.",
                    "Contour simplify epsilon.",
                )
            )
        if getattr(self, "metal_min_points_spin", None) is not None:
            self.metal_min_points_spin.setToolTip(
                tt("Минимальное число вершин упрощённого полигона для принятия.", "Minimum vertex count.")
            )
        if getattr(self, "metal_min_angle_spin", None) is not None:
            self.metal_min_angle_spin.setToolTip(
                tt(
                    "Подавляет острые «шипы» на контуре: вершины с меньшим внутренним углом выкидываются при упрощении.",
                    "Minimum interior angle at simplified vertices.",
                )
            )
        if getattr(self, "metal_approximation_checkbox", None) is not None:
            self.metal_approximation_checkbox.setToolTip(
                tt("Включить упрощение контура (approxPolyDP); выключите только для отладки сырой цепочки.", "Enable DP simplify.")
            )
        if getattr(self, "metal_hierarchy_combo", None) is not None:
            self.metal_hierarchy_combo.setToolTip(
                tt(
                    "Полная иерархия (RETR_TREE) учитывает вложенность контуров; только внешние — быстрее и проще, если дырки не нужны.",
                    "Contour hierarchy retrieval mode.",
                )
            )
        if getattr(self, "metal_allowed_angles_combo", None) is not None:
            self.metal_allowed_angles_combo.setToolTip(
                tt(
                    "Ограничение на углы трассировки после упрощения: ортогональ, 45°/90° или без ограничений.\n"
                    "Жёстче режим — меньше ложных изломанных контуров, но риск отсечь слегка «кривую» реальную дорожку.",
                    "Allowed routing angles.",
                )
            )
        if getattr(self, "metal_angle_tolerance_spin", None) is not None:
            self.metal_angle_tolerance_spin.setToolTip(
                tt(
                    "На сколько градусов можно отклониться от идеальных 0/45/90°, чтобы угол всё ещё считался допустимым.\n"
                    "Увеличьте при шумном крае; уменьшите, если просачиваются диагональные артефакты. Типично 5–10°.",
                    "Angular tolerance in degrees.",
                )
            )
        if getattr(self, "metal_straightness_spin", None) is not None:
            self.metal_straightness_spin.setToolTip(
                tt(
                    "Отношение «длина по minAreaRect» к периметру: низкие значения характерны для рыхлых, извилистых шумовых масок.\n"
                    "Повышение отсекает пятна и ветвистый мусор; понижение спасает сложные, но реальные формы. Старт 0.55–0.7.",
                    "Minimum straightness metric.",
                )
            )
        if getattr(self, "metal_t_junction_checkbox", None) is not None:
            self.metal_t_junction_checkbox.setToolTip(
                tt(
                    "Разрешать T-образные соединения в растровой маске (один связный компонент с разветвлением).\n"
                    "Выключение слегка ужесточает отбор по выпуклым дефектам — полезно, если шум даёт ложные «тройники» внутри одного контура.",
                    "Allow T-junction topology in mask components.",
                )
            )
        if getattr(self, "metal_border_handling_combo", None) is not None:
            self.metal_border_handling_combo.setToolTip(
                tt(
                    "«Игнорировать» — отбрасывать всё, что касается края кадра; «Принимать» — не отличать; "
                    "«Помечать» — принять, но выделить отдельно (часто обрезанные проводники).",
                    "How to treat image-border-touching components.",
                )
            )
        if getattr(self, "metal_validity_checkbox", None) is not None:
            self.metal_validity_checkbox.setToolTip(
                tt(
                    "Проверка простого замкнутого контура без самопересечений и лишних самокасаний на упрощённой цепочке.\n"
                    "Отключайте только временно для отладки сырой векторизации — иначе в выдачу могут попасть некорректные полигоны.",
                    "Validate simplified ring geometry.",
                )
            )
        if getattr(self, "metal_morph_close_spin", None) is not None:
            self.metal_morph_close_spin.setToolTip(
                tt(
                    "Радиус морфологического closing после порога: склеивает мелкие разрывы маски.\n"
                    "Держите низким (2–4), иначе сливаются близкие несвязанные объекты.",
                    "Closing radius; keep small.",
                )
            )
        if getattr(self, "metal_morph_open_spin", None) is not None:
            self.metal_morph_open_spin.setToolTip(
                tt("Opening для удаления тонкого соли-and-pepper шума; 0 — отключено.", "Opening radius, 0 = off.")
            )
        if getattr(self, "metal_preview_mask_button", None) is not None:
            self.metal_preview_mask_button.setToolTip(
                tt("Переключить оверлей на бинарную маску и включить показ.", "Jump to binary mask overlay.")
            )
        if getattr(self, "metal_reset_params_button", None) is not None:
            self.metal_reset_params_button.setToolTip(
                tt("Сбросить параметры восстановления к значениям по умолчанию.", "Reset metal parameters to defaults.")
            )

    def _update_via_size_controls_state(self) -> None:
        fixed_mode = normalize_via_size_mode(self.via_size_mode_combo.currentData()) == VIA_SIZE_MODE_FIXED
        range_widgets = [
            (self.min_via_width_label_widget, self.via_width_range_widget),
            (self.min_via_height_label_widget, self.via_height_range_widget),
        ]
        fixed_widgets = [
            (self.fixed_vias_label_widget, self.fixed_vias_widget),
        ]
        for label_widget, field_widget in range_widgets:
            if label_widget is not None:
                label_widget.setVisible(not fixed_mode)
            field_widget.setVisible(not fixed_mode)
        for label_widget, field_widget in fixed_widgets:
            if label_widget is not None:
                label_widget.setVisible(fixed_mode)
            field_widget.setVisible(fixed_mode)
        self._update_via_threshold_controls_state()

    def _update_via_threshold_controls_state(self) -> None:
        mode = normalize_via_search_mode(self.via_search_mode_combo.currentData())
        advanced = self._advanced_extraction_enabled()
        bright_enabled = mode == VIA_SEARCH_MODE_HEURISTIC
        blob_enabled = False
        template_enabled = mode == VIA_SEARCH_MODE_TEMPLATE
        for label_widget, field_widget in (
            (self.via_min_score_label_widget, self.via_min_score_spin),
            (self.via_min_contrast_label_widget, self.via_min_contrast_spin),
            (self.via_min_edge_coverage_label_widget, self.via_min_edge_coverage_spin),
            (self.via_spot_line_suppression_label_widget, self.via_spot_line_suppression_spin),
        ):
            if label_widget is not None:
                label_widget.setVisible(advanced and blob_enabled)
            field_widget.setVisible(advanced and blob_enabled)
        if self.via_template_min_score_label_widget is not None:
            self.via_template_min_score_label_widget.setVisible(advanced and template_enabled)
        self.via_template_min_score_spin.setVisible(advanced and template_enabled)
        if self.via_templates_label_widget is not None:
            self.via_templates_label_widget.setVisible(template_enabled)
        self.via_templates_widget.setVisible(template_enabled)
        if hasattr(self, "via_range_checkboxes_label_widget") and self.via_range_checkboxes_label_widget is not None:
            self.via_range_checkboxes_label_widget.setVisible(advanced and not bright_enabled)
        if hasattr(self, "via_range_checkboxes_widget"):
            self.via_range_checkboxes_widget.setVisible(advanced and not bright_enabled)

        white_enabled = self.via_white_range_checkbox.isChecked()
        self.via_white_range_min_spin.setEnabled(white_enabled)
        self.via_white_range_max_spin.setEnabled(white_enabled)
        if self.via_white_range_label_widget is not None:
            self.via_white_range_label_widget.setVisible(advanced and white_enabled and not bright_enabled)
        self.via_white_range_widget.setVisible(advanced and white_enabled and not bright_enabled)
        black_enabled = self.via_black_range_checkbox.isChecked()
        self.via_black_range_min_spin.setEnabled(black_enabled)
        self.via_black_range_max_spin.setEnabled(black_enabled)
        if self.via_black_range_label_widget is not None:
            self.via_black_range_label_widget.setVisible(advanced and black_enabled and not bright_enabled)
        self.via_black_range_widget.setVisible(advanced and black_enabled and not bright_enabled)
        if hasattr(self, "bright_via_group") and hasattr(self, "recognition_mode_combo"):
            self.bright_via_group.setVisible(
                self._active_extraction_profile == "vias"
                and str(self.recognition_mode_combo.currentData() or "") == "via"
            )

    def _update_extraction_profile_controls_state(self) -> None:
        rec = str(self.recognition_mode_combo.currentData() or "conductors") if hasattr(self, "recognition_mode_combo") else "conductors"
        is_via_profile = self._active_extraction_profile == "vias"
        advanced = self._advanced_extraction_enabled()
        show_legacy_via = is_via_profile and rec == "disabled"
        conductors_recognition = rec == "conductors"
        if hasattr(self, "advanced_extraction_checkbox"):
            self.advanced_extraction_checkbox.setVisible(not conductors_recognition)
        if conductors_recognition:
            self.basic_filters_group.setVisible(False)
            self.geometry_filters_group.setVisible(False)
            self.topology_group.setVisible(False)
        else:
            self.basic_filters_group.setVisible(advanced)
            self.geometry_filters_group.setVisible(advanced)
            self.topology_group.setVisible(advanced and (not is_via_profile or rec == "conductors"))
        self.conductor_group.setEnabled(False)
        self.conductor_group.setVisible(False)
        self.via_group.setEnabled(show_legacy_via)
        self.via_group.setVisible(show_legacy_via)
        advanced_via_widgets = [
            (self.via_range_checkboxes_label_widget, self.via_range_checkboxes_widget),
            (self.via_white_range_label_widget, self.via_white_range_widget),
            (self.via_black_range_label_widget, self.via_black_range_widget),
            (self.via_min_score_label_widget, self.via_min_score_spin),
            (self.via_min_contrast_label_widget, self.via_min_contrast_spin),
            (self.via_min_edge_coverage_label_widget, self.via_min_edge_coverage_spin),
            (self.via_spot_line_suppression_label_widget, self.via_spot_line_suppression_spin),
            (self.via_roundness_label_widget, self.via_roundness_spin),
        ]
        in_via_extraction = rec in ("via", "disabled")
        for label_widget, field_widget in advanced_via_widgets:
            if label_widget is not None:
                label_widget.setVisible(advanced and is_via_profile and in_via_extraction)
            field_widget.setVisible(advanced and is_via_profile and in_via_extraction)
        if hasattr(self, "bright_via_group"):
            self.bright_via_group.setVisible(is_via_profile and rec == "via")
        self._sync_recognition_stack_visibility()
        self._update_via_threshold_controls_state()

    def _sync_recognition_stack_visibility(self) -> None:
        if not hasattr(self, "recognition_mode_combo") or not hasattr(self, "recognition_stack"):
            return
        data = str(self.recognition_mode_combo.currentData() or "conductors")
        if data == "via":
            self.recognition_stack.setVisible(False)
        else:
            self.recognition_stack.setVisible(True)
            self.recognition_stack.setCurrentIndex(0 if data == "disabled" else 1)

    def _advanced_extraction_enabled(self) -> bool:
        return bool(hasattr(self, "advanced_extraction_checkbox") and self.advanced_extraction_checkbox.isChecked())

    def _on_advanced_extraction_toggled(self, *_args) -> None:
        self._update_extraction_profile_controls_state()

    def _build_visual_panel(self) -> QWidget:
        return build_visual_panel(self)

    def _build_editor_toolbar(self) -> QWidget:
        return build_editor_toolbar(self)

    def _sync_editor_via_size(self) -> None:
        self.polygon_editor.set_via_size(float(self.via_width_spin.value()), float(self.via_height_spin.value()))

    def _configure_toolbar_button(
        self,
        button: QToolButton,
        icon: QIcon,
        text: str,
        *,
        checkable: bool = False,
    ) -> None:
        button.setIcon(icon)
        button.setIconSize(QSize(self._toolbar_icon_size_px(), self._toolbar_icon_size_px()))
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setToolTip(text)
        button.setStatusTip(text)
        button.setAccessibleName(text)
        button.setAutoRaise(False)
        button.setFixedSize(self._toolbar_button_size_px(), self._toolbar_button_size_px())
        button.setCheckable(checkable)

    def _create_editor_tool_icon(self, tool: EditorTool) -> QIcon:
        return create_editor_tool_icon(tool)

    def _create_editor_action_icon(self, action: str) -> QIcon:
        return create_editor_action_icon(action)

    @staticmethod
    def _toolbar_icon_size_px() -> int:
        return TOOLBAR_ICON_SIZE_PX

    @staticmethod
    def _toolbar_button_size_px() -> int:
        return TOOLBAR_BUTTON_SIZE_PX

    @staticmethod
    def _toolbar_icon_canvas_size_px() -> int:
        return TOOLBAR_ICON_CANVAS_SIZE_PX

    def _tr(self, key: str, default: str = "", **kwargs) -> str:
        return tr(key, default=default, language=self._ui_language, **kwargs)

    def _set_common_tooltip(self, widget: QWidget | None, key: str) -> None:
        if widget is None:
            return
        tooltip = _localized_text(GENERAL_CONTROL_TOOLTIPS, key, self._ui_language)
        widget.setToolTip(tooltip)
        widget.setStatusTip(tooltip)

    def _mode_text(self, key: str) -> str:
        if self._ui_language == "ru":
            mapping = {
                "polygon_points": "По точкам",
                "polygon_rectangle": "Прямоугольник",
                "brush_freeform": "Произвольная",
                "brush_45deg": "45° шаг",
                "brush_stamp_add": "Кружок (добавить)",
                "brush_stamp_erase": "Кружок (стереть)",
                "delete_single": "Вершина",
                "delete_area": "Область",
            }
        else:
            mapping = {
                "polygon_points": "By points",
                "polygon_rectangle": "Rectangle",
                "brush_freeform": "Freeform",
                "brush_45deg": "45° constrained",
                "brush_stamp_add": "Circle (add)",
                "brush_stamp_erase": "Circle (erase)",
                "delete_single": "Single vertex",
                "delete_area": "Area",
            }
        return mapping[key]

    def _busy_indicator_text(self) -> str:
        return "Обработка..." if self._ui_language == "ru" else "Processing..."

    def _set_progress_status(self, key: str, **kwargs) -> None:
        self._progress_status_key = key
        self._progress_status_kwargs = dict(kwargs)

    def set_ui_language(self, language: str | None) -> None:
        self._ui_language = active_language(language)
        self._batch_processor.set_ui_language(self._ui_language)
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_ui_language(self._ui_language)
        self._retranslate_ui()

    def _retranslate_ui(self) -> None:
        retranslate_ui(self)

    def _update_tool_button_texts(self) -> None:
        texts = {
            EditorTool.RULER: self._tr("tool_ruler", "Ruler"),
            EditorTool.ADD_VIA: self._tr("tool_add_via", "Via"),
            EditorTool.SELECT: self._tr("tool_select", "Выбор" if self._ui_language == "ru" else "Select"),
            EditorTool.PAN: self._tr("tool_pan", "Панорамирование" if self._ui_language == "ru" else "Pan"),
            EditorTool.ADD_POLYGON: self._tr(
                "tool_add_polygon", "Полигон" if self._ui_language == "ru" else "Add polygon"
            ),
            EditorTool.BRUSH: self._tr("tool_brush", "Кисть" if self._ui_language == "ru" else "Brush"),
            EditorTool.ADD_VERTEX: self._tr(
                "tool_add_vertex", "Добавить вершину" if self._ui_language == "ru" else "Add vertex"
            ),
            EditorTool.DELETE_VERTEX: self._tr(
                "tool_delete_vertex", "Удалить вершину" if self._ui_language == "ru" else "Delete vertex"
            ),
            EditorTool.MOVE_VERTEX: self._tr(
                "tool_move_vertex", "Переместить вершину" if self._ui_language == "ru" else "Move vertex"
            ),
            EditorTool.DELETE_POLYGON: self._tr(
                "tool_delete_polygon", "Удалить полигон" if self._ui_language == "ru" else "Delete polygon"
            ),
        }
        for tool, button in self._tool_buttons.items():
            label = texts.get(tool, tool.value)
            if tool == EditorTool.RULER:
                label = self._tr("tool_ruler", "Линейка" if self._ui_language == "ru" else "Ruler")
            tooltip_pair = EDITOR_TOOL_TOOLTIPS.get(tool)
            base_tip = (tooltip_pair[0] if self._ui_language == "ru" else tooltip_pair[1]) if tooltip_pair else label
            shortcut_tip = tool_shortcut_native_text(tool)
            tooltip = append_shortcut_to_tooltip(base_tip, shortcut_tip)
            button.setToolTip(tooltip)
            button.setStatusTip(tooltip)
            button.setAccessibleName(label)

    def _update_action_button_texts(self) -> None:
        undo_key = QKeySequence(QKeySequence.StandardKey.Undo).toString(QKeySequence.SequenceFormat.NativeText)
        redo_key = QKeySequence(QKeySequence.StandardKey.Redo).toString(QKeySequence.SequenceFormat.NativeText)
        for button, label in [
            (self.undo_button, self._tr("undo_button", "Отменить" if self._ui_language == "ru" else "Undo")),
            (self.redo_button, self._tr("redo_button", "Повторить" if self._ui_language == "ru" else "Redo")),
            (self.zoom_in_button, self._tr("zoom_in_button", "Увеличить" if self._ui_language == "ru" else "Zoom in")),
            (
                self.zoom_out_button,
                self._tr("zoom_out_button", "Уменьшить" if self._ui_language == "ru" else "Zoom out"),
            ),
            (self.fit_button, self._tr("fit_button", "Подогнать" if self._ui_language == "ru" else "Fit")),
        ]:
            button.setAccessibleName(label)
        shortcuts_map = {
            self.undo_button: undo_key,
            self.redo_button: redo_key,
            self.zoom_in_button: "",
            self.zoom_out_button: "",
            self.fit_button: "",
        }
        for button, tooltip_key in (
            (self.undo_button, "undo_button"),
            (self.redo_button, "redo_button"),
            (self.zoom_in_button, "zoom_in_button"),
            (self.zoom_out_button, "zoom_out_button"),
            (self.fit_button, "fit_button"),
        ):
            tooltip = _localized_text(EDITOR_ACTION_TOOLTIPS, tooltip_key, self._ui_language)
            shortcut = shortcuts_map.get(button, "")
            full_tip = append_shortcut_to_tooltip(tooltip, shortcut) if shortcut else tooltip
            button.setToolTip(full_tip)
            button.setStatusTip(full_tip)

    def _on_editor_tool_changed(self, tool) -> None:
        is_ruler = tool == EditorTool.RULER
        self.ruler_status_label.setVisible(is_ruler)
        if is_ruler and not self.ruler_status_label.text():
            self.ruler_status_label.setText(
                self._tr(
                    "ruler_idle_label",
                    "Потяните на изображении для измерения"
                    if self._ui_language == "ru"
                    else "Drag on the image to measure",
                )
            )
        elif not is_ruler:
            self.ruler_status_label.clear()
        if hasattr(self, "_polygon_toolbar_block"):
            self._polygon_toolbar_block.setVisible(tool == EditorTool.ADD_POLYGON)
        if hasattr(self, "_brush_toolbar_block"):
            self._brush_toolbar_block.setVisible(tool == EditorTool.BRUSH)
        if hasattr(self, "_via_toolbar_block"):
            self._via_toolbar_block.setVisible(tool == EditorTool.ADD_VIA)
        if hasattr(self, "_delete_vertex_toolbar_block"):
            self._delete_vertex_toolbar_block.setVisible(tool == EditorTool.DELETE_VERTEX)
        self._on_effective_polygon_create_mode_changed(self.polygon_editor.effective_polygon_create_mode())

    def _on_effective_polygon_create_mode_changed(self, mode: PolygonCreateMode) -> None:
        if not hasattr(self, "polygon_draw_mode_indicator"):
            return
        if self.polygon_editor.current_tool != EditorTool.ADD_POLYGON:
            self.polygon_draw_mode_indicator.clear()
            return
        if mode == PolygonCreateMode.POINTS:
            text = self._tr(
                "polygon_draw_now_points",
                "Сейчас: по точкам" if self._ui_language == "ru" else "Now: by points",
            )
        else:
            text = self._tr(
                "polygon_draw_now_rectangle",
                "Сейчас: прямоугольник" if self._ui_language == "ru" else "Now: rectangle",
            )
        self.polygon_draw_mode_indicator.setText(text)

    def _update_ruler_status(self, text: str) -> None:
        if not text:
            if self.polygon_editor.current_tool == EditorTool.RULER:
                self.ruler_status_label.setText(
                    self._tr(
                        "ruler_idle_label",
                        "Потяните на изображении для измерения"
                        if self._ui_language == "ru"
                        else "Drag on the image to measure",
                    )
                )
            else:
                self.ruler_status_label.clear()
            return
        self.ruler_status_label.setText(text)

    def _retranslate_editor_mode_combos(self) -> None:
        polygon_mode = self.polygon_mode_combo.currentData()
        brush_mode = self.brush_mode_combo.currentData()
        delete_mode = self.delete_vertex_mode_combo.currentData()

        self.polygon_mode_combo.setItemText(0, self._mode_text("polygon_points"))
        self.polygon_mode_combo.setItemText(1, self._mode_text("polygon_rectangle"))
        self.brush_mode_combo.setItemText(0, self._mode_text("brush_freeform"))
        self.brush_mode_combo.setItemText(1, self._mode_text("brush_45deg"))
        self.brush_mode_combo.setItemText(2, self._mode_text("brush_stamp_add"))
        self.brush_mode_combo.setItemText(3, self._mode_text("brush_stamp_erase"))
        self.delete_vertex_mode_combo.setItemText(0, self._mode_text("delete_single"))
        self.delete_vertex_mode_combo.setItemText(1, self._mode_text("delete_area"))

        polygon_index = self.polygon_mode_combo.findData(polygon_mode)
        brush_index = self.brush_mode_combo.findData(brush_mode)
        delete_index = self.delete_vertex_mode_combo.findData(delete_mode)
        if polygon_index >= 0:
            self.polygon_mode_combo.setCurrentIndex(polygon_index)
        if brush_index >= 0:
            self.brush_mode_combo.setCurrentIndex(brush_index)
        if delete_index >= 0:
            self.delete_vertex_mode_combo.setCurrentIndex(delete_index)

        self._on_effective_polygon_create_mode_changed(self.polygon_editor.effective_polygon_create_mode())

    def _retranslate_contour_mode_combos(self) -> None:
        current_retrieval = self.retrieval_mode_combo.currentData()
        for index in range(self.retrieval_mode_combo.count()):
            mode_name = str(self.retrieval_mode_combo.itemData(index))
            self.retrieval_mode_combo.setItemText(index, self._tr(f"retrieval_mode.{mode_name}", default=mode_name))
        if current_retrieval is not None:
            self.retrieval_mode_combo.setCurrentIndex(self.retrieval_mode_combo.findData(current_retrieval))

        current_approximation = self.approximation_mode_combo.currentData()
        for index in range(self.approximation_mode_combo.count()):
            mode_name = str(self.approximation_mode_combo.itemData(index))
            self.approximation_mode_combo.setItemText(
                index,
                self._tr(f"approximation_mode.{mode_name}", default=mode_name),
            )
        if current_approximation is not None:
            self.approximation_mode_combo.setCurrentIndex(self.approximation_mode_combo.findData(current_approximation))

    def _wrap_group(self, title: str, widget: QWidget) -> QWidget:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _build_checkbox_spin_row(self, checkbox: QCheckBox, spinbox: QAbstractSpinBox) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(checkbox, 1)
        layout.addWidget(spinbox)
        return widget

    def _build_checkbox_range_row(
        self,
        checkbox: QCheckBox,
        min_spinbox: QAbstractSpinBox,
        max_spinbox: QAbstractSpinBox,
    ) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(checkbox, 1)
        layout.addWidget(min_spinbox)
        layout.addWidget(max_spinbox)
        return widget

    def _build_range_row(self, min_spinbox: QAbstractSpinBox, max_spinbox: QAbstractSpinBox) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(min_spinbox)
        layout.addWidget(max_spinbox)
        return widget

    def _configure_icon_only_button(self, button: QPushButton, icon: QIcon) -> None:
        button.setText("")
        button.setIcon(icon)
        button.setIconSize(QSize(20, 20))
        button.setFixedWidth(36)
        button.setMinimumHeight(30)

    def _refresh_files_icon(self) -> QIcon:
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor("#22C55E"), 2.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(5, 5, 14, 14, 35 * 16, 285 * 16)
        arrow = QPolygonF([QPointF(18.0, 5.0), QPointF(18.2, 11.0), QPointF(13.2, 8.0)])
        painter.setBrush(QColor("#22C55E"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(arrow)
        painter.end()
        return QIcon(pixmap)

    def _configure_compact_form(self, form: QFormLayout) -> None:
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setVerticalSpacing(2)
        form.setHorizontalSpacing(6)

    def _disable_spinbox_wheel_changes(self) -> None:
        for spinbox in self.findChildren(QAbstractSpinBox):
            spinbox.installEventFilter(self)

    def _register_spinbox(self, spinbox: QAbstractSpinBox) -> None:
        spinbox.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if isinstance(watched, QAbstractSpinBox) and event.type() == QEvent.Type.Wheel:
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def _build_color_button(self, color: str, handler) -> QPushButton:
        button = QPushButton(color)
        button.clicked.connect(handler)
        self._update_color_button(button, color)
        return button

    def _update_color_button(self, button: QPushButton, color_value: str) -> None:
        button.setText(color_value)
        button.setStyleSheet(f"background-color: {color_value}; color: #111111;")

    def _populate_pipeline_operations(self) -> None:
        selected_operation = self._selected_available_operation_name()
        self.operation_tree.clear()
        for _group_key, labels, operations in PIPELINE_OPERATION_GROUPS:
            group_item = QTreeWidgetItem([labels[0] if self._ui_language == "ru" else labels[1]])
            group_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            for operation_name in operations:
                child_item = QTreeWidgetItem([get_operation_display_name(operation_name, self._ui_language)])
                child_item.setData(0, Qt.ItemDataRole.UserRole, operation_name)
                summary, use_case = self._operation_help_entry(operation_name)
                child_item.setToolTip(
                    0,
                    f"{summary}\n\n"
                    + (("Когда использовать: " if self._ui_language == "ru" else "When to use: ") + use_case),
                )
                group_item.addChild(child_item)
            group_item.setExpanded(True)
            self.operation_tree.addTopLevelItem(group_item)
        target_operation = selected_operation or self._all_operation_names()[0]
        target_item = self._find_operation_tree_item(target_operation)
        if target_item is not None:
            self.operation_tree.setCurrentItem(target_item)
            self._update_pipeline_help_preview(target_operation)
        self._refresh_pipeline_preset_combo()

    def _built_in_pipeline_presets(self) -> dict[str, dict[str, object]]:
        return built_in_pipeline_presets(self._ui_language)

    def _refresh_pipeline_preset_combo(self) -> None:
        if not hasattr(self, "pipeline_preset_combo"):
            return
        current_name = self.pipeline_preset_combo.currentText()
        self.pipeline_preset_combo.clear()
        for name in self._built_in_pipeline_presets():
            self.pipeline_preset_combo.addItem(name, name)
        index = self.pipeline_preset_combo.findText(current_name)
        if index >= 0:
            self.pipeline_preset_combo.setCurrentIndex(index)

    def _apply_selected_pipeline_preset(self) -> None:
        if not hasattr(self, "pipeline_preset_combo"):
            return
        preset_name = str(self.pipeline_preset_combo.currentData() or self.pipeline_preset_combo.currentText() or "")
        payload = self._built_in_pipeline_presets().get(preset_name)
        if not isinstance(payload, dict):
            return
        self._pipeline = PreprocessingPipeline.from_dict(payload)
        self._populate_pipeline_list()
        self.process_current_image(debounced=True)

    def _populate_pipeline_list(self) -> None:
        self._ignore_pipeline_item_change = True
        self.pipeline_list.clear()
        for step in self._pipeline.steps:
            label = get_operation_display_name(step.operation, self._ui_language)
            item = QListWidgetItem(label)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsDragEnabled
            )
            item.setData(Qt.ItemDataRole.UserRole, self.pipeline_list.count())
            item.setData(Qt.ItemDataRole.UserRole + 1, step.operation)
            item.setCheckState(Qt.CheckState.Checked if step.enabled else Qt.CheckState.Unchecked)
            self.pipeline_list.addItem(item)
        self._ignore_pipeline_item_change = False
        if self.pipeline_list.count():
            self.pipeline_list.setCurrentRow(0)
            self._render_pipeline_parameters(0)
        else:
            self._clear_parameters_form()

    def _clear_parameters_form(self) -> None:
        while self.parameters_form.count():
            item = self.parameters_form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._parameter_widgets.clear()

    def _on_pipeline_step_selected(self, row: int) -> None:
        self._render_pipeline_parameters(row)

    def _render_pipeline_parameters(self, row: int) -> None:
        self._clear_parameters_form()
        if row < 0 or row >= len(self._pipeline.steps):
            self._set_color_pick_active(None)
            return
        step = self._pipeline.steps[row]
        descriptor = get_operation_descriptor(step.operation)
        for spec in descriptor.parameters:
            value = step.parameters.get(spec.name, spec.default)
            if spec.kind == "bool":
                widget = QCheckBox()
                widget.setChecked(bool(value))
                widget.stateChanged.connect(
                    lambda _state, name=spec.name, row_index=row, w=widget: self._update_step_parameter(
                        row_index, name, w.isChecked()
                    )
                )
            elif spec.kind == "choice":
                widget = QComboBox()
                for option in spec.options:
                    widget.addItem(get_choice_display_label(spec.name, str(option), self._ui_language), option)
                selected_index = widget.findData(value)
                if selected_index >= 0:
                    widget.setCurrentIndex(selected_index)
                widget.currentIndexChanged.connect(
                    lambda _index, name=spec.name, row_index=row, w=widget: self._update_step_parameter(
                        row_index,
                        name,
                        w.currentData(),
                    )
                )
            elif spec.kind == "int":
                widget = QSpinBox()
                self._register_spinbox(widget)
                widget.setRange(int(spec.minimum or -1_000_000), int(spec.maximum or 1_000_000))
                widget.setSingleStep(int(spec.step or 1))
                widget.setValue(int(value))
                widget.valueChanged.connect(
                    lambda new_value, name=spec.name, row_index=row: self._update_step_parameter(
                        row_index, name, int(new_value)
                    )
                )
            else:
                widget = QDoubleSpinBox()
                self._register_spinbox(widget)
                widget.setDecimals(spec.decimals)
                widget.setRange(float(spec.minimum or -1_000_000), float(spec.maximum or 1_000_000))
                widget.setSingleStep(float(spec.step or 0.1))
                widget.setValue(float(value))
                widget.valueChanged.connect(
                    lambda new_value, name=spec.name, row_index=row: self._update_step_parameter(
                        row_index, name, float(new_value)
                    )
                )
            tooltip = spec.tooltip or self._pipeline_parameter_tooltip(step.operation, spec.name)
            widget.setToolTip(tooltip)
            self._parameter_widgets[spec.name] = widget
            label_widget = QLabel(get_parameter_display_label(spec, self._ui_language))
            label_widget.setToolTip(tooltip)
            self.parameters_form.addRow(label_widget, widget)
        if step.operation == "color_binarize":
            self._render_color_binarize_parameters(row)
        else:
            self._set_color_pick_active(None)

    def _update_step_parameter(self, row: int, parameter_name: str, value) -> None:
        if row < 0 or row >= len(self._pipeline.steps):
            return
        self._pipeline.steps[row].parameters[parameter_name] = value
        self._auto_apply_pipeline()

    def _color_selection_entries(self, row: int) -> list[dict[str, object]]:
        if row < 0 or row >= len(self._pipeline.steps):
            return []
        entries = self._pipeline.steps[row].parameters.get("selected_colors", [])
        if not isinstance(entries, list):
            entries = []
        normalized: list[dict[str, object]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rgb = entry.get("rgb")
            if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
                continue
            try:
                parsed_rgb = [max(0, min(255, int(channel))) for channel in rgb]
            except (TypeError, ValueError):
                continue
            normalized.append({"rgb": parsed_rgb, "enabled": bool(entry.get("enabled", True))})
        self._pipeline.steps[row].parameters["selected_colors"] = normalized
        return normalized

    def _render_color_binarize_parameters(self, row: int) -> None:
        entries = self._color_selection_entries(row)
        group = QGroupBox(
            self._tr(
                "color_binarize_group_title",
                "Цвета для бинаризации" if self._ui_language == "ru" else "Colors for binarization",
            )
        )
        layout = QVBoxLayout(group)
        hint = QLabel(
            self._tr(
                "color_binarize_hint",
                "Включите выбор и кликните по изображению, чтобы добавить цвет. Галочкой можно временно отключить цвет."
                if self._ui_language == "ru"
                else "Enable picking and click the image to add a color. Uncheck an item to disable it temporarily.",
            )
        )
        hint.setWordWrap(True)
        hint.setToolTip(
            "Цвета из списка используются для построения бинарной маски; допуск задается параметром delta."
            if self._ui_language == "ru"
            else "Colors in the list are used to build the binary mask; tolerance is controlled by delta."
        )
        layout.addWidget(hint)
        color_list = QListWidget()
        color_list.setToolTip(
            "Отмеченные цвета участвуют в бинаризации. Снимите галочку, чтобы временно исключить цвет из маски."
            if self._ui_language == "ru"
            else "Checked colors participate in binarization. Uncheck a color to temporarily exclude it from the mask."
        )
        color_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for entry in entries:
            rgb = entry["rgb"]
            item = QListWidgetItem(f"#{int(rgb[0]):02X}{int(rgb[1]):02X}{int(rgb[2]):02X}")
            item.setToolTip(
                "Этот цвет добавляет похожие пиксели в маску; галочка включает или выключает его."
                if self._ui_language == "ru"
                else "This color adds similar pixels to the mask; the checkbox enables or disables it."
            )
            item.setData(Qt.ItemDataRole.UserRole, list(rgb))
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(Qt.CheckState.Checked if entry.get("enabled", True) else Qt.CheckState.Unchecked)
            item.setBackground(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
            brightness = int(rgb[0]) * 0.299 + int(rgb[1]) * 0.587 + int(rgb[2]) * 0.114
            item.setForeground(QColor("#111111" if brightness > 150 else "#F8FAFC"))
            color_list.addItem(item)
        color_list.itemChanged.connect(
            lambda item, row_index=row, widget=color_list: self._on_color_entry_changed(row_index, widget, item)
        )
        layout.addWidget(color_list)

        buttons_row = QWidget()
        buttons_layout = QHBoxLayout(buttons_row)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        pick_button = QPushButton(
            self._tr("pick_colors_button", "Выбор с изображения" if self._ui_language == "ru" else "Pick from image")
        )
        pick_button.setCheckable(True)
        pick_button.setToolTip(
            "Включает выбор цвета с изображения: кликните по нужному пикселю, чтобы добавить его в список."
            if self._ui_language == "ru"
            else "Enables picking from the image: click a pixel to add its color to the list."
        )
        pick_button.setChecked(self._color_pick_pipeline_row == row)
        pick_button.toggled.connect(
            lambda checked, row_index=row: self._set_color_pick_active(row_index if checked else None)
        )
        remove_button = QPushButton(
            self._tr(
                "remove_selected_color_button", "Удалить выбранный" if self._ui_language == "ru" else "Remove selected"
            )
        )
        remove_button.setToolTip(
            "Удаляет выбранный цвет из списка бинаризации."
            if self._ui_language == "ru"
            else "Removes the selected color from the binarization list."
        )
        remove_button.clicked.connect(
            lambda _checked=False, row_index=row, widget=color_list: self._remove_selected_color_entry(
                row_index, widget
            )
        )
        clear_button = QPushButton(
            self._tr("clear_colors_button", "Очистить список" if self._ui_language == "ru" else "Clear list")
        )
        clear_button.setToolTip(
            "Очищает весь список цветов для этого шага бинаризации."
            if self._ui_language == "ru"
            else "Clears the whole color list for this binarization step."
        )
        clear_button.clicked.connect(lambda _checked=False, row_index=row: self._clear_color_entries(row_index))
        buttons_layout.addWidget(pick_button)
        buttons_layout.addWidget(remove_button)
        buttons_layout.addWidget(clear_button)
        layout.addWidget(buttons_row)
        self.parameters_form.addRow(group)

    def _on_color_entry_changed(self, row: int, color_list: QListWidget, item: QListWidgetItem) -> None:
        entries = self._color_selection_entries(row)
        index = color_list.row(item)
        if index < 0 or index >= len(entries):
            return
        entries[index]["enabled"] = item.checkState() == Qt.CheckState.Checked
        self._pipeline.steps[row].parameters["selected_colors"] = entries
        self._auto_apply_pipeline()

    def _remove_selected_color_entry(self, row: int, color_list: QListWidget) -> None:
        index = color_list.currentRow()
        if index < 0:
            return
        entries = self._color_selection_entries(row)
        if index >= len(entries):
            return
        entries.pop(index)
        self._pipeline.steps[row].parameters["selected_colors"] = entries
        self._render_pipeline_parameters(row)
        self._auto_apply_pipeline()

    def _clear_color_entries(self, row: int) -> None:
        if row < 0 or row >= len(self._pipeline.steps):
            return
        self._pipeline.steps[row].parameters["selected_colors"] = []
        self._render_pipeline_parameters(row)
        self._auto_apply_pipeline()

    def _set_color_pick_active(self, row: int | None) -> None:
        self._color_pick_pipeline_row = row
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_image_click_mode(row is not None)

    def _set_via_template_pick_active(self, enabled: bool) -> None:
        if enabled:
            self._set_color_pick_active(None)
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_image_region_selection_mode(enabled)

    def _refresh_via_template_list(self) -> None:
        if not hasattr(self, "via_template_list"):
            return
        self.via_template_list.clear()
        for index, template in enumerate(self._via_template_images, start=1):
            height, width = template.shape[:2]
            item = QListWidgetItem(f"{index}: {width}x{height}")
            preview_pixmap = QPixmap.fromImage(cv_to_qimage(template)).scaled(
                56,
                56,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            item.setIcon(QIcon(preview_pixmap))
            item.setToolTip(
                f"Шаблон via #{index}: {width}x{height} пикс."
                if self._ui_language == "ru"
                else f"Via template #{index}: {width}x{height} px"
            )
            self.via_template_list.addItem(item)

    def _normalize_via_template_images(self, payload: list[object]) -> list[np.ndarray]:
        templates: list[np.ndarray] = []
        for item in payload:
            try:
                image = np.asarray(item, dtype=np.uint8)
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
        if hasattr(self, "add_via_template_button"):
            self.add_via_template_button.setChecked(False)
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
        self._via_template_images.append(template.astype(np.uint8, copy=False))
        self._refresh_via_template_list()
        self._on_extraction_settings_changed()
        self._append_log(
            self._tr(
                "via_template_added_log",
                "Добавлен шаблон via {width}x{height}. Всего шаблонов: {count}."
                if self._ui_language == "ru"
                else "Added via template {width}x{height}. Total templates: {count}.",
                width=right - left,
                height=bottom - top,
                count=len(self._via_template_images),
            )
        )

    def _clear_via_templates(self, *_args) -> None:
        self._via_template_images.clear()
        self._refresh_via_template_list()
        self._on_extraction_settings_changed()

    def _remove_selected_via_template(self, *_args) -> None:
        row = self.via_template_list.currentRow() if hasattr(self, "via_template_list") else -1
        if row < 0 or row >= len(self._via_template_images):
            return
        self._via_template_images.pop(row)
        self._refresh_via_template_list()
        if self._via_template_images:
            self.via_template_list.setCurrentRow(min(row, len(self._via_template_images) - 1))
        self._on_extraction_settings_changed()

    def _built_in_via_presets(self) -> dict[str, dict[str, object]]:
        return built_in_via_presets(self._ui_language)

    def _noisy_traces_via_preset_payload(self) -> dict[str, object]:
        return noisy_traces_via_preset_payload()

    def _blurred_via_preset_payload(self) -> dict[str, object]:
        return blurred_via_preset_payload()

    def _load_user_via_presets(self) -> dict[str, dict[str, object]]:
        settings = QSettings("ViaLaNet", "Contour")
        raw_payload = settings.value(VIA_PRESETS_SETTINGS_KEY, "{}", type=str)
        try:
            payload = json.loads(str(raw_payload or "{}"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(name): dict(value) for name, value in payload.items() if isinstance(value, dict)}

    def _save_user_via_presets(self) -> None:
        settings = QSettings("ViaLaNet", "Contour")
        settings.setValue(
            VIA_PRESETS_SETTINGS_KEY, json.dumps(self._user_via_presets, ensure_ascii=False, sort_keys=True)
        )
        settings.sync()

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
        excluded_keys = {
            "via_template_images",
            "fixed_via_widths",
            "fixed_via_heights",
            "min_via_width",
            "max_via_width",
            "min_via_height",
            "max_via_height",
            "via_size_mode",
        }
        return {
            key: value
            for key, value in payload.items()
            if (key.startswith("via_") or key.startswith("bright_via_")) and key not in excluded_keys
        } | {
            "debug_enabled": self.debug_candidates_checkbox.isChecked()
        }

    def _apply_via_preset_payload(self, payload: dict[str, object]) -> None:
        blockers = [
            QSignalBlocker(self.via_search_mode_combo),
            QSignalBlocker(self.via_white_range_checkbox),
            QSignalBlocker(self.via_white_range_min_spin),
            QSignalBlocker(self.via_white_range_max_spin),
            QSignalBlocker(self.via_black_range_checkbox),
            QSignalBlocker(self.via_black_range_min_spin),
            QSignalBlocker(self.via_black_range_max_spin),
            QSignalBlocker(self.via_min_score_spin),
            QSignalBlocker(self.via_min_contrast_spin),
            QSignalBlocker(self.via_min_edge_coverage_spin),
            QSignalBlocker(self.via_spot_line_suppression_spin),
            QSignalBlocker(self.via_template_min_score_spin),
            QSignalBlocker(self.bright_via_diameter_min_spin),
            QSignalBlocker(self.bright_via_diameter_max_spin),
            QSignalBlocker(self.bright_via_clahe_clip_spin),
            QSignalBlocker(self.bright_via_clahe_tile_spin),
            QSignalBlocker(self.bright_via_median_kernel_spin),
            QSignalBlocker(self.bright_via_tophat_kernel_spin),
            QSignalBlocker(self.bright_via_dog_small_spin),
            QSignalBlocker(self.bright_via_dog_large_spin),
            QSignalBlocker(self.bright_via_threshold_percentile_spin),
            QSignalBlocker(self.bright_via_mask_combine_combo),
            QSignalBlocker(self.bright_via_min_area_factor_spin),
            QSignalBlocker(self.bright_via_max_area_factor_spin),
            QSignalBlocker(self.bright_via_min_circularity_spin),
            QSignalBlocker(self.bright_via_min_aspect_spin),
            QSignalBlocker(self.bright_via_max_aspect_spin),
            QSignalBlocker(self.bright_via_bright_center_score_spin),
            QSignalBlocker(self.bright_via_metal_constraint_combo),
            QSignalBlocker(self.bright_via_metal_fraction_spin),
            QSignalBlocker(self.bright_via_max_radial_asymmetry_spin),
            QSignalBlocker(self.bright_via_max_edge_likeness_spin),
            QSignalBlocker(self.bright_via_max_line_likeness_spin),
            QSignalBlocker(self.bright_via_nms_distance_spin),
            QSignalBlocker(self.bright_via_min_final_score_spin),
            QSignalBlocker(self.bright_via_show_rejected_checkbox),
            QSignalBlocker(self.bright_via_hard_asym_checkbox),
            QSignalBlocker(self.bright_via_hard_edge_checkbox),
            QSignalBlocker(self.bright_via_hard_line_checkbox),
            QSignalBlocker(self.debug_candidates_checkbox),
            QSignalBlocker(self.via_roundness_spin),
        ]
        try:
            mode_index = self.via_search_mode_combo.findData(
                normalize_via_search_mode(payload.get("via_search_mode", self.via_search_mode_combo.currentData()))
            )
            if mode_index >= 0:
                self.via_search_mode_combo.setCurrentIndex(mode_index)
            self.via_white_range_checkbox.setChecked(
                bool(payload.get("via_white_range_enabled", self.via_white_range_checkbox.isChecked()))
            )
            self.via_white_range_min_spin.setValue(
                int(payload.get("via_white_range_min", self.via_white_range_min_spin.value()))
            )
            self.via_white_range_max_spin.setValue(
                int(payload.get("via_white_range_max", self.via_white_range_max_spin.value()))
            )
            self.via_black_range_checkbox.setChecked(
                bool(payload.get("via_black_range_enabled", self.via_black_range_checkbox.isChecked()))
            )
            self.via_black_range_min_spin.setValue(
                int(payload.get("via_black_range_min", self.via_black_range_min_spin.value()))
            )
            self.via_black_range_max_spin.setValue(
                int(payload.get("via_black_range_max", self.via_black_range_max_spin.value()))
            )
            self.via_min_score_spin.setValue(float(payload.get("via_min_score", self.via_min_score_spin.value())))
            self.via_min_contrast_spin.setValue(
                float(payload.get("via_min_contrast", self.via_min_contrast_spin.value()))
            )
            self.via_min_edge_coverage_spin.setValue(
                float(payload.get("via_min_edge_coverage", self.via_min_edge_coverage_spin.value()))
            )
            self.via_spot_line_suppression_spin.setValue(
                float(payload.get("via_spot_line_suppression", self.via_spot_line_suppression_spin.value()))
            )
            self.via_template_min_score_spin.setValue(
                float(payload.get("via_template_min_score", self.via_template_min_score_spin.value()))
            )
            self.via_roundness_spin.setValue(float(payload.get("via_min_roundness", self.via_roundness_spin.value())))
            self.bright_via_diameter_min_spin.setValue(
                int(payload.get("bright_via_diameter_min", self.bright_via_diameter_min_spin.value()))
            )
            self.bright_via_diameter_max_spin.setValue(
                int(payload.get("bright_via_diameter_max", self.bright_via_diameter_max_spin.value()))
            )
            self.bright_via_clahe_clip_spin.setValue(
                float(payload.get("bright_via_clahe_clip_limit", self.bright_via_clahe_clip_spin.value()))
            )
            self.bright_via_clahe_tile_spin.setValue(
                int(payload.get("bright_via_clahe_tile_grid_size", self.bright_via_clahe_tile_spin.value()))
            )
            self.bright_via_median_kernel_spin.setValue(
                int(payload.get("bright_via_median_blur_kernel", self.bright_via_median_kernel_spin.value()))
            )
            self.bright_via_tophat_kernel_spin.setValue(
                int(payload.get("bright_via_tophat_kernel_size", self.bright_via_tophat_kernel_spin.value()))
            )
            self.bright_via_dog_small_spin.setValue(
                float(payload.get("bright_via_dog_sigma_small", self.bright_via_dog_small_spin.value()))
            )
            self.bright_via_dog_large_spin.setValue(
                float(payload.get("bright_via_dog_sigma_large", self.bright_via_dog_large_spin.value()))
            )
            self.bright_via_threshold_percentile_spin.setValue(
                float(
                    payload.get(
                        "bright_via_threshold_percentile", self.bright_via_threshold_percentile_spin.value()
                    )
                )
            )
            combine_index = self.bright_via_mask_combine_combo.findData(
                str(payload.get("bright_via_mask_combine_mode", self.bright_via_mask_combine_combo.currentData()))
            )
            if combine_index >= 0:
                self.bright_via_mask_combine_combo.setCurrentIndex(combine_index)
            self.bright_via_min_area_factor_spin.setValue(
                float(payload.get("bright_via_min_area_factor", self.bright_via_min_area_factor_spin.value()))
            )
            self.bright_via_max_area_factor_spin.setValue(
                float(payload.get("bright_via_max_area_factor", self.bright_via_max_area_factor_spin.value()))
            )
            self.bright_via_min_circularity_spin.setValue(
                float(payload.get("bright_via_min_circularity", self.bright_via_min_circularity_spin.value()))
            )
            self.bright_via_min_aspect_spin.setValue(
                float(payload.get("bright_via_min_aspect", self.bright_via_min_aspect_spin.value()))
            )
            self.bright_via_max_aspect_spin.setValue(
                float(payload.get("bright_via_max_aspect", self.bright_via_max_aspect_spin.value()))
            )
            self.bright_via_bright_center_score_spin.setValue(
                float(
                    payload.get(
                        "bright_via_bright_center_min_score",
                        self.bright_via_bright_center_score_spin.value(),
                    )
                )
            )
            metal_mode = _normalize_bright_via_metal_constraint_mode(
                payload.get("bright_via_metal_constraint_mode", self.bright_via_metal_constraint_combo.currentData())
            )
            metal_index = self.bright_via_metal_constraint_combo.findData(metal_mode)
            if metal_index >= 0:
                self.bright_via_metal_constraint_combo.setCurrentIndex(metal_index)
            self.bright_via_metal_fraction_spin.setValue(
                float(payload.get("bright_via_metal_fraction_min", self.bright_via_metal_fraction_spin.value()))
            )
            self.bright_via_max_radial_asymmetry_spin.setValue(
                float(
                    payload.get(
                        "bright_via_max_radial_asymmetry",
                        self.bright_via_max_radial_asymmetry_spin.value(),
                    )
                )
            )
            self.bright_via_max_edge_likeness_spin.setValue(
                float(payload.get("bright_via_max_edge_likeness", self.bright_via_max_edge_likeness_spin.value()))
            )
            self.bright_via_max_line_likeness_spin.setValue(
                float(payload.get("bright_via_max_line_likeness", self.bright_via_max_line_likeness_spin.value()))
            )
            self.bright_via_nms_distance_spin.setValue(
                int(payload.get("bright_via_nms_distance", self.bright_via_nms_distance_spin.value()))
            )
            self.bright_via_min_final_score_spin.setValue(
                float(payload.get("bright_via_min_final_score", self.bright_via_min_final_score_spin.value()))
            )
            self.bright_via_show_rejected_checkbox.setChecked(
                bool(payload.get("bright_via_show_rejected", self.bright_via_show_rejected_checkbox.isChecked()))
            )
            self.bright_via_hard_asym_checkbox.setChecked(
                bool(
                    payload.get(
                        "bright_via_hard_reject_on_asymmetry", self.bright_via_hard_asym_checkbox.isChecked()
                    )
                )
            )
            self.bright_via_hard_edge_checkbox.setChecked(
                bool(payload.get("bright_via_hard_reject_on_edge", self.bright_via_hard_edge_checkbox.isChecked()))
            )
            self.bright_via_hard_line_checkbox.setChecked(
                bool(payload.get("bright_via_hard_reject_on_line", self.bright_via_hard_line_checkbox.isChecked()))
            )
            self.debug_candidates_checkbox.setChecked(
                bool(payload.get("debug_enabled", self.debug_candidates_checkbox.isChecked()))
            )
        finally:
            del blockers
        self._update_via_threshold_controls_state()
        self._on_extraction_settings_changed()

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
        self._user_via_presets[name] = self._current_via_preset_payload()
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

    def _apply_noisy_traces_via_preset(self, *_args) -> None:
        self._apply_via_preset_payload(self._noisy_traces_via_preset_payload())

    def _apply_blurred_via_preset(self, *_args) -> None:
        self._apply_via_preset_payload(self._blurred_via_preset_payload())

    def _reset_via_search_parameters(self, *_args) -> None:
        blockers = [
            QSignalBlocker(self.via_search_mode_combo),
            QSignalBlocker(self.via_min_score_spin),
            QSignalBlocker(self.via_min_contrast_spin),
            QSignalBlocker(self.via_min_edge_coverage_spin),
            QSignalBlocker(self.via_spot_line_suppression_spin),
            QSignalBlocker(self.via_template_min_score_spin),
            QSignalBlocker(self.via_roundness_spin),
        ]
        try:
            mode_index = self.via_search_mode_combo.findData("template")
            if mode_index >= 0:
                self.via_search_mode_combo.setCurrentIndex(mode_index)
            self.via_min_score_spin.setValue(0.35)
            self.via_min_contrast_spin.setValue(14.0)
            self.via_min_edge_coverage_spin.setValue(0.45)
            self.via_spot_line_suppression_spin.setValue(0.65)
            self.via_template_min_score_spin.setValue(0.35)
            self.via_roundness_spin.setValue(40.0)
        finally:
            del blockers
        self._update_via_threshold_controls_state()
        self._on_extraction_settings_changed()

    def _select_bright_via_mode(self) -> None:
        ridx = self.recognition_mode_combo.findData("via")
        if ridx >= 0 and self.recognition_mode_combo.currentIndex() != ridx:
            self.recognition_mode_combo.setCurrentIndex(ridx)
        mode_index = self.via_search_mode_combo.findData(VIA_SEARCH_MODE_HEURISTIC)
        if mode_index >= 0 and self.via_search_mode_combo.currentIndex() != mode_index:
            self.via_search_mode_combo.setCurrentIndex(mode_index)

    def _preview_bright_via_mask(self, *_args) -> None:
        self._select_bright_via_mode()
        self.debug_candidates_checkbox.setChecked(True)
        self._show_gradient_debug_window()

    def _reset_bright_via_parameters(self, *_args) -> None:
        blockers = [
            QSignalBlocker(self.bright_via_diameter_min_spin),
            QSignalBlocker(self.bright_via_diameter_max_spin),
            QSignalBlocker(self.bright_via_clahe_clip_spin),
            QSignalBlocker(self.bright_via_clahe_tile_spin),
            QSignalBlocker(self.bright_via_median_kernel_spin),
            QSignalBlocker(self.bright_via_tophat_kernel_spin),
            QSignalBlocker(self.bright_via_dog_small_spin),
            QSignalBlocker(self.bright_via_dog_large_spin),
            QSignalBlocker(self.bright_via_threshold_percentile_spin),
            QSignalBlocker(self.bright_via_mask_combine_combo),
            QSignalBlocker(self.bright_via_min_area_factor_spin),
            QSignalBlocker(self.bright_via_max_area_factor_spin),
            QSignalBlocker(self.bright_via_min_circularity_spin),
            QSignalBlocker(self.bright_via_min_aspect_spin),
            QSignalBlocker(self.bright_via_max_aspect_spin),
            QSignalBlocker(self.bright_via_bright_center_score_spin),
            QSignalBlocker(self.bright_via_metal_constraint_combo),
            QSignalBlocker(self.bright_via_metal_fraction_spin),
            QSignalBlocker(self.bright_via_max_radial_asymmetry_spin),
            QSignalBlocker(self.bright_via_max_edge_likeness_spin),
            QSignalBlocker(self.bright_via_max_line_likeness_spin),
            QSignalBlocker(self.bright_via_nms_distance_spin),
            QSignalBlocker(self.bright_via_min_final_score_spin),
            QSignalBlocker(self.bright_via_show_rejected_checkbox),
            QSignalBlocker(self.bright_via_hard_asym_checkbox),
            QSignalBlocker(self.bright_via_hard_edge_checkbox),
            QSignalBlocker(self.bright_via_hard_line_checkbox),
        ]
        try:
            self.bright_via_diameter_min_spin.setValue(6)
            self.bright_via_diameter_max_spin.setValue(8)
            self.bright_via_clahe_clip_spin.setValue(2.0)
            self.bright_via_clahe_tile_spin.setValue(8)
            self.bright_via_median_kernel_spin.setValue(3)
            self.bright_via_tophat_kernel_spin.setValue(11)
            self.bright_via_dog_small_spin.setValue(0.8)
            self.bright_via_dog_large_spin.setValue(2.0)
            self.bright_via_threshold_percentile_spin.setValue(99.0)
            combine_index = self.bright_via_mask_combine_combo.findData("OR")
            if combine_index >= 0:
                self.bright_via_mask_combine_combo.setCurrentIndex(combine_index)
            self.bright_via_min_area_factor_spin.setValue(0.45)
            self.bright_via_max_area_factor_spin.setValue(1.8)
            self.bright_via_min_circularity_spin.setValue(0.30)
            self.bright_via_min_aspect_spin.setValue(0.45)
            self.bright_via_max_aspect_spin.setValue(2.2)
            self.bright_via_bright_center_score_spin.setValue(6.0)
            metal_index = self.bright_via_metal_constraint_combo.findData("soft")
            if metal_index >= 0:
                self.bright_via_metal_constraint_combo.setCurrentIndex(metal_index)
            self.bright_via_metal_fraction_spin.setValue(0.3)
            self.bright_via_max_radial_asymmetry_spin.setValue(18.0)
            self.bright_via_max_edge_likeness_spin.setValue(35.0)
            self.bright_via_max_line_likeness_spin.setValue(65.0)
            self.bright_via_nms_distance_spin.setValue(5)
            self.bright_via_min_final_score_spin.setValue(38.0)
            self.bright_via_show_rejected_checkbox.setChecked(True)
            self.bright_via_hard_asym_checkbox.setChecked(False)
            self.bright_via_hard_edge_checkbox.setChecked(False)
            self.bright_via_hard_line_checkbox.setChecked(False)
        finally:
            del blockers
        self._on_extraction_settings_changed()

    def _add_color_selection(self, row: int, rgb: tuple[int, int, int]) -> None:
        entries = self._color_selection_entries(row)
        for entry in entries:
            if tuple(entry["rgb"]) == tuple(rgb):
                entry["enabled"] = True
                self._pipeline.steps[row].parameters["selected_colors"] = entries
                self._render_pipeline_parameters(row)
                self._auto_apply_pipeline()
                return
        entries.append({"rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])], "enabled": True})
        self._pipeline.steps[row].parameters["selected_colors"] = entries
        self._render_pipeline_parameters(row)
        self._auto_apply_pipeline()

    def _on_editor_image_clicked(self, x_coord: float, y_coord: float) -> None:
        row = self._color_pick_pipeline_row
        if row is None or row < 0 or row >= len(self._pipeline.steps):
            return
        current_state = self._workspace.current_state
        if current_state is None or current_state.source_image is None:
            return
        image = np.asarray(current_state.source_image)
        x_index = round(x_coord)
        y_index = round(y_coord)
        if y_index < 0 or x_index < 0 or y_index >= image.shape[0] or x_index >= image.shape[1]:
            return
        if image.ndim == 2:
            value = int(image[y_index, x_index])
            rgb = (value, value, value)
        else:
            pixel = image[y_index, x_index]
            if image.shape[2] >= 3:
                rgb = (int(pixel[2]), int(pixel[1]), int(pixel[0]))
            else:
                value = int(pixel[0])
                rgb = (value, value, value)
        self._add_color_selection(row, rgb)
        self._append_log(
            self._tr(
                "color_picked_log",
                "Добавлен цвет {color}" if self._ui_language == "ru" else "Added color {color}",
                color=f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
            )
        )

    def _on_via_debug_requested(self, polygon: PolygonData) -> None:
        current_state = self._workspace.current_state
        candidates = list(current_state.debug_candidates) if current_state is not None else []
        is_via_like = (polygon.shape_hint or "") == "box" or (polygon.category or "") == "via"
        title = (
            ("Отладка via" if self._ui_language == "ru" else "Via debug")
            if is_via_like
            else ("Отладка полигона" if self._ui_language == "ru" else "Polygon debug")
        )
        if not candidates:
            message = (
                "Для текущего кадра нет отладочных данных. Включите 'Проверять по клику' и дождитесь повторной обработки."
                if self._ui_language == "ru"
                else "There is no debug data for the current frame. Enable 'Inspect by click' and wait for processing to finish."
            )
            QMessageBox.information(self, title, message)
            return
        candidate = self._best_debug_candidate_for_polygon(polygon, candidates)
        if candidate is None:
            message = (
                "Для этой via не найден исходный кандидат распознавания. Вероятно, объект был создан или изменен вручную после обработки."
                if self._ui_language == "ru"
                else "No source recognition candidate was found for this via. The object was likely created or edited manually after processing."
            )
            QMessageBox.information(self, title, message)
            return
        source = self._debug_candidate_source(candidate)
        reason = str(getattr(candidate, "reason", "") or "")
        accepted = bool(getattr(candidate, "accepted", False))
        bbox = getattr(candidate, "bbox", (0, 0, 0, 0))
        status = (
            "принята"
            if accepted and self._ui_language == "ru"
            else "accepted"
            if accepted
            else "отклонена"
            if self._ui_language == "ru"
            else "rejected"
        )
        lines = [
            f"{'Статус' if self._ui_language == 'ru' else 'Status'}: {status}",
            f"{'Метод' if self._ui_language == 'ru' else 'Method'}: {self._debug_method_text(source)}",
            f"{'Критерий' if self._ui_language == 'ru' else 'Criterion'}: {self._debug_criterion_text(source, reason, accepted)}",
            f"{'Причина' if self._ui_language == 'ru' else 'Reason'}: {reason or '-'}",
        ]
        if is_via_like:
            lines += [
                f"{'Оценка' if self._ui_language == 'ru' else 'Score'}: {float(getattr(candidate, 'score', 0.0)):.1f}",
                f"{'Округлость' if self._ui_language == 'ru' else 'Roundness'}: {float(getattr(candidate, 'roundness', 0.0)):.1f}",
            ]
        else:
            area_v = float(getattr(candidate, "area", 0.0) or 0.0)
            per_v = float(getattr(candidate, "perimeter", 0.0) or 0.0)
            ew = float(getattr(candidate, "effective_width", 0.0) or 0.0)
            wm = str(getattr(candidate, "width_metric", "") or "")
            wline = f"{'Оценка ширины' if self._ui_language == 'ru' else 'Width estimate'}: {ew:.2f} px"
            if wm:
                wline += f" ({wm})"
            lines += [
                f"{'Площадь' if self._ui_language == 'ru' else 'Area'}: {area_v:.1f} px²",
                f"{'Периметр' if self._ui_language == 'ru' else 'Perimeter'}: {per_v:.1f} px",
                wline,
            ]
        lines += [
            f"{'Размер кандидата' if self._ui_language == 'ru' else 'Candidate size'}: {int(bbox[2])} x {int(bbox[3])} px",
            f"{'Позиция' if self._ui_language == 'ru' else 'Position'}: x={int(bbox[0])}, y={int(bbox[1])}",
        ]
        message = "\n".join(lines)
        self._append_log(message.replace("\n", " | "))
        QMessageBox.information(self, title, message)

    def _on_metal_overlay_detail_requested(self, layer_key: str, reason: str) -> None:
        ru = self._ui_language == "ru"
        titles = {
            "rejected": "Отклонённый проводник" if ru else "Rejected conductor",
            "suspicious": "Сомнительный проводник" if ru else "Suspicious conductor",
            "border": "Проводник у границы кадра" if ru else "Border-touching conductor",
            "wide_pairs_suspicious": "Широкий проводник (сомнительно)" if ru else "Wide trace (suspicious)",
            "wide_pairs_rejected": "Широкий проводник (отклонён)" if ru else "Wide trace (rejected)",
        }
        title = titles.get(layer_key, "Металлизация" if ru else "Metal recovery")
        r = (reason or "").strip()
        if not r:
            body = (
                "Для этого слоя подробная причина не сохранена."
                if ru
                else "No detailed reason was stored for this overlay."
            )
        else:
            body = f"{'Причина' if ru else 'Reason'}:\n{r}"
        self._append_log(f"{title}: {r or body}")
        QMessageBox.information(self, title, body)

    def _on_middle_preview_hold_changed(self, active: bool) -> None:
        should_show_source = bool(active and self._is_filters_tab_active())
        if self._show_source_while_middle_held == should_show_source:
            return
        self._show_source_while_middle_held = should_show_source
        self._sync_current_state_views()

    def _is_filters_tab_active(self) -> bool:
        if not hasattr(self, "control_tabs") or not hasattr(self, "pipeline_tab"):
            return False
        return self.control_tabs.currentWidget() is self.pipeline_tab

    def _on_control_tab_changed(self, _index: int) -> None:
        if not self._show_source_while_middle_held:
            return
        if self._is_filters_tab_active():
            return
        self._show_source_while_middle_held = False
        self._sync_current_state_views()

    def _show_gradient_debug_window(self) -> None:
        title = "Отладка градиентов" if self._ui_language == "ru" else "Gradient debug"
        current_state = self._workspace.current_state
        maps: dict[str, object] = {}
        if current_state is not None:
            maps = dict(getattr(current_state, "debug_gradient_maps", {}) or {})
        if not maps:
            try:
                maps = self._compute_gradient_debug_maps_on_demand()
            except Exception as exc:  # pragma: no cover - defensive UI path
                QMessageBox.warning(
                    self,
                    title,
                    (
                        f"Не удалось построить карту градиентов: {exc}"
                        if self._ui_language == "ru"
                        else f"Could not build the gradient map: {exc}"
                    ),
                )
                return
        if not maps:
            message = (
                "Нет карт градиентов. Включите 'Проверять по клику' и подождите пересчёта."
                if self._ui_language == "ru"
                else "No gradient maps available. Enable 'Inspect by click' and wait for reprocessing."
            )
            QMessageBox.information(self, title, message)
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(1100, 780)
        layout = QVBoxLayout(dialog)
        tabs = QTabWidget(dialog)
        layout.addWidget(tabs, 1)
        ordering = [
            "source_gray",
            "gradient_elevation",
            "gradient_color",
            "scharr",
            "phase_congruency",
            "structured",
            "ridge",
            "conductor_gradient_elevation",
            "spot_response",
            "spot_response_dark",
            "raw_gray",
            "processed",
            "tophat",
            "dog",
            "tophat_mask",
            "dog_mask",
            "via_mask",
            "candidate_mask",
            "metal_mask",
            "radial_symmetry",
            "edge_likeness",
            "line_likeness",
            "distance_to_edge",
            "final_overlay",
            "mask",
        ]
        pretty_names = {
            "source_gray": "Source gray" if self._ui_language != "ru" else "Исходное (серое)",
            "gradient_elevation": "Gradient elevation" if self._ui_language != "ru" else "Карта градиента",
            "gradient_color": "Gradient heatmap" if self._ui_language != "ru" else "Тепловая карта",
            "scharr": "Scharr",
            "phase_congruency": "Phase congruency",
            "structured": "Structured edges" if self._ui_language != "ru" else "Структурные границы",
            "ridge": "Ridge response" if self._ui_language != "ru" else "Хребтовая реакция",
            "conductor_gradient_elevation": (
                "Conductor gradient" if self._ui_language != "ru" else "Градиент проводников"
            ),
            "spot_response": "Spot response (bright)" if self._ui_language != "ru" else "Отклик (светлые)",
            "spot_response_dark": "Spot response (dark)" if self._ui_language != "ru" else "Отклик (тёмные)",
            "raw_gray": "Normalized input" if self._ui_language != "ru" else "Исходное (нормализовано 0–255)",
            "processed": "Processed" if self._ui_language != "ru" else "После предобработки",
            "tophat": "Top-hat response" if self._ui_language != "ru" else "Top-hat отклик",
            "dog": "DoG response" if self._ui_language != "ru" else "DoG отклик",
            "tophat_mask": "Top-hat mask" if self._ui_language != "ru" else "Top-hat маска",
            "dog_mask": "DoG mask" if self._ui_language != "ru" else "DoG маска",
            "via_mask": "Combined mask (OR/AND)" if self._ui_language != "ru" else "Маска OR/AND (порог)",
            "candidate_mask": "Candidate union" if self._ui_language != "ru" else "Маска кандидатов (объединение)",
            "metal_mask": "Metal mask" if self._ui_language != "ru" else "Маска металла",
            "radial_symmetry": "Radial asymmetry" if self._ui_language != "ru" else "Радиальная асимметрия",
            "edge_likeness": "Edge-likeness" if self._ui_language != "ru" else "Похожесть на край",
            "line_likeness": "Line-likeness" if self._ui_language != "ru" else "Линейность",
            "distance_to_edge": "Distance to edge" if self._ui_language != "ru" else "Дистанция до края",
            "final_overlay": "Final overlay" if self._ui_language != "ru" else "Итоговый overlay",
            "mask": "Mask" if self._ui_language != "ru" else "Маска",
        }
        seen: set[str] = set()
        for key in ordering + sorted(maps.keys()):
            if key in seen or key not in maps:
                continue
            seen.add(key)
            array = maps.get(key)
            if array is None:
                continue
            try:
                image = np.asarray(array)
            except Exception:  # pragma: no cover - defensive
                continue
            if image.size == 0:
                continue
            pixmap = self._gradient_debug_pixmap(image)
            if pixmap is None:
                continue
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(4, 4, 4, 4)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setPixmap(pixmap)
            scroll.setWidget(label)
            page_layout.addWidget(scroll, 1)
            info = QLabel(
                f"{image.shape[1]} x {image.shape[0]} px"
                + (f" · dtype={image.dtype}" if hasattr(image, "dtype") else "")
            )
            page_layout.addWidget(info)
            tabs.addTab(page, pretty_names.get(key, key))
        close_button = QPushButton("Close" if self._ui_language != "ru" else "Закрыть")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        dialog.exec()

    def _compute_gradient_debug_maps_on_demand(self) -> dict[str, object]:
        current_state = self._workspace.current_state
        if current_state is None or current_state.source_image is None:
            return {}
        from .application.use_cases.processing import build_detection_debug_maps

        settings = self._current_contour_settings()
        preprocessed = current_state.preprocessed_image
        if preprocessed is None:
            preprocessed = current_state.source_image
        maps = build_detection_debug_maps(current_state.source_image, preprocessed, settings)
        try:
            current_state.debug_gradient_maps = dict(maps)
        except Exception:  # pragma: no cover - defensive
            pass
        return maps

    def _on_gradient_overlay_toggled(self, _checked: bool = False) -> None:
        if not self.gradient_overlay_checkbox.isChecked():
            if hasattr(self, "polygon_editor"):
                self.polygon_editor.clear_gradient_overlay()
            return
        self._refresh_gradient_overlay()

    def _on_gradient_overlay_opacity_changed(self, value: float) -> None:
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_gradient_overlay_opacity(float(value))

    def _refresh_gradient_overlay(self) -> None:
        if not hasattr(self, "polygon_editor"):
            return
        rec = (
            str(self.recognition_mode_combo.currentData() or "")
            if hasattr(self, "recognition_mode_combo")
            else ""
        )
        if (
            rec == "conductors"
            and hasattr(self, "metal_show_mask_checkbox")
            and self.metal_show_mask_checkbox.isChecked()
        ):
            _st = self._workspace.current_state
            _maps: dict = getattr(_st, "debug_gradient_maps", None) or {} if _st is not None else {}
            if any(k in _maps for k in ("metal_filtered_mask", "metal_binary_mask", "metal_mask")):
                self._apply_metal_visual_overlay()
                return
        if not hasattr(self, "gradient_overlay_checkbox"):
            self.polygon_editor.clear_gradient_overlay()
            return
        if not self.gradient_overlay_checkbox.isChecked():
            self.polygon_editor.clear_gradient_overlay()
            return
        current_state = self._workspace.current_state
        if current_state is None or current_state.source_image is None:
            self.polygon_editor.clear_gradient_overlay()
            return
        try:
            overlay = self._build_gradient_overlay_image(current_state.source_image)
        except Exception:  # pragma: no cover - defensive: UI must never crash
            self.polygon_editor.clear_gradient_overlay()
            return
        if overlay is None:
            self.polygon_editor.clear_gradient_overlay()
            return
        self.polygon_editor.set_gradient_overlay(overlay, float(self.gradient_overlay_opacity_spin.value()))

    def _apply_metal_visual_overlay(self) -> None:
        if not hasattr(self, "polygon_editor"):
            return
        current_state = self._workspace.current_state
        if current_state is None:
            self.polygon_editor.clear_gradient_overlay()
            return
        maps: dict = getattr(current_state, "debug_gradient_maps", None) or {}
        mode = (
            str(self.metal_debug_visual_combo.currentData() or "overlay")
            if hasattr(self, "metal_debug_visual_combo")
            else "overlay"
        )
        op = float(self.metal_overlay_opacity_spin.value()) if hasattr(self, "metal_overlay_opacity_spin") else 0.45
        try:
            if mode == "overlay":
                src = current_state.source_image
                if src is None:
                    self.polygon_editor.clear_gradient_overlay()
                    return
                vis = np.asarray(src)
                if vis.ndim == 2:
                    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                m = maps.get("metal_filtered_mask") or maps.get("metal_binary_mask") or maps.get("metal_mask")
                if m is None or np.asarray(m).size == 0:
                    self.polygon_editor.clear_gradient_overlay()
                    return
                binm = (np.asarray(m) > 0).astype(np.uint8)
                tint = np.zeros_like(vis)
                tint[:, :, 1] = binm * 200
                tint[:, :, 0] = binm * 40
                out = cv2.addWeighted(vis, 1.0 - 0.55 * op, tint, 0.55 * op, 0)
                if current_state and getattr(current_state, "polygons", None):
                    for poly in current_state.polygons:
                        if str(getattr(poly, "category", "")) != "metal_wide_gradient":
                            continue
                        if len(poly.points) < 2:
                            continue
                        pts = np.array([(int(x), int(y)) for x, y in poly.points], dtype=np.int32).reshape(
                            -1, 1, 2
                        )
                        cv2.polylines(out, [pts], True, (255, 120, 40), 2)
                self.polygon_editor.set_gradient_overlay(out, 1.0)
                return
            arr = maps.get(mode)
            if arr is None:
                self.polygon_editor.clear_gradient_overlay()
                return
            image = np.asarray(arr)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.polygon_editor.set_gradient_overlay(image, min(1.0, max(0.05, op)))
        except Exception:  # pragma: no cover
            self.polygon_editor.clear_gradient_overlay()

    def _build_gradient_overlay_image(self, source_image: np.ndarray) -> np.ndarray | None:
        from .application.use_cases.processing import (
            _resolve_conductor_edge_method,
            _resolve_via_edge_method,
            _via_grayscale,
        )
        from .edge_detection import build_gradient_elevation

        settings = self._current_contour_settings()
        if settings.object_type == "via" or settings.output_mode == "box":
            method = _resolve_via_edge_method(settings)
        else:
            method = _resolve_conductor_edge_method(settings)
        gray = _via_grayscale(source_image)
        if gray.size == 0:
            return None
        elevation = build_gradient_elevation(gray, method)
        mode = str(self.gradient_overlay_mode_combo.currentData() or "heatmap")
        if mode == "elevation":
            return cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR)
        if mode == "threshold":
            threshold = float(settings.via_min_contrast)
            mask = elevation >= threshold
            overlay = np.zeros((elevation.shape[0], elevation.shape[1], 3), dtype=np.uint8)
            overlay[..., 1] = mask.astype(np.uint8) * 230
            overlay[..., 2] = mask.astype(np.uint8) * 60
            return overlay
        heatmap = cv2.applyColorMap(elevation, cv2.COLORMAP_TURBO)
        threshold = float(settings.via_min_contrast)
        if settings.object_type == "via" or settings.output_mode == "box":
            below = (elevation < max(0.0, threshold)).astype(np.uint8)
            if below.any():
                dimmed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                below3 = below[..., None]
                heatmap = heatmap * (1 - below3) + (dimmed // 3) * below3
                heatmap = heatmap.astype(np.uint8)
        return heatmap

    def _gradient_debug_pixmap(self, image: np.ndarray) -> QPixmap | None:
        data = np.asarray(image)
        if data.size == 0:
            return None
        if data.dtype != np.uint8:
            if data.dtype == bool:
                data = data.astype(np.uint8) * 255
            else:
                as_float = data.astype(np.float32)
                max_val = float(as_float.max()) if as_float.size else 0.0
                if max_val <= 1.0001:
                    data = np.clip(as_float * 255.0, 0, 255).astype(np.uint8)
                else:
                    min_val = float(as_float.min())
                    span = max_val - min_val
                    if span <= 1e-6:
                        data = np.clip(as_float, 0, 255).astype(np.uint8)
                    else:
                        data = np.clip((as_float - min_val) / span * 255.0, 0, 255).astype(np.uint8)
        try:
            qimage = cv_to_qimage(data)
        except Exception:  # pragma: no cover - defensive
            return None
        return QPixmap.fromImage(qimage)

    def _best_debug_candidate_for_polygon(self, polygon: PolygonData, candidates: list[object]) -> object | None:
        polygon_rect = self._polygon_rect(polygon)
        if polygon_rect.isNull() or not candidates:
            return None
        polygon_center = polygon_rect.center()
        best_candidate: object | None = None
        best_rank: tuple[int, int, float, float] | None = None
        for index, candidate in enumerate(candidates):
            bbox = getattr(candidate, "bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            candidate_rect = QRectF(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])).normalized()
            if candidate_rect.isNull():
                continue
            overlap = self._rect_overlap_area(polygon_rect, candidate_rect)
            candidate_center = candidate_rect.center()
            dx = polygon_center.x() - candidate_center.x()
            dy = polygon_center.y() - candidate_center.y()
            distance_sq = dx * dx + dy * dy
            max_span = max(
                polygon_rect.width(), polygon_rect.height(), candidate_rect.width(), candidate_rect.height(), 1.0
            )
            if overlap <= 0.0 and distance_sq > (max_span * 1.5) * (max_span * 1.5):
                continue
            accepted_rank = 1 if bool(getattr(candidate, "accepted", False)) else 0
            rank = (accepted_rank, 1 if overlap > 0.0 else 0, overlap, -distance_sq - index * 1e-9)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_candidate = candidate
        return best_candidate

    @staticmethod
    def _polygon_rect(polygon: PolygonData) -> QRectF:
        if polygon.points:
            x_values = [point[0] for point in polygon.points]
            y_values = [point[1] for point in polygon.points]
            return QRectF(
                min(x_values),
                min(y_values),
                max(x_values) - min(x_values),
                max(y_values) - min(y_values),
            ).normalized()
        x_coord, y_coord, width, height = polygon.bbox
        return QRectF(float(x_coord), float(y_coord), float(width), float(height)).normalized()

    @staticmethod
    def _rect_overlap_area(first: QRectF, second: QRectF) -> float:
        overlap = first.intersected(second)
        if overlap.isNull():
            return 0.0
        return max(0.0, overlap.width()) * max(0.0, overlap.height())

    @staticmethod
    def _debug_candidate_source(candidate: object) -> str:
        source = str(getattr(candidate, "source", "") or "")
        reason = str(getattr(candidate, "reason", "") or "")
        if not source and ":" in reason:
            source = reason.split(":", 1)[1]
        return source

    def _debug_method_text(self, source: str) -> str:
        source = source.lower()
        labels = {
            "range-components": ("Диапазон яркости + компоненты", "Intensity range + components"),
            "range-contours": ("Диапазон яркости + контуры", "Intensity range + contours"),
            "gradient": ("Градиент округлой формы", "Round gradient"),
            "spot": ("Локальная яркая/темная точка", "Local bright/dark spot"),
            "hough-gray": ("HoughCircles по grayscale", "HoughCircles on grayscale"),
            "hough": ("HoughCircles", "HoughCircles"),
            "components": ("Связанные компоненты", "Connected components"),
            "contours-response": ("Контуры по карте отклика", "Contours on response map"),
            "contours": ("Контуры", "Contours"),
            "morphology": ("Морфологические пики", "Morphology peaks"),
            "template": ("Шаблон", "Template"),
            "blob": ("Blob detector", "Blob detector"),
        }
        for prefix, pair in labels.items():
            if source.startswith(prefix):
                return pair[0] if self._ui_language == "ru" else pair[1]
        return source or ("Неизвестно" if self._ui_language == "ru" else "Unknown")

    def _debug_criterion_text(self, source: str, reason: str, accepted: bool) -> str:
        if not accepted:
            rejection_labels = {
                "duplicate": (
                    "дубликат более сильного ближайшего кандидата",
                    "duplicate of a stronger nearby candidate",
                ),
                "component_score": (
                    "отклик компоненты ниже заданного порога",
                    "component response is below the configured threshold",
                ),
                "contour_score": (
                    "отклик контура ниже заданного порога",
                    "contour response is below the configured threshold",
                ),
                "min_via_width": ("ширина меньше допустимого минимума", "width is below the allowed minimum"),
                "max_via_width": ("ширина больше допустимого максимума", "width is above the allowed maximum"),
                "min_via_height": ("высота меньше допустимого минимума", "height is below the allowed minimum"),
                "max_via_height": ("высота больше допустимого максимума", "height is above the allowed maximum"),
                "min_aspect_ratio": (
                    "соотношение сторон ниже допустимого",
                    "aspect ratio is below the allowed minimum",
                ),
                "max_aspect_ratio": (
                    "соотношение сторон выше допустимого",
                    "aspect ratio is above the allowed maximum",
                ),
                "roundness": ("округлость ниже заданного порога", "roundness is below the configured threshold"),
                "empty_geometry": ("пустая геометрия кандидата", "candidate geometry is empty"),
                "min_polygon_width": (
                    "Отклонено: ширина меньше минимальной",
                    "rejected: width below the configured minimum",
                ),
            }
            pair = rejection_labels.get(reason)
            if pair is not None:
                return pair[0] if self._ui_language == "ru" else pair[1]
            return reason or (
                "кандидат не прошел фильтры" if self._ui_language == "ru" else "candidate did not pass filters"
            )
        source = source.lower()
        accepted_labels = {
            "range-components": (
                "пиксели попали в заданный диапазон яркости, компонент прошел фильтры размера, формы и округлости",
                "pixels matched the configured intensity range, and the component passed size, shape, and roundness filters",
            ),
            "range-contours": (
                "контур из диапазона яркости прошел фильтры размера, формы и округлости",
                "an intensity-range contour passed size, shape, and roundness filters",
            ),
            "gradient": (
                "найден локальный круглый перепад яркости с достаточной силой и покрытием границы",
                "a local round brightness gradient had enough strength and edge coverage",
            ),
            "spot": (
                "найдена компактная локальная точка после подавления длинных дорожек",
                "a compact local spot was found after suppressing long traces",
            ),
            "hough-gray": (
                "HoughCircles нашел окружность на подготовленном grayscale-изображении",
                "HoughCircles found a circle on the prepared grayscale image",
            ),
            "hough": (
                "HoughCircles нашел окружность на карте отклика",
                "HoughCircles found a circle on the response map",
            ),
            "components": (
                "связанная компонента прошла порог отклика и геометрические фильтры",
                "a connected component passed the response threshold and geometry filters",
            ),
            "contours-response": (
                "контур на карте отклика прошел порог и геометрические фильтры",
                "a response-map contour passed the threshold and geometry filters",
            ),
            "contours": (
                "контур прошел порог отклика и геометрические фильтры",
                "a contour passed the response threshold and geometry filters",
            ),
            "morphology": (
                "морфологический пик прошел фильтры размера и формы",
                "a morphology peak passed size and shape filters",
            ),
            "template": (
                "область совпала с одним из сохраненных шаблонов выше заданного порога",
                "the area matched one of the saved templates above the configured threshold",
            ),
            "blob": (
                "Blob detector нашел компактное пятно с достаточной округлостью",
                "Blob detector found a compact spot with enough circularity",
            ),
        }
        for prefix, pair in accepted_labels.items():
            if source.startswith(prefix):
                return pair[0] if self._ui_language == "ru" else pair[1]
        return (
            "кандидат прошел фильтры размера, пропорций и округлости"
            if self._ui_language == "ru"
            else "candidate passed size, aspect, and roundness filters"
        )

    def _add_pipeline_step(self) -> None:
        operation_name = self._selected_available_operation_name()
        if not operation_name:
            return
        self._pipeline.steps.append(PreprocessingPipeline.create_step(operation_name))
        self._populate_pipeline_list()
        self.pipeline_list.setCurrentRow(len(self._pipeline.steps) - 1)
        self._auto_apply_pipeline()

    def _remove_pipeline_step(self) -> None:
        row = self.pipeline_list.currentRow()
        if row < 0:
            return
        self._pipeline.steps.pop(row)
        self._populate_pipeline_list()
        self._auto_apply_pipeline()

    def _move_pipeline_step_up(self) -> None:
        row = self.pipeline_list.currentRow()
        if row <= 0:
            return
        self._pipeline.steps[row - 1], self._pipeline.steps[row] = (
            self._pipeline.steps[row],
            self._pipeline.steps[row - 1],
        )
        self._populate_pipeline_list()
        self.pipeline_list.setCurrentRow(row - 1)
        self._auto_apply_pipeline()

    def _move_pipeline_step_down(self) -> None:
        row = self.pipeline_list.currentRow()
        if row < 0 or row >= len(self._pipeline.steps) - 1:
            return
        self._pipeline.steps[row + 1], self._pipeline.steps[row] = (
            self._pipeline.steps[row],
            self._pipeline.steps[row + 1],
        )
        self._populate_pipeline_list()
        self.pipeline_list.setCurrentRow(row + 1)
        self._auto_apply_pipeline()

    def _on_pipeline_item_changed(self, item: QListWidgetItem) -> None:
        if self._ignore_pipeline_item_change:
            return
        row = self.pipeline_list.row(item)
        if row < 0 or row >= len(self._pipeline.steps):
            return
        self._pipeline.steps[row].enabled = item.checkState() == Qt.CheckState.Checked
        self._auto_apply_pipeline()

    def _sync_pipeline_order_from_list(self) -> None:
        if self._ignore_pipeline_item_change:
            return
        old_steps = list(self._pipeline.steps)
        new_steps = []
        for row in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(row)
            old_index = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(old_index, int) or old_index < 0 or old_index >= len(old_steps):
                return
            new_steps.append(old_steps[old_index])
        if len(new_steps) != len(old_steps) or all(
            first is second for first, second in zip(new_steps, old_steps, strict=False)
        ):
            return
        self._pipeline.steps = new_steps
        for row in range(self.pipeline_list.count()):
            self.pipeline_list.item(row).setData(Qt.ItemDataRole.UserRole, row)
        self._render_pipeline_parameters(self.pipeline_list.currentRow())
        self._auto_apply_pipeline()

    def _on_epsilon_spin_value_changed(self, *_args) -> None:
        if hasattr(self, "epsilon_slider"):
            self.epsilon_slider.blockSignals(True)
            try:
                self.epsilon_slider.setValue(min(1000, max(0, round(self.epsilon_spin.value() * 100.0))))
            finally:
                self.epsilon_slider.blockSignals(False)
        self._on_extraction_settings_changed()

    def _on_epsilon_slider_value_changed(self, value: int) -> None:
        self.epsilon_spin.blockSignals(True)
        try:
            self.epsilon_spin.setValue(value / 100.0)
        finally:
            self.epsilon_spin.blockSignals(False)
        self._on_extraction_settings_changed()

    def _on_extraction_settings_changed(self, *_args) -> None:
        # Stop in-flight preview immediately (cooperative cancel); keep prepared-image
        # workers running — pipeline / source unchanged.
        self._abort_in_flight_interactive_processing(preview=True, prepared=False)
        if not hasattr(self, "_extraction_settings_debounce"):
            self._extraction_settings_debounce = QTimer(self)
            self._extraction_settings_debounce.setSingleShot(True)
            self._extraction_settings_debounce.timeout.connect(self._flush_extraction_settings_changed)
        self._extraction_settings_debounce.stop()
        self._extraction_settings_debounce.start(120)

    def _flush_extraction_settings_changed(self) -> None:
        if hasattr(self, "via_white_range_checkbox"):
            self._update_via_threshold_controls_state()
        self._store_active_extraction_profile_settings()
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_debug_candidates([])
            self.polygon_editor.set_via_debug_inspection_enabled(self._via_debug_inspection_enabled())
        self._refresh_gradient_overlay()
        self._auto_apply_pipeline()

    def _on_via_search_method_changed(self, *_args) -> None:
        if hasattr(self, "bright_via_mode_stack"):
            self.bright_via_mode_stack.setCurrentIndex(
                1 if self.via_search_mode_combo.currentData() == VIA_SEARCH_MODE_TEMPLATE else 0
            )

    def _sync_via_diameter_size_mode(self, *_args) -> None:
        if hasattr(self, "via_size_mode_combo") and hasattr(self, "via_diameter_size_mode_combo"):
            with QSignalBlocker(self.via_size_mode_combo):
                self.via_size_mode_combo.setCurrentIndex(self.via_diameter_size_mode_combo.currentIndex())
        self._on_via_size_mode_changed()

    def _on_via_size_mode_changed(self, *_args) -> None:
        self._update_via_size_controls_state()
        if (
            normalize_via_size_mode(self.via_size_mode_combo.currentData()) == VIA_SIZE_MODE_FIXED
            and not self._fixed_via_rows
        ):
            self._add_fixed_via_row(width=1, height=1)
            return
        self._on_extraction_settings_changed()

    def _on_extraction_profile_changed(self, *_args) -> None:
        """Legacy hook; profile is controlled by recognition mode."""

    def _store_active_extraction_profile_settings(self) -> None:
        if not hasattr(self, "recognition_mode_combo"):
            return
        rec = str(self.recognition_mode_combo.currentData() or "conductors")
        profile = "vias" if rec == "via" else "conductors"
        self._active_extraction_profile = profile
        settings = self._current_contour_settings()
        settings.extraction_profile = profile
        settings.object_type = "via" if profile == "vias" else "conductor"
        self._contour_settings_profiles[profile] = settings

    def _save_pipeline_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("save_pipeline_dialog_title"),
            "",
            self._tr("json_file_filter"),
        )
        if not path:
            return
        save_pipeline_config_to_path(path, self.get_pipeline())
        self._append_log(self._tr("pipeline_saved_log", path=path))

    def _load_pipeline_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("load_pipeline_dialog_title"),
            "",
            self._tr("json_file_filter"),
        )
        if not path:
            return
        payload = load_pipeline_config_from_path(path)
        self.set_pipeline(payload)
        self._append_log(self._tr("pipeline_loaded_log", path=path))

    def _on_image_item_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        if previous is not None and not self._try_leave_current_frame():
            self.image_list.blockSignals(True)
            try:
                self.image_list.setCurrentItem(previous)
            finally:
                self.image_list.blockSignals(False)
            self._sync_frame_navigation_controls()
            return
        if current is None:
            self._sync_frame_navigation_controls()
            return
        image_path = current.data(Qt.ItemDataRole.UserRole)
        if image_path:
            try:
                self.load_image(str(image_path))
            except Exception as exc:
                self._append_log(self._tr("failed_to_load_image_log", image_path=image_path, error=exc))
                QMessageBox.warning(self, self._tr("image_load_error_title"), str(exc))
        self._sync_frame_navigation_controls()

    def _prune_tagged_sets_for_images(self, retained_paths: list[str]) -> None:
        retained = {str(Path(p)) for p in retained_paths}
        stems = {Path(p).stem.lower() for p in retained_paths}
        self._persisted_highlight_paths.intersection_update(retained)
        self._viewed_image_paths.intersection_update(retained)
        self._cif_load_failure_stems.intersection_update(stems)

    def _matching_report(self):
        return build_image_cif_matching_report(
            list(self._workspace.image_paths),
            self._workspace.cif_paths_by_stem,
        )

    def _log_matching_gaps_after_refresh(self, report) -> None:
        if report.stems_with_image_but_no_cif:
            sample = sorted(report.stems_with_image_but_no_cif)[:12]
            more_txt = ""
            extra = len(report.stems_with_image_but_no_cif) - len(sample)
            if extra > 0:
                more_txt = f" (+{extra})"
            self._append_log(
                self._tr(
                    "images_without_matching_cif_log",
                    count=len(report.stems_with_image_but_no_cif),
                    sample=", ".join(sample),
                    more=more_txt,
                )
            )
        if report.stems_with_cif_but_no_image:
            sample = sorted(report.stems_with_cif_but_no_image)[:12]
            more_txt = ""
            extra = len(report.stems_with_cif_but_no_image) - len(sample)
            if extra > 0:
                more_txt = f" (+{extra})"
            self._append_log(
                self._tr(
                    "cif_without_matching_image_log",
                    count=len(report.stems_with_cif_but_no_image),
                    sample=", ".join(sample),
                    more=more_txt,
                )
            )

    def _complete_directory_scan_turn(self) -> str | None:
        self._directory_scan_busy = False
        if hasattr(self, "files_scan_progress_bar"):
            self.files_scan_progress_bar.setVisible(False)
            self.files_scan_progress_bar.setRange(0, 100)
            self.files_scan_progress_bar.setValue(0)
        pending = self._directory_scan_pending_directory
        self._directory_scan_pending_directory = None
        return pending

    def _begin_async_directory_scan(self, directory: str) -> None:
        normalized = str(Path(directory))
        if self._directory_scan_busy:
            self._directory_scan_pending_directory = normalized
            return
        self._run_directory_scan_now(normalized)

    def _maybe_start_pending_directory_scan(self, pending_directory: str | None) -> None:
        if pending_directory:
            self._run_directory_scan_now(pending_directory)

    def _run_directory_scan_now(self, directory: str) -> None:
        self._directory_scan_busy = True
        scan_generation = self._scan_generation
        if hasattr(self, "files_scan_progress_bar"):
            self.files_scan_progress_bar.setVisible(True)
            self.files_scan_progress_bar.setRange(0, 0)
            self.files_scan_progress_bar.setFormat(self._tr("scanning_directory_progress"))
        runnable = ScanInputDirectoryRunnable(
            directory=directory,
            signals=self._directory_scan_signals,
            run_generation=scan_generation,
        )
        self._scan_thread_pool.start(runnable)

    def _on_input_directory_scan_finished(self, paths: list[str], run_generation: int) -> None:
        pending = self._complete_directory_scan_turn()
        if run_generation != self._scan_generation:
            self._maybe_start_pending_directory_scan(pending)
            return
        self.load_images(paths)
        self._maybe_start_pending_directory_scan(pending)

    def _on_input_directory_scan_failed(self, message: str, run_generation: int) -> None:
        pending = self._complete_directory_scan_turn()
        if run_generation == self._scan_generation:
            self._append_log(
                self._tr(
                    "scan_input_directory_failed_log",
                    error=message,
                )
            )
        self._maybe_start_pending_directory_scan(pending)

    def _on_sidebar_list_mode_changed(self, index: int) -> None:
        if index < 0:
            return
        if hasattr(self, "sidebar_list_stack"):
            self.sidebar_list_stack.setCurrentIndex(0 if index == 0 else 1)
        if index == 1:
            # Defer arming: the mouse release that closes the combo popup is often
            # delivered to the vector list in the *next* event-loop tick.
            def _arm_suppress() -> None:
                self._vectors_list_ignore_navigate_until = time.monotonic() + 0.55

            QTimer.singleShot(0, _arm_suppress)
        else:
            self._vectors_list_ignore_navigate_until = 0.0

    def _image_path_for_cif_stem(self, stem: str) -> str | None:
        target = stem.lower()
        for path in self._workspace.image_paths:
            if Path(path).stem.lower() == target:
                return str(Path(path))
        return None

    def _paint_vector_list_item(self, item: QListWidgetItem, stem: str) -> None:
        stem_lower = stem.lower()
        status = self._vector_status_enum_for_stem(stem_lower)
        item.setData(FRAME_STATUS_ROLE, status.value)
        hex_background = background_hex_vector_status(status)
        if hex_background:
            tint = QColor(hex_background)
            item.setBackground(QBrush(tint))
            # Windows / some styles ignore setBackground for items; BackgroundRole is respected more reliably.
            item.setData(Qt.ItemDataRole.BackgroundRole, tint)
        else:
            item.setBackground(QBrush())
            item.setData(Qt.ItemDataRole.BackgroundRole, None)
        raw_path = item.data(Qt.ItemDataRole.UserRole)
        if raw_path:
            item.setText(Path(str(raw_path)).stem)

    def _vector_status_enum_for_stem(self, stem_lower: str):
        ipath = self._image_path_for_cif_stem(stem_lower)
        has_matching = ipath is not None
        cif_failed = stem_lower in self._cif_load_failure_stems
        normalized = "" if ipath is None else str(Path(ipath))
        never_opened = (not normalized) or (normalized not in self._viewed_image_paths)
        dirty = bool(ipath is not None and self._workspace.image_has_changes(normalized))
        persist = normalized in self._persisted_highlight_paths if normalized else False
        return classify_vector_side_status(
            has_matching_image=has_matching,
            cif_load_failed=cif_failed,
            image_never_viewed=never_opened,
            polygons_dirty=dirty,
            persist_highlight=persist,
        )

    def _rebuild_vector_list(self) -> None:
        if not hasattr(self, "vector_list"):
            return
        self.vector_list.blockSignals(True)
        self.vector_list.clear()
        mapping = sorted(self._workspace.cif_paths_by_stem.items(), key=lambda kv: kv[0].lower())
        for stem, cif_path in mapping:
            item = QListWidgetItem(Path(cif_path).stem)
            item.setToolTip(cif_path)
            item.setData(Qt.ItemDataRole.UserRole, cif_path)
            self._paint_vector_list_item(item, stem)
            self.vector_list.addItem(item)
        self.vector_list.blockSignals(False)

    def _configure_thumbnail_grid_geometry(self) -> None:
        if not hasattr(self, "thumbnail_grid"):
            return
        columns = self._thumbnail_columns()
        icon_size = self._thumbnail_icon_size if hasattr(self, "_thumbnail_icon_size") else QSize(64, 48)
        cell_w = int(icon_size.width())
        cell_h = int(icon_size.height())
        self.thumbnail_grid.setIconSize(icon_size)
        self.thumbnail_grid.setGridSize(QSize(cell_w, cell_h))
        self.thumbnail_grid.setSpacing(0)
        frame = 2 * int(self.thumbnail_grid.frameWidth())
        self.thumbnail_grid.setMinimumWidth(max(cell_w + frame, columns * cell_w + frame))
        visible_rows = 2
        self.thumbnail_grid.setMinimumHeight(cell_h + frame)
        self.thumbnail_grid.setMaximumHeight(max(cell_h + frame, visible_rows * cell_h + frame))

    def _thumbnail_columns(self) -> int:
        if not hasattr(self, "neighbor_columns_spin"):
            return 3
        try:
            return max(1, int(self.neighbor_columns_spin.value()))
        except (TypeError, ValueError):
            return 1

    def _thumbnail_placeholder(self) -> QIcon:
        if not getattr(self, "_thumbnail_placeholder_icon", QIcon()).isNull():
            return self._thumbnail_placeholder_icon
        size = self._thumbnail_icon_size if hasattr(self, "_thumbnail_icon_size") else QSize(64, 48)
        pixmap = QPixmap(size)
        pixmap.fill(QColor("#1F2937"))
        self._thumbnail_placeholder_icon = QIcon(pixmap)
        return self._thumbnail_placeholder_icon

    def _rebuild_thumbnail_grid(self) -> None:
        if not hasattr(self, "thumbnail_grid"):
            return
        self._thumbnail_generation += 1
        generation = self._thumbnail_generation
        self._configure_thumbnail_grid_geometry()
        self.thumbnail_grid.blockSignals(True)
        try:
            self.thumbnail_grid.clear()
            for path in self._workspace.image_paths:
                item = QListWidgetItem(self._thumbnail_placeholder(), "")
                item.setToolTip(Path(str(path)).stem)
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self._paint_image_row_item(item, str(path), show_text=False)
                self.thumbnail_grid.addItem(item)
                runnable = ThumbnailLoadRunnable(
                    generation,
                    str(path),
                    self._thumbnail_icon_size.width(),
                    self._thumbnail_icon_size.height(),
                )
                runnable.signals.result.connect(self._on_thumbnail_loaded)
                self._thumbnail_thread_pool.start(runnable)
        finally:
            self.thumbnail_grid.blockSignals(False)
        self._update_thumbnail_grid_selection()

    def _on_thumbnail_loaded(self, generation: int, path: str, image: object) -> None:
        if generation != self._thumbnail_generation or not hasattr(self, "thumbnail_grid"):
            return
        target = str(Path(path))
        item = None
        for index in range(self.thumbnail_grid.count()):
            candidate = self.thumbnail_grid.item(index)
            if candidate is not None and str(candidate.data(Qt.ItemDataRole.UserRole) or "") == target:
                item = candidate
                break
        if item is None:
            return
        if image is None:
            item.setIcon(self._thumbnail_placeholder())
            return
        pixmap = QPixmap.fromImage(cv_to_qimage(image))
        if pixmap.isNull():
            item.setIcon(self._thumbnail_placeholder())
        else:
            item.setIcon(QIcon(pixmap))

    def _update_thumbnail_grid_selection(self) -> None:
        if not hasattr(self, "thumbnail_grid"):
            return
        current = self._workspace.current_image_path
        matched = False
        self.thumbnail_grid.blockSignals(True)
        try:
            for index in range(self.thumbnail_grid.count()):
                item = self.thumbnail_grid.item(index)
                selected = bool(current and item is not None and item.data(Qt.ItemDataRole.UserRole) == current)
                if selected:
                    self.thumbnail_grid.setCurrentRow(index)
                    matched = True
                if item is not None:
                    path = str(item.data(Qt.ItemDataRole.UserRole) or "")
                    if selected:
                        item.setBackground(QBrush(QColor("#1D4ED8")))
                        item.setData(Qt.ItemDataRole.BackgroundRole, QColor("#1D4ED8"))
                    elif path:
                        self._paint_image_row_item(item, path, show_text=False)
            if not matched:
                self.thumbnail_grid.clearSelection()
                self.thumbnail_grid.setCurrentRow(-1)
        finally:
            self.thumbnail_grid.blockSignals(False)

    def _on_thumbnail_item_clicked(self, item: QListWidgetItem) -> None:
        if item is None:
            return
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not path:
            return
        image_item = self._find_image_list_item(path)
        if image_item is not None:
            self.image_list.setCurrentItem(image_item)

    def _refresh_vector_rows_for_workspace(self) -> None:
        if not hasattr(self, "vector_list"):
            return
        for index in range(self.vector_list.count()):
            row = self.vector_list.item(index)
            if row is None:
                continue
            tip = row.toolTip()
            if not tip:
                continue
            self._paint_vector_list_item(row, Path(tip).stem.lower())

    def _refresh_vector_items_for_stems(self, stems: set[str]) -> None:
        if not stems or not hasattr(self, "vector_list"):
            return
        lowered = {s.lower() for s in stems}
        for idx in range(self.vector_list.count()):
            item = self.vector_list.item(idx)
            if item is None:
                continue
            tip = item.toolTip()
            if not tip:
                continue
            stem = Path(tip).stem.lower()
            if stem in lowered:
                self._paint_vector_list_item(item, stem)

    def _select_input_image_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            self._tr("select_image_files_dialog"),
            self.input_dir_edit.text() or str(Path.home()),
            self._tr("supported_image_files_filter"),
        )
        if not paths:
            return
        self.input_dir_edit.setText(str(Path(paths[0]).parent))
        self._save_persisted_paths()
        self.load_images([str(Path(p)) for p in paths])

    def _merge_cif_files_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            self._tr("merge_cif_files_dialog"),
            self.cif_dir_edit.text() or str(Path.home()),
            self._tr("cif_files_filter"),
        )
        if not paths:
            return
        additions = index_cif_file_paths(paths)
        if not additions:
            return
        self._workspace.merge_cif_paths(additions)
        self.cif_dir_edit.setText(str(Path(paths[0]).parent))
        self._save_persisted_paths()
        self._append_log(self._tr("cif_indexed_log", count=len(self._workspace.cif_paths_by_stem)))
        self._sync_after_cif_index_changed()

    def _sync_after_cif_index_changed(self) -> None:
        self._clear_cif_transient_hints()
        self._rebuild_vector_list()
        self._refresh_image_list_item_states()
        cur = self._workspace.current_image_path
        if cur:
            try:
                self.load_image(cur)
            except Exception as exc:
                self._append_log(self._tr("reload_with_cif_failed_log", error=exc))
        report = self._matching_report()
        self._log_matching_gaps_after_refresh(report)

    def _clear_cif_transient_hints(self) -> None:
        self._cif_load_failure_stems.clear()

    def _invalidate_cif_overlay_for_stems(self, stems: set[str]) -> list[str]:
        if not stems:
            return []
        paths: list[str] = []
        for path in self._workspace.image_paths:
            stem = Path(path).stem.lower()
            if stem in stems:
                self._cif_load_failure_stems.discard(stem)
                paths.append(str(Path(path)))
        if paths:
            self._workspace.invalidate_image_states(paths)
        return paths

    def _reload_cif_overlays_for_selected_vectors(self) -> None:
        stems: set[str] = set()
        for row in self.vector_list.selectedItems():
            tip = row.toolTip()
            if tip:
                stems.add(Path(tip).stem.lower())
        if not stems:
            self._append_log(self._tr("no_vector_selection_for_reload_log"))
            return
        affected = self._invalidate_cif_overlay_for_stems(stems)
        self._finalize_overlay_reload(affected)

    def _reload_cif_overlays_for_selected_images(self) -> None:
        stems: set[str] = {Path(str(row.data(Qt.ItemDataRole.UserRole))).stem.lower() for row in self.image_list.selectedItems()}
        cur = self._workspace.current_image_path
        if not stems and cur:
            stems.add(Path(cur).stem.lower())
        if not stems:
            self._append_log(self._tr("no_image_selection_for_reload_log"))
            return
        affected = self._invalidate_cif_overlay_for_stems(stems)
        self._finalize_overlay_reload(affected)

    def _finalize_overlay_reload(self, paths_for_message: list[str]) -> None:
        if not paths_for_message:
            self._append_log(self._tr("cif_reload_no_matching_images_log"))
            return
        unique_stems = sorted({Path(p).stem.lower() for p in paths_for_message})[:16]
        more_txt = ""
        extra = len({Path(p).stem.lower() for p in paths_for_message}) - len(unique_stems)
        if extra > 0:
            more_txt = f" (+{extra})"
        self._append_log(
            self._tr(
                "cif_reload_invalidate_log",
                stems=", ".join(unique_stems),
                more=more_txt,
            )
        )
        self._refresh_image_list_item_states()
        self._refresh_vector_rows_for_workspace()
        cur = self._workspace.current_image_path
        if cur and cur in paths_for_message:
            try:
                self.load_image(cur)
            except Exception as exc:
                self._append_log(self._tr("reload_with_cif_failed_log", error=exc))

    def _sync_frame_navigation_controls(self) -> None:
        if not hasattr(self, "frame_nav_spin"):
            return
        paths = list(self._workspace.image_paths)
        total = len(paths)
        self.frame_nav_spin.blockSignals(True)
        if total <= 0:
            self.visual_frame_nav_widget.setEnabled(False)
            self.frame_nav_spin.setMinimum(1)
            self.frame_nav_spin.setMaximum(1)
            self.frame_nav_spin.setValue(1)
            self.frame_nav_total_label.setText("/ 0")
        else:
            self.visual_frame_nav_widget.setEnabled(True)
            self.frame_nav_spin.setMinimum(1)
            self.frame_nav_spin.setMaximum(total)
            current = self._workspace.current_image_path
            position = paths.index(current) + 1 if current and current in paths else min(self.frame_nav_spin.value(), total)
            position = max(1, min(position, total))
            self.frame_nav_spin.setValue(position)
            self.frame_nav_total_label.setText(f"/ {total}")
        self.frame_nav_spin.blockSignals(False)

    def _on_frame_nav_spin_changed(self, value: int) -> None:
        paths = list(self._workspace.image_paths)
        if not paths:
            return
        idx = max(0, min(int(value), len(paths)) - 1)
        target_item = self.image_list.item(idx)
        if target_item is not None:
            self.image_list.setCurrentItem(target_item)

    def _frame_nav_previous(self) -> None:
        if not hasattr(self, "frame_nav_spin"):
            return
        self.frame_nav_spin.setValue(max(1, self.frame_nav_spin.value() - 1))

    def _frame_nav_next(self) -> None:
        if not hasattr(self, "frame_nav_spin"):
            return
        total = len(self._workspace.image_paths)
        if not total:
            return
        self.frame_nav_spin.setValue(min(total, self.frame_nav_spin.value() + 1))

    def _on_image_selection_changed(self) -> None:
        rows = sorted({self.image_list.row(i) for i in self.image_list.selectedItems()})
        paths = list(self._workspace.image_paths)
        if len(rows) == 1 and paths:
            with QSignalBlocker(self.frame_nav_spin):
                self.frame_nav_spin.setValue(min(max(1, rows[0] + 1), len(paths)))

    def _on_vector_item_navigate_request(self, item: QListWidgetItem) -> None:
        """Jump to the matching image frame (Files → Images) after an explicit click."""

        if item is None:
            return
        if time.monotonic() < self._vectors_list_ignore_navigate_until:
            return
        tip = item.toolTip()
        if not tip:
            return
        stem = Path(tip).stem.lower()
        image_path = self._image_path_for_cif_stem(stem)
        if not image_path:
            self._append_log(self._tr("vector_row_no_matching_image_loaded_log"))
            return
        image_item = self._find_image_list_item(image_path)
        if image_item is None:
            return
        if hasattr(self, "sidebar_list_mode_combo"):
            with QSignalBlocker(self.sidebar_list_mode_combo):
                self.sidebar_list_mode_combo.setCurrentIndex(0)
        if hasattr(self, "sidebar_list_stack"):
            self.sidebar_list_stack.setCurrentIndex(0)
        self.image_list.blockSignals(True)
        self.image_list.setCurrentItem(image_item)
        self.image_list.blockSignals(False)

    def _select_input_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_input_directory_dialog"),
            self.input_dir_edit.text(),
        )
        if path:
            self.set_input_directory(path)

    def _select_cif_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_cif_directory_dialog"),
            self.cif_dir_edit.text(),
        )
        if path:
            self.set_cif_directory(path)

    def _select_output_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_output_directory_dialog"),
            self.output_dir_edit.text(),
        )
        if path:
            self.set_output_directory(path)

    def _select_dataset_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._tr("select_dataset_directory_dialog"),
            self.dataset_dir_edit.text(),
        )
        if path:
            self.set_dataset_directory(path)

    def _apply_input_directory_edit(self) -> None:
        path = self.input_dir_edit.text().strip()
        if path:
            self.set_input_directory(path)
        else:
            self._workspace.replace_image_selection([], is_supported_image=is_image_path)
            self.image_list.clear()
            self._rebuild_thumbnail_grid()
            self._rebuild_vector_list()
            self._refresh_image_list_item_states()
            self._sync_frame_navigation_controls()
            self._sync_current_state_views()
            self._save_persisted_paths()

    def _apply_cif_directory_edit(self) -> None:
        path = self.cif_dir_edit.text().strip()
        if path:
            self.set_cif_directory(path)
        else:
            self._workspace.clear_cif_index()
            self._save_persisted_paths()
            self._rebuild_vector_list()
            self._refresh_image_list_item_states()
            if self._workspace.current_image_path:
                try:
                    self.load_image(self._workspace.current_image_path)
                except Exception as exc:
                    self._append_log(self._tr("reload_with_cif_failed_log", error=exc))

    def _apply_output_directory_edit(self) -> None:
        self.set_output_directory(self.output_dir_edit.text().strip())

    def _apply_dataset_directory_edit(self) -> None:
        self.set_dataset_directory(self.dataset_dir_edit.text().strip())

    def _choose_external_color(self) -> None:
        self._choose_color("external_color", self.external_color_button)

    def _choose_hole_color(self) -> None:
        self._choose_color("hole_color", self.hole_color_button)

    def _choose_selected_color(self) -> None:
        self._choose_color("selected_color", self.selected_color_button)

    def _choose_conductor_hover_highlight_color(self) -> None:
        self._choose_color("conductor_hover_highlight_color", self.conductor_hover_highlight_color_button)

    def _choose_vertex_color(self) -> None:
        self._choose_color("vertex_color", self.vertex_color_button)

    def _choose_color(self, attribute_name: str, button: QPushButton) -> None:
        initial = QColor(getattr(self._display_settings, attribute_name))
        color = QColorDialog.getColor(initial, self, self._tr("select_color_dialog_title"))
        if not color.isValid():
            return
        value = color.name(QColor.NameFormat.HexRgb)
        setattr(self._display_settings, attribute_name, value)
        self._update_color_button(button, value)
        self._apply_display_settings()

    def _apply_display_settings(self) -> None:
        if hasattr(self, "line_width_spin"):
            self._display_settings.line_width = float(self.line_width_spin.value())
            self._display_settings.vertex_size = float(self.vertex_size_spin.value())
            self._display_settings.fill_opacity = float(self.fill_opacity_spin.value())
            self._display_settings.show_vertices = bool(self.show_vertices_checkbox.isChecked())
            self._display_settings.show_labels = bool(self.show_labels_checkbox.isChecked())
            if hasattr(self, "polygon_editor"):
                self.polygon_editor.set_display_settings(self._display_settings)
                if hasattr(self, "random_object_colors_checkbox"):
                    self.polygon_editor.set_random_object_colors_enabled(self.random_object_colors_checkbox.isChecked())
        self._apply_vector_geometry_editor_config()
        self._save_persisted_display_settings()

    def _on_neighbor_display_settings_changed(self, *_args) -> None:
        self._sync_neighbor_frames()
        self._configure_thumbnail_grid_geometry()
        self._save_persisted_display_settings()

    def _refresh_extra_layers_list(self) -> None:
        if not hasattr(self, "extra_layers_list"):
            return
        current_row = self.extra_layers_list.currentRow()
        self.extra_layers_list.clear()
        for layer in self._extra_layers:
            name = str(layer.get("name", "Layer"))
            visible = bool(layer.get("visible", True))
            pixmap = layer.get("pixmap")
            loaded = isinstance(pixmap, QPixmap) and not pixmap.isNull()
            prefix = "[x] " if visible else "[ ] "
            if not loaded:
                prefix = "[!] "
            item = QListWidgetItem(prefix + name)
            item.setToolTip(str(layer.get("path", "")))
            self.extra_layers_list.addItem(item)
        if self._extra_layers:
            self.extra_layers_list.setCurrentRow(max(0, min(current_row, len(self._extra_layers) - 1)))
        else:
            self._on_extra_layer_selected(-1)

    def _sync_extra_layers(self) -> None:
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_extra_layers(self._extra_layers)

    def _load_extra_layers(self) -> None:
        file_paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Выберите изображения слоев" if self._ui_language == "ru" else "Select layer images",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All files (*)",
        )
        for file_path in file_paths:
            layer = self._extra_layer_from_path(file_path)
            if layer is not None:
                self._extra_layers.append(layer)
        self._refresh_extra_layers_list()
        self._sync_extra_layers()

    def _browse_selected_extra_layer_path(self) -> None:
        current_path = ""
        row = self.extra_layers_list.currentRow() if hasattr(self, "extra_layers_list") else -1
        if 0 <= row < len(self._extra_layers):
            current_path = str(self._extra_layers[row].get("path", ""))
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение слоя" if self._ui_language == "ru" else "Select layer image",
            str(Path(current_path).parent) if current_path else "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All files (*)",
        )
        if not file_path:
            return
        self.extra_layer_path_edit.setText(file_path)
        self._on_extra_layer_path_changed()

    def _on_extra_layer_path_changed(self) -> None:
        path = self.extra_layer_path_edit.text().strip()
        if not path:
            return
        layer = self._extra_layer_from_path(path)
        if layer is None:
            return
        row = self.extra_layers_list.currentRow() if hasattr(self, "extra_layers_list") else -1
        if row < 0 or row >= len(self._extra_layers):
            self._extra_layers.append(layer)
            row = len(self._extra_layers) - 1
        else:
            existing = self._extra_layers[row]
            existing["name"] = layer["name"]
            existing["path"] = layer["path"]
            existing["pixmap"] = layer["pixmap"]
        self._refresh_extra_layers_list()
        self.extra_layers_list.setCurrentRow(row)
        self._sync_extra_layers()

    def _extra_layer_from_path(self, file_path: str) -> dict[str, object] | None:
        path = Path(file_path.strip().strip("\"'")).expanduser()
        if not path.is_file():
            self._append_log(
                self._tr(
                    "extra_layer_missing_file_log",
                    "Файл слоя не найден: {path}" if self._ui_language == "ru" else "Layer file not found: {path}",
                    path=file_path,
                )
            )
            return None
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._append_log(
                self._tr(
                    "extra_layer_load_failed_log",
                    "Не удалось загрузить изображение слоя: {path}"
                    if self._ui_language == "ru"
                    else "Failed to load layer image: {path}",
                    path=str(path),
                )
            )
            return None
        return {
            "name": path.name,
            "path": str(path),
            "pixmap": pixmap,
            "visible": True,
            "opacity": 0.35,
            "dx": 0.0,
            "dy": 0.0,
        }

    def _remove_selected_extra_layer(self) -> None:
        row = self.extra_layers_list.currentRow() if hasattr(self, "extra_layers_list") else -1
        if row < 0 or row >= len(self._extra_layers):
            return
        self._extra_layers.pop(row)
        self._refresh_extra_layers_list()
        self._sync_extra_layers()

    def _on_extra_layer_selected(self, row: int) -> None:
        blockers = [
            QSignalBlocker(self.extra_layer_path_edit),
            QSignalBlocker(self.extra_layer_visible_checkbox),
            QSignalBlocker(self.extra_layer_opacity_spin),
            QSignalBlocker(self.extra_layer_dx_spin),
            QSignalBlocker(self.extra_layer_dy_spin),
        ]
        if row < 0 or row >= len(self._extra_layers):
            try:
                self.extra_layer_path_edit.clear()
                self.extra_layer_visible_checkbox.setChecked(False)
                self.extra_layer_opacity_spin.setValue(0.35)
                self.extra_layer_dx_spin.setValue(0.0)
                self.extra_layer_dy_spin.setValue(0.0)
            finally:
                del blockers
            return
        layer = self._extra_layers[row]
        try:
            self.extra_layer_path_edit.setText(str(layer.get("path", "")))
            self.extra_layer_visible_checkbox.setChecked(bool(layer.get("visible", True)))
            self.extra_layer_opacity_spin.setValue(float(layer.get("opacity", 0.35) or 0.35))
            self.extra_layer_dx_spin.setValue(float(layer.get("dx", 0.0) or 0.0))
            self.extra_layer_dy_spin.setValue(float(layer.get("dy", 0.0) or 0.0))
        finally:
            del blockers

    def _on_extra_layer_controls_changed(self, *_args) -> None:
        row = self.extra_layers_list.currentRow() if hasattr(self, "extra_layers_list") else -1
        if row < 0 or row >= len(self._extra_layers):
            return
        layer = self._extra_layers[row]
        layer["visible"] = self.extra_layer_visible_checkbox.isChecked()
        layer["opacity"] = float(self.extra_layer_opacity_spin.value())
        layer["dx"] = float(self.extra_layer_dx_spin.value())
        layer["dy"] = float(self.extra_layer_dy_spin.value())
        self._refresh_extra_layers_list()
        self.extra_layers_list.setCurrentRow(row)
        self._sync_extra_layers()

    def _auto_apply_pipeline(self) -> None:
        if not self._workspace.current_image_path:
            return
        if hasattr(self, "auto_apply_checkbox") and not self.auto_apply_checkbox.isChecked():
            return
        self._abort_in_flight_interactive_processing(preview=True, prepared=True)
        self.process_current_image(debounced=False)

    def _try_extract_if_recognition_enabled(self) -> None:
        if not hasattr(self, "recognition_mode_combo"):
            return
        if str(self.recognition_mode_combo.currentData() or "") == "disabled":
            return
        current_path = self._workspace.current_image_path
        if not current_path:
            return
        state = self._workspace.current_state
        if state is None or state.image_path != current_path or state.preprocessed_image is None:
            return
        if state.pipeline_config != self.get_pipeline():
            return
        self.process_current_image(debounced=False)

    def _start_auto_tune_from_reference(self) -> None:
        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        reference_polygons = self.get_polygons()

        if current_state is None or current_state.source_image is None or current_image_path is None:
            self._append_log(
                self._tr(
                    "no_image_selected_log",
                    "Изображение не выбрано." if self._ui_language == "ru" else "No image selected.",
                )
            )
            return
        if not reference_polygons:
            self._append_log(
                self._tr(
                    "auto_tune_no_reference_log",
                    "Для автоподбора сначала нарисуйте эталонный полигон или область."
                    if self._ui_language == "ru"
                    else "Draw at least one reference polygon before running auto-fit.",
                )
            )
            return
        if self._auto_tune_running_request_id is not None:
            self._append_log(
                self._tr(
                    "auto_tune_already_running_log",
                    "Автоподбор уже выполняется." if self._ui_language == "ru" else "Auto-fit is already running.",
                )
            )
            return

        self._auto_tune_request_serial += 1
        request_id = self._auto_tune_request_serial
        self._auto_tune_running_request_id = request_id
        self._append_log(
            self._tr(
                "auto_tune_started_log",
                "Запущен автоподбор по {count} полигонам."
                if self._ui_language == "ru"
                else "Auto-fit started using {count} reference polygons.",
                count=len(reference_polygons),
            )
        )
        worker = AutoTuneRunnable(
            request_id=request_id,
            image_path=current_image_path,
            source_image=current_state.source_image,
            reference_polygons=reference_polygons,
        )
        worker.signals.result.connect(self._on_auto_tune_result)
        worker.signals.error.connect(self._on_auto_tune_error)
        worker.signals.finished.connect(self._on_auto_tune_finished)
        self._auto_tune_thread_pool.start(worker)
        self._refresh_busy_indicator()

    def _apply_auto_tune_result(self, result: AutoTuneResult) -> None:
        self._pipeline = PreprocessingPipeline.from_dict(result.pipeline_config)
        self._populate_pipeline_list()
        self._set_extraction_settings(result.contour_settings)
        self.process_current_image()

    def _set_extraction_settings(self, settings: ContourExtractionSettings) -> None:
        blockers = [
            QSignalBlocker(self.retrieval_mode_combo),
            QSignalBlocker(self.approximation_mode_combo),
            QSignalBlocker(self.epsilon_spin),
            QSignalBlocker(self.epsilon_slider),
            QSignalBlocker(self.epsilon_relative_checkbox),
            QSignalBlocker(self.min_area_spin),
            QSignalBlocker(self.max_area_spin),
            QSignalBlocker(self.min_perimeter_spin),
            QSignalBlocker(self.min_points_spin),
            QSignalBlocker(self.min_polygon_width_spin),
            QSignalBlocker(self.max_perimeter_spin),
            QSignalBlocker(self.min_bbox_width_spin),
            QSignalBlocker(self.max_bbox_width_spin),
            QSignalBlocker(self.min_bbox_height_spin),
            QSignalBlocker(self.max_bbox_height_spin),
            QSignalBlocker(self.min_aspect_ratio_spin),
            QSignalBlocker(self.max_aspect_ratio_spin),
            QSignalBlocker(self.exclude_border_touching_checkbox),
            QSignalBlocker(self.min_solidity_spin),
            QSignalBlocker(self.min_extent_spin),
            QSignalBlocker(self.min_polygon_angle_spin),
            QSignalBlocker(self.conductor_gradient_checkbox),
            QSignalBlocker(self.conductor_gradient_min_strength_spin),
            QSignalBlocker(self.conductor_gradient_band_radius_spin),
            QSignalBlocker(self.via_size_mode_combo),
            QSignalBlocker(self.via_search_mode_combo),
            QSignalBlocker(self.via_white_range_checkbox),
            QSignalBlocker(self.via_white_range_min_spin),
            QSignalBlocker(self.via_white_range_max_spin),
            QSignalBlocker(self.via_black_range_checkbox),
            QSignalBlocker(self.via_black_range_min_spin),
            QSignalBlocker(self.via_black_range_max_spin),
            QSignalBlocker(self.via_min_score_spin),
            QSignalBlocker(self.via_min_contrast_spin),
            QSignalBlocker(self.via_min_edge_coverage_spin),
            QSignalBlocker(self.via_spot_line_suppression_spin),
            QSignalBlocker(self.via_template_min_score_spin),
            QSignalBlocker(self.debug_candidates_checkbox),
            QSignalBlocker(self.via_roundness_spin),
            QSignalBlocker(self.min_via_width_spin),
            QSignalBlocker(self.max_via_width_spin),
            QSignalBlocker(self.min_via_height_spin),
            QSignalBlocker(self.max_via_height_spin),
            QSignalBlocker(self.min_hierarchy_depth_spin),
            QSignalBlocker(self.max_hierarchy_depth_spin),
            QSignalBlocker(self.max_hole_area_ratio_spin),
            QSignalBlocker(self.advanced_extraction_checkbox),
        ]
        for _mw in (
            "metal_preset_combo",
            "metal_sensitivity_slider",
            "metal_min_width_spin",
            "metal_max_width_spin",
            "metal_min_length_spin",
            "metal_use_wide_gradient_checkbox",
            "metal_segmentation_method_combo",
            "metal_sensitivity_combo",
            "metal_show_conductors_checkbox",
            "metal_show_rejected_checkbox",
            "metal_show_suspicious_checkbox",
            "metal_show_border_checkbox",
            "metal_show_mask_checkbox",
            "metal_debug_visual_combo",
            "metal_overlay_opacity_spin",
            "metal_min_area_spin",
            "metal_max_area_spin",
            "metal_min_perimeter_spin",
            "metal_max_perimeter_spin",
            "metal_epsilon_spin",
            "metal_min_points_spin",
            "metal_min_angle_spin",
            "metal_approximation_checkbox",
            "metal_hierarchy_combo",
            "metal_allowed_angles_combo",
            "metal_angle_tolerance_spin",
            "metal_straightness_spin",
            "metal_t_junction_checkbox",
            "metal_border_handling_combo",
            "metal_validity_checkbox",
            "metal_morph_close_spin",
            "metal_morph_open_spin",
            "metal_wide_grad_radius_spin",
            "metal_wide_grad_conf_spin",
            "metal_wide_grad_pair_len_spin",
            "metal_wide_grad_parallel_spin",
            "metal_wide_grad_gap_spin",
            "metal_wide_grad_overlap_spin",
            "metal_advanced_group",
        ):
            _w = getattr(self, _mw, None)
            if _w is not None:
                blockers.append(QSignalBlocker(_w))
        if hasattr(self, "recognition_mode_combo"):
            blockers.append(QSignalBlocker(self.recognition_mode_combo))
        if hasattr(self, "via_search_sensitivity_combo"):
            blockers.append(QSignalBlocker(self.via_search_sensitivity_combo))
        if hasattr(self, "via_show_detected_checkbox"):
            blockers.append(QSignalBlocker(self.via_show_detected_checkbox))
        if hasattr(self, "via_debug_gradient_map_checkbox"):
            blockers.append(QSignalBlocker(self.via_debug_gradient_map_checkbox))
        try:
            prof = str(getattr(settings, "extraction_profile", "conductors") or "conductors")
            rm = normalize_recognition_mode(getattr(settings, "recognition_mode", "conductors"))
            if prof == "vias" or (getattr(settings, "object_type", "conductor") == "via" and rm != "disabled"):
                rdata = "via"
            elif rm == "disabled":
                rdata = "disabled"
            else:
                rdata = "conductors"
            if hasattr(self, "recognition_mode_combo"):
                ridx = self.recognition_mode_combo.findData(rdata)
                if ridx >= 0:
                    self.recognition_mode_combo.setCurrentIndex(ridx)
            if hasattr(self, "recognition_stack"):
                if rdata == "via":
                    self.recognition_stack.setVisible(False)
                else:
                    self.recognition_stack.setVisible(True)
                    self.recognition_stack.setCurrentIndex(0 if rdata == "disabled" else 1)
            self._active_extraction_profile = "vias" if rdata == "via" else "conductors"
            retrieval_index = self.retrieval_mode_combo.findData(settings.retrieval_mode)
            if retrieval_index >= 0:
                self.retrieval_mode_combo.setCurrentIndex(retrieval_index)
            approximation_index = self.approximation_mode_combo.findData(settings.approximation_mode)
            if approximation_index >= 0:
                self.approximation_mode_combo.setCurrentIndex(approximation_index)
            self.epsilon_spin.setValue(float(settings.epsilon))
            if hasattr(self, "epsilon_slider"):
                self.epsilon_slider.setValue(min(1000, max(0, round(float(settings.epsilon) * 100.0))))
            self.epsilon_relative_checkbox.setChecked(bool(settings.epsilon_relative))
            self.min_area_spin.setValue(float(settings.min_area))
            self.max_area_spin.setValue(0.0 if settings.max_area is None else float(settings.max_area))
            self.min_perimeter_spin.setValue(float(settings.min_perimeter))
            self.max_perimeter_spin.setValue(0.0 if settings.max_perimeter is None else float(settings.max_perimeter))
            self.min_points_spin.setValue(int(settings.min_points))
            self.min_polygon_width_spin.setValue(float(getattr(settings, "min_polygon_width_px", 0.0) or 0.0))
            self.min_bbox_width_spin.setValue(int(settings.min_bbox_width))
            self.max_bbox_width_spin.setValue(0 if settings.max_bbox_width is None else int(settings.max_bbox_width))
            self.min_bbox_height_spin.setValue(int(settings.min_bbox_height))
            self.max_bbox_height_spin.setValue(0 if settings.max_bbox_height is None else int(settings.max_bbox_height))
            self.min_aspect_ratio_spin.setValue(float(settings.min_aspect_ratio))
            self.max_aspect_ratio_spin.setValue(
                0.0 if settings.max_aspect_ratio is None else float(settings.max_aspect_ratio)
            )
            self.exclude_border_touching_checkbox.setChecked(bool(settings.exclude_border_touching))
            self.min_solidity_spin.setValue(float(settings.min_solidity))
            self.min_extent_spin.setValue(float(settings.min_extent))
            self.min_polygon_angle_spin.setValue(float(settings.min_polygon_angle))
            self.conductor_gradient_checkbox.setChecked(bool(settings.conductor_gradient_enabled))
            self.conductor_gradient_min_strength_spin.setValue(float(settings.conductor_gradient_min_strength))
            self.conductor_gradient_band_radius_spin.setValue(int(settings.conductor_gradient_band_radius))
            via_size_mode_index = self.via_size_mode_combo.findData(normalize_via_size_mode(settings.via_size_mode))
            if via_size_mode_index >= 0:
                self.via_size_mode_combo.setCurrentIndex(via_size_mode_index)
            via_search_mode_index = self.via_search_mode_combo.findData(
                normalize_via_search_mode(settings.via_search_mode)
            )
            if via_search_mode_index >= 0:
                self.via_search_mode_combo.setCurrentIndex(via_search_mode_index)
            if hasattr(self, "via_diameter_size_mode_combo"):
                _di = self.via_diameter_size_mode_combo.findData(normalize_via_size_mode(settings.via_size_mode))
                if _di >= 0:
                    self.via_diameter_size_mode_combo.setCurrentIndex(_di)
            if hasattr(self, "via_heuristic_polarity_combo"):
                _po = str(getattr(settings, "via_heuristic_polarity", "auto") or "auto")
                _pidx = self.via_heuristic_polarity_combo.findData(_po)
                if _pidx >= 0:
                    self.via_heuristic_polarity_combo.setCurrentIndex(_pidx)
            if hasattr(self, "via_fixed_diameters_edit"):
                self.via_fixed_diameters_edit.setText(
                    str(getattr(settings, "via_fixed_diameters_text", "6, 8, 10") or "6, 8, 10")
                )
            if hasattr(self, "via_template_nms_distance_spin"):
                self.via_template_nms_distance_spin.setValue(
                    int(getattr(settings, "via_template_nms_distance", 4) or 4)
                )
            if hasattr(self, "via_template_scale_min_spin"):
                self.via_template_scale_min_spin.setValue(
                    float(getattr(settings, "via_template_scale_min", 0.9) or 0.9)
                )
            if hasattr(self, "via_template_scale_max_spin"):
                self.via_template_scale_max_spin.setValue(
                    float(getattr(settings, "via_template_scale_max", 1.1) or 1.1)
                )
            if hasattr(self, "via_template_scale_step_spin"):
                self.via_template_scale_step_spin.setValue(
                    float(getattr(settings, "via_template_scale_step", 0.1) or 0.1)
                )
            if hasattr(self, "heuristic_background_sigma_spin"):
                self.heuristic_background_sigma_spin.setValue(
                    float(getattr(settings, "heuristic_background_sigma", 25.0) or 25.0)
                )
            if hasattr(self, "heuristic_analysis_window_scale_spin"):
                self.heuristic_analysis_window_scale_spin.setValue(
                    float(getattr(settings, "heuristic_analysis_window_scale", 3.0) or 3.0)
                )
            if hasattr(self, "heuristic_min_center_contrast_spin"):
                self.heuristic_min_center_contrast_spin.setValue(
                    float(getattr(settings, "heuristic_min_center_contrast", 6.0) or 0.0)
                )
            if hasattr(self, "heuristic_min_peak_prominence_spin"):
                self.heuristic_min_peak_prominence_spin.setValue(
                    float(getattr(settings, "heuristic_min_peak_prominence", 4.0) or 0.0)
                )
            if hasattr(self, "heuristic_min_compactness_spin"):
                self.heuristic_min_compactness_spin.setValue(
                    float(getattr(settings, "heuristic_min_compactness", 0.12) or 0.0)
                )
            if hasattr(self, "heuristic_max_elongation_spin"):
                self.heuristic_max_elongation_spin.setValue(
                    float(getattr(settings, "heuristic_max_elongation", 3.2) or 3.2)
                )
            if hasattr(self, "heuristic_line_penalty_spin"):
                self.heuristic_line_penalty_spin.setValue(
                    float(getattr(settings, "heuristic_line_penalty_scale", 1.0) or 1.0)
                )
            if hasattr(self, "heuristic_border_penalty_spin"):
                self.heuristic_border_penalty_spin.setValue(
                    float(getattr(settings, "heuristic_border_penalty_scale", 1.0) or 1.0)
                )
            if hasattr(self, "heuristic_local_binarize_percentile_spin"):
                self.heuristic_local_binarize_percentile_spin.setValue(
                    float(getattr(settings, "heuristic_local_binarize_percentile", 88.0) or 88.0)
                )
            if hasattr(self, "heuristic_min_abs_peak_spin"):
                self.heuristic_min_abs_peak_spin.setValue(
                    float(getattr(settings, "heuristic_min_abs_peak", 0.0) or 0.0)
                )
            if hasattr(self, "heuristic_use_bilateral_checkbox"):
                self.heuristic_use_bilateral_checkbox.setChecked(
                    bool(getattr(settings, "heuristic_use_bilateral", False))
                )
            if hasattr(self, "bright_via_mode_stack") and hasattr(self, "via_search_mode_combo"):
                _ist = self.via_search_mode_combo.currentData() == VIA_SEARCH_MODE_TEMPLATE
                self.bright_via_mode_stack.setCurrentIndex(1 if _ist else 0)
            self.via_white_range_checkbox.setChecked(bool(settings.via_white_range_enabled))
            self.via_white_range_min_spin.setValue(int(settings.via_white_range_min))
            self.via_white_range_max_spin.setValue(int(settings.via_white_range_max))
            self.via_black_range_checkbox.setChecked(bool(settings.via_black_range_enabled))
            self.via_black_range_min_spin.setValue(int(settings.via_black_range_min))
            self.via_black_range_max_spin.setValue(int(settings.via_black_range_max))
            self.via_min_score_spin.setValue(float(settings.via_min_score))
            self.via_min_contrast_spin.setValue(float(settings.via_min_contrast))
            self.via_min_edge_coverage_spin.setValue(float(settings.via_min_edge_coverage))
            self.via_template_min_score_spin.setValue(float(settings.via_template_min_score))
            self.via_spot_line_suppression_spin.setValue(float(settings.via_spot_line_suppression))
            self.bright_via_diameter_min_spin.setValue(int(settings.bright_via_diameter_min))
            self.bright_via_diameter_max_spin.setValue(int(settings.bright_via_diameter_max))
            self.bright_via_clahe_clip_spin.setValue(float(settings.bright_via_clahe_clip_limit))
            self.bright_via_clahe_tile_spin.setValue(int(settings.bright_via_clahe_tile_grid_size))
            self.bright_via_median_kernel_spin.setValue(int(settings.bright_via_median_blur_kernel))
            self.bright_via_tophat_kernel_spin.setValue(int(settings.bright_via_tophat_kernel_size))
            self.bright_via_dog_small_spin.setValue(float(settings.bright_via_dog_sigma_small))
            self.bright_via_dog_large_spin.setValue(float(settings.bright_via_dog_sigma_large))
            self.bright_via_threshold_percentile_spin.setValue(float(settings.bright_via_threshold_percentile))
            combine_index = self.bright_via_mask_combine_combo.findData(settings.bright_via_mask_combine_mode)
            if combine_index >= 0:
                self.bright_via_mask_combine_combo.setCurrentIndex(combine_index)
            self.bright_via_min_area_factor_spin.setValue(float(settings.bright_via_min_area_factor))
            self.bright_via_max_area_factor_spin.setValue(float(settings.bright_via_max_area_factor))
            self.bright_via_min_circularity_spin.setValue(float(settings.bright_via_min_circularity))
            self.bright_via_min_aspect_spin.setValue(float(settings.bright_via_min_aspect))
            self.bright_via_max_aspect_spin.setValue(float(settings.bright_via_max_aspect))
            self.bright_via_bright_center_score_spin.setValue(float(settings.bright_via_bright_center_min_score))
            metal_index = self.bright_via_metal_constraint_combo.findData(
                _normalize_bright_via_metal_constraint_mode(settings.bright_via_metal_constraint_mode)
            )
            if metal_index >= 0:
                self.bright_via_metal_constraint_combo.setCurrentIndex(metal_index)
            self.bright_via_metal_fraction_spin.setValue(float(settings.bright_via_metal_fraction_min))
            self.bright_via_max_radial_asymmetry_spin.setValue(float(settings.bright_via_max_radial_asymmetry))
            self.bright_via_max_edge_likeness_spin.setValue(float(settings.bright_via_max_edge_likeness))
            self.bright_via_max_line_likeness_spin.setValue(float(settings.bright_via_max_line_likeness))
            self.bright_via_nms_distance_spin.setValue(int(settings.bright_via_nms_distance))
            self.bright_via_min_final_score_spin.setValue(float(settings.bright_via_min_final_score))
            self.bright_via_show_rejected_checkbox.setChecked(bool(settings.bright_via_show_rejected))
            self.bright_via_hard_asym_checkbox.setChecked(bool(settings.bright_via_hard_reject_on_asymmetry))
            self.bright_via_hard_edge_checkbox.setChecked(bool(settings.bright_via_hard_reject_on_edge))
            self.bright_via_hard_line_checkbox.setChecked(bool(settings.bright_via_hard_reject_on_line))
            self._via_template_images = self._normalize_via_template_images(settings.via_template_images)
            self._refresh_via_template_list()
            self.debug_candidates_checkbox.setChecked(bool(settings.debug_enabled))
            if hasattr(self, "via_debug_gradient_map_checkbox"):
                self.via_debug_gradient_map_checkbox.setChecked(bool(settings.debug_gradient_map_enabled))
            if hasattr(self, "via_show_detected_checkbox"):
                self.via_show_detected_checkbox.setChecked(bool(getattr(settings, "via_display_show_detected", True)))
            if hasattr(self, "via_search_sensitivity_combo"):
                vs = str(getattr(settings, "via_search_sensitivity", "medium") or "medium")
                vs_idx = self.via_search_sensitivity_combo.findData(vs)
                if vs_idx >= 0:
                    self.via_search_sensitivity_combo.setCurrentIndex(vs_idx)
            self.via_roundness_spin.setValue(float(settings.via_min_roundness))
            self.min_via_width_spin.setValue(int(settings.min_via_width))
            self.max_via_width_spin.setValue(0 if settings.max_via_width is None else int(settings.max_via_width))
            self.min_via_height_spin.setValue(int(settings.min_via_height))
            self.max_via_height_spin.setValue(0 if settings.max_via_height is None else int(settings.max_via_height))
            self._suspend_fixed_via_updates = True
            self._clear_fixed_via_rows()
            for width, height in zip(settings.fixed_via_widths, settings.fixed_via_heights, strict=False):
                self._add_fixed_via_row(width=width, height=height)
            self._suspend_fixed_via_updates = False
            self.min_hierarchy_depth_spin.setValue(int(settings.min_hierarchy_depth))
            self.max_hierarchy_depth_spin.setValue(
                0 if settings.max_hierarchy_depth is None else int(settings.max_hierarchy_depth)
            )
            self.max_hole_area_ratio_spin.setValue(
                0.0 if settings.max_hole_area_ratio is None else float(settings.max_hole_area_ratio)
            )
            if hasattr(self, "metal_preset_combo"):
                mp = self.metal_preset_combo.findData(str(getattr(settings, "metal_preset", "standard") or "standard"))
                if mp >= 0:
                    self.metal_preset_combo.setCurrentIndex(mp)
                self.metal_sensitivity_slider.setValue(int(getattr(settings, "metal_sensitivity_0_100", 50)))
                if hasattr(self, "metal_sensitivity_value_label"):
                    self.metal_sensitivity_value_label.setText(str(self.metal_sensitivity_slider.value()))
                self.metal_min_width_spin.setValue(float(getattr(settings, "metal_min_trace_width_px", 8.0) or 8.0))
                mw = getattr(settings, "metal_max_trace_width_px", None)
                self.metal_max_width_spin.setValue(0.0 if mw is None else float(mw))
                self.metal_min_length_spin.setValue(
                    float(getattr(settings, "metal_min_trace_length_px", 8.0) or 8.0)
                )
                if hasattr(self, "metal_use_wide_gradient_checkbox"):
                    self.metal_use_wide_gradient_checkbox.setChecked(
                        bool(getattr(settings, "metal_use_wide_conductor_gradient", False))
                    )
                if hasattr(self, "metal_wide_grad_radius_spin"):
                    self.metal_wide_grad_radius_spin.setValue(
                        int(getattr(settings, "metal_wide_gradient_profile_radius_px", 8) or 8)
                    )
                if hasattr(self, "metal_wide_grad_conf_spin"):
                    self.metal_wide_grad_conf_spin.setValue(
                        float(getattr(settings, "metal_wide_gradient_min_direction_confidence", 0.15) or 0.15)
                    )
                if hasattr(self, "metal_wide_grad_pair_len_spin"):
                    self.metal_wide_grad_pair_len_spin.setValue(
                        float(getattr(settings, "metal_wide_gradient_min_pair_length_px", 24.0) or 24.0)
                    )
                if hasattr(self, "metal_wide_grad_parallel_spin"):
                    self.metal_wide_grad_parallel_spin.setValue(
                        float(getattr(settings, "metal_wide_gradient_parallel_tolerance_deg", 10.0) or 10.0)
                    )
                if hasattr(self, "metal_wide_grad_gap_spin"):
                    self.metal_wide_grad_gap_spin.setValue(
                        int(getattr(settings, "metal_wide_gradient_max_edge_gap_px", 5) or 5)
                    )
                if hasattr(self, "metal_wide_grad_overlap_spin"):
                    self.metal_wide_grad_overlap_spin.setValue(
                        float(getattr(settings, "metal_wide_gradient_min_overlap_ratio", 0.5) or 0.5)
                    )
                _smi = self.metal_segmentation_method_combo.findData(
                    str(getattr(settings, "metal_segmentation_method", "none") or "none")
                )
                if _smi >= 0:
                    self.metal_segmentation_method_combo.setCurrentIndex(_smi)
                _st = str(getattr(settings, "metal_sensitivity", "medium") or "medium")
                _sti = self.metal_sensitivity_combo.findData(_st)
                if _sti >= 0:
                    self.metal_sensitivity_combo.setCurrentIndex(_sti)
                self.metal_min_area_spin.setValue(float(getattr(settings, "metal_min_area", 60.0) or 60.0))
                ma = getattr(settings, "metal_max_area", None)
                self.metal_max_area_spin.setValue(0.0 if ma is None else float(ma))
                self.metal_min_perimeter_spin.setValue(
                    float(getattr(settings, "metal_min_perimeter", 32.0) or 32.0)
                )
                mp2 = getattr(settings, "metal_max_perimeter", None)
                self.metal_max_perimeter_spin.setValue(0.0 if mp2 is None else float(mp2))
                self.metal_epsilon_spin.setValue(float(settings.epsilon))
                self.metal_min_points_spin.setValue(int(settings.min_points))
                self.metal_min_angle_spin.setValue(float(settings.min_polygon_angle))
                self.metal_approximation_checkbox.setChecked(
                    bool(getattr(settings, "metal_approximation_enabled", True))
                )
                _hm = self.metal_hierarchy_combo.findData(
                    str(getattr(settings, "metal_hierarchy_mode", "full") or "full")
                )
                if _hm >= 0:
                    self.metal_hierarchy_combo.setCurrentIndex(_hm)
                _aa = str(getattr(settings, "metal_allowed_angles", "free") or "free")
                _aai = self.metal_allowed_angles_combo.findData(_aa)
                if _aai >= 0:
                    self.metal_allowed_angles_combo.setCurrentIndex(_aai)
                self.metal_angle_tolerance_spin.setValue(
                    float(getattr(settings, "metal_angle_tolerance_deg", 7.0) or 7.0)
                )
                self.metal_straightness_spin.setValue(
                    float(getattr(settings, "metal_min_straightness", 0.2) or 0.2)
                )
                self.metal_t_junction_checkbox.setChecked(
                    bool(getattr(settings, "metal_allow_t_junction", True))
                )
                _bh = str(getattr(settings, "metal_border_handling", "mark") or "mark")
                _bhi = self.metal_border_handling_combo.findData(_bh)
                if _bhi >= 0:
                    self.metal_border_handling_combo.setCurrentIndex(_bhi)
                self.metal_validity_checkbox.setChecked(
                    bool(getattr(settings, "metal_check_contour_validity", True))
                )
                self.metal_morph_close_spin.setValue(int(getattr(settings, "metal_morph_close_radius", 1) or 1))
                self.metal_morph_open_spin.setValue(int(getattr(settings, "metal_morph_open_radius", 0) or 0))
                self.metal_show_conductors_checkbox.setChecked(
                    bool(getattr(settings, "metal_display_show_conductors", True))
                )
                self.metal_show_rejected_checkbox.setChecked(
                    bool(getattr(settings, "metal_display_show_rejected", False))
                )
                self.metal_show_suspicious_checkbox.setChecked(
                    bool(getattr(settings, "metal_display_show_suspicious", True))
                )
                self.metal_show_border_checkbox.setChecked(
                    bool(getattr(settings, "metal_display_show_border_highlight", True))
                )
                self.metal_show_mask_checkbox.setChecked(bool(getattr(settings, "metal_display_show_mask", True)))
                _dv = str(getattr(settings, "metal_debug_visual", "overlay") or "overlay")
                _dvi = self.metal_debug_visual_combo.findData(_dv)
                if _dvi >= 0:
                    self.metal_debug_visual_combo.setCurrentIndex(_dvi)
                self.metal_overlay_opacity_spin.setValue(
                    float(getattr(settings, "metal_overlay_opacity", 0.45) or 0.45)
                )
            self._update_via_size_controls_state()
            self._update_via_threshold_controls_state()
            self._update_extraction_profile_controls_state()
        finally:
            self._suspend_fixed_via_updates = False
            self._ignore_extraction_profile_change = False
            del blockers

    def _current_contour_settings(self) -> ContourExtractionSettings:
        max_area = self.max_area_spin.value()
        max_perimeter = self.max_perimeter_spin.value()
        max_bbox_width = self.max_bbox_width_spin.value()
        max_bbox_height = self.max_bbox_height_spin.value()
        max_aspect_ratio = self.max_aspect_ratio_spin.value()
        max_via_width = self.max_via_width_spin.value()
        max_via_height = self.max_via_height_spin.value()
        raw_rec = (
            str(self.recognition_mode_combo.currentData() or "conductors")
            if hasattr(self, "recognition_mode_combo")
            else "conductors"
        )
        rec_mode = normalize_recognition_mode(raw_rec)
        if rec_mode == "via" and hasattr(self, "via_diameter_size_mode_combo"):
            via_size_mode = normalize_via_size_mode(self.via_diameter_size_mode_combo.currentData())
        else:
            via_size_mode = normalize_via_size_mode(self.via_size_mode_combo.currentData())
        via_search_mode_effective = normalize_via_search_mode(self.via_search_mode_combo.currentData())
        if rec_mode == "via":
            extraction_profile = "vias"
            object_type = "via"
            output_mode = "box"
            algorithm_backend = normalize_algorithm_backend("sem")
            via_search_mode_effective = normalize_via_search_mode(self.via_search_mode_combo.currentData())
        else:
            extraction_profile = "conductors"
            object_type = "conductor"
            output_mode = "polygon"
            algorithm_backend = normalize_algorithm_backend("legacy")
            via_search_mode_effective = normalize_via_search_mode(self.via_search_mode_combo.currentData())
        fixed_via_pairs = self._fixed_via_pairs()
        fixed_via_widths = [width for width, _height in fixed_via_pairs]
        fixed_via_heights = [height for _width, height in fixed_via_pairs]
        max_hierarchy_depth = self.max_hierarchy_depth_spin.value()
        max_hole_area_ratio = self.max_hole_area_ratio_spin.value()
        return ContourExtractionSettings(
            algorithm_backend=algorithm_backend,
            sem_noise_level="medium",
            extraction_profile=extraction_profile,
            object_type=object_type,
            output_mode=output_mode,
            retrieval_mode=str(self.retrieval_mode_combo.currentData() or self.retrieval_mode_combo.currentText()),
            approximation_mode=str(
                self.approximation_mode_combo.currentData() or self.approximation_mode_combo.currentText()
            ),
            epsilon=self.metal_epsilon_spin.value()
            if hasattr(self, "metal_epsilon_spin") and rec_mode != "via"
            else self.epsilon_spin.value(),
            epsilon_relative=self.epsilon_relative_checkbox.isChecked(),
            min_polygon_angle=self.metal_min_angle_spin.value()
            if hasattr(self, "metal_min_angle_spin") and rec_mode != "via"
            else self.min_polygon_angle_spin.value(),
            min_area=self.min_area_spin.value(),
            max_area=None if max_area <= 0 else max_area,
            min_perimeter=self.min_perimeter_spin.value(),
            min_points=self.metal_min_points_spin.value()
            if hasattr(self, "metal_min_points_spin") and rec_mode != "via"
            else self.min_points_spin.value(),
            max_perimeter=None if max_perimeter <= 0 else max_perimeter,
            min_bbox_width=self.min_bbox_width_spin.value(),
            max_bbox_width=None if max_bbox_width <= 0 else max_bbox_width,
            min_bbox_height=self.min_bbox_height_spin.value(),
            max_bbox_height=None if max_bbox_height <= 0 else max_bbox_height,
            min_aspect_ratio=self.min_aspect_ratio_spin.value(),
            max_aspect_ratio=None if max_aspect_ratio <= 0 else max_aspect_ratio,
            exclude_border_touching=self.exclude_border_touching_checkbox.isChecked(),
            min_solidity=self.min_solidity_spin.value(),
            min_extent=self.min_extent_spin.value(),
            min_polygon_width_px=self.min_polygon_width_spin.value(),
            conductor_gradient_enabled=False,
            conductor_gradient_min_strength=self.conductor_gradient_min_strength_spin.value()
            if hasattr(self, "conductor_gradient_min_strength_spin")
            else 18.0,
            conductor_gradient_band_radius=self.conductor_gradient_band_radius_spin.value()
            if hasattr(self, "conductor_gradient_band_radius_spin")
            else 3,
            via_size_mode=via_size_mode,
            via_search_mode=via_search_mode_effective,
            via_white_range_enabled=self.via_white_range_checkbox.isChecked(),
            via_white_range_min=self.via_white_range_min_spin.value(),
            via_white_range_max=self.via_white_range_max_spin.value(),
            via_black_range_enabled=self.via_black_range_checkbox.isChecked(),
            via_black_range_min=self.via_black_range_min_spin.value(),
            via_black_range_max=self.via_black_range_max_spin.value(),
            via_min_score=self.via_min_score_spin.value(),
            via_min_contrast=self.via_min_contrast_spin.value(),
            via_min_edge_coverage=self.via_min_edge_coverage_spin.value(),
            via_template_min_score=self.via_template_min_score_spin.value(),
            via_spot_line_suppression=self.via_spot_line_suppression_spin.value(),
            bright_via_diameter_min=self.bright_via_diameter_min_spin.value(),
            bright_via_diameter_max=self.bright_via_diameter_max_spin.value(),
            bright_via_clahe_clip_limit=self.bright_via_clahe_clip_spin.value(),
            bright_via_clahe_tile_grid_size=self.bright_via_clahe_tile_spin.value(),
            bright_via_median_blur_kernel=self.bright_via_median_kernel_spin.value(),
            bright_via_tophat_kernel_size=self.bright_via_tophat_kernel_spin.value(),
            bright_via_dog_sigma_small=self.bright_via_dog_small_spin.value(),
            bright_via_dog_sigma_large=self.bright_via_dog_large_spin.value(),
            bright_via_threshold_percentile=self.bright_via_threshold_percentile_spin.value(),
            bright_via_mask_combine_mode=str(self.bright_via_mask_combine_combo.currentData() or "OR"),
            bright_via_min_area_factor=self.bright_via_min_area_factor_spin.value(),
            bright_via_max_area_factor=self.bright_via_max_area_factor_spin.value(),
            bright_via_min_circularity=self.bright_via_min_circularity_spin.value(),
            bright_via_min_aspect=self.bright_via_min_aspect_spin.value(),
            bright_via_max_aspect=self.bright_via_max_aspect_spin.value(),
            bright_via_bright_center_min_score=self.bright_via_bright_center_score_spin.value(),
            bright_via_metal_constraint_mode=_normalize_bright_via_metal_constraint_mode(
                self.bright_via_metal_constraint_combo.currentData()
            ),
            bright_via_use_metal_mask=str(self.bright_via_metal_constraint_combo.currentData()) != "disabled",
            bright_via_metal_fraction_min=self.bright_via_metal_fraction_spin.value(),
            bright_via_max_radial_asymmetry=self.bright_via_max_radial_asymmetry_spin.value(),
            bright_via_max_edge_likeness=self.bright_via_max_edge_likeness_spin.value(),
            bright_via_max_line_likeness=self.bright_via_max_line_likeness_spin.value(),
            bright_via_nms_distance=self.bright_via_nms_distance_spin.value(),
            bright_via_min_final_score=self.bright_via_min_final_score_spin.value(),
            bright_via_show_rejected=self.bright_via_show_rejected_checkbox.isChecked(),
            bright_via_hard_reject_on_asymmetry=self.bright_via_hard_asym_checkbox.isChecked(),
            bright_via_hard_reject_on_edge=self.bright_via_hard_edge_checkbox.isChecked(),
            bright_via_hard_reject_on_line=self.bright_via_hard_line_checkbox.isChecked(),
            via_template_images=[template.copy() for template in self._via_template_images],
            via_template_nms_distance=self.via_template_nms_distance_spin.value()
            if hasattr(self, "via_template_nms_distance_spin")
            else 4,
            via_template_scale_min=self.via_template_scale_min_spin.value()
            if hasattr(self, "via_template_scale_min_spin")
            else 0.9,
            via_template_scale_max=self.via_template_scale_max_spin.value()
            if hasattr(self, "via_template_scale_max_spin")
            else 1.1,
            via_template_scale_step=self.via_template_scale_step_spin.value()
            if hasattr(self, "via_template_scale_step_spin")
            else 0.1,
            via_heuristic_polarity=str(
                self.via_heuristic_polarity_combo.currentData() or "auto"
            )
            if hasattr(self, "via_heuristic_polarity_combo")
            else "auto",
            via_fixed_diameters_text=str(self.via_fixed_diameters_edit.text() or "6, 8, 10")
            if hasattr(self, "via_fixed_diameters_edit")
            else "6, 8, 10",
            heuristic_background_sigma=self.heuristic_background_sigma_spin.value()
            if hasattr(self, "heuristic_background_sigma_spin")
            else 25.0,
            heuristic_analysis_window_scale=self.heuristic_analysis_window_scale_spin.value()
            if hasattr(self, "heuristic_analysis_window_scale_spin")
            else 3.0,
            heuristic_min_center_contrast=self.heuristic_min_center_contrast_spin.value()
            if hasattr(self, "heuristic_min_center_contrast_spin")
            else 6.0,
            heuristic_min_peak_prominence=self.heuristic_min_peak_prominence_spin.value()
            if hasattr(self, "heuristic_min_peak_prominence_spin")
            else 4.0,
            heuristic_min_compactness=self.heuristic_min_compactness_spin.value()
            if hasattr(self, "heuristic_min_compactness_spin")
            else 0.12,
            heuristic_max_elongation=self.heuristic_max_elongation_spin.value()
            if hasattr(self, "heuristic_max_elongation_spin")
            else 3.2,
            heuristic_line_penalty_scale=self.heuristic_line_penalty_spin.value()
            if hasattr(self, "heuristic_line_penalty_spin")
            else 1.0,
            heuristic_border_penalty_scale=self.heuristic_border_penalty_spin.value()
            if hasattr(self, "heuristic_border_penalty_spin")
            else 1.0,
            heuristic_local_binarize_percentile=self.heuristic_local_binarize_percentile_spin.value()
            if hasattr(self, "heuristic_local_binarize_percentile_spin")
            else 88.0,
            heuristic_min_abs_peak=self.heuristic_min_abs_peak_spin.value()
            if hasattr(self, "heuristic_min_abs_peak_spin")
            else 0.0,
            heuristic_use_bilateral=self.heuristic_use_bilateral_checkbox.isChecked()
            if hasattr(self, "heuristic_use_bilateral_checkbox")
            else False,
            debug_enabled=self.debug_candidates_checkbox.isChecked(),
            debug_gradient_map_enabled=(
                self.via_debug_gradient_map_checkbox.isChecked()
                if hasattr(self, "via_debug_gradient_map_checkbox")
                else self.debug_candidates_checkbox.isChecked()
            ),
            recognition_mode=raw_rec,
            via_search_sensitivity=str(
                self.via_search_sensitivity_combo.currentData() or "medium"
            )
            if hasattr(self, "via_search_sensitivity_combo")
            else "medium",
            via_display_show_detected=(
                self.via_show_detected_checkbox.isChecked()
                if hasattr(self, "via_show_detected_checkbox")
                else True
            ),
            via_display_show_candidates=self.debug_candidates_checkbox.isChecked(),
            metal_structural_pipeline=(raw_rec == "conductors"),
            metal_preset=str(self.metal_preset_combo.currentData() or "standard")
            if hasattr(self, "metal_preset_combo")
            else "standard",
            metal_segmentation_method=str(self.metal_segmentation_method_combo.currentData() or "none")
            if hasattr(self, "metal_segmentation_method_combo")
            else "none",
            metal_sensitivity=str(self.metal_sensitivity_combo.currentData() or "medium")
            if hasattr(self, "metal_sensitivity_combo")
            else "medium",
            metal_sensitivity_0_100=int(self.metal_sensitivity_slider.value())
            if hasattr(self, "metal_sensitivity_slider")
            else 50,
            metal_min_object_area=self.metal_min_area_spin.value()
            if hasattr(self, "metal_min_area_spin")
            else 60.0,
            metal_min_trace_width_px=float(self.metal_min_width_spin.value())
            if hasattr(self, "metal_min_width_spin")
            else 8.0,
            metal_max_trace_width_px=None
            if not hasattr(self, "metal_max_width_spin") or self.metal_max_width_spin.value() <= 0
            else float(self.metal_max_width_spin.value()),
            metal_min_trace_length_px=float(self.metal_min_length_spin.value())
            if hasattr(self, "metal_min_length_spin")
            else 8.0,
            metal_use_wide_conductor_gradient=(
                self.metal_use_wide_gradient_checkbox.isChecked()
                if hasattr(self, "metal_use_wide_gradient_checkbox")
                else False
            ),
            metal_wide_gradient_profile_radius_px=int(self.metal_wide_grad_radius_spin.value())
            if hasattr(self, "metal_wide_grad_radius_spin")
            else 8,
            metal_wide_gradient_min_direction_confidence=float(self.metal_wide_grad_conf_spin.value())
            if hasattr(self, "metal_wide_grad_conf_spin")
            else 0.15,
            metal_wide_gradient_min_pair_length_px=float(self.metal_wide_grad_pair_len_spin.value())
            if hasattr(self, "metal_wide_grad_pair_len_spin")
            else 24.0,
            metal_wide_gradient_parallel_tolerance_deg=float(self.metal_wide_grad_parallel_spin.value())
            if hasattr(self, "metal_wide_grad_parallel_spin")
            else 10.0,
            metal_wide_gradient_max_edge_gap_px=int(self.metal_wide_grad_gap_spin.value())
            if hasattr(self, "metal_wide_grad_gap_spin")
            else 5,
            metal_wide_gradient_min_overlap_ratio=float(self.metal_wide_grad_overlap_spin.value())
            if hasattr(self, "metal_wide_grad_overlap_spin")
            else 0.5,
            metal_allowed_angles=str(self.metal_allowed_angles_combo.currentData() or "free")
            if hasattr(self, "metal_allowed_angles_combo")
            else "free",
            metal_angle_tolerance_deg=float(self.metal_angle_tolerance_spin.value())
            if hasattr(self, "metal_angle_tolerance_spin")
            else 7.0,
            metal_min_straightness=float(self.metal_straightness_spin.value())
            if hasattr(self, "metal_straightness_spin")
            else 0.2,
            metal_allow_t_junction=self.metal_t_junction_checkbox.isChecked()
            if hasattr(self, "metal_t_junction_checkbox")
            else True,
            metal_border_handling=str(self.metal_border_handling_combo.currentData() or "mark")
            if hasattr(self, "metal_border_handling_combo")
            else "mark",
            metal_check_contour_validity=self.metal_validity_checkbox.isChecked()
            if hasattr(self, "metal_validity_checkbox")
            else True,
            metal_hierarchy_mode=str(self.metal_hierarchy_combo.currentData() or "full")
            if hasattr(self, "metal_hierarchy_combo")
            else "full",
            metal_min_area=self.metal_min_area_spin.value() if hasattr(self, "metal_min_area_spin") else 60.0,
            metal_max_area=None
            if not hasattr(self, "metal_max_area_spin") or self.metal_max_area_spin.value() <= 0
            else float(self.metal_max_area_spin.value()),
            metal_min_perimeter=float(self.metal_min_perimeter_spin.value())
            if hasattr(self, "metal_min_perimeter_spin")
            else 32.0,
            metal_max_perimeter=None
            if not hasattr(self, "metal_max_perimeter_spin") or self.metal_max_perimeter_spin.value() <= 0
            else float(self.metal_max_perimeter_spin.value()),
            metal_approximation_enabled=self.metal_approximation_checkbox.isChecked()
            if hasattr(self, "metal_approximation_checkbox")
            else True,
            metal_morph_close_radius=self.metal_morph_close_spin.value()
            if hasattr(self, "metal_morph_close_spin")
            else 1,
            metal_morph_open_radius=self.metal_morph_open_spin.value()
            if hasattr(self, "metal_morph_open_spin")
            else 0,
            metal_display_show_conductors=self.metal_show_conductors_checkbox.isChecked()
            if hasattr(self, "metal_show_conductors_checkbox")
            else True,
            metal_display_show_mask=self.metal_show_mask_checkbox.isChecked()
            if hasattr(self, "metal_show_mask_checkbox")
            else True,
            metal_display_show_contours=self.metal_show_conductors_checkbox.isChecked()
            if hasattr(self, "metal_show_conductors_checkbox")
            else True,
            metal_display_show_rejected=self.metal_show_rejected_checkbox.isChecked()
            if hasattr(self, "metal_show_rejected_checkbox")
            else False,
            metal_display_show_suspicious=self.metal_show_suspicious_checkbox.isChecked()
            if hasattr(self, "metal_show_suspicious_checkbox")
            else True,
            metal_display_show_border_highlight=self.metal_show_border_checkbox.isChecked()
            if hasattr(self, "metal_show_border_checkbox")
            else True,
            metal_debug_visual=str(self.metal_debug_visual_combo.currentData() or "overlay")
            if hasattr(self, "metal_debug_visual_combo")
            else "overlay",
            metal_overlay_opacity=float(self.metal_overlay_opacity_spin.value())
            if hasattr(self, "metal_overlay_opacity_spin")
            else 0.45,
            via_min_roundness=self.via_roundness_spin.value(),
            min_via_width=self.min_via_width_spin.value(),
            max_via_width=None if max_via_width <= 0 else max_via_width,
            min_via_height=self.min_via_height_spin.value(),
            max_via_height=None if max_via_height <= 0 else max_via_height,
            fixed_via_widths=fixed_via_widths,
            fixed_via_heights=fixed_via_heights,
            min_hierarchy_depth=self.min_hierarchy_depth_spin.value(),
            max_hierarchy_depth=None if max_hierarchy_depth <= 0 else max_hierarchy_depth,
            max_hole_area_ratio=None if max_hole_area_ratio <= 0 else max_hole_area_ratio,
        )

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
            self.via_group.setVisible(self._active_extraction_profile == "vias" and data == "disabled")
        self.polygon_editor.set_debug_candidates([])
        if hasattr(self, "_update_extraction_profile_controls_state"):
            self._update_extraction_profile_controls_state()
        self._on_extraction_settings_changed()

    def _on_via_search_sensitivity_changed(self, *_args) -> None:
        self._apply_via_search_sensitivity_profile()
        self._on_extraction_settings_changed()

    def _apply_via_search_sensitivity_profile(self) -> None:
        if not hasattr(self, "via_search_sensitivity_combo"):
            return
        level = str(self.via_search_sensitivity_combo.currentData() or "medium")
        profiles = {
            "low": (99.5, 8.0, 55.0, 0.40, True, True, True, True),
            "medium": (99.0, 6.0, 38.0, 0.30, False, False, False, False),
            "high": (98.0, 4.0, 32.0, 0.22, False, False, False, False),
        }
        pct, bright, final, circ, ha, he, hl, _ = profiles.get(level, profiles["medium"])
        blockers = [
            QSignalBlocker(self.bright_via_threshold_percentile_spin),
            QSignalBlocker(self.bright_via_bright_center_score_spin),
            QSignalBlocker(self.bright_via_min_final_score_spin),
            QSignalBlocker(self.bright_via_min_circularity_spin),
            QSignalBlocker(self.bright_via_hard_asym_checkbox),
            QSignalBlocker(self.bright_via_hard_edge_checkbox),
            QSignalBlocker(self.bright_via_hard_line_checkbox),
        ]
        try:
            self.bright_via_threshold_percentile_spin.setValue(pct)
            self.bright_via_bright_center_score_spin.setValue(bright)
            self.bright_via_min_final_score_spin.setValue(final)
            self.bright_via_min_circularity_spin.setValue(circ)
            self.bright_via_hard_asym_checkbox.setChecked(ha)
            self.bright_via_hard_edge_checkbox.setChecked(he)
            self.bright_via_hard_line_checkbox.setChecked(hl)
        finally:
            del blockers

    def _on_via_display_settings_changed(self, *_args) -> None:
        if hasattr(self, "polygon_editor") and hasattr(self, "via_show_detected_checkbox"):
            self.polygon_editor.set_polygon_category_visible("via", self.via_show_detected_checkbox.isChecked())
        self._on_extraction_settings_changed()

    def _on_metal_overlay_opacity_changed(self, value: float) -> None:
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_gradient_overlay_opacity(float(value))
        self._on_extraction_settings_changed()

    def _on_metal_sensitivity_slider_changed(self, value: int) -> None:
        if hasattr(self, "metal_sensitivity_value_label"):
            self.metal_sensitivity_value_label.setText(str(int(value)))
        self._on_extraction_settings_changed()

    def _metal_preset_table(self) -> dict[str, dict[str, float | int | str]]:
        return {
            "standard": {
                "sens": 50,
                "close": 1,
                "open": 0,
                "min_w": 8.0,
                "max_w": 0.0,
                "min_l": 8.0,
                "min_a": 60.0,
                "max_a": 0.0,
                "min_p": 32.0,
                "max_p": 0.0,
                "str": 0.2,
                "tol": 7.0,
                "tok": "medium",
                "angles": "free",
            },
            "dense": {
                "sens": 42,
                "close": 5,
                "open": 0,
                "min_w": 6.0,
                "max_w": 85.0,
                "min_l": 18.0,
                "min_a": 45.0,
                "max_a": 0.0,
                "min_p": 28.0,
                "max_p": 0.0,
                "str": 0.52,
                "tol": 9.0,
                "tok": "high",
            },
            "thin_traces": {
                "sens": 58,
                "close": 2,
                "open": 1,
                "min_w": 4.0,
                "max_w": 24.0,
                "min_l": 28.0,
                "min_a": 35.0,
                "max_a": 0.0,
                "min_p": 26.0,
                "max_p": 0.0,
                "str": 0.64,
                "tol": 6.0,
                "tok": "medium",
            },
            "noisy_sem": {
                "sens": 36,
                "close": 4,
                "open": 1,
                "min_w": 10.0,
                "max_w": 0.0,
                "min_l": 32.0,
                "min_a": 85.0,
                "max_a": 0.0,
                "min_p": 40.0,
                "max_p": 0.0,
                "str": 0.68,
                "tol": 10.0,
                "tok": "low",
            },
            "conservative": {
                "sens": 62,
                "close": 2,
                "open": 0,
                "min_w": 10.0,
                "max_w": 48.0,
                "min_l": 36.0,
                "min_a": 100.0,
                "max_a": 0.0,
                "min_p": 44.0,
                "max_p": 0.0,
                "str": 0.72,
                "tol": 5.0,
                "tok": "low",
            },
        }

    def _on_metal_preset_changed(self, *_args) -> None:
        if not hasattr(self, "metal_preset_combo"):
            return
        key = str(self.metal_preset_combo.currentData() or "standard")
        pr = self._metal_preset_table().get(key)
        if not pr:
            self._on_extraction_settings_changed()
            return
        self.metal_sensitivity_slider.setValue(int(pr["sens"]))
        self.metal_morph_close_spin.setValue(int(pr["close"]))
        self.metal_morph_open_spin.setValue(int(pr["open"]))
        self.metal_min_width_spin.setValue(float(pr["min_w"]))
        self.metal_max_width_spin.setValue(float(pr["max_w"]))
        self.metal_min_length_spin.setValue(float(pr["min_l"]))
        self.metal_min_area_spin.setValue(float(pr["min_a"]))
        self.metal_max_area_spin.setValue(float(pr["max_a"]))
        self.metal_min_perimeter_spin.setValue(float(pr["min_p"]))
        self.metal_max_perimeter_spin.setValue(float(pr["max_p"]))
        self.metal_straightness_spin.setValue(float(pr["str"]))
        self.metal_angle_tolerance_spin.setValue(float(pr["tol"]))
        _ti = self.metal_sensitivity_combo.findData(str(pr["tok"]))
        if _ti >= 0:
            self.metal_sensitivity_combo.setCurrentIndex(_ti)
        if hasattr(self, "metal_allowed_angles_combo") and "angles" in pr:
            _ai = self.metal_allowed_angles_combo.findData(str(pr["angles"]))
            if _ai >= 0:
                self.metal_allowed_angles_combo.setCurrentIndex(_ai)
        self._on_extraction_settings_changed()

    def _preview_metal_mask(self, *_args) -> None:
        if hasattr(self, "metal_debug_visual_combo"):
            idx = self.metal_debug_visual_combo.findData("metal_binary_mask")
            if idx >= 0:
                self.metal_debug_visual_combo.setCurrentIndex(idx)
        if hasattr(self, "metal_show_mask_checkbox"):
            self.metal_show_mask_checkbox.setChecked(True)
        self._refresh_gradient_overlay()

    def _reset_metal_parameters(self, *_args) -> None:
        defaults = ContourExtractionSettings()
        if hasattr(self, "metal_preset_combo"):
            self.metal_preset_combo.setCurrentIndex(self.metal_preset_combo.findData("standard"))
        if hasattr(self, "metal_sensitivity_slider"):
            self.metal_sensitivity_slider.setValue(int(defaults.metal_sensitivity_0_100))
        if hasattr(self, "metal_sensitivity_value_label"):
            self.metal_sensitivity_value_label.setText(str(int(defaults.metal_sensitivity_0_100)))
        if hasattr(self, "metal_min_width_spin"):
            self.metal_min_width_spin.setValue(float(defaults.metal_min_trace_width_px))
        if hasattr(self, "metal_max_width_spin"):
            mw = defaults.metal_max_trace_width_px
            self.metal_max_width_spin.setValue(0.0 if mw is None else float(mw))
        if hasattr(self, "metal_min_length_spin"):
            self.metal_min_length_spin.setValue(float(defaults.metal_min_trace_length_px))
        if hasattr(self, "metal_segmentation_method_combo"):
            ix = self.metal_segmentation_method_combo.findData(defaults.metal_segmentation_method)
            if ix >= 0:
                self.metal_segmentation_method_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_sensitivity_combo"):
            ix = self.metal_sensitivity_combo.findData(defaults.metal_sensitivity)
            if ix >= 0:
                self.metal_sensitivity_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_min_area_spin"):
            self.metal_min_area_spin.setValue(float(defaults.metal_min_area))
        if hasattr(self, "metal_max_area_spin"):
            ma = defaults.metal_max_area
            self.metal_max_area_spin.setValue(0.0 if ma is None else float(ma))
        if hasattr(self, "metal_min_perimeter_spin"):
            self.metal_min_perimeter_spin.setValue(float(defaults.metal_min_perimeter))
        if hasattr(self, "metal_max_perimeter_spin"):
            mp = defaults.metal_max_perimeter
            self.metal_max_perimeter_spin.setValue(0.0 if mp is None else float(mp))
        if hasattr(self, "metal_epsilon_spin"):
            self.metal_epsilon_spin.setValue(float(defaults.epsilon))
        if hasattr(self, "metal_min_points_spin"):
            self.metal_min_points_spin.setValue(int(defaults.min_points))
        if hasattr(self, "metal_min_angle_spin"):
            self.metal_min_angle_spin.setValue(float(defaults.min_polygon_angle))
        if hasattr(self, "metal_approximation_checkbox"):
            self.metal_approximation_checkbox.setChecked(bool(defaults.metal_approximation_enabled))
        if hasattr(self, "metal_hierarchy_combo"):
            ix = self.metal_hierarchy_combo.findData(defaults.metal_hierarchy_mode)
            if ix >= 0:
                self.metal_hierarchy_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_allowed_angles_combo"):
            ix = self.metal_allowed_angles_combo.findData(defaults.metal_allowed_angles)
            if ix >= 0:
                self.metal_allowed_angles_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_angle_tolerance_spin"):
            self.metal_angle_tolerance_spin.setValue(float(defaults.metal_angle_tolerance_deg))
        if hasattr(self, "metal_straightness_spin"):
            self.metal_straightness_spin.setValue(float(defaults.metal_min_straightness))
        if hasattr(self, "metal_t_junction_checkbox"):
            self.metal_t_junction_checkbox.setChecked(bool(defaults.metal_allow_t_junction))
        if hasattr(self, "metal_border_handling_combo"):
            ix = self.metal_border_handling_combo.findData(defaults.metal_border_handling)
            if ix >= 0:
                self.metal_border_handling_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_validity_checkbox"):
            self.metal_validity_checkbox.setChecked(bool(defaults.metal_check_contour_validity))
        if hasattr(self, "metal_morph_close_spin"):
            self.metal_morph_close_spin.setValue(int(defaults.metal_morph_close_radius))
        if hasattr(self, "metal_morph_open_spin"):
            self.metal_morph_open_spin.setValue(int(defaults.metal_morph_open_radius))
        if hasattr(self, "metal_use_wide_gradient_checkbox"):
            self.metal_use_wide_gradient_checkbox.setChecked(bool(defaults.metal_use_wide_conductor_gradient))
        if hasattr(self, "metal_wide_grad_radius_spin"):
            self.metal_wide_grad_radius_spin.setValue(int(defaults.metal_wide_gradient_profile_radius_px))
        if hasattr(self, "metal_wide_grad_conf_spin"):
            self.metal_wide_grad_conf_spin.setValue(float(defaults.metal_wide_gradient_min_direction_confidence))
        if hasattr(self, "metal_wide_grad_pair_len_spin"):
            self.metal_wide_grad_pair_len_spin.setValue(float(defaults.metal_wide_gradient_min_pair_length_px))
        if hasattr(self, "metal_wide_grad_parallel_spin"):
            self.metal_wide_grad_parallel_spin.setValue(float(defaults.metal_wide_gradient_parallel_tolerance_deg))
        if hasattr(self, "metal_wide_grad_gap_spin"):
            self.metal_wide_grad_gap_spin.setValue(int(defaults.metal_wide_gradient_max_edge_gap_px))
        if hasattr(self, "metal_wide_grad_overlap_spin"):
            self.metal_wide_grad_overlap_spin.setValue(float(defaults.metal_wide_gradient_min_overlap_ratio))
        if hasattr(self, "metal_show_conductors_checkbox"):
            self.metal_show_conductors_checkbox.setChecked(bool(defaults.metal_display_show_conductors))
        if hasattr(self, "metal_show_rejected_checkbox"):
            self.metal_show_rejected_checkbox.setChecked(bool(defaults.metal_display_show_rejected))
        if hasattr(self, "metal_show_suspicious_checkbox"):
            self.metal_show_suspicious_checkbox.setChecked(bool(defaults.metal_display_show_suspicious))
        if hasattr(self, "metal_show_border_checkbox"):
            self.metal_show_border_checkbox.setChecked(bool(defaults.metal_display_show_border_highlight))
        if hasattr(self, "metal_show_mask_checkbox"):
            self.metal_show_mask_checkbox.setChecked(bool(defaults.metal_display_show_mask))
        if hasattr(self, "metal_debug_visual_combo"):
            ix = self.metal_debug_visual_combo.findData(defaults.metal_debug_visual)
            if ix >= 0:
                self.metal_debug_visual_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_overlay_opacity_spin"):
            self.metal_overlay_opacity_spin.setValue(float(defaults.metal_overlay_opacity))
        self._on_extraction_settings_changed()

    def _set_recognition_status(self, kind: str, message: str | None = None) -> None:
        if not hasattr(self, "recognition_status_label"):
            return
        if self._ui_language == "ru":
            texts = {
                "idle": "Готово",
                "disabled": "Извлечение отключено",
                "updating": "Выполняется обработка…",
                "error": "Ошибка",
            }
        else:
            texts = {
                "idle": "Ready",
                "disabled": "Recognition off",
                "updating": "Updating…",
                "error": "Error",
            }
        text = message or texts.get(kind, texts["idle"])
        self.recognition_status_label.setText(text)

    def _current_save_options(self) -> SaveOptions:
        return SaveOptions(
            save_cif=self.save_cif_checkbox.isChecked(),
            save_csv=self.save_csv_checkbox.isChecked(),
            save_txt=self.save_txt_checkbox.isChecked(),
            save_svg=self.save_svg_checkbox.isChecked(),
            save_preview=self.save_preview_checkbox.isChecked(),
        )

    def _paint_image_row_item(self, item: QListWidgetItem, image_path: str, *, show_text: bool = True) -> None:
        normalized = str(Path(image_path))
        painted = classify_image_side_paint_status(
            never_opened=normalized not in self._viewed_image_paths,
            polygons_dirty=self._workspace.image_has_changes(normalized),
            persist_highlight=normalized in self._persisted_highlight_paths,
        )
        item.setData(FRAME_STATUS_ROLE, painted.value)
        hex_background = background_hex_image_paint_status(painted)
        if hex_background:
            tint = QColor(hex_background)
            item.setBackground(QBrush(tint))
            item.setData(Qt.ItemDataRole.BackgroundRole, tint)
        else:
            item.setBackground(QBrush())
            item.setData(Qt.ItemDataRole.BackgroundRole, None)
        fg = QColor(foreground_hex_image_has_vector_overlay(bool(self._workspace.resolve_cif_path(normalized))))
        item.setForeground(QBrush(fg))
        item.setText(Path(normalized).stem if show_text else "")

    def _find_image_list_item(self, image_path: str) -> QListWidgetItem | None:
        target = str(Path(image_path))
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            if item is not None and str(item.data(Qt.ItemDataRole.UserRole) or "") == target:
                return item
        return None

    def _update_frame_item_status(self, image_path: str | None) -> None:
        if not image_path:
            return
        item = self._find_image_list_item(image_path)
        if item is None:
            return
        self._paint_image_row_item(item, str(Path(image_path)))
        self._refresh_vector_items_for_stems({Path(str(image_path)).stem.lower()})
        self._update_thumbnail_item_status(image_path)

    def _refresh_image_list_item_states(self) -> None:
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            if item is None:
                continue
            image_path = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if image_path:
                self._paint_image_row_item(item, image_path)
        self._refresh_vector_rows_for_workspace()
        self._update_thumbnail_grid_selection()

    def _update_thumbnail_item_status(self, image_path: str | None) -> None:
        if not image_path or not hasattr(self, "thumbnail_grid"):
            return
        normalized = str(Path(image_path))
        for index in range(self.thumbnail_grid.count()):
            item = self.thumbnail_grid.item(index)
            if item is not None and str(item.data(Qt.ItemDataRole.UserRole) or "") == normalized:
                self._paint_image_row_item(item, normalized, show_text=False)
                break

    def _update_vector_edit_status_label(self) -> None:
        if not hasattr(self, "vector_edit_status_label"):
            return
        if self._workspace.current_image_path is None:
            self.vector_edit_status_label.clear()
            return
        if not self._updating_views:
            self._workspace.update_current_polygons(self.get_polygons())
        dirty = self._workspace.current_image_has_changes()
        if self._ui_language == "ru":
            self.vector_edit_status_label.setText("Изменено" if dirty else "Сохранено")
        else:
            self.vector_edit_status_label.setText("Modified" if dirty else "Saved")

    def _persist_current_overlay_changes(self) -> bool:
        """Persist editor polygons for the current frame (dataset export and/or linked CIF)."""

        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        if current_state is None or current_image_path is None:
            return True
        current_polygons = self.get_polygons()
        self._workspace.update_current_polygons(current_polygons)
        if not self._workspace.current_image_has_changes():
            self._update_frame_item_status(current_image_path)
            self._update_vector_edit_status_label()
            return True

        want_dataset = bool(self.dataset_mode_checkbox.isChecked())
        can_cif = bool(current_state.loaded_cif_path and current_state.source_image is not None)

        if not want_dataset and not can_cif:
            self._append_log(
                self._tr(
                    "vector_save_no_target_log",
                    "Нет каталога набора данных или связанного CIF для сохранения правок текущего кадра."
                    if self._ui_language == "ru"
                    else "No dataset directory or linked CIF available to save edits for the current frame.",
                )
            )
            return False

        if want_dataset:
            saved_ds = self._export_dataset_frame_for_state(current_image_path, current_state, current_polygons)
            if not saved_ds:
                return False

        if can_cif:
            image_size = (int(current_state.source_image.shape[1]), int(current_state.source_image.shape[0]))
            try:
                save_polygons_cif(
                    current_state.loaded_cif_path,
                    current_image_path,
                    current_polygons,
                    image_size=image_size,
                )
            except Exception as exc:
                self._append_log(
                    self._tr(
                        "autosave_failed_log",
                        "Не удалось сохранить CIF {path}: {error}"
                        if self._ui_language == "ru"
                        else "Failed to save CIF {path}: {error}",
                        path=current_state.loaded_cif_path,
                        error=exc,
                    )
                )
                return False

        persisted_path = str(Path(current_image_path))
        self._persisted_highlight_paths.add(persisted_path)
        self._workspace.sync_polygon_reference_to_current(persisted_path)
        self._append_log(
            self._tr(
                "vectors_persisted_transition_log",
                "Изменения векторов сохранены для кадра {name}"
                if self._ui_language == "ru"
                else "Vector edits saved for frame {name}",
                name=Path(current_image_path).name,
            )
        )
        self._update_frame_item_status(current_image_path)
        self._update_vector_edit_status_label()
        return True

    def _discard_current_vector_changes(self) -> None:
        state = self._workspace.current_state
        path = self._workspace.current_image_path
        if state is None or path is None:
            return
        restored = [polygon.clone() for polygon in state.reference_polygons]
        self._updating_views = True
        try:
            state.polygons = restored
            self._workspace.update_current_polygons(restored)
            self.polygon_editor.set_polygons(restored)
        finally:
            self._updating_views = False
        self._persisted_highlight_paths.discard(str(Path(path)))
        self._update_frame_item_status(path)
        self._update_vector_edit_status_label()

    def _prompt_transition_vector_save_dialog(self) -> TransitionPromptChoice:
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle(
            self._tr(
                "unsaved_vectors_dialog_title",
                "Несохранённые изменения" if self._ui_language == "ru" else "Unsaved changes",
            )
        )
        msg.setText(
            self._tr(
                "unsaved_vectors_dialog_text",
                "Сохранить изменения?" if self._ui_language == "ru" else "Save changes?",
            )
        )
        save_button = msg.addButton(
            "Сохранить" if self._ui_language == "ru" else "Save",
            QMessageBox.ButtonRole.AcceptRole,
        )
        discard_button = msg.addButton(
            "Не сохранять" if self._ui_language == "ru" else "Don't save",
            QMessageBox.ButtonRole.DestructiveRole,
        )
        cancel_button = msg.addButton(
            "Отмена" if self._ui_language == "ru" else "Cancel",
            QMessageBox.ButtonRole.RejectRole,
        )
        autosave_checkbox = QCheckBox(
            "Автосохранение при переходе к следующему кадру"
            if self._ui_language == "ru"
            else "Autosave on next frame"
        )
        autosave_checkbox.setChecked(False)
        msg.setCheckBox(autosave_checkbox)
        msg.setDefaultButton(save_button)
        result = msg.exec()
        clicked = msg.clickedButton()
        if clicked is None:
            if result == QMessageBox.StandardButton.Cancel:
                return TransitionPromptChoice.CANCEL
            if result == QMessageBox.StandardButton.Discard:
                return TransitionPromptChoice.DISCARD
            if result == QMessageBox.StandardButton.Save and autosave_checkbox.isChecked() and hasattr(
                self, "autosave_on_frame_transition_checkbox"
            ):
                self.autosave_on_frame_transition_checkbox.setChecked(True)
            return TransitionPromptChoice.SAVE
        if clicked is cancel_button:
            return TransitionPromptChoice.CANCEL
        if clicked is discard_button:
            return TransitionPromptChoice.DISCARD
        if clicked is save_button and autosave_checkbox.isChecked() and hasattr(
            self, "autosave_on_frame_transition_checkbox"
        ):
            self.autosave_on_frame_transition_checkbox.setChecked(True)
        return TransitionPromptChoice.SAVE

    def _warn_transition_blocked_after_failed_autosave(self) -> None:
        QMessageBox.warning(
            self,
            self._tr(
                "autosave_transition_blocked_title",
                "Не удалось сохранить" if self._ui_language == "ru" else "Save failed",
            ),
            self._tr(
                "autosave_transition_blocked_text",
                "Автосохранение не выполнено; переход отменён, данные не потеряны."
                if self._ui_language == "ru"
                else "Autosave failed; navigation cancelled — your edits were kept.",
            ),
        )

    def _warn_transition_blocked_after_failed_manual_save(self) -> None:
        QMessageBox.warning(
            self,
            self._tr(
                "manual_save_transition_blocked_title",
                "Не удалось сохранить" if self._ui_language == "ru" else "Save failed",
            ),
            self._tr(
                "manual_save_transition_blocked_text",
                "Сохранение не выполнено; переход отменён, данные не потеряны."
                if self._ui_language == "ru"
                else "Save failed; navigation cancelled — your edits were kept.",
            ),
        )

    def _try_leave_current_frame(self) -> bool:
        self._workspace.update_current_polygons(self.get_polygons())
        dirty = self._workspace.current_image_has_changes()
        if not dirty:
            self._update_vector_edit_status_label()
            return True

        autosave_on = bool(
            hasattr(self, "autosave_on_frame_transition_checkbox")
            and self.autosave_on_frame_transition_checkbox.isChecked()
        )
        if autosave_on:
            save_ok = self._persist_current_overlay_changes()
            allowed = navigation_allowed_after_autosave_attempt(dirty=True, save_ok=save_ok)
            if not allowed:
                self._warn_transition_blocked_after_failed_autosave()
            return allowed

        choice = self._prompt_transition_vector_save_dialog()
        if choice == TransitionPromptChoice.CANCEL:
            return False

        if choice == TransitionPromptChoice.DISCARD:
            self._discard_current_vector_changes()
            return True

        save_ok = self._persist_current_overlay_changes()
        allowed = navigation_allowed_after_prompt(dirty=True, choice=TransitionPromptChoice.SAVE, save_ok=save_ok)
        if not allowed:
            self._warn_transition_blocked_after_failed_manual_save()
        return allowed

    def confirm_ok_to_leave_current_vectors(self) -> bool:
        """Ask before closing or reloading images when the active frame has unsaved vector edits."""

        return self._try_leave_current_frame()

    def _sync_current_state_views(self) -> None:
        self._updating_views = True
        try:
            display_image = self._display_image_for_current_state()
            current_state = self._workspace.current_state
            polygons_synced = [polygon.clone() for polygon in current_state.polygons] if current_state else []
            self.polygon_editor.set_image(display_image)
            self.polygon_editor.set_polygons(polygons_synced)
            self.polygon_editor.set_debug_candidates(list(current_state.debug_candidates) if current_state else [])
            self.polygon_editor.set_via_debug_inspection_enabled(self._via_debug_inspection_enabled())
            if hasattr(self, "via_show_detected_checkbox"):
                self.polygon_editor.set_polygon_category_visible(
                    "via", self.via_show_detected_checkbox.isChecked()
                )
            if hasattr(self, "polygon_editor") and hasattr(self, "metal_show_rejected_checkbox"):
                layers = getattr(current_state, "metal_overlay_polygons", None) or {}
                self.polygon_editor.set_metal_overlays(
                    layers,
                    {
                        "rejected": self.metal_show_rejected_checkbox.isChecked(),
                        "suspicious": self.metal_show_suspicious_checkbox.isChecked(),
                        "border": self.metal_show_border_checkbox.isChecked(),
                        "wide_pairs_suspicious": self.metal_show_suspicious_checkbox.isChecked(),
                        "wide_pairs_rejected": self.metal_show_rejected_checkbox.isChecked(),
                    },
                )
            if hasattr(self, "metal_show_conductors_checkbox"):
                show_c = self.metal_show_conductors_checkbox.isChecked()
                self.polygon_editor.set_polygon_category_visible("conductor", show_c)
                self.polygon_editor.set_polygon_category_visible("metal_border", show_c)
                self.polygon_editor.set_polygon_category_visible("metal_wide_gradient", show_c)
            self._sync_neighbor_frames()
            self._sync_extra_layers()
            self._refresh_gradient_overlay()
        finally:
            self._updating_views = False
        self._update_vector_edit_status_label()

    def _display_image_for_current_state(self):
        current_state = self._workspace.current_state
        if self._show_source_while_middle_held and current_state is not None and current_state.source_image is not None:
            return current_state.source_image
        return self._workspace.current_display_image()

    def _via_debug_inspection_enabled(self) -> bool:
        return bool(hasattr(self, "debug_candidates_checkbox") and self.debug_candidates_checkbox.isChecked())

    def _neighbor_frame_image(self, image_path: str):
        state = getattr(self._workspace, "_state_cache", {}).get(image_path)
        if state is not None:
            return state.preprocessed_image if state.preprocessed_image is not None else state.source_image
        cached = self._neighbor_image_cache.get(image_path)
        if cached is not None:
            return cached
        try:
            image = load_image_color(image_path)
        except Exception as exc:
            self._append_log(
                self._tr(
                    "neighbor_frame_load_failed_log",
                    "Не удалось загрузить соседний кадр {path}: {error}"
                    if self._ui_language == "ru"
                    else "Failed to load neighbor frame {path}: {error}",
                    path=image_path,
                    error=exc,
                )
            )
            return None
        self._neighbor_image_cache[image_path] = image
        return image

    def _odd_neighbor_grid_size(self, value: int) -> int:
        size = max(3, min(7, int(value)))
        return size if size % 2 else size - 1

    def _neighbor_grid_size_for_zoom(self) -> int:
        max_grid = self._odd_neighbor_grid_size(self.neighbor_max_grid_spin.value())
        zoom = self.polygon_editor.zoom_factor() if hasattr(self, "polygon_editor") else 1.0
        requested = 7 if zoom < 0.25 else 5 if zoom < 0.45 else 3
        return min(max_grid, requested)

    def _sync_neighbor_frames(self) -> None:
        if not hasattr(self, "polygon_editor"):
            return
        if not hasattr(self, "show_neighbor_frames_checkbox") or not self.show_neighbor_frames_checkbox.isChecked():
            self.polygon_editor.set_neighbor_frames([], 0.0, 0, False)
            return
        current_path = self._workspace.current_image_path
        image_paths = list(self._workspace.image_paths)
        if not current_path or current_path not in image_paths:
            self.polygon_editor.set_neighbor_frames([], 0.0, 0, False)
            return
        current_index = image_paths.index(current_path)
        columns = max(1, int(self.neighbor_columns_spin.value()))
        current_row = current_index // columns
        current_column = current_index % columns
        radius = self._neighbor_grid_size_for_zoom() // 2
        frames: list[tuple[int, int, object, str]] = []
        for row_offset in range(-radius, radius + 1):
            for column_offset in range(-radius, radius + 1):
                if row_offset == 0 and column_offset == 0:
                    continue
                row = current_row + row_offset
                column = current_column + column_offset
                if row < 0 or column < 0 or column >= columns:
                    continue
                index = row * columns + column
                if index < 0 or index >= len(image_paths):
                    continue
                image_path = image_paths[index]
                image = self._neighbor_frame_image(image_path)
                if image is None:
                    continue
                frames.append((column_offset, row_offset, image, image_path))
        self.polygon_editor.set_neighbor_frames(
            frames,
            float(self.neighbor_opacity_spin.value()),
            int(self.neighbor_overlap_spin.value()),
            True,
        )

    def _on_neighbor_frame_activated(self, image_path: str) -> None:
        if image_path in self._workspace.image_paths:
            item = self._find_image_list_item(image_path)
            if item is not None:
                self.image_list.setCurrentItem(item)
            else:
                self.load_image(image_path)

    def _abort_in_flight_interactive_processing(self, *, preview: bool, prepared: bool) -> None:
        self._preview_update_timer.stop()
        if preview:
            if self._preview_run_cancel is not None:
                self._preview_run_cancel.set()
            self._preview_pending_request = None
            self._preview_pending_signature = None
        if prepared:
            if self._prepared_image_run_cancel is not None:
                self._prepared_image_run_cancel.set()
            self._prepared_image_pending_request = None
            self._prepared_image_pending_signature = None
        self._refresh_busy_indicator()

    def _queue_prepared_image_update(self, image_path: str, source_image) -> None:
        request = PreparedImageRequest(
            image_path=image_path,
            source_image=source_image,
            pipeline_config=self.get_pipeline(),
        )
        signature = self._prepared_image_request_signature(request)
        if signature == self._prepared_image_running_signature or signature == self._prepared_image_pending_signature:
            self._refresh_busy_indicator()
            return
        if self._prepared_image_run_cancel is not None:
            self._prepared_image_run_cancel.set()
        self._prepared_image_pending_request = request
        self._prepared_image_pending_signature = signature
        self._refresh_busy_indicator()
        self._start_pending_prepared_image_update()

    def _start_pending_prepared_image_update(self) -> None:
        if self._prepared_image_pending_request is None:
            return
        if self._prepared_image_running_request_id is not None:
            if self._prepared_image_run_cancel is not None:
                self._prepared_image_run_cancel.set()
            return
        request = self._prepared_image_pending_request
        self._prepared_image_pending_request = None
        request_signature = self._prepared_image_pending_signature
        self._prepared_image_pending_signature = None
        self._prepared_image_request_serial += 1
        request_id = self._prepared_image_request_serial
        self._prepared_image_running_request_id = request_id
        self._prepared_image_running_signature = request_signature
        cancel = threading.Event()
        self._prepared_image_run_cancel = cancel
        worker = PreparedImageRunnable(request_id=request_id, request=request, cancel_event=cancel)
        worker.signals.result.connect(self._on_prepared_image_result)
        worker.signals.error.connect(self._on_prepared_image_error)
        worker.signals.finished.connect(self._on_prepared_image_finished)
        self._prepared_image_thread_pool.start(worker)
        self._refresh_busy_indicator()

    def _build_preview_request(self) -> PreviewProcessingRequest | None:
        if not self._workspace.current_image_path:
            return None
        source_image = None
        preprocessed_image = None
        current_state = self._workspace.current_state
        pipeline_config = self.get_pipeline()
        if current_state is not None and current_state.image_path == self._workspace.current_image_path:
            source_image = current_state.source_image
            if current_state.preprocessed_image is not None and current_state.pipeline_config == pipeline_config:
                preprocessed_image = current_state.preprocessed_image
        passthrough: tuple[PolygonData, ...] | None = None
        if hasattr(self, "recognition_mode_combo") and str(self.recognition_mode_combo.currentData() or "") == "disabled":
            passthrough = tuple(polygon.clone() for polygon in self.get_polygons())
        return PreviewProcessingRequest(
            image_path=self._workspace.current_image_path,
            pipeline_config=pipeline_config,
            contour_settings=self._current_contour_settings(),
            source_image=source_image,
            preprocessed_image=preprocessed_image,
            passthrough_polygons=passthrough,
        )

    def _preview_request_signature(self, request: PreviewProcessingRequest) -> tuple[str, str, str, int]:
        return build_preview_request_signature(request)

    def _prepared_image_request_signature(self, request: PreparedImageRequest) -> tuple[str, str]:
        return build_prepared_image_signature(request)

    def _queue_preview_processing(self, *, debounced: bool) -> None:
        request = self._build_preview_request()
        if request is None:
            self._append_log(self._tr("no_image_selected_log"))
            return
        if hasattr(self, "recognition_mode_combo"):
            self._set_recognition_status("updating")
        signature = self._preview_request_signature(request)
        if signature == self._preview_running_signature or signature == self._preview_pending_signature:
            self._refresh_busy_indicator()
            return
        self._preview_update_timer.stop()
        if self._preview_run_cancel is not None:
            self._preview_run_cancel.set()
        self._preview_pending_request = request
        self._preview_pending_signature = signature
        self._refresh_busy_indicator()
        if debounced:
            self._preview_update_timer.start()
            return
        self._preview_update_timer.stop()
        self._start_pending_preview_processing()

    def _start_pending_preview_processing(self) -> None:
        if self._preview_pending_request is None:
            return
        if self._preview_running_request_id is not None:
            if self._preview_run_cancel is not None:
                self._preview_run_cancel.set()
            return
        request = self._preview_pending_request
        self._preview_pending_request = None
        request_signature = self._preview_pending_signature
        self._preview_pending_signature = None
        self._preview_request_serial += 1
        request_id = self._preview_request_serial
        self._preview_running_request_id = request_id
        self._preview_running_signature = request_signature
        cancel = threading.Event()
        self._preview_run_cancel = cancel
        worker = PreviewProcessingRunnable(request_id=request_id, request=request, cancel_event=cancel)
        worker.signals.result.connect(self._on_preview_processing_result)
        worker.signals.error.connect(self._on_preview_processing_error)
        worker.signals.finished.connect(self._on_preview_processing_finished)
        self._preview_thread_pool.start(worker)
        self._refresh_busy_indicator()

    def _append_log(self, message: str) -> None:
        self.logMessage.emit(message)

    def _refresh_busy_indicator(self) -> None:
        active = any(
            (
                self._preview_running_request_id is not None,
                self._preview_pending_request is not None,
                self._preview_update_timer.isActive(),
                self._prepared_image_running_request_id is not None,
                self._prepared_image_pending_request is not None,
                self._auto_tune_running_request_id is not None,
            )
        )
        if hasattr(self, "preview_busy_label"):
            self.preview_busy_label.setText(self._busy_indicator_text())
            self.preview_busy_label.setVisible(active)
        if hasattr(self, "preview_busy_progress"):
            self.preview_busy_progress.setVisible(active)
        if hasattr(self, "auto_tune_button"):
            self.auto_tune_button.setEnabled(self._auto_tune_running_request_id is None)

    def _on_prepared_image_result(
        self, request_id: int, image_path: str, preprocessed_image, pipeline_config: dict
    ) -> None:
        if request_id != self._prepared_image_running_request_id:
            return
        if pipeline_config != self.get_pipeline():
            return
        if self._workspace.store_preprocessed_image(image_path, preprocessed_image, pipeline_config):
            self._sync_current_state_views()
            self._try_extract_if_recognition_enabled()

    def _on_prepared_image_error(self, request_id: int, message: str) -> None:
        if request_id != self._prepared_image_running_request_id:
            return
        self._append_log(self._tr("processing_failed_log", error=message))

    def _on_prepared_image_finished(self, request_id: int) -> None:
        if request_id == self._prepared_image_running_request_id:
            self._prepared_image_running_request_id = None
            self._prepared_image_running_signature = None
            self._prepared_image_run_cancel = None
        if self._prepared_image_pending_request is not None:
            self._start_pending_prepared_image_update()
        self._refresh_busy_indicator()

    def _on_auto_tune_result(self, request_id: int, result: AutoTuneResult) -> None:
        if request_id != self._auto_tune_running_request_id:
            return
        self._apply_auto_tune_result(result)
        roi_width = result.roi_bbox[2]
        roi_height = result.roi_bbox[3]
        self._append_log(
            self._tr(
                "auto_tune_finished_log",
                "Автоподбор завершён: score={score:.3f}, ROI={width}x{height}, проверок={evaluations}."
                if self._ui_language == "ru"
                else "Auto-fit completed: score={score:.3f}, ROI={width}x{height}, evaluations={evaluations}.",
                score=result.score,
                width=roi_width,
                height=roi_height,
                evaluations=result.evaluations,
            )
        )

    def _on_auto_tune_error(self, request_id: int, message: str) -> None:
        if request_id != self._auto_tune_running_request_id:
            return
        self._append_log(
            self._tr(
                "auto_tune_failed_log",
                "Ошибка автоподбора: {error}" if self._ui_language == "ru" else "Auto-fit failed: {error}",
                error=message,
            )
        )

    def _on_auto_tune_finished(self, request_id: int) -> None:
        if request_id == self._auto_tune_running_request_id:
            self._auto_tune_running_request_id = None
        self._refresh_busy_indicator()

    def _on_preview_processing_result(self, request_id: int, result) -> None:
        if request_id != self._preview_running_request_id:
            return
        if self._workspace.current_image_path != result.image_path:
            return

        if self._workspace.apply_processing_result(result):
            self._sync_current_state_views()
        self._update_frame_item_status(result.image_path)
        if hasattr(self, "recognition_mode_combo"):
            if str(self.recognition_mode_combo.currentData() or "") == "disabled":
                self._set_recognition_status("disabled")
            else:
                self._set_recognition_status("idle")
        self._set_progress_status("current_image_processed_status")
        self._append_log(
            self._tr(
                "current_image_processed_log",
                image_name=Path(result.image_path).name,
                count=len(result.polygons),
            )
        )
        self.imageProcessed.emit(result.image_path, result.polygons)

    def _on_preview_processing_error(self, request_id: int, message: str) -> None:
        if request_id != self._preview_running_request_id:
            return
        if hasattr(self, "recognition_mode_combo"):
            self._set_recognition_status("error", message)
        self._append_log(self._tr("processing_failed_log", error=message))

    def _on_preview_processing_finished(self, request_id: int) -> None:
        if request_id == self._preview_running_request_id:
            self._preview_running_request_id = None
            self._preview_running_signature = None
            self._preview_run_cancel = None
        if self._preview_pending_request is not None and not self._preview_update_timer.isActive():
            self._start_pending_preview_processing()
        self._refresh_busy_indicator()

    def _show_batch_progress(self, total: int) -> None:
        if not self._batch_progress_enabled:
            self._hide_batch_progress()
            return
        self.batch_progress_bar.setRange(0, max(1, total))
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setVisible(True)

    def _hide_batch_progress(self) -> None:
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)

    def _on_polygons_edited(self) -> None:
        if self._updating_views:
            return
        if self._workspace.update_current_polygons(self.get_polygons()):
            current_path = self._workspace.current_image_path
            if current_path:
                self._persisted_highlight_paths.discard(str(Path(current_path)))
            self._update_frame_item_status(self._workspace.current_image_path)
            self._update_vector_edit_status_label()
            self.polygonsEdited.emit()

    def _on_batch_result(self, result) -> None:
        self.imageProcessed.emit(result.image_path, result.polygons)
        self._append_log(
            self._tr(
                "batch_result_log",
                image_name=Path(result.image_path).name,
                count=len(result.polygons),
            )
        )

    def _on_batch_progress(self, current: int, total: int) -> None:
        if self._batch_progress_enabled:
            self.batch_progress_bar.setRange(0, max(1, total))
            self.batch_progress_bar.setValue(current)
        self._set_progress_status("batch_progress_status", current=current, total=total)
        self.batchProgress.emit(current, total)

    def _on_batch_finished(self) -> None:
        self._batch_progress_enabled = False
        self._hide_batch_progress()
        self._set_progress_status("batch_finished_status")
        self.batchFinished.emit()

    def _on_batch_error(self, image_path: str, message: str) -> None:
        self._append_log(self._tr("batch_error_log", image_name=Path(image_path).name, message=message))

    def refresh_image_list(self) -> None:
        directory = self.input_dir_edit.text().strip()
        if not directory:
            self._append_log(self._tr("input_directory_empty_log"))
            return
        self._begin_async_directory_scan(directory)

    def set_input_directory(self, path: str) -> None:
        normalized = str(Path(path))
        root = Path(normalized)
        if not root.exists() or not root.is_dir():
            self._append_log(
                self._tr(
                    "input_directory_missing_log",
                    directory=normalized,
                )
            )
            return
        self.input_dir_edit.setText(normalized)
        self._save_persisted_paths()
        self._begin_async_directory_scan(normalized)

    def set_cif_directory(self, path: str) -> None:
        directory_state = index_cif_directory(path)
        self.cif_dir_edit.setText(directory_state.directory)
        self._save_persisted_paths()
        self._workspace.set_cif_index(directory_state.indexed_paths)
        if directory_state.available:
            self._append_log(self._tr("cif_indexed_log", count=len(directory_state.indexed_paths)))
        else:
            self._append_log(self._tr("cif_directory_unavailable_log"))
        self._sync_after_cif_index_changed()

    def set_output_directory(self, path: str) -> None:
        self.output_dir_edit.setText(path)
        self._save_persisted_paths()

    def set_dataset_directory(self, path: str) -> None:
        self.dataset_dir_edit.setText(path)
        self._save_persisted_paths()

    def load_images(self, paths: list[str]) -> None:
        if self._workspace.current_state is not None and not self._try_leave_current_frame():
            return
        self._scan_generation += 1
        normalized_paths = self._workspace.replace_image_selection(paths, is_supported_image=is_image_path)
        self._neighbor_image_cache.clear()
        self._prune_tagged_sets_for_images(normalized_paths)
        self._abort_in_flight_interactive_processing(preview=True, prepared=True)
        self.image_list.clear()
        for path in normalized_paths:
            item = QListWidgetItem(Path(path).stem)
            item.setToolTip(f"Путь к файлу: {path}" if self._ui_language == "ru" else f"File path: {path}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._paint_image_row_item(item, path)
            self.image_list.addItem(item)
        self._rebuild_thumbnail_grid()
        if normalized_paths:
            self.image_list.setCurrentRow(0)
        else:
            self._sync_current_state_views()
        self._rebuild_vector_list()
        self._refresh_vector_rows_for_workspace()
        self._sync_frame_navigation_controls()
        self._log_matching_gaps_after_refresh(self._matching_report())

    def _find_matching_cif_path(self, image_path: str) -> str | None:
        return self._workspace.resolve_cif_path(image_path)

    def _load_cif_overlay_polygons(self, image_path: str) -> list[PolygonData]:
        stem_key = Path(image_path).stem.lower()
        cif_path = self._find_matching_cif_path(image_path)
        if not cif_path:
            self._cif_load_failure_stems.discard(stem_key)
            return []
        try:
            referenced_image, image_size, polygons = load_polygons_cif(cif_path)
        except Exception as exc:
            self._cif_load_failure_stems.add(stem_key)
            self._append_log(self._tr("cif_load_failed_log", file_name=Path(cif_path).name, error=exc))
            return []
        self._cif_load_failure_stems.discard(stem_key)
        if referenced_image and Path(referenced_image).stem.lower() != Path(image_path).stem.lower():
            self._append_log(
                self._tr(
                    "cif_reference_name_diff_log",
                    file_name=Path(cif_path).name,
                    referenced_image=referenced_image,
                )
            )
        if image_size is not None:
            self._append_log(
                self._tr(
                    "cif_overlay_loaded_with_size_log",
                    file_name=Path(cif_path).name,
                    width=image_size[0],
                    height=image_size[1],
                    count=len(polygons),
                )
            )
        else:
            self._append_log(self._tr("cif_overlay_loaded_log", file_name=Path(cif_path).name, count=len(polygons)))
        return polygons

    def load_image(self, path: str) -> None:
        self._abort_in_flight_interactive_processing(preview=True, prepared=True)
        image_result = self._workspace.load_image(
            path,
            load_source_image=load_image_color,
            load_cif_overlay=self._load_cif_overlay_polygons,
        )
        self._viewed_image_paths.add(str(Path(image_result.image_path)))
        if image_result.state is not None and not image_result.cache_hit and not image_result.reused_current_state:
            image_result.state.loaded_cif_path = self._find_matching_cif_path(image_result.image_path)
            image_result.state.reference_polygons = [polygon.clone() for polygon in image_result.state.polygons]
        if image_result.reused_current_state:
            self._update_frame_item_status(image_result.image_path)
            self._update_thumbnail_grid_selection()
            self._sync_frame_navigation_controls()
            return
        self._sync_current_state_views()
        self._update_frame_item_status(image_result.image_path)
        self._update_thumbnail_grid_selection()
        self._sync_frame_navigation_controls()
        if (
            image_result.prepared_image_required
            and image_result.state is not None
            and image_result.state.source_image is not None
        ):
            self._queue_prepared_image_update(image_result.image_path, image_result.state.source_image)
        if image_result.cache_hit:
            self._append_log(self._tr("loaded_cached_state_log", image_path=image_result.image_path))
        else:
            self._append_log(self._tr("loaded_image_log", image_path=image_result.image_path))
        self._try_extract_if_recognition_enabled()

    def get_polygons(self) -> list[PolygonData]:
        return self.polygon_editor.get_polygons()

    def set_pipeline(self, config: dict) -> None:
        self._pipeline = PreprocessingPipeline.from_dict(config)
        self._populate_pipeline_list()
        self._auto_apply_pipeline()

    def get_pipeline(self) -> dict:
        return self._pipeline.to_dict()

    def process_current_image(self, *_args, debounced: bool = False) -> None:
        self._queue_preview_processing(debounced=debounced)

    def _export_dataset_frame_for_state(
        self,
        image_path: str,
        state: ImageProcessingState,
        polygons: list[PolygonData],
        dataset_directory: str | None = None,
    ) -> dict[str, str]:
        target_directory = dataset_directory or self.dataset_dir_edit.text().strip()
        result = export_frame_to_dataset(
            dataset_directory=target_directory,
            image_path=image_path,
            state=state,
            polygons=polygons,
        )
        if result.message_key is not None:
            self._append_log(self._tr(result.message_key, **(result.message_kwargs or {})))
        return result.saved_files

    def export_current_frame_to_dataset(self, dataset_directory: str | None = None) -> dict[str, str]:
        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        if current_state is None or current_image_path is None:
            self._append_log(self._tr("nothing_to_save_log"))
            return {}
        current_polygons = self.get_polygons()
        self._workspace.update_current_polygons(current_polygons)
        self._update_frame_item_status(current_image_path)
        return self._export_dataset_frame_for_state(
            current_image_path,
            current_state,
            current_polygons,
            dataset_directory=dataset_directory,
        )

    def save_current_result(
        self,
        output_directory: str | None = None,
        save_options: SaveOptions | None = None,
    ) -> dict[str, str]:
        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        if current_state is None or current_image_path is None:
            self._append_log(self._tr("nothing_to_save_log"))
            return {}
        target_directory = output_directory or self.output_dir_edit.text().strip()
        if not target_directory:
            self._append_log(self._tr("output_directory_not_set_log"))
            return {}
        self._workspace.update_current_polygons(self.get_polygons())
        had_vector_edits = self._workspace.current_image_has_changes()
        saved_files = save_result_bundle(
            output_directory=target_directory,
            image_path=current_image_path,
            polygons=self.get_polygons(),
            source_image=current_state.source_image,
            display_settings=self._display_settings,
            save_options=save_options or self._current_save_options(),
            metadata={
                "contour_settings": self._current_contour_settings().to_dict(),
                "pipeline": self.get_pipeline(),
            },
        )
        if saved_files:
            self._append_log(self._tr("saved_result_log", saved_files=saved_files))
            saved_key = str(Path(current_image_path))
            if had_vector_edits:
                self._persisted_highlight_paths.add(saved_key)
            self._workspace.sync_polygon_reference_to_current(saved_key)
            self._update_frame_item_status(current_image_path)
            self._update_vector_edit_status_label()
        return saved_files

    def start_batch_processing(
        self,
        image_paths: list[str] | None = None,
        max_workers: int | None = None,
    ) -> None:
        if self._batch_controller.is_running:
            self._append_log(self._tr("batch_already_running_log"))
            return
        paths = image_paths or list(self._workspace.image_paths)
        if not paths:
            self._append_log(self._tr("batch_no_images_log"))
            return
        output_directory = self.output_dir_edit.text().strip() or None
        save_options = self._current_save_options()
        started = self._batch_controller.start(
            BatchStartRequest(
                image_paths=list(paths),
                pipeline_config=self.get_pipeline(),
                contour_settings=self._current_contour_settings(),
                display_settings=self._display_settings,
                save_options=save_options,
                output_directory=output_directory,
                max_workers=max_workers or self.max_workers_spin.value(),
            )
        )
        if not started:
            return
        self._batch_progress_enabled = self._batch_controller.progress_enabled
        self._show_batch_progress(len(paths))
        self._set_progress_status("batch_started_status")

    def stop_batch_processing(self) -> None:
        self._batch_controller.stop()
