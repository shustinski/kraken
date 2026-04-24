from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, QSettings, QSignalBlocker, QSize, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
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
    QToolButton,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .adapters.qt.image_conversion import cv_to_qimage
from .adapters.qt.preview import AutoTuneRunnable, PreparedImageRunnable, PreviewProcessingRunnable
from .application.dto import PersistedPaths
from .application.processing import (
    VIA_SIZE_MODE_FIXED,
    ContourExtractionSettings,
    DisplaySettings,
    ImageProcessingState,
    SaveOptions,
    normalize_algorithm_backend,
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
from .application.use_cases import (
    AutoTuneResult,
    PreparedImageRequest,
    PreviewProcessingRequest,
    build_prepared_image_signature,
    build_preview_request_signature,
    index_cif_directory,
    load_input_directory,
)
from .batch_processor import BatchProcessor
from .domain import PolygonData
from .graphics_view import EditorTool
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
from .utils import is_image_path, load_image_color, scan_image_files

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
FRAME_STATUS_UNCHANGED = "unchanged"
FRAME_STATUS_VIEWED = "viewed"
FRAME_STATUS_MODIFIED = "modified"
VIA_PRESETS_SETTINGS_KEY = "via_search/user_presets"


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
                algorithm_backend="sem",
                sem_noise_level="medium",
                extraction_profile="conductors",
                object_type="conductor",
                output_mode="polygon",
                min_polygon_angle=90.0,
            ),
            "vias": ContourExtractionSettings(
                algorithm_backend="sem",
                sem_noise_level="medium",
                extraction_profile="vias",
                object_type="via",
                output_mode="box",
                via_search_mode="hybrid",
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
        self._preview_update_timer.setInterval(180)
        self._preview_update_timer.timeout.connect(self._start_pending_preview_processing)
        self._preview_request_serial = 0
        self._preview_running_request_id: int | None = None
        self._preview_pending_request: PreviewProcessingRequest | None = None
        self._preview_running_signature: tuple[str, str, str] | None = None
        self._preview_pending_signature: tuple[str, str, str] | None = None
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
        self._auto_tune_thread_pool = QThreadPool(self)
        self._auto_tune_thread_pool.setMaxThreadCount(1)
        self._auto_tune_thread_pool.setExpiryTimeout(-1)
        self._auto_tune_request_serial = 0
        self._auto_tune_running_request_id: int | None = None
        self._neighbor_image_cache: dict[str, object] = {}
        self._show_source_while_middle_held = False

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
        ]
        self._restoring_display_settings = True
        try:
            self._update_color_button(self.external_color_button, self._display_settings.external_color)
            self._update_color_button(self.hole_color_button, self._display_settings.hole_color)
            self._update_color_button(self.selected_color_button, self._display_settings.selected_color)
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
            self._restore_main_splitter_sizes(payload.get("main_splitter_sizes"))
        finally:
            self._restoring_display_settings = False
            del blockers
        self._sync_neighbor_frames()

    def _current_display_settings_payload(self) -> dict[str, object]:
        return {
            **self._display_settings.to_dict(),
            "random_object_colors": bool(self.random_object_colors_checkbox.isChecked()),
            "show_neighbor_frames": bool(self.show_neighbor_frames_checkbox.isChecked()),
            "neighbor_columns": int(self.neighbor_columns_spin.value()),
            "neighbor_max_grid": int(self.neighbor_max_grid_spin.value()),
            "neighbor_opacity": float(self.neighbor_opacity_spin.value()),
            "neighbor_overlap_pixels": int(self.neighbor_overlap_spin.value()),
            "main_splitter_sizes": self.main_splitter.sizes() if hasattr(self, "main_splitter") else [],
        }

    def _save_persisted_display_settings(self) -> None:
        if self._restoring_display_settings or not hasattr(self, "line_width_spin"):
            return
        self._display_settings_store.save(self._current_display_settings_payload())

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
        overview_action = self._help_menu.addAction(
            self._tr(
                "help_all_filters_action", "Все преобразования" if self._ui_language == "ru" else "All transformations"
            )
        )
        overview_action.triggered.connect(lambda _checked=False: self._show_help_dialog())
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
        self._set_field_tooltip(
            self.extraction_profile_label_widget, self.extraction_profile_combo, "extraction_profile"
        )
        self._set_field_tooltip(self.retrieval_mode_label_widget, self.retrieval_mode_combo, "retrieval_mode")
        self._set_field_tooltip(
            self.approximation_mode_label_widget, self.approximation_mode_combo, "approximation_mode"
        )
        self._set_field_tooltip(self.epsilon_label_widget, self.epsilon_spin, "epsilon")
        self._set_field_tooltip(self.epsilon_mode_label_widget, self.epsilon_relative_checkbox, "epsilon_mode")
        self._set_field_tooltip(self.min_area_label_widget, self.min_area_spin, "min_area")
        self._set_field_tooltip(self.max_area_label_widget, self.max_area_spin, "max_area")
        self._set_field_tooltip(self.min_perimeter_label_widget, self.min_perimeter_spin, "min_perimeter")
        self._set_field_tooltip(self.max_perimeter_label_widget, self.max_perimeter_spin, "max_perimeter")
        self._set_field_tooltip(self.min_point_count_label_widget, self.min_points_spin, "min_points")
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
        self._set_field_tooltip(self.via_search_mode_label_widget, self.via_search_mode_combo, "via_search_mode")
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
        self._set_field_tooltip(
            self.noisy_traces_via_preset_label_widget,
            self.noisy_traces_via_preset_button,
            "via_noisy_traces_preset",
        )
        self._set_field_tooltip(
            self.blurred_via_preset_label_widget,
            self.blurred_via_preset_button,
            "via_blurred_preset",
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
        blob_enabled = mode in {"hybrid", "blob"}
        template_enabled = mode in {"hybrid", "template"}
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

        white_enabled = self.via_white_range_checkbox.isChecked()
        self.via_white_range_min_spin.setEnabled(white_enabled)
        self.via_white_range_max_spin.setEnabled(white_enabled)
        if self.via_white_range_label_widget is not None:
            self.via_white_range_label_widget.setVisible(advanced and white_enabled)
        self.via_white_range_widget.setVisible(advanced and white_enabled)
        black_enabled = self.via_black_range_checkbox.isChecked()
        self.via_black_range_min_spin.setEnabled(black_enabled)
        self.via_black_range_max_spin.setEnabled(black_enabled)
        if self.via_black_range_label_widget is not None:
            self.via_black_range_label_widget.setVisible(advanced and black_enabled)
        self.via_black_range_widget.setVisible(advanced and black_enabled)

    def _update_extraction_profile_controls_state(self) -> None:
        is_via_profile = self._active_extraction_profile == "vias"
        advanced = self._advanced_extraction_enabled()
        self.basic_filters_group.setVisible(advanced)
        self.geometry_filters_group.setVisible(advanced)
        self.conductor_group.setEnabled(not is_via_profile)
        self.conductor_group.setVisible(advanced and not is_via_profile)
        self.via_group.setEnabled(is_via_profile)
        self.via_group.setVisible(is_via_profile)
        self.topology_group.setVisible(advanced and not is_via_profile)
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
        for label_widget, field_widget in advanced_via_widgets:
            if label_widget is not None:
                label_widget.setVisible(advanced and is_via_profile)
            field_widget.setVisible(advanced and is_via_profile)
        self._update_via_threshold_controls_state()

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
                "delete_single": "Вершина",
                "delete_area": "Область",
            }
        else:
            mapping = {
                "polygon_points": "By points",
                "polygon_rectangle": "Rectangle",
                "brush_freeform": "Freeform",
                "brush_45deg": "45° constrained",
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
            EditorTool.SELECT_AREA: self._tr(
                "tool_select_area", "Выбор рамкой" if self._ui_language == "ru" else "Area select"
            ),
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
            tooltip = (tooltip_pair[0] if self._ui_language == "ru" else tooltip_pair[1]) if tooltip_pair else label
            button.setToolTip(tooltip)
            button.setStatusTip(tooltip)
            button.setAccessibleName(label)

    def _update_action_button_texts(self) -> None:
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
            button.setToolTip(label)
            button.setStatusTip(label)
            button.setAccessibleName(label)
        for button, tooltip_key in (
            (self.undo_button, "undo_button"),
            (self.redo_button, "redo_button"),
            (self.zoom_in_button, "zoom_in_button"),
            (self.zoom_out_button, "zoom_out_button"),
            (self.fit_button, "fit_button"),
        ):
            tooltip = _localized_text(EDITOR_ACTION_TOOLTIPS, tooltip_key, self._ui_language)
            button.setToolTip(tooltip)
            button.setStatusTip(tooltip)

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
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

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
        settings = QSettings("ViaLaNet", "PolygonWidget")
        raw_payload = settings.value(VIA_PRESETS_SETTINGS_KEY, "{}", type=str)
        try:
            payload = json.loads(str(raw_payload or "{}"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(name): dict(value) for name, value in payload.items() if isinstance(value, dict)}

    def _save_user_via_presets(self) -> None:
        settings = QSettings("ViaLaNet", "PolygonWidget")
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
        return {key: value for key, value in payload.items() if key.startswith("via_") and key not in excluded_keys} | {
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
        title = "Отладка via" if self._ui_language == "ru" else "Via debug"
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
            f"{'Оценка' if self._ui_language == 'ru' else 'Score'}: {float(getattr(candidate, 'score', 0.0)):.1f}",
            f"{'Округлость' if self._ui_language == 'ru' else 'Roundness'}: {float(getattr(candidate, 'roundness', 0.0)):.1f}",
            f"{'Размер кандидата' if self._ui_language == 'ru' else 'Candidate size'}: {int(bbox[2])} x {int(bbox[3])} px",
            f"{'Позиция' if self._ui_language == 'ru' else 'Position'}: x={int(bbox[0])}, y={int(bbox[1])}",
        ]
        message = "\n".join(lines)
        self._append_log(message.replace("\n", " | "))
        QMessageBox.information(self, title, message)

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
        if not hasattr(self, "polygon_editor") or not hasattr(self, "gradient_overlay_checkbox"):
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

    def _on_extraction_settings_changed(self, *_args) -> None:
        if hasattr(self, "via_white_range_checkbox"):
            self._update_via_threshold_controls_state()
        self._store_active_extraction_profile_settings()
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_debug_candidates([])
            self.polygon_editor.set_via_debug_inspection_enabled(self._via_debug_inspection_enabled())
        self._refresh_gradient_overlay()
        self._auto_apply_pipeline()

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
        if self._ignore_extraction_profile_change:
            return
        self._store_active_extraction_profile_settings()
        profile = str(self.extraction_profile_combo.currentData() or "conductors")
        self._active_extraction_profile = profile
        self._set_extraction_settings(self._contour_settings_profiles[profile])
        self._update_extraction_profile_controls_state()
        self._refresh_gradient_overlay()
        self._auto_apply_pipeline()

    def _store_active_extraction_profile_settings(self) -> None:
        if not hasattr(self, "extraction_profile_combo"):
            return
        profile = str(self._active_extraction_profile or self.extraction_profile_combo.currentData() or "conductors")
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
        if previous is not None:
            self._autosave_current_overlay_if_needed()
        if current is None:
            return
        image_path = current.data(Qt.ItemDataRole.UserRole)
        if image_path:
            try:
                self.load_image(str(image_path))
            except Exception as exc:
                self._append_log(self._tr("failed_to_load_image_log", image_path=image_path, error=exc))
                QMessageBox.warning(self, self._tr("image_load_error_title"), str(exc))

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
            self._sync_current_state_views()
            self._save_persisted_paths()

    def _apply_cif_directory_edit(self) -> None:
        path = self.cif_dir_edit.text().strip()
        if path:
            self.set_cif_directory(path)
        else:
            self._workspace.clear_cif_index()
            self._save_persisted_paths()
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
        self._save_persisted_display_settings()

    def _on_neighbor_display_settings_changed(self, *_args) -> None:
        self._sync_neighbor_frames()
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
        self.process_current_image(debounced=True)

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
            QSignalBlocker(self.extraction_profile_combo),
            QSignalBlocker(self.algorithm_backend_combo),
            QSignalBlocker(self.result_shape_combo),
            QSignalBlocker(self.sem_noise_combo),
            QSignalBlocker(self.retrieval_mode_combo),
            QSignalBlocker(self.approximation_mode_combo),
            QSignalBlocker(self.epsilon_spin),
            QSignalBlocker(self.epsilon_relative_checkbox),
            QSignalBlocker(self.min_area_spin),
            QSignalBlocker(self.max_area_spin),
            QSignalBlocker(self.min_perimeter_spin),
            QSignalBlocker(self.min_points_spin),
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
        try:
            profile_index = self.extraction_profile_combo.findData(settings.extraction_profile)
            if profile_index >= 0:
                self._ignore_extraction_profile_change = True
                self.extraction_profile_combo.setCurrentIndex(profile_index)
                self._ignore_extraction_profile_change = False
            self._active_extraction_profile = str(settings.extraction_profile or self._active_extraction_profile)
            backend_index = self.algorithm_backend_combo.findData(
                normalize_algorithm_backend(settings.algorithm_backend)
            )
            if backend_index >= 0:
                self.algorithm_backend_combo.setCurrentIndex(backend_index)
            shape_index = self.result_shape_combo.findData(settings.output_mode)
            if shape_index >= 0:
                self.result_shape_combo.setCurrentIndex(shape_index)
            noise_index = self.sem_noise_combo.findData(settings.sem_noise_level)
            if noise_index >= 0:
                self.sem_noise_combo.setCurrentIndex(noise_index)
            retrieval_index = self.retrieval_mode_combo.findData(settings.retrieval_mode)
            if retrieval_index >= 0:
                self.retrieval_mode_combo.setCurrentIndex(retrieval_index)
            approximation_index = self.approximation_mode_combo.findData(settings.approximation_mode)
            if approximation_index >= 0:
                self.approximation_mode_combo.setCurrentIndex(approximation_index)
            self.epsilon_spin.setValue(float(settings.epsilon))
            self.epsilon_relative_checkbox.setChecked(bool(settings.epsilon_relative))
            self.min_area_spin.setValue(float(settings.min_area))
            self.max_area_spin.setValue(0.0 if settings.max_area is None else float(settings.max_area))
            self.min_perimeter_spin.setValue(float(settings.min_perimeter))
            self.max_perimeter_spin.setValue(0.0 if settings.max_perimeter is None else float(settings.max_perimeter))
            self.min_points_spin.setValue(int(settings.min_points))
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
            self._via_template_images = self._normalize_via_template_images(settings.via_template_images)
            self._refresh_via_template_list()
            self.debug_candidates_checkbox.setChecked(bool(settings.debug_enabled))
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
        via_size_mode = normalize_via_size_mode(self.via_size_mode_combo.currentData())
        via_search_mode = normalize_via_search_mode(self.via_search_mode_combo.currentData())
        fixed_via_pairs = self._fixed_via_pairs()
        fixed_via_widths = [width for width, _height in fixed_via_pairs]
        fixed_via_heights = [height for _width, height in fixed_via_pairs]
        max_hierarchy_depth = self.max_hierarchy_depth_spin.value()
        max_hole_area_ratio = self.max_hole_area_ratio_spin.value()
        extraction_profile = str(
            self.extraction_profile_combo.currentData() or self._active_extraction_profile or "conductors"
        )
        object_type = "via" if extraction_profile == "vias" else "conductor"
        output_mode = str(
            self.result_shape_combo.currentData() or ("box" if extraction_profile == "vias" else "polygon")
        )
        return ContourExtractionSettings(
            algorithm_backend=normalize_algorithm_backend(self.algorithm_backend_combo.currentData()),
            sem_noise_level=str(self.sem_noise_combo.currentData() or "medium"),
            extraction_profile=extraction_profile,
            object_type=object_type,
            output_mode=output_mode,
            retrieval_mode=str(self.retrieval_mode_combo.currentData() or self.retrieval_mode_combo.currentText()),
            approximation_mode=str(
                self.approximation_mode_combo.currentData() or self.approximation_mode_combo.currentText()
            ),
            epsilon=self.epsilon_spin.value(),
            epsilon_relative=self.epsilon_relative_checkbox.isChecked(),
            min_polygon_angle=self.min_polygon_angle_spin.value(),
            min_area=self.min_area_spin.value(),
            max_area=None if max_area <= 0 else max_area,
            min_perimeter=self.min_perimeter_spin.value(),
            min_points=self.min_points_spin.value(),
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
            conductor_gradient_enabled=self.conductor_gradient_checkbox.isChecked(),
            conductor_gradient_min_strength=self.conductor_gradient_min_strength_spin.value(),
            conductor_gradient_band_radius=self.conductor_gradient_band_radius_spin.value(),
            via_size_mode=via_size_mode,
            via_search_mode=via_search_mode,
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
            via_template_images=[template.copy() for template in self._via_template_images],
            debug_enabled=self.debug_candidates_checkbox.isChecked(),
            debug_gradient_map_enabled=self.debug_candidates_checkbox.isChecked(),
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

    def _current_save_options(self) -> SaveOptions:
        return SaveOptions(
            save_cif=self.save_cif_checkbox.isChecked(),
            save_csv=self.save_csv_checkbox.isChecked(),
            save_txt=self.save_txt_checkbox.isChecked(),
            save_svg=self.save_svg_checkbox.isChecked(),
            save_preview=self.save_preview_checkbox.isChecked(),
        )

    def _frame_status_for_image(self, image_path: str) -> str:
        if self._workspace.image_has_changes(image_path):
            return FRAME_STATUS_MODIFIED
        if str(Path(image_path)) in self._viewed_image_paths:
            return FRAME_STATUS_VIEWED
        return FRAME_STATUS_UNCHANGED

    def _frame_status_brush(self, status: str) -> QBrush:
        if status == FRAME_STATUS_MODIFIED:
            return QBrush(QColor("#86EFAC"))
        if status == FRAME_STATUS_VIEWED:
            return QBrush(QColor("#D1D5DB"))
        return QBrush(QColor("#D1D5DB"))

    def _apply_frame_status_to_item(self, item: QListWidgetItem, status: str) -> None:
        item.setData(FRAME_STATUS_ROLE, status)
        item.setBackground(self._frame_status_brush(status))

    def _find_image_list_item(self, image_path: str) -> QListWidgetItem | None:
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            if item is not None and str(item.data(Qt.ItemDataRole.UserRole) or "") == image_path:
                return item
        return None

    def _update_frame_item_status(self, image_path: str | None) -> None:
        if not image_path:
            return
        item = self._find_image_list_item(image_path)
        if item is None:
            return
        self._apply_frame_status_to_item(item, self._frame_status_for_image(image_path))

    def _refresh_image_list_item_states(self) -> None:
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            if item is None:
                continue
            image_path = str(item.data(Qt.ItemDataRole.UserRole) or "")
            self._apply_frame_status_to_item(item, self._frame_status_for_image(image_path))

    def _autosave_current_overlay_if_needed(self) -> None:
        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        if current_state is None or current_image_path is None:
            return
        current_polygons = self.get_polygons()
        self._workspace.update_current_polygons(current_polygons)
        current_has_changes = self._workspace.current_image_has_changes()
        if current_has_changes and self.dataset_mode_checkbox.isChecked():
            self._export_dataset_frame_for_state(current_image_path, current_state, current_polygons)
        if not current_state.loaded_cif_path or current_state.source_image is None:
            self._update_frame_item_status(current_image_path)
            return
        if not current_has_changes:
            self._update_frame_item_status(current_image_path)
            return
        image_size = (int(current_state.source_image.shape[1]), int(current_state.source_image.shape[0]))
        try:
            save_polygons_cif(
                current_state.loaded_cif_path,
                current_image_path,
                current_polygons,
                image_size=image_size,
            )
            self._append_log(
                self._tr(
                    "autosaved_cif_log",
                    "Автосохранен CIF: {path}" if self._ui_language == "ru" else "Autosaved CIF: {path}",
                    path=current_state.loaded_cif_path,
                )
            )
        except Exception as exc:
            self._append_log(
                self._tr(
                    "autosave_failed_log",
                    "Не удалось автосохранить CIF {path}: {error}"
                    if self._ui_language == "ru"
                    else "Failed to autosave CIF {path}: {error}",
                    path=current_state.loaded_cif_path,
                    error=exc,
                )
            )
        self._update_frame_item_status(current_image_path)

    def _sync_current_state_views(self) -> None:
        self._updating_views = True
        try:
            display_image = self._display_image_for_current_state()
            current_state = self._workspace.current_state
            polygons = current_state.polygons if current_state else []
            self.polygon_editor.set_image(display_image)
            self.polygon_editor.set_polygons(polygons)
            self.polygon_editor.set_debug_candidates([])
            self.polygon_editor.set_via_debug_inspection_enabled(self._via_debug_inspection_enabled())
            self._sync_neighbor_frames()
            self._sync_extra_layers()
            self._refresh_gradient_overlay()
        finally:
            self._updating_views = False

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
            self._autosave_current_overlay_if_needed()
            item = self._find_image_list_item(image_path)
            if item is not None:
                self.image_list.setCurrentItem(item)
            else:
                self.load_image(image_path)

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
        self._prepared_image_pending_request = request
        self._prepared_image_pending_signature = signature
        self._refresh_busy_indicator()
        self._start_pending_prepared_image_update()

    def _start_pending_prepared_image_update(self) -> None:
        if self._prepared_image_running_request_id is not None or self._prepared_image_pending_request is None:
            return
        request = self._prepared_image_pending_request
        self._prepared_image_pending_request = None
        request_signature = self._prepared_image_pending_signature
        self._prepared_image_pending_signature = None
        self._prepared_image_request_serial += 1
        request_id = self._prepared_image_request_serial
        self._prepared_image_running_request_id = request_id
        self._prepared_image_running_signature = request_signature

        worker = PreparedImageRunnable(request_id=request_id, request=request)
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
        return PreviewProcessingRequest(
            image_path=self._workspace.current_image_path,
            pipeline_config=pipeline_config,
            contour_settings=self._current_contour_settings(),
            source_image=source_image,
            preprocessed_image=preprocessed_image,
        )

    def _preview_request_signature(self, request: PreviewProcessingRequest) -> tuple[str, str, str]:
        return build_preview_request_signature(request)

    def _prepared_image_request_signature(self, request: PreparedImageRequest) -> tuple[str, str]:
        return build_prepared_image_signature(request)

    def _queue_preview_processing(self, *, debounced: bool) -> None:
        request = self._build_preview_request()
        if request is None:
            self._append_log(self._tr("no_image_selected_log"))
            return
        signature = self._preview_request_signature(request)
        if signature == self._preview_running_signature or signature == self._preview_pending_signature:
            self._refresh_busy_indicator()
            return
        self._preview_pending_request = request
        self._preview_pending_signature = signature
        self._refresh_busy_indicator()
        if debounced:
            self._preview_update_timer.start()
            return
        self._preview_update_timer.stop()
        self._start_pending_preview_processing()

    def _start_pending_preview_processing(self) -> None:
        if self._preview_running_request_id is not None or self._preview_pending_request is None:
            return
        request = self._preview_pending_request
        self._preview_pending_request = None
        request_signature = self._preview_pending_signature
        self._preview_pending_signature = None
        self._preview_request_serial += 1
        request_id = self._preview_request_serial
        self._preview_running_request_id = request_id
        self._preview_running_signature = request_signature

        worker = PreviewProcessingRunnable(request_id=request_id, request=request)
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

    def _on_prepared_image_error(self, request_id: int, message: str) -> None:
        if request_id != self._prepared_image_running_request_id:
            return
        self._append_log(self._tr("processing_failed_log", error=message))

    def _on_prepared_image_finished(self, request_id: int) -> None:
        if request_id == self._prepared_image_running_request_id:
            self._prepared_image_running_request_id = None
            self._prepared_image_running_signature = None
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
        self._append_log(self._tr("processing_failed_log", error=message))

    def _on_preview_processing_finished(self, request_id: int) -> None:
        if request_id == self._preview_running_request_id:
            self._preview_running_request_id = None
            self._preview_running_signature = None
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
            self._update_frame_item_status(self._workspace.current_image_path)
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
        self.load_images(scan_image_files(directory))

    def set_input_directory(self, path: str) -> None:
        directory_state = load_input_directory(path, scan_images=scan_image_files)
        self.input_dir_edit.setText(directory_state.directory)
        self._save_persisted_paths()
        self.load_images(list(directory_state.image_paths))

    def set_cif_directory(self, path: str) -> None:
        directory_state = index_cif_directory(path)
        self.cif_dir_edit.setText(directory_state.directory)
        self._save_persisted_paths()
        self._workspace.set_cif_index(directory_state.indexed_paths)
        self._refresh_image_list_item_states()
        if directory_state.available:
            self._append_log(self._tr("cif_indexed_log", count=len(directory_state.indexed_paths)))
        else:
            self._append_log(self._tr("cif_directory_unavailable_log"))

        if self._workspace.current_image_path:
            try:
                self.load_image(self._workspace.current_image_path)
            except Exception as exc:
                self._append_log(self._tr("reload_with_cif_failed_log", error=exc))

    def set_output_directory(self, path: str) -> None:
        self.output_dir_edit.setText(path)
        self._save_persisted_paths()

    def set_dataset_directory(self, path: str) -> None:
        self.dataset_dir_edit.setText(path)
        self._save_persisted_paths()

    def load_images(self, paths: list[str]) -> None:
        normalized_paths = self._workspace.replace_image_selection(paths, is_supported_image=is_image_path)
        self._neighbor_image_cache.clear()
        self._viewed_image_paths.intersection_update(str(Path(path)) for path in normalized_paths)
        self._preview_update_timer.stop()
        self._preview_pending_request = None
        self._preview_pending_signature = None
        self._prepared_image_pending_request = None
        self._prepared_image_pending_signature = None
        self._refresh_busy_indicator()
        self.image_list.clear()
        for path in normalized_paths:
            item = QListWidgetItem(Path(path).name)
            item.setToolTip(f"Путь к файлу: {path}" if self._ui_language == "ru" else f"File path: {path}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._apply_frame_status_to_item(item, self._frame_status_for_image(path))
            self.image_list.addItem(item)
        if normalized_paths:
            self.image_list.setCurrentRow(0)
        else:
            self._sync_current_state_views()

    def _find_matching_cif_path(self, image_path: str) -> str | None:
        return self._workspace.resolve_cif_path(image_path)

    def _load_cif_overlay_polygons(self, image_path: str) -> list[PolygonData]:
        cif_path = self._find_matching_cif_path(image_path)
        if not cif_path:
            return []
        try:
            referenced_image, image_size, polygons = load_polygons_cif(cif_path)
        except Exception as exc:
            self._append_log(self._tr("cif_load_failed_log", file_name=Path(cif_path).name, error=exc))
            return []
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
        self._preview_update_timer.stop()
        self._preview_pending_request = None
        self._preview_pending_signature = None
        self._prepared_image_pending_request = None
        self._prepared_image_pending_signature = None
        self._refresh_busy_indicator()
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
            return
        self._sync_current_state_views()
        self._update_frame_item_status(image_result.image_path)
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
