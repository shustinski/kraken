from __future__ import annotations

import threading
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import (
    QEvent,
    QPointF,
    QRectF,
    QSignalBlocker,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QBrush, QCloseEvent, QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QListView,
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

from .adapters.qt.image_conversion import cv_to_qimage
from .adapters.qt.preview import AutoTuneRunnable, PreparedImageRunnable, PreviewProcessingRunnable
from .adapters.qt.thumbnails import ThumbnailLoadRunnable
from .application.dto import PersistedPaths
from .application.extraction_profiles import default_contour_settings_profiles
from .application.frame_asset_sync import (
    build_frame_asset_sets,
    build_image_cif_matching_report,
    classify_vector_side_status,
    index_cif_file_paths,
)
from .application.frame_layers import (
    build_additional_layer_frame_map,
    build_base_frame_number_map,
    build_base_frame_records,
)
from .application.processing import (
    VIA_SEARCH_MODE_BRIGHT_TOPHAT_DOG,
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
    DirectoryScanController,
    PathSettingsController,
    VectorIndexController,
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
from .gamification import GamificationProfileService, GamificationService
from .i18n import active_language, tr
from .infrastructure import (
    WidgetDisplaySettingsStore,
    WidgetGamificationProfileStore,
    WidgetPathSettingsStore,
    WidgetSessionSettingsStore,
    WidgetViaPresetSettingsStore,
)
from .pipeline import (
    PreprocessingPipeline,
    available_operations,
    get_choice_display_label,
    get_operation_descriptor,
    get_operation_display_name,
    get_parameter_display_label,
)
from .serializers import load_polygons_vector, save_polygons_vector, save_result_bundle
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
from .ui.frame_path_list_model import FramePathFilterProxyModel, FramePathListModel
from .ui.item_status_painting import FRAME_STATUS_ROLE, paint_image_row_item, paint_vector_row_item
from .ui.large_dataset import LARGE_FRAME_COUNT_THRESHOLD
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
from .utils import is_image_path, is_visible_image_path, load_image_color, scan_image_files

from .widget_parts import (
    WidgetDebugMixin,
    WidgetExtractionControlsMixin,
    WidgetExtractionSettingsMixin,
    WidgetHelpMixin,
    WidgetNavigationMixin,
    WidgetPipelineActionsMixin,
    WidgetPipelineMixin,
    WidgetProcessingMixin,
    WidgetSettingsMixin,
    WidgetUiHelpersMixin,
)

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


class PolygonExtractionWidget(
    WidgetSettingsMixin,
    WidgetHelpMixin,
    WidgetExtractionControlsMixin,
    WidgetUiHelpersMixin,
    WidgetPipelineMixin,
    WidgetDebugMixin,
    WidgetPipelineActionsMixin,
    WidgetNavigationMixin,
    WidgetExtractionSettingsMixin,
    WidgetProcessingMixin,
    QWidget,
):
    _ui_theme: str
    show_frame_matrix_checkbox: QCheckBox
    show_neighbor_vectors_checkbox: QCheckBox
    image_list: QListView

    imageProcessed = pyqtSignal(str, list)
    batchProgress = pyqtSignal(int, int)
    batchFinished = pyqtSignal()
    polygonsEdited = pyqtSignal()
    logMessage = pyqtSignal(str)
    workSimulationActiveChanged = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._apply_contour_application_icon()
        self.setObjectName("polygonExtractionWidget")
        self._ui_language = active_language()
        self._ui_theme = "dark"
        self._path_settings = PathSettingsController(WidgetPathSettingsStore())
        self._display_settings_store = WidgetDisplaySettingsStore()
        self._session_settings_store = WidgetSessionSettingsStore()
        self._via_preset_settings_store = WidgetViaPresetSettingsStore()
        self._gamification_profile_service = GamificationProfileService(WidgetGamificationProfileStore())
        self._gamification_service = GamificationService(self._gamification_profile_service)
        self._workspace = WorkspaceSession()
        self._pipeline = PreprocessingPipeline()
        self._display_settings = DisplaySettings()
        self._contour_settings_profiles = default_contour_settings_profiles()
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
        self._preview_running_request_for_progress: PreviewProcessingRequest | None = None
        self._busy_progress_timer = QTimer(self)
        self._busy_progress_timer.setInterval(250)
        self._busy_progress_timer.timeout.connect(self._advance_busy_progress)
        self._busy_progress_value = 0
        self._busy_progress_stage = ""
        self._help_menu: QMenu | None = None
        self._color_pick_pipeline_row: int | None = None
        self._via_template_images: list[np.ndarray] = []
        self._viewed_image_paths: set[str] = set()
        self._user_via_presets: dict[str, dict[str, object]] = self._load_user_via_presets()
        self._extra_layers: list[dict[str, object]] = []
        self._next_extra_layer_id = 1
        self._base_frame_numbers: set[int] = set()
        self._base_frame_number_by_path: dict[str, int] = {}
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
        self._neighbor_image_dimensions: dict[str, tuple[int, int]] = {}
        self._neighbor_vector_cache: dict[str, tuple[list[PolygonData], tuple[int, int] | None]] = {}
        self._neighbor_thread_pool = QThreadPool(self)
        self._neighbor_thread_pool.setMaxThreadCount(2)
        self._neighbor_thread_pool.setExpiryTimeout(30000)
        self._neighbor_sync_image_path: str | None = None
        self._neighbor_frame_specs: list[tuple[int, int, str]] = []
        self._neighbor_queued_paths: set[str] = set()
        self._neighbor_sync_timer = QTimer(self)
        self._neighbor_sync_timer.setSingleShot(True)
        self._neighbor_sync_timer.timeout.connect(self._sync_neighbor_frames)
        self._neighbor_apply_timer = QTimer(self)
        self._neighbor_apply_timer.setSingleShot(True)
        self._neighbor_apply_timer.timeout.connect(self._apply_cached_neighbor_frames)
        self._thumbnail_thread_pool = QThreadPool(self)
        self._thumbnail_thread_pool.setMaxThreadCount(3)
        self._thumbnail_thread_pool.setExpiryTimeout(30000)
        self._frame_load_thread_pool = QThreadPool(self)
        self._frame_load_thread_pool.setMaxThreadCount(2)
        self._frame_load_thread_pool.setExpiryTimeout(30000)
        self._frame_load_request_serial = 0
        self._frame_load_running_path: str | None = None
        self._frame_load_pending: tuple[str, bool] | None = None
        self._defer_vector_load_until_cif_index = False
        self._pending_thumbnail_rebuild_after_vectors = False
        self._thumbnail_flush_retry_count = 0
        self._editor_display_thread_pool = QThreadPool(self)
        self._editor_display_thread_pool.setMaxThreadCount(1)
        self._editor_display_request_serial = 0
        self._editor_pixmap_cache: dict[tuple[str, str], QPixmap] = {}
        self._editor_polygons_signature: tuple[str, int, int] | None = None
        self._pending_editor_frame_apply: tuple[str, list, bool] | None = None
        self._frame_switch_profile = None
        self._frame_switch_profile_generation = 0
        self._thumbnail_path_to_row: dict[str, int] = {}
        self._pending_frame_chrome_path: str | None = None
        self._thumbnail_generation = 0
        self._thumbnail_icon_size = QSize(64, 48)
        self._thumbnail_placeholder_icon = QIcon()
        self._thumbnail_loaded_generation: dict[str, int] = {}
        self._thumbnail_loaded_sizes: dict[str, tuple[int, int]] = {}
        self._thumbnail_queued_paths: set[str] = set()
        self._thumbnail_queued_sizes: dict[str, tuple[int, int]] = {}
        self._thumbnail_rebuild_in_progress = False
        self._thumbnail_selected_path: str | None = None
        self._thumbnail_build_chunk_size = 50
        self._thumbnail_build_interval_ms = 25
        self._thumbnail_pending_apply: dict[str, object] = {}
        self._thumbnail_icon_cache: dict[object, QIcon] = {}
        self._thumbnail_disk_cache_dir = Path(tempfile.gettempdir()) / "contour-frame-thumbnails"
        self._thumbnail_disk_cache_key: str | None = None
        self._thumbnail_disk_cache_dir.mkdir(parents=True, exist_ok=True)
        self._thumbnail_apply_timer = QTimer(self)
        self._thumbnail_apply_timer.setSingleShot(True)
        self._thumbnail_apply_timer.timeout.connect(self._flush_thumbnail_icon_batch)
        self._thumbnail_visible_load_timer = QTimer(self)
        self._thumbnail_visible_load_timer.setSingleShot(True)
        self._thumbnail_visible_load_timer.timeout.connect(self._schedule_visible_thumbnail_loads)
        self._thumbnail_scroll_settle_timer = QTimer(self)
        self._thumbnail_scroll_settle_timer.setSingleShot(True)
        self._thumbnail_scroll_settle_timer.timeout.connect(self._on_thumbnail_scroll_settled)
        self._thumbnail_radial_paths: list[str] = []
        self._thumbnail_radial_cursor = 0
        self._thumbnail_radial_center_path: str | None = None
        self._thumbnail_radial_pump_timer = QTimer(self)
        self._thumbnail_radial_pump_timer.setSingleShot(True)
        self._thumbnail_radial_pump_timer.timeout.connect(self._pump_thumbnail_radial_loads)
        self._show_source_while_middle_held = False

        self._persisted_highlight_paths: set[str] = set()
        self._cif_load_failure_stems: set[str] = set()
        self._closing = False
        self._loading_image_path: str | None = None
        self._pending_restore_current_image_path = self._session_settings_store.load_current_image_path()
        self._directory_scan_append_mode = False
        self._directory_scanner = DirectoryScanController(self)
        self._directory_scanner.started.connect(self._on_input_directory_scan_started)
        self._directory_scanner.idle.connect(self._on_input_directory_scan_idle)
        self._directory_scanner.finished.connect(self._on_input_directory_scan_finished)
        self._directory_scanner.failed.connect(self._on_input_directory_scan_failed)
        self._vector_indexer = VectorIndexController(self)
        self._vector_indexer.started.connect(self._on_cif_directory_index_started)
        self._vector_indexer.idle.connect(self._on_cif_directory_index_idle)
        self._vector_indexer.finished.connect(self._on_cif_directory_index_finished)
        self._vector_indexer.failed.connect(self._on_cif_directory_index_failed)
        self._vectors_list_ignore_navigate_until: float = 0.0
        self._image_list_build_generation = 0
        self._image_list_build_chunk_size = 250
        self._image_list_rebuild_in_progress = False
        self._pending_image_list_post_build: dict[str, object] | None = None
        self._asset_list_build_generation = 0
        self._pending_cif_directory_state: object | None = None
        self._pending_cif_directory_path_after_images: str | None = None
        self._indexed_cif_directory: str | None = None
        self._asset_filter_match_only = False
        self._image_path_to_index: dict[str, int] = {}
        self._work_simulation_interval_ms = 200
        self._work_simulation_timer = QTimer(self)
        self._work_simulation_timer.setSingleShot(False)
        self._work_simulation_timer.timeout.connect(self._advance_work_simulation)
        self._work_simulation_running = False
        self._work_simulation_paths: list[str] = []
        self._work_simulation_path_index = -1
        self._work_simulation_target_polygons: list[PolygonData] = []
        self._work_simulation_visible_points = 0
        self._work_simulation_total_points = 0
        self._work_simulation_original_dirty: bool | None = None
        self._work_simulation_original_reference_polygons: list[PolygonData] = []

        self._batch_processor = BatchProcessor(self)
        self._batch_processor.set_ui_language(self._ui_language)
        self._batch_processor.resultReady.connect(self._on_batch_result)
        self._batch_processor.progressChanged.connect(self._on_batch_progress)
        self._batch_processor.finished.connect(self._on_batch_finished)
        self._batch_processor.errorOccurred.connect(self._on_batch_error)
        self._batch_processor.logMessage.connect(self._append_log)
        self._batch_controller = BatchController(self._batch_processor)

        self._build_ui()
        self._image_list_model = FramePathListModel(self)
        self._image_list_proxy = FramePathFilterProxyModel(self)
        self._image_list_proxy.setSourceModel(self._image_list_model)
        self.image_list.setModel(self._image_list_proxy)
        image_list_selection = self.image_list.selectionModel()
        if image_list_selection is not None:
            image_list_selection.currentChanged.connect(self._on_image_list_current_changed)
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
        self._update_extra_layers_enabled_state()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._closing = True
        self._frame_load_request_serial = int(getattr(self, "_frame_load_request_serial", 0)) + 1
        self._frame_load_pending = None
        self._frame_load_running_path = None
        self._loading_image_path = None
        if hasattr(self, "_cancel_thumbnail_loading"):
            self._cancel_thumbnail_loading()
        self._persist_session_state()
        super().closeEvent(event)
