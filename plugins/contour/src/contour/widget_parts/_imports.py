from __future__ import annotations

import threading
import time
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
from PyQt6.QtGui import QBrush, QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
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

from ..adapters.qt.image_conversion import cv_to_qimage
from ..adapters.qt.preview import AutoTuneRunnable, PreparedImageRunnable, PreviewProcessingRunnable
from ..adapters.qt.thumbnails import ThumbnailLoadRunnable
from ..application.dto import PersistedPaths
from ..application.extraction_profiles import default_contour_settings_profiles
from ..application.frame_asset_sync import (
    build_image_cif_matching_report,
    classify_vector_side_status,
    index_cif_file_paths,
)
from ..application.frame_layers import (
    build_additional_layer_frame_map,
    build_base_frame_number_map,
    build_base_frame_records,
)
from ..application.polygon_antialiasing import antialias_polygons
from ..application.processing import (
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
from ..application.services import (
    BatchController,
    BatchStartRequest,
    DirectoryScanController,
    PathSettingsController,
    WorkspaceSession,
    export_frame_to_dataset,
    load_pipeline_config_from_path,
    save_pipeline_config_to_path,
)
from ..application.transition_save_guard import (
    TransitionPromptChoice,
    navigation_allowed_after_autosave_attempt,
    navigation_allowed_after_prompt,
)
from ..application.use_cases import (
    AutoTuneResult,
    PreparedImageRequest,
    PreviewProcessingRequest,
    build_prepared_image_signature,
    build_preview_request_signature,
    index_cif_directory,
)
from ..application.vector_geometry_postprocess import VectorGeometrySettings
from ..batch_processor import BatchProcessor
from ..domain import PolygonData
from ..graphics.editor_hotkeys import (
    append_shortcut_to_tooltip,
    build_editor_hotkeys_plain_text,
    tool_shortcut_native_text,
)
from ..graphics_view import BrushMode, DeleteVertexMode, EditorTool, PolygonCreateMode
from ..gamification import (
    CorrectionEvent,
    CorrectionType,
    GamificationProfileService,
    GamificationService,
    RewardEventType,
)
from ..gamification.ui import GamificationPanel
from ..i18n import active_language, tr
from ..infrastructure import (
    WidgetDisplaySettingsStore,
    WidgetGamificationProfileStore,
    WidgetPathSettingsStore,
    WidgetSessionSettingsStore,
    WidgetViaPresetSettingsStore,
)
from ..pipeline import (
    PreprocessingPipeline,
    available_operations,
    get_choice_display_label,
    get_operation_descriptor,
    get_operation_display_name,
    get_parameter_display_label,
)
from ..serializers import load_polygons_vector, save_polygons_vector, save_result_bundle
from ..ui.builders import (
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
from ..ui.editor_icons import (
    TOOLBAR_BUTTON_SIZE_PX,
    TOOLBAR_ICON_CANVAS_SIZE_PX,
    TOOLBAR_ICON_SIZE_PX,
    create_editor_action_icon,
    create_editor_tool_icon,
)
from ..ui.item_status_painting import FRAME_STATUS_ROLE, paint_image_row_item, paint_vector_row_item
from ..ui.i18n_content import (
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
from ..ui.pipeline_presets import built_in_pipeline_presets
from ..ui.retranslate import retranslate_ui
from ..ui.styles import COMPACT_UI_STYLE
from ..ui.via_presets import (
    blurred_via_preset_payload,
    built_in_via_presets,
    noisy_traces_via_preset_payload,
)
from ..utils import is_image_path, is_visible_image_path, load_image_color, scan_image_files

__all__ = [name for name in globals() if not name.startswith("__")]


