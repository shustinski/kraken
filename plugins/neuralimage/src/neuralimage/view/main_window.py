import os
import math
import time
from pathlib import Path
from PyQt6.QtCore import QEvent, QSize, Qt, QSettings, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QFontMetrics, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QDockWidget,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from neuralimage.application.dto import MainWindowState
from neuralimage.UI import ClickableLabel
import numpy as np
from neuralimage.lib.data_interfaces import WorkMode
from neuralimage.lib.logging_policy import MAX_LOG_MESSAGES
from neuralimage.lib.runtime_paths import resolve_internal_path
from neuralimage.lib.shared_styles import load_stylesheet, resolve_shared_style_path
from neuralimage.lib.ui_texts import get_ui_language, get_ui_section
from neuralimage.lib.version import get_app_title
from neuralimage.view.changelog_dialog import show_changelog_dialog
from neuralimage.view.help_dialog import show_help_dialog
from neuralimage.view.metrics_panel import TrainingMetricsDock
from neuralimage.view.settings_panel import create_spinbox
from neuralimage.view.tic_tac_toe_dialog import TicTacToeDialog

from .main_window_parts.module_state import (
    WINDOW_LAYOUT_STATE_VERSION,
    PREVIEW_MINIMUM_SIZE,
)
from .main_window_parts.construction import ConstructionMixin
from .main_window_parts.setup_ui import SetupUiMixin
from .main_window_parts.style_action_button import StyleActionButtonMixin
from .main_window_parts.update_item import UpdateItemMixin
from .main_window_parts.row_progress import RowProgressMixin
from .main_window_parts.construction_2 import Construction2Mixin
from .main_window_parts.setup_ui_2 import SetupUi2Mixin
from .main_window_parts.restore_window_layout import RestoreWindowLayoutMixin
from .main_window_parts.create_menubar import CreateMenubarMixin
from .main_window_parts.apply_work_mode import ApplyWorkModeMixin
from .main_window_parts.connect_internal_signals import ConnectInternalSignalsMixin
from .main_window_parts.configure_update_channels import ConfigureUpdateChannelsMixin
from .main_window_parts.append_metrics import AppendMetricsMixin
from .main_window_parts.update_memory_runtime_label import UpdateMemoryRuntimeLabelMixin
from .main_window_parts.task_queue_items import TaskQueueItemsMixin
from .main_window_parts.apply_ui_language import ApplyUiLanguageMixin
from .main_window_parts.update_task_queue_item import UpdateTaskQueueItemMixin
from .main_window_parts.responsive_preview_label import (
    _main_window_qsettings,
    _load_persisted_ui_mode,
    _save_persisted_ui_mode,
    ResponsivePreviewLabel,
    _modal_dialogs_enabled,
    load_qss_from_resource,
    _load_menu_icon,
    TaskQueueListWidget,
)

class TaskQueueRowWidget(
    ConstructionMixin,
    SetupUiMixin,
    StyleActionButtonMixin,
    UpdateItemMixin,
    RowProgressMixin,
    QWidget,
):
    name_changed: pyqtSignal = pyqtSignal(int, str)
    pause_continue_requested: pyqtSignal = pyqtSignal(int)
    move_up_requested: pyqtSignal = pyqtSignal(int)
    move_down_requested: pyqtSignal = pyqtSignal(int)
    remove_requested: pyqtSignal = pyqtSignal(int)
    retry_requested: pyqtSignal = pyqtSignal(int)

class MainView(
    Construction2Mixin,
    SetupUi2Mixin,
    RestoreWindowLayoutMixin,
    CreateMenubarMixin,
    ApplyWorkModeMixin,
    ConnectInternalSignalsMixin,
    ConfigureUpdateChannelsMixin,
    AppendMetricsMixin,
    UpdateMemoryRuntimeLabelMixin,
    TaskQueueItemsMixin,
    ApplyUiLanguageMixin,
    UpdateTaskQueueItemMixin,
    QMainWindow,
):
    sample_type_changed: pyqtSignal = pyqtSignal(str)
    source_path_requested: pyqtSignal = pyqtSignal()
    result_path_requested: pyqtSignal = pyqtSignal()
    label_path_requested: pyqtSignal = pyqtSignal()
    jpg_path_requested: pyqtSignal = pyqtSignal()
    model_path_requested: pyqtSignal = pyqtSignal()
    open_config_requested: pyqtSignal = pyqtSignal()
    start_requested: pyqtSignal = pyqtSignal()
    stop_requested: pyqtSignal = pyqtSignal()
    queue_context_remove_requested: pyqtSignal = pyqtSignal(int)
    queue_properties_requested: pyqtSignal = pyqtSignal(int)
    queue_report_requested: pyqtSignal = pyqtSignal(int)
    queue_retry_requested: pyqtSignal = pyqtSignal(int)
    queue_pause_continue_requested: pyqtSignal = pyqtSignal(int)
    queue_move_up_requested: pyqtSignal = pyqtSignal(int)
    queue_move_down_requested: pyqtSignal = pyqtSignal(int)
    queue_rows_reordered: pyqtSignal = pyqtSignal(int, int)
    queue_task_name_changed: pyqtSignal = pyqtSignal(int, str)
    epochs_changed: pyqtSignal = pyqtSignal()
    recursive_file_search_changed: pyqtSignal = pyqtSignal()
    request_close: pyqtSignal = pyqtSignal()
    log_message: pyqtSignal = pyqtSignal(object)
    log_message_with_delete_last: pyqtSignal = pyqtSignal(object)
    metrics_message: pyqtSignal = pyqtSignal(object)
    enable_start: pyqtSignal = pyqtSignal(bool)
    show_info: pyqtSignal = pyqtSignal(str)
    show_warning: pyqtSignal = pyqtSignal(str)
    toggle_start_stop: pyqtSignal = pyqtSignal(bool)
    batch_preview_visibility_changed: pyqtSignal = pyqtSignal(bool)
    release_memory_requested: pyqtSignal = pyqtSignal()
    open_tic_tac_toe_requested: pyqtSignal = pyqtSignal()
    open_validation_gradient_requested: pyqtSignal = pyqtSignal()
    update_check_requested: pyqtSignal = pyqtSignal()
    update_channel_selected: pyqtSignal = pyqtSignal(str)
    ui_language_selected: pyqtSignal = pyqtSignal(str)
    theme_selected: pyqtSignal = pyqtSignal(str)
    ui_mode_selected: pyqtSignal = pyqtSignal(str)
    simple_workflow_requested: pyqtSignal = pyqtSignal(str)

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = MainView(QWidget())
    window.show()
    sys.exit(app.exec())

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    WINDOW_LAYOUT_STATE_VERSION,
    PREVIEW_MINIMUM_SIZE,
    _main_window_qsettings,
    _load_persisted_ui_mode,
    _save_persisted_ui_mode,
    ResponsivePreviewLabel,
    _modal_dialogs_enabled,
    load_qss_from_resource,
    _load_menu_icon,
    TaskQueueListWidget,
)
