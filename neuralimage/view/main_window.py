
import os
import math
import time
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QSettings
from PyQt6.QtGui import QAction, QActionGroup, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from application.dto import MainWindowState
from UI import ClickableLabel
import numpy as np
from lib.data_interfaces import WorkMode
from lib.logging_policy import MAX_LOG_MESSAGES
from lib.runtime_paths import resolve_internal_path
from lib.shared_styles import load_stylesheet, resolve_shared_style_path
from lib.ui_texts import get_ui_language, get_ui_section
from lib.version import get_app_title
from view.changelog_dialog import show_changelog_dialog
from view.help_dialog import show_help_dialog
from view.metrics_panel import TrainingMetricsDock
from view.settings_panel import create_spinbox
from view.tic_tac_toe_dialog import TicTacToeDialog




def _main_window_qsettings() -> QSettings:
    root = os.getenv('NEURALIMAGE_SETTINGS_DIR')
    if root:
        settings_root = Path(root)
        settings_root.mkdir(parents=True, exist_ok=True)
        return QSettings(
            str(settings_root / 'NeuralImage_MainWindow.ini'),
            QSettings.Format.IniFormat,
        )
    return QSettings('NeuralImage', 'MainWindow')


def _load_persisted_ui_mode() -> str:
    settings = _main_window_qsettings()
    value = settings.value('ui_mode', 'simple', type=str)
    settings.sync()
    return 'advanced' if value == 'advanced' else 'simple'


def _save_persisted_ui_mode(mode: str) -> None:
    settings = _main_window_qsettings()
    settings.setValue('ui_mode', 'advanced' if mode == 'advanced' else 'simple')
    settings.sync()


def _modal_dialogs_enabled() -> bool:
    disabled_flag = str(os.getenv('NEURALIMAGE_DISABLE_MODAL_DIALOGS', '') or '').strip().lower()
    if disabled_flag in {'1', 'true', 'yes', 'on'}:
        return False
    return not bool(os.getenv('PYTEST_CURRENT_TEST'))


def load_qss_from_resource(qss_path: str):
    return load_stylesheet(qss_path)


def _load_menu_icon() -> QIcon:
    return QIcon(str(resolve_internal_path('settings_icon.png')))


class MainView(QMainWindow):
    sample_type_changed: pyqtSignal = pyqtSignal(str)

    source_path_requested: pyqtSignal = pyqtSignal()
    result_path_requested: pyqtSignal = pyqtSignal()

    label_path_requested: pyqtSignal = pyqtSignal()
    jpg_path_requested: pyqtSignal = pyqtSignal()

    model_path_requested: pyqtSignal = pyqtSignal()
    open_config_requested: pyqtSignal = pyqtSignal()

    start_requested: pyqtSignal = pyqtSignal()
    stop_requested: pyqtSignal = pyqtSignal()
    queue_remove_requested: pyqtSignal = pyqtSignal()
    queue_pause_toggle_requested: pyqtSignal = pyqtSignal()
    queue_context_remove_requested: pyqtSignal = pyqtSignal(int)
    queue_properties_requested: pyqtSignal = pyqtSignal(int)

    epochs_changed: pyqtSignal = pyqtSignal()
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
    developer_tools_requested: pyqtSignal = pyqtSignal()
    update_check_requested: pyqtSignal = pyqtSignal()
    update_channel_selected: pyqtSignal = pyqtSignal(str)
    ui_language_selected: pyqtSignal = pyqtSignal(str)
    theme_selected: pyqtSignal = pyqtSignal(str)
    ui_mode_selected: pyqtSignal = pyqtSignal(str)
    simple_workflow_requested: pyqtSignal = pyqtSignal(str)

    def __init__(self, side_panel: QWidget | None = None):
        super().__init__()
        self.setWindowTitle(get_app_title())
        self.setWindowIcon(QIcon(str(resolve_internal_path('icon.png'))))
        self.setGeometry(200, 200, 1200, 740)
        

        self.settings_dock = side_panel if isinstance(side_panel, QDockWidget) else None
        self._close_allowed = False
        self.log_scroll: QScrollArea | None = None

        self._batch_points_by_epoch: dict[int, list[tuple[float, float]]] = {}
        self._tic_tac_toe_dialog: TicTacToeDialog | None = None
        self._ram_mb: float | None = None
        self._vram_alloc_mb: float | None = None
        self._vram_reserved_mb: float | None = None
        self._train_speed_batches_per_sec: float | None = None
        self._recognition_speed_images_per_sec: float | None = None
        self._sample_count_value = 0
        self._sample_count_pending = False
        self._current_preview_sample_name = ""
        self._current_preview_mode = "train"
        self._last_status_message = ""
        self._recognition_started_at: float | None = None
        self._recognition_last_current = 0
        self._recognition_last_total = 0
        self._last_validation_metrics: tuple[float, float, float] | None = None
        self._last_performance_metrics: dict[str, float] | None = None
        self._ui_language = get_ui_language()
        self._theme = "dark"
        self._file_menu = None
        self._settings_menu = None
        self._view_menu = None
        self._plugins_menu = None
        self._tools_menu = None
        self._info_menu = None
        self._language_menu = None
        self._theme_menu = None
        self._open_config_action = None
        self._settings_sample_action = None
        self._settings_train_action = None
        self._settings_pred_action = None
        self._metrics_toggle_action = None
        self._log_toggle_action = None
        self._settings_toggle_action = None
        self._help_action = None
        self._changelog_action = None
        self._check_updates_action = None
        self._update_channel_menu = None
        self._update_channel_action_group = None
        self._update_channel_actions: dict[str, QAction] = {}
        self._available_update_channels: list[str] = []
        self._selected_update_channel = 'stable'
        self._open_validation_gradient_action = None
        self._developer_tools_action = None
        self._ui_mode_menu = None
        self._ui_mode_simple_action = None
        self._ui_mode_advanced_action = None
        self._central_scroll: QScrollArea | None = None
        self._central_content: QWidget | None = None
        self._ui_mode = _load_persisted_ui_mode()
        self._current_work_mode = ''
        self._selected_simple_workflow: str | None = None

        self._setup_ui()

    def _setup_ui(self):

        t = get_ui_section("main_window")
        self._texts = t
        self._central_content = QWidget(self)
        self.main_grid = QGridLayout(self._central_content)

        row = 0
        self.main_grid.setColumnStretch(0, 1)
        self.main_grid.setColumnStretch(1, 10)
        self.sample_count_top_label = QLabel(t.get("samples_count", "Кадров в выборке: 0"))
        self.sample_count_top_label.hide()

        self.work_mode_group = QGroupBox(t["mode"])
        sample_type_layout = QHBoxLayout(self.work_mode_group)

        self.rb_train_and_recognition = QRadioButton(t["mode_train_and_rec"])
        self.rb_further_train_model = QRadioButton(t["mode_ft_and_rec"])
        self.rb_recognition = QRadioButton(t["mode_rec"])
        self.rb_train_only = QRadioButton(t["mode_train"])

        sample_type_layout.addWidget(self.rb_train_and_recognition)
        sample_type_layout.addWidget(self.rb_further_train_model)
        sample_type_layout.addWidget(self.rb_recognition)
        sample_type_layout.addWidget(self.rb_train_only)
        self.main_grid.addWidget(self.work_mode_group, row, 0, 1, 2)

        row += 1
        self.source_title_label = QLabel(t["source"])
        self.main_grid.addWidget(self.source_title_label, row, 0)
        self.lbl_source = ClickableLabel()
        self.main_grid.addWidget(self.lbl_source, row, 1)

        row += 1
        self.result_title_label = QLabel(t["result"])
        self.main_grid.addWidget(self.result_title_label, row, 0)
        self.lbl_result = ClickableLabel()
        self.main_grid.addWidget(self.lbl_result, row, 1)

        row += 1
        self.sample_path_group = QGroupBox(t["sample"])
        self.main_grid.addWidget(self.sample_path_group, row, 0, 1, 2)
        sample_path_form = QFormLayout(self.sample_path_group)

        self.sample_path = ClickableLabel()
        self.sample_path.setToolTip(t["sample_tip"])
        sample_path_form.addRow(t["sample_src"], self.sample_path)
        self.sample_src_title_label = sample_path_form.labelForField(self.sample_path)

        self.label_path = ClickableLabel()
        self.label_path.setToolTip(t["label_tip"])
        sample_path_form.addRow(t["labels"], self.label_path)
        self.label_path_title_label = sample_path_form.labelForField(self.label_path)

        row += 1
        self.model_title_label = QLabel(t["model"])
        self.main_grid.addWidget(self.model_title_label, row, 0)
        self.model_path = ClickableLabel()
        self.main_grid.addWidget(self.model_path, row, 1)

        self.epochs_title_label = QLabel(t["epochs"])
        self.epochs_title_label.hide()
        self.le_epochs = create_spinbox((0, 1000), 1, 40)
        self.le_epochs.hide()

        row += 1
        self.buttons_row = QWidget()
        buttons_layout = QHBoxLayout(self.buttons_row)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_start = QPushButton(t["start"])
        self.btn_start.setEnabled(True)
        self.btn_start.setStyleSheet(
            """background-color: #C62828; color: white; padding: 8px 20px;
               border: none; border-radius: 5px;"""
        )
        self.btn_stop = QPushButton(t["stop"])
        self.btn_stop.setVisible(False)
        buttons_layout.addWidget(self.btn_start)
        buttons_layout.addWidget(self.btn_stop)
        self.main_grid.addWidget(self.buttons_row, row, 0, 1, 2)

        row += 1
        self.simple_workflows_group = QGroupBox(t.get("simple_workflows_group", "Simple workflows"))
        simple_workflows_root_layout = QVBoxLayout(self.simple_workflows_group)
        simple_workflows_layout = QHBoxLayout()
        self.btn_simple_conductors = QPushButton(
            t.get("simple_workflow_conductors", "Conductor recognition")
        )
        self.btn_simple_contacts = QPushButton(
            t.get("simple_workflow_contacts", "Contact recognition")
        )
        self.btn_simple_memory = QPushButton(
            t.get("simple_workflow_memory", "Memory recognition")
        )
        simple_workflows_layout.addWidget(self.btn_simple_conductors)
        simple_workflows_layout.addWidget(self.btn_simple_contacts)
        simple_workflows_layout.addWidget(self.btn_simple_memory)
        simple_workflows_root_layout.addLayout(simple_workflows_layout)
        self.simple_workflow_label = QLabel()
        simple_workflows_root_layout.addWidget(self.simple_workflow_label)
        self.main_grid.addWidget(self.simple_workflows_group, row, 0, 1, 2)
        self.simple_workflows_group.setVisible(False)
        self._update_simple_workflow_label()

        row += 1
        self.queue_group = QGroupBox(t["queue"])
        queue_layout = QVBoxLayout(self.queue_group)
        self.queue_list = QListWidget()
        self.queue_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        queue_layout.addWidget(self.queue_list)
        queue_buttons_layout = QHBoxLayout()
        self.btn_queue_remove = QPushButton(t["queue_remove"])
        self.btn_queue_pause_toggle = QPushButton(t["queue_pause"])
        queue_buttons_layout.addWidget(self.btn_queue_remove)
        queue_buttons_layout.addWidget(self.btn_queue_pause_toggle)
        queue_layout.addLayout(queue_buttons_layout)
        self.main_grid.addWidget(self.queue_group, row, 0, 1, 2)

        row += 1
        self.progress_group = QGroupBox(t["progress_group"])
        progress_layout = QFormLayout(self.progress_group)
        self.epoch_progress_bar = QProgressBar()
        self.batch_progress_bar = QProgressBar()
        self.recognition_progress_bar = QProgressBar()
        for progress_bar in (self.epoch_progress_bar, self.batch_progress_bar, self.recognition_progress_bar):
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_bar.setFormat("%p%")
        progress_layout.addRow(t["progress_epochs"], self.epoch_progress_bar)
        self.progress_epochs_title_label = progress_layout.labelForField(self.epoch_progress_bar)
        progress_layout.addRow(t["progress_batches"], self.batch_progress_bar)
        self.progress_batches_title_label = progress_layout.labelForField(self.batch_progress_bar)
        recognition_speed_default = (
            "Recognition speed: —" if self._ui_language == "en" else "Скорость распознавания: —"
        )
        self.recognition_speed_label = QLabel(t.get("recognition_speed_default", recognition_speed_default))
        progress_layout.addRow(self.recognition_speed_label)
        progress_layout.addRow(t["progress_recognition"], self.recognition_progress_bar)
        self.progress_recognition_title_label = progress_layout.labelForField(self.recognition_progress_bar)
        self.memory_usage_label = QLabel(t["memory_label_default"])
        progress_layout.addRow(self.memory_usage_label)
        self.validation_quality_label = QLabel(t["validation_quality_default"])
        progress_layout.addRow(self.validation_quality_label)
        self.performance_label = QLabel(t["performance_label_default"])
        progress_layout.addRow(self.performance_label)
        self.main_grid.addWidget(self.progress_group, row, 0, 1, 2)

        row += 1
        self.preview_group = QGroupBox(t["preview_group"])
        preview_layout = QVBoxLayout(self.preview_group)
        self.preview_frame_name_label = QLabel(t.get("preview_current_frame_default", "Кадр: —"))
        self.preview_frame_name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        preview_layout.addWidget(self.preview_frame_name_label)

        preview_row = QHBoxLayout()
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.setSpacing(8)

        self.preview_image_title_label = QLabel(t["preview_image"])
        self.preview_label_title_label = QLabel(t["preview_label"])
        self.preview_output_title_label = QLabel(t["preview_output"])
        self.preview_image_label = QLabel()
        self.preview_label_label = QLabel()
        self.preview_output_label = QLabel()
        for preview in (self.preview_image_label, self.preview_label_label, self.preview_output_label):
            preview.setFixedSize(220, 220)
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setStyleSheet("border: 1px solid #666; background: #111;")
        for title in (self.preview_image_title_label, self.preview_label_title_label, self.preview_output_title_label):
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for attr_prefix, title, preview in (
            ("preview_image", self.preview_image_title_label, self.preview_image_label),
            ("preview_label", self.preview_label_title_label, self.preview_label_label),
            ("preview_output", self.preview_output_title_label, self.preview_output_label),
        ):
            column_widget = QWidget()
            column_layout = QVBoxLayout(column_widget)
            column_layout.setContentsMargins(0, 0, 0, 0)
            column_layout.setSpacing(4)
            column_layout.addWidget(title)
            column_layout.addWidget(preview)
            setattr(self, f"{attr_prefix}_column_widget", column_widget)
            preview_row.addWidget(column_widget)
        preview_layout.addLayout(preview_row)
        self.main_grid.addWidget(self.preview_group, row, 0, 1, 2)

        row += 1
        self.log_scroll = QScrollArea()
        self.log_scroll.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.log_scroll.setWidgetResizable(True)

        self.log_container = QWidget()
        self.log_layout = QVBoxLayout(self.log_container)
        self.log_layout.setContentsMargins(5, 5, 5, 5)
        self.log_layout.setSpacing(2)
        self.log_scroll.setWidget(self.log_container)
        self.log_dock = QDockWidget(t.get("log_dock_title", "Лог"), self)
        self.log_dock.setObjectName("logDock")
        self.log_dock.setWidget(self.log_scroll)
        self.log_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.main_grid.setRowStretch(row, 10)

        self._central_scroll = QScrollArea(self)
        self._central_scroll.setWidgetResizable(True)
        self._central_scroll.setWidget(self._central_content)
        self.setCentralWidget(self._central_scroll)
        self.statusBar().showMessage("")

        self.metrics_panel = TrainingMetricsDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.metrics_panel)
        self.metrics_panel.setMinimumHeight(220)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.log_dock)
        self.tabifyDockWidget(self.metrics_panel, self.log_dock)

        if self.settings_dock is not None:
            self.settings_dock.setWindowTitle(t["settings_dock_title"])
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.settings_dock)
            self.tabifyDockWidget(self.metrics_panel, self.settings_dock)
            self.tabifyDockWidget(self.log_dock, self.settings_dock)
            self.settings_dock.show()
            self.settings_dock.raise_()

        self._create_menubar(t)
        self._apply_theme(self._theme)

    def _create_menubar(self, t: dict[str, str]):
        menubar = self.menuBar()
        if menubar is None:
            return
        file_menu = menubar.addMenu(t.get("menu_file", "Файл"))
        settings_menu = menubar.addMenu(t["menu_settings"])
        self._view_menu = menubar.addMenu(t["menu_view"])
        view_menu = self._view_menu
        self._plugins_menu = menubar.addMenu(t.get("menu_plugins", "Plugins"))
        plugins_menu = self._plugins_menu
        self._tools_menu = menubar.addMenu(t.get("menu_tools", "Инструменты"))
        tools_menu = self._tools_menu
        info_menu = menubar.addMenu(t["menu_help"])
        self._file_menu = file_menu
        self._settings_menu = settings_menu
        self._info_menu = info_menu
        if file_menu is None or settings_menu is None or view_menu is None or plugins_menu is None or tools_menu is None or info_menu is None:
            return

        self._open_config_action = QAction(t.get("menu_open_config", "Открыть"), self)
        file_menu.addAction(self._open_config_action)
        menu_icon = _load_menu_icon()
        self._settings_sample_action = QAction(menu_icon, t["menu_sample"], self)
        self._settings_train_action = QAction(menu_icon, t["menu_train"], self)
        self._settings_pred_action = QAction(menu_icon, t["menu_pred"], self)
        settings_menu.addAction(self._settings_sample_action)
        settings_menu.addAction(self._settings_train_action)
        settings_menu.addAction(self._settings_pred_action)
        metrics_action = self.metrics_panel.toggleViewAction()
        if metrics_action is not None:
            metrics_action.setText(t["menu_metrics"])
            view_menu.addAction(metrics_action)
            self._metrics_toggle_action = metrics_action
        log_action = self.log_dock.toggleViewAction()
        if log_action is not None:
            log_action.setText(t.get("menu_log_panel", "Панель лога"))
            view_menu.addAction(log_action)
            self._log_toggle_action = log_action
        if self.settings_dock is not None:
            settings_action = self.settings_dock.toggleViewAction()
            if settings_action is not None:
                settings_action.setText(t["menu_settings_panel"])
                view_menu.addAction(settings_action)
                self._settings_toggle_action = settings_action
        self.batch_preview_action = QAction(t["menu_batch_preview"], self)
        self.batch_preview_action.setCheckable(True)
        self.batch_preview_action.setChecked(True)
        view_menu.addAction(self.batch_preview_action)
        self.release_memory_action = QAction(t["menu_release_memory"], self)
        view_menu.addAction(self.release_memory_action)
        self.open_tic_tac_toe_action = QAction(
            t.get("menu_open_tic_tac_toe", "Крестики-нолики (нейросеть)"),
            self,
        )
        view_menu.addAction(self.open_tic_tac_toe_action)
        self._open_validation_gradient_action = QAction(
            t.get("menu_open_validation_gradient", "Open Validation gradient"),
            self,
        )
        plugins_menu.addAction(self._open_validation_gradient_action)
        self._developer_tools_action = QAction(t.get("menu_developer", "Разработчик"), self)
        tools_menu.addAction(self._developer_tools_action)
        view_menu.addSeparator()
        self._language_menu = view_menu.addMenu(t.get("menu_language", "Язык"))
        language_group = QActionGroup(self)
        language_group.setExclusive(True)
        self.ui_language_ru_action = QAction(t.get("lang_ru", "Русский"), self)
        self.ui_language_ru_action.setCheckable(True)
        self.ui_language_ru_action.setData("ru")
        language_group.addAction(self.ui_language_ru_action)
        self._language_menu.addAction(self.ui_language_ru_action)
        self.ui_language_en_action = QAction(t.get("lang_en", "English"), self)
        self.ui_language_en_action.setCheckable(True)
        self.ui_language_en_action.setData("en")
        language_group.addAction(self.ui_language_en_action)
        self._language_menu.addAction(self.ui_language_en_action)
        self._theme_menu = view_menu.addMenu(t.get("menu_theme", "Тема"))
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        self.theme_dark_action = QAction(t.get("theme_dark", "Темная"), self)
        self.theme_dark_action.setCheckable(True)
        self.theme_dark_action.setData("dark")
        theme_group.addAction(self.theme_dark_action)
        self._theme_menu.addAction(self.theme_dark_action)
        self.theme_light_action = QAction(t.get("theme_light", "Светлая"), self)
        self.theme_light_action.setCheckable(True)
        self.theme_light_action.setData("light")
        theme_group.addAction(self.theme_light_action)
        self._theme_menu.addAction(self.theme_light_action)
        self._ui_mode_menu = view_menu.addMenu(t.get("menu_ui_mode", "Interface mode"))
        ui_mode_group = QActionGroup(self)
        ui_mode_group.setExclusive(True)
        self._ui_mode_simple_action = QAction(t.get("menu_ui_mode_simple", "Simple"), self)
        self._ui_mode_simple_action.setCheckable(True)
        self._ui_mode_simple_action.setData("simple")
        ui_mode_group.addAction(self._ui_mode_simple_action)
        self._ui_mode_menu.addAction(self._ui_mode_simple_action)
        self._ui_mode_advanced_action = QAction(t.get("menu_ui_mode_advanced", "Advanced"), self)
        self._ui_mode_advanced_action.setCheckable(True)
        self._ui_mode_advanced_action.setData("advanced")
        ui_mode_group.addAction(self._ui_mode_advanced_action)
        self._ui_mode_menu.addAction(self._ui_mode_advanced_action)
        self._sync_language_menu_checks()
        self._sync_theme_menu_checks()
        self._sync_ui_mode_menu_checks()
        help_action = QAction(t["menu_open_help"], self)
        help_action.triggered.connect(lambda: show_help_dialog(self))
        info_menu.addAction(help_action)
        self._help_action = help_action
        changelog_action = QAction(t.get("menu_open_changelog", "Список изменений"), self)
        changelog_action.triggered.connect(lambda: show_changelog_dialog(self))
        info_menu.addAction(changelog_action)
        self._changelog_action = changelog_action
        check_updates_action = QAction(t.get("menu_check_updates", "Проверить обновления"), self)
        info_menu.addAction(check_updates_action)
        self._check_updates_action = check_updates_action
        self._update_channel_menu = info_menu.addMenu(t.get("menu_update_channel", "Канал обновлений"))
        self.configure_update_channels(self._available_update_channels or ['stable'], self._selected_update_channel)
        menu_action = info_menu.menuAction()
        if menu_action is not None:
            menu_action.setVisible(True)

    def set_batch_preview_enabled(self, enabled: bool) -> None:
        action = getattr(self, "batch_preview_action", None)
        if action is not None:
            action.setChecked(enabled)
        preview_group = getattr(self, "preview_group", None)
        if preview_group is not None:
            preview_group.setVisible(bool(enabled))

    def is_batch_preview_enabled(self) -> bool:
        action = getattr(self, "batch_preview_action", None)
        if action is None:
            return True
        return bool(action.isChecked())

    def _current_side_panel_width(self) -> int:
        if self.settings_dock is None or not self.settings_dock.isVisible():
            return 0
        return self.settings_dock.width()

    def _compute_target_side_panel_width(self) -> int:
        if self.settings_dock is None:
            return 0
        return self.settings_dock.width()

    def _show_side_panel_fully(self):
        if self.settings_dock is None:
            return
        self.settings_dock.show()
        self.settings_dock.raise_()

    def show_settings_dock(self) -> None:
        self._show_side_panel_fully()

    def _hide_side_panel(self):
        if self.settings_dock is None:
            return
        self.settings_dock.hide()

    def _toogle_button_clicked(self):
        if self.settings_dock is None:
            return
        panel_visible = self.settings_dock.isVisible()
        if panel_visible:
            self._hide_side_panel()
        else:
            self._show_side_panel_fully()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        return

    def apply_work_mode(self, mode: str) -> None:
        self._current_work_mode = str(mode or '').strip()
        resolved_mode = self._current_work_mode
        training_only = resolved_mode == WorkMode.train_only.value
        recognition_only = resolved_mode == WorkMode.recognition_only.value
        uses_model = resolved_mode in {
            WorkMode.recognition_only.value,
            WorkMode.further_training.value,
        }
        uses_epochs = resolved_mode in {
            WorkMode.train_only.value,
            WorkMode.train_and_recognition.value,
            WorkMode.further_training.value,
        }

        if not resolved_mode:
            self.lbl_source.setEnabled(True)
            self.lbl_result.setEnabled(True)
            self.sample_path_group.setEnabled(True)
            self.sample_path_group.setVisible(True)
            self.model_path.setEnabled(True)
        else:
            self.lbl_source.setEnabled(not training_only)
            self.lbl_result.setEnabled(not training_only)
            self.sample_path_group.setEnabled(not recognition_only)
            self.sample_path_group.setVisible(not recognition_only)
            self.model_path.setEnabled(uses_model)

        show_model = True if not resolved_mode else uses_model
        show_epochs = False
        self.model_title_label.setVisible(show_model)
        self.model_path.setVisible(show_model)
        self.epochs_title_label.setVisible(show_epochs)
        self.le_epochs.setVisible(show_epochs)

    def connect_internal_signals(self):
        self.rb_train_and_recognition.clicked.connect(
            lambda _: self.sample_type_changed.emit(WorkMode.train_and_recognition.value)
        )
        self.rb_recognition.clicked.connect(lambda _: self.sample_type_changed.emit(WorkMode.recognition_only.value))
        self.rb_further_train_model.clicked.connect(
            lambda _: self.sample_type_changed.emit(WorkMode.further_training.value)
        )
        self.rb_train_only.clicked.connect(lambda _: self.sample_type_changed.emit(WorkMode.train_only.value))

        self.lbl_source.clicked.connect(lambda: self.source_path_requested.emit())
        self.lbl_result.clicked.connect(lambda: self.result_path_requested.emit())

        self.label_path.clicked.connect(lambda: self.label_path_requested.emit())
        self.sample_path.clicked.connect(lambda: self.jpg_path_requested.emit())

        self.model_path.clicked.connect(lambda: self.model_path_requested.emit())
        if self._open_config_action is not None:
            self._open_config_action.triggered.connect(lambda _checked=False: self.open_config_requested.emit())
        if self._settings_sample_action is not None and self.settings_dock is not None:
            self._settings_sample_action.triggered.connect(lambda _checked=False: self._show_settings_page("base"))
        if self._settings_train_action is not None and self.settings_dock is not None:
            self._settings_train_action.triggered.connect(
                lambda _checked=False: self._show_settings_page("training")
            )
        if self._settings_pred_action is not None and self.settings_dock is not None:
            self._settings_pred_action.triggered.connect(
                lambda _checked=False: self._show_settings_page("recognition")
            )

        self.le_epochs.valueChanged.connect(lambda _: self.epochs_changed.emit())

        self.btn_start.clicked.connect(lambda: self.start_requested.emit())
        self.btn_stop.clicked.connect(lambda: self.stop_requested.emit())
        self.btn_queue_remove.clicked.connect(lambda: self.queue_remove_requested.emit())
        self.btn_queue_pause_toggle.clicked.connect(lambda: self.queue_pause_toggle_requested.emit())
        self.queue_list.customContextMenuRequested.connect(self._show_queue_context_menu)
        if hasattr(self, "batch_preview_action"):
            self.batch_preview_action.toggled.connect(self.batch_preview_visibility_changed.emit)
            self.batch_preview_action.toggled.connect(self.set_batch_preview_enabled)
        if hasattr(self, "release_memory_action"):
            self.release_memory_action.triggered.connect(self.release_memory_requested.emit)
        if hasattr(self, "open_tic_tac_toe_action"):
            self.open_tic_tac_toe_action.triggered.connect(self.open_tic_tac_toe_requested.emit)
        if self._open_validation_gradient_action is not None:
            self._open_validation_gradient_action.triggered.connect(self.open_validation_gradient_requested.emit)
        if self._developer_tools_action is not None:
            self._developer_tools_action.triggered.connect(self.developer_tools_requested.emit)
        if self._check_updates_action is not None:
            self._check_updates_action.triggered.connect(self.update_check_requested.emit)
        if hasattr(self, "ui_language_ru_action"):
            self.ui_language_ru_action.triggered.connect(lambda: self._handle_ui_language_action("ru"))
        if hasattr(self, "ui_language_en_action"):
            self.ui_language_en_action.triggered.connect(lambda: self._handle_ui_language_action("en"))
        if hasattr(self, "theme_dark_action"):
            self.theme_dark_action.triggered.connect(lambda: self._handle_theme_action("dark"))
        if hasattr(self, "theme_light_action"):
            self.theme_light_action.triggered.connect(lambda: self._handle_theme_action("light"))
        if self._ui_mode_simple_action is not None:
            self._ui_mode_simple_action.triggered.connect(lambda: self._handle_ui_mode_action("simple"))
        if self._ui_mode_advanced_action is not None:
            self._ui_mode_advanced_action.triggered.connect(lambda: self._handle_ui_mode_action("advanced"))
        self.btn_simple_conductors.clicked.connect(lambda: self._handle_simple_workflow_click("conductors"))
        self.btn_simple_contacts.clicked.connect(lambda: self._handle_simple_workflow_click("contacts"))
        self.btn_simple_memory.clicked.connect(lambda: self._handle_simple_workflow_click("memory"))
        self.open_tic_tac_toe_requested.connect(self._open_tic_tac_toe_dialog)

        self.log_message.connect(self._append_log)
        self.log_message_with_delete_last.connect(self.append_with_delete_previous)
        self.metrics_message.connect(self._append_metrics)
        self.enable_start.connect(self._set_start_enabled)
        self.show_info.connect(self._show_info_message)
        self.show_warning.connect(self._show_warning_message)
        self.toggle_start_stop.connect(self._switch_start_stop)

    def _show_queue_context_menu(self, position) -> None:
        item = self.queue_list.itemAt(position)
        if item is None:
            return
        row = self.queue_list.row(item)
        if row < 0:
            return

        self.queue_list.setCurrentRow(row)
        texts = self._main_texts()
        menu = QMenu(self.queue_list)
        remove_action = menu.addAction(str(texts.get("queue_remove", "Удалить из очереди")))
        properties_action = menu.addAction(str(texts.get("queue_properties", "Свойства")))
        selected_action = menu.exec(self.queue_list.viewport().mapToGlobal(position))

        if selected_action is remove_action:
            self.queue_context_remove_requested.emit(row)
        elif selected_action is properties_action:
            self.queue_properties_requested.emit(row)

    def _show_settings_page(self, page_key: str) -> None:
        self.show_settings_dock()
        if self.settings_dock is not None and hasattr(self.settings_dock, "show_settings_page"):
            self.settings_dock.show_settings_page(page_key)

    def _main_texts(self) -> dict[str, str]:
        texts = getattr(self, "_texts", None)
        return texts if isinstance(texts, dict) else get_ui_section("main_window")

    def _format_update_channel_label(self, channel: str) -> str:
        normalized_channel = str(channel or "").strip().lower()
        if not normalized_channel:
            normalized_channel = "stable"
        default_label = normalized_channel.replace("_", " ").title()
        return str(
            self._main_texts().get(
                f"update_channel_{normalized_channel}",
                default_label,
            )
        )

    def configure_update_channels(
        self,
        available_channels: list[str] | tuple[str, ...] | None,
        selected_channel: str | None,
    ) -> None:
        normalized_channels: list[str] = []
        for channel in available_channels or ("stable",):
            normalized = str(channel or "").strip().lower()
            if normalized and normalized not in normalized_channels:
                normalized_channels.append(normalized)
        if not normalized_channels:
            normalized_channels = ["stable"]

        resolved_selected = str(selected_channel or "").strip().lower()
        if resolved_selected not in normalized_channels:
            resolved_selected = normalized_channels[0]

        self._available_update_channels = normalized_channels
        self._selected_update_channel = resolved_selected
        self._update_channel_actions = {}

        if self._update_channel_menu is None:
            return

        self._update_channel_menu.clear()
        self._update_channel_action_group = QActionGroup(self)
        self._update_channel_action_group.setExclusive(True)

        for channel in normalized_channels:
            action = QAction(self._format_update_channel_label(channel), self)
            action.setCheckable(True)
            action.setData(channel)
            action.setChecked(channel == resolved_selected)
            action.triggered.connect(
                lambda checked=False, selected=channel: (
                    self.update_channel_selected.emit(selected) if checked else None
                )
            )
            self._update_channel_action_group.addAction(action)
            self._update_channel_menu.addAction(action)
            self._update_channel_actions[channel] = action

    def _show_info_message(self, text: str) -> None:
        if not _modal_dialogs_enabled():
            self.statusBar().showMessage(str(text))
            return
        QMessageBox.information(self, str(self._main_texts().get("info", "Информация")), text)

    def _show_warning_message(self, text: str) -> None:
        if not _modal_dialogs_enabled():
            self.statusBar().showMessage(str(text))
            return
        QMessageBox.warning(self, str(self._main_texts().get("warning", "Предупреждение")), text)

    def _format_validation_quality_text(self, iou: float, dice: float, f1: float) -> str:
        t = self._main_texts()
        template = str(
            t.get(
                "validation_quality_template",
                "IoU: {iou} | Dice: {dice} | F1: {f1}",
            )
        )
        return template.format(
            iou=f"{float(iou):.2%}",
            dice=f"{float(dice):.2%}",
            f1=f"{float(f1):.2%}",
        )

    def _format_performance_text(
        self,
        data_wait_ms: float,
        augmentation_ms: float,
        forward_ms: float,
        backward_ms: float,
        optimizer_ms: float,
        total_ms: float,
    ) -> str:
        t = self._main_texts()
        template = str(
            t.get(
                "performance_label_template",
                "Batch timing | data: {data_wait_ms:.1f} ms | aug: {augmentation_ms:.1f} ms | "
                "forward: {forward_ms:.1f} ms | "
                "backward: {backward_ms:.1f} ms | optimizer: {optimizer_ms:.1f} ms | total: {total_ms:.1f} ms",
            )
        )
        return template.format(
            data_wait_ms=float(data_wait_ms),
            augmentation_ms=float(augmentation_ms),
            forward_ms=float(forward_ms),
            backward_ms=float(backward_ms),
            optimizer_ms=float(optimizer_ms),
            total_ms=float(total_ms),
        )

    def _append_log(self, data):
        layout: QVBoxLayout = self.log_layout
        message_text = str(data)
        new_label = QLabel(message_text)
        layout.addWidget(new_label)
        self._last_status_message = message_text
        self.statusBar().showMessage(message_text)
        while layout.count() > MAX_LOG_MESSAGES:
            item = layout.takeAt(0)
            old_widget = item.widget() if item is not None else None
            if isinstance(old_widget, QWidget):
                old_widget.deleteLater()

        if isinstance(self.log_scroll, QScrollArea):
            vbar = self.log_scroll.verticalScrollBar()
            if vbar is not None:
                vbar.setValue(vbar.maximum())

    def append_with_delete_previous(self, data: str) -> None:
        layout: QVBoxLayout = self.log_layout
        count: int = layout.count()

        if count:
            item = layout.itemAt(count - 1)
            old_widget = item.widget() if item is not None else None
            if isinstance(old_widget, QWidget):
                old_widget.deleteLater()
                layout.removeWidget(old_widget)

        self._append_log(data)

    def _append_metrics(self, data):
        if not isinstance(data, dict):
            return

        metric_type = data.get("type")
        if metric_type == "train_epoch":
            self.metrics_panel.add_train_epoch_point(int(data.get("epoch", 0)), float(data.get("loss", 0.0)))
            return

        if metric_type == "val_epoch":
            self.metrics_panel.add_val_epoch_point(int(data.get("epoch", 0)), float(data.get("loss", 0.0)))
            iou = data.get("iou")
            dice = data.get("dice")
            f1 = data.get("f1")
            if iou is not None and dice is not None and f1 is not None:
                self.metrics_panel.add_val_quality_point(
                    int(data.get("epoch", 0)),
                    float(iou),
                    float(dice),
                )
                self._last_validation_metrics = (float(iou), float(dice), float(f1))
                self.validation_quality_label.setText(
                    self._format_validation_quality_text(float(iou), float(dice), float(f1))
                )
            return

        if metric_type == "train_batch":
            epoch = int(data.get("epoch", 0))
            batch_index = float(data.get("batch_index", 0.0))
            loss = float(data.get("loss", 0.0))
            epoch_points = self._batch_points_by_epoch.setdefault(epoch, [])
            epoch_points.append((batch_index, loss))
            self.metrics_panel.set_batch_points(epoch, self._sparsify_batch_points(epoch_points))
            return

        if metric_type == "train_epoch_progress":
            self._set_progress_bar(self.epoch_progress_bar, int(data.get("current", 0)), int(data.get("total", 0)))
            return

        if metric_type == "train_batch_progress":
            self._set_progress_bar(self.batch_progress_bar, int(data.get("current", 0)), int(data.get("total", 0)))
            return

        if metric_type == "recognition_progress":
            current = int(data.get("current", 0))
            total = int(data.get("total", 0))
            self._set_progress_bar(self.recognition_progress_bar, current, total)
            self._update_recognition_speed(current, total)
            return

        if metric_type == "train_batch_preview":
            image = data.get("image")
            label = data.get("label")
            outputs = data.get("outputs", data.get("output"))
            sample_name = str(data.get("sample_name", data.get("frame_name", ""))).strip()
            self._set_preview_mode("train")
            self._set_preview_image(self.preview_image_label, image)
            self._set_preview_image(self.preview_label_label, label)
            self._set_preview_image(self.preview_output_label, outputs)
            self._set_preview_frame_name(sample_name)
            return

        if metric_type == "recognition_preview":
            image = data.get("image")
            outputs = data.get("outputs", data.get("output", data.get("result")))
            sample_name = str(data.get("sample_name", data.get("frame_name", ""))).strip()
            self._set_preview_mode("recognition")
            self.preview_label_label.clear()
            self._set_preview_image(self.preview_image_label, image)
            self._set_preview_image(self.preview_output_label, outputs)
            self._set_preview_frame_name(sample_name)
            return

        if metric_type == "system_memory":
            ram_mb = data.get("ram_mb")
            vram_alloc_mb = data.get("vram_allocated_mb")
            vram_reserved_mb = data.get("vram_reserved_mb")
            self._ram_mb = float(ram_mb) if ram_mb is not None else None
            self._vram_alloc_mb = float(vram_alloc_mb) if vram_alloc_mb is not None else None
            self._vram_reserved_mb = float(vram_reserved_mb) if vram_reserved_mb is not None else None
            self._update_memory_runtime_label()
            return

        if metric_type in ("train_perf", "train_perf_epoch"):
            data_wait_ms = float(data.get("data_wait_ms", 0.0))
            augmentation_ms = float(data.get("augmentation_ms", 0.0))
            forward_ms = float(data.get("forward_ms", 0.0))
            backward_ms = float(data.get("backward_ms", 0.0))
            optimizer_ms = float(data.get("optimizer_ms", 0.0))
            total_ms = float(data.get("total_ms", 0.0))
            self._last_performance_metrics = {
                "data_wait_ms": data_wait_ms,
                "augmentation_ms": augmentation_ms,
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "optimizer_ms": optimizer_ms,
                "total_ms": total_ms,
            }
            self.performance_label.setText(
                self._format_performance_text(
                    data_wait_ms,
                    augmentation_ms,
                    forward_ms,
                    backward_ms,
                    optimizer_ms,
                    total_ms,
                )
            )
            if math.isfinite(total_ms) and total_ms > 0.0:
                self._train_speed_batches_per_sec = 1000.0 / total_ms
            else:
                self._train_speed_batches_per_sec = None
            self._update_memory_runtime_label()
            return

    def _update_memory_runtime_label(self) -> None:
        t = self._main_texts()
        no_runtime_data = (
            self._ram_mb is None
            and self._vram_alloc_mb is None
            and self._vram_reserved_mb is None
            and self._train_speed_batches_per_sec is None
        )
        if no_runtime_data:
            self.memory_usage_label.setText(str(t.get("memory_label_default", "Память: —")))
            return

        memory_unit = str(t.get("memory_unit", "МБ"))
        speed_unit = str(t.get("speed_unit", "batch/s"))
        ram_label = str(t.get("runtime_ram_label", "RAM"))
        vram_label = str(t.get("runtime_vram_label", "VRAM"))
        speed_label = str(t.get("runtime_speed_label", "Скорость"))

        ram_text = f"{ram_label}: {self._ram_mb:.0f} {memory_unit}" if self._ram_mb is not None else f"{ram_label}: —"
        if self._vram_alloc_mb is None:
            vram_text = f"{vram_label}: —"
        else:
            reserved_text = f"/{self._vram_reserved_mb:.0f}" if self._vram_reserved_mb is not None else ""
            vram_text = f"{vram_label}: {self._vram_alloc_mb:.0f}{reserved_text} {memory_unit}"
        speed_text = (
            f"{speed_label}: {self._train_speed_batches_per_sec:.2f} {speed_unit}"
            if self._train_speed_batches_per_sec is not None
            else f"{speed_label}: — {speed_unit}"
        )
        self.memory_usage_label.setText(f"{ram_text} | {vram_text} | {speed_text}")

    def _update_recognition_speed(self, current: int, total: int) -> None:
        if total <= 0:
            self._recognition_speed_images_per_sec = None
            self._recognition_started_at = None
            self._recognition_last_current = 0
            self._recognition_last_total = 0
            self._update_recognition_speed_label()
            return

        now = time.perf_counter()
        run_restarted = (
            self._recognition_started_at is None
            or total != self._recognition_last_total
            or current < self._recognition_last_current
            or current == 0
        )
        if run_restarted:
            self._recognition_started_at = now
            self._recognition_speed_images_per_sec = None
        elif self._recognition_started_at is not None and current > 0:
            elapsed_seconds = max(1e-6, now - self._recognition_started_at)
            self._recognition_speed_images_per_sec = current / elapsed_seconds

        self._recognition_last_current = current
        self._recognition_last_total = total
        self._update_recognition_speed_label()

    def _update_recognition_speed_label(self) -> None:
        t = self._main_texts()
        label_fallback = "Recognition speed" if self._ui_language == "en" else "Скорость распознавания"
        default_fallback = "Recognition speed: —" if self._ui_language == "en" else "Скорость распознавания: —"
        unit_fallback = "img/s" if self._ui_language == "en" else "изобр./с"
        default_text = str(t.get("recognition_speed_default", default_fallback))
        label = str(t.get("recognition_speed_label", label_fallback))
        unit = str(t.get("recognition_speed_unit", unit_fallback))
        if self._recognition_speed_images_per_sec is None:
            self.recognition_speed_label.setText(default_text)
            return
        self.recognition_speed_label.setText(
            f"{label}: {self._recognition_speed_images_per_sec:.2f} {unit}"
        )

    def _set_preview_frame_name(self, sample_name: str) -> None:
        self._current_preview_sample_name = str(sample_name).strip()
        if self._current_preview_sample_name:
            template = str(self._main_texts().get("preview_current_frame", "Frame: {name}"))
            self.preview_frame_name_label.setText(template.format(name=self._current_preview_sample_name))
            return
        self.preview_frame_name_label.setText(
            str(self._main_texts().get("preview_current_frame_default", "Frame: —"))
        )

    def _set_preview_mode(self, mode: str) -> None:
        resolved_mode = "recognition" if str(mode).strip().lower() == "recognition" else "train"
        self._current_preview_mode = resolved_mode
        preview_label_column = getattr(self, "preview_label_column_widget", None)
        if preview_label_column is not None:
            preview_label_column.setVisible(resolved_mode != "recognition")
        self.preview_image_title_label.setText(self._main_texts()["preview_image"])
        self.preview_label_title_label.setText(self._main_texts()["preview_label"])
        self.preview_output_title_label.setText(self._main_texts()["preview_output"])

    def _reset_runtime_metrics(self) -> None:
        self._ram_mb = None
        self._vram_alloc_mb = None
        self._vram_reserved_mb = None
        self._train_speed_batches_per_sec = None
        self._recognition_speed_images_per_sec = None
        self._recognition_started_at = None
        self._recognition_last_current = 0
        self._recognition_last_total = 0
        self._last_validation_metrics = None
        self._last_performance_metrics = None

    @staticmethod
    def _sparsify_batch_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(points) > 1000:
            return points[::2]
        return points

    @staticmethod
    def _set_progress_bar(progress_bar: QProgressBar, current: int, total: int):
        if total <= 0:
            progress_bar.setValue(0)
            progress_bar.setFormat("0%")
            return
        value = max(0, min(100, int((current / total) * 100)))
        progress_bar.setValue(value)
        progress_bar.setFormat(f"{value}% ({current}/{total})")

    @staticmethod
    def _set_preview_image(widget: QLabel, image_data):
        if not isinstance(image_data, np.ndarray) or image_data.size == 0:
            return
        arr = image_data
        if arr.ndim == 2:
            qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format.Format_Grayscale8).copy()
        elif arr.ndim == 3 and arr.shape[2] == 3:
            qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format.Format_RGB888).copy()
        else:
            return
        pix = QPixmap.fromImage(qimg).scaled(
            widget.width(),
            widget.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        widget.setText("")
        widget.setPixmap(pix)

    def _set_start_enabled(self, ok: bool):
        self.btn_start.setEnabled(True)
        self.btn_start.setToolTip('' if ok else 'Нажмите кнопку, чтобы увидеть причину невозможности запуска.')
        self.btn_start.setStyleSheet(
            """background-color: #4CAF50; color: white; padding: 8px 20px;
               border: none; border-radius: 5px;"""
            if ok
            else """background-color: #C62828; color: white; padding: 8px 20px;
               border: none; border-radius: 5px;"""
        )

    def _switch_start_stop(self, show_stop: bool):
        self.btn_start.setVisible(True)
        self.btn_stop.setVisible(show_stop)
        if show_stop:
            self._reset_runtime_metrics()
            self.metrics_panel.clear()
            self._batch_points_by_epoch.clear()
            self._set_progress_bar(self.epoch_progress_bar, 0, 0)
            self._set_progress_bar(self.batch_progress_bar, 0, 0)
            self._set_progress_bar(self.recognition_progress_bar, 0, 0)
            self._update_recognition_speed_label()
            self.preview_image_label.clear()
            self.preview_label_label.clear()
            self.preview_output_label.clear()
            self._set_preview_mode("train")
            self._set_preview_frame_name("")
            self._update_memory_runtime_label()
            self.validation_quality_label.setText(get_ui_section("main_window")["validation_quality_default"])
            self.performance_label.setText(get_ui_section("main_window")["performance_label_default"])

    def _open_tic_tac_toe_dialog(self):
        if self._tic_tac_toe_dialog is None:
            self._tic_tac_toe_dialog = TicTacToeDialog(self)
        self._tic_tac_toe_dialog.show()
        self._tic_tac_toe_dialog.raise_()
        self._tic_tac_toe_dialog.activateWindow()

    def get_selected_queue_row(self) -> int:
        return self.queue_list.currentRow()

    def set_task_queue_items(self, items: list[str], selected_row: int = -1):
        self.queue_list.clear()
        self.queue_list.addItems(items)
        if 0 <= selected_row < len(items):
            self.queue_list.setCurrentRow(selected_row)

    def set_stylesheet(self, style):
        self.setStyleSheet(style)

    def _theme_qss_path(self, theme: str) -> str:
        if theme == "light":
            return str(resolve_shared_style_path('style.qss'))
        return str(resolve_shared_style_path('dark_modern.qss'))

    def _sync_language_menu_checks(self) -> None:
        if hasattr(self, "ui_language_ru_action"):
            self.ui_language_ru_action.setChecked(self._ui_language == "ru")
        if hasattr(self, "ui_language_en_action"):
            self.ui_language_en_action.setChecked(self._ui_language == "en")

    def _sync_theme_menu_checks(self) -> None:
        if hasattr(self, "theme_dark_action"):
            self.theme_dark_action.setChecked(self._theme == "dark")
        if hasattr(self, "theme_light_action"):
            self.theme_light_action.setChecked(self._theme == "light")

    def _sync_ui_mode_menu_checks(self) -> None:
        if self._ui_mode_simple_action is not None:
            self._ui_mode_simple_action.setChecked(self._ui_mode == "simple")
        if self._ui_mode_advanced_action is not None:
            self._ui_mode_advanced_action.setChecked(self._ui_mode == "advanced")

    def _apply_theme(self, theme: str) -> None:
        self._theme = "light" if theme == "light" else "dark"
        qss_path = self._theme_qss_path(self._theme)
        self.set_stylesheet(load_qss_from_resource(qss_path))
        self._sync_theme_menu_checks()

    def apply_theme(self, theme: str) -> None:
        self._apply_theme(theme)

    def _handle_theme_action(self, theme: str) -> None:
        self._apply_theme(theme)
        self.theme_selected.emit(self._theme)

    def _handle_ui_mode_action(self, mode: str) -> None:
        self.apply_ui_mode(mode)
        self.ui_mode_selected.emit(self._ui_mode)

    def apply_ui_mode(self, mode: str) -> None:
        normalized_mode = "simple" if mode == "simple" else "advanced"
        self._ui_mode = normalized_mode
        is_simple = normalized_mode == "simple"
        self.simple_workflows_group.setVisible(is_simple)
        self.log_dock.setVisible(not is_simple)
        self.metrics_panel.setVisible(not is_simple)
        if self.settings_dock is not None:
            self.settings_dock.setVisible(not is_simple)
        if self._metrics_toggle_action is not None:
            self._metrics_toggle_action.setVisible(not is_simple)
        if self._log_toggle_action is not None:
            self._log_toggle_action.setVisible(not is_simple)
        if self._settings_toggle_action is not None:
            self._settings_toggle_action.setVisible(not is_simple)
        if self._settings_menu is not None:
            self._settings_menu.menuAction().setVisible(not is_simple)
        self.apply_work_mode(self._current_work_mode)
        self._sync_ui_mode_menu_checks()
        _save_persisted_ui_mode(self._ui_mode)

    def current_ui_mode(self) -> str:
        return self._ui_mode

    def _handle_simple_workflow_click(self, profile_key: str) -> None:
        self.set_simple_workflow_profile(profile_key)
        self.simple_workflow_requested.emit(profile_key)

    def set_simple_workflow_profile(self, profile_key: str | None) -> None:
        self._selected_simple_workflow = str(profile_key) if profile_key else None
        self._update_simple_workflow_label()

    def _simple_workflow_display_name(self, profile_key: str | None) -> str:
        mapping = {
            'conductors': self.btn_simple_conductors.text(),
            'contacts': self.btn_simple_contacts.text(),
            'memory': self.btn_simple_memory.text(),
        }
        return mapping.get(str(profile_key), self._main_texts().get('simple_workflow_none', 'Not selected'))

    def _update_simple_workflow_label(self) -> None:
        t = self._main_texts()
        template = str(t.get('simple_workflow_selected_template', 'Current profile: {profile}'))
        self.simple_workflow_label.setText(
            template.format(profile=self._simple_workflow_display_name(self._selected_simple_workflow))
        )

    def apply_ui_language(self, language: str) -> None:
        self._ui_language = "en" if language == "en" else "ru"
        t = get_ui_section("main_window")
        self._texts = t
        if self._sample_count_pending:
            self.set_samples_count_loading()
        else:
            self.set_samples_count(self._sample_count_value)
        self.work_mode_group.setTitle(t["mode"])
        self.rb_train_and_recognition.setText(t["mode_train_and_rec"])
        self.rb_further_train_model.setText(t["mode_ft_and_rec"])
        self.rb_recognition.setText(t["mode_rec"])
        self.rb_train_only.setText(t["mode_train"])
        self.source_title_label.setText(t["source"])
        self.result_title_label.setText(t["result"])
        self.sample_path_group.setTitle(t["sample"])
        self.sample_path.setToolTip(t["sample_tip"])
        self.label_path.setToolTip(t["label_tip"])
        if self.sample_src_title_label is not None:
            self.sample_src_title_label.setText(t["sample_src"])
        if self.label_path_title_label is not None:
            self.label_path_title_label.setText(t["labels"])
        self.model_title_label.setText(t["model"])
        self.epochs_title_label.setText(t["epochs"])
        self.btn_start.setText(t["start"])
        self.btn_stop.setText(t["stop"])
        self.queue_group.setTitle(t["queue"])
        self.btn_queue_remove.setText(t["queue_remove"])
        self.btn_queue_pause_toggle.setText(t["queue_pause"])
        self.simple_workflows_group.setTitle(t.get("simple_workflows_group", "Simple workflows"))
        self.btn_simple_conductors.setText(t.get("simple_workflow_conductors", "Conductor recognition"))
        self.btn_simple_contacts.setText(t.get("simple_workflow_contacts", "Contact recognition"))
        self.btn_simple_memory.setText(t.get("simple_workflow_memory", "Memory recognition"))
        self._update_simple_workflow_label()
        self.progress_group.setTitle(t["progress_group"])
        if self.progress_epochs_title_label is not None:
            self.progress_epochs_title_label.setText(t["progress_epochs"])
        if self.progress_batches_title_label is not None:
            self.progress_batches_title_label.setText(t["progress_batches"])
        if self.progress_recognition_title_label is not None:
            self.progress_recognition_title_label.setText(t["progress_recognition"])
        self._update_recognition_speed_label()
        self.preview_group.setTitle(t["preview_group"])
        self._set_preview_frame_name(self._current_preview_sample_name)
        self._set_preview_mode(self._current_preview_mode)
        self.statusBar().showMessage(self._last_status_message)
        if self._settings_menu is not None:
            self._settings_menu.setTitle(t["menu_settings"])
        if self._file_menu is not None:
            self._file_menu.setTitle(t.get("menu_file", "Файл"))
        if self._view_menu is not None:
            self._view_menu.setTitle(t["menu_view"])
        if self._info_menu is not None:
            self._info_menu.setTitle(t["menu_help"])
        if self._plugins_menu is not None:
            self._plugins_menu.setTitle(t.get("menu_plugins", "Plugins"))
        if self._tools_menu is not None:
            self._tools_menu.setTitle(t.get("menu_tools", "Инструменты"))
        if self._update_channel_menu is not None:
            self._update_channel_menu.setTitle(t.get("menu_update_channel", "Update channel"))
        if self._open_config_action is not None:
            self._open_config_action.setText(t.get("menu_open_config", "Открыть"))
        if self._language_menu is not None:
            self._language_menu.setTitle(t.get("menu_language", "Язык"))
        if self._theme_menu is not None:
            self._theme_menu.setTitle(t.get("menu_theme", "Тема"))
        if self._ui_mode_menu is not None:
            self._ui_mode_menu.setTitle(t.get("menu_ui_mode", "Interface mode"))
        if self._settings_sample_action is not None:
            self._settings_sample_action.setText(t.get("menu_sample", "Выборка"))
        if self._settings_train_action is not None:
            self._settings_train_action.setText(t.get("menu_train", "Обучение"))
        if self._settings_pred_action is not None:
            self._settings_pred_action.setText(t.get("menu_pred", "Распознавание"))
        if self._metrics_toggle_action is not None:
            self._metrics_toggle_action.setText(t.get("menu_metrics", "Панель графиков"))
        if self._log_toggle_action is not None:
            self._log_toggle_action.setText(t.get("menu_log_panel", "Панель лога"))
        if self._settings_toggle_action is not None:
            self._settings_toggle_action.setText(t.get("menu_settings_panel", "Панель настроек"))
        if hasattr(self, "batch_preview_action"):
            self.batch_preview_action.setText(t.get("menu_batch_preview", "Превью пакета обучения"))
        if hasattr(self, "release_memory_action"):
            self.release_memory_action.setText(t.get("menu_release_memory", "Освободить память GPU"))
        if hasattr(self, "open_tic_tac_toe_action"):
            self.open_tic_tac_toe_action.setText(
                t.get("menu_open_tic_tac_toe", "Крестики-нолики (нейросеть)")
            )
        if self._open_validation_gradient_action is not None:
            self._open_validation_gradient_action.setText(
                t.get("menu_open_validation_gradient", "Open Validation gradient")
            )
        if self._developer_tools_action is not None:
            self._developer_tools_action.setText(t.get("menu_developer", "Разработчик"))
        if self._help_action is not None:
            self._help_action.setText(t.get("menu_open_help", "Открыть справку"))
        if self._changelog_action is not None:
            self._changelog_action.setText(t.get("menu_open_changelog", "Список изменений"))
        if self._check_updates_action is not None:
            self._check_updates_action.setText(t.get("menu_check_updates", "Проверить обновления"))
        for channel, action in self._update_channel_actions.items():
            action.setText(self._format_update_channel_label(channel))
        if hasattr(self, "ui_language_ru_action"):
            self.ui_language_ru_action.setText(t.get("lang_ru", "Русский"))
        if hasattr(self, "ui_language_en_action"):
            self.ui_language_en_action.setText(t.get("lang_en", "English"))
        if hasattr(self, "theme_dark_action"):
            self.theme_dark_action.setText(t.get("theme_dark", "Темная"))
        if hasattr(self, "theme_light_action"):
            self.theme_light_action.setText(t.get("theme_light", "Светлая"))
        if self._ui_mode_simple_action is not None:
            self._ui_mode_simple_action.setText(t.get("menu_ui_mode_simple", "Simple"))
        if self._ui_mode_advanced_action is not None:
            self._ui_mode_advanced_action.setText(t.get("menu_ui_mode_advanced", "Advanced"))
        self.log_dock.setWindowTitle(t.get("log_dock_title", "Лог"))
        if self.settings_dock is not None:
            self.settings_dock.setWindowTitle(t.get("settings_dock_title", "Настройки"))
        self._update_memory_runtime_label()
        if self._last_validation_metrics is not None:
            iou, dice, f1 = self._last_validation_metrics
            self.validation_quality_label.setText(self._format_validation_quality_text(iou, dice, f1))
        else:
            self.validation_quality_label.setText(t["validation_quality_default"])
        if self._last_performance_metrics is not None:
            self.performance_label.setText(
                self._format_performance_text(**self._last_performance_metrics)
            )
        else:
            self.performance_label.setText(t["performance_label_default"])
        self._sync_language_menu_checks()

    def _handle_ui_language_action(self, language: str) -> None:
        if self.settings_dock is not None and hasattr(self.settings_dock, "set_ui_language"):
            self.settings_dock.set_ui_language(language)
        else:
            self.apply_ui_language(language)
        self.ui_language_selected.emit(language)

    def set_source_path(self, path: str):
        self.lbl_source.setText(path)

    def set_result_path(self, path: str):
        self.lbl_result.setText(path)

    def set_samples_count(self, total_samples: int) -> None:
        try:
            self._sample_count_value = int(total_samples)
        except (TypeError, ValueError):
            self._sample_count_value = 0
        self._sample_count_pending = False
        template = str(
            self._texts.get("samples_count_template", self._texts.get("samples_count", "Кадров в выборке: {count}"))
        )
        if "{count}" in template:
            text = template.format(count=self._sample_count_value)
        else:
            text = template
        self.sample_count_top_label.setText(text)

    def set_samples_count_loading(self) -> None:
        self._sample_count_pending = True
        self.sample_count_top_label.setText(str(self._texts.get("samples_count_loading", "Идет расчет...")))

    def set_label_path(self, path: str):
        self.label_path.setText(path)

    def set_jpg_path(self, path: str):
        self.sample_path.setText(path)

    def restore_from_dataclass(self, state: MainWindowState):
        self.set_source_path(state.source_folder)
        self.set_result_path(state.result_folder)
        self.label_path.setText(state.label_folder)
        self.sample_path.setText(state.sample_folder)
        self.model_path.setText(state.model_path)
        self.le_epochs.setValue(state.epochs)

    def closeEvent(self, event):
        if self._close_allowed:
            event.accept()
            return
        self.request_close.emit()
        event.ignore()

    def allow_close(self):
        self._close_allowed = True
        self.close()

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = MainView(QWidget())
    window.show()
    sys.exit(app.exec())



