from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kraken_core.theme import apply_app_theme, normalize_theme
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction, QActionGroup, QCloseEvent, QIcon
from PyQt6.QtWidgets import QDockWidget, QMainWindow, QMenu, QMenuBar, QStatusBar, QWidget

from ..__version__ import __version__
from ..infrastructure import WidgetAppearanceSettingsStore
from ..updater import create_contour_update_controller
from ..widget import PolygonExtractionWidget
from .styles import resolve_style_path

WINDOW_SCREEN_MARGIN_PX = 32
MIN_INITIAL_WINDOW_WIDTH = 640
MIN_INITIAL_WINDOW_HEIGHT = 420
CONTOUR_ENABLE_WORK_SIMULATION = True


def _required_qt_object[_QtObjectT](value: _QtObjectT | None, description: str) -> _QtObjectT:
    if value is None:
        raise RuntimeError(f"Qt did not create {description}")
    return value


def _try_apply_app_icon(window: QMainWindow) -> None:
    icon_path = resolve_style_path("icons", "icon.png")
    if icon_path.exists():
        window.setWindowIcon(QIcon(str(icon_path)))


def _bounded_initial_window_size(
    width: int,
    height: int,
    available_width: int,
    available_height: int,
) -> QSize:
    max_width = max(320, int(available_width) - WINDOW_SCREEN_MARGIN_PX)
    max_height = max(240, int(available_height) - WINDOW_SCREEN_MARGIN_PX)

    def _clamp(requested: int, preferred_minimum: int, maximum: int) -> int:
        requested = max(1, int(requested))
        if maximum < preferred_minimum:
            return maximum
        return min(max(requested, preferred_minimum), maximum)

    return QSize(
        _clamp(width, MIN_INITIAL_WINDOW_WIDTH, max_width),
        _clamp(height, MIN_INITIAL_WINDOW_HEIGHT, max_height),
    )


def _main_menu_bar(window: QMainWindow) -> QMenuBar:
    return _required_qt_object(window.menuBar(), "main menu bar")


def _status_bar(window: QMainWindow) -> QStatusBar:
    return _required_qt_object(window.statusBar(), "status bar")


def _add_menu(parent: QMenuBar | QMenu, title: str) -> QMenu:
    return _required_qt_object(parent.addMenu(title), "menu")


def _add_action(menu: QMenu, text: str) -> QAction:
    return _required_qt_object(menu.addAction(text), "menu action")


def _dock_toggle_action(dock: QDockWidget) -> QAction:
    return _required_qt_object(dock.toggleViewAction(), "dock toggle action")


class ContourMainView(QMainWindow):
    def __init__(self, appearance_settings_store: WidgetAppearanceSettingsStore | None = None) -> None:
        super().__init__()
        self._presenter: Any | None = None
        self._theme = "dark"
        self._appearance_settings_store = appearance_settings_store
        self._update_controller = None
        self._update_menu_action = None
        self._widget = PolygonExtractionWidget(self)
        self.setCentralWidget(self._widget)
        menu_bar = _main_menu_bar(self)
        self._file_menu = _add_menu(menu_bar, "")
        self._new_project_action = _add_action(self._file_menu, "")
        self._new_project_action.triggered.connect(lambda _checked=False: self._widget.reset_project())
        self._refresh_file_menu()
        self._view_menu = _add_menu(menu_bar, "")
        self._files_dock = self._create_panel_dock("filesDock", self._widget._take_files_panel())
        self._paths_dock = self._create_panel_dock("pathsDock", self._widget._take_paths_panel())
        self._pipeline_dock = self._create_panel_dock("pipelineDock", self._widget._take_pipeline_panel())
        self._display_dock = self._create_panel_dock("displayDock", self._widget._take_display_panel())
        self._recognition_dock = self._create_panel_dock("recognitionDock", self._widget._take_recognition_panel())
        self._run_dock = self._create_panel_dock("runDock", self._widget._take_run_panel())
        self._thumbnail_matrix_dock = self._create_panel_dock(
            "thumbnailMatrixDock",
            self._widget._take_thumbnail_matrix_panel(),
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._files_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._thumbnail_matrix_dock)
        self.splitDockWidget(self._files_dock, self._thumbnail_matrix_dock, Qt.Orientation.Vertical)
        self._files_dock.raise_()
        for dock in (
            self._paths_dock,
            self._pipeline_dock,
            self._recognition_dock,
            self._display_dock,
        ):
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.tabifyDockWidget(self._paths_dock, self._pipeline_dock)
        self.tabifyDockWidget(self._paths_dock, self._recognition_dock)
        self.tabifyDockWidget(self._paths_dock, self._display_dock)
        self._paths_dock.raise_()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._run_dock)
        self._run_dock.hide()
        self._files_toggle_action = _dock_toggle_action(self._files_dock)
        self._paths_toggle_action = _dock_toggle_action(self._paths_dock)
        self._pipeline_toggle_action = _dock_toggle_action(self._pipeline_dock)
        self._display_toggle_action = _dock_toggle_action(self._display_dock)
        self._recognition_toggle_action = _dock_toggle_action(self._recognition_dock)
        self._run_toggle_action = _dock_toggle_action(self._run_dock)
        self._thumbnail_matrix_toggle_action = _dock_toggle_action(self._thumbnail_matrix_dock)
        self._thumbnail_matrix_toggle_action.triggered.connect(self._on_thumbnail_matrix_toggle_triggered)
        if hasattr(self._widget, "show_frame_matrix_checkbox"):
            self._widget.show_frame_matrix_checkbox.stateChanged.connect(self._sync_thumbnail_matrix_dock_visibility)
        self._paths_dock.setWindowTitle("Paths")
        self._paths_toggle_action.setText("Paths")
        self._pipeline_dock.setWindowTitle("Pipeline")
        self._pipeline_toggle_action.setText("Pipeline")
        self._display_dock.setWindowTitle("Display")
        self._display_toggle_action.setText("Display")
        self._run_dock.setWindowTitle("Run")
        self._run_toggle_action.setText("Run")
        self._view_menu.addAction(self._files_toggle_action)
        self._view_menu.addAction(self._paths_toggle_action)
        self._view_menu.addAction(self._pipeline_toggle_action)
        self._view_menu.addAction(self._display_toggle_action)
        self._view_menu.addAction(self._recognition_toggle_action)
        self._view_menu.addAction(self._run_toggle_action)
        self._view_menu.addAction(self._thumbnail_matrix_toggle_action)
        self._view_menu.addSeparator()
        self._sync_thumbnail_matrix_dock_visibility()
        self._theme_menu = self._build_theme_menu()
        self._view_menu.addMenu(self._theme_menu)
        self._language_menu = self._build_language_menu()
        self._view_menu.addMenu(self._language_menu)
        self._tools_menu = _add_menu(menu_bar, "")
        self._work_simulation_action = _add_action(self._tools_menu, "")
        self._work_simulation_action.setVisible(CONTOUR_ENABLE_WORK_SIMULATION)
        self._work_simulation_action.triggered.connect(lambda _checked=False: self._widget._toggle_work_simulation())
        self._widget.workSimulationActiveChanged.connect(self._refresh_work_simulation_action)
        self._help_menu = _add_menu(menu_bar, self._widget.help_menu_title())
        self._widget.attach_help_menu(self._help_menu)
        self._update_controller = create_contour_update_controller(self)
        self._attach_update_menu_action()
        self._refresh_view_and_tools_menus()
        _try_apply_app_icon(self)

    @property
    def widget(self) -> PolygonExtractionWidget:
        return self._widget

    def _create_panel_dock(self, object_name: str, panel: QWidget) -> QDockWidget:
        dock = QDockWidget("", self)
        dock.setObjectName(object_name)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock.setWidget(panel)
        return dock

    def _on_thumbnail_matrix_toggle_triggered(self, checked: bool) -> None:
        if not self._widget._frame_matrix_enabled():
            self._thumbnail_matrix_dock.hide()
            self._thumbnail_matrix_toggle_action.setChecked(False)
            return
        if checked:
            self._restore_thumbnail_matrix_dock()

    def _restore_thumbnail_matrix_dock(self) -> None:
        if not self._widget._frame_matrix_enabled():
            return
        dock = self._thumbnail_matrix_dock
        if self.dockWidgetArea(dock) == Qt.DockWidgetArea.NoDockWidgetArea:
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        if dock.isFloating():
            dock.setFloating(False)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.splitDockWidget(self._files_dock, dock, Qt.Orientation.Vertical)
        dock.show()
        dock.raise_()

    def _sync_thumbnail_matrix_dock_visibility(self, *_args) -> None:
        if self._widget._frame_matrix_enabled():
            self._restore_thumbnail_matrix_dock()
            self._thumbnail_matrix_toggle_action.setEnabled(True)
            self._thumbnail_matrix_toggle_action.setChecked(True)
        else:
            self._thumbnail_matrix_dock.hide()
            self._thumbnail_matrix_toggle_action.setChecked(False)
            self._thumbnail_matrix_toggle_action.setEnabled(False)

    def _build_theme_menu(self) -> QMenu:
        menu = QMenu("", self)
        self._theme_action_group = QActionGroup(self)
        self._theme_action_group.setExclusive(True)
        self._theme_dark_action = QAction("", self)
        self._theme_dark_action.setCheckable(True)
        self._theme_dark_action.setData("dark")
        self._theme_light_action = QAction("", self)
        self._theme_light_action.setCheckable(True)
        self._theme_light_action.setData("light")
        for action in (self._theme_dark_action, self._theme_light_action):
            self._theme_action_group.addAction(action)
            menu.addAction(action)
        self._theme_dark_action.triggered.connect(lambda _checked=False: self.apply_theme("dark"))
        self._theme_light_action.triggered.connect(lambda _checked=False: self.apply_theme("light"))
        self._sync_theme_actions()
        return menu

    def _build_language_menu(self) -> QMenu:
        menu = QMenu("", self)
        self._language_action_group = QActionGroup(self)
        self._language_action_group.setExclusive(True)
        self._language_ru_action = QAction("", self)
        self._language_ru_action.setCheckable(True)
        self._language_ru_action.setData("ru")
        self._language_en_action = QAction("", self)
        self._language_en_action.setCheckable(True)
        self._language_en_action.setData("en")
        for action in (self._language_ru_action, self._language_en_action):
            self._language_action_group.addAction(action)
            menu.addAction(action)
        self._language_ru_action.triggered.connect(lambda _checked=False: self.set_ui_language("ru"))
        self._language_en_action.triggered.connect(lambda _checked=False: self.set_ui_language("en"))
        self._sync_language_actions()
        return menu

    def apply_theme(self, theme: str) -> None:
        self._theme = apply_app_theme(theme)
        self._widget._ui_theme = self._theme
        if hasattr(self._widget, "_refresh_image_list_item_states"):
            self._widget._refresh_image_list_item_states()
        if self._appearance_settings_store is not None:
            self._appearance_settings_store.save_theme(self._theme)
        self._sync_theme_actions()

    def _sync_theme_actions(self) -> None:
        theme = normalize_theme(self._theme)
        if hasattr(self, "_theme_dark_action"):
            self._theme_dark_action.setChecked(theme == "dark")
        if hasattr(self, "_theme_light_action"):
            self._theme_light_action.setChecked(theme == "light")

    def _sync_language_actions(self) -> None:
        language = getattr(self._widget, "_ui_language", "ru")
        if hasattr(self, "_language_ru_action"):
            self._language_ru_action.setChecked(language == "ru")
        if hasattr(self, "_language_en_action"):
            self._language_en_action.setChecked(language == "en")

    def set_presenter(self, presenter: Any) -> None:
        self._presenter = presenter

    def set_window_title(self, title: str) -> None:
        versioned_title = f"{title} {__version__}" if __version__ not in title else title
        self.setWindowTitle(versioned_title)

    def resize_window(self, width: int, height: int) -> None:
        screen = self.screen()
        if screen is None:
            self.resize(width, height)
            return
        available = screen.availableGeometry()
        self.resize(
            _bounded_initial_window_size(
                width,
                height,
                available.width(),
                available.height(),
            )
        )

    def set_ui_language(self, language: str) -> None:
        self._widget.set_ui_language(language)
        if self._appearance_settings_store is not None:
            self._appearance_settings_store.save_language(getattr(self._widget, "_ui_language", language))
        self._attach_update_menu_action()
        self._refresh_file_menu()
        self._refresh_view_and_tools_menus()
        self._help_menu.setTitle(self._widget.help_menu_title())
        self._sync_language_actions()

    def _attach_update_menu_action(self) -> None:
        if self._update_controller is None:
            return
        language = getattr(self._widget, "_ui_language", "ru")
        is_ru = language == "ru"
        self._update_menu_action = self._update_controller.add_menu_action(
            self._help_menu,
            "Проверить обновления" if is_ru else "Check for updates",
            submenu_title="Обновление" if is_ru else "Update",
            submenu_object_name="contourUpdateMenu",
            action_object_name="contourCheckUpdatesAction",
        )

    def _refresh_file_menu(self) -> None:
        language = getattr(self._widget, "_ui_language", "ru")
        self._file_menu.setTitle("Файл" if language == "ru" else "File")
        self._new_project_action.setText("Новый проект" if language == "ru" else "New project")

    def _refresh_view_and_tools_menus(self) -> None:
        language = getattr(self._widget, "_ui_language", "ru")
        self._view_menu.setTitle("Вид" if language == "ru" else "View")
        self._files_dock.setWindowTitle("Файлы" if language == "ru" else "Files")
        self._files_toggle_action.setText("Файлы" if language == "ru" else "Files")
        self._paths_dock.setWindowTitle("Пути" if language == "ru" else "Paths")
        self._paths_toggle_action.setText("Пути" if language == "ru" else "Paths")
        self._pipeline_dock.setWindowTitle("Pipeline")
        self._pipeline_toggle_action.setText("Pipeline")
        self._display_dock.setWindowTitle("Отображение" if language == "ru" else "Display")
        self._display_toggle_action.setText("Отображение" if language == "ru" else "Display")
        self._recognition_dock.setWindowTitle("Распознавание" if language == "ru" else "Recognition")
        self._recognition_toggle_action.setText("Распознавание" if language == "ru" else "Recognition")
        self._run_dock.setWindowTitle("Обработка" if language == "ru" else "Run")
        self._run_toggle_action.setText("Обработка" if language == "ru" else "Run")
        self._thumbnail_matrix_dock.setWindowTitle("Матрица кадров" if language == "ru" else "Frame matrix")
        self._thumbnail_matrix_toggle_action.setText("Матрица кадров" if language == "ru" else "Frame matrix")
        self._theme_menu.setTitle("Тема" if language == "ru" else "Theme")
        self._theme_dark_action.setText("Темная" if language == "ru" else "Dark")
        self._theme_light_action.setText("Светлая" if language == "ru" else "Light")
        self._language_menu.setTitle("Язык" if language == "ru" else "Language")
        self._language_ru_action.setText("Русский" if language == "ru" else "Russian")
        self._language_en_action.setText("Английский" if language == "ru" else "English")
        self._tools_menu.setTitle("Инструменты" if language == "ru" else "Tools")
        self._refresh_work_simulation_action()

    def _refresh_work_simulation_action(self) -> None:
        language = getattr(self._widget, "_ui_language", "ru")
        running = bool(getattr(self._widget, "_work_simulation_running", False))
        if running:
            self._work_simulation_action.setText("Прекратить симуляцию" if language == "ru" else "Stop simulation")
        else:
            self._work_simulation_action.setText("Симуляция работы" if language == "ru" else "Work simulation")

    def set_input_directory(self, path: str) -> None:
        self._widget.set_input_directory(path)

    def set_output_directory(self, path: str) -> None:
        self._widget.set_output_directory(path)

    def set_cif_directory(self, path: str) -> None:
        self._widget.set_cif_directory(path)

    def set_pipeline(self, payload: dict) -> None:
        self._widget.set_pipeline(payload)

    def load_images(self, paths: list[str]) -> None:
        self._widget.load_images(paths)

    def show_status_message(self, message: str, timeout_ms: int = 0) -> None:
        _status_bar(self).showMessage(message, timeout_ms)

    def bind_log_message(self, handler: Callable[[str], None]) -> None:
        self._widget.logMessage.connect(handler)

    def bind_image_processed(self, handler: Callable[[str, list], None]) -> None:
        self._widget.imageProcessed.connect(handler)

    def closeEvent(self, event: QCloseEvent) -> None:
        if hasattr(self._widget, "confirm_ok_to_leave_current_vectors") and not self._widget.confirm_ok_to_leave_current_vectors():
            event.ignore()
            return
        self._widget._persist_session_state()
        super().closeEvent(event)


class ContourStandaloneWindow(ContourMainView):
    """Backward-compatible alias for the standalone main window."""
