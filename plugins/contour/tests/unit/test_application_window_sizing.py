from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDockWidget, QTabBar

from contour.application.model import ContourApplicationModel
from contour.application.view import ContourMainView, _bounded_initial_window_size


def _app() -> QApplication:
    instance = QApplication.instance()
    return instance if instance is not None else QApplication([])


def test_model_initial_size_matches_compact_window_minimum() -> None:
    assert ContourApplicationModel(width=1, height=1).initial_size == (640, 420)


def test_initial_window_size_is_bounded_by_available_screen() -> None:
    size = _bounded_initial_window_size(
        width=1680,
        height=980,
        available_width=1024,
        available_height=768,
    )

    assert size.width() == 992
    assert size.height() == 736


def test_initial_window_size_keeps_preferred_minimum_when_space_allows() -> None:
    size = _bounded_initial_window_size(
        width=320,
        height=240,
        available_width=1920,
        available_height=1080,
    )

    assert size.width() == 640
    assert size.height() == 420


def test_initial_window_size_can_fit_very_small_screens() -> None:
    size = _bounded_initial_window_size(
        width=1680,
        height=980,
        available_width=500,
        available_height=360,
    )

    assert size.width() == 468
    assert size.height() == 328


def test_main_view_exposes_file_menu_new_project_action() -> None:
    _app()
    view = ContourMainView()
    try:
        file_menu = view.menuBar().actions()[0].menu()

        assert file_menu is not None
        assert file_menu.title() in {"Файл", "File"}
        assert any(action.text() in {"Новый проект", "New project"} for action in file_menu.actions())
    finally:
        view.close()
        view.deleteLater()


def test_main_view_docks_thumbnail_matrix_and_keeps_file_menu_first() -> None:
    _app()
    view = ContourMainView()
    try:
        assert view.menuBar().actions()[0].menu() is not None
        files_dock = view.findChild(QDockWidget, "filesDock")
        recognition_dock = view.findChild(QDockWidget, "recognitionDock")
        dock = view.findChild(QDockWidget, "thumbnailMatrixDock")

        assert files_dock is not None
        assert files_dock.widget() is view.widget.files_tab
        assert recognition_dock is not None
        assert recognition_dock.widget() is view.widget.extraction_tab
        assert dock is not None
        assert dock.widget() is view.widget.thumbnail_matrix_panel
        assert view.widget.thumbnail_grid_scroll_area.widget() is view.widget.thumbnail_grid
        assert not view.widget.thumbnail_grid_label.isVisible()
    finally:
        view.close()
        view.deleteLater()


def test_main_view_restores_floating_thumbnail_matrix_from_view_toggle() -> None:
    app = _app()
    view = ContourMainView()
    try:
        view.show()
        app.processEvents()
        dock = view.findChild(QDockWidget, "thumbnailMatrixDock")
        assert dock is not None

        dock.setFloating(True)
        dock.hide()
        app.processEvents()
        assert dock.isFloating()
        assert not dock.isVisible()

        view._thumbnail_matrix_toggle_action.trigger()
        app.processEvents()

        assert dock.isVisible()
        assert not dock.isFloating()
        assert view.dockWidgetArea(dock) == Qt.DockWidgetArea.RightDockWidgetArea
    finally:
        view.close()
        view.deleteLater()


def test_main_view_default_left_dock_tab_order_is_paths_pipeline_recognition_display() -> None:
    _app()
    view = ContourMainView()
    try:
        view.set_ui_language("en")
        expected = ["Paths", "Pipeline", "Recognition", "Display"]
        matching_bars = [
            [bar.tabText(index) for index in range(bar.count())]
            for bar in view.findChildren(QTabBar)
            if all(label in [bar.tabText(index) for index in range(bar.count())] for label in expected)
        ]

        assert expected in matching_bars
    finally:
        view.close()
        view.deleteLater()


def test_main_view_exposes_dock_theme_and_language_actions_in_view_menu() -> None:
    _app()
    view = ContourMainView()
    try:
        view_menu = view.menuBar().actions()[1].menu()
        assert view_menu is not None

        action_texts = {action.text() for action in view_menu.actions() if action.text()}
        assert {"Файлы", "Распознавание", "Матрица кадров"} <= action_texts
        assert view._theme_menu.menuAction() in view_menu.actions()
        assert view._language_menu.menuAction() in view_menu.actions()
        assert {action.data() for action in view._theme_menu.actions()} == {"dark", "light"}
        assert {action.data() for action in view._language_menu.actions()} == {"ru", "en"}

        view.set_ui_language("en")
        assert view._language_en_action.isChecked()
        assert view._view_menu.title() == "View"
        assert view._theme_menu.title() == "Theme"
        assert view._language_menu.title() == "Language"
    finally:
        view.close()
        view.deleteLater()


def test_main_view_persists_language_and_theme_menu_changes() -> None:
    _app()

    class _AppearanceStore:
        def __init__(self) -> None:
            self.language = ""
            self.theme = ""

        def save_language(self, language: str | None) -> None:
            self.language = str(language or "")

        def save_theme(self, theme: str | None) -> None:
            self.theme = str(theme or "")

    store = _AppearanceStore()
    view = ContourMainView(appearance_settings_store=store)
    try:
        view.set_ui_language("en")
        view.apply_theme("light")

        assert store.language == "en"
        assert store.theme == "light"
        assert view._language_en_action.isChecked()
        assert view._theme_light_action.isChecked()
    finally:
        view.close()
        view.deleteLater()


def test_main_view_exposes_update_action_in_help_menu() -> None:
    _app()
    view = ContourMainView()
    try:
        found_menu = False
        found_action = False

        def walk(menu) -> None:
            nonlocal found_menu, found_action
            for action in menu.actions():
                submenu = action.menu()
                if action.objectName() == "contourCheckUpdatesAction":
                    found_action = True
                if submenu is not None:
                    if submenu.objectName() == "contourUpdateMenu":
                        found_menu = True
                    walk(submenu)

        for action in view.menuBar().actions():
            menu = action.menu()
            if menu is not None:
                walk(menu)

        assert found_menu
        assert found_action
    finally:
        view.close()
        view.deleteLater()
