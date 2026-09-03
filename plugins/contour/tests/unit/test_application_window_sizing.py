from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDockWidget, QGroupBox, QTabBar, QToolBar

from contour.application.model import ContourApplicationModel
from contour.application.view import (
    DEFAULT_PANEL_DOCK_MIN_WIDTH,
    DEFAULT_RIGHT_DOCK_MIN_WIDTH,
    ContourMainView,
    _bounded_initial_window_size,
)
from contour.domain import PolygonData


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


def test_main_view_dock_contents_do_not_force_excessive_minimum_height() -> None:
    app = _app()
    view = ContourMainView()
    try:
        view.show()
        app.processEvents()

        assert view.minimumSizeHint().height() <= 560
        assert view._pipeline_dock.widget() is view.widget.pipeline_panel_scroll
        assert view._display_dock.widget() is view.widget.display_panel_scroll
        assert view.findChild(QDockWidget, "pathsDock") is None

        view.resize(800, 560)
        app.processEvents()
        assert view.height() <= 560
    finally:
        view.close()
        view.deleteLater()


def test_persisted_session_restore_runs_only_after_window_is_shown() -> None:
    app = _app()
    view = ContourMainView()
    restored_while_visible: list[bool] = []
    view.widget._restore_persisted_session_selection = lambda: restored_while_visible.append(view.isVisible())
    try:
        view.defer_persisted_session_selection_restore()
        app.processEvents()
        assert restored_while_visible == []

        view.show()
        assert restored_while_visible == []
        for _attempt in range(3):
            app.processEvents()
            if restored_while_visible:
                break

        assert restored_while_visible == [True]
    finally:
        view.close()
        view.deleteLater()


def test_main_view_status_bar_shows_selected_object_count() -> None:
    app = _app()
    view = ContourMainView()
    try:
        view.widget.set_ui_language("ru")
        contact = PolygonData(
            id=1,
            points=[(10, 10), (20, 10), (20, 20), (10, 20)],
            category="via",
            shape_hint="box",
            bbox=(10, 10, 11, 11),
        )
        view.widget.polygon_editor.set_polygons([contact])
        view.widget.polygon_editor._editor_scene.select_polygon(1)
        app.processEvents()

        assert view._selection_status_label.text() == "Выделено контактов: 1"

        view.widget.polygon_editor._editor_scene.select_polygon(None)
        app.processEvents()
        assert view._selection_status_label.text() == ""

        polygon = contact.clone()
        polygon.category = "conductor"
        polygon.shape_hint = "polygon"
        view.widget.polygon_editor.set_polygons([polygon])
        view.widget.polygon_editor._editor_scene.select_polygon(1)
        app.processEvents()
        assert view._selection_status_label.text() == "Выделено полигонов: 1"
    finally:
        view.close()
        view.deleteLater()


def test_main_view_exposes_file_menu_new_project_action() -> None:
    _app()
    view = ContourMainView()
    try:
        file_menu = view.menuBar().actions()[0].menu()

        assert file_menu is not None
        assert file_menu.title() in {"Файл", "File"}
        assert any(action.text() in {"Новый проект", "New project"} for action in file_menu.actions())
        load_labels = {action.text() for action in file_menu.actions()}
        assert "Загрузить кадры из папки…" in load_labels or "Load frames from folder…" in load_labels
        assert "Загрузить кадры…" in load_labels or "Load frame files…" in load_labels
        assert "Загрузить векторы из папки…" in load_labels or "Load vectors from folder…" in load_labels
        assert "Убрать вектор" in load_labels or "Clear vectors" in load_labels
        assert view.findChild(QAction, "loadImagesFolderAction") is not None
        assert view.findChild(QAction, "clearVectorsAction") is not None
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
        assert recognition_dock.widget() is view.widget.recognition_panel_scroll
        assert view.widget.recognition_panel_scroll.widget() is view.widget.extraction_tab
        assert dock is not None
        assert dock.widget() is view.widget.thumbnail_matrix_panel
        assert view.widget.thumbnail_grid_scroll_area.widget() is view.widget.thumbnail_grid
        assert not view.widget.thumbnail_grid_label.isVisible()
    finally:
        view.close()
        view.deleteLater()


def test_middle_button_only_hides_vectors_and_filter_checkbox_controls_raster() -> None:
    app = _app()
    view = ContourMainView()
    try:
        editor = view.widget.polygon_editor
        source = np.full((100, 100), 23, dtype=np.uint8)
        filtered = np.full((100, 100), 197, dtype=np.uint8)
        image_path = "middle-preview.png"
        conductor = PolygonData(
            id=1,
            points=[(20, 20), (80, 20), (80, 80), (20, 80)],
            category="conductor",
            shape_hint="polygon",
            bbox=(20, 20, 61, 61),
        )
        view.widget._workspace.apply_loaded_frame(
            image_path,
            source_image=source,
            polygons=[conductor],
        )
        state = view.widget._workspace.current_state
        assert state is not None
        state.reference_polygons = [conductor.clone()]
        view.widget._pending_restore_current_image_path = None
        view.show()

        view._pipeline_dock.raise_()
        app.processEvents()
        view.widget._sync_current_state_views(sync_neighbors=False)
        source_key = (image_path, "source", "")
        view.widget._editor_display_thread_pool.waitForDone()
        for _attempt in range(100):
            app.processEvents()
            if source_key in view.widget._editor_pixmap_cache:
                break
            QTest.qWait(5)
        assert source_key in view.widget._editor_pixmap_cache

        view.widget._prepared_image_running_request_id = 7
        view.widget._on_prepared_image_result(
            7,
            image_path,
            filtered,
            view.widget.get_pipeline(),
        )
        filtered_key = view.widget._editor_display_cache_key(image_path, state)
        view.widget._editor_display_thread_pool.waitForDone()
        for _attempt in range(100):
            app.processEvents()
            if filtered_key in view.widget._editor_pixmap_cache:
                break
            QTest.qWait(5)
        assert filtered_key in view.widget._editor_pixmap_cache
        assert editor._editor_scene._image_item.pixmap().toImage().pixelColor(0, 0).red() == 197
        app.processEvents()
        viewport_point = editor.mapFromScene(QPointF(10.0, 10.0))
        rendered = editor.viewport().grab().toImage()
        assert rendered.pixelColor(viewport_point).red() >= 190

        assert not hasattr(view.widget, "pipeline_preset_combo")
        assert not hasattr(view.widget, "apply_pipeline_preset_button")
        assert not hasattr(view.widget, "auto_tune_button")
        assert view.widget.show_applied_filters_checkbox.isChecked()
        QTest.mousePress(editor.viewport(), Qt.MouseButton.MiddleButton)
        app.processEvents()
        assert not editor.polygon_overlays_visible()
        assert editor._editor_scene._image_item.pixmap().toImage().pixelColor(0, 0).red() == 197
        QTest.mouseRelease(editor.viewport(), Qt.MouseButton.MiddleButton)
        app.processEvents()
        assert editor.polygon_overlays_visible()
        assert editor._editor_scene._image_item.pixmap().toImage().pixelColor(0, 0).red() == 197

        view.widget.show_applied_filters_checkbox.setChecked(False)
        app.processEvents()
        assert editor.polygon_overlays_visible()
        assert editor._editor_scene._image_item.pixmap().toImage().pixelColor(0, 0).red() == 23
        rendered = editor.viewport().grab().toImage()
        assert rendered.pixelColor(viewport_point).red() <= 30
        view.widget.show_applied_filters_checkbox.setChecked(True)
        app.processEvents()
        assert editor._editor_scene._image_item.pixmap().toImage().pixelColor(0, 0).red() == 197
        rendered = editor.viewport().grab().toImage()
        assert rendered.pixelColor(viewport_point).red() >= 190

        view._recognition_dock.raise_()
        app.processEvents()
        QTest.mousePress(editor.viewport(), Qt.MouseButton.MiddleButton)
        app.processEvents()
        assert not editor.polygon_overlays_visible()
        assert editor._editor_scene._image_item.pixmap().toImage().pixelColor(0, 0).red() == 197
        QTest.mouseRelease(editor.viewport(), Qt.MouseButton.MiddleButton)
        app.processEvents()
        assert editor.polygon_overlays_visible()
    finally:
        view.close()
        view.deleteLater()


def test_main_view_uses_flat_dock_roots_and_full_width_editor_toolbar() -> None:
    app = _app()
    view = ContourMainView()
    try:
        widget = view.widget
        widget.recognition_mode_combo.setCurrentIndex(widget.recognition_mode_combo.findData("via"))
        widget.via_search_mode_combo.setCurrentIndex(widget.via_search_mode_combo.findData("template"))
        widget._via_template_images = [np.zeros((11, 11), dtype=np.uint8)]
        widget._via_template_min_scores = [0.35]
        widget._via_template_diameters = [11]
        widget._refresh_via_template_list()
        view.resize(1400, 900)
        view.show()
        view._recognition_dock.raise_()
        app.processEvents()

        toolbar = view.findChild(QToolBar, "editorToolsToolbar")
        assert toolbar is not None
        assert toolbar.width() >= view.width() - 4
        assert toolbar.isAncestorOf(widget.editor_toolbar_scroll)
        assert not isinstance(widget.editor_group, QGroupBox)

        assert not isinstance(widget.path_group, QGroupBox)
        assert not widget.path_panel.isVisible()
        assert widget.files_tab.isAncestorOf(widget.extra_layers_group)
        assert not isinstance(widget.run_group, QGroupBox)
        assert view._run_dock.isAncestorOf(widget.save_group)
        assert not widget.extraction_tab.isAncestorOf(widget.save_group)

        assert not widget.recognition_panel_scroll.horizontalScrollBar().isVisible()
        assert not widget.via_template_table.horizontalScrollBar().isVisible()
        assert widget.via_template_table.height() < 120
    finally:
        view.close()
        view.deleteLater()


def test_main_view_gives_central_image_area_default_priority() -> None:
    app = _app()
    view = ContourMainView()
    try:
        # Verify the default layout independently of state saved by another
        # window test in the same QApplication session.
        view._pending_window_layout_restore = False
        view.resize(1400, 900)
        view.show()
        app.processEvents()
        view._apply_default_dock_layout()
        app.processEvents()

        files_dock = view.findChild(QDockWidget, "filesDock")
        pipeline_dock = view.findChild(QDockWidget, "pipelineDock")
        thumbnail_dock = view.findChild(QDockWidget, "thumbnailMatrixDock")

        assert files_dock is not None
        assert pipeline_dock is not None
        assert thumbnail_dock is not None
        assert files_dock.minimumWidth() >= DEFAULT_RIGHT_DOCK_MIN_WIDTH
        assert thumbnail_dock.minimumWidth() >= DEFAULT_RIGHT_DOCK_MIN_WIDTH
        assert pipeline_dock.minimumWidth() >= DEFAULT_PANEL_DOCK_MIN_WIDTH
        assert view.widget.width() > files_dock.width()
        assert view.widget.width() > pipeline_dock.width()
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


def test_main_view_default_left_dock_tab_order_is_pipeline_recognition_display() -> None:
    _app()
    view = ContourMainView()
    try:
        view.set_ui_language("en")
        expected = ["Pipeline", "Recognition", "Display"]
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
        assert "Пути" not in action_texts
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


def test_main_view_does_not_duplicate_update_menu_on_language_change() -> None:
    _app()
    view = ContourMainView()
    try:
        view.set_ui_language("en")
        view.set_ui_language("ru")
        update_menus = [
            action.menu()
            for action in view._help_menu.actions()
            if action.menu() is not None and action.menu().objectName() == "contourUpdateMenu"
        ]
        assert len(update_menus) == 1
        assert update_menus[0].title() == "Обновление"
    finally:
        view.close()
        view.deleteLater()
