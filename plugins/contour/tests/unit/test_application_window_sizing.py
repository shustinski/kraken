from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

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
