"""test_hub_manager_app: test matrix context menu."""
from __future__ import annotations
from types import SimpleNamespace
import pytest
from PyQt6.QtWidgets import (
    QMenu,
    QWidget,
)
from kraken_core.frame_matrix.qt import FrameRect
from kraken_hub.manager_app import (
    DesktopController,
)
from kraken_manager.domain.identity import Permission
from kraken_manager.presentation.qt import (
    FrameSelection,
)

def test_matrix_context_menu_exports_and_opens_vector_properties(
    qapp,
    monkeypatch,
) -> None:
    controller = object.__new__(DesktopController)
    controller.shell = QWidget()
    series = SimpleNamespace(name="frame.cif", representation_id="vector-1")
    version = SimpleNamespace(filename="frame.cif")
    controller._active_frame_files = lambda _x, _y: (
        ("CIF", series, version),
    )
    controller._workspace = SimpleNamespace(
        vector_representation_combo=SimpleNamespace(currentData=lambda: "vector-1"),
    )
    exported = []
    opened = []
    controller._export_selected_frame_files = exported.append
    controller._show_frame_file_properties = (
        lambda *values: opened.append(values)
    )
    context = SimpleNamespace(
        x=2,
        y=3,
        selection=FrameSelection.single(2, 3),
    )

    monkeypatch.setattr(
        QMenu,
        "exec",
        lambda menu, *_args: next(
            action
            for action in menu.actions()
            if action.text().startswith("Выгрузить файлы")
        ),
    )
    controller._show_matrix_context_menu(context, controller.shell.pos())
    assert exported == [context]

    def select_vector_properties(menu: QMenu, *_args):
        return next(
            action
            for action in menu.actions()
            if action.text() == "Свойства вектора…"
        )

    monkeypatch.setattr(QMenu, "exec", select_vector_properties)
    controller._show_matrix_context_menu(context, controller.shell.pos())
    assert opened == [("CIF", series, version)]

def test_matrix_context_menu_hides_properties_for_multiple_frames(qapp, monkeypatch) -> None:
    controller = object.__new__(DesktopController)
    controller.shell = QWidget()
    controller._project_id = "project-1"
    controller._workspace = SimpleNamespace(
        vector_representation_combo=SimpleNamespace(currentData=lambda: "vector-1"),
    )
    controller._has_permission = lambda _permission: False
    controller._active_frame_files = lambda _x, _y: pytest.fail(
        "properties must not inspect one frame when several frames are selected"
    )
    context = SimpleNamespace(
        x=2,
        y=3,
        selection=FrameSelection((FrameRect(2, 3, 3, 3),)),
    )

    def inspect_menu(menu: QMenu, *_args):
        labels = tuple(action.text() for action in menu.actions())
        assert "Свойства вектора…" not in labels
        assert "Свойства кадра…" not in labels

    monkeypatch.setattr(QMenu, "exec", inspect_menu)

    controller._show_matrix_context_menu(context, controller.shell.pos())

def test_matrix_context_menu_preserves_review_action(qapp, monkeypatch) -> None:
    controller = object.__new__(DesktopController)
    controller.shell = QWidget()
    controller._project_id = "project-1"
    controller._active_frame_files = lambda _x, _y: ()
    controller._has_permission = lambda permission: permission is Permission.ASSIGN_WORK
    requested = []
    controller.send_selection_for_review = lambda: requested.append(True)
    context = SimpleNamespace(
        x=2,
        y=3,
        selection=FrameSelection.single(2, 3),
    )
    monkeypatch.setattr(
        QMenu,
        "exec",
        lambda menu, *_args: next(
            action
            for action in menu.actions()
            if action.text() == "Выдать работу…"
        ),
    )

    controller._show_matrix_context_menu(context, controller.shell.pos())

    assert requested == [True]

    requested.clear()
    monkeypatch.setattr(QMenu, "exec", lambda *_args: None)
    controller._show_matrix_context_menu(context, controller.shell.pos())
    assert requested == []
