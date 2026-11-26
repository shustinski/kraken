"""test_hub_manager_app: test matrix context menu."""
from __future__ import annotations
from types import SimpleNamespace
import pytest
from PyQt6.QtWidgets import (
    QWidget,
)
from kraken_core.frame_matrix.qt import FrameContext, FrameRect
from kraken_hub.presentation.qt.layer_matrix import LayerFrameMatrixWidget
from kraken_hub.presentation.qt.manager_app import (
    DesktopController,
)
from kraken_manager.domain.identity import Permission
from kraken_manager.presentation.qt import (
    FrameSelection,
)


def _titles(controller, selection: FrameSelection) -> tuple[str, ...]:
    matrix = LayerFrameMatrixWidget(4, 4)
    matrix.set_action_host(controller)
    context = FrameContext(2, 3, matrix.cell_data(2, 3), selection)
    return tuple(title for _action, title in matrix.visible_frame_actions(context))


def test_matrix_context_menu_exports_and_opens_vector_properties(qapp) -> None:
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

    assert "Свойства вектора…" in _titles(controller, context.selection)
    controller._on_matrix_frame_action("export_files", context)
    assert exported == [context]

    controller._on_matrix_frame_action("vector_properties", context)
    assert opened == [("CIF", series, version)]


def test_matrix_context_menu_hides_properties_for_multiple_frames(qapp) -> None:
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

    labels = _titles(controller, FrameSelection((FrameRect(2, 3, 3, 3),)))

    assert "Свойства вектора…" not in labels
    assert "Свойства кадра…" not in labels


def test_matrix_context_menu_preserves_review_action(qapp) -> None:
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

    assert "Выдать работу…" in _titles(controller, context.selection)
    controller._on_matrix_frame_action("assign_work", context)

    assert requested == [True]
