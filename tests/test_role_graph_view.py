from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

from kraken_manager.domain.identity import ProjectRole
from kraken_manager.presentation.qt.role_management import RoleGraphView, RoleManagementDialog


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def test_role_graph_view_initializes_its_qt_base(qapp) -> None:
    del qapp
    view = RoleGraphView()
    assert view.scene() is view.graph_scene
    assert set(view._nodes) == {role.value for role in ProjectRole}


def test_role_management_dialog_opens_with_an_empty_acting_role(qapp) -> None:
    del qapp

    class _Person:
        id = "11111111-1111-1111-1111-111111111111"
        display_name = "Admin"

    dialog = RoleManagementDialog(
        None,
        project_name="Demo",
        principals=[_Person()],
        roles_for=lambda _person: frozenset(),
        change_role=lambda *_args: None,
        acting_roles=[],
        on_acting_role=lambda _role: None,
    )
    assert dialog.graph.scene() is not None
    dialog.close()
