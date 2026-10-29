from __future__ import annotations

import pytest
from types import SimpleNamespace

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication, QPushButton, QTableWidgetItem, QWidget
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QImage, QPainter

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


def test_role_graph_background_renders_fractional_scene_coordinates(qapp) -> None:
    del qapp
    view = RoleGraphView()
    image = QImage(200, 160, QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    try:
        view.drawBackground(painter, QRectF(-35.5, -20.25, 180.75, 140.5))
    finally:
        painter.end()


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


def test_admin_project_role_graph_button_opens_and_paints_dialog(qtbot) -> None:
    from kraken_admin.presentation.qt.roles_panel import RolesPanel

    parent = QWidget()
    qtbot.addWidget(parent)
    person = SimpleNamespace(id="11111111-1111-1111-1111-111111111111", display_name="Admin")
    admin = SimpleNamespace(
        window=parent,
        _services=SimpleNamespace(project_roles=lambda *_args: {"roles": ["maintainer"]}),
    )
    panel = RolesPanel(admin)
    qtbot.addWidget(panel)
    panel.people = [person]
    panel.projects = [{"project_id": "project", "name": "Demo"}]
    panel.table.setRowCount(1)
    panel.table.setColumnCount(2)
    panel.table.setItem(0, 1, QTableWidgetItem("Maintainer"))
    panel.table.setCurrentCell(0, 1)
    rendered = []

    def inspect_dialog() -> None:
        dialog = parent.findChild(RoleManagementDialog)
        if dialog is None:
            return
        try:
            # grab forces painting, including the QGraphicsView background.
            rendered.append((dialog.graph.grab(), dialog._held()))
        finally:
            dialog.accept()

    timer = QTimer(parent)
    timer.setSingleShot(True)
    timer.timeout.connect(inspect_dialog)
    timer.start(100)
    button = next(button for button in panel.findChildren(QPushButton) if "Граф ролей" in button.text())
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert len(rendered) == 1
    assert not rendered[0][0].isNull()
    assert rendered[0][1] == {"maintainer"}
