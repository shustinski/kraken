"""Each menu uses one fresh permission snapshot, never one request per action."""

from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QMenu

from kraken_hub.presentation.qt.manager_app import DesktopController
from kraken_manager.domain.identity import Permission
from kraken_manager.presentation.qt.layer_management_parts.format_scalar import (
    LayerPipelineSnapshot,
    PipelineLane,
    PipelineNode,
)
from kraken_manager.presentation.qt.layer_management_parts.layer_manager_dialog import LayerManagerDialog
from kraken_manager.presentation.qt.models import LayerListItem


@pytest.mark.parametrize("target", ["layer", "pipeline"])
def test_menu_fetches_permissions_once_and_refreshes_after_revocation(qtbot, monkeypatch, target):
    calls = []
    permissions = frozenset({Permission.ASSIGN_WORK, Permission.MANAGE_STRUCTURE, Permission.RUN_PLUGIN})

    def project_permissions(project, principal):
        calls.append((project, principal))
        return permissions

    controller = object.__new__(DesktopController)
    controller._project_id = "project"
    controller.session = SimpleNamespace(principal=object())
    controller.service = SimpleNamespace(project_permissions=project_permissions)
    controller.plugin_items = {}
    dialog = LayerManagerDialog(
        "project",
        action_availability=controller._action_availability,
        action_availability_factory=controller._menu_action_availability,
    )
    qtbot.addWidget(dialog)
    dialog.set_layers([LayerListItem("layer", "Layer", "source", "#ffffff")])
    dialog.set_pipeline(
        LayerPipelineSnapshot(
            "project", "layer", (PipelineLane("lane", "Lane", (PipelineNode("node", "Source", "source"),), ()),)
        )
    )

    def choose(menu, _position):
        rename = (
            next(action for action in menu.actions() if action.text() == "Переименовать…")
            if target == "layer"
            else next(action for action in menu.actions() if action.text() == "Выдать работу…")
        )
        assert rename.isEnabled() == bool(permissions)
        assert rename.isVisible() == bool(permissions)
        return None

    monkeypatch.setattr(QMenu, "exec", choose)

    def open_menu():
        if target == "layer":
            dialog.layer_list._context_menu(dialog.layer_list.visualItemRect(dialog.layer_list.item(0)).center())
        else:
            dialog.graph._items["node"].contextMenuEvent(SimpleNamespace(screenPos=lambda: QPoint()))

    open_menu()
    assert len(calls) == 1
    permissions = frozenset()
    open_menu()
    assert len(calls) == 2
    assert not controller._action_availability("rename_layer")[0]
    assert len(calls) == 3
