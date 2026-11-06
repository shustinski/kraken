"""Layer menu opening is profiled before waiting for a user's choice."""

from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QPoint, QSettings
from PyQt6.QtWidgets import QMenu

from kraken_manager.presentation.qt.layer_management_parts.format_scalar import (
    LayerPipelineSnapshot,
    PipelineLane,
    PipelineNode,
)
from kraken_manager.presentation.qt.layer_management_parts.layer_order_list import LayerOrderList
from kraken_manager.presentation.qt.layer_management_parts.pipeline_graph_view import PipelineGraphView
from kraken_manager.presentation.qt.models import LayerListItem
from kraken_manager.presentation.qt.layer_management_parts import menu_profiling


def test_disabled_profiling_produces_no_report(monkeypatch, capsys):
    monkeypatch.setattr(menu_profiling, "PROFILE_LAYER_ACTION_MENU", False)
    with menu_profiling.profile_menu_open("disabled") as checkpoint:
        checkpoint("availability:vectorize")
    assert not capsys.readouterr().err


def test_profile_reports_failed_preparation_and_propagates_error(capsys):
    import sys

    previous = sys.getprofile()
    with pytest.raises(ValueError, match="unavailable"):
        with menu_profiling.profile_menu_open("failed") as checkpoint:
            checkpoint("selection_and_actions")
            raise ValueError("unavailable")
    assert sys.getprofile() is previous
    assert "[layer-menu-profile] failed" in capsys.readouterr().err


@pytest.mark.parametrize("target", ["layer", "pipeline"])
def test_menu_profile_is_reported_before_exec(qtbot, tmp_path, monkeypatch, capsys, target):
    checked = []

    def availability(code):
        checked.append(code)
        return True, ""

    def choose(menu, position):
        report = capsys.readouterr().err
        assert f"[layer-menu-profile] {target}" in report
        assert "total_ms=" in report
        assert "availability:" + checked[0] in report
        assert "availability" in report
        assert "cumulative" in report
        return None

    monkeypatch.setattr(QMenu, "exec", choose)
    if target == "layer":
        widget = LayerOrderList(action_availability=availability)
        qtbot.addWidget(widget)
        widget.set_layers([LayerListItem("layer", "Layer", "source", "#ffffff")])
        widget._context_menu(widget.visualItemRect(widget.item(0)).center())
    else:
        settings = QSettings(str(tmp_path / "graph.ini"), QSettings.Format.IniFormat)
        settings.setValue("layer-manager/project/layer/expanded-lane", "lane")
        widget = PipelineGraphView(settings, action_availability=availability)
        qtbot.addWidget(widget)
        widget.set_snapshot(
            LayerPipelineSnapshot(
                "project", "layer", (PipelineLane("lane", "Lane", (PipelineNode("node", "Source", "source"),), ()),)
            )
        )
        widget._items["node"].contextMenuEvent(SimpleNamespace(screenPos=lambda: QPoint()))
    assert checked
