from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QPoint, QPointF, QSettings, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QGraphicsTextItem, QInputDialog, QMenu, QMessageBox

from kraken_manager.presentation.qt.layer_management_parts.format_scalar import (
    LayerPipelineSnapshot, PipelineLane, PipelineNode,
)
from kraken_manager.presentation.qt.layer_management_parts.pipeline_graph_view import PipelineGraphView
from kraken_hub.presentation.qt.parts.layer_manager_action import LayerManagerActionMixin
from kraken_hub.presentation.qt.parts.archive_layer_representation import ArchiveLayerRepresentationMixin


def graph(qtbot, tmp_path, nodes):
    settings = QSettings(str(tmp_path / "graph.ini"), QSettings.Format.IniFormat)
    settings.setValue("layer-manager/project/layer/expanded-lane", "lane")
    view = PipelineGraphView(settings)
    qtbot.addWidget(view)
    view.resize(850, 400)
    snapshot = LayerPipelineSnapshot("project", "layer", (
        PipelineLane("lane", "Lane", tuple(nodes), ()),
    ))
    view.set_snapshot(snapshot)
    view.show()
    return view, snapshot


@pytest.mark.parametrize("kind", ["source", "vector"])
def test_drag_does_not_activate_and_survives_refresh(qtbot, tmp_path, kind):
    view, snapshot = graph(qtbot, tmp_path, [PipelineNode("node", "Title", kind)])
    activated = []
    view.nodeActivated.connect(activated.append)
    item = view._items["node"]
    original = QPointF(item.pos())
    start = view.mapFromScene(item.mapToScene(QPointF(80, 20)))
    end = start + QPoint(70, 45)
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view.viewport(), end, delay=30)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
    assert not activated
    assert item.pos() != original
    moved = QPointF(item.pos())
    view.set_snapshot(snapshot)
    assert view._items["node"].pos() == moved
    click = view.mapFromScene(view._items["node"].mapToScene(QPointF(80, 20)))
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=click)
    assert len(activated) == 1


def test_note_wraps_and_persists_and_can_be_cleared(qtbot, tmp_path, monkeypatch):
    view, snapshot = graph(qtbot, tmp_path, [PipelineNode("job", "Title", "job", "Status")])
    original_height = view._items["job"].rect().height()
    body = "A long note with multiple words " * 6 + "\nSecond line"
    monkeypatch.setattr(QInputDialog, "getMultiLineText", lambda *a, **k: (body, True))
    view._request_node_action(view._items["job"].node, "edit_node_note")
    view.set_snapshot(snapshot)
    item = view._items["job"]
    note = next(child for child in item.childItems() if isinstance(child, QGraphicsTextItem))
    assert note.toPlainText() == body
    assert item.rect().height() > original_height
    assert note.pos().y() == 30
    assert note.boundingRect().width() <= 166
    monkeypatch.setattr(QInputDialog, "getMultiLineText", lambda *a, **k: ("", True))
    view._request_node_action(item.node, "edit_node_note")
    assert view._items["job"].rect().height() == original_height


def test_new_pipeline_node_does_not_overlap_saved_job_position(qtbot, tmp_path):
    source = PipelineNode("source", "Images", "source")
    job = PipelineNode("job", "Completed", "job")
    view, _ = graph(qtbot, tmp_path, [source, job])
    view.set_snapshot(
        LayerPipelineSnapshot(
            "project", "layer",
            (PipelineLane("lane", "Lane", (source, PipelineNode("dataset", "Dataset", "dataset"), job), ()),),
        )
    )

    assert not view._items["dataset"].sceneBoundingRect().intersects(view._items["job"].sceneBoundingRect())


@pytest.mark.parametrize("state, present", [("waiting_for_user", True), ("succeeded", False)])
def test_continue_menu_dispatch(qtbot, tmp_path, monkeypatch, state, present):
    view, _ = graph(qtbot, tmp_path, [PipelineNode(
        "job", "Title", "job", state=state, details={"job_id": "job"},
    )])
    observed = []
    view.nodeActionRequested.connect(lambda node, action: observed.append(action))

    def choose(menu, _pos):
        action = next((a for a in menu.actions() if a.text() == "Продолжить выполнение"), None)
        assert (action is not None) == present
        return action

    monkeypatch.setattr(QMenu, "exec", choose)
    view._items["job"].contextMenuEvent(SimpleNamespace(screenPos=lambda: QPoint()))
    assert observed == (["continue_pipeline_step"] if present else [])


def test_cif_menu_offers_removal(qtbot, tmp_path, monkeypatch):
    view, _ = graph(qtbot, tmp_path, [
        PipelineNode("cif-id", "CIF — Исходные изображения", "vector", representation_id="cif-id"),
    ])
    observed = []
    view.nodeActionRequested.connect(lambda node, action: observed.append((node.node_id, action)))

    def choose(menu, _pos):
        return next(action for action in menu.actions() if action.text() == "Удалить блок CIF из проекта")

    monkeypatch.setattr(QMenu, "exec", choose)
    view._items["cif-id"].contextMenuEvent(SimpleNamespace(screenPos=lambda: QPoint()))
    assert observed == [("cif-id", "archive_representation")]


def test_cif_removal_archives_representation_and_refreshes_board(monkeypatch):
    controller = ArchiveLayerRepresentationMixin()
    controller._project_id = "project"
    controller._workspace = object()
    controller._layer_dialog = SimpleNamespace(set_pipeline=lambda snapshot: refreshed.append(snapshot))
    controller.session = SimpleNamespace(principal="actor")
    representation = SimpleNamespace(id="cif-id", name="CIF — Исходные изображения")
    controller.service = SimpleNamespace(
        get_project=lambda _id: "project",
        list_layers=lambda _id: [SimpleNamespace(id="layer")],
        list_representations=lambda _project, _layer: [representation],
        archive_representation=lambda **kwargs: archived.append(kwargs),
    )
    controller._error = pytest.fail
    refreshed = []
    archived = []
    controller._load_representations = lambda *_args: None
    controller._pipeline_snapshot = lambda *_args: "updated board"
    prompts = []

    def confirm(_parent, title, message):
        prompts.append((title, message))
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", confirm)
    controller._archive_layer_representation(
        "layer", PipelineNode("cif-id", representation.name, "vector", representation_id="cif-id"),
    )
    assert archived[0]["representation"] is representation
    assert refreshed == ["updated board"]
    assert prompts[0][0] == "Удалить блок CIF"


def test_delete_pipeline_step_writes_a_profile(tmp_path, monkeypatch):
    from kraken_core.frame_matrix import profiling

    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text("[profiling]\npipeline_step_removal = true\n", encoding="utf-8")
    profiling.configure(tmp_path)
    controller = LayerManagerActionMixin()
    controller._project_id = "project"
    controller.session = SimpleNamespace(principal="actor")
    controller._layer_dialog = SimpleNamespace(set_pipeline=lambda snapshot: refreshed.append(snapshot))
    controller._pipeline_snapshot = lambda *_args: "updated board"
    controller._error = pytest.fail
    removed = []

    def remove_layer_pipeline_action(**kwargs):
        removed.append(kwargs)

    controller.service = SimpleNamespace(
        plugin_jobs=lambda: [],
        remove_layer_pipeline_action=remove_layer_pipeline_action,
    )
    refreshed = []
    node = PipelineNode(
        "step",
        "Step",
        "operation",
        details={"pipeline_event_id": "event-1", "deletable": True},
    )

    controller._layer_manager_action("layer", node, "delete_pipeline_step")

    report = (tmp_path / "profiling.log").read_text(encoding="utf-8")
    assert report.count("[pipeline_step_removal]") == 1
    assert "remove_layer_pipeline_action" in report
    assert removed[0]["action_event_id"] == "event-1"
    assert refreshed == ["updated board"]


def test_continue_opens_existing_waiting_job():
    controller = LayerManagerActionMixin()
    controller._project_id = "project"
    job = SimpleNamespace(id="job", project_id="project", layer_id="layer", state=SimpleNamespace(value="waiting_for_user"))
    controller.service = SimpleNamespace(plugin_jobs=lambda: [job], is_remote_project=lambda _: False)
    controller._action_availability = lambda _: (True, "")
    controller._error = pytest.fail
    calls = []
    controller.launch_selected_work = lambda *args: calls.append(args)
    controller._layer_manager_action("layer", PipelineNode("job", "Step", "job", details={"job_id": "job"}), "continue_pipeline_step")
    assert calls == [("job", job)]
