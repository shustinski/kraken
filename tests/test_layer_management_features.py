from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from kraken_agent.jobs import DurableJobStore
from kraken_core.external_model import ExternalModelLink
from kraken_core.plugin_protocol import (
    PluginAsset,
    PluginAssetScope,
    PluginJobManifestV2,
    PluginResultPublicationV2,
    parse_plugin_job_json,
)
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.domain import GridOrientation, LayerType


def _asset(identifier: str, path: str, *, role: str, frame_id: str | None = None) -> PluginAsset:
    return PluginAsset(
        asset_id=identifier,
        role=role,
        scope=PluginAssetScope.LAYER if frame_id is None else PluginAssetScope.FRAME,
        relative_path=path,
        sha256="a" * 64,
        media_type="application/octet-stream",
        frame_id=frame_id,
        x=None if frame_id is None else 1,
        y=None if frame_id is None else 1,
    )


def test_plugin_protocol_v2_allows_role_aware_multi_inputs_and_publications(tmp_path: Path) -> None:
    frame_id = str(uuid4())
    manifest = PluginJobManifestV2(
        job_id=str(uuid4()),
        operation="layer.confidence.analyze.v1",
        project_id=str(uuid4()),
        layer_id=str(uuid4()),
        actor_id=str(uuid4()),
        inputs=(
            _asset("source", "inputs/source.png", role="source", frame_id=frame_id),
            _asset("binary", "inputs/binary.png", role="binary", frame_id=frame_id),
        ),
    )
    assert isinstance(parse_plugin_job_json(manifest.to_json()), PluginJobManifestV2)

    store = DurableJobStore(tmp_path / "agent.sqlite3")
    job = store.enqueue(manifest)
    first = PluginResultPublicationV2(
        job_id=manifest.job_id,
        publication_id=str(uuid4()),
        sequence=1,
        plugin_id="karakal",
        plugin_version="2.0",
        outputs=(),
        frame_values={frame_id: {"confidence": 0.25}},
    )
    job, duplicate = store.record_result(
        first,
        callback_key=f"publication:{first.publication_id}",
        expected_revision=job.revision,
    )
    assert not duplicate
    second = PluginResultPublicationV2(
        job_id=manifest.job_id,
        publication_id=str(uuid4()),
        sequence=2,
        plugin_id="karakal",
        plugin_version="2.0",
        outputs=(),
        frame_values={frame_id: {"confidence": 0.75}},
        final=True,
    )
    store.record_result(
        second,
        callback_key=f"publication:{second.publication_id}",
        expected_revision=job.revision,
    )
    assert [item.sequence for item in store.list_publications(manifest.job_id)] == [1, 2]


def test_external_model_link_rehashes_each_staged_run(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"first")
    link = ExternalModelLink.observe(model)
    model.write_bytes(b"second")

    staged = link.stage(tmp_path / "staging")

    assert staged.changed_since_observation
    assert staged.observed_sha256 == link.observed_sha256
    assert (tmp_path / "staging" / staged.relative_path).read_bytes() == b"second"


def test_pipeline_step_removal_is_an_audited_tombstone(tmp_path: Path) -> None:
    service = EmbeddedProjectService(tmp_path / "data")
    session = service.create_initial_account("owner", "Owner", "")
    project = service.create_project(
        principal=session.principal,
        name="Pipeline",
        width=1,
        height=1,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key="project",
    )
    layer = service.create_layer(
        principal=session.principal,
        project=project,
        name="Metal",
        layer_type=LayerType.METAL,
        order=0,
        idempotency_key="layer",
    )
    action = service.record_layer_pipeline_action(
        principal=session.principal,
        project_id=project.id,
        layer_id=layer.id,
        action="prepare_dataset",
        node_id="source",
        plugin_id="contour",
        capability="frames.dataset.prepare.v1",
        mode="interactive",
    )

    removed = service.remove_layer_pipeline_action(
        principal=session.principal,
        project_id=project.id,
        layer_id=layer.id,
        action_event_id=action.event_id,
    )

    assert removed.event_type == "LayerPipelineActionRemoved"
    assert removed.payload["action_event_id"] == action.event_id
    assert any(event.event_id == action.event_id for event in service.history(project.id))


pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, QSettings  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMenu  # noqa: E402

from kraken_manager.presentation.qt import (  # noqa: E402
    FrameCellData,
    FrameMatrixView,
    LayerListItem,
    LayerManagerDialog,
    LayerPipelineSnapshot,
    ObjectHistoryEntry,
    ObjectPropertiesDialog,
    ObjectPropertiesSnapshot,
    PipelineLane,
    PipelineNode,
    ProjectManagerShell,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_workspace_menus_and_modeless_layer_manager(qapp, monkeypatch) -> None:
    settings = QSettings("Kraken", "KrakenHub")
    settings.remove("matrix/border-mode")
    settings.remove("matrix/fill-mode")
    shell = ProjectManagerShell()
    workspace = shell.open_project_workspace()

    assert shell.layers_action.text().startswith("Слои")
    assert shell.layers_action.isEnabled()
    assert shell.view_menu.isEnabled()
    assert shell.visual_modes() == ("status", "thumbnail")
    assert not workspace.image_representation_combo.isVisible()
    assert not workspace.vector_representation_combo.isVisible()

    dialog = LayerManagerDialog(str(uuid4()), shell)
    dialog.set_layers(
        [
            LayerListItem("layer-a", "Metal 1", "metal"),
            LayerListItem("layer-b", "Metal 2", "metal"),
        ],
        "layer-a",
    )
    dialog.set_pipeline(
        LayerPipelineSnapshot(
            dialog.project_id,
            "layer-a",
            (
                PipelineLane(
                    "empty-lane",
                    "Empty source",
                    (
                        PipelineNode("source", "Исходники", "source"),
                        PipelineNode("missing", "CIF не получен", "missing"),
                    ),
                    (("source", "missing"),),
                ),
                PipelineLane(
                    "lane-a",
                    "Source A",
                    (
                        PipelineNode("source-a", "Исходник A", "source"),
                        PipelineNode("dataset-a", "Prepared dataset", "dataset"),
                        PipelineNode("missing-a", "CIF не получен", "missing"),
                    ),
                    (("source-a", "dataset-a"), ("dataset-a", "missing-a")),
                ),
                PipelineLane(
                    "lane-b",
                    "Source B",
                    (
                        PipelineNode("source-b", "Исходник B", "source"),
                        PipelineNode("job-b", "Recognition", "job"),
                        PipelineNode("missing-b", "CIF не получен", "missing"),
                    ),
                    (("source-b", "job-b"), ("job-b", "missing-b")),
                ),
            ),
        )
    )

    assert not dialog.isModal()
    assert dialog.layer_list.layer_ids() == ("layer-a", "layer-b")
    assert dialog.graph.scene().items()
    blackbox_ids = {
        item.node.node_id
        for item in dialog.graph.scene().items()
        if getattr(item, "node", None) is not None and item.node.kind == "blackbox"
    }
    assert blackbox_ids == {"lane-a:blackbox", "lane-b:blackbox"}

    dialog.graph.expand_lane("lane-a")
    assert dialog.graph._expanded_lane_id == "lane-a"
    assert "dataset-a" in dialog.graph._items
    assert "lane-a:blackbox" not in dialog.graph._items
    assert "lane-b:blackbox" in dialog.graph._items
    assert dialog.graph._items["dataset-a"]._collapse is not None

    dialog.graph.expand_lane("lane-b")
    assert dialog.graph._expanded_lane_id == "lane-b"
    assert "lane-a:blackbox" in dialog.graph._items
    assert "lane-b:blackbox" not in dialog.graph._items

    dialog.graph.collapse_lane("lane-b")
    assert dialog.graph._expanded_lane_id == ""
    assert "lane-a:blackbox" in dialog.graph._items
    assert "lane-b:blackbox" in dialog.graph._items
    dialog.show()
    qapp.processEvents()
    assert not dialog.graph.grab().isNull()

    requested_menu_label = {"value": "Добавить слой изображений…"}

    def select_requested_action(menu: QMenu, *_args):
        return next(
            action
            for action in menu.actions()
            if action.text() == requested_menu_label["value"]
        )

    monkeypatch.setattr(QMenu, "exec", select_requested_action)
    layer_actions = []
    dialog.layerActionRequested.connect(
        lambda layer_id, action: layer_actions.append((layer_id, action))
    )
    first_item = dialog.layer_list.item(0)
    dialog.layer_list._context_menu(
        dialog.layer_list.visualItemRect(first_item).center()
    )
    assert layer_actions[-1] == ("layer-a", "add_image_representation")
    layer_properties = []
    dialog.layerPropertiesRequested.connect(layer_properties.append)
    requested_menu_label["value"] = "Свойства"
    dialog.layer_list._context_menu(
        dialog.layer_list.visualItemRect(first_item).center()
    )
    assert layer_properties == ["layer-a"]

    class ContextEvent:
        @staticmethod
        def screenPos():
            return QPoint(0, 0)

    node_actions = []
    dialog.nodeActionRequested.connect(
        lambda layer_id, node, action: node_actions.append(
            (layer_id, node.node_id, action)
        )
    )
    requested_menu_label["value"] = "Удалить слой изображений из проекта"
    dialog.graph._items["source-a"].contextMenuEvent(ContextEvent())
    assert node_actions[-1] == ("layer-a", "source-a", "archive_representation")

    requested_menu_label["value"] = "Добавить CIF из внешнего источника…"
    dialog.graph._items["missing-a"].contextMenuEvent(ContextEvent())
    assert node_actions[-1] == ("layer-a", "missing-a", "add_external_vector")

    node_properties = []
    dialog.nodePropertiesRequested.connect(
        lambda layer_id, node: node_properties.append((layer_id, node.node_id))
    )
    requested_menu_label["value"] = "Свойства"
    dialog.graph._items["missing-a"].contextMenuEvent(ContextEvent())
    dialog.graph._items["lane-b:blackbox"].contextMenuEvent(ContextEvent())
    assert node_properties == [
        ("layer-a", "missing-a"),
        ("layer-a", "lane-b:blackbox"),
    ]
    dialog.close()


def test_object_properties_dialog_renders_nested_values_and_local_history(qapp) -> None:
    snapshot = ObjectPropertiesSnapshot(
        title="Metal 1",
        object_kind="layer",
        properties=(
            ("Название", "Metal 1"),
            ("Примечание", None),
            ("Параметры", {"quality": 95, "flags": ["a", "b"]}),
        ),
        history=(
            ObjectHistoryEntry(
                "2026-07-28T10:00:00+00:00",
                "Оператор",
                "LayerCreated",
                {
                    "layer_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                    "name": "Metal 1",
                    "type": "binary",
                    "order": 2,
                    "state": "active",
                },
            ),
        ),
    )

    dialog = ObjectPropertiesDialog(snapshot)

    assert dialog.windowTitle() == "Свойства: Metal 1"
    assert dialog.properties_table.item(1, 1).text() == "—"
    assert '"quality": 95' in dialog.properties_table.item(2, 1).text()
    assert dialog.history_table.item(0, 1).text() == "Оператор"
    assert dialog.history_table.item(0, 2).text() == "Создан слой"
    assert dialog.history_table.item(0, 2).toolTip() == "LayerCreated"
    details = dialog.history_table.item(0, 3).text()
    assert "Название: Metal 1" in details
    assert "Тип: binary" in details
    assert "layer_id" not in details
    assert dialog.history_table.item(0, 0).toolTip() == "2026-07-28T10:00:00+00:00"
    dialog.close()


def test_history_payload_summary_prefers_human_fields_over_ids() -> None:
    from kraken_manager.presentation.qt.layer_management import (
        _event_type_label,
        _format_history_payload,
    )

    assert _event_type_label("ReviewChangesRequested") == "Запрошена доработка"
    summary = _format_history_payload(
        "ProjectCreated",
        {
            "project_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "name": "Demo",
            "width": 12,
            "height": 8,
            "orientation": "xy",
            "state": "active",
        },
    )
    assert "Название: Demo" in summary
    assert "Ширина: 12" in summary
    assert "Состояние: активен" in summary
    assert "project_id" not in summary

    rename = _format_history_payload(
        "ProjectRenamed",
        {"project": {"id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "name": "New name", "state": "active"}},
    )
    assert "Название: New name" in rename

    review = _format_history_payload(
        "ReviewChangesRequested",
        {
            "review_batch_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "reason": "Нужны правки по контуру",
            "state": "changes_requested",
        },
    )
    assert "Причина: Нужны правки по контуру" in review
    assert "Состояние: запрошена доработка" in review


def test_matrix_semantic_colors_follow_time_quality_and_review_rules(qapp) -> None:
    view = FrameMatrixView(2, 1)
    oldest = FrameCellData(
        1,
        1,
        payload={
            "modified_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
            "quality": 0.0,
            "review_status": "not_checked",
        },
    )
    newest = FrameCellData(
        2,
        1,
        payload={
            "modified_at": datetime(2026, 1, 2, tzinfo=UTC).isoformat(),
            "quality": 1.0,
            "review_status": "checked",
        },
    )
    view.set_cells((oldest, newest))

    old_time = view._cell_visual_color(oldest, "time")
    new_time = view._cell_visual_color(newest, "time")
    assert old_time.red() > old_time.green()
    assert new_time.green() > new_time.red()
    assert view._cell_visual_color(oldest, "quality").red() > 200
    assert view._cell_visual_color(newest, "quality").green() > 200
    assert view._cell_visual_color(oldest, "status").name() == "#64748b"
    assert view._cell_visual_color(newest, "status").name() == "#22c55e"
