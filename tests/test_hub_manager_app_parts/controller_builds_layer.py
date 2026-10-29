"""test_hub_manager_app: test controller builds layer."""
from __future__ import annotations
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from kraken_hub.presentation.qt.manager_app import (
    DesktopController,
)
from kraken_manager.presentation.qt import (
    LayerPipelineSnapshot,
    PipelineLane,
    PipelineNode,
)

from test_hub_manager_app_parts.event import _event

def test_controller_builds_layer_and_representation_properties(tmp_path: Path) -> None:
    image_directory = tmp_path / "images"
    image_directory.mkdir()
    (image_directory / "0.jpg").write_bytes(b"image")
    layer = SimpleNamespace(
        id="layer-1",
        project_id="project-1",
        name="Metal 1",
        type=SimpleNamespace(value="metal"),
        order=2,
        state=SimpleNamespace(value="active"),
        revision=3,
        created_at=datetime(2026, 7, 28, 10, 0, tzinfo=UTC),
    )
    binding = SimpleNamespace(
        image_directory=str(image_directory),
        ssc_directory="",
        prv_directory="",
        aux_directory="",
        import_root=str(tmp_path),
        mode=SimpleNamespace(value="managed_copy"),
        frame_positions={"0.jpg": 1},
        conversion=SimpleNamespace(target_format="jpg", jpeg_quality=95),
    )
    representation = SimpleNamespace(
        id="representation-1",
        project_id="project-1",
        layer_id="layer-1",
        name="Исходные изображения",
        kind=SimpleNamespace(value="image"),
        purpose=SimpleNamespace(value="source"),
        note="Проверено",
        source=str(image_directory),
        source_image_representation_id=None,
        active=True,
        state=SimpleNamespace(value="active"),
        revision=1,
        created_at=datetime(2026, 7, 28, 10, 1, tzinfo=UTC),
    )
    events = (
        _event("LayerCreated", {"layer_id": "layer-1"}),
        _event(
            "RepresentationCreated",
            {
                "layer_id": "layer-1",
                "representation_id": "representation-1",
                "note": "Проверено",
            },
            minute=1,
            event_id="event-2",
            stream_id="layer:layer-1",
        ),
    )

    class Service:
        @staticmethod
        def list_layers(_project_id):
            return (layer,)

        @staticmethod
        def list_representations(_project_id, _layer_id):
            return (representation,)

        @staticmethod
        def layer_file_binding(_project_id, _layer_id):
            return binding

        @staticmethod
        def history(_project_id):
            return events

        @staticmethod
        def frame_cells(_project_id, _layer_id, _representation_id):
            return (SimpleNamespace(frame_id="frame-1"),)

        @staticmethod
        def list_derived_runs(_project_id, _layer_id):
            return ()

    controller = object.__new__(DesktopController)
    controller.service = Service()
    controller._project_id = "project-1"
    controller._pipeline_snapshot = lambda *_args: LayerPipelineSnapshot(
        "project-1",
        "layer-1",
        (
            PipelineLane(
                "representation-1",
                "Исходные изображения",
                (
                    PipelineNode(
                        "representation-1",
                        "Исходные изображения",
                        "source",
                        representation_id="representation-1",
                    ),
                ),
                (),
            ),
        ),
    )
    snapshots = []
    controller._open_properties = snapshots.append

    controller._show_layer_properties("layer-1")
    layer_properties = dict(snapshots[-1].properties)
    assert layer_properties["Путь"] == str(image_directory)
    assert layer_properties["Количество файлов"] == 1
    assert layer_properties["Кто добавил"] == "Оператор"
    assert len(snapshots[-1].history) == 2

    node = PipelineNode(
        "representation-1",
        "Исходные изображения",
        "source",
        representation_id="representation-1",
    )
    controller._show_node_properties("layer-1", node)
    representation_properties = dict(snapshots[-1].properties)
    assert representation_properties["Примечание"] == "Проверено"
    assert representation_properties["Количество файлов"] == 1
    assert representation_properties["Ревизия"] == 1
    assert [item.event_type for item in snapshots[-1].history] == [
        "RepresentationCreated"
    ]
