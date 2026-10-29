"""test_hub_manager_app: test remote project can."""
from __future__ import annotations
from types import SimpleNamespace
from PyQt6.QtWidgets import (
    QInputDialog,
    QWidget,
)
from kraken_hub.presentation.qt.manager_app import (
    DesktopController,
)
from kraken_manager.domain.project import LayerType, RepresentationKind
from kraken_manager.presentation.qt import (
    LayerPipelineSnapshot,
    PipelineLane,
    PipelineNode,
)

def test_remote_project_can_create_logical_layer(qapp, monkeypatch) -> None:
    project = SimpleNamespace(id="project-1")
    created = SimpleNamespace(id="layer-1")
    calls = []

    class ServiceStub:
        @staticmethod
        def get_project(_project_id):
            return project

        @staticmethod
        def project_workspace(_project_id):
            return None

        @staticmethod
        def is_remote_project(_project_id):
            return True

        @staticmethod
        def list_layers(_project_id):
            return ()

        @staticmethod
        def create_layer(**kwargs):
            calls.append(kwargs)
            return created

    controller = object.__new__(DesktopController)
    controller.service = ServiceStub()
    controller.session = SimpleNamespace(principal=object())
    controller.shell = QWidget()
    shown = []
    controller._show_created_layer = lambda *args: shown.append(args)
    workspace = QWidget()

    monkeypatch.setattr(QInputDialog, "getText", lambda *_args, **_kwargs: ("Metal", True))
    monkeypatch.setattr(
        QInputDialog,
        "getItem",
        lambda *_args, **_kwargs: (_args[3][0], True),
    )

    controller._add_layer(workspace, project.id)

    assert calls[0]["layer_type"] is LayerType.METAL
    assert calls[0]["order"] == 0
    assert shown == [(workspace, project.id, created)]

def test_external_cif_import_uses_source_from_clicked_pipeline_lane(monkeypatch) -> None:
    controller = object.__new__(DesktopController)
    controller._workspace = object()
    controller._project_id = "project-1"
    missing = PipelineNode("missing-cif", "CIF не получен", "missing")
    controller._pipeline_snapshot = lambda *_args: LayerPipelineSnapshot(
        "project-1",
        "layer-1",
        (
            PipelineLane(
                "source-representation-2",
                "Source 2",
                (
                    PipelineNode("source-representation-2", "Source 2", "source"),
                    missing,
                ),
                (("source-representation-2", "missing-cif"),),
            ),
        ),
    )
    calls = []
    monkeypatch.setattr(
        controller,
        "_add_representation",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    controller._layer_manager_action("layer-1", missing, "add_external_vector")

    assert calls
    assert calls[0][0][2] is RepresentationKind.VECTOR
    assert calls[0][1]["source_image_id"] == "source-representation-2"
