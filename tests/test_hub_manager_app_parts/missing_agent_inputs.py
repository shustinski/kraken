"""test_hub_manager_app: test missing agent inputs."""
from __future__ import annotations
from time import monotonic, sleep
from types import SimpleNamespace
from PyQt6.QtWidgets import (
    QMainWindow,
)
from kraken_hub.manager_app import (
    DesktopController,
)

def test_missing_agent_inputs_are_prepared_in_background_and_retried(qapp) -> None:
    project = SimpleNamespace(id="project-1")
    layer = SimpleNamespace(id="layer-1")
    representation = SimpleNamespace(id="representation-1")
    prepared_cells = []

    class ServiceStub:
        @staticmethod
        def get_project(_project_id):
            return project

        @staticmethod
        def frame_cells(*_args):
            return tuple(prepared_cells)

        @staticmethod
        def list_layers(_project_id):
            return (layer,)

        @staticmethod
        def list_representations(_project_id, _layer_id):
            return (representation,)

        @staticmethod
        def materialize_representation_inputs(**_kwargs):
            prepared_cells.append(SimpleNamespace(x=1, y=1))
            return tuple(prepared_cells)

    shell = QMainWindow()
    controller = object.__new__(DesktopController)
    controller.shell = shell
    controller.service = ServiceStub()
    controller.session = SimpleNamespace(principal=object())
    controller._project_id = project.id
    controller._background_workers = set()
    controller._source_preparation_keys = set()
    retried = []

    ready = controller._ensure_agent_source_ready(
        layer_id=layer.id,
        source_representation_id=representation.id,
        retry=lambda: retried.append(True),
    )

    assert ready is False
    deadline = monotonic() + 3
    while not retried and monotonic() < deadline:
        qapp.processEvents()
        sleep(0.01)

    assert retried == [True]
    assert prepared_cells
    assert not controller._source_preparation_keys
