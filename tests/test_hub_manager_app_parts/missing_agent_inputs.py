"""test_hub_manager_app: test missing agent inputs."""
from __future__ import annotations
from time import monotonic, sleep
from types import SimpleNamespace
import pytest
from PyQt6.QtWidgets import (
    QMainWindow,
)
from kraken_hub.presentation.qt.manager_app import (
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


def test_partial_binary_result_is_ready_without_directory_import(qapp) -> None:
    from kraken_manager.domain.project import RepresentationPurpose

    project = SimpleNamespace(id="project")
    layer = SimpleNamespace(id="layer")
    binary = SimpleNamespace(id="binary", purpose=RepresentationPurpose.BINARY, source=None)
    controller = object.__new__(DesktopController)
    controller._project_id = project.id
    controller.shell = QMainWindow()
    controller.session = SimpleNamespace(principal=object())
    controller.service = SimpleNamespace(
        get_project=lambda _: project,
        frame_cells=lambda *_: (SimpleNamespace(x=1, y=1),),
        layer_file_binding=lambda *_: SimpleNamespace(frame_positions=((1, 1), (1, 2))),
        list_layers=lambda *_: (layer,),
        list_representations=lambda *_: (binary,),
        materialize_representation_inputs=lambda **_: pytest.fail("Binary result was imported as a source directory"),
    )
    controller._error = pytest.fail

    assert controller._ensure_agent_source_ready(
        layer_id=layer.id, source_representation_id=binary.id, retry=lambda: None,
    ) is True


def test_unpublished_binary_result_reports_missing_frames(qapp) -> None:
    from kraken_manager.domain.project import RepresentationPurpose

    project = SimpleNamespace(id="project")
    layer = SimpleNamespace(id="layer")
    binary = SimpleNamespace(id="binary", purpose=RepresentationPurpose.BINARY, source=None)
    errors = []
    controller = object.__new__(DesktopController)
    controller._project_id = project.id
    controller.shell = QMainWindow()
    controller.session = SimpleNamespace(principal=object())
    controller.service = SimpleNamespace(
        get_project=lambda _: project,
        frame_cells=lambda *_: (),
        list_layers=lambda *_: (layer,),
        list_representations=lambda *_: (binary,),
        materialize_representation_inputs=lambda **_: pytest.fail("Binary result was imported as a source directory"),
    )
    controller._error = errors.append

    assert controller._ensure_agent_source_ready(
        layer_id=layer.id, source_representation_id=binary.id, retry=lambda: None,
    ) is False
    assert errors == ["Бинарные изображения ещё не опубликованы. Дождитесь завершения бинаризации."]
