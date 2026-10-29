"""test_hub_manager_app: test contour vectorize receives."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from PyQt6.QtWidgets import (
    QPushButton,
)
from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.presentation.qt.manager_app import (
    DesktopController,
    _configure_administration_page,
)
from kraken_manager.domain.project import RepresentationKind
from kraken_manager.presentation.qt import (
    PipelineNode,
    ProjectManagerShell,
)

def test_contour_vectorize_receives_staged_base_layer_path(tmp_path: Path) -> None:
    representation = SimpleNamespace(
        id="binary-representation-1",
        kind=RepresentationKind.IMAGE,
        source="managed-import",
    )

    class ServiceStub:
        data_dir = tmp_path

        @staticmethod
        def list_representations(_project_id, _layer_id):
            return (representation,)

        @staticmethod
        def frame_cells(_project_id, _layer_id, _representation_id):
            return (
                SimpleNamespace(x=3, y=4, sha256="a" * 64),
            )

        @staticmethod
        def read_project_blob(_project_id, _sha256):
            return b"image"

    controller = object.__new__(DesktopController)
    controller.service = ServiceStub()
    controller._project_id = "project-1"
    node = PipelineNode(
        "binary-representation-1",
        "Binary",
        "binary",
        representation_id="binary-representation-1",
    )

    arguments, parameters = controller._contour_launch_arguments(
        layer_id="layer-1",
        node=node,
        action="vectorize",
        source_representation_id="source-representation-1",
    )

    input_directory = Path(arguments[arguments.index("--input-dir") + 1])
    output_directory = Path(arguments[arguments.index("--output-dir") + 1])
    assert input_directory.is_dir()
    assert (input_directory / "3_4.png").read_bytes() == b"image"
    assert output_directory.is_dir()
    assert parameters["input_representation_id"] == "binary-representation-1"

def test_local_administration_page_is_wired_for_desktop_session(qapp, tmp_path) -> None:
    service = EmbeddedProjectService(tmp_path)
    session = service.create_initial_account(
        username="admin",
        display_name="Administrator",
        password="",
    )
    shell = ProjectManagerShell()

    _configure_administration_page(shell, service, session)

    administration_item = shell._navigation_items["administration"]
    assert not administration_item.isHidden()
    page = shell.page("administration")
    assert page is not None
    labels = [button.text() for button in page.findChildren(QPushButton)]
    assert "Проверить целостность" in labels
    assert "Создать резервную копию" in labels
    assert "Восстановить резервную копию" in labels
    assert "Подключить проект из локального каталога…" in labels

def test_layer_manager_reports_history_failure_without_opening_dialog() -> None:
    from kraken_hub.infrastructure.remote.client import RemoteServerError

    controller = object.__new__(DesktopController)
    controller._workspace = SimpleNamespace(layer_model=SimpleNamespace(items=lambda: ()), _selected_layer_id="layer")
    controller._project_id = "project"
    controller.shell = SimpleNamespace(current_page_key=lambda: "workspace")
    shown = []
    pipelines = []
    controller._layer_dialog = SimpleNamespace(
        set_layers=lambda *_args: None,
        set_pipeline=pipelines.append,
        show=lambda: shown.append(True),
    )
    errors = []
    controller._error = errors.append

    def fail(*_args):
        raise RemoteServerError("History unavailable", status=500)

    controller._pipeline_snapshot = fail
    controller.open_layer_manager()
    assert len(errors) == 1
    assert "History unavailable" in errors[0]
    assert shown == []
    assert pipelines == []

    snapshot = SimpleNamespace(layer_id="layer")
    controller._pipeline_snapshot = lambda *_args: snapshot
    assert controller._refresh_layer_pipeline("project", "layer")
    assert pipelines == [snapshot]
