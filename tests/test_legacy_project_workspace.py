from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image
from PyQt6.QtWidgets import QDialog, QMessageBox

from kraken_hub.presentation.qt import manager_app
from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.composition.catalog import DualCatalogService
from kraken_hub.presentation.qt.manager_app import DesktopController
from kraken_manager.application.dto import CommandContext, CreateProjectCommand
from kraken_manager.application.errors import AuthorizationError
from kraken_manager.application.use_cases import CreateProjectHandler
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind
from kraken_manager.presentation.qt import ProjectListItem, ProjectManagerShell
from kraken_manager.workspace import ImageConversionSettings


@pytest.fixture
def legacy_project(tmp_path: Path):
    service = EmbeddedProjectService(tmp_path / "data")
    session = service.create_initial_account("owner", "Owner", "")
    command = CreateProjectCommand(
        context=CommandContext(actor=session.principal, idempotency_key="project"),
        name="Old project",
        width=2,
        height=2,
        orientation=GridOrientation.Y_DOWN,
        storage_profile_id=service.profile.id,
    )
    project = CreateProjectHandler(service._uow(str(command.project_id)), service.profiles, service.clock)(command)
    assert service.project_workspace(project.id) is None
    return service, session, project


def _bind(service, session, project):
    return service.bind_project_workspace(
        principal=session.principal,
        project=project,
        source_root=service.default_source_root,
        derived_root=service.default_derived_root,
        idempotency_key="workspace",
    )


def test_bind_legacy_project_preserves_data_and_enables_disk_import(legacy_project, tmp_path):
    service, session, project = legacy_project
    images = tmp_path / "old-images"
    images.mkdir()
    image = images / "frame_0.jpg"
    Image.new("RGB", (2, 2), "red").save(image)
    original_bytes = image.read_bytes()
    layer = service.create_layer(
        principal=session.principal,
        project=project,
        name="Old layer",
        layer_type=LayerType.METAL,
        order=1,
        idempotency_key="layer",
    )
    representation = service.create_representation(
        principal=session.principal,
        project=project,
        layer=layer,
        name="Old images",
        kind=RepresentationKind.IMAGE,
        source=str(images),
        active=True,
        idempotency_key="representation",
    )
    before = service.history(project.id)
    current = service.get_project(project.id)
    layers = service.list_layers(project.id)
    representations = service.list_representations(project.id, layer.id)

    binding = _bind(DualCatalogService(service), session, project)

    assert service.get_project(project.id) == current
    assert service.list_layers(project.id) == layers
    assert service.list_representations(project.id, layer.id) == representations
    viewport = service.matrix_viewport(
        project.id,
        layer_id=layer.id,
        representation_ids=(representation.id,),
        x1=1,
        y1=1,
        x2=2,
        y2=2,
    )
    assert viewport["cells"][0]["asset_path"] == str(image)
    assert service.history(project.id)[:-1] == before
    assert service.history(project.id)[-1].event_type == "ProjectWorkspaceBoundV1"
    assert image.read_bytes() == original_bytes
    assert Path(binding.source_project_dir).is_dir()
    assert Path(binding.derived_project_dir).is_dir()
    restarted = EmbeddedProjectService(service.data_dir)
    assert restarted.project_workspace(project.id) == binding
    assert _bind(service, session, project) == binding
    assert service.history(project.id)[:-1] == before

    imported, files, _representation = service.create_layer_from_disk(
        principal=session.principal,
        project=current,
        name="New layer",
        layer_type=LayerType.CONTACT,
        order=2,
        idempotency_key="import",
        scan=service.scan_layer_source(images, maximum_frames=project.frame_count),
        conversion=ImageConversionSettings(),
    )
    assert imported.name == "New layer"
    assert (Path(files.image_directory) / image.name).is_file()
    assert image.read_bytes() == original_bytes


def test_bind_legacy_project_rolls_back_on_history_failure(legacy_project, tmp_path, monkeypatch):
    service, session, project = legacy_project
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (2, 2)).save(images / "frame_0.png")
    layer, files, _representation = service.create_external_layer(
        principal=session.principal,
        project=project,
        name="External",
        layer_type=LayerType.METAL,
        order=1,
        image_directory=images,
        ssc_directory=None,
        prv_directory=None,
        idempotency_key="external",
    )
    before = service.history(project.id)

    def fail_history(**_kwargs):
        raise OSError("history unavailable")

    monkeypatch.setattr(service, "_record_workspace_event", fail_history)
    with pytest.raises(OSError, match="history unavailable"):
        _bind(service, session, project)

    assert service.project_workspace(project.id) is None
    assert service.layer_file_binding(project.id, layer.id) == files
    assert service.history(project.id) == before
    assert (images / "frame_0.png").is_file()
    assert not (service.default_source_root / project.name).exists()
    assert not (service.default_derived_root / project.name).exists()


def test_bind_legacy_project_refuses_existing_folders(legacy_project):
    service, session, project = legacy_project
    occupied = service.default_source_root / project.name
    occupied.mkdir()
    marker = occupied / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _bind(service, session, project)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert service.project_workspace(project.id) is None
    assert not (service.default_derived_root / project.name).exists()


def test_bind_legacy_project_requires_permission(legacy_project):
    service, session, project = legacy_project
    stranger = Principal.local(subject="other", display_name="Other")
    with pytest.raises(AuthorizationError):
        _bind(service, replace(session, principal=stranger), project)
    assert service.project_workspace(project.id) is None
    assert not (service.default_source_root / project.name).exists()


@pytest.mark.parametrize("configure", [False, None, True])
def test_desktop_opens_legacy_project_with_optional_setup(
    legacy_project,
    qapp,
    monkeypatch,
    configure,
):
    service, session, project = legacy_project
    shell = ProjectManagerShell()
    controller = object.__new__(DesktopController)
    controller.shell = shell
    controller.service = service
    controller.session = session
    controller.thumbnail_store_uri = "memory://"
    controller._project_id = None
    controller._workspace = None
    controller._layer_dialog = None
    controller._error = pytest.fail
    controller._configure_analysis = lambda *_args: None
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args: QMessageBox.StandardButton.No if configure is False else QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        manager_app,
        "_configure_workspace_roots",
        lambda *_args, **_kwargs: (
            None if configure is None else (str(service.default_source_root), str(service.default_derived_root))
        ),
    )

    controller.open_project(ProjectListItem(str(project.id), project.name, 2, 2, "Local"))

    assert controller._project_id == str(project.id)
    assert controller._workspace is not None
    assert (service.project_workspace(project.id) is not None) is (configure is True)
    controller._workspace.matrix_view.close()
    shell.close()


def test_add_layer_configures_workspace_before_showing_import_dialog(legacy_project, qapp, monkeypatch):
    service, session, project = legacy_project
    controller = object.__new__(DesktopController)
    controller.shell = ProjectManagerShell()
    controller.service = service
    controller.session = session
    controller._error = pytest.fail
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(
        manager_app,
        "_configure_workspace_roots",
        lambda *_args, **_kwargs: (str(service.default_source_root), str(service.default_derived_root)),
    )
    shown = []

    def reject_import(_dialog):
        shown.append(service.project_workspace(project.id))
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(manager_app.LayerCreationDialog, "exec", reject_import)
    controller._add_layer(object(), project.id)
    assert len(shown) == 1
    assert shown[0] is not None
    controller.shell.close()
