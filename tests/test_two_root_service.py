from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from kraken_core.plugin_protocol import WorkspacePluginResultV1
from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.manager_app import DesktopController
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind
from kraken_manager.workspace import (
    DerivedRunKind,
    ImageConversionSettings,
    LayerSourceMode,
    WorkspaceValidationError,
)


def _project(tmp_path: Path):
    service = EmbeddedProjectService(tmp_path / "data")
    session = service.create_initial_account("owner", "Owner", "")
    project = service.create_project(
        principal=session.principal,
        name="Chip",
        width=2,
        height=2,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key="project",
    )
    return service, session, project


def test_external_layer_uses_numbered_positions_and_delete_keeps_source(
    tmp_path: Path,
) -> None:
    images = tmp_path / "external"
    images.mkdir()
    Image.new("RGB", (2, 2), "red").save(images / "frame_0.png")
    Image.new("RGB", (2, 2), "blue").save(images / "frame_2.png")
    service, session, project = _project(tmp_path)

    layer, binding, representation = service.create_external_layer(
        principal=session.principal,
        project=project,
        name="Metal",
        layer_type=LayerType.METAL,
        order=1,
        image_directory=images,
        ssc_directory=None,
        prv_directory=None,
        idempotency_key="external-layer",
    )

    assert binding.mode is LayerSourceMode.EXTERNAL
    assert representation.kind is RepresentationKind.IMAGE
    assert representation.source == binding.image_directory
    assert representation.active
    assert representation.name == "Исходные изображения"
    assert service.list_representations(project.id, layer.id) == (representation,)
    assert service.frame_cells(project.id, layer.id, representation.id) == ()
    prepared = service.materialize_representation_inputs(
        principal=session.principal,
        project=project,
        layer=layer,
        representation=representation,
    )
    assert {(cell.x, cell.y) for cell in prepared} == {(1, 1), (1, 2)}
    controller = object.__new__(DesktopController)
    controller.service = service
    snapshot = controller._pipeline_snapshot(project.id, layer.id)
    source_node = snapshot.lanes[0].nodes[0]
    assert source_node.kind == "source"
    assert binding.image_directory in source_node.details.values()
    viewport = service.matrix_viewport(
        project.id,
        layer_id=layer.id,
        representation_ids=(representation.id,),
        x1=1,
        y1=1,
        x2=2,
        y2=2,
    )
    assert {(cell["x"], cell["y"]) for cell in viewport["cells"]} == {(1, 1), (1, 2)}
    assert {event.event_type for event in service.history(project.id)} >= {
        "ProjectWorkspaceBoundV1",
        "LayerFileBoundV1",
    }

    latest_project = service.get_project(project.id)
    assert latest_project is not None
    service.delete_layer(
        principal=session.principal,
        project=latest_project,
        layer=layer,
        confirmation_name="Metal",
        idempotency_key="delete-layer",
    )

    assert (images / "frame_0.png").is_file()
    assert service.layer_file_binding(project.id, layer.id) is None
    assert "LayerDeletedV1" in {
        event.event_type for event in service.history(project.id)
    }


def test_disk_import_registers_copied_directory_as_image_representation(
    tmp_path: Path,
) -> None:
    microscope = tmp_path / "microscope"
    microscope.mkdir()
    Image.new("RGB", (2, 2), "red").save(microscope / "frame_0.jpg")
    service, session, project = _project(tmp_path)

    layer, binding, representation = service.create_layer_from_disk(
        principal=session.principal,
        project=project,
        name="Metal",
        layer_type=LayerType.METAL,
        order=1,
        scan=service.scan_layer_source(
            microscope,
            maximum_frames=project.frame_count,
        ),
        conversion=ImageConversionSettings(),
        idempotency_key="managed-layer",
    )

    assert binding.mode is LayerSourceMode.MANAGED_COPY
    assert Path(binding.image_directory).is_dir()
    assert representation.kind is RepresentationKind.IMAGE
    assert representation.source == binding.image_directory
    assert representation.active
    assert representation.name == "Исходные изображения"
    assert service.list_representations(project.id, layer.id) == (representation,)
    cells = service.frame_cells(project.id, layer.id, representation.id)
    assert {(cell.x, cell.y) for cell in cells} == {(1, 1)}
    assert service.managed_artifact_path(
        project.id,
        cells[0].artifact_version_id,
    ).is_file()


def test_project_rename_rolls_back_both_workspace_roots_on_domain_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, session, project = _project(tmp_path)
    original = service.project_workspace(project.id)
    assert original is not None
    source_before = Path(original.source_project_dir)
    derived_before = Path(original.derived_project_dir)
    assert source_before.is_dir()
    assert derived_before.is_dir()

    class FailingRenameHandler:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __call__(self, _command):
            raise RuntimeError("forced domain failure")

    monkeypatch.setattr(
        "kraken_hub.composition.RenameProjectHandler",
        FailingRenameHandler,
    )

    try:
        service.rename_project(
            principal=session.principal,
            project=project,
            name="Renamed",
            idempotency_key="rename",
        )
    except RuntimeError as exc:
        assert str(exc) == "forced domain failure"
    else:
        raise AssertionError("rename_project must propagate the domain failure")

    restored = service.project_workspace(project.id)
    assert restored == original
    assert source_before.is_dir()
    assert derived_before.is_dir()
    assert not (source_before.parent / "Renamed").exists()
    assert not (derived_before.parent / "Renamed").exists()
    assert service.get_project(project.id).name == "Chip"


def test_managed_layer_rename_rolls_back_files_and_registry_on_domain_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    microscope = tmp_path / "microscope"
    microscope.mkdir()
    Image.new("RGB", (2, 2), "red").save(microscope / "frame_0.jpg")
    service, session, project = _project(tmp_path)
    layer, binding, _representation = service.create_layer_from_disk(
        principal=session.principal,
        project=project,
        name="Metal",
        layer_type=LayerType.METAL,
        order=1,
        scan=service.scan_layer_source(
            microscope,
            maximum_frames=project.frame_count,
        ),
        conversion=ImageConversionSettings(),
        idempotency_key="managed-layer",
    )

    class FailingRenameHandler:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __call__(self, _command):
            raise RuntimeError("forced layer failure")

    monkeypatch.setattr(
        "kraken_hub.composition.RenameLayerHandler",
        FailingRenameHandler,
    )
    try:
        service.rename_layer(
            principal=session.principal,
            project=service.get_project(project.id),
            layer=layer,
            name="Metal renamed",
            idempotency_key="rename-layer",
        )
    except RuntimeError as exc:
        assert str(exc) == "forced layer failure"
    else:
        raise AssertionError("rename_layer must propagate the domain failure")

    restored = service.layer_file_binding(project.id, layer.id)
    assert restored == binding
    for value in (
        binding.image_directory,
        binding.ssc_directory,
        binding.prv_directory,
        binding.aux_directory,
    ):
        assert Path(value).is_dir()
        assert not (Path(value).parent / "Metal renamed").exists()
    assert service.list_layers(project.id)[0].name == "Metal"


def test_managed_layer_mutations_are_blocked_by_active_agent_job(tmp_path: Path) -> None:
    microscope = tmp_path / "microscope"
    microscope.mkdir()
    Image.new("RGB", (2, 2), "red").save(microscope / "frame_0.jpg")
    service, session, project = _project(tmp_path)
    layer, binding, _representation = service.create_layer_from_disk(
        principal=session.principal,
        project=project,
        name="Metal",
        layer_type=LayerType.METAL,
        order=1,
        scan=service.scan_layer_source(
            microscope,
            maximum_frames=project.frame_count,
        ),
        conversion=ImageConversionSettings(),
        idempotency_key="managed-layer",
    )
    service.plugin_jobs = lambda: (
        SimpleNamespace(
            project_id=project.id,
            layer_id=layer.id,
            state=SimpleNamespace(value="running"),
        ),
    )

    try:
        service.rename_layer(
            principal=session.principal,
            project=service.get_project(project.id),
            layer=layer,
            name="Metal renamed",
            idempotency_key="rename-layer",
        )
    except WorkspaceValidationError as exc:
        assert "выполняются задания" in str(exc)
    else:
        raise AssertionError("active Agent jobs must block managed layer renames")

    assert service.layer_file_binding(project.id, layer.id) == binding
    assert Path(binding.image_directory).is_dir()
    assert not (Path(binding.image_directory).parent / "Metal renamed").exists()

    try:
        service.delete_layer(
            principal=session.principal,
            project=service.get_project(project.id),
            layer=layer,
            confirmation_name=layer.name,
            idempotency_key="delete-layer",
        )
    except WorkspaceValidationError as exc:
        assert "active plugin job" in str(exc)
    else:
        raise AssertionError("active Agent jobs must block managed layer deletion")

    assert service.layer_file_binding(project.id, layer.id) == binding
    assert Path(binding.image_directory).is_dir()


def test_vector_run_publication_creates_versioned_representation(
    tmp_path: Path,
) -> None:
    images = tmp_path / "external"
    images.mkdir()
    Image.new("RGB", (2, 2), "red").save(images / "frame_0.png")
    service, session, project = _project(tmp_path)
    layer, _binding, _representation = service.create_external_layer(
        principal=session.principal,
        project=project,
        name="Metal",
        layer_type=LayerType.METAL,
        order=1,
        image_directory=images,
        ssc_directory=None,
        prv_directory=None,
        idempotency_key="external-layer",
    )
    run = service.begin_derived_run(
        project_id=project.id,
        layer_id=layer.id,
        layer_name=layer.name,
        kind=DerivedRunKind.VECTOR,
        plugin_id="contour",
        operation="frames.vectorize.v1",
        principal=session.principal,
    )
    external_result = tmp_path / "contour-output"
    external_result.mkdir()
    (external_result / "frame_1.cif").write_text("E\n", encoding="utf-8")
    result = WorkspacePluginResultV1(
        run_id=run.run_id,
        plugin_id="contour",
        operation=run.operation,
        outcome="succeeded",
        output_directory=str(external_result.resolve()),
        provenance={"plugin_version": "test"},
    )
    current_project = service.get_project(project.id)
    assert current_project is not None

    published, vector = service.publish_workspace_plugin_result(
        principal=session.principal,
        project=current_project,
        layer=layer,
        result=result,
    )

    assert published.path == run.path
    assert (Path(run.path) / "frame_1.cif").is_file()
    assert vector is not None
    assert vector.kind is RepresentationKind.VECTOR
    assert vector.active
    assert "DerivedRunPublishedV1" in {
        event.event_type for event in service.history(project.id)
    }


def test_binary_run_uses_frame_number_suffix_and_preserves_gaps(
    tmp_path: Path,
) -> None:
    images = tmp_path / "external"
    images.mkdir()
    Image.new("RGB", (2, 2), "red").save(images / "frame_0.png")
    Image.new("RGB", (2, 2), "blue").save(images / "frame_2.png")
    service, session, project = _project(tmp_path)
    layer, _binding, _source = service.create_external_layer(
        principal=session.principal,
        project=project,
        name="Metal",
        layer_type=LayerType.METAL,
        order=1,
        image_directory=images,
        ssc_directory=None,
        prv_directory=None,
        idempotency_key="external-layer",
    )
    run = service.begin_derived_run(
        project_id=project.id,
        layer_id=layer.id,
        layer_name=layer.name,
        kind=DerivedRunKind.RESULT,
        plugin_id="neuralimage",
        operation="frames.binary-segment.v2",
        principal=session.principal,
    )
    outputs = tmp_path / "binary-output"
    outputs.mkdir()
    Image.new("L", (2, 2), 0).save(outputs / "000001_1_1_0.png")
    Image.new("L", (2, 2), 255).save(outputs / "000002_1_2_2.png")
    current_project = service.get_project(project.id)
    assert current_project is not None
    _published, binary = service.publish_workspace_plugin_result(
        principal=session.principal,
        project=current_project,
        layer=layer,
        result=WorkspacePluginResultV1(
            run_id=run.run_id,
            plugin_id="neuralimage",
            operation=run.operation,
            outcome="succeeded",
            output_directory=str(outputs.resolve()),
        ),
    )
    assert binary is not None

    viewport = service.matrix_viewport(
        project.id,
        layer_id=layer.id,
        representation_ids=(binary.id,),
        x1=1,
        y1=1,
        x2=2,
        y2=2,
    )
    assert {(cell["x"], cell["y"]) for cell in viewport["cells"]} == {(1, 1), (1, 2)}
