from __future__ import annotations

from pathlib import Path

from PIL import Image

from kraken_core.plugin_protocol import WorkspacePluginResultV1
from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.manager_app import DesktopController
from kraken_manager.workspace import (
    DerivedRunKind,
    ImageConversionSettings,
    LayerSourceMode,
)
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind


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
