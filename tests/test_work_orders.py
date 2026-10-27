"""Issued work packages for vectorization, training, stitching and vector correction."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

import pytest

from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.infrastructure.work_package import WorkIssueRequest, place_work_order
from kraken_manager.domain.common import LayerId, PerformerId, PrincipalId, ProjectId
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind
from kraken_manager.domain.work_orders import WorkKind, WorkOrder, WorkPlacement
from tests.test_hub_embedded_service_parts.memory_secret_store import MemorySecretStoreMixin


def _service(tmp_path: Path) -> EmbeddedProjectService:
    return EmbeddedProjectService(tmp_path / "data", secret_store=MemorySecretStoreMixin._memory_secret_store())


def _project_with_frames(service: EmbeddedProjectService, root: Path, *, with_vectors: bool = True):
    owner = service.create_initial_account("owner", "Owner", "")
    performer = service.create_manual_performer(name="Исполнитель", color="#336699")
    project = service.create_project(
        principal=owner.principal,
        name="new_test",
        width=2,
        height=1,
        orientation=GridOrientation.Y_DOWN,
        idempotency_key="project",
    )
    layer = service.create_layer(
        principal=owner.principal,
        project=project,
        name="M1",
        layer_type=LayerType.METAL,
        order=0,
        idempotency_key="layer",
    )
    images = service.create_representation(
        principal=owner.principal,
        project=service.get_project(project.id),
        layer=layer,
        name="Images",
        kind=RepresentationKind.IMAGE,
        active=True,
        idempotency_key="images",
    )
    layer = service.list_layers(project.id)[0]
    vectors = None
    if with_vectors:
        vectors = service.create_representation(
            principal=owner.principal,
            project=service.get_project(project.id),
            layer=layer,
            name="CIF",
            kind=RepresentationKind.VECTOR,
            source_image_representation_id=images.id,
            active=True,
            idempotency_key="vectors",
        )
    root.mkdir(parents=True, exist_ok=True)
    image_dir = root / "images"
    image_dir.mkdir()
    (image_dir / "1_1.png").write_bytes(b"image-a")
    (image_dir / "2_1.png").write_bytes(b"image-b")
    service.commit_managed_import(
        principal=owner.principal,
        project=service.get_project(project.id),
        layer=layer,
        representation=images,
        plan=service.plan_import_directory(project=project, directory=image_dir),
        idempotency_key="import-images",
    )
    if vectors is not None:
        vector_dir = root / "vectors"
        vector_dir.mkdir()
        (vector_dir / "1_1.cif").write_text("DS 1 1 1;\nDF;\nE\n", encoding="utf-8")
        (vector_dir / "2_1.cif").write_text("DS 1 1 1;\nDF;\nE\n", encoding="utf-8")
        service.commit_managed_import(
            principal=owner.principal,
            project=service.get_project(project.id),
            layer=service.list_layers(project.id)[0],
            representation=vectors,
            plan=service.plan_import_directory(project=project, directory=vector_dir),
            idempotency_key="import-vectors",
        )
    return owner, performer, service.get_project(project.id), service.list_layers(project.id)[0], images, vectors


def test_vectorize_external_package_keeps_distinct_frame_coordinates(tmp_path: Path) -> None:
    service = _service(tmp_path)
    owner, performer, project, layer, images, _vectors = _project_with_frames(service, tmp_path / "source")
    destination = tmp_path / "usb"
    destination.mkdir()
    order = service.issue_work(
        principal=owner.principal,
        project_id=project.id,
        layer_id=layer.id,
        kind="vectorize",
        placement="external",
        assignee_id=performer.id,
        coordinates=((1, 1), (2, 1)),
        image_representation_id=str(images.id),
        instructions="Векторизовать ряд",
        destination_parent=destination,
        idempotency_key="vectorize-1",
    )
    package = Path(order.destination)
    assert (package / "images" / "1_1.png").read_bytes() == b"image-a"
    assert (package / "images" / "2_1.png").read_bytes() == b"image-b"
    manifest = json.loads((package / "job.json").read_text(encoding="utf-8"))
    coordinates = {(item["x"], item["y"]) for item in manifest["inputs"]}
    assert coordinates == {(1, 1), (2, 1)}
    assert len({item["relative_path"] for item in manifest["inputs"]}) == 2
    from contour.kraken_bridge import ContourKrakenSession

    session = ContourKrakenSession.load(
        job_manifest=package / "job.json",
        result_manifest=package / "result.json",
        staging_root=package,
    )
    assert [item.x for item in session.manifest.inputs] == [1, 2]
    assert all(item.y == 1 for item in session.manifest.inputs)
    repeated = service.issue_work(
        principal=owner.principal,
        project_id=project.id,
        layer_id=layer.id,
        kind="vectorize",
        placement="external",
        assignee_id=performer.id,
        coordinates=((1, 1), (2, 1)),
        image_representation_id=str(images.id),
        destination_parent=destination,
        idempotency_key="vectorize-1",
    )
    assert repeated.id == order.id


def test_server_http_package_does_not_copy_images_and_file_edges_fail(tmp_path: Path) -> None:
    service = _service(tmp_path)
    owner, performer, project, layer, images, vectors = _project_with_frames(service, tmp_path / "source")
    order = service.issue_work(
        principal=owner.principal,
        project_id=project.id,
        layer_id=layer.id,
        kind="vectorize",
        placement="server",
        assignee_id=performer.id,
        coordinates=((1, 1),),
        image_representation_id=str(images.id),
        idempotency_key="http-vectorize",
        transport="http",
    )
    package = Path(order.destination)
    assert order.transport == "http"
    assert order.launch_mode == "plugin-http"
    assert not (package / "images").exists()
    chosen = tmp_path / "chosen-server-folder"
    chosen.mkdir()
    automatic = service.issue_work(
        principal=owner.principal,
        project_id=project.id,
        layer_id=layer.id,
        kind="vectorize",
        placement="server",
        assignee_id=performer.id,
        coordinates=((1, 1),),
        image_representation_id=str(images.id),
        destination_parent=chosen,
        idempotency_key="automatic-server",
    )
    assert automatic.transport == "http"
    assert list(chosen.iterdir()) == []
    assert chosen not in Path(automatic.destination).parents
    description = json.loads((package / "kraken-work.json").read_text(encoding="utf-8"))
    assert description["frames"] == [{"x": 1, "y": 1, "path": "images/1_1.png", "frame_id": description["frames"][0]["frame_id"]}]
    with pytest.raises(ValueError, match="HTTP"):
        service.complete_work_order(order, idempotency_key="complete-http")
    with pytest.raises(ValueError, match="каталог"):
        service.issue_work(
            principal=owner.principal,
            project_id=project.id,
            layer_id=layer.id,
            kind="vectorize",
            placement="external",
            assignee_id=performer.id,
            coordinates=((1, 1),),
            image_representation_id=str(images.id),
            destination_parent=tmp_path / "missing-usb",
            idempotency_key="missing-usb",
        )
    file_parent = tmp_path / "not-a-directory"
    file_parent.write_text("usb", encoding="utf-8")
    with pytest.raises(ValueError, match="каталог"):
        service.issue_work(
            principal=owner.principal,
            project_id=project.id,
            layer_id=layer.id,
            kind="vectorize",
            placement="external",
            assignee_id=performer.id,
            coordinates=((1, 1),),
            image_representation_id=str(images.id),
            destination_parent=file_parent,
            idempotency_key="file-usb",
        )
    with pytest.raises(ValueError):
        service.issue_work(
            principal=owner.principal,
            project_id=project.id,
            layer_id=layer.id,
            kind="vectorize",
            placement="server",
            assignee_id=performer.id,
            coordinates=((1, 2),),
            image_representation_id=str(images.id),
            idempotency_key="outside",
        )
    archived = service.archive_performer(performer.id)
    assert archived.active is False
    with pytest.raises(ValueError, match="архив"):
        service.issue_work(
            principal=owner.principal,
            project_id=project.id,
            layer_id=layer.id,
            kind="vectorize",
            placement="server",
            assignee_id=performer.id,
            coordinates=((1, 1),),
            image_representation_id=str(images.id),
            idempotency_key="archived",
        )
    assert vectors is not None


def test_stitch_completion_requires_a_result_file_and_cancel_is_terminal(tmp_path: Path) -> None:
    service = _service(tmp_path)
    owner, performer, project, layer, images, _vectors = _project_with_frames(service, tmp_path / "source", with_vectors=False)
    destination = tmp_path / "usb"
    destination.mkdir()
    order = service.issue_work(
        principal=owner.principal,
        project_id=project.id,
        layer_id=layer.id,
        kind="stitch",
        placement="external",
        assignee_id=performer.id,
        coordinates=((2, 1),),
        image_representation_id=str(images.id),
        destination_parent=destination,
        idempotency_key="stitch",
    )
    assert (Path(order.destination) / "sources" / "2_1.png").read_bytes() == b"image-b"
    with pytest.raises(ValueError, match="результата"):
        service.complete_work_order(order, idempotency_key="early")
    (Path(order.destination) / "seam.bin").write_bytes(b"seam")
    completed = service.complete_work_order(order, idempotency_key="done")
    assert completed.state.value == "completed"
    with pytest.raises(ValueError, match="закрыто"):
        service.cancel_work_order(completed, idempotency_key="cancel-late")
    fresh = service.issue_work(
        principal=owner.principal,
        project_id=project.id,
        layer_id=layer.id,
        kind="stitch",
        placement="server",
        assignee_id=performer.id,
        coordinates=((1, 1),),
        image_representation_id=str(images.id),
        idempotency_key="stitch-server",
    )
    assert fresh.placement is WorkPlacement.SERVER
    assert Path(fresh.destination).is_relative_to(service.data_dir)
    cancelled = service.cancel_work_order(fresh, idempotency_key="cancel")
    assert cancelled.state.value == "cancelled"


def test_training_requires_dataset_and_review_requires_vectors(tmp_path: Path) -> None:
    service = _service(tmp_path)
    owner, performer, project, layer, images, _vectors = _project_with_frames(service, tmp_path / "source", with_vectors=False)
    destination = tmp_path / "usb"
    destination.mkdir()
    with pytest.raises(ValueError, match="выборк"):
        service.issue_work(
            principal=owner.principal,
            project_id=project.id,
            layer_id=layer.id,
            kind="train",
            placement="external",
            assignee_id=performer.id,
            coordinates=(),
            destination_parent=destination,
            idempotency_key="train-missing",
        )
    with pytest.raises(ValueError, match="CIF"):
        service.issue_work(
            principal=owner.principal,
            project_id=project.id,
            layer_id=layer.id,
            kind="vector_review",
            placement="external",
            assignee_id=performer.id,
            coordinates=((1, 1),),
            image_representation_id=str(images.id),
            destination_parent=destination,
            idempotency_key="review-missing",
        )


def test_training_archive_is_unpacked_and_review_package_is_tracked(tmp_path: Path) -> None:
    archive = io.BytesIO()
    with ZipFile(archive, "w") as stream:
        stream.writestr("images/frame.png", b"png")
        stream.writestr("cif/frame.cif", b"cif")
    content = archive.getvalue()
    version_id = str(uuid4())

    class Version:
        id = version_id
        media_type = "application/zip"
        filename = "dataset.zip"
        sha256 = hashlib.sha256(content).hexdigest()
        parameters = {"plugin_output_role": "dataset"}
        created_at = datetime.now(UTC)

    class Project:
        id = ProjectId(str(uuid4()))
        name = "Train"
        width = 1
        height = 1

    class Layer:
        id = LayerId(str(uuid4()))

    now = datetime.now(UTC)
    order = place_work_order(
        WorkIssueRequest(
            project=Project(),
            layer=Layer(),
            kind=WorkKind.TRAIN,
            placement=WorkPlacement.EXTERNAL,
            assignee_id=str(PerformerId(str(uuid4()))),
            created_by=str(PrincipalId(str(uuid4()))),
            coordinates=(),
            image_representation_id="",
            vector_representation_id="",
            instructions="Обучить",
            due_at=now + timedelta(days=1),
            destination_parent=tmp_path,
            server_directory_for=lambda _order_id: tmp_path / "unused",
            idempotency_key="train",
            transport="files",
            remote=False,
            now=now,
        ),
        active_version=lambda _series: None,
        read_bytes=lambda version: content if version == version_id else b"",
        dataset_version=lambda: Version(),
    )
    dataset = Path(order.destination) / "dataset"
    assert (dataset / "images" / "frame.png").read_bytes() == b"png"
    assert (dataset / "cif" / "frame.cif").read_bytes() == b"cif"
    (tmp_path / "second").mkdir()
    unsafe = io.BytesIO()
    with ZipFile(unsafe, "w") as stream:
        stream.writestr("../escape.png", b"no")
    with pytest.raises(ValueError, match="небезопасный"):
        place_work_order(
            WorkIssueRequest(
                project=Project(),
                layer=Layer(),
                kind=WorkKind.TRAIN,
                placement=WorkPlacement.EXTERNAL,
                assignee_id=str(uuid4()),
                created_by=str(uuid4()),
                coordinates=(),
                image_representation_id="",
                vector_representation_id="",
                instructions="",
                due_at=None,
                destination_parent=tmp_path / "second",
                server_directory_for=lambda _order_id: tmp_path / "unused",
                idempotency_key="bad-zip",
                transport="files",
                remote=False,
                now=now + timedelta(seconds=1),
            ),
            active_version=lambda _series: None,
            read_bytes=lambda _version: unsafe.getvalue(),
            dataset_version=lambda: Version(),
        )

    service = _service(tmp_path / "review-data")
    owner, performer, project, layer, images, vectors = _project_with_frames(service, tmp_path / "review-source")
    parent = tmp_path / "review-usb"
    parent.mkdir()
    review = service.issue_work(
        principal=owner.principal,
        project_id=project.id,
        layer_id=layer.id,
        kind="vector_review",
        placement="external",
        assignee_id=performer.id,
        coordinates=((1, 1), (2, 1)),
        image_representation_id=str(images.id),
        vector_representation_id=str(vectors.id),
        destination_parent=parent,
        idempotency_key="review",
    )
    assert (Path(review.destination) / "kraken-review.json").is_file()
    assert review.related_batch_id
    assert service.work_orders()[0].id == review.id
    with pytest.raises(ValueError, match="принятием"):
        service.complete_work_order(review, idempotency_key="review-complete")
    closed = WorkOrder.create(
        project_id=project.id,
        layer_id=layer.id,
        kind=WorkKind.STITCH,
        placement=WorkPlacement.SERVER,
        assignee_id=performer.id,
        created_by=owner.principal.id,
        destination=str(tmp_path / "closed"),
        frame_count=1,
        selection=review.selection,
        created_at=now,
    ).cancel(at=now + timedelta(seconds=1))
    with pytest.raises(Exception):
        closed.cancel(at=now + timedelta(seconds=2))
