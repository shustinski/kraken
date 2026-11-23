"""Палитра W3 — векторизация двух слоёв отдельными заданиями агента (live, gateway). G17."""

from __future__ import annotations

from uuid import uuid4

from PIL import Image, ImageDraw

from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind
from tests.test_live_server_worker_acceptance import _upload_frame, live_worker


def _active_variant(remote, project, layer_name: str, shade: int, folder):
    principal = remote.auth.principal
    layer = remote.create_layer(
        principal=principal, project=project, name=layer_name, layer_type=LayerType.METAL,
        order=shade, idempotency_key=str(uuid4()),
    )
    first = remote.create_representation(
        principal=principal, project=project, layer=layer, name="Первый",
        kind=RepresentationKind.IMAGE, active=True, idempotency_key=str(uuid4()),
    )
    layer = next(item for item in remote.list_layers(project.id) if item.id == layer.id)
    second = remote.create_representation(
        principal=principal, project=project, layer=layer, name="Второй",
        kind=RepresentationKind.IMAGE, active=False, idempotency_key=str(uuid4()),
    )
    layer = next(item for item in remote.list_layers(project.id) if item.id == layer.id)
    active = remote.activate_representation(
        principal=principal, project=remote.get_project(project.id), layer=layer,
        representation=second, idempotency_key=str(uuid4()),
    )
    source = folder / f"layer-{shade}.png"
    image = Image.new("RGB", (64, 64), "black")
    ImageDraw.Draw(image).rectangle((4, 4, 20 + shade, 40), fill="white")
    image.save(source)
    layer = next(item for item in remote.list_layers(project.id) if item.id == layer.id)
    version = _upload_frame(remote, project.id, layer.id, active.id, source, 1, 1)
    layer = next(item for item in remote.list_layers(project.id) if item.id == layer.id)
    vector = remote.create_representation(
        principal=principal, project=remote.get_project(project.id), layer=layer, name="Вектор",
        kind=RepresentationKind.VECTOR, source_image_representation_id=active.id, active=True,
        idempotency_key=str(uuid4()),
    )
    return layer, first, active, vector, version


def _vectorize(remote, worker, project, layer, image, vector):
    job = remote.submit_plugin_job(
        principal=remote.auth.principal, gateway=None, project_id=project.id, layer_id=layer.id,
        source_representation_id=image.id, target_representation_id=vector.id, coordinates=[(1, 1)],
        capability="frames.vectorize.v1",
        parameters={"input_transport": "server", "execution_mode": "headless", "execution_host": "server"},
        idempotency_key=str(uuid4()),
    )
    worker.allowed_jobs.add(str(job.id))
    assert job.state.value == "queued"
    assert worker.run_once(job_id=str(job.id))
    done = next(item for item in remote.plugin_jobs() if item.id == job.id)
    assert done.state.value == "succeeded", (done.state, done.error)
    return done


def test_g17_two_layers_vectorized_separately_active_variant(live_worker, tmp_path):
    remote, worker = live_worker.remote, live_worker.worker
    project = remote.create_project(
        principal=remote.auth.principal, name=f"palette-g17-{uuid4().hex[:8]}",
        width=1, height=1, orientation=GridOrientation.Y_DOWN, idempotency_key=str(uuid4()),
    )
    try:
        metal = _active_variant(remote, project, "Металл", 8, tmp_path)
        dielectric = _active_variant(remote, remote.get_project(project.id), "Диэлектрик", 24, tmp_path)
        _vectorize(remote, worker, project, metal[0], metal[2], metal[3])
        _vectorize(remote, worker, project, dielectric[0], dielectric[2], dielectric[3])
        digests = []
        for layer, _first, active, vector, version in (metal, dielectric):
            cells = [cell for cell in remote.frame_cells(project.id, layer.id, vector.id) if cell.sha256]
            assert len(cells) == 1
            saved = remote.artifact_version(project.id, cells[0].artifact_version_id)
            assert list(map(str, saved.input_version_ids)) == [str(version.id)]
            assert str(active.id) != str(_first.id)
            digests.append(saved.sha256)
        assert digests[0] != digests[1]
    finally:
        current = remote.get_project(project.id)
        if current.state.value != "archived":
            remote.archive_project(
                principal=remote.auth.principal, project=current, idempotency_key=str(uuid4()),
            )


def test_g17_cancel_marks_failed_project_intact(live_worker, tmp_path):
    remote = live_worker.remote
    project = remote.create_project(
        principal=remote.auth.principal, name=f"palette-g17-{uuid4().hex[:8]}",
        width=1, height=1, orientation=GridOrientation.Y_DOWN, idempotency_key=str(uuid4()),
    )
    job = None
    try:
        layer, _first, active, vector, _version = _active_variant(remote, project, "Металл", 12, tmp_path)
        before = remote.get_project(project.id).revision
        job = remote.submit_plugin_job(
            principal=remote.auth.principal, gateway=None, project_id=project.id, layer_id=layer.id,
            source_representation_id=active.id, target_representation_id=vector.id, coordinates=[(1, 1)],
            capability="frames.vectorize.v1",
            parameters={"input_transport": "server", "execution_mode": "headless", "execution_host": "server"},
            idempotency_key=str(uuid4()),
        )
        cancelled = remote.cancel_plugin_job(
            principal=remote.auth.principal, gateway=None, job=job, idempotency_key=str(uuid4()),
        )
        assert cancelled.state.value in {"cancelled", "failed"}
        assert remote.get_project(project.id).revision == before
        assert not [cell for cell in remote.frame_cells(project.id, layer.id, vector.id) if cell.sha256]
    finally:
        if job is not None:
            current_job = next((item for item in remote.plugin_jobs() if item.id == job.id), None)
            if current_job is not None and current_job.state.value in {"queued", "staging", "running", "recovery_required"}:
                remote.cancel_plugin_job(
                    principal=remote.auth.principal, gateway=None, job=current_job, idempotency_key=str(uuid4()),
                )
        current = remote.get_project(project.id)
        if current.state.value != "archived":
            remote.archive_project(
                principal=remote.auth.principal, project=current, idempotency_key=str(uuid4()),
            )
