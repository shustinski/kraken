"""Палитра W3: подготовка выборки агентом на живом Gateway. G18."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4
from zipfile import ZipFile
from textwrap import dedent

import pytest
from PIL import Image

from kraken_agent.runner import PluginProcessSpec, PluginRegistry
from kraken_agent.server_worker import ServerAgentWorker
from kraken_core.plugin_protocol import PluginFrameInput, PluginJobManifest
from kraken_manager.domain.project import GridOrientation, LayerType, RepresentationKind
from kraken_server.configuration import unprotect_secret
from tests.test_live_server_worker_acceptance import ObservedWorker, _connect, _protected_snapshot


class GatewayDatasetWorker(ObservedWorker):
    def _request(self, method, path, payload=None, **kwargs):
        result = super()._request(method, path, payload, **kwargs)
        if path.endswith("/download-ticket") or path.endswith("/upload-ticket"):
            assert result["mode"] == "gateway", result
            self.gateway_tickets.append(path)
        return result


@pytest.fixture
def dataset_stand(tmp_path):
    filename = os.environ.get("KRAKEN_LIVE_WORKER_CONFIG")
    if not filename:
        pytest.skip("Нужен KRAKEN_LIVE_WORKER_CONFIG выделенного beta-стенда")
    config = json.loads(Path(filename).read_text(encoding="utf-8-sig"))
    assert "dataset_machine_secret" in config
    remote = _connect(config, tmp_path / "client")
    protected = _protected_snapshot(remote, config["protected_project_id"])
    active = {"queued", "staging", "running", "recovery_required"}
    assert not [job for job in remote.plugin_jobs() if job.state.value in active]
    worker = GatewayDatasetWorker(
        server_url=config["server_url"],
        token=unprotect_secret(Path(config["dataset_machine_secret"])),
        data_dir=tmp_path / "worker",
        registry=PluginRegistry({
            "frames.dataset.prepare.v1": PluginProcessSpec(
                "frames.dataset.prepare.v1", (sys.executable, "-m", "contour.worker")
            )
        }),
    )
    worker.transitions = []
    worker.allowed_jobs = set()
    worker.gateway_tickets = []
    context = SimpleNamespace(remote=remote, worker=worker, projects=[], jobs=[])
    try:
        yield context
    finally:
        worker.stop()
        for job_id in context.jobs:
            job = next((item for item in remote.plugin_jobs() if str(item.id) == job_id), None)
            if job is not None and job.state.value in active:
                remote.cancel_plugin_job(
                    principal=remote.auth.principal, gateway=None, job=job, idempotency_key=str(uuid4())
                )
        for project_id in context.projects:
            project = remote.get_project(project_id)
            if project is not None and project.state.value != "archived":
                remote.archive_project(
                    principal=remote.auth.principal, project=project, idempotency_key=str(uuid4())
                )
        assert not [job for job in remote.plugin_jobs() if job.state.value in active]
        assert _protected_snapshot(remote, config["protected_project_id"]) == protected


def _project_with_paired_inputs(env, tmp_path):
    remote = env.remote
    principal = remote.auth.principal
    project = remote.create_project(
        principal=principal, name=f"palette-g18-{uuid4().hex[:12]}", width=2, height=1,
        orientation=GridOrientation.Y_DOWN, idempotency_key=str(uuid4()),
    )
    env.projects.append(project.id)
    layer = remote.create_layer(
        principal=principal, project=project, name="Metal", layer_type=LayerType.METAL,
        order=0, idempotency_key=str(uuid4()),
    )
    image_rep = remote.create_representation(
        principal=principal, project=remote.get_project(project.id), layer=layer,
        name="Image", kind=RepresentationKind.IMAGE, active=True, idempotency_key=str(uuid4()),
    )
    images = tmp_path / "images"
    images.mkdir()
    image = images / "1_1.png"
    Image.new("RGB", (16, 16), "white").save(image)
    image_plan = remote.plan_import_directory(project=project, directory=images, kind=RepresentationKind.IMAGE)
    assert image_plan.ready and len(image_plan.items) == 1
    remote.commit_managed_import(
        principal=principal, project=project, layer=layer, representation=image_rep,
        plan=image_plan, idempotency_key=str(uuid4()),
    )
    layer = next(item for item in remote.list_layers(project.id) if item.id == layer.id)
    vector_rep = remote.create_representation(
        principal=principal, project=remote.get_project(project.id), layer=layer,
        name="CIF", kind=RepresentationKind.VECTOR, source_image_representation_id=image_rep.id,
        active=True, idempotency_key=str(uuid4()),
    )
    vectors = tmp_path / "vectors"
    vectors.mkdir()
    (vectors / "1_1.cif").write_text(
        "DS 1 1 1; L M1; ( S 16 16 ); P 0 0 8 0 8 8 0 8; DF; E;\n", encoding="utf-8"
    )
    vector_plan = remote.plan_import_directory(project=project, directory=vectors, kind=RepresentationKind.VECTOR)
    assert vector_plan.ready and len(vector_plan.items) == 1
    remote.commit_managed_import(
        principal=principal, project=project, layer=layer, representation=vector_rep,
        plan=vector_plan, idempotency_key=str(uuid4()),
    )
    return project, layer, image_rep, vector_rep


def test_g18_dataset_prepare_worker_imports_dataset(dataset_stand, tmp_path):
    env = dataset_stand
    remote, worker = env.remote, env.worker
    project, layer, image_rep, vector_rep = _project_with_paired_inputs(env, tmp_path)
    job = remote.submit_plugin_job(
        principal=remote.auth.principal, gateway=None, project_id=project.id,
        layer_id=layer.id, source_representation_id=image_rep.id,
        target_representation_id=image_rep.id, coordinates=[(1, 1)],
        capability="frames.dataset.prepare.v1",
        parameters={
            "execution_host": "server", "input_transport": "server",
            "vector_representation_id": str(vector_rep.id),
        },
        idempotency_key=str(uuid4()),
    )
    env.jobs.append(str(job.id))
    worker.allowed_jobs.add(str(job.id))
    assert job.state.value == "queued"
    assert worker.run_once(job_id=str(job.id))
    completed = next(item for item in remote.plugin_jobs() if item.id == job.id)
    assert completed.state.value == "succeeded", completed.error
    assert any(entry["path"].endswith("/lease") for entry in worker.transitions)
    assert worker.gateway_tickets
    series = remote.list_artifact_series(project.id, layer_id=layer.id)
    datasets = [
        version for item in series for version in remote.artifact_versions(project.id, item.id)
        if version.media_type == "application/zip"
    ]
    assert len(datasets) == 1
    version = datasets[0]
    path = remote.managed_artifact_path(project.id, version.id)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == version.sha256
    with ZipFile(path) as archive:
        assert any(name.startswith("images/") for name in archive.namelist())
        assert any(name.startswith("cif/") for name in archive.namelist())


def test_g18_interactive_prepare_via_hub_gateway_path(dataset_stand, tmp_path, monkeypatch):
    env = dataset_stand
    remote = env.remote
    project, layer, image_rep, vector_rep = _project_with_paired_inputs(env, tmp_path)
    job = remote.submit_plugin_job(
        principal=remote.auth.principal, gateway=None, project_id=project.id,
        layer_id=layer.id, source_representation_id=image_rep.id,
        target_representation_id=image_rep.id, coordinates=[(1, 1)],
        capability="frames.dataset.prepare.v1",
        parameters={
            "execution_host": "desktop", "input_transport": "server",
            "vector_representation_id": str(vector_rep.id),
        },
        idempotency_key=str(uuid4()),
    )
    env.jobs.append(str(job.id))
    script = tmp_path / "interactive_contour.py"
    script.write_text(dedent('''
        from pathlib import Path
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QMessageBox
        from contour.kraken_bridge import prepare_contour_launch
        from contour.application import bootstrap
        from contour.application.cli import main

        session, arguments = prepare_contour_launch([])
        assert len(session.input_paths) == 1
        original_build = bootstrap.build_application

        def build(argv):
            app, window = original_build([*argv, "--no-qss", "--log-file", str(session.staging_root / "contour.log")])

            def publish():
                assert window.isVisible()
                assert window._widget._image_list_model.rowCount() == 1
                image = session.input_paths[0]
                output_images = session.output_directory / "images"
                output_cif = session.output_directory / "cif"
                output_images.mkdir(exist_ok=True)
                output_cif.mkdir(exist_ok=True)
                if session.server_source is not None:
                    with session.server_source.open(image.relative_to(session.staging_root).as_posix()) as stream:
                        (output_images / image.name).write_bytes(stream.read())
                else:
                    (output_images / image.name).write_bytes(image.read_bytes())
                vector = session.vector_paths[0]
                (output_cif / (image.stem + ".cif")).write_bytes(vector.read_bytes())
                QMessageBox.information = lambda *args: QMessageBox.StandardButton.Ok
                QMessageBox.question = lambda *args: QMessageBox.StandardButton.Yes
                window._kraken_return_action.trigger()
                app.quit()

            QTimer.singleShot(500, publish)
            return app, window

        bootstrap.build_application = build
        main([])
    '''), encoding="utf-8")
    monkeypatch.setenv("VIALANET_SETTINGS_DIR", str(tmp_path / "contour-settings"))
    observed_tickets = []
    original_request = ServerAgentWorker._request

    def observe_request(self, method, path, payload=None, **kwargs):
        result = original_request(self, method, path, payload, **kwargs)
        if path.endswith("/download-ticket") or path.endswith("/upload-ticket"):
            assert result["mode"] == "gateway", result
            observed_tickets.append(path)
        return result

    monkeypatch.setattr(ServerAgentWorker, "_request", observe_request)
    registry = PluginRegistry({
        "frames.dataset.prepare.v1": PluginProcessSpec(
            "frames.dataset.prepare.v1", (sys.executable, str(script)), interactive=True
        )
    })
    result = remote.run_interactive_job(project.id, job.id, registry=registry)
    assert result["imported"]["job"]["state"] == "succeeded"
    assert len(result["imported"]["versions"]) == 1
    assert observed_tickets
    version = result["imported"]["versions"][0]
    assert version["media_type"] == "application/zip"
    assert hashlib.sha256(remote.managed_artifact_path(project.id, version["id"]).read_bytes()).hexdigest() == version["sha256"]


def test_g18_headless_worker_rejects_unpaired_cif(tmp_path):
    from contour.worker import run

    staged = tmp_path / "inputs" / "1_1"
    staged.mkdir(parents=True)
    image = staged / "frame.png"
    image.write_bytes(b"test image bytes")
    manifest = PluginJobManifest(
        job_id="dataset-job", operation="frames.dataset.prepare.v1",
        project_id="project", layer_id="layer", actor_id="actor",
        target_representation_id="target",
        inputs=(PluginFrameInput(
            frame_id="frame", x=1, y=1, artifact_version_id="image-version",
            sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            media_type="image/png", relative_path="inputs/1_1/frame.png",
        ),),
        parameters={},
    )
    job = tmp_path / "job.json"
    result = tmp_path / "result.json"
    job.write_text(manifest.to_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset frame has no CIF"):
        run(job, result, tmp_path)
    assert not (tmp_path / "outputs" / "dataset.zip").exists()


def test_g18_cancel_queued_dataset_job_preserves_project(dataset_stand, tmp_path):
    env = dataset_stand
    remote = env.remote
    project, layer, image_rep, vector_rep = _project_with_paired_inputs(env, tmp_path)
    before = [event.event_id for event in remote.history(project.id)]
    job = remote.submit_plugin_job(
        principal=remote.auth.principal, gateway=None, project_id=project.id,
        layer_id=layer.id, source_representation_id=image_rep.id,
        target_representation_id=image_rep.id, coordinates=[(1, 1)],
        capability="frames.dataset.prepare.v1",
        parameters={
            "execution_host": "server", "input_transport": "server",
            "vector_representation_id": str(vector_rep.id),
        },
        idempotency_key=str(uuid4()),
    )
    env.jobs.append(str(job.id))
    assert job.state.value == "queued"
    cancelled = remote.cancel_plugin_job(
        principal=remote.auth.principal, gateway=None, job=job, idempotency_key=str(uuid4())
    )
    assert cancelled.state.value == "cancelled"
    assert remote.get_project(project.id) is not None
    assert [event.event_id for event in remote.history(project.id)][:len(before)] == before
