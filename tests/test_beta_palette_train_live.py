"""G19: bounded CPU model training through the real Server machine lease."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from uuid import uuid4
from zipfile import ZipFile

from PIL import Image, ImageDraw
from sqlalchemy import create_engine

from contour.domain import PolygonData
from contour.serializers import save_polygons_cif
from kraken_agent.runner import PluginProcessSpec, PluginRegistry
from kraken_manager.domain.project import RepresentationKind
from kraken_server.agent_auth import PostgresAgentTokenStore
from tests.test_beta_palette_agent_ops import GatewayDatasetWorker, _project_with_paired_inputs

pytest_plugins = ("tests.test_beta_palette_agent_ops",)


def _bounded_dataset(path: Path) -> Path:
    source = path / "training-source"
    for index in range(3):
        image_dir, cif_dir = source / "images", source / "cif"
        image_dir.mkdir(parents=True, exist_ok=True)
        cif_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{index}.png"
        image = Image.new("RGB", (128, 128), (20 + index, 20, 20))
        ImageDraw.Draw(image).rectangle((20 + index, 30, 100, 90), fill=(200, 200, 200))
        image.save(image_path)
        save_polygons_cif(
            cif_dir / f"{index}.cif", str(image_path),
            [PolygonData(1, [(20 + index, 30), (100, 30), (100, 90), (20 + index, 90)])],
            (128, 128),
        )
    archive_path = path / "dataset.zip"
    with ZipFile(archive_path, "w") as archive:
        for item in sorted(source.rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(source).as_posix())
    return archive_path


def test_g19_server_training_publishes_verified_checkpoint(dataset_stand, tmp_path, monkeypatch):
    env = dataset_stand
    remote = env.remote
    project, layer, image_rep, _ = _project_with_paired_inputs(env, tmp_path)
    archive = _bounded_dataset(tmp_path)
    series = remote._call(
        "POST", f"/api/v1/projects/{project.id}/artifacts",
        payload={"name": "Bounded CPU training dataset", "scope": "layer_attachment", "layer_id": str(layer.id)},
        idempotency_key=str(uuid4()),
    )
    metadata = {
        "filename": archive.name, "media_type": "application/zip",
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "size_bytes": archive.stat().st_size, "parameters": {"artifact_role": "dataset"},
    }
    key = str(uuid4())
    uploads = f"/api/v1/projects/{project.id}/artifacts/{series['id']}/uploads"
    revision = remote.artifact_stream_revision(project.id, series["id"])
    ticket = remote._call("POST", uploads, payload=metadata, idempotency_key=key, if_match=revision)
    assert ticket["mode"] == "gateway"
    remote._http.upload_url(ticket["url"], token=ticket["token"], source=archive)
    dataset = remote._call(
        "POST", uploads + "/complete", payload={**metadata, "upload_token": ticket["token"]},
        idempotency_key=key, if_match=revision,
    )
    layer = next(item for item in remote.list_layers(project.id) if item.id == layer.id)
    target = remote.create_representation(
        principal=remote.auth.principal, project=remote.get_project(project.id), layer=layer,
        name="Trained model", kind=RepresentationKind.IMAGE, idempotency_key=str(uuid4()),
    )
    settings = {
        "model": "S 660k", "sample_size": [128, 128], "train_patch_size": [128, 128],
        "step": 128, "batch_size": 2, "train_batch_size": 2, "vertical_rotation": False,
        "horizontal_rotation": False, "use_validation": True, "validation_source": "split",
        "dataloader_num_workers": 0, "early_stopping_enabled": True,
    }
    job = remote.submit_plugin_job(
        principal=remote.auth.principal, gateway=None, project_id=project.id,
        layer_id=layer.id, source_representation_id=image_rep.id,
        target_representation_id=target.id, coordinates=[(1, 1)], capability="dataset.model.train.v1",
        parameters={
            "dataset_version_id": dataset["id"], "input_transport": "server", "execution_host": "server",
            "execution_mode": "headless",
            "training": {"epochs": 1, "seed": 1234, "settings": settings},
        },
        idempotency_key=str(uuid4()),
    )
    env.jobs.append(str(job.id))
    engine = create_engine(os.environ["KRAKEN_TEST_POSTGRES_URL"], pool_pre_ping=True)
    tokens = PostgresAgentTokenStore(engine)
    identity, token = tokens.create(f"palette-g19-train-{uuid4().hex[:8]}", frozenset({"dataset.model.train.v1"}))
    worker = GatewayDatasetWorker(
        server_url=env.worker.server_url, token=token, data_dir=tmp_path / "training-worker",
        registry=PluginRegistry({
            "dataset.model.train.v1": PluginProcessSpec(
                "dataset.model.train.v1", (sys.executable, "-m", "neuralimage")
            )
        }),
    )
    worker.allowed_jobs = {str(job.id)}
    worker.transitions = []
    worker.gateway_tickets = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1" if os.name == "nt" else "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    try:
        assert worker.run_once(job_id=str(job.id))
        completed = next(item for item in remote.plugin_jobs() if item.id == job.id)
        assert completed.state.value == "succeeded", completed.error
        assert worker.gateway_tickets
        versions = [
            version for item in remote.list_artifact_series(project.id, layer_id=layer.id)
            for version in remote.artifact_versions(project.id, item.id)
            if version.media_type == "application/octet-stream"
            and version.parameters.get("plugin_output_role") == "model"
        ]
        assert len(versions) == 1
        model = versions[0]
        assert model.size_bytes > 100000
        assert model.parameters["managed_execution"]["epoch_upper_bound"] == 1
        assert model.parameters["managed_execution"]["cuda_available"] is False
        assert str(dataset["id"]) in set(map(str, model.input_version_ids))
        assert hashlib.sha256(remote.managed_artifact_path(project.id, model.id).read_bytes()).hexdigest() == model.sha256
    finally:
        worker.stop()
        assert tokens.revoke(identity.token_id)
        engine.dispose()
