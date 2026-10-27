"""Opt-in bounded actual CPU training -> managed model port -> binary inference.

No quality metrics: synthetic fixture proves publication, transport and lineage.
The entire child process has a hard 180 second limit.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4
import zipfile

import pytest


@pytest.mark.skipif(os.environ.get("KRAKEN_RUN_MODEL_FUNCTIONAL") != "1", reason="Explicit bounded actual CPU model gate")
def test_actual_cpu_training_to_pipeline_model_port_and_inference(tmp_path):
    # This Windows CUDA runtime treats an empty value as visible GPUs; -1 was
    # checked with a real torch child and hides them. Verify actual runtime below.
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1" if os.name == "nt" else "", QT_QPA_PLATFORM="offscreen",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (tmp_path / "gate.log").open("wb") as log:
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
            env=environment, stdout=log, stderr=subprocess.STDOUT, **options)
        try:
            returncode = process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            else:
                import signal
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            raise
    assert returncode == 0, (tmp_path / "gate.log").read_text(encoding="utf-8", errors="replace")[-10000:]
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"


def run_gate(root):
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(3)
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    from PIL import Image, ImageDraw
    from contour.domain import PolygonData
    from contour.serializers import save_polygons_cif
    from kraken_core.plugins import PluginInventoryItem, load_plugin_metadata_from_directory
    from kraken_hub.agent_runtime import LocalAgentRuntime
    from kraken_hub.composition import EmbeddedProjectService
    from kraken_manager.application.dto import AddArtifactVersionCommand, CommandContext
    from kraken_manager.application.use_cases import AddArtifactVersionHandler
    from kraken_manager.domain.artifacts import ArtifactScope
    from kraken_manager.domain.pipeline import LayerPipelineDefinition, PipelineOperation, PipelineModelInput, PipelinePolicy, PipelineResources
    from kraken_manager.domain.project import RepresentationKind, RepresentationPurpose
    from kraken_manager.infrastructure.blob.filesystem import FilesystemBlobStore
    from kraken_manager.infrastructure.plugin import AgentPluginGateway
    from tests.test_work_orders import _service, _project_with_frames

    started = time.monotonic()
    local = _service(root)
    owner, _, project, layer, images, _ = _project_with_frames(local, root / "initial", with_vectors=False)
    dataset = root / "dataset"
    for index in range(3):
        split = dataset if index < 2 else dataset / "validation"
        for directory in ("images", "cif"):
            (split / directory).mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (128, 128), (20 + index, 20, 20))
        ImageDraw.Draw(image).rectangle((20 + index, 30, 100, 90), fill=(200, 200, 200))
        path = split / "images" / f"{index}.png"
        image.save(path)
        save_polygons_cif(split / "cif" / f"{index}.cif", str(path),
            [PolygonData(1, [(20 + index, 30), (100, 30), (100, 90), (20 + index, 90)])], (128, 128))
    source = root / "source"
    source.mkdir()
    (source / "1_1.png").write_bytes((dataset / "images/0.png").read_bytes())
    local.commit_managed_import(principal=owner.principal, project=local.get_project(project.id), layer=layer,
        representation=images, plan=local.plan_import_directory(project=project, directory=source), idempotency_key="real-image")
    image_series = next(s for s in local.list_artifact_series(project.id) if s.frame_id == project.frame_id_at(1, 1))
    image_version = local.active_artifact_version(project.id, image_series.id)
    archive_path = root / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dataset.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(dataset).as_posix())
    series = local.create_artifact_series(principal=owner.principal, project_id=project.id, layer_id=layer.id,
        scope=ArtifactScope.LAYER_ATTACHMENT, name="Synthetic bounded dataset", idempotency_key="dataset-series")
    dataset_version = AddArtifactVersionHandler(local._uow(str(project.id)), local.profiles, local.clock)(
        AddArtifactVersionCommand(context=CommandContext(actor=owner.principal, idempotency_key="dataset-version"),
            project_id=project.id, series_id=series.id, filename="dataset.zip", media_type="application/zip",
            expected_series_revision=local.artifact_stream_revision(project.id, series.id),
            parameters={"artifact_role": "dataset"}), (archive_path.read_bytes(),))

    def representation(name, purpose=None):
        options = {} if purpose is None else {"purpose": purpose}
        return local.create_representation(principal=owner.principal, project=local.get_project(project.id),
            layer=local.list_layers(project.id)[0], name=name, kind=RepresentationKind.IMAGE,
            active=False, idempotency_key=name, **options)

    dataset_rep, model_rep, mask_rep = representation("Dataset port"), representation("Model port"), representation("Mask", RepresentationPurpose.BINARY)
    metadata = load_plugin_metadata_from_directory(repository / "plugins/neuralimage")
    train_id, infer_id = str(uuid4()), str(uuid4())
    settings = dict(model="S 660k", sample_size=[128, 128], train_patch_size=[128, 128], step=128,
        batch_size=2, train_batch_size=2, vertical_rotation=False, horizontal_rotation=False,
        use_validation=True, validation_source="external", dataloader_num_workers=0, early_stopping_enabled=True)
    train = PipelineOperation(id=train_id, title="Bounded one epoch CPU train", capability="dataset.model.train.v1",
        algorithm_version=metadata.version, source_representation_id=dataset_rep.id, target_representation_id=model_rep.id,
        input_kind="dataset", output_kind="model", coordinates=((1, 1),),
        parameters={"dataset_version_id": dataset_version.id, "training": {"epochs": 1, "seed": 1234,
            "validation_relative_root": "validation", "settings": settings}})
    inference = PipelineOperation(id=infer_id, title="Infer using published model", capability="frames.binary-segment.v2",
        algorithm_version=metadata.version, source_representation_id=images.id, target_representation_id=mask_rep.id,
        input_kind="image", output_kind="binary_mask", coordinates=((1, 1),), dependencies=(train_id,),
        model_input=PipelineModelInput(producer_id=train_id), parameters={"execution_mode": "headless", "patch_size": 128})
    definition = LayerPipelineDefinition(str(project.id), str(layer.id), (train, inference),
        PipelinePolicy(capacity=PipelineResources(cpu=1, memory_mb=1024)))
    state = local.define_layer_pipeline(project_id=project.id, layer_id=layer.id, definition=definition.to_dict(),
        principal=owner.principal, expected_revision=0, idempotency_key="define")
    state = local.command_layer_pipeline(project_id=project.id, layer_id=layer.id, operation="start",
        principal=owner.principal, expected_revision=state["revision"], idempotency_key="start")
    runtime = LocalAgentRuntime(root / "agent", (PluginInventoryItem(metadata, installed=False),))
    try:
        capabilities = runtime.ensure_started()
        blobs = FilesystemBlobStore.for_project(local.catalog_root, str(project.id))
        gateway = AgentPluginGateway(base_url=runtime.base_url, token=runtime.token, staging_root=root / "agent/staging",
            capabilities=capabilities, source_for_version=lambda identifier: blobs.path_for_read(local.artifact_version(project.id, identifier).blob),
            coordinate_for_frame=lambda identifier: (1, 1),
            media_type_for_version=lambda identifier: local.artifact_version(project.id, identifier).media_type)
        while time.monotonic() - started < 150:
            state = local.tick_layer_pipeline(project_id=project.id, layer_id=layer.id, principal=owner.principal, gateway=gateway)
            (root / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
            for job in local.synchronize_plugin_jobs(principal=owner.principal, gateway=gateway):
                if job.state.value == "importing":
                    imported = local.import_agent_result(principal=owner.principal, result_payload=gateway.get_result(job.id), staging_root=root / "agent/staging")
                    assert not imported.requires_partial_confirmation
                    gateway.complete_import(job.id)
            if state["status"] in {"completed", "failed"}:
                break
            time.sleep(0.1)
        assert state["status"] == "completed", json.dumps(state)
        model_id, = state["nodes"][train_id]["attempts"][0]["output_version_ids"]
        mask_id, = state["nodes"][infer_id]["attempts"][0]["output_version_ids"]
        model = local.artifact_version(project.id, model_id)
        mask = local.artifact_version(project.id, mask_id)
        execution = model.parameters["managed_execution"]
        assert execution["epoch_upper_bound"] == 1
        assert execution["train_images"] == 2 and execution["validation_images"] == 1
        assert execution["cuda_available"] is False, "Agent must preserve the explicit CPU environment"
        assert model.parameters["main_window_state"]["epochs"] == 1
        assert mask.parameters["model_sha256"] == model.sha256
        assert mask.parameters["model_version"] == model_id
        assert mask.parameters["lossless_binary_png"] is True
        assert dataset_version.id in model.input_version_ids
        assert {model_id, image_version.id} <= set(mask.input_version_ids)
        assert model.blob.size_bytes > 100000
        for version in (model, mask):
            blobs.verify(version.blob)
            assert hashlib.sha256(blobs.path_for_read(version.blob).read_bytes()).hexdigest() == version.sha256
        with Image.open(blobs.path_for_read(mask.blob)) as output:
            assert output.format == "PNG" and output.size == (128, 128)
            assert set(output.convert("L").tobytes()) <= {0, 255}
        from neuralimage.model.NeuralNetwork.model_io import load_model_artifact
        checkpoint = load_model_artifact(blobs.path_for_read(model.blob), map_location="cpu")
        assert sum(p.numel() for p in checkpoint.parameters()) > 100000
        reopened = EmbeddedProjectService(local.data_dir, secret_store=local.secrets)
        assert reopened.artifact_version(project.id, model_id) == model
        assert reopened.artifact_version(project.id, mask_id) == mask
        assert reopened.get_layer_pipeline(project.id, layer.id)["status"] == "completed"
        receipt = {"status": "passed", "seconds": round(time.monotonic() - started, 3),
            "synthetic": True, "training": "2 train + 1 external validation, 128x128, 1 epoch, CPU",
            "project_id": str(project.id), "dataset_version_id": dataset_version.id,
            "image_version_id": image_version.id, "model_version_id": model_id, "mask_version_id": mask_id,
            "model_sha256": model.sha256, "mask_sha256": mask.sha256,
            "cuda_available": execution["cuda_available"], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "epoch_upper_bound": execution["epoch_upper_bound"], "train_images": execution["train_images"],
            "validation_images": execution["validation_images"], "model_size_bytes": model.blob.size_bytes,
            "limitations": "No quality metrics, no electrical ground truth; local Agent path only"}
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt), flush=True)
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    run_gate(Path(sys.argv[1]).resolve())
