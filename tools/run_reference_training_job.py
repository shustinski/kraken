"""Run a real managed NeuralImage training job on the published reference corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from kraken_core.plugins import PluginInventoryItem, load_plugin_metadata_from_directory
from kraken_hub.agent_runtime import LocalAgentRuntime
from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.secret_store import KeyringSecretStore
from kraken_manager.application.dto import AddArtifactVersionCommand, CommandContext, StoredContent
from kraken_manager.application.use_cases import AddArtifactVersionHandler
from kraken_manager.domain.artifacts import is_training_dataset_artifact
from kraken_manager.infrastructure.blob.filesystem import FilesystemBlobStore
from kraken_manager.infrastructure.plugin.agent_gateway import AgentPluginGateway


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--deadline-hours", type=float, default=6)
    args = parser.parse_args()
    if min(args.epochs, args.patch_size, args.batch_size, args.deadline_hours) <= 0:
        parser.error("Training dimensions, epochs and observation deadline must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = json.loads((args.profile / "state.json").read_text())
    dataset = json.loads(args.dataset_report.read_text())
    assert dataset["status"] == "passed" and dataset["project_id"] == state["project_id"]
    service = EmbeddedProjectService(args.profile / "data", secret_store=KeyringSecretStore(state["secret_service"]))
    owner = service.identities.get(state["owner_id"])
    assert owner is not None and owner.active
    version = service.artifact_version(state["project_id"], dataset["dataset_version_id"])
    assert version is not None and version.sha256 == dataset["dataset_sha256"]
    # The first corpus bundle predates the explicit manual artifact-role contract.
    # Correct classification with a NEW immutable version, preserving its parent,
    # bytes and every source reference; do not pretend a plugin published it.
    if not is_training_dataset_artifact(version):
        version = AddArtifactVersionHandler(service._uow(state["project_id"]), service.profiles, service.clock)(
            AddArtifactVersionCommand(
                context=CommandContext(actor=owner, idempotency_key="typed-reference-dataset:" + version.sha256),
                project_id=state["project_id"],
                series_id=version.series_id,
                filename=version.filename,
                media_type=version.media_type,
                expected_series_revision=service.artifact_stream_revision(state["project_id"], version.series_id),
                parent_version_id=version.id,
                input_version_ids=version.input_version_ids,
                expected_sha256=version.sha256,
                tool_name="Kraken reference dataset publisher",
                tool_version="1",
                parameters={
                    **dict(version.parameters),
                    "artifact_role": "dataset",
                    "dataset_format": "kraken.reference-training.v1",
                },
            ),
            stored=StoredContent(version.blob, already_existed=True),
        )
    repo = Path(__file__).resolve().parents[1]
    metadata = load_plugin_metadata_from_directory(repo / "plugins/neuralimage")
    runtime = LocalAgentRuntime(output / "agent", (PluginInventoryItem(metadata, installed=False),))
    evidence = dict(
        status="starting",
        project_id=state["project_id"],
        dataset_version_id=str(version.id),
        dataset_sha256=version.sha256,
        train_count=dataset["train_count"],
        validation_count=dataset["validation_count"],
        heldout_excluded=dataset["heldout_excluded"],
        truth_status="supplied_annotations_not_independently_verified",
    )
    started = time.monotonic()
    job_id = None

    def save():
        from kraken_manager.infrastructure.filesystem._atomic import atomic_write_json

        atomic_write_json(output / "report.json", evidence)

    save()
    try:
        capabilities = runtime.ensure_started()
        blobs = FilesystemBlobStore.for_project(service.catalog_root, state["project_id"])
        project = service.get_project(state["project_id"])
        first_frame = project.frame_id_at(1, 1)
        gateway = AgentPluginGateway(
            base_url=runtime.base_url,
            token=runtime.token,
            staging_root=output / "agent/staging",
            capabilities=capabilities,
            timeout=15,
            source_for_version=lambda identifier: blobs.path_for_read(
                service.artifact_version(project.id, identifier).blob
            ),
            coordinate_for_frame=lambda identifier: (
                (1, 1)
                if str(identifier) == str(first_frame)
                else (_ for _ in ()).throw(ValueError("Training job frame outside its declared selection"))
            ),
            media_type_for_version=lambda identifier: service.artifact_version(project.id, identifier).media_type,
        )
        parameters = dict(
            execution_mode="headless",
            dataset_version_id=str(version.id),
            execution_resources=dict(cpu=2, gpu=1, memory_mb=4096),
            training=dict(
                epochs=args.epochs,
                seed=20261025,
                validation_relative_root="validation",
                settings=dict(
                    model="S 660k",
                    sample_size=[args.patch_size, args.patch_size],
                    train_patch_size=[args.patch_size, args.patch_size],
                    step=args.patch_size,
                    batch_size=args.batch_size,
                    train_batch_size=args.batch_size,
                    dataloader_num_workers=0,
                    vertical_rotation=False,
                    horizontal_rotation=False,
                    use_validation=True,
                    validation_source="external",
                    resize_enabled=False,
                    early_stopping_enabled=True,
                ),
            ),
        )
        evidence["parameters"] = parameters
        save()
        job = service.submit_plugin_job(
            principal=owner,
            gateway=gateway,
            project_id=project.id,
            layer_id=state["layer_id"],
            source_representation_id=state["image_representation_id"],
            target_representation_id=state["image_representation_id"],
            coordinates=((1, 1),),
            capability="dataset.model.train.v1",
            parameters=parameters,
            idempotency_key="reference-real-training:" + version.sha256 + ":" + str(args.epochs),
        )
        job_id = str(job.id)
        evidence.update(status="running", job_id=job_id, algorithm_version=metadata.version)
        save()
        print("Actual NeuralImage training submitted:", job_id, flush=True)
        deadline = time.monotonic() + args.deadline_hours * 3600
        last_status = None
        while time.monotonic() < deadline:
            for current in service.synchronize_plugin_jobs(principal=owner, gateway=gateway):
                if str(current.id) != job_id:
                    continue
                status = current.state.value
                if status != last_status:
                    evidence.update(status=status, seconds=round(time.monotonic() - started, 3))
                    save()
                    print("Training state:", status, "elapsed", evidence["seconds"], flush=True)
                    last_status = status
                if status == "importing":
                    result = service.import_agent_result(
                        principal=owner,
                        result_payload=gateway.get_result(job_id),
                        staging_root=output / "agent/staging",
                    )
                    assert not result.requires_partial_confirmation and not result.awaiting_authorization
                    gateway.complete_import(job_id)
                    models = [
                        value for value in result.versions if value.parameters.get("plugin_output_role") == "model"
                    ]
                    assert len(models) == 1
                    model = models[0]
                    assert str(version.id) in set(map(str, model.input_version_ids))
                    evidence.update(
                        status="passed",
                        model_version_id=str(model.id),
                        model_sha256=model.sha256,
                        model_size_bytes=model.size_bytes,
                        model_path=str(blobs.path_for_read(model.blob)),
                        model_parameters=dict(model.parameters),
                        seconds=round(time.monotonic() - started, 3),
                    )
                    save()
                    print(json.dumps({k: v for k, v in evidence.items() if k != "model_parameters"}), flush=True)
                    return 0
                if status in {"failed", "cancelled", "recovery_required", "awaiting_authorization"}:
                    raise RuntimeError("Actual managed training did not succeed: " + status + ": " + str(current.error))
            time.sleep(5)
        raise TimeoutError("Actual training exceeded its explicit observation deadline; not certified")
    except BaseException as exc:
        evidence.update(
            status="failed", error=f"{type(exc).__name__}: {exc}", seconds=round(time.monotonic() - started, 3)
        )
        save()
        raise
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
