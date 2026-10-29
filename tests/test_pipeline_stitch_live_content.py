"""Actual CPU inference -> same Pipeline layer port -> mosaic; no training."""
import json
from pathlib import Path
import time
from uuid import uuid4

from kraken_manager.domain.pipeline import LayerPipelineDefinition, PipelineModelInput, PipelineOperation
from kraken_manager.domain.pipeline_layer import PipelineLayerContent, PipelineLayerOperation
from kraken_manager.domain.multilayer import MultilayerModel
from kraken_manager.infrastructure.filesystem import FilesystemEventStore
from kraken_manager.infrastructure.mosaic import MosaicReader
from tests.test_stitch_plan_workflow import geometry as geometry


def test_actual_inference_outputs_are_sampled_without_replacing_geometry(geometry, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("NEURALIMAGE_TORCH_COMPILE", "0")
    monkeypatch.delenv("NEURALIMAGE_SERVER_URL", raising=False)
    monkeypatch.delenv("NEURALIMAGE_REMOTE_ONLY", raising=False)
    import torch
    from neuralimage.model.NeuralNetwork import create_model
    from neuralimage.model.NeuralNetwork.model_io import save_model_artifact
    from kraken_core.plugins import PluginInventoryItem, load_plugin_metadata_from_directory
    from kraken_hub.infrastructure.agent.runtime import LocalAgentRuntime
    from kraken_manager.application.dto import AddArtifactVersionCommand, CommandContext
    from kraken_manager.application.use_cases import AddArtifactVersionHandler
    from kraken_manager.domain.artifacts import ArtifactScope
    from kraken_manager.domain.project import RepresentationKind, RepresentationPurpose
    from kraken_manager.infrastructure.plugin.agent_gateway import AgentPluginGateway
    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    local, session, project, layer, sources, arguments, command = geometry
    plan = MultilayerModel.from_dict(command("plan_layer_stitch", arguments)).stitches[-1]
    checkpoint = tmp_path / "untrained-functional-checkpoint.pth"
    save_model_artifact(create_model("S 660k", 1), checkpoint, model_name="S 660k", input_channels=1)
    series = local.create_artifact_series(principal=session.principal, project_id=project.id, layer_id=layer.id,
        scope=ArtifactScope.LAYER_ATTACHMENT, name="Explicit untrained functional model", idempotency_key="checkpoint-series")
    model = AddArtifactVersionHandler(local._uow(str(project.id)), local.profiles, local.clock)(
        AddArtifactVersionCommand(context=CommandContext(actor=session.principal, idempotency_key="checkpoint"),
            project_id=project.id, series_id=series.id, filename=checkpoint.name, media_type="application/octet-stream",
            expected_series_revision=local.artifact_stream_revision(project.id, series.id),
            parameters={"artifact_role": "model", "training_started": False}), (checkpoint.read_bytes(),))
    target = local.create_representation(principal=session.principal, project=local.get_project(project.id),
        layer=local.list_layers(project.id)[0], name="Derived native masks", kind=RepresentationKind.IMAGE,
        purpose=RepresentationPurpose.BINARY, active=False, idempotency_key="mask-representation")
    source_series = local._projection(project.id).get_artifact_series(sources[0].series_id)
    lookup = {str(project.frame_id_at(x, y)): (x, y) for x in range(1, 3) for y in range(1, 3)}
    coords = tuple(lookup[tile.frame_id] for tile in plan.tiles)
    metadata = load_plugin_metadata_from_directory(Path(__file__).resolve().parents[1] / "plugins/neuralimage")
    inference = PipelineOperation(str(uuid4()), "Actual CPU inference, no training", "frames.binary-segment.v2", metadata.version,
        str(source_series.representation_id), str(target.id), "image", "binary_mask", coords,
        parameters={"execution_mode": "headless", "patch_size": 32, "batch_size": 1, "overlap": 0},
        model_input=PipelineModelInput(version_id=str(model.id)))
    mosaic = PipelineLayerOperation(str(uuid4()), "Mosaic of the actual derived masks", "layer.stitch.v1", "1.0.0",
        plan.id, plan.revision, dependencies=(inference.id,), content=PipelineLayerContent("binary_mask", producer_id=inference.id),
        parameters={"block_size": 8})
    definition = LayerPipelineDefinition(str(project.id), str(layer.id), (inference, mosaic))
    revision = FilesystemEventStore(local.catalog_root, project.id).current_revision(f"layer-pipeline:{layer.id}")
    common = dict(project_id=project.id, layer_id=layer.id, principal=session.principal)
    state = local.define_layer_pipeline(**common, definition=definition.to_dict(), expected_revision=revision, idempotency_key="define-mixed")
    state = local.command_layer_pipeline(**common, operation="start", expected_revision=state["revision"], idempotency_key="start-mixed")
    runtime = LocalAgentRuntime(tmp_path / "mixed-agent", (PluginInventoryItem(metadata, installed=False),), builtin_stitching=True)
    try:
        capabilities = runtime.ensure_started()
        gateway = AgentPluginGateway(base_url=runtime.base_url, token=runtime.token, staging_root=runtime.data_dir / "staging",
            capabilities=capabilities, source_for_version=lambda identifier: local.artifact_source_path(project.id, identifier),
            coordinate_for_frame=lambda identifier: lookup[str(identifier)],
            media_type_for_version=lambda identifier: local.get_artifact_version(project.id, identifier).media_type)
        started = time.monotonic()
        while time.monotonic()-started < 80:
            state = local.tick_layer_pipeline(**common, gateway=gateway)
            for job in local.synchronize_plugin_jobs(principal=session.principal, gateway=gateway):
                if gateway.get_job(job.id)["state"] == "importing":
                    local.import_agent_result(principal=session.principal, result_payload=gateway.get_result(job.id),
                                              staging_root=runtime.data_dir / "staging")
                    gateway.complete_import(job.id)
            if state["status"] in {"completed", "failed"}:
                break
            time.sleep(.05)
        assert state["status"] == "completed", json.dumps(state)
        mask_ids = set(state["nodes"][inference.id]["attempts"][-1]["output_version_ids"])
        output_id, = state["nodes"][mosaic.id]["attempts"][-1]["output_version_ids"]
        output = local.get_artifact_version(project.id, output_id)
        assert mask_ids <= set(output.input_version_ids)
        assert {str(v.id) for v in sources} <= set(output.input_version_ids)
        with MosaicReader(local.artifact_source_path(project.id, output.id)) as reader:
            reader.validate(expected_plan=plan)
            assert reader.metadata["sampling"] == "nearest"
            assert {p["artifact_version_id"] for p in reader.metadata["content_inputs"]} == mask_ids
            assert {p.calibration.artifact.version_id for p in reader.plan.tiles} == {str(v.id) for v in sources}
            assert set(reader.read_block(0, 0).pixels.flatten()) <= {0, 255}
        (tmp_path / "receipt.json").write_text(json.dumps({"status": "passed", "training_started": False,
            "model_kind": "explicit untrained checkpoint for execution only", "mask_versions": sorted(mask_ids),
            "mosaic_version": str(output.id), "mosaic_sha256": output.sha256, "seconds": time.monotonic()-started}), encoding="utf-8")
    finally:
        runtime.shutdown()
