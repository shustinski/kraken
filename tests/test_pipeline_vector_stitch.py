"""Actual Contour -> typed layer vector stitching -> canonical mosaic receipt."""
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

from PIL import Image, ImageDraw

from kraken_core.plugins import PluginInventoryItem, load_plugin_metadata_from_directory
from kraken_hub.infrastructure.agent.runtime import LocalAgentRuntime
from kraken_manager.domain.artifacts import ArtifactScope, deterministic_frame_series_id
from kraken_manager.domain.multilayer import MultilayerModel
from kraken_manager.domain.pipeline import LayerPipelineDefinition, PipelineOperation
from kraken_manager.domain.pipeline_layer import PipelineLayerContent, PipelineLayerOperation
from kraken_manager.infrastructure.mosaic import MosaicReader, block_preview
from kraken_manager.infrastructure.plugin.agent_gateway import AgentPluginGateway
from tests.test_work_orders import _service, _project_with_frames


def test_actual_contour_outputs_are_physically_stitched_with_visible_disagreement(tmp_path):
    local = _service(tmp_path)
    owner, _, project, layer, image_rep, vector_rep = _project_with_frames(local, tmp_path / "source")
    actor = owner.principal
    source = tmp_path / "actual-images"
    source.mkdir()
    for x in (1, 2):
        image = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(image).rectangle((8, 12, 55, 48), fill=255)
        image.save(source / f"{x}_1.png")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    local.commit_managed_import(principal=actor, project=local.get_project(project.id), layer=local.list_layers(project.id)[0],
        representation=image_rep, plan=local.plan_import_directory(project=project, directory=source), idempotency_key="actual-images")
    evidence = local.create_artifact_series(principal=actor, project_id=project.id, layer_id=layer.id,
        scope=ArtifactScope.LAYER_ATTACHMENT, name="Controlled stage transform", idempotency_key="evidence-series")
    evidence_path = tmp_path / "geometry.json"
    evidence_path.write_text('{"scope":"functional fixture, not measured stage ground truth"}')
    version = local.add_managed_artifact_version(principal=actor, project_id=project.id, series_id=evidence.id,
        source=evidence_path, idempotency_key="evidence")
    def command(operation, arguments):
        return local.mutate_multilayer_model(project_id=project.id, principal=actor,
            expected_revision=local.get_multilayer_model(project.id)["model"]["revision"],
            operation=operation, arguments=arguments, idempotency_key=str(uuid4()))
    command("set_stack", {"layers": [{"layer_id": str(layer.id), "layer_type": layer.type, "z_m": 0, "thickness_m": 0}]})
    sources = [local._projection(project.id).get_active_artifact_version(deterministic_frame_series_id(image_rep.id,
        project.frame_id_at(x, 1))) for x in (1, 2)]
    tiles = [{"frame_id": str(project.frame_id_at(x, 1)), "artifact_version_id": str(v.id), "width": 64, "height": 64,
        "pixel_size_m": [1e-6, 1e-6], "y_axis": "down", "catalog_xy": [x, 1]} for x, v in zip((1, 2), sources)]
    tiles[1].update(declared_affine=[1, 0, 40e-6, 0, 1, 0], note="Controlled functional pose, not a catalog offset")
    model = MultilayerModel.from_dict(command("plan_layer_stitch", {"layer_id": str(layer.id), "plan_version_id": str(version.id),
        "reference_frame_id": tiles[0]["frame_id"], "tiles": tiles, "max_error_m": 1e-12}))
    plan = model.stitches[-1]
    metadata = load_plugin_metadata_from_directory(Path(__file__).resolve().parents[1] / "plugins/contour")
    contour = PipelineOperation(str(uuid4()), "Actual Contour", "frames.vectorize.v1", metadata.version,
        str(image_rep.id), str(vector_rep.id), "image", "vector", ((1, 1), (2, 1)), parameters={"execution_mode": "headless"})
    stitch = PipelineLayerOperation(str(uuid4()), "Physical vector mosaic", "layer.stitch.v1", "1.0.0", plan.id, plan.revision,
        dependencies=(contour.id,), content=PipelineLayerContent("vector", producer_id=contour.id), parameters={"block_size": 32})
    common = dict(project_id=project.id, layer_id=layer.id, principal=actor)
    definition = LayerPipelineDefinition(str(project.id), str(layer.id), (contour, stitch))
    state = local.define_layer_pipeline(**common, definition=definition.to_dict(), expected_revision=0, idempotency_key="define")
    local.command_layer_pipeline(**common, operation="start", expected_revision=state["revision"], idempotency_key="start")
    runtime = LocalAgentRuntime(tmp_path / "agent", (PluginInventoryItem(metadata, installed=False),), builtin_stitching=True)
    started = time.monotonic()
    try:
        capabilities = runtime.ensure_started()
        coordinates = {str(project.frame_id_at(x, 1)): (x, 1) for x in (1, 2)}
        gateway = AgentPluginGateway(base_url=runtime.base_url, token=runtime.token, staging_root=runtime.data_dir / "staging",
            capabilities=capabilities, source_for_version=lambda v: local.artifact_source_path(project.id, v),
            coordinate_for_frame=lambda frame: coordinates[str(frame)], media_type_for_version=lambda v: local.get_artifact_version(project.id, v).media_type)
        while time.monotonic()-started < 80:
            state = local.tick_layer_pipeline(**common, gateway=gateway)
            for job in local.synchronize_plugin_jobs(principal=actor, gateway=gateway):
                if gateway.get_job(job.id)["state"] == "importing":
                    local.import_agent_result(principal=actor, result_payload=gateway.get_result(job.id), staging_root=runtime.data_dir / "staging")
                    gateway.complete_import(job.id)
            if state["status"] in {"completed", "failed"}:
                break
            time.sleep(.05)
        assert state["status"] == "completed", json.dumps(state)
        output_id, = state["nodes"][stitch.id]["attempts"][-1]["output_version_ids"]
        output = local.get_artifact_version(project.id, output_id)
        vectors = state["nodes"][contour.id]["attempts"][-1]["output_version_ids"]
        assert set(vectors) <= set(output.input_version_ids)
        with MosaicReader(local.artifact_source_path(project.id, output_id)) as reader:
            reader.validate(expected_plan=plan)
            assert reader.metadata["vector_stitch"]["source_features"] >= 2
            assert reader.metadata["vector_stitch"]["conflict_blocks"] > 0
            assert reader.metadata["vector_stitch"]["electrical_connectivity"] == "unverified"
            previews = [block_preview(reader, column, row) for row in range(2) for column in range(4)]
            assert any(v[1].get("vector_stitch") for v in previews)
        from kraken_manager.infrastructure.vector_exchange import VectorExchangeReader
        exported = tmp_path / "vectors.jsonl"
        job_id = state["nodes"][stitch.id]["attempts"][-1]["job_id"]
        before = local.history(project.id)
        local.export_stitch_vectors(project_id=project.id, job_id=job_id, destination=exported)
        with VectorExchangeReader(exported) as exchange:
            exported_blocks = list(exchange.blocks())
            assert exchange.verified and exchange.header["output_version_id"] == output_id
            assert any(block["conflicts"] for block in exported_blocks)
        assert local.history(project.id) == before
        assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
        (tmp_path / "receipt.json").write_text(json.dumps({"status": "passed", "training_started": False,
            "project_id": str(project.id), "data_dir": str(local.data_dir), "vector_versions": vectors,
            "mosaic_version": str(output.id), "mosaic_sha256": output.sha256, "seconds": time.monotonic()-started}))
    finally:
        runtime.shutdown()
