"""Opt-in real Agent/Contour DAG, publication and restart acceptance."""

import hashlib
import json
import os
from pathlib import Path
import time
from uuid import uuid4

import pytest
from PIL import Image, ImageDraw

from kraken_core.plugins import PluginInventoryItem, load_plugin_metadata_from_directory
from kraken_hub.agent_runtime import LocalAgentRuntime
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.domain.pipeline import LayerPipelineDefinition, PipelineOperation, PipelinePolicy, PipelineResources
from kraken_manager.domain.project import RepresentationKind
from kraken_manager.infrastructure.blob.filesystem import FilesystemBlobStore
from kraken_manager.infrastructure.plugin import AgentPluginGateway
from tests.test_work_orders import _service, _project_with_frames


@pytest.mark.skipif(
    os.environ.get("KRAKEN_RUN_PIPELINE_LIVE") != "1", reason="explicit real Agent/Contour execution gate"
)
def test_actual_contour_jobs_follow_pipeline_dependencies_and_publish_exact_versions(tmp_path):
    local = _service(tmp_path)
    owner, _, project, layer, image_rep, vector_rep = _project_with_frames(local, tmp_path / "source")
    source = tmp_path / "actual-images"
    source.mkdir()
    for x in (1, 2):
        image = Image.new("L", (96, 96), 0)
        ImageDraw.Draw(image).rectangle((8 * x, 12, 65, 70), fill=255)
        image.save(source / f"{x}_1.png")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    local.commit_managed_import(
        principal=owner.principal,
        project=local.get_project(project.id),
        layer=local.list_layers(project.id)[0],
        representation=image_rep,
        plan=local.plan_import_directory(project=project, directory=source),
        idempotency_key="actual-images",
    )
    outputs = [vector_rep]
    for index in (2, 3):
        outputs.append(
            local.create_representation(
                principal=owner.principal,
                project=local.get_project(project.id),
                layer=local.list_layers(project.id)[0],
                name=f"CIF {index}",
                kind=RepresentationKind.VECTOR,
                source_image_representation_id=image_rep.id,
                active=False,
                idempotency_key=f"output-{index}",
            )
        )
    root = Path(__file__).resolve().parents[1]
    metadata = load_plugin_metadata_from_directory(root / "plugins/contour")
    a, b, c = (str(uuid4()) for _ in range(3))
    operations = tuple(
        PipelineOperation(
            id=identifier,
            title=f"Contour {index}",
            capability="frames.vectorize.v1",
            algorithm_version=metadata.version,
            source_representation_id=image_rep.id,
            target_representation_id=output.id,
            input_kind="image",
            output_kind="vector",
            coordinates=((1, 1), (2, 1)),
            dependencies=(a, b) if identifier == c else (),
        )
        for index, (identifier, output) in enumerate(zip((a, b, c), outputs, strict=True))
    )
    definition = LayerPipelineDefinition(
        str(project.id),
        str(layer.id),
        operations,
        PipelinePolicy(max_parallel=2, capacity=PipelineResources(cpu=2, memory_mb=512)),
    )
    state = local.define_layer_pipeline(
        project_id=project.id,
        layer_id=layer.id,
        definition=definition.to_dict(),
        principal=owner.principal,
        expected_revision=0,
        idempotency_key="define",
    )
    state = local.command_layer_pipeline(
        project_id=project.id,
        layer_id=layer.id,
        operation="start",
        principal=owner.principal,
        expected_revision=state["revision"],
        idempotency_key="start",
    )
    runtime = LocalAgentRuntime(tmp_path / "agent", (PluginInventoryItem(metadata, installed=False),))
    try:
        capabilities = runtime.ensure_started()
        blobs = FilesystemBlobStore.for_project(local.catalog_root, str(project.id))
        coordinates = {project.frame_id_at(x, 1): (x, 1) for x in (1, 2)}
        gateway = AgentPluginGateway(
            base_url=runtime.base_url,
            token=runtime.token,
            staging_root=tmp_path / "agent/staging",
            capabilities=capabilities,
            source_for_version=lambda identifier: blobs.path_for_read(
                local.artifact_version(project.id, identifier).blob
            ),
            coordinate_for_frame=lambda identifier: coordinates[identifier],
            media_type_for_version=lambda identifier: local.artifact_version(project.id, identifier).media_type,
        )
        deadline = time.monotonic() + 90
        restarted = False
        while time.monotonic() < deadline:
            state = local.tick_layer_pipeline(
                project_id=project.id, layer_id=layer.id, principal=owner.principal, gateway=gateway
            )
            if not restarted and state["nodes"][a]["attempts"]:
                # Re-open the application composition while real jobs are active.
                local = EmbeddedProjectService(local.data_dir, secret_store=local.secrets)
                restarted = True
            for job in local.synchronize_plugin_jobs(principal=owner.principal, gateway=gateway):
                if job.state.value == "importing":
                    imported = local.import_agent_result(
                        principal=owner.principal,
                        result_payload=gateway.get_result(job.id),
                        staging_root=tmp_path / "agent/staging",
                    )
                    assert not imported.requires_partial_confirmation
                    gateway.complete_import(job.id)
            if state["status"] in {"completed", "failed"}:
                break
            time.sleep(0.1)
        assert state["status"] == "completed", json.dumps(state, ensure_ascii=False)
        assert restarted
        assert all(len(node["attempts"]) == 1 for node in state["nodes"].values())
        assert len(local.plugin_jobs()) == 3
        assert all(job.state.value == "succeeded" for job in local.plugin_jobs())
        latest_end = max(state["nodes"][key]["attempts"][-1]["finished_at"] for key in (a, b))
        assert state["nodes"][c]["attempts"][0]["created_at"] >= latest_end
        for node in state["nodes"].values():
            attempt = node["attempts"][0]
            assert len(attempt["output_version_ids"]) == 2
            for identifier in attempt["output_version_ids"]:
                version = local.artifact_version(project.id, identifier)
                assert version.tool_version == metadata.version
                assert version.media_type == "application/x-cif"
                assert version.input_version_ids
                assert blobs.path_for_read(version.blob).read_bytes()
        assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    finally:
        runtime.shutdown()
