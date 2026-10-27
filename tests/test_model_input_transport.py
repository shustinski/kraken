from dataclasses import replace
import hashlib
import threading
from uuid import uuid4

import pytest

from kraken_core.model_input import PinnedModelInput
from kraken_agent.jobs import StagingWorkspace
from kraken_agent.runner import PluginRegistry
from kraken_agent.server_worker import ServerAgentWorker
from kraken_agent.service import AgentControlServer
from kraken_manager.infrastructure.plugin.agent_gateway import AgentPluginGateway
from tests.test_agent_handoff_idempotency import domain_manifest


@pytest.mark.parametrize(
    "filename", ["../model.pth", "x/model.pth", "C:model.pth", "NUL.pth", "model.pth.", "x\\model.pth"]
)
def test_pinned_model_rejects_unsafe_filename_before_staging(filename):
    with pytest.raises(ValueError):
        PinnedModelInput.from_parameters(
            {"model_version_id": str(uuid4()), "model_sha256": "a" * 64, "model_filename": filename}
        )


@pytest.mark.parametrize("v2", [False, True], ids=["v1-auxiliary-model", "v2-layer-asset"])
def test_local_agent_stages_pinned_model_as_auxiliary_input_with_actual_http(tmp_path, v2):
    data, weights = b"source transport fixture", b"opaque model transport fixture; no algorithm evaluation"
    source, model_file = tmp_path / "source.bin", tmp_path / "weights.pth"
    source.write_bytes(data)
    model_file.write_bytes(weights)
    model_id = str(uuid4())
    value = domain_manifest(data)
    value = replace(
        value,
        parameters={
            "model_version_id": model_id,
            "model_sha256": hashlib.sha256(weights).hexdigest(),
            "model_filename": model_file.name,
        },
    )
    control = AgentControlServer.create(tmp_path / "jobs.sqlite3", token="test-token")
    server = control.build_http_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway = AgentPluginGateway(
        base_url=f"http://127.0.0.1:{server.server_address[1]}",
        token="test-token",
        staging_root=tmp_path / "staging",
        source_for_version=lambda identifier: model_file if str(identifier) == model_id else source,
        coordinate_for_frame=lambda _: (1, 1),
        media_type_for_version=lambda _: "application/octet-stream",
        capabilities=frozenset({value.capability}),
        v2_capabilities=frozenset({value.capability}) if v2 else frozenset(),
    )
    try:
        gateway.submit(value)
        transport = control.store.get(value.job_id).manifest
        assert transport.parameters["model_relative_path"] == "inputs/model/weights.pth"
        assert (tmp_path / "staging" / value.job_id / "inputs/model/weights.pth").read_bytes() == weights
        assert "model_source_path" not in transport.parameters
        if v2:
            models = [item for item in transport.inputs if item.role == "model"]
            assert len(models) == 1 and models[0].scope == "layer" and models[0].asset_id == model_id
            assert models[0].sha256 == hashlib.sha256(weights).hexdigest()
        else:
            assert len(transport.inputs) == 1 and transport.inputs[0].artifact_version_id != model_id
        gateway.submit(value)
        assert control.store.get(value.job_id).manifest.to_json() == transport.to_json()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_server_worker_downloads_exact_pinned_model_and_rejects_damaged_cache(tmp_path, monkeypatch):
    from kraken_core.plugin_protocol import PluginFrameInput, PluginJobManifest

    model_id = str(uuid4())
    weights = b"opaque model bytes"
    value = domain_manifest(b"data")
    transport = PluginJobManifest(
        job_id=value.job_id,
        operation=value.capability,
        project_id=value.project_id,
        layer_id=value.layer_id,
        actor_id=value.actor_principal_id,
        target_representation_id=value.target_representation_id,
        inputs=(
            PluginFrameInput(
                "frame", 1, 1, "version", hashlib.sha256(b"data").hexdigest(), "image/png", "inputs/1.png"
            ),
        ),
        parameters={
            "model_version_id": model_id,
            "model_sha256": hashlib.sha256(weights).hexdigest(),
            "model_filename": "weights.pth",
        },
    )
    worker = ServerAgentWorker(
        server_url="http://127.0.0.1", token="test", data_dir=tmp_path, registry=PluginRegistry()
    )
    workspace = StagingWorkspace(tmp_path / "staged", value.job_id)
    workspace.create()
    paths = []

    def download(url, destination, *, expected_sha256, staging_root):
        assert expected_sha256 == hashlib.sha256(weights).hexdigest() and staging_root == workspace.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(weights)
        paths.append(url)
        return "application/x-pytorch"

    monkeypatch.setattr(worker, "_download", download)
    staged = worker._stage_model_input(transport, workspace)
    assert paths == [f"/api/v1/agent/jobs/{value.job_id}/inputs/{model_id}"]
    assert staged.parameters["model_relative_path"] == "inputs/model/weights.pth"
    assert worker._stage_model_input(transport, workspace) == staged
    assert len(paths) == 1
    (workspace.path / "inputs/model/weights.pth").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="does not match"):
        worker._stage_model_input(transport, workspace)
