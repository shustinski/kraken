"""test_server_workspace_inputs: test server recognition stages."""
from __future__ import annotations
from types import SimpleNamespace
import pytest
from PIL import Image
from kraken_hub.presentation.qt.manager_app import DesktopController

from test_server_workspace_inputs_parts.desktop_job import _desktop_job

def test_server_task_cancellation_does_not_start_local_agent():
    controller = object.__new__(DesktopController)
    calls = []
    controller.service = SimpleNamespace(
        is_remote_project=lambda _id: True, cancel_plugin_job=lambda **kwargs: calls.append(kwargs)
    )
    controller.session = SimpleNamespace(principal="owner")
    controller.my_work_refresh = None
    controller._gateway_for_job = lambda _job: pytest.fail("Server cancellation must not start the local Agent")
    job = SimpleNamespace(project_id="server")
    controller.cancel_agent_job(job)
    assert calls[0]["gateway"] is None
    assert calls[0]["job"] is job

@pytest.mark.parametrize("capability", ["frames.binary-segment.v2", "frames.confidence.v1"])
def test_server_recognition_stages_model_with_its_original_extension(tmp_path, capability):
    import hashlib
    from kraken_agent.jobs import StagingWorkspace
    from kraken_agent.server_worker import ServerAgentWorker
    from kraken_agent.runner import PluginRegistry
    from kraken_manager.infrastructure.filesystem._codec import decode_model
    from kraken_manager.domain.workflows import PluginJobManifestV1
    from neuralimage.kraken_bridge import HeadlessOptions, NeuralImageKrakenSession

    _server, _local, _session, _project, _layer, _target, gateway, payload = _desktop_job(
        tmp_path, "frames.dataset.prepare.v1"
    )
    from kraken_server.agent_auth import AgentIdentity

    lease = gateway.lease(
        AgentIdentity(f"desktop:{payload['actor_principal_id']}", "desktop", frozenset({"frames.dataset.prepare.v1"})),
        job_id=payload["id"],
    )
    raw = lease["manifest"]
    model = tmp_path / "cached-model-without-extension"
    model.write_bytes(b"test-weights")
    raw["capability"] = capability
    raw["parameters"] = {
        "execution_host": "desktop",
        "model_source_path": str(model),
        "model_filename": "weights.pth",
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    }
    manifest = decode_model(PluginJobManifestV1, raw)
    workspace = StagingWorkspace(tmp_path / "staging", payload["id"])
    workspace.create()
    worker = ServerAgentWorker(
        server_url="http://localhost", token="owner", data_dir=tmp_path / "worker", registry=PluginRegistry({})
    )

    def download(_endpoint, destination, **_kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12), "red").save(destination)
        return "image/jpeg"

    worker._download = download
    transport = worker._transport_manifest(manifest, workspace)
    assert transport.parameters["model_relative_path"] == "inputs/model/weights.pth"
    assert (workspace.path / "inputs/model/weights.pth").read_bytes() == model.read_bytes()
    # The normal recognition bridge accepts the staged, pinned model.
    session = NeuralImageKrakenSession(
        transport, workspace.path, workspace.path / "result.json", (), (), workspace.path / "outputs"
    )
    options = HeadlessOptions.from_session(session)
    assert options.model_path.suffix == ".pth"

@pytest.mark.parametrize("media_type", [None, "application/octet-stream", "image/jpeg"])
def test_server_stream_manifest_infers_image_type_without_staging_inputs(tmp_path, monkeypatch, media_type):
    from kraken_agent.jobs import StagingWorkspace
    from kraken_agent.runner import PluginRegistry
    from kraken_agent.server_worker import ServerAgentWorker
    from kraken_manager.domain.workflows import PluginJobManifestV1
    from kraken_manager.infrastructure.filesystem._codec import decode_model
    from kraken_server.agent_auth import AgentIdentity

    _server, _local, _session, _project, _layer, _target, gateway, payload = _desktop_job(
        tmp_path, "frames.dataset.prepare.v1"
    )
    leased = gateway.lease(
        AgentIdentity(f"desktop:{payload['actor_principal_id']}", "desktop", frozenset({"frames.dataset.prepare.v1"})),
        job_id=payload["id"],
    )
    raw = leased["manifest"]
    raw["parameters"] = {"input_transport": "server"}
    if media_type is not None:
        raw["parameters"]["input_media_types"] = {item["artifact_version_id"]: media_type for item in raw["inputs"]}
    manifest = decode_model(PluginJobManifestV1, raw)
    workspace = StagingWorkspace(tmp_path / "staging", payload["id"])
    workspace.create()
    worker = ServerAgentWorker(
        server_url="http://localhost", token="owner", data_dir=tmp_path / "worker", registry=PluginRegistry({})
    )
    monkeypatch.setattr(worker, "_request", lambda *_args: pytest.fail("Image MIME caused an extra HTTP request"))
    monkeypatch.setattr(worker, "_download", lambda *_args, **_kwargs: pytest.fail("Image staged as a file"))
    transport = worker._transport_manifest(manifest, workspace)
    assert transport.inputs
    assert all(item.media_type == "image/jpeg" for item in transport.inputs)
    assert all(not (workspace.path / item.relative_path).exists() for item in transport.inputs)
    assert transport.parameters["server_url"] == "http://localhost"
