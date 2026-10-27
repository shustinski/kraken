from dataclasses import replace
import hashlib
from io import BytesIO
import threading
from uuid import uuid4

from PIL import Image
import pytest

from kraken_agent.service import AgentControlServer
from kraken_core.plugin_protocol import PluginJobManifest, PluginJobManifestV2
from kraken_manager.domain.project import FrameCoordinate
from kraken_manager.domain.selection import FrameRowRange, FrameSelectionV1
from kraken_manager.domain.workflows import PluginInputV1, PluginJobManifestV1
from kraken_manager.infrastructure.plugin.agent_gateway import AgentPluginGateway, AgentUnavailable
from kraken_agent.server_worker import ServerAgentWorker
from kraken_agent.runner import PluginRegistry
from kraken_agent.jobs import StagingWorkspace


def domain_manifest(content):
    project = str(uuid4())
    frame = FrameCoordinate(1, 1).frame_id(project)
    return PluginJobManifestV1(
        job_id=str(uuid4()),
        project_id=project,
        layer_id=str(uuid4()),
        target_representation_id=str(uuid4()),
        actor_principal_id=str(uuid4()),
        selection=FrameSelectionV1(row_ranges=(FrameRowRange(1, 1, 1),)),
        capability="transport.probe.v1",
        inputs=(PluginInputV1(frame, str(uuid4()), hashlib.sha256(content).hexdigest(), "inputs/1.png"),),
        parameters={},
    )


@pytest.mark.parametrize("v2", [False, True], ids=["v1", "v2"])
def test_repeated_authoritative_handoff_is_idempotent_on_actual_agent_http(tmp_path, v2):
    encoded = BytesIO()
    Image.new("RGB", (2, 2), "red").save(encoded, format="PNG")
    data = encoded.getvalue()
    source = tmp_path / "source.png"
    source.write_bytes(data)
    value = domain_manifest(data)
    control = AgentControlServer.create(tmp_path / "agent.sqlite3", token="test-token")
    server = control.build_http_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway = AgentPluginGateway(
        base_url=f"http://127.0.0.1:{server.server_address[1]}",
        token="test-token",
        staging_root=tmp_path / "staging",
        source_for_version=lambda _: source,
        coordinate_for_frame=lambda _: (1, 1),
        media_type_for_version=lambda _: "image/png",
        capabilities=frozenset({value.capability}),
        v2_capabilities=frozenset({value.capability}) if v2 else frozenset(),
    )
    try:
        gateway.submit(value)
        first = control.store.get(value.job_id)
        assert isinstance(first.manifest, PluginJobManifestV2 if v2 else PluginJobManifest)
        gateway.submit(value)
        assert control.store.get(value.job_id) == first
        assert (tmp_path / "staging" / value.job_id / "inputs/1.png").read_bytes() == data
        with pytest.raises(AgentUnavailable):
            gateway.submit(replace(value, parameters={"different_request": True}))
        assert control.store.get(value.job_id) == first
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_server_rebuilds_same_transport_manifest_across_lease_claims(tmp_path, monkeypatch):
    data = b"pinned download"
    value = domain_manifest(data)
    worker = ServerAgentWorker(
        server_url="http://127.0.0.1", token="test", data_dir=tmp_path, registry=PluginRegistry()
    )

    def download(url, path, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return "application/octet-stream"

    monkeypatch.setattr(worker, "_download", download)
    first_workspace = StagingWorkspace(tmp_path / "first", value.job_id)
    first_workspace.create()
    first = worker._transport_manifest(value, first_workspace)
    # Preserve a real prior transport timestamp rather than minting another
    # timestamp while rebuilding the same authoritative manifest after recovery.
    first = replace(first, created_at="2026-10-25T00:00:00+00:00")
    worker.store.enqueue(first)
    next_workspace = StagingWorkspace(tmp_path / "next", value.job_id)
    next_workspace.create()
    assert worker._transport_manifest(value, next_workspace).to_json() == first.to_json()
