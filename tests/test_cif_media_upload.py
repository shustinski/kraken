"""Actual modern/legacy HTTP uploads preserve Kraken's Caltech CIF type."""
import socket
import threading
import time

import uvicorn

from kraken_hub.remote_client import RemoteAuth, RemoteServerProjectService
from kraken_manager.infrastructure.work_package import media_type_for
from kraken_server.app import SessionPrincipal, create_app
from kraken_server.services import CommandContext
from tests.test_admin_project_workflows import project

pytest_plugins = ["tests.test_admin_project_workflows"]


def test_remote_upload_protocols_assign_caltech_type_independent_of_system_registry(database, tmp_path):
    service, _, actor, _ = database
    pid = project(service, actor, "CIF media contract")["project_id"]
    layer = service.create_layer(pid, {"name": "M1", "type": "metal", "order": 0},
        CommandContext(actor.id, "layer", service.get_project(pid)["revision"]))
    series = service.create_artifact_series(pid, {"name": "CIF evidence", "scope": "layer_attachment", "layer_id": layer["layer_id"]},
        CommandContext(actor.id, "series", None))
    app = create_app(services=service, session_resolver=lambda token: SessionPrincipal(str(actor.id), "local", token) if token == "owner" else None)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    source = tmp_path / "vector.CIF"
    source.write_bytes(b"DS 1 1 1;\nL NM;\n( S 4 4 );\nP 0 0 3 0 3 3;\nDF;\nE\n")
    try:
        deadline = time.monotonic()+5
        while not server.started and time.monotonic() < deadline:
            time.sleep(.01)
        assert server.started
        remote = RemoteServerProjectService(url, auth=RemoteAuth(access_token="owner", principal=actor), data_dir=tmp_path / "client")
        first = remote.add_managed_artifact_version(project_id=pid, series_id=series["id"], principal=actor, source=source, idempotency_key="modern")
        second = remote._add_managed_artifact_version_proxy(project_id=pid, series_id=series["id"], principal=actor,
            source=source, parent_version_id=first.id, idempotency_key="proxy")
        for version in (first, second):
            assert version.media_type == "application/x-cif"
            assert service.projections.get_artifact_version(version.id).media_type == "application/x-cif"
            assert service.uow_factory.blobs.path_for_read(version.blob).read_bytes() == source.read_bytes()
        assert media_type_for("vector.CIF") == "application/x-cif"
        assert media_type_for("vector.CIF", "legacy/type") == "legacy/type"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
