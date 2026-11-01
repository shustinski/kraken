"""Live dual catalog, websocket outbox and deletion request on isolated PostgreSQL."""
import socket
import threading

import uvicorn

from kraken_hub.composition.catalog import DualCatalogService
from kraken_hub.infrastructure.remote.client import RemoteAuth, RemoteServerProjectService
from kraken_manager.domain.project import GridOrientation
from kraken_server.app import SessionPrincipal, create_app
from kraken_server.outbox import ConnectionHub, OutboxPublisher
from kraken_server.project_deletion import DeletionRequests
from tests.test_admin_project_workflows import project
from tests.test_work_orders import _service

pytest_plugins = ["tests.test_admin_project_workflows"]


def test_live_dual_catalog_receives_ws_and_requests_deletion_without_deleting(database, tmp_path, qtbot):
    services, _, owner, _ = database
    remote_id = project(services, owner, "Live remote")["project_id"]
    hub = ConnectionHub()
    publisher = OutboxPublisher(services.engine, hub, interval_seconds=0.05)
    app = create_app(services=services, session_resolver=lambda token: (
        SessionPrincipal(str(owner.id), "local", "admin") if token == "owner" else None
    ), connection_hub=hub, outbox_publisher=publisher)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    local = _service(tmp_path / "local")
    local_owner = local.create_initial_account("local", "Local", "")
    local_project = local.create_project(principal=local_owner.principal, name="Live local", width=1, height=1,
                                         orientation=GridOrientation.Y_DOWN, idempotency_key="local")
    remote = RemoteServerProjectService(url, auth=RemoteAuth(access_token="owner", principal=owner),
                                        data_dir=tmp_path / "http-client")
    dual = None
    try:
        qtbot.waitUntil(lambda: server.started, timeout=5000)
        dual = DualCatalogService(local, remote)
        projects = dual.list_projects()
        assert {p.name for p in projects} == {"Live local", "Live remote"}
        assert dual.is_remote_project(remote_id)
        assert not dual.is_remote_project(local_project.id)
        wakes, catalogs = [], []
        dual.add_wake_handler(wakes.append)
        dual.add_catalog_handler(lambda: catalogs.append(True))
        dual.subscribe_project(remote_id)

        def subscribed():
            with hub._lock:
                return bool(hub._catalog_subs) and any(remote_id in ids for ids in hub._project_subs.values())

        qtbot.waitUntil(subscribed, timeout=10000)
        updated = dual.rename_project(principal=owner, project=dual.get_project(remote_id),
                                      name="Remote renamed via Hub", idempotency_key="rename")
        qtbot.waitUntil(lambda: any(str(w.project_id) == remote_id for w in wakes), timeout=10000)
        qtbot.waitUntil(lambda: bool(catalogs), timeout=10000)
        assert local.get_project(local_project.id).name == "Live local"
        assert remote.get_project(remote_id).name == "Remote renamed via Hub"
        request = dual.request_project_deletion(project=updated, reason="Isolated acceptance")
        assert request["state"] == "pending"
        assert services.get_project(remote_id)["name"] == "Remote renamed via Hub"
        assert len(DeletionRequests(services.engine).list(remote_id)) == 1
    finally:
        if dual is not None:
            dual.stop_sync()
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive()
