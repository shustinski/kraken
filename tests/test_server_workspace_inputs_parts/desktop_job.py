"""test_server_workspace_inputs: desktop job."""
from __future__ import annotations
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from kraken_hub.manager_app import DesktopController
from kraken_manager.application.plugin_jobs import SubmitPluginJobHandler
from kraken_manager.domain.project import RepresentationKind
from kraken_server.app import SessionPrincipal, create_app
from kraken_server.services import CommandContext

from test_server_workspace_inputs_parts.server_layer import _server_layer

def test_partial_server_inputs_are_completed_before_launch(qtbot):
    from PyQt6.QtWidgets import QMainWindow

    cells = [SimpleNamespace(x=1, y=1)]
    layer = SimpleNamespace(id="layer")
    representation = SimpleNamespace(id="source")

    def prepare(**_kwargs):
        cells.append(SimpleNamespace(x=1, y=2))
        return tuple(cells)

    controller = object.__new__(DesktopController)
    controller._project_id = "project"
    controller.session = SimpleNamespace(principal=object())
    controller.shell = QMainWindow()
    qtbot.addWidget(controller.shell)
    controller._background_workers = set()
    controller._source_preparation_keys = set()
    controller.service = SimpleNamespace(
        get_project=lambda _id: SimpleNamespace(id="project"),
        frame_cells=lambda *_args: tuple(cells),
        layer_file_binding=lambda *_args: SimpleNamespace(frame_positions={"0000.jpg": 1, "0002.jpg": 3}),
        list_layers=lambda _id: (layer,),
        list_representations=lambda *_args: (representation,),
        materialize_representation_inputs=prepare,
    )
    retried = []
    assert (
        controller._ensure_agent_source_ready(
            layer_id="layer",
            source_representation_id="source",
            retry=lambda: retried.append(True),
        )
        is False
    )
    qtbot.waitUntil(lambda: bool(retried), timeout=3000)
    qtbot.waitUntil(lambda: not controller._background_workers, timeout=3000)
    assert len(cells) == 2

def _desktop_job(tmp_path, capability):
    import sqlalchemy as sa
    from kraken_manager.infrastructure.blob.filesystem import FilesystemBlobStore
    from kraken_server.agent_gateway import PostgresAgentGateway

    server, local, session, project, layer, _binding, source, context = _server_layer(tmp_path)
    server.materialize_representation_inputs(str(project.id), str(layer.id), str(source.id), context)
    target = (
        local.create_representation(
            principal=session.principal,
            project=project,
            layer=layer,
            name="Contour output",
            kind=RepresentationKind.VECTOR,
            source_image_representation_id=source.id,
            idempotency_key="target",
        )
        if capability == "frames.vectorize.v1"
        else source
    )
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'queue.sqlite3'}")
    blobs = FilesystemBlobStore.for_project(local.catalog_root, str(project.id))
    gateway = PostgresAgentGateway(engine, tokens=None, blobs=blobs)
    gateway.metadata.create_all(engine)
    server.uow_factory = local._uow(str(project.id))
    server.uow_factory.blobs = blobs
    server.profiles = local.profiles
    server.identities = local.identities
    server.project_ids_for_principal = lambda principal_id: (
        (str(project.id),) if local.project_roles(project.id, principal_id) else ()
    )
    server.agent_gateway = gateway
    server._submit_plugin_job = SubmitPluginJobHandler(server.uow_factory, local.profiles, local.clock, gateway)
    job = server.submit_plugin_job(
        str(project.id),
        {
            "layer_id": str(layer.id),
            "source_representation_id": str(source.id),
            "target_representation_id": str(target.id),
            "coordinates": [[1, 1], [1, 2]],
            "capability": capability,
            "parameters": {"execution_host": "desktop"},
        },
        CommandContext(str(session.principal.id), "desktop-submit", None),
    )
    return server, local, session, project, layer, target, gateway, job

@pytest.mark.parametrize("acting_role", [None, "maintainer"])
def test_desktop_lease_is_limited_to_its_initiator_and_job(tmp_path, acting_role):
    from uuid import uuid4
    from kraken_server.agent_auth import AgentIdentity

    server, _local, session, project, _layer, _target, gateway, job = _desktop_job(tmp_path, "frames.vectorize.v1")
    app = create_app(
        services=server,
        session_resolver=lambda token: SessionPrincipal(
            str(session.principal.id) if token == "owner" else str(uuid4()), "local", token
        ),
        agent_gateway=gateway,
    )
    client = TestClient(app)
    endpoint = f"/api/v1/projects/{project.id}/jobs/{job['id']}/desktop-lease"
    assert client.post(endpoint).status_code == 401
    assert client.post(endpoint, headers={"Authorization": "Bearer other"}).status_code == 403
    assert gateway.lease(AgentIdentity("machine", "server", frozenset({"frames.vectorize.v1"}))) is None
    headers = {"Authorization": "Bearer owner"}
    if acting_role is not None:
        headers["X-Kraken-Role"] = acting_role
    assert client.post(endpoint, headers={**headers, "X-Kraken-Role": "viewer"}).status_code == 403
    assert client.post(endpoint.replace(str(project.id), str(uuid4())), headers=headers).status_code == (
        403 if acting_role else 404
    )
    response = client.post(endpoint, headers=headers)
    assert response.status_code == 200
    assert response.json()["job"]["job_id"] == job["id"]
    assert client.post(endpoint, headers=headers).status_code == 409
    assert client.post("/api/v1/agent/lease", headers=headers, json={}).status_code == 401
    assert client.post(f"/api/v1/agent/jobs/{job['id']}/heartbeat", headers=headers, json={}).status_code == 200
    from kraken_manager.application.plugin_jobs import CancelPluginJobHandler

    server._cancel_plugin_job = CancelPluginJobHandler(server.uow_factory, server.profiles, _local.clock, gateway)
    current = _local.plugin_jobs()[0]
    assert current.state.value == "waiting_for_user"
    cancelled = client.post(
        f"/api/v1/projects/{project.id}/jobs/{job['id']}/cancel",
        json={},
        headers={**headers, "Idempotency-Key": "cancel", "If-Match": str(current.revision)},
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "cancelled"
    assert client.post(f"/api/v1/agent/jobs/{job['id']}/heartbeat", headers=headers, json={}).status_code == 403


def test_agent_queue_cancel_ignores_a_missing_job(tmp_path):
    import sqlalchemy as sa
    from kraken_server.agent_gateway import PostgresAgentGateway

    engine = sa.create_engine(f"sqlite:///{tmp_path / 'queue.sqlite3'}")
    gateway = PostgresAgentGateway(engine, tokens=None, blobs=None)
    gateway.metadata.create_all(engine)
    gateway.cancel("11111111-1111-1111-1111-111111111111")


def test_cancel_succeeds_when_the_agent_queue_is_already_finished(tmp_path):
    import sqlalchemy as sa

    server, local, session, project, _layer, _target, gateway, job = _desktop_job(tmp_path, "frames.vectorize.v1")
    app = create_app(
        services=server,
        session_resolver=lambda token: SessionPrincipal(str(session.principal.id), "local", token),
        agent_gateway=gateway,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer owner"}
    leased = client.post(
        f"/api/v1/projects/{project.id}/jobs/{job['id']}/desktop-lease",
        headers=headers,
    )
    assert leased.status_code == 200
    with gateway.engine.begin() as connection:
        connection.execute(
            sa.update(gateway.jobs)
            .where(gateway.jobs.c.job_id == job["id"])
            .values(state="completed", lease_owner=None, lease_until=None)
        )
    from kraken_manager.application.plugin_jobs import CancelPluginJobHandler

    server._cancel_plugin_job = CancelPluginJobHandler(server.uow_factory, server.profiles, local.clock, gateway)
    current = local.plugin_jobs()[0]
    cancelled = client.post(
        f"/api/v1/projects/{project.id}/jobs/{job['id']}/cancel",
        json={},
        headers={**headers, "Idempotency-Key": "cancel-finished", "If-Match": str(current.revision)},
    )
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["state"] == "cancelled"
