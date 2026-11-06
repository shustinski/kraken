"""Partial-result approval stays with the job's owning backend."""

from types import SimpleNamespace
from uuid import uuid4
import hashlib
import pytest

from fastapi.testclient import TestClient

from kraken_core.plugin_protocol import PluginFrameOutput, PluginResultManifest
from kraken_hub.presentation.qt.parts.gateway_job import GatewayJobMixin
from kraken_manager.application import CommandContext, ImportPluginResultCommand
from kraken_server.app import SessionPrincipal, create_app
from test_plugin_job_application import Fixture
from test_server_workspace_inputs_parts.desktop_job import _desktop_job


@pytest.mark.parametrize("repeat_while_importing", [False, True])
def test_agent_partial_approval_and_import_completion_are_idempotent(tmp_path, repeat_while_importing):
    from kraken_agent.jobs import AgentJobState, DurableJobStore
    from kraken_manager.infrastructure.plugin.agent_gateway import AgentPluginGateway
    from test_agent_jobs import manifest

    store = DurableJobStore(tmp_path / "agent.sqlite3")
    job = store.enqueue(manifest())
    for state in (AgentJobState.STAGING, AgentJobState.RUNNING, AgentJobState.PARTIAL):
        job = store.transition(job.job_id, state, expected_revision=job.revision)
    posts = []

    def request(method, path, payload=None):
        current = store.get(job.job_id)
        if method == "POST":
            posts.append(path)
            target = AgentJobState.IMPORTING if path.endswith("/confirm-partial") else AgentJobState.SUCCEEDED
            current = store.transition(current.job_id, target, expected_revision=payload["expected_revision"])
        return {"state": current.state.value, "revision": current.revision}

    gateway = object.__new__(AgentPluginGateway)
    gateway._request = request
    first = gateway.confirm_partial(job.job_id)
    assert first["state"] == "importing"
    if repeat_while_importing:
        assert gateway.confirm_partial(job.job_id) == first
    completed = gateway.complete_import(job.job_id)
    assert completed["state"] == "succeeded"
    assert gateway.confirm_partial(job.job_id) == completed
    assert gateway.complete_import(job.job_id) == completed
    assert posts == [f"/api/v1/jobs/{job.job_id}/confirm-partial", f"/api/v1/jobs/{job.job_id}/complete-import"]


def test_local_partial_approval_imports_available_outputs(tmp_path):
    fixture = Fixture(frame_count=2)
    job = fixture.submit()
    manifest = fixture.result(job, count=1)
    importer = fixture.importer()
    calls = []
    completed = []

    def import_result(**kwargs):
        calls.append(kwargs.get("confirm_partial", False))
        return importer(
            ImportPluginResultCommand(
                context=CommandContext(actor=fixture.actor, idempotency_key="approve"),
                manifest=manifest,
                confirm_partial=kwargs.get("confirm_partial", False),
            )
        )

    gateway = SimpleNamespace(
        confirm_partial=lambda _id: None,
        get_publications=lambda _id: (),
        get_result=lambda _id: {},
        complete_import=lambda identifier: completed.append(identifier),
    )
    controller = SimpleNamespace(
        service=SimpleNamespace(
            data_dir=tmp_path, is_remote_project=lambda _id: False, import_agent_result=import_result
        ),
        session=SimpleNamespace(principal=fixture.actor),
        _gateway_for_job=lambda _job: gateway,
        _poll_agent_jobs=lambda: None,
        _refresh_current_matrix=lambda: None,
        my_work_refresh=None,
    )
    GatewayJobMixin.import_partial_agent_job(controller, job)
    assert calls == [True]
    assert fixture.reader.reads == 1
    assert completed == [job.id]


def test_remote_partial_approval_never_starts_local_agent():
    calls = []
    job = SimpleNamespace(id="server-job", project_id="server-project")
    controller = SimpleNamespace(
        service=SimpleNamespace(
            is_remote_project=lambda _id: True, confirm_partial_plugin_job=lambda **kwargs: calls.append(kwargs)
        ),
        session=SimpleNamespace(principal="owner"),
        _gateway_for_job=lambda _job: (_ for _ in ()).throw(AssertionError("Local Agent used")),
        _poll_agent_jobs=lambda: None,
        _refresh_current_matrix=lambda: None,
        my_work_refresh=None,
    )
    GatewayJobMixin.import_partial_agent_job(controller, job)
    assert len(calls) == 1
    assert calls[0]["job"] is job


def test_catalog_partial_approval_calls_authenticated_server_client():
    from kraken_hub.composition.catalog import DualCatalogService
    from kraken_hub.infrastructure.remote.parts.submit_plugin_job import SubmitPluginJobMixin

    calls = []
    remote = SimpleNamespace(_call=lambda *args, **kwargs: calls.append((args, kwargs)))
    remote.confirm_partial_plugin_job = lambda **kwargs: SubmitPluginJobMixin.confirm_partial_plugin_job(
        remote, **kwargs
    )
    catalog = object.__new__(DualCatalogService)
    catalog.remote = remote
    catalog._remote_ids = {"project"}
    job = SimpleNamespace(id="job", project_id="project")
    catalog.confirm_partial_plugin_job(principal="owner", job=job, idempotency_key="approval")
    assert calls == [
        (("POST", "/api/v1/projects/project/jobs/job/confirm-partial"), {"payload": {}, "idempotency_key": "approval"})
    ]


@pytest.mark.parametrize("transport_state", ["succeeded", "importing", "unavailable"])
def test_local_service_materializes_partial_only_after_approval(tmp_path, monkeypatch, transport_state):
    _server, local, session, project, _layer, _target, _gateway, job = _desktop_job(tmp_path, "frames.vectorize.v1")
    materialized = []
    monkeypatch.setattr(
        "kraken_hub.composition.parts.import_agent_result.materialize_vector_result",
        lambda *args, **kwargs: materialized.append(kwargs["manifest"]),
    )
    content = b"L M1; E;\n"
    coordinate = next(local.plugin_jobs()[0].selection.iter_coordinates())
    output = PluginFrameOutput(
        output_id=str(uuid4()),
        frame_id=str(coordinate.frame_id(project.id)),
        relative_path="outputs/1_1.cif",
        sha256=hashlib.sha256(content).hexdigest(),
        media_type="application/x-cif",
        role="vector",
    )
    staging = tmp_path / "staging"
    path = staging / job["id"] / output.relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    manifest = PluginResultManifest(
        job_id=job["id"],
        outcome="partial",
        plugin_id="Contour",
        plugin_version="1",
        outputs=(output,),
    )
    arguments = dict(principal=session.principal, result_payload=manifest.to_dict(), staging_root=staging)
    preview = local.import_agent_result(**arguments)
    assert preview.requires_partial_confirmation is True
    assert preview.versions == ()
    assert materialized == []
    awaiting_import = local.synchronize_plugin_jobs(
        principal=session.principal,
        gateway=SimpleNamespace(get_job=lambda _identifier: {"state": "importing"}),
    )
    assert awaiting_import[0].state.value == "importing"
    approved = local.import_agent_result(**arguments, confirm_partial=True)
    assert approved.requires_partial_confirmation is False
    assert len(approved.versions) == 1
    assert len(materialized) == 1
    assert approved.job.state.value == "partial"
    assert approved.job.progress == 0.5

    def get_job(_identifier):
        if transport_state == "unavailable":
            raise OSError("Agent is offline after importing the result")
        return {"state": transport_state, "progress": {"completed_frames": 2, "total_frames": 2}}

    synchronized = local.synchronize_plugin_jobs(principal=session.principal, gateway=SimpleNamespace(get_job=get_job))
    assert len(synchronized) == 1
    assert synchronized[0] == approved.job


def test_server_partial_confirmation_after_agent_lease_ends(tmp_path, monkeypatch):
    server, local, session, project, _layer, _target, gateway, job = _desktop_job(tmp_path, "frames.vectorize.v1")
    materialized = []
    monkeypatch.setattr(
        "kraken_server.persistent_services_parts.import_agent_result.materialize_vector_result",
        lambda *args, **kwargs: materialized.append(kwargs["manifest"]),
    )
    app = create_app(
        services=server,
        agent_gateway=gateway,
        session_resolver=lambda token: SessionPrincipal(
            str(session.principal.id) if token == "owner" else str(uuid4()),
            "local",
            token,
        ),
    )
    client = TestClient(app)
    owner = {"Authorization": "Bearer owner"}
    base = f"/api/v1/projects/{project.id}/jobs/{job['id']}"
    leased = client.post(base + "/desktop-lease", headers=owner)
    assert leased.status_code == 200, leased.text
    worker = {**owner, "X-Kraken-Lease-Claim": leased.json()["job"]["lease_claim"]}
    content = b"L M1; E;\n"
    coordinate = next(local.plugin_jobs()[0].selection.iter_coordinates())
    output = PluginFrameOutput(
        output_id=str(uuid4()),
        frame_id=str(coordinate.frame_id(project.id)),
        relative_path="outputs/1_1.cif",
        sha256=hashlib.sha256(content).hexdigest(),
        media_type="application/x-cif",
        role="vector",
    )
    agent_base = f"/api/v1/agent/jobs/{job['id']}"
    uploaded = client.post(
        agent_base + f"/outputs/{output.output_id}?sha256={output.sha256}", headers=worker, content=content
    )
    assert uploaded.status_code == 200, uploaded.text
    manifest = PluginResultManifest(
        job_id=job["id"], outcome="partial", plugin_id="Contour", plugin_version="1", outputs=(output,)
    )
    assert client.post(agent_base + "/publications", headers=worker, json=manifest.to_dict()).status_code == 200
    finished = client.post(agent_base + "/complete", headers=worker)
    assert finished.status_code == 200, finished.text
    assert finished.json()["imported"]["requires_partial_confirmation"] is True
    assert materialized == []
    endpoint = base + "/confirm-partial"
    assert client.post(endpoint).status_code == 401
    assert client.post(endpoint, headers=owner).status_code == 422
    assert (
        client.post(endpoint, headers={"Authorization": "Bearer other", "Idempotency-Key": "other"}).status_code == 403
    )
    assert (
        client.post(
            endpoint.replace(str(project.id), str(uuid4())), headers={**owner, "Idempotency-Key": "wrong-project"}
        ).status_code
        == 404
    )
    assert (
        client.post(endpoint, headers={**owner, "Idempotency-Key": "viewer", "X-Kraken-Role": "viewer"}).status_code
        == 403
    )
    import sqlalchemy as sa

    with gateway.engine.begin() as connection:
        connection.execute(sa.update(gateway.jobs).where(gateway.jobs.c.job_id == job["id"]).values(state="leased"))
    incomplete = client.post(endpoint, headers={**owner, "Idempotency-Key": "incomplete"})
    assert incomplete.status_code == 409, incomplete.text
    assert materialized == []
    with gateway.engine.begin() as connection:
        connection.execute(sa.update(gateway.jobs).where(gateway.jobs.c.job_id == job["id"]).values(state="completed"))
    approved = client.post(endpoint, headers={**owner, "Idempotency-Key": "approve"})
    assert approved.status_code == 200, approved.text
    assert len(approved.json()["versions"]) == 1
    assert approved.json()["requires_partial_confirmation"] is False
    assert len(materialized) == 1
    repeated = client.post(endpoint, headers={**owner, "Idempotency-Key": "approve"})
    assert repeated.status_code == 200, repeated.text
    assert repeated.json()["already_imported"] is True
    assert [item["id"] for item in repeated.json()["versions"]] == [item["id"] for item in approved.json()["versions"]]
