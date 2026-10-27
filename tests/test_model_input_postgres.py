"""Real leased model-download authorization; no inference or model-quality claims."""

from dataclasses import replace
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from kraken_manager.domain.selection import FrameSelectionV1
from kraken_manager.domain.workflows import PluginInputV1, PluginJobManifestV1
from kraken_server.agent_auth import PostgresAgentTokenStore
from kraken_server.agent_gateway import PostgresAgentGateway
from kraken_server.app import create_app
from kraken_server.services import CommandContext
from tests.test_admin_project_workflows import project
from tests.test_agent_queue_admission import expire

pytest_plugins = ["tests.test_admin_project_workflows"]


def setup_model(database, *, role="model"):
    services, _accounts, owner, _config = database
    created = project(services, owner, "Model input transport")
    pid = created["project_id"]
    layer = services.create_layer(
        pid,
        {"name": "M1", "type": "metal", "order": 0},
        CommandContext(str(owner.id), str(uuid4()), services.get_project(pid)["revision"]),
    )
    series = services.create_artifact_series(
        pid,
        {"scope": "layer_attachment", "layer_id": layer["layer_id"], "name": "Model"},
        CommandContext(str(owner.id), str(uuid4()), None),
    )
    content = b"opaque pinned model transfer bytes; no inference evaluation"
    version = services.add_managed_artifact_version(
        pid,
        series["id"],
        {"filename": "weights.pth", "media_type": "application/x-pytorch", "parameters": {"artifact_role": role}},
        (content,),
        CommandContext(str(owner.id), str(uuid4()), services.artifact_stream_revision(pid, series["id"])),
    )
    assert version["parameters"] == {"artifact_role": role}
    value = PluginJobManifestV1(
        job_id=str(uuid4()),
        project_id=pid,
        layer_id=layer["layer_id"],
        target_representation_id=str(uuid4()),
        actor_principal_id=str(owner.id),
        selection=FrameSelectionV1.rectangle(1, 1, 1, 1),
        capability="frames.binary-segment.v2",
        inputs=(PluginInputV1(str(uuid4()), str(uuid4()), "a" * 64, "inputs/1.png"),),
        parameters={
            "model_version_id": version["id"],
            "model_sha256": version["sha256"],
            "model_filename": version["filename"],
        },
    )
    queue = PostgresAgentGateway(services.engine, None, services.uow_factory.blobs)
    tokens = PostgresAgentTokenStore(services.engine)
    _, token = tokens.create("model-worker", frozenset({value.capability}))
    client = TestClient(
        create_app(services=services, session_resolver=lambda _: None, agent_token_store=tokens, agent_gateway=queue)
    )
    return services, queue, client, value, version, content, {"Authorization": f"Bearer {token}"}


def leased(client, headers):
    response = client.post("/api/v1/agent/lease", headers=headers, json={})
    assert response.status_code == 200, response.text
    result = response.json()["job"]
    assert result is not None
    return {**headers, "X-Kraken-Lease-Claim": result["lease_claim"]}


def test_exact_model_download_and_ticket_are_claim_fenced_with_real_postgres(database):
    _services, queue, client, value, model, content, authorization = setup_model(database)
    queue.submit(value)
    old = leased(client, authorization)
    base = f"/api/v1/agent/jobs/{value.job_id}/inputs/{model['id']}"
    response = client.get(base, headers=old)
    assert response.status_code == 200 and response.content == content
    ticket = client.get(base + "/download-ticket", headers=old)
    assert ticket.status_code == 200 and ticket.json() == {"mode": "proxy", "media_type": "application/x-pytorch"}
    unknown = f"/api/v1/agent/jobs/{value.job_id}/inputs/{uuid4()}"
    assert client.get(unknown, headers=old).status_code == 403
    expire(queue, value.job_id)
    current = leased(client, authorization)
    assert current != old
    assert client.get(base, headers=old).status_code == 403
    assert client.get(base + "/download-ticket", headers=old).status_code == 403
    assert client.get(base, headers=current).content == content


@pytest.mark.parametrize("case", ["digest", "filename", "role", "layer", "project"])
def test_leased_model_read_rechecks_authoritative_scope_role_and_metadata(database, case):
    services, queue, client, value, model, _content, authorization = setup_model(
        database, role="document" if case == "role" else "model"
    )
    changes = dict(value.parameters)
    if case == "digest":
        changes["model_sha256"] = "f" * 64
    elif case == "filename":
        changes["model_filename"] = "other.pth"
    elif case == "layer":
        value = replace(value, layer_id=str(uuid4()))
    elif case == "project":
        other = project(services, database[2], "Other project")
        value = replace(value, project_id=other["project_id"])
    value = replace(value, parameters=changes)
    queue.submit(value)
    headers = leased(client, authorization)
    base = f"/api/v1/agent/jobs/{value.job_id}/inputs/{model['id']}"
    for path in (base, base + "/download-ticket"):
        response = client.get(path, headers=headers)
        assert response.status_code in {403, 404}, response.text
        assert "opaque pinned" not in response.text
