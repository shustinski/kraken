from __future__ import annotations

import pytest


pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kraken_server.app import SessionPrincipal, create_app
from kraken_server.agent_auth import AgentIdentity
from kraken_server.services import ConflictError, InMemoryServerServices


def test_development_api_accepts_large_sparse_project_and_frames_resource() -> None:
    client = TestClient(create_app(development=True))
    headers = {"Authorization": "Bearer developer", "Idempotency-Key": "large-grid"}
    created = client.post(
        "/api/v1/projects",
        headers=headers,
        json={
            "name": "Large sparse grid",
            "width": 2_000_000_000,
            "height": 2_000_000_000,
            "orientation": "y_down",
            "storage_profile_id": "server-postgres",
        },
    )
    assert created.status_code == 201
    project_id = created.json()["project_id"]

    frames = client.get(
        f"/api/v1/projects/{project_id}/frames",
        headers={"Authorization": "Bearer developer"},
        params={"layer_id": "layer", "representation_id": "representation"},
    )
    assert frames.status_code == 200
    assert frames.json()["items"] == []


def test_analysis_and_pipeline_events_are_revisioned_server_resources() -> None:
    client = TestClient(create_app(development=True))
    authorization = {"Authorization": "Bearer developer"}
    created = client.post(
        "/api/v1/projects",
        headers={**authorization, "Idempotency-Key": "project"},
        json={"name": "Shared", "width": 10, "height": 10, "orientation": "y_down"},
    ).json()
    project_id = created["project_id"]
    layer = client.post(
        f"/api/v1/projects/{project_id}/layers",
        headers={
            **authorization,
            "Idempotency-Key": "layer",
            "If-Match": "0",
        },
        json={"name": "Metal", "type": "metal", "order": 1},
    ).json()
    layer_id = layer["layer_id"]

    analysis = client.post(
        f"/api/v1/projects/{project_id}/analyses/karakal",
        headers={
            **authorization,
            "Idempotency-Key": "analysis",
            "If-Match": "0",
        },
        json={
            "layer_id": layer_id,
            "frame_confidence": {"frame": 0.75},
            "report": {},
            "parameters": {},
            "plugin_version": "1.2.3",
        },
    )
    assert analysis.status_code == 201
    assert analysis.json()["revision"] == 1

    pipeline = client.post(
        f"/api/v1/projects/{project_id}/pipeline-actions",
        headers={
            **authorization,
            "Idempotency-Key": "pipeline",
            "If-Match": "0",
        },
        json={
            "event_type": "LayerPipelineActionRequested",
            "layer_id": layer_id,
            "action": "vectorize",
            "node_id": "node",
            "plugin_id": "contour",
            "capability": "vectorize",
            "mode": "server",
            "parameters": {},
        },
    )
    assert pipeline.status_code == 201
    assert pipeline.json()["revision"] == 1

    history = client.get(
        f"/api/v1/projects/{project_id}/history", headers=authorization
    ).json()["items"]
    assert {item["event_type"] for item in history} >= {
        "KarakalAnalysisPublished",
        "LayerPipelineActionRequested",
    }


class _ConflictServices(InMemoryServerServices):
    def rename_project(self, project_id, name, context):
        del project_id, name, context
        raise ConflictError("Expected project revision 3, found 4")


def test_revision_conflict_problem_identifies_entity_and_actual_revision() -> None:
    app = create_app(
        services=_ConflictServices(),
        session_resolver=lambda token: SessionPrincipal("user", "gitlab", token),
        live_gitlab_verifier=lambda _session: True,
    )
    response = TestClient(app).patch(
        "/api/v1/projects/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        headers={
            "Authorization": "Bearer user-token",
            "Idempotency-Key": "rename",
            "If-Match": "3",
        },
        json={"name": "Concurrent"},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "revision.conflict"
    assert response.json()["entity_kind"] == "project"
    assert response.json()["entity_id"] == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert response.json()["actual_revision"] == 4


class _AgentTokens:
    def resolve(self, token):
        if token == "agent-token":
            return AgentIdentity(
                "agent", "worker", frozenset(("vectorize",))
            )
        return None


class _AgentGateway:
    def lease(self, agent, *, seconds):
        del agent, seconds
        return None


def test_user_and_agent_tokens_are_not_interchangeable() -> None:
    app = create_app(
        services=InMemoryServerServices(),
        session_resolver=lambda token: (
            SessionPrincipal("user", "local", token) if token == "user-token" else None
        ),
        agent_token_store=_AgentTokens(),
        agent_gateway=_AgentGateway(),
    )
    client = TestClient(app)

    assert client.get(
        "/api/v1/projects", headers={"Authorization": "Bearer agent-token"}
    ).status_code == 401
    assert client.post(
        "/api/v1/agent/lease",
        headers={"Authorization": "Bearer user-token"},
        json={"capabilities": ["vectorize"]},
    ).status_code == 401
    assert client.post(
        "/api/v1/agent/lease",
        headers={"Authorization": "Bearer agent-token"},
        json={"capabilities": ["vectorize"]},
    ).status_code == 200


def test_session_endpoint_returns_authenticated_principal() -> None:
    principal_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    app = create_app(
        services=InMemoryServerServices(),
        session_resolver=lambda token: (
            SessionPrincipal(principal_id, "gitlab", token)
            if token == "user-token"
            else None
        ),
    )
    response = TestClient(app).get(
        "/api/v1/session", headers={"Authorization": "Bearer user-token"}
    )
    assert response.status_code == 200
    assert response.json()["principal_id"] == principal_id
    assert response.json()["provider"] == "gitlab"
