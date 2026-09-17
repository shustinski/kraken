from __future__ import annotations

from fastapi.testclient import TestClient

from kraken_server.app import create_app
from kraken_server.services import InMemoryServerServices


def test_development_backend_accepts_plugin_job_submit_and_list() -> None:
    client = TestClient(create_app(services=InMemoryServerServices(), development=True))
    headers = {"Authorization": "Bearer developer", "Idempotency-Key": "proj-1"}
    created = client.post(
        "/api/v1/projects",
        headers=headers,
        json={
            "name": "Jobs",
            "width": 2,
            "height": 2,
            "orientation": "y_down",
            "storage_profile_id": "server-postgres",
        },
    )
    assert created.status_code == 201
    project_id = created.json()["project_id"]
    layer_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    target_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"

    submitted = client.post(
        f"/api/v1/projects/{project_id}/jobs",
        headers={"Authorization": "Bearer developer", "Idempotency-Key": "job-1"},
        json={
            "layer_id": layer_id,
            "source_representation_id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
            "target_representation_id": target_id,
            "coordinates": [[1, 1]],
            "capability": "vectorize",
            "parameters": {},
        },
    )
    assert submitted.status_code == 202, submitted.text
    body = submitted.json()
    assert body["state"] == "queued"
    assert body["project_id"] == project_id

    listed = client.get(
        f"/api/v1/projects/{project_id}/jobs",
        headers={"Authorization": "Bearer developer"},
    )
    assert listed.status_code == 200
    assert listed.json()["items"][0]["id"] == body["id"]
