"""Public HTTP contracts must survive moving route handlers between modules."""

from fastapi.testclient import TestClient

from kraken_server.app import create_app


def test_transport_objects_are_not_query_parameters():
    schema = create_app(development=True).openapi()
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            for parameter in operation.get("parameters", []):
                assert parameter["name"] not in {"request", "websocket"}, (path, method, parameter)


def test_transfer_and_reorder_routes_remain_registered():
    app = create_app(development=True)
    routes = {(route.path, method) for route in app.routes for method in getattr(route, "methods", ())}
    assert ("/api/v1/projects/{project_id}/layers/{layer_id}/reorder", "POST") in routes
    assert ("/api/v1/projects/{project_id}/artifacts/{series_id}/uploads/complete", "POST") in routes
    assert ("/api/v1/agent/jobs/{job_id}/inputs/{version_id}", "GET") in routes


def test_project_creation_accepts_http_request_injection():
    with TestClient(create_app(development=True)) as client:
        response = client.post(
            "/api/v1/projects",
            headers={"Authorization": "Bearer developer", "Idempotency-Key": "route-regression"},
            json={"name": "Route contract", "width": 1, "height": 1},
        )
    assert response.status_code == 201, response.text
