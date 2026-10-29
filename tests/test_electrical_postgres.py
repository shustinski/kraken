"""Actual electrical artifact/evidence path in owned PostgreSQL UUID schemas."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import sqlalchemy as sa

from kraken_admin.infrastructure.composition import open_project_services
from kraken_manager.infrastructure.postgres.event_store import PostgresEventStore
from kraken_server.services import CommandContext, ConflictError
from tests.test_electrical_graph import png
from tests.test_multilayer_postgres import setup_physical

pytest_plugins = ["tests.test_admin_project_workflows"]


def setup_electrical(database):
    services, owner, identifier, config, layers, observations, _stack, physical_command, _initial = setup_physical(database)
    image = png([[0] * 16 for _ in range(5)] + [[0] + [255] * 14 + [0]] + [[0] * 16 for _ in range(10)])
    masks = []
    for index in (0, 2):
        series = services.create_artifact_series(identifier, {"scope": "layer_attachment", "name": f"Binary mask {index}",
            "layer_id": layers[index]}, CommandContext(str(owner.id), f"binary-series-{index}", None))
        version = services.add_managed_artifact_version(identifier, series["id"], {
            "filename": f"binary-mask-{index}.png", "media_type": "image/png"}, (image,),
            CommandContext(str(owner.id), f"binary-mask-{index}", services.artifact_stream_revision(identifier, series["id"])))
        masks.append(version)
        physical_command("calibrate", {"layer_id": layers[index], "artifact_version_id": version["id"],
            "pixel_size_m": [1e-6, 1e-6], "y_axis": "up", "uncertainty_m": 1e-9}, f"binary-calibration-{index}")
    physical = physical_command("register_affine", {"source_layer_id": layers[0], "target_layer_id": layers[2],
        "max_error_m": 1e-9, "control_points": [{"source_xy": p, "target_xy": p, "uncertainty_m": 1e-9}
                                               for p in [(0, 0), (10, 0), (0, 10), (10, 10)]]}, "binary-register")
    reg = physical["model"]["registrations"][-1]
    proposed = physical_command("propose_contact", {"source_layer_id": layers[0], "target_layer_id": layers[2],
        "contact_layer_id": layers[1], "registration_id": reg["id"], "registration_revision": reg["revision"],
        "position_target_m": [5e-6, 5e-6], "radius_m": 1e-8}, "binary-contact")
    contact = proposed["model"]["contacts"][-1]
    physical_command("review_contact", {"contact_id": contact["id"], "decision": "confirmed",
        "evidence_version_ids": [observations[1]["id"]], "review_note": "Independent synthetic contact-layer observation"}, "binary-contact-reviewed")

    def command(operation, arguments, key):
        snapshot = services.get_electrical_model(identifier)
        return services.transport_mutate_electrical_model(identifier, {
            "operation": operation, "arguments": arguments, "expected_physical_revision": snapshot["physical_model_revision"]},
            CommandContext(str(owner.id), key, snapshot["electrical"]["revision"]))

    components = []
    for i, index in enumerate((0, 2)):
        snapshot = command("extract_conductors", {"layer_id": layers[index], "mask_version_id": masks[i]["id"],
            "connectivity": 4, "foreground_value": 255}, f"extract-{index}")
        component = snapshot["electrical"]["extractions"][-1]["components"][0]["id"]
        components.append(component)
        command("review_components", {"component_ids": [component], "decision": "confirmed",
            "evidence_version_ids": [observations[index]["id"]], "note": "Independent material observation", "declared_net_name": "SIGNAL"}, f"material-{index}")
    snapshot = command("bind_contact", {"contact_id": contact["id"], "source_component_id": components[0],
        "target_component_id": components[1], "note": "Explicit physical contact/component correspondence"}, "contact-binding")
    return services, owner, identifier, config, observations, components, command, snapshot


def test_postgres_http_electrical_netlist_reopen_and_real_concurrent_cas(database, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from kraken_server.app import SessionPrincipal, create_app
    from krona.bootstrap import logic_functions_to_dict

    service, owner, identifier, config, observations, components, command, snapshot = setup_electrical(database)
    assert len(snapshot["graph"]["nets"]) == 1
    app = create_app(services=service, session_resolver=lambda token: SessionPrincipal(str(owner.id), "local", token))
    with TestClient(app) as client:
        base = f"/api/v1/projects/{identifier}/electrical"
        assert client.get(base).status_code == 401
        headers = {"Authorization": "Bearer owner"}
        for i, (index, port, direction) in enumerate([(0, "A", "input"), (2, "Y", "output")]):
            current = client.get(base, headers=headers).json()
            body = {"expected_physical_revision": current["physical_model_revision"], "operation": "bind_terminal", "arguments": {
                "component_id": components[i], "port": port, "direction": direction,
                "evidence_version_ids": [observations[index]["id"]], "note": "Independent synthetic pad observation"}}
            request_headers = headers | {"Idempotency-Key": f"terminal-{port}"}
            assert client.post(base + "/commands", json=body, headers=request_headers).status_code == 422
            request_headers["If-Match"] = str(current["electrical"]["revision"])
            response = client.post(base + "/commands", json=body, headers=request_headers)
            assert response.status_code == 200, response.text
            assert client.post(base + "/commands", json=body, headers=request_headers).json() == response.json()
        exported = client.get(base + "/export", headers=headers)
        assert exported.status_code == 200, exported.text
        output = tmp_path / "postgres-electrical.edf"
        output.write_text(exported.json()["content"])
        logic = logic_functions_to_dict(output)
        assert [row["value"] for row in logic["outputs"]["Y"]["truth_table"]] == [0, 1]
    reopened = open_project_services(service.engine.url.render_as_string(hide_password=False), config.blob_root, engine=service.engine)
    persisted = reopened.get_electrical_model(identifier)
    assert persisted == service.get_electrical_model(identifier)
    assert reopened.export_electrical_model(identifier) == exported.json()
    revision = persisted["electrical"]["revision"]
    physical_revision = persisted["physical_model_revision"]
    barrier = Barrier(2)
    append = PostgresEventStore.append

    def synchronized(self, stream_id, *, expected_revision, events):
        if stream_id == f"electrical:{identifier}" and expected_revision == revision:
            barrier.wait(timeout=15)
        return append(self, stream_id, expected_revision=expected_revision, events=events)

    monkeypatch.setattr(PostgresEventStore, "append", synchronized)

    def write(index):
        try:
            return service.transport_mutate_electrical_model(identifier, {"operation": "bind_terminal",
                "expected_physical_revision": physical_revision, "arguments": {"component_id": components[0],
                    "port": f"PAD{index}", "direction": "input", "evidence_version_ids": [observations[0]["id"]],
                    "note": "Concurrent explicit pad observation"}}, CommandContext(str(owner.id), f"cas-{index}", revision))
        except ConflictError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, range(2)))
    assert sum(result is not None for result in results) == 1
    assert reopened.get_electrical_model(identifier)["electrical"]["revision"] == revision + 1
    with service.engine.connect() as connection:
        assert connection.execute(sa.text(
            "SELECT count(*) FROM transactional_outbox WHERE event_type='ElectricalModelChangedV1'"
        )).scalar_one() == revision + 1


def test_postgres_electrical_failure_rolls_back_snapshot_idempotency_and_outbox(database, monkeypatch):
    service, owner, identifier, _config, observations, components, _command, snapshot = setup_electrical(database)
    original = PostgresEventStore.append

    def fail(self, stream_id, *, expected_revision, events):
        result = original(self, stream_id, expected_revision=expected_revision, events=events)
        if stream_id == f"electrical:{identifier}":
            raise OSError("injected electrical persistence failure")
        return result

    monkeypatch.setattr(PostgresEventStore, "append", fail)
    payload = {"operation": "bind_terminal", "expected_physical_revision": snapshot["physical_model_revision"], "arguments": {
        "component_id": components[0], "port": "A", "direction": "input", "evidence_version_ids": [observations[0]["id"]],
        "note": "Reviewed explicit input pad"}}
    context = CommandContext(str(owner.id), "rollback-electrical", snapshot["electrical"]["revision"])
    with pytest.raises(OSError, match="injected"):
        service.transport_mutate_electrical_model(identifier, payload, context)
    assert service.get_electrical_model(identifier) == snapshot
    with service.engine.connect() as connection:
        assert connection.execute(sa.text(
            "SELECT count(*) FROM command_idempotency WHERE idempotency_key='rollback-electrical'"
        )).scalar_one() == 0
    monkeypatch.setattr(PostgresEventStore, "append", original)
    assert service.transport_mutate_electrical_model(identifier, payload, context)["electrical"]["revision"] == snapshot["electrical"]["revision"] + 1
