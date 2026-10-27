"""Actual HTTP/PostgreSQL device-candidate review; synthetic masks and annotations."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import sqlalchemy as sa

from kraken_admin.local_admin import open_project_services
from kraken_manager.infrastructure.postgres.event_store import PostgresEventStore
from kraken_server.services import CommandContext, ConflictError
from tests.test_admin_project_workflows import project
from tests.test_electrical_graph import png

pytest_plugins = ["tests.test_admin_project_workflows"]


def setup_device(database):
    service, _accounts, owner, config = database
    created = project(service, owner, "Synthetic device review")
    identifier = created["project_id"]
    layers = []
    for i, (name, kind) in enumerate([("POLY", "gate"), ("ACTIVE", "diffusion")]):
        layer = service.create_layer(identifier, {"name": name, "type": kind, "order": i},
            CommandContext(str(owner.id), f"device-layer-{i}", service.get_project(identifier)["revision"]))
        layers.append(layer["layer_id"])

    def artifact(layer, name, data, media_type="image/png"):
        series = service.create_artifact_series(identifier, {"scope": "layer_attachment", "name": name, "layer_id": layer},
            CommandContext(str(owner.id), f"series-{name}", None))
        return service.add_managed_artifact_version(identifier, series["id"], {"filename": name, "media_type": media_type}, (data,),
            CommandContext(str(owner.id), f"version-{name}", service.artifact_stream_revision(identifier, series["id"])))

    pixels = [[[0] * 9 for _ in range(9)] for _ in range(2)]
    for y in range(1, 8):
        pixels[0][y][4] = 255
    for x in (1, 2, 3, 5, 6, 7):
        pixels[1][4][x] = 255
    masks = [artifact(layer, f"mask-{i}.png", png(pixels[i])) for i, layer in enumerate(layers)]
    sources = [artifact(layer, f"source-{i}.png", png([[200 if p else 20 + x for x, p in enumerate(row)] for row in pixels[i]]))
               for i, layer in enumerate(layers)]
    process = artifact(layers[1], "synthetic-process.txt", b"Synthetic fixture's independently supplied process annotation: N-channel.", "text/plain")

    def physical(operation, arguments, key):
        revision = service.get_multilayer_model(identifier)["model"]["revision"]
        return service.transport_mutate_multilayer_model(identifier, {"operation": operation, "arguments": arguments},
            CommandContext(str(owner.id), key, revision))

    physical("set_stack", {"layers": [{"layer_id": layer, "layer_type": kind, "z_m": i * 1e-6,
        "thickness_m": 1e-7, "conductive": True} for i, (layer, kind) in enumerate(zip(layers, ["gate", "diffusion"]))]}, "device-stack")
    for i, layer in enumerate(layers):
        physical("calibrate", {"layer_id": layer, "artifact_version_id": masks[i]["id"], "pixel_size_m": [1e-6, 1e-6],
            "y_axis": "up", "uncertainty_m": 1e-9}, f"device-calibration-{i}")
    physical("register_affine", {"source_layer_id": layers[0], "target_layer_id": layers[1], "max_error_m": 1e-12,
        "control_points": [{"source_xy": p, "target_xy": p, "uncertainty_m": 1e-9} for p in ((0, 0), (8, 0), (0, 8), (8, 8))]}, "device-registration")
    for i, layer in enumerate(layers):
        current = service.get_electrical_model(identifier)
        extracted = service.transport_mutate_electrical_model(identifier, {"operation": "extract_conductors",
            "expected_physical_revision": current["physical_model_revision"], "arguments": {"layer_id": layer,
                "mask_version_id": masks[i]["id"], "connectivity": 4, "foreground_value": 255}},
            CommandContext(str(owner.id), f"extract-{i}", current["electrical"]["revision"]))
        extraction = next(e for e in extracted["electrical"]["extractions"] if e["layer_id"] == layer)
        service.transport_mutate_electrical_model(identifier, {"operation": "review_components",
            "expected_physical_revision": extracted["physical_model_revision"], "arguments": {
                "component_ids": [c["id"] for c in extraction["components"]], "decision": "confirmed",
                "evidence_version_ids": [sources[i]["id"]], "note": "Independent synthetic source-image review"}},
            CommandContext(str(owner.id), f"material-{i}", extracted["electrical"]["revision"]))
    snapshot = service.get_electrical_model(identifier)
    candidate = snapshot["device_candidates"][0]
    diffusion = next(e for e in snapshot["electrical"]["extractions"] if e["layer_id"] == layers[1])
    source, drain = [c["id"] for c in diffusion["components"]]
    arguments = {"candidate_id": candidate["id"], "instance": "MX1", "kind": "NMOS", "source_component_id": source,
        "drain_component_id": drain, "evidence_version_ids": [process["id"]], "evidence_kind": "process_documentation",
        "note": "Independent synthetic process annotation and explicit port assignments reviewed"}
    return service, owner, identifier, config, snapshot, arguments


def test_postgres_http_device_candidate_review_and_edif_reopen_preserve_three_separate_nets(database, tmp_path):
    from fastapi.testclient import TestClient
    from kraken_server.app import SessionPrincipal, create_app
    from krona.analysis.structural.model import DeviceKind
    from krona.analysis.structural.parsers import EdifStructuralParser
    from krona.infrastructure.edif_repository import EdifRepository

    service, owner, identifier, config, snapshot, arguments = setup_device(database)
    assert snapshot["device_candidates"][0]["state"] == "candidate"
    app = create_app(services=service, session_resolver=lambda token: SessionPrincipal(str(owner.id), "local", token))
    with TestClient(app) as client:
        base = f"/api/v1/projects/{identifier}/electrical"
        headers = {"Authorization": "Bearer owner", "Idempotency-Key": "http-device"}
        body = {"operation": "realize_device", "arguments": arguments, "expected_physical_revision": snapshot["physical_model_revision"]}
        assert client.post(base + "/commands", json=body, headers=headers).status_code == 422
        headers["If-Match"] = str(snapshot["electrical"]["revision"])
        response = client.post(base + "/commands", json=body, headers=headers)
        assert response.status_code == 200, response.text
        assert client.post(base + "/commands", json=body, headers=headers).json() == response.json()
        saved = response.json()
        assert saved["graph"]["devices"][0]["kind"] == "NMOS"
        assert saved["graph"]["devices"][0]["bulk_status"] == "unresolved"
        assert len(saved["graph"]["nets"]) == 3 and saved["graph"]["connections"] == []
        duplicate = client.post(base + "/commands", json={**body, "arguments": arguments | {"instance": "DUPLICATE"}},
            headers=headers | {"Idempotency-Key": "duplicate-device", "If-Match": str(saved["electrical"]["revision"])})
        assert duplicate.status_code == 409, duplicate.text
        exported = client.get(base + "/export", headers=headers)
        assert exported.status_code == 200, exported.text
    output = tmp_path / "postgres-reviewed-device.edf"
    output.write_text(exported.json()["content"])
    device = EdifStructuralParser(EdifRepository()).parse(output).devices["MX1"]
    assert device.kind is DeviceKind.NMOS
    assert len(set(device.pins.values())) == 3
    reopened = open_project_services(service.engine.url.render_as_string(hide_password=False), config.blob_root, engine=service.engine)
    assert reopened.get_electrical_model(identifier) == saved
    assert reopened.export_electrical_model(identifier) == exported.json()
    with service.engine.connect() as connection:
        assert connection.execute(sa.text("SELECT count(*) FROM transactional_outbox WHERE event_type='ElectricalModelChangedV1'")).scalar_one() == saved["electrical"]["revision"]


def test_postgres_device_review_compare_and_swap_prevents_duplicate_realization(database, monkeypatch):
    service, owner, identifier, _config, snapshot, arguments = setup_device(database)
    expected = snapshot["electrical"]["revision"]
    barrier, original = Barrier(2), PostgresEventStore.append

    def synchronized(self, stream_id, *, expected_revision, events):
        if stream_id == f"electrical:{identifier}" and expected_revision == expected:
            barrier.wait(timeout=15)
        return original(self, stream_id, expected_revision=expected_revision, events=events)

    monkeypatch.setattr(PostgresEventStore, "append", synchronized)

    def write(index):
        try:
            return service.transport_mutate_electrical_model(identifier, {"operation": "realize_device",
                "arguments": arguments | {"instance": f"MX{index}"}, "expected_physical_revision": snapshot["physical_model_revision"]},
                CommandContext(str(owner.id), f"device-cas-{index}", expected))
        except ConflictError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, range(2)))
    assert sum(result is not None for result in results) == 1
    saved = service.get_electrical_model(identifier)
    assert saved["electrical"]["revision"] == expected + 1
    assert len(saved["graph"]["devices"]) == 1
    assert len(saved["graph"]["nets"]) == 3
