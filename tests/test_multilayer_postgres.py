"""Physical revisions in fixture-owned PostgreSQL UUID schemas, never public."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import sqlalchemy as sa

from kraken_admin.infrastructure.composition import open_project_services
from kraken_manager.infrastructure.postgres.event_store import PostgresEventStore
from kraken_server.services import CommandContext, ConflictError
from tests.test_admin_project_workflows import project

pytest_plugins = ["tests.test_admin_project_workflows"]


def setup_physical(database):
    service, _accounts, owner, config = database
    created = project(service, owner, "Physical test")
    identifier = created["project_id"]
    layers, versions = [], []
    for order, (name, kind) in enumerate([("M1", "metal"), ("V1", "contact"), ("M2", "metal")]):
        layer = service.create_layer(identifier, {"name": name, "type": kind, "order": order},
            CommandContext(str(owner.id), f"layer-{order}", service.get_project(identifier)["revision"]))
        layers.append(layer["layer_id"])
        series = service.create_artifact_series(identifier, {"scope": "layer_attachment", "name": f"Synthetic input {name}",
            "layer_id": layer["layer_id"]}, CommandContext(str(owner.id), f"series-{order}", None))
        image = b"P5\n2 2\n255\n" + bytes([0, order + 1, 255, 0])
        versions.append(service.add_managed_artifact_version(identifier, series["id"], {
            "filename": f"synthetic-{name}.pgm", "media_type": "image/x-portable-graymap"}, (image,),
            CommandContext(str(owner.id), f"version-{order}", service.artifact_stream_revision(identifier, series["id"]))))

    def command(operation, args, key):
        revision = service.get_multilayer_model(identifier)["model"]["revision"]
        return service.transport_mutate_multilayer_model(identifier, {"operation": operation, "arguments": args},
            CommandContext(str(owner.id), key, revision))

    stack = {"layers": [{"layer_id": layer, "layer_type": "contact" if i == 1 else "metal", "z_m": i * 1e-6,
                          "thickness_m": 1e-7, "conductive": i != 1} for i, layer in enumerate(layers)]}
    command("set_stack", stack, "stack")
    for i in (0, 2):
        command("calibrate", {"layer_id": layers[i], "artifact_version_id": versions[i]["id"],
            "pixel_size_m": [1e-6, 1e-6], "y_axis": "up", "uncertainty_m": 1e-9}, f"calibration-{i}")
    result = command("register_affine", {"source_layer_id": layers[0], "target_layer_id": layers[2],
        "max_error_m": 1e-9, "control_points": [{"source_xy": p, "target_xy": p, "uncertainty_m": 1e-9}
            for p in [(0, 0), (10, 0), (0, 10), (10, 10)]]}, "registration")
    registration = result["model"]["registrations"][0]
    result = command("propose_contact", {"source_layer_id": layers[0], "target_layer_id": layers[2],
        "contact_layer_id": layers[1], "registration_id": registration["id"], "registration_revision": 0,
        "position_target_m": [5e-6, 5e-6], "radius_m": 1e-8}, "contact")
    return service, owner, identifier, config, layers, versions, stack, command, result


def test_postgres_affine_contact_review_reopen_outbox_and_concurrent_cas(database, monkeypatch):
    service, owner, identifier, config, layers, versions, stack, command, result = setup_physical(database)
    assert result["model"]["registrations"][0]["fit"]["status"] == "fitted"
    contact = result["model"]["contacts"][0]
    confirmed = command("review_contact", {"contact_id": contact["id"], "decision": "confirmed",
        "evidence_version_ids": [versions[1]["id"]], "review_note": "Independent contact-layer observation reviewed"}, "confirm")
    reopened = open_project_services(service.engine.url.render_as_string(hide_password=False), config.blob_root, engine=service.engine)
    assert reopened.get_multilayer_model(identifier) == confirmed
    assert confirmed["assessment"]["contacts"][0]["electrical_connectivity"] == "engineer_confirmed"
    revision = confirmed["model"]["revision"]
    barrier = Barrier(2)
    original = PostgresEventStore.append

    def synchronized(self, stream_id, *, expected_revision, events):
        if stream_id == f"multilayer:{identifier}" and expected_revision == revision:
            barrier.wait(timeout=15)
        return original(self, stream_id, expected_revision=expected_revision, events=events)

    monkeypatch.setattr(PostgresEventStore, "append", synchronized)

    def write(index):
        try:
            return service.transport_mutate_multilayer_model(identifier, {"operation": "set_stack", "arguments": stack},
                CommandContext(str(owner.id), f"cas-{index}", revision))
        except ConflictError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, range(2)))
    assert sum(result is not None for result in results) == 1
    after = reopened.get_multilayer_model(identifier)
    assert after["model"]["revision"] == revision + 1
    assert after["assessment"]["contacts"][0]["effective_state"] == "stale"
    assert after["model"]["contacts"][0]["state"] == "confirmed"
    with service.engine.connect() as connection:
        count = connection.execute(sa.text(
            "SELECT count(*) FROM transactional_outbox WHERE event_type='MultilayerModelChangedV1'"
        )).scalar_one()
    assert count == revision + 1


def test_postgres_failed_physical_event_rolls_back_idempotency_and_outbox(database, monkeypatch):
    import pytest

    service, owner, identifier, _config, _layers, _versions, stack, _command, result = setup_physical(database)
    original = PostgresEventStore.append

    def fail(self, stream_id, *, expected_revision, events):
        response = original(self, stream_id, expected_revision=expected_revision, events=events)
        if stream_id == f"multilayer:{identifier}":
            raise OSError("injected physical persistence failure")
        return response

    monkeypatch.setattr(PostgresEventStore, "append", fail)
    revision = result["model"]["revision"]
    context = CommandContext(str(owner.id), "rollback-physical", revision)
    payload = {"operation": "set_stack", "arguments": stack}
    with pytest.raises(OSError, match="injected"):
        service.transport_mutate_multilayer_model(identifier, payload, context)
    assert service.get_multilayer_model(identifier) == result
    with service.engine.connect() as connection:
        assert connection.execute(sa.text(
            "SELECT count(*) FROM command_idempotency WHERE idempotency_key='rollback-physical'"
        )).scalar_one() == 0
    monkeypatch.setattr(PostgresEventStore, "append", original)
    assert service.transport_mutate_multilayer_model(identifier, payload, context)["model"]["revision"] == revision + 1
