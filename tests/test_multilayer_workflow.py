"""Real artifacts, durable physical revisions, independent evidence and HTTP guards."""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from kraken_manager.application.errors import AuthorizationError, ConcurrencyError, ConflictError
from kraken_manager.domain.artifacts import ArtifactScope
from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.multilayer import MultilayerModel
from kraken_manager.domain.project import LayerType
from kraken_manager.infrastructure.filesystem import FilesystemEventStore
from tests.test_work_planning_integration import setup_project
from tests.test_work_orders import _service


@pytest.fixture
def physical(tmp_path, monkeypatch):
    monkeypatch.setenv("KRAKEN_PROJECT_ACCESS_MODE", "acl")
    service, owner, project, metal1, performer = setup_project(tmp_path)
    layers = [metal1]
    for order, (name, kind) in enumerate([("V1", LayerType.CONTACT), ("M2", LayerType.METAL)], 1):
        layers.append(service.create_layer(principal=owner, project=service.get_project(project.id), name=name,
                                            layer_type=kind, order=order, idempotency_key=f"layer-{order}"))
    versions, series = [], []
    for i, layer in enumerate(layers):
        series.append(service.create_artifact_series(principal=owner, project_id=project.id,
            scope=ArtifactScope.LAYER_ATTACHMENT, layer_id=layer.id, name=f"Physical input {i}", idempotency_key=f"series-{i}"))
        image = tmp_path / f"synthetic-{i}.pgm"
        image.write_bytes(b"P5\n16 16\n255\n" + bytes((j + i) % 256 for j in range(256)))
        versions.append(service.add_managed_artifact_version(principal=owner, project_id=project.id,
            series_id=series[-1].id, source=image, idempotency_key=f"version-{i}"))
    command_count = 0

    def command(operation, arguments, *, principal=None, key=None, expected=None):
        nonlocal command_count
        command_count += 1
        current = service.get_multilayer_model(project.id)["model"]["revision"]
        return service.mutate_multilayer_model(
            project_id=project.id, principal=principal or owner,
            expected_revision=current if expected is None else expected,
            operation=operation, arguments=arguments, idempotency_key=key or f"physical-{command_count}",
        )

    stack = {"layers": [{"layer_id": layer.id, "layer_type": layer.type, "z_m": i * 1e-6,
                         "thickness_m": 1e-7, "conductive": layer.type is LayerType.METAL}
                        for i, layer in enumerate(layers)]}
    command("set_stack", stack)
    for i in (0, 2):
        command("calibrate", {"layer_id": layers[i].id, "artifact_version_id": versions[i].id,
                              "pixel_size_m": [1e-6, 1e-6], "y_axis": "up", "uncertainty_m": 1e-9})
    register = {"source_layer_id": layers[0].id, "target_layer_id": layers[2].id,
                "control_points": [{"source_xy": p, "target_xy": p, "uncertainty_m": 1e-9}
                                   for p in [(0, 0), (10, 0), (0, 10), (10, 10)]], "max_error_m": 1e-9}
    payload = command("register_affine", register, key="first-registration")
    registration = payload["model"]["registrations"][-1]
    propose = {"source_layer_id": layers[0].id, "target_layer_id": layers[2].id,
               "contact_layer_id": layers[1].id, "registration_id": registration["id"],
               "registration_revision": registration["revision"], "position_target_m": [5e-6, 5e-6], "radius_m": 1e-8}
    payload = command("propose_contact", propose)
    contact = payload["model"]["contacts"][-1]
    return SimpleNamespace(service=service, owner=owner, project=project, layers=layers, versions=versions,
                           series=series, command=command, register=register, registration=registration,
                           contact=contact, stack=stack, tmp_path=tmp_path)


def test_real_physical_fit_contact_review_roundtrip_reopen_and_audit(physical):
    q = physical
    assert q.registration["fit"]["status"] == "fitted"
    assert q.registration["fit"]["control_error_bound_m"] > 0
    assert q.service.get_multilayer_model(q.project.id)["assessment"]["contacts"][0]["electrical_connectivity"] == "unknown"
    payload = q.command("review_contact", {"contact_id": q.contact["id"], "decision": "confirmed",
        "evidence_version_ids": [q.versions[1].id], "review_note": "Independent contact-layer observation reviewed"})
    assert payload["assessment"]["contacts"][0]["electrical_connectivity"] == "engineer_confirmed"
    model = MultilayerModel.from_dict(payload)
    assert model.contacts[0].reviewed_by == q.owner.id
    assert model.contacts[0].evidence[0].sha256 == q.versions[1].sha256
    reopened = _service(q.tmp_path)
    assert reopened.get_multilayer_model(q.project.id) == payload
    history = FilesystemEventStore(q.service.catalog_root, q.project.id).load_stream(f"multilayer:{q.project.id}")
    assert [e.revision for e in history] == list(range(1, model.revision + 1))
    assert history[-1].event_type == "MultilayerModelChangedV1"
    assert history[-1].payload["operation"] == "review_contact"
    assert history[-1].actor.principal_id == q.owner.id


def test_no_contact_confirmation_without_independent_contact_layer_evidence(physical):
    q = physical
    for evidence in ([], [q.versions[0].id], [str(uuid4())]):
        with pytest.raises(DomainValidationError):
            q.command("review_contact", {"contact_id": q.contact["id"], "decision": "confirmed",
                "evidence_version_ids": evidence, "review_note": "Geometry alone is insufficient"})
    assert q.service.get_multilayer_model(q.project.id)["model"]["contacts"][0]["state"] == "candidate"


def test_recalibration_marks_prior_registration_and_contact_stale_without_auto_recompute(physical):
    q = physical
    q.command("review_contact", {"contact_id": q.contact["id"], "decision": "confirmed",
        "evidence_version_ids": [q.versions[1].id], "review_note": "Reviewed independently"})
    payload = q.command("calibrate", {"layer_id": q.layers[0].id, "artifact_version_id": q.versions[0].id,
                                      "pixel_size_m": [2e-6, 2e-6], "y_axis": "up"})
    assert payload["assessment"]["registrations"][0]["current"] is False
    assert payload["assessment"]["contacts"][0]["effective_state"] == "stale"
    assert payload["assessment"]["contacts"][0]["electrical_connectivity"] == "unknown"
    assert payload["assessment"]["automatic_recalculation"] is False
    assert payload["model"]["contacts"][0]["state"] == "confirmed"  # Historical review is preserved.
    with pytest.raises(ConflictError, match="Stale"):
        q.command("review_contact", {"contact_id": q.contact["id"], "decision": "confirmed",
            "evidence_version_ids": [q.versions[1].id], "review_note": "Must re-register first"})
    replay = q.command("register_affine", q.register, key="first-registration", expected=0)
    assert replay["assessment"]["registrations"][0]["current"] is False


def test_revision_replay_payload_conflict_foreign_inputs_and_rbac(physical):
    q = physical
    revision = q.service.get_multilayer_model(q.project.id)["model"]["revision"]
    with pytest.raises(ConcurrencyError):
        q.command("set_stack", q.stack, expected=0)
    with pytest.raises(ConflictError, match="Idempotency"):
        q.command("register_affine", q.register | {"max_error_m": 1e-5}, key="first-registration")
    with pytest.raises(DomainValidationError):
        q.command("calibrate", {"layer_id": q.layers[0].id, "artifact_version_id": q.versions[2].id,
                                "pixel_size_m": [1e-6, 1e-6]})
    outsider = Principal.local(subject="outsider", display_name="Outsider")
    q.service.identities.save(outsider)
    with pytest.raises(AuthorizationError):
        q.command("set_stack", q.stack, principal=outsider)
    assert q.service.get_multilayer_model(q.project.id)["model"]["revision"] == revision


def test_replacing_source_artifact_marks_registration_stale(physical):
    q = physical
    image = q.tmp_path / "updated.pgm"
    image.write_bytes(b"P5\n1 1\n255\n\x01")
    updated = q.service.add_managed_artifact_version(
        principal=q.owner, project_id=q.project.id, series_id=q.series[0].id, source=image,
        parent_version_id=q.versions[0].id, idempotency_key="updated-physical-source",
    )
    assert q.service._projection(q.project.id).get_active_artifact_version(q.series[0].id).id == updated.id
    result = q.service.get_multilayer_model(q.project.id)
    assert result["assessment"]["registrations"][0]["current"] is False
    with pytest.raises(ConflictError, match="stale"):
        q.command("register_affine", q.register)


def test_underdetermined_registration_is_audited_and_cannot_confirm_contact(physical):
    q = physical
    result = q.command("register_affine", q.register | {"control_points": q.register["control_points"][:2]})
    registration = result["model"]["registrations"][-1]
    assert registration["fit"]["status"] == "insufficient"
    assert registration["fit"]["affine"] is None
    assert result["assessment"]["registrations"][0]["current"] is False  # Prior registration superseded.
    assert result["assessment"]["contacts"][0]["effective_state"] == "stale"


def test_http_physical_commands_require_revision_authorization_and_use_same_durable_model(physical):
    from fastapi.testclient import TestClient
    from kraken_server.app import SessionPrincipal, create_app
    from kraken_server.persistent_services import PostgresServerServices

    q = physical
    server = object.__new__(PostgresServerServices)
    server.projections = q.service._projection(q.project.id)
    server.events = FilesystemEventStore(q.service.catalog_root, q.project.id)
    server.uow_factory = q.service._uow(str(q.project.id))
    server.profiles = q.service.profiles
    server.identities = q.service.identities
    server.project_ids_for_principal = lambda actor_id: (
        (str(q.project.id),) if q.service.identities.roles_for(q.project.id, actor_id) else ()
    )
    app = create_app(services=server,
        session_resolver=lambda token: SessionPrincipal(str(q.owner.id), "local", token))
    with TestClient(app) as client:
        base = f"/api/v1/projects/{q.project.id}/multilayer"
        assert client.get(base).status_code == 401
        headers = {"Authorization": "Bearer owner", "Idempotency-Key": "http-contact"}
        current = client.get(base, headers=headers)
        assert current.status_code == 200, current.text
        revision = current.json()["model"]["revision"]
        body = {"operation": "review_contact", "arguments": {"contact_id": q.contact["id"], "decision": "confirmed",
                "evidence_version_ids": [q.versions[1].id], "review_note": "Independent contact-layer evidence reviewed"}}
        assert client.post(base + "/commands", headers=headers, json=body).status_code == 422
        headers |= {"If-Match": str(revision)}
        response = client.post(base + "/commands", headers=headers, json=body)
        assert response.status_code == 200, response.text
        assert response.json()["assessment"]["contacts"][0]["electrical_connectivity"] == "engineer_confirmed"
        assert client.post(base + "/commands", headers=headers, json=body).json() == response.json()
        assert q.service.get_multilayer_model(q.project.id) == response.json()
        stale = client.post(base + "/commands", headers=headers | {"Idempotency-Key": "stale-http"}, json=body)
        assert stale.status_code == 409, stale.text
        unknown = client.post(base + "/commands", headers=headers | {
            "Idempotency-Key": "invalid-operation", "If-Match": str(revision + 1)},
            json={"operation": "force_confirm", "arguments": {}})
        assert unknown.status_code == 422, unknown.text
