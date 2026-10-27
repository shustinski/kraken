"""Real managed binary artifacts, physical reviews and durable electrical evidence."""
from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace

from PIL import Image
import pytest

from kraken_manager.application.errors import AuthorizationError, ConcurrencyError, ConflictError
from kraken_manager.domain.artifacts import ArtifactScope
from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.identity import Principal
from kraken_manager.infrastructure.filesystem import FilesystemEventStore
from tests.test_work_orders import _service

pytest_plugins = ["tests.test_multilayer_workflow"]


@pytest.fixture
def electrical(physical):
    q = physical
    versions = []
    for index in (0, 2):
        series = q.service.create_artifact_series(principal=q.owner, project_id=q.project.id,
            scope=ArtifactScope.LAYER_ATTACHMENT, layer_id=q.layers[index].id,
            name=f"Binary conductor mask {index}", idempotency_key=f"binary-series-{index}")
        path = q.tmp_path / f"binary-{index}.png"
        image = Image.new("L", (16, 16), 0)
        for x in range(1, 15):
            image.putpixel((x, 5), 255)
        image.save(path)
        version = q.service.add_managed_artifact_version(principal=q.owner, project_id=q.project.id,
            series_id=series.id, source=path, idempotency_key=f"binary-version-{index}")
        versions.append(version)
        q.command("calibrate", {"layer_id": q.layers[index].id, "artifact_version_id": version.id,
            "pixel_size_m": [1e-6, 1e-6], "y_axis": "up", "uncertainty_m": 1e-9}, key=f"binary-calibration-{index}")
    registered = q.command("register_affine", q.register, key="binary-registration")
    reg = registered["model"]["registrations"][-1]
    proposed = q.command("propose_contact", {"source_layer_id": q.layers[0].id, "target_layer_id": q.layers[2].id,
        "contact_layer_id": q.layers[1].id, "registration_id": reg["id"], "registration_revision": reg["revision"],
        "position_target_m": [5e-6, 5e-6], "radius_m": 1e-8}, key="binary-contact")
    contact = proposed["model"]["contacts"][-1]
    q.command("review_contact", {"contact_id": contact["id"], "decision": "confirmed",
        "evidence_version_ids": [q.versions[1].id], "review_note": "Synthetic independent contact-layer observation"}, key="binary-contact-review")
    count, requests = 0, []

    def command(operation, arguments, *, principal=None, key=None, expected=None, physical_revision=None):
        nonlocal count
        count += 1
        current = q.service.get_electrical_model(q.project.id)
        request = dict(project_id=q.project.id, principal=principal or q.owner,
            expected_revision=current["electrical"]["revision"] if expected is None else expected,
            expected_physical_revision=current["physical_model_revision"] if physical_revision is None else physical_revision,
            operation=operation, arguments=arguments, idempotency_key=key or f"electrical-{count}")
        requests.append(request)
        return q.service.mutate_electrical_model(**request)

    extractions = []
    for i, index in enumerate((0, 2)):
        result = command("extract_conductors", {"layer_id": q.layers[index].id,
            "mask_version_id": versions[i].id, "connectivity": 4, "foreground_value": 255})
        extractions.append(result["electrical"]["extractions"][-1])
    return SimpleNamespace(q=q, versions=versions, extractions=extractions, contact=contact, command=command, requests=requests)


def complete(e):
    q = e.q
    for i, index in enumerate((0, 2)):
        e.command("review_components", {"component_ids": [e.extractions[i]["components"][0]["id"]], "decision": "confirmed",
            "evidence_version_ids": [q.versions[index].id], "note": "Independent synthetic material observation", "declared_net_name": "SIGNAL"})
    e.command("bind_contact", {"contact_id": e.contact["id"], "source_component_id": e.extractions[0]["components"][0]["id"],
        "target_component_id": e.extractions[1]["components"][0]["id"], "note": "Observed contact mapped explicitly to both components"})
    for i, (index, port, direction) in enumerate([(0, "A", "input"), (2, "Y", "output")]):
        result = e.command("bind_terminal", {"component_id": e.extractions[i]["components"][0]["id"], "port": port,
            "direction": direction, "evidence_version_ids": [q.versions[index].id], "note": "Explicit synthetic pad observation"})
    return result


def test_blob_mask_reviews_explicit_contact_terminals_edif_reopen_and_audit(electrical):
    from krona.bootstrap import logic_functions_to_dict

    e, q = electrical, electrical.q
    before = q.service.get_electrical_model(q.project.id)
    assert before["graph"]["nets"] == []
    assert len(before["graph"]["nodes"]) == 2
    result = complete(e)
    assert len(result["graph"]["nets"]) == 1
    assert result["graph"]["connections"][0]["evidence"][0]["sha256"] == q.versions[1].sha256
    exported = q.service.export_electrical_model(q.project.id)
    assert exported["sha256"] == sha256(exported["content"].encode()).hexdigest()
    output = q.tmp_path / "electrical.edf"
    output.write_text(exported["content"])
    logic = logic_functions_to_dict(output)
    assert [row["value"] for row in logic["outputs"]["Y"]["truth_table"]] == [0, 1]
    reopened = _service(q.tmp_path)
    assert reopened.get_electrical_model(q.project.id) == result
    assert reopened.export_electrical_model(q.project.id) == exported
    events = FilesystemEventStore(q.service.catalog_root, q.project.id).load_stream(f"electrical:{q.project.id}")
    assert len(events) == result["electrical"]["revision"] == 7
    assert all(event.event_type == "ElectricalModelChangedV1" for event in events)
    assert events[-1].actor.principal_id == q.owner.id
    assert events[-1].payload["arguments"]["note"] == "Explicit synthetic pad observation"


def test_revisions_rbac_independent_evidence_and_idempotency_are_enforced(electrical):
    e, q = electrical, electrical.q
    request = e.requests[-1]
    replay = q.service.mutate_electrical_model(**request)
    assert replay["electrical"]["revision"] == 2
    with pytest.raises(ConflictError):
        q.service.mutate_electrical_model(**(request | {"arguments": request["arguments"] | {"connectivity": 8}}))
    for expected, physical_revision in [(0, None), (None, 0)]:
        with pytest.raises(ConcurrencyError):
            e.command("extract_conductors", request["arguments"], expected=expected, physical_revision=physical_revision)
    with pytest.raises(DomainValidationError, match="contents"):
        e.command("review_components", {"component_ids": [e.extractions[0]["components"][0]["id"]], "decision": "confirmed",
            "evidence_version_ids": [e.versions[0].id], "note": "Mask itself is not independent material evidence"})
    outsider = Principal.local(subject="outsider-electrical", display_name="Outsider")
    q.service.identities.save(outsider)
    with pytest.raises(AuthorizationError):
        e.command("review_components", {"component_ids": [e.extractions[0]["components"][0]["id"]], "decision": "confirmed",
            "evidence_version_ids": [q.versions[0].id], "note": "Unauthorized confirmation"}, principal=outsider)
    assert q.service.get_electrical_model(q.project.id)["electrical"]["revision"] == 2


def test_later_physical_change_and_review_rejection_remove_current_connectivity(electrical):
    e, q = electrical, electrical.q
    complete(e)
    binding_request = next(request for request in e.requests if request["operation"] == "bind_contact")
    e.command("review_components", {"component_ids": [e.extractions[0]["components"][0]["id"]], "decision": "rejected",
        "evidence_version_ids": [], "note": "Material observation rejected after independent review"})
    replay = q.service.mutate_electrical_model(**binding_request)
    assert replay["electrical"]["revision"] == 5  # Original snapshot retained.
    assert replay["graph_electrical_revision"] == 8  # Present evidence assessment uses latest review.
    assert replay["graph"]["connections"] == []
    q.command("calibrate", {"layer_id": q.layers[0].id, "artifact_version_id": e.versions[0].id,
        "pixel_size_m": [2e-6, 2e-6], "y_axis": "up"})
    graph = q.service.get_electrical_model(q.project.id)["graph"]
    assert graph["connections"] == []
    assert any(u["code"] == "stale_or_unresolved_artifact_calibration" for u in graph["unknowns"])


def test_conflicting_declarations_block_export_and_explicit_retraction_is_audited(electrical):
    e, q = electrical, electrical.q
    complete(e)
    for i, (index, label) in enumerate([(0, "VDD"), (2, "GND")]):
        result = e.command("review_components", {"component_ids": [e.extractions[i]["components"][0]["id"]],
            "decision": "confirmed", "evidence_version_ids": [q.versions[index].id],
            "note": "Explicit canonical net identity", "declared_net_name": label})
    assert result["graph"]["connections"] == []
    assert any(c["code"] == "contradictory_confirmed_net_identities" for c in result["graph"]["conflicts"])
    with pytest.raises(DomainValidationError, match="contradictions"):
        q.service.export_electrical_model(q.project.id)
    result = e.command("retract_contact_binding", {"contact_id": e.contact["id"], "note": "Contact assignment withdrawn pending investigation"})
    assert result["graph"]["conflicts"] == []
    assert len(result["graph"]["nets"]) == 2
