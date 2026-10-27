"""Managed synthetic gate/diffusion evidence, explicit review and durable devices."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from kraken_manager.application.errors import AuthorizationError, ConflictError, NotFoundError
from kraken_manager.domain.artifacts import ArtifactScope
from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.project import LayerType
from kraken_manager.infrastructure.filesystem import FilesystemEventStore
from tests.test_electrical_graph import png
from tests.test_work_orders import _service
from tests.test_work_planning_integration import setup_project


@pytest.fixture
def device_project(tmp_path, monkeypatch):
    monkeypatch.setenv("KRAKEN_PROJECT_ACCESS_MODE", "acl")
    service, owner, project, _metal, _performer = setup_project(tmp_path)
    layers = [service.create_layer(principal=owner, project=service.get_project(project.id), name=name,
        layer_type=kind, order=i + 1, idempotency_key=f"device-layer-{i}")
        for i, (name, kind) in enumerate([("POLY", LayerType.GATE), ("ACTIVE", LayerType.DIFFUSION)])]
    rows = [[[0] * 9 for _ in range(9)] for _ in range(2)]
    for y in range(1, 8):
        rows[0][y][4] = 255
    for x in (1, 2, 3, 5, 6, 7):
        rows[1][4][x] = 255

    def artifact(layer, name, data):
        series = service.create_artifact_series(principal=owner, project_id=project.id, scope=ArtifactScope.LAYER_ATTACHMENT,
            layer_id=layer.id, name=name, idempotency_key=f"series-{name}")
        file = tmp_path / name
        file.write_bytes(data)
        return service.add_managed_artifact_version(principal=owner, project_id=project.id, series_id=series.id,
                                                    source=file, idempotency_key=f"version-{name}")

    masks = [artifact(layer, f"device-mask-{i}.png", png(rows[i])) for i, layer in enumerate(layers)]
    observations = [artifact(layer, f"device-source-{i}.png", png([[200 if v else 20 + x for x, v in enumerate(row)] for row in rows[i]]))
                    for i, layer in enumerate(layers)]
    process = artifact(layers[1], "synthetic-process-annotation.txt", b"Synthetic known fixture design: annotated N-channel device. No silicon claim.")
    sequence = 0

    def physical(operation, arguments):
        nonlocal sequence
        sequence += 1
        return service.mutate_multilayer_model(project_id=project.id, principal=owner,
            expected_revision=service.get_multilayer_model(project.id)["model"]["revision"], operation=operation,
            arguments=arguments, idempotency_key=f"device-physical-{sequence}")

    physical("set_stack", {"layers": [{"layer_id": layer.id, "layer_type": layer.type.value, "z_m": i * 1e-6,
        "thickness_m": 1e-7, "conductive": True} for i, layer in enumerate(layers)]})
    for layer, mask in zip(layers, masks):
        physical("calibrate", {"layer_id": layer.id, "artifact_version_id": mask.id,
            "pixel_size_m": [1e-6, 1e-6], "y_axis": "up", "uncertainty_m": 1e-9})
    physical("register_affine", {"source_layer_id": layers[0].id, "target_layer_id": layers[1].id, "max_error_m": 1e-12,
        "control_points": [{"source_xy": p, "target_xy": p, "uncertainty_m": 1e-9} for p in ((0, 0), (8, 0), (0, 8), (8, 8))]})

    def command(operation, arguments, principal=None):
        nonlocal sequence
        sequence += 1
        snapshot = service.get_electrical_model(project.id)
        return service.mutate_electrical_model(project_id=project.id, principal=principal or owner,
            expected_revision=snapshot["electrical"]["revision"], expected_physical_revision=snapshot["physical_model_revision"],
            operation=operation, arguments=arguments, idempotency_key=f"device-command-{sequence}")

    for layer, mask, observation in zip(layers, masks, observations):
        result = command("extract_conductors", {"layer_id": layer.id, "mask_version_id": mask.id, "connectivity": 4, "foreground_value": 255})
        extracted = next(e for e in result["electrical"]["extractions"] if e["layer_id"] == layer.id)
        command("review_components", {"component_ids": [c["id"] for c in extracted["components"]], "decision": "confirmed",
            "evidence_version_ids": [observation.id], "note": "Independent synthetic material observation"})
    description = service.get_electrical_model(project.id)
    candidate = description["device_candidates"][0]
    diffusion = next(e for e in description["electrical"]["extractions"] if e["layer_id"] == layers[1].id)
    source, drain = [c["id"] for c in diffusion["components"]]
    arguments = {"candidate_id": candidate["id"], "instance": "MX1", "kind": "NMOS", "source_component_id": source,
        "drain_component_id": drain, "evidence_version_ids": [process.id], "evidence_kind": "process_documentation",
        "note": "Synthetic fixture process annotation and explicit source/drain roles reviewed"}
    return SimpleNamespace(service=service, owner=owner, project=project, layers=layers, masks=masks, observations=observations,
        process=process, candidate=candidate, source=source, drain=drain, arguments=arguments,
        command=command, physical=physical, tmp_path=tmp_path)


def test_real_managed_candidate_requires_review_then_persists_explicit_mos_and_reopens(device_project):
    from krona.analysis.structural.model import DeviceKind
    from krona.analysis.structural.parsers import EdifStructuralParser
    from krona.infrastructure.edif_repository import EdifRepository

    q = device_project
    assert q.candidate["state"] == "candidate"
    initial = q.service.get_electrical_model(q.project.id)
    assert initial["graph"]["devices"] == []
    assert initial["graph"]["connections"] == []
    assert len(initial["graph"]["nets"]) == 3
    saved = q.command("realize_device", q.arguments)
    assert saved["graph"]["devices"][0]["kind"] == "NMOS"
    assert saved["graph"]["devices"][0]["bulk_status"] == "unresolved"
    assert saved["graph"]["devices"][0]["kind_evidence"][0]["sha256"] == q.process.sha256
    assert len(saved["graph"]["nets"]) == 3 and saved["graph"]["connections"] == []
    exported = q.service.export_electrical_model(q.project.id)
    file = q.tmp_path / "device-reviewed.edf"
    file.write_text(exported["content"])
    parsed = EdifStructuralParser(EdifRepository()).parse(file)
    assert parsed.devices["MX1"].kind is DeviceKind.NMOS
    assert len(set(parsed.devices["MX1"].pins.values())) == 3
    assert _service(q.tmp_path).get_electrical_model(q.project.id) == saved
    events = FilesystemEventStore(q.service.catalog_root, q.project.id).load_stream(f"electrical:{q.project.id}")
    assert events[-1].payload["operation"] == "realize_device"
    assert events[-1].payload["arguments"]["evidence_kind"] == "process_documentation"
    assert events[-1].actor.principal_id == q.owner.id


def test_realization_rejects_forged_kind_evidence_roles_and_unauthorized_actor(device_project):
    q = device_project
    for changes in ({"source_component_id": q.drain}, {"kind": "AUTO_GUESS"},
                    {"evidence_version_ids": []}, {"evidence_version_ids": [q.masks[0].id]}, {"evidence_kind": "nearest_net"}):
        with pytest.raises(DomainValidationError):
            q.command("realize_device", q.arguments | changes)
    outsider = Principal.local(subject="device-outsider", display_name="Outsider")
    q.service.identities.save(outsider)
    with pytest.raises(AuthorizationError):
        q.command("realize_device", q.arguments, principal=outsider)
    assert q.service.get_electrical_model(q.project.id)["graph"]["devices"] == []


def test_device_change_requires_explicit_retraction_and_new_coordinate_revision_invalidates_realization(device_project):
    q = device_project
    q.command("realize_device", q.arguments)
    with pytest.raises(ConflictError, match="retract"):
        q.command("realize_device", q.arguments | {"kind": "PMOS"})
    q.physical("calibrate", {"layer_id": q.layers[0].id, "artifact_version_id": q.masks[0].id,
        "pixel_size_m": [2e-6, 2e-6], "y_axis": "up"})
    stale = q.service.get_electrical_model(q.project.id)
    assert stale["graph"]["devices"] == []
    assert any(u["code"] == "device_candidate_or_material_review_stale" for u in stale["graph"]["unknowns"])
    assert stale["electrical"]["device_realizations"][0]["kind"] == "NMOS"  # Historical review retained.
    q.command("retract_device", {"instance": "MX1", "note": "Coordinate calibration replaced; re-observation required"})
    with pytest.raises(NotFoundError):
        q.command("realize_device", q.arguments)
