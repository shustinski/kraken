"""Synthetic binary/physical evidence; no microscopy or silicon ground truth."""
from __future__ import annotations

import base64
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import re
from types import SimpleNamespace
from uuid import uuid4

from PIL import Image
import pytest

from kraken_manager.application.electrical_graph import build_electrical_graph, export_reviewed_edif
from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.electrical import (
    ComponentReview, ConductorExtraction, ContactBinding, ProjectEvidence, TerminalBinding,
)
from kraken_manager.domain.multilayer import (
    ArtifactAnchor, ControlPoint, CoordinateCalibration, InterlayerContact, MultilayerModel,
    PhysicalLayer, RegistrationRevision,
)
from kraken_manager.domain.project import LayerType
from kraken_manager.infrastructure.conductor_masks import extract_conductors
from kraken_manager.infrastructure.registration import fit_affine


def uid():
    return str(uuid4())


def png(rows, *, mode="L"):
    image = Image.new(mode, (len(rows[0]), len(rows)))
    image.putdata([value for row in rows for value in row])
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def anchor(data):
    return ArtifactAnchor(uid(), sha256(data).hexdigest())


def extraction(rows, *, connectivity=4, foreground_value=255, nodata_value=None, mode="L"):
    data = png(rows, mode=mode)
    mask = anchor(data)
    calibration = CoordinateCalibration(uid(), 0, mask, (1e-6, 2e-6), y_axis="up", uncertainty_m=1e-9)
    return extract_conductors(data, project_id=uid(), mask=mask, calibration=calibration,
        connectivity=connectivity, foreground_value=foreground_value, nodata_value=nodata_value)


def graph_case():
    project, reviewer = uid(), uid()
    layers = [PhysicalLayer(uid(), kind, i * 1e-6, 1e-7, kind is LayerType.METAL)
              for i, kind in enumerate([LayerType.METAL, LayerType.CONTACT, LayerType.METAL])]
    data = png([[0] * 8, [0] * 8, [0, 255, 255, 255, 255, 255, 255, 0], [0] * 8])
    masks = [anchor(data), anchor(data)]
    calibrations = [CoordinateCalibration(layers[i].layer_id, 0, masks[j], (1e-6, 1e-6), y_axis="up", uncertainty_m=1e-9)
                    for j, i in enumerate([0, 2])]
    extractions = [extract_conductors(data, project_id=project, mask=masks[j], calibration=c,
                                     connectivity=4, foreground_value=255) for j, c in enumerate(calibrations)]
    controls = tuple(ControlPoint(p, p, 1e-9) for p in [(0, 0), (7, 0), (0, 3), (7, 3)])
    registration = RegistrationRevision(uid(), 0, 1, *calibrations, controls,
        fit_affine(*calibrations, controls, max_error_m=1e-9))
    material = [anchor(b"independent source layer observation"), anchor(b"independent target layer observation")]
    via = anchor(b"independent contact plane observation")
    contact = InterlayerContact(uid(), layers[0].layer_id, layers[2].layer_id, layers[1].layer_id,
        registration.id, 0, (3e-6, 2e-6), 1e-8, "confirmed", (via,), reviewer, "Contact observed independently")
    model = MultilayerModel(project, 5, 1, tuple(layers), tuple(calibrations), (registration,), (contact,))
    reviews = [ComponentReview(e.components[0].id, reviewer, (material[i],), "Observed continuous material", declared_net_name="SIGNAL")
               for i, e in enumerate(extractions)]
    binding = ContactBinding(contact.id, extractions[0].components[0].id, extractions[1].components[0].id,
        reviewer, (via,), 'Review (not proximity); note contains ") (net INJECTED)" safely')
    terminals = [TerminalBinding(extractions[0].components[0].id, "A", "input", reviewer, (material[0],), "Input pad observed"),
                 TerminalBinding(extractions[1].components[0].id, "Y", "output", reviewer, (material[1],), "Output pad observed")]
    catalog = {}
    for i, e in enumerate(extractions):
        for evidence in [e.mask, material[i]]:
            catalog[evidence.version_id] = ProjectEvidence(project, e.layer_id, evidence, True)
    catalog[via.version_id] = ProjectEvidence(project, layers[1].layer_id, via, True)
    arguments = dict(physical_model=model, extractions=extractions, component_reviews=reviews,
        contact_bindings=[binding], terminal_bindings=terminals,
        current_artifacts={e.layer_id: e.mask for e in extractions}, evidence_catalog=catalog)
    return SimpleNamespace(args=arguments, model=model, reviews=reviews, extractions=extractions, binding=binding,
                           terminals=terminals, catalog=catalog, material=material, via=via, reviewer=reviewer)


def test_explicit_four_and_eight_adjacency_differ_and_geometry_roundtrips():
    rows = [[0] * 4, [0, 255, 0, 0], [0, 0, 255, 0], [0] * 4]
    four, eight = extraction(rows, connectivity=4), extraction(rows, connectivity=8)
    assert len(four.components) == 2
    assert len(eight.components) == 1
    assert eight.components[0].diagonal_bridge_count == 1
    assert eight.components[0].pixel_count == 2
    assert four.components[0].runs == ((1, 1, 2),)
    assert ConductorExtraction.from_dict(json.loads(json.dumps(eight.to_dict()))) == eight
    assert eight.backend_version


def test_voids_are_unknown_and_never_foreground_or_connected_background():
    result = extraction([[0, 0, 0, 0, 0], [0, 255, 128, 255, 0], [0, 0, 0, 0, 0]], nodata_value=128)
    assert len(result.components) == 2
    assert result.nodata_runs == ((1, 2, 3),)
    assert all(c.touches_nodata for c in result.components)
    assert sum(c.pixel_count for c in result.components) == 2
    transparent = extraction([[(255, 0), (0, 255)], [(0, 255), (0, 255)]], mode="LA")
    assert transparent.components == ()
    assert transparent.nodata_runs == ((0, 0, 1),)


def test_nonbinary_samples_bad_hash_and_coordinate_revision_are_rejected():
    with pytest.raises(DomainValidationError, match="nonbinary"):
        extraction([[0, 100, 255]])
    data = png([[0, 255]])
    mask = anchor(data)
    calibration = CoordinateCalibration(uid(), 0, mask, (1e-6, 1e-6))
    with pytest.raises(DomainValidationError, match="SHA-256"):
        extract_conductors(data + b"bad", project_id=uid(), mask=mask, calibration=calibration, connectivity=4, foreground_value=255)
    with pytest.raises(DomainValidationError, match="own version"):
        extract_conductors(data, project_id=uid(), mask=mask, calibration=replace(calibration, artifact=anchor(data)),
                           connectivity=4, foreground_value=255)


def test_coincident_layers_and_confirmed_contact_do_not_connect_unreviewed_material():
    q = graph_case()
    graph = build_electrical_graph(**(q.args | {"component_reviews": []}))
    assert graph["connections"] == []
    assert graph["nets"] == []
    assert len(graph["nodes"]) == 2
    assert all(node["state"] == "unreviewed" for node in graph["nodes"])


def test_candidate_contact_never_unions_fully_overlapping_reviewed_components():
    q = graph_case()
    candidate = replace(q.model.contacts[0], state="candidate", evidence=(), reviewed_by=None, review_note="")
    model = replace(q.model, contacts=(candidate,))
    graph = build_electrical_graph(**(q.args | {"physical_model": model}))
    assert len(graph["nets"]) == 2
    assert graph["connections"] == []
    assert any(u["code"] == "contact_not_engineer_confirmed" for u in graph["unknowns"])


def test_reviewed_connection_carries_reason_evidence_and_roundtrips_strict_krona(tmp_path):
    from krona.bootstrap import logic_functions_to_dict
    from krona.infrastructure.edif_repository import EdifRepository

    q = graph_case()
    graph = build_electrical_graph(**q.args)
    assert len(graph["nets"]) == 1
    assert graph["connections"][0]["reason"] == q.binding.note
    assert graph["connections"][0]["evidence"][0]["sha256"] == q.via.sha256
    assert graph["extractions"][0]["calibration"]["pixel_size_m"] == [1e-6, 1e-6]
    assert graph["electrical_completeness"] == "partial_reviewed_subgraph"
    edif = export_reviewed_edif(graph)
    encoded = re.search(r'KRAKEN_EVIDENCE_JSON_BASE64 \(string "([A-Za-z0-9+/=]+)"', edif).group(1)
    proof = json.loads(base64.b64decode(encoded))
    assert proof["connections"][0]["reason"] == q.binding.note
    assert len(proof["terminal_bindings"]) == 2
    path = tmp_path / "reviewed.edf"
    path.write_text(edif)
    parsed = EdifRepository().read_top_level_netlist(path)
    assert len(parsed.nets) == 1
    assert {c.port for c in parsed.nets[0].connections} == {"A", "Y"}
    result = logic_functions_to_dict(path)
    assert result["inputs"] == ["A"]
    assert [row["value"] for row in result["outputs"]["Y"]["truth_table"]] == [0, 1]
    assert result["meta"]["electrical_connectivity"] == "unverified"  # Import does not certify silicon.


def test_contradictory_confirmed_identities_are_conflicts_not_unconditional_union():
    q = graph_case()
    reviews = [replace(q.reviews[0], declared_net_name="VDD"), replace(q.reviews[1], declared_net_name="GND")]
    graph = build_electrical_graph(**(q.args | {"component_reviews": reviews}))
    assert graph["connections"] == []
    assert len(graph["nets"]) == 2
    assert any(c["code"] == "contradictory_confirmed_net_identities" for c in graph["conflicts"])
    with pytest.raises(DomainValidationError, match="contradictions"):
        export_reviewed_edif(graph)


def test_contradictory_contact_assignments_and_terminal_membership_are_conflicts():
    q = graph_case()
    second = replace(q.binding, target_component_id=uid())
    graph = build_electrical_graph(**(q.args | {"contact_bindings": [q.binding, second]}))
    assert graph["connections"] == []
    assert any(c["code"] == "contradictory_contact_bindings" for c in graph["conflicts"])
    terminals = [q.terminals[0], replace(q.terminals[0], component_id=q.terminals[1].component_id, evidence=q.terminals[1].evidence)]
    graph = build_electrical_graph(**(q.args | {"contact_bindings": [], "terminal_bindings": terminals}))
    assert graph["terminals"] == []
    assert len(graph["nets"]) == 2
    assert any(c["code"] == "conflicting_terminal_bindings" for c in graph["conflicts"])


def test_stale_calibration_or_artifact_never_reuses_earlier_contact_proof():
    q = graph_case()
    changed_calibration = replace(q.model.calibrations[0], revision=1)
    model = replace(q.model, calibrations=(changed_calibration, q.model.calibrations[1]))
    graph = build_electrical_graph(**(q.args | {"physical_model": model}))
    assert graph["connections"] == []
    assert any(u["code"] == "stale_or_unresolved_artifact_calibration" for u in graph["unknowns"])
    active = dict(q.args["current_artifacts"])
    active[q.extractions[0].layer_id] = anchor(b"replaced mask")
    assert build_electrical_graph(**(q.args | {"current_artifacts": active}))["connections"] == []


def test_same_content_copy_is_not_independent_material_evidence():
    q = graph_case()
    copied = ArtifactAnchor(uid(), q.extractions[0].mask.sha256)
    catalog = q.catalog | {copied.version_id: ProjectEvidence(q.model.project_id, q.extractions[0].layer_id, copied, True)}
    review = replace(q.reviews[0], evidence=(copied,))
    graph = build_electrical_graph(**(q.args | {"component_reviews": [review, q.reviews[1]], "evidence_catalog": catalog}))
    assert graph["connections"] == []
    assert graph["nodes"][0]["state"] == "independent_material_evidence_missing"


def test_explicit_contact_mapping_outside_observed_regions_is_a_conflict_not_a_nearest_match():
    q = graph_case()
    far = replace(q.model.contacts[0], position_target_m=(1, 1))
    graph = build_electrical_graph(**(q.args | {"physical_model": replace(q.model, contacts=(far,))}))
    assert graph["connections"] == []
    assert len(graph["nets"]) == 2
    assert any(c["code"] == "contact_component_observation_mismatch" for c in graph["conflicts"])
