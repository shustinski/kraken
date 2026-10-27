"""Synthetic geometry candidates; channel polarity and terminal roles are reviewed."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kraken_manager.application.device_candidates import detect_mos_candidates, realize_mos_candidate
from kraken_manager.application.electrical_graph import build_electrical_graph, export_reviewed_edif
from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.device_candidates import MosCandidate, MosRealization
from kraken_manager.domain.electrical import ComponentReview, ProjectEvidence, TerminalBinding
from kraken_manager.domain.multilayer import ControlPoint, CoordinateCalibration, MultilayerModel, PhysicalLayer, RegistrationRevision
from kraken_manager.domain.project import LayerType
from kraken_manager.infrastructure.conductor_masks import extract_conductors
from kraken_manager.infrastructure.registration import fit_affine
from tests.test_electrical_graph import anchor, png, uid


def mos_case(mode="identity", *, continuous=False, complex_gate=False, multiple=False, void_gate=False, bypass=False):
    project, reviewer, gate_layer, diffusion_layer = uid(), uid(), uid(), uid()
    gate, diffusion = [[0] * 9 for _ in range(9)], [[0] * 9 for _ in range(9)]
    if mode == "shear":
        for x in range(1, 8):
            gate[4][x] = 255
        for y in (1, 2, 3, 5, 6, 7):
            diffusion[y][4] = 255
        def transform(p):
            return (p[0] + 0.25 * p[1], p[1])
    else:
        for y in range(1, 8):
            gate[y][4] = 255
        if mode == "rotation":
            for y in (1, 2, 3, 5, 6, 7):
                diffusion[y][4] = 255
            def transform(p):
                return (8 - p[1], p[0])
        else:
            for y in ((2, 6) if multiple else (4,)):
                for x in ((1, 2, 3, 4, 5, 6, 7) if continuous else (1, 2, 3, 5, 6, 7)):
                    diffusion[y][x] = 255
            def transform(p):
                return p
    if complex_gate:
        gate[1][5] = 255
    if void_gate:
        gate[4][4] = 128
    if bypass:
        for x in range(1, 8):
            diffusion[0][x] = 255
        for y in range(5):
            diffusion[y][1] = diffusion[y][7] = 255
    data = [png(gate), png(diffusion)]
    masks = [anchor(d) for d in data]
    calibrations = [CoordinateCalibration(layer, 0, mask, (1e-6, 1e-6), y_axis="up", uncertainty_m=1e-9)
                    for layer, mask in zip((gate_layer, diffusion_layer), masks)]
    extractions = [extract_conductors(d, project_id=project, mask=m, calibration=c, connectivity=4,
                                     foreground_value=255, nodata_value=128 if void_gate else None)
                   for d, m, c in zip(data, masks, calibrations)]
    points = tuple(ControlPoint(p, transform(p), 1e-9) for p in ((0, 0), (8, 0), (0, 8), (8, 8)))
    fit = fit_affine(*calibrations, points, max_error_m=1e-12)
    registration = RegistrationRevision(uid(), 0, 1, *calibrations, points, fit)
    layers = (PhysicalLayer(gate_layer, LayerType.GATE, 0, 1e-7, True),
              PhysicalLayer(diffusion_layer, LayerType.DIFFUSION, 1e-6, 1e-7, True))
    model = MultilayerModel(project, 4, 1, layers, tuple(calibrations), (registration,))
    observations = [anchor(b"synthetic primary gate image"), anchor(b"synthetic primary diffusion image")]
    polarity = anchor(b"explicit independent process annotation: reviewed channel type")
    catalog = {polarity.version_id: ProjectEvidence(project, diffusion_layer, polarity, True)}
    reviews = []
    for i, extraction in enumerate(extractions):
        for a in (extraction.mask, observations[i]):
            catalog[a.version_id] = ProjectEvidence(project, extraction.layer_id, a, True)
        reviews.extend(ComponentReview(c.id, reviewer, (observations[i],), "Observed synthetic material") for c in extraction.components)
    arguments = dict(physical_model=model, extractions=extractions, component_reviews=reviews,
        contact_bindings=[], terminal_bindings=[], current_artifacts={e.layer_id: e.mask for e in extractions}, evidence_catalog=catalog)
    graph = build_electrical_graph(**arguments)
    candidates = detect_mos_candidates(physical_model=model, gate_extraction=extractions[0], diffusion_extraction=extractions[1],
                                       evidence_graph=graph, current_artifacts=arguments["current_artifacts"])
    return SimpleNamespace(project=project, reviewer=reviewer, model=model, extractions=extractions, observations=observations,
        polarity=polarity, catalog=catalog, graph=graph, arguments=arguments, candidates=candidates)


@pytest.mark.parametrize("mode", ["identity", "rotation", "shear"])
def test_real_registered_masks_yield_only_an_unordered_geometry_candidate(mode):
    q = mos_case(mode)
    assert len(q.candidates) == 1
    candidate = q.candidates[0]
    assert candidate.state == "candidate"
    assert set(candidate.diffusion_terminal_candidates) == {c.id for c in q.extractions[1].components}
    assert candidate.gate_component_id == q.extractions[0].components[0].id
    assert len(candidate.gate_polygon_target_m) == 4
    assert "polarity_not_inferred" in candidate.uncertainties
    assert "source_drain_roles_unassigned" in candidate.uncertainties
    assert len(q.graph["nets"]) == 3
    assert q.graph["connections"] == []  # A geometric gate crossing is not a via.
    assert MosCandidate.from_dict(json.loads(json.dumps(candidate.to_dict()))) == candidate


@pytest.mark.parametrize("options,reason", [
    ({"continuous": True}, "continuous_diffusion_under_gate_requires_reviewed_channel_partition"),
    ({"complex_gate": True}, "nonrectangular_shared_gate_geometry_not_supported"),
    ({"multiple": True}, "multiple_channels_or_ambiguous_diffusion_terminals"),
])
def test_unsupported_or_continuous_diffusion_geometry_is_not_invented_as_a_mos(options, reason):
    q = mos_case(**options)
    candidate = q.candidates[0]
    assert candidate.state == "unresolved"
    assert candidate.reason == reason
    assert candidate.diffusion_terminal_candidates == ()
    if options.get("continuous"):
        assert candidate.channel_overlap_components
        assert len(q.extractions[1].components) == 1  # No silent channel cut or fake source/drain islands.


def test_candidate_stays_unknown_without_current_calibration_registration_or_material_review():
    q = mos_case()
    active = dict(q.arguments["current_artifacts"])
    active[q.extractions[0].layer_id] = anchor(b"new gate mask")
    result = detect_mos_candidates(physical_model=q.model, gate_extraction=q.extractions[0], diffusion_extraction=q.extractions[1],
                                    evidence_graph=q.graph, current_artifacts=active)
    assert result[0].state == "unresolved"
    assert result[0].reason == "unreviewed_or_stale_material"
    graph = build_electrical_graph(**(q.arguments | {"component_reviews": []}))
    assert detect_mos_candidates(physical_model=q.model, gate_extraction=q.extractions[0], diffusion_extraction=q.extractions[1],
        evidence_graph=graph, current_artifacts=q.arguments["current_artifacts"])[0].state == "unresolved"


def test_nodata_and_a_real_diffusion_bypass_never_become_an_automatic_transistor():
    void = mos_case(void_gate=True)
    assert len(void.candidates) == 2
    assert all(c.state == "unresolved" and c.reason == "unobserved_pixels_touch_device_geometry" for c in void.candidates)
    assert void.extractions[0].nodata_runs
    bypass = mos_case(bypass=True)
    assert len(bypass.extractions[1].components) == 1
    assert bypass.candidates[0].state == "unresolved"
    assert bypass.candidates[0].reason == "same_diffusion_component_bypasses_channel"


@pytest.mark.parametrize("kind", ["NMOS", "PMOS"])
def test_explicit_review_produces_three_separate_pins_and_strict_edif_device_roundtrip(tmp_path, kind):
    from krona.analysis.structural.model import DeviceKind
    from krona.analysis.structural.parsers import EdifStructuralParser
    from krona.bootstrap import logic_functions_to_dict
    from krona.infrastructure.edif_repository import EdifRepository

    q = mos_case()
    source, drain = [c.id for c in q.extractions[1].components]
    realization = realize_mos_candidate(candidate=q.candidates[0], physical_model=q.model, evidence_graph=q.graph,
        evidence_catalog=q.catalog, instance="MX1", kind=kind, source_component_id=source, drain_component_id=drain,
        reviewed_by=q.reviewer, kind_evidence=(q.polarity,), evidence_kind="process_documentation", note="Reviewed process annotation and explicit terminal roles")
    assert realization.bulk_status == "unresolved"
    assert MosRealization.from_dict(json.loads(json.dumps(realization.to_dict()))) == realization
    ports = [TerminalBinding(q.candidates[0].gate_component_id, "A", "input", q.reviewer, (q.observations[0],), "Gate stimulus pad"),
             TerminalBinding(source, "GND", "inout", q.reviewer, (q.observations[1],), "Explicit grounded terminal"),
             TerminalBinding(drain, "Y", "output", q.reviewer, (q.observations[1],), "Explicit output terminal")]
    graph = build_electrical_graph(**(q.arguments | {"terminal_bindings": ports, "device_realizations": [realization]}))
    assert len(graph["nets"]) == 3
    assert graph["connections"] == []
    assert graph["devices"][0]["kind_evidence"][0]["sha256"] == q.polarity.sha256
    assert any(u["code"] == "device_bulk_unresolved" for u in graph["unknowns"])
    path = tmp_path / f"reviewed-{kind}.edf"
    path.write_text(export_reviewed_edif(graph))
    device = EdifStructuralParser(EdifRepository()).parse(path).devices["MX1"]
    assert device.kind is DeviceKind[kind]
    assert len(set(device.pins.values())) == 3
    assert set(device.pins) == {"GATE", "SOURCE", "DRAIN"}
    logic = logic_functions_to_dict(path)
    assert logic["meta"]["transistor_count"] == 1
    values = [r["value"] for r in logic["outputs"]["Y"]["truth_table"]]
    assert values == ([None, 0] if kind == "NMOS" else [0, None])


def test_polarity_and_source_drain_roles_cannot_be_guessed_or_reuse_mask_as_evidence():
    q = mos_case()
    source, drain = [c.id for c in q.extractions[1].components]
    args = dict(candidate=q.candidates[0], physical_model=q.model, evidence_graph=q.graph, evidence_catalog=q.catalog,
        instance="MX1", kind="NMOS", source_component_id=source, drain_component_id=drain, reviewed_by=q.reviewer,
        kind_evidence=(q.polarity,), evidence_kind="process_documentation", note="Explicit independent review")
    for change in ({"kind": "guess"}, {"source_component_id": drain}, {"kind_evidence": ()},
                   {"kind_evidence": (q.extractions[0].mask,)}, {"evidence_kind": "geometry_proximity"}):
        with pytest.raises(DomainValidationError):
            realize_mos_candidate(**(args | change))
