"""A narrow, reviewable single-channel MOS geometry detector.

Observed rectangular gate strips and distinct diffusion regions are compared in
the recorded affine coordinate frame. Boundary intersection proposes roles only;
it never joins nets, assigns polarity, splits a conductor or invents a bulk net.
"""
from __future__ import annotations

from math import hypot
from sys import float_info
from uuid import UUID, uuid5

from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.device_candidates import MosCandidate, MosRealization
from kraken_manager.domain.electrical import TerminalBinding
from kraken_manager.domain.multilayer import ArtifactAnchor
from kraken_manager.domain.project import LayerType


def _area(polygon):
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(polygon, polygon[1:] + polygon[:1])) / 2


def _clip(polygon, bounds):
    output = polygon
    for axis, limit, sign in ((0, bounds[0], 1), (0, bounds[2], -1), (1, bounds[1], 1), (1, bounds[3], -1)):
        source, output = output, []
        if not source:
            break
        previous = source[-1]
        for point in source:
            current_in, previous_in = sign * (point[axis] - limit) >= 0, sign * (previous[axis] - limit) >= 0
            if current_in != previous_in:
                ratio = (limit - previous[axis]) / (point[axis] - previous[axis])
                output.append(tuple(previous[j] + ratio * (point[j] - previous[j]) for j in (0, 1)))
            if current_in:
                output.append(point)
            previous = point
    return output


def _boundary_span(a, b, polygon, epsilon):
    """Parameter span of a nominal gate edge inside a convex pixel-run support."""
    orientation = 1 if _area(polygon) >= 0 else -1
    lower, upper = 0.0, 1.0
    vx, vy = b[0] - a[0], b[1] - a[1]
    for p, q in zip(polygon, polygon[1:] + polygon[:1]):
        dx, dy = q[0] - p[0], q[1] - p[1]
        c0 = orientation * (dx * (a[1] - p[1]) - dy * (a[0] - p[0]))
        slope = orientation * (dx * vy - dy * vx)
        if abs(slope) <= epsilon:
            if c0 < -epsilon:
                return 0
        elif slope > 0:
            lower = max(lower, (-epsilon - c0) / slope)
        else:
            upper = min(upper, (-epsilon - c0) / slope)
    return max(0.0, upper - lower)


def _rectangle(component):
    runs = component.runs
    first = runs[0]
    if (len(runs) != runs[-1][0] - first[0] + 1
            or any((r[1], r[2]) != (first[1], first[2]) or r[0] != first[0] + i for i, r in enumerate(runs))):
        return None
    return (first[1] - 0.5, first[0] - 0.5, first[2] - 0.5, runs[-1][0] + 0.5)


def _inverse_affine(fit, point):
    a, b, tx, c, d, ty = fit.affine
    determinant = a * d - b * c
    x, y = point[0] - tx, point[1] - ty
    return ((d * x - b * y) / determinant, (-c * x + a * y) / determinant)


def detect_mos_candidates(*, physical_model, gate_extraction, diffusion_extraction, evidence_graph, current_artifacts):
    if gate_extraction.project_id != physical_model.project_id or diffusion_extraction.project_id != physical_model.project_id:
        raise DomainValidationError("device inputs belong to another project")
    layers = {layer.layer_id: layer for layer in physical_model.layers}
    if (layers.get(gate_extraction.layer_id) is None or layers[gate_extraction.layer_id].layer_type is not LayerType.GATE
            or layers.get(diffusion_extraction.layer_id) is None or layers[diffusion_extraction.layer_id].layer_type is not LayerType.DIFFUSION):
        raise DomainValidationError("supported MOS geometry requires explicit gate and diffusion layers")
    nodes = {node["component_id"]: node for node in evidence_graph["nodes"]}
    registrations = [r for r in physical_model.registrations if r.source == gate_extraction.calibration
                     and r.target == diffusion_extraction.calibration and physical_model.registration_current(r)]
    reg = registrations[0] if len(registrations) == 1 else None
    try:
        current = (evidence_graph["project_id"] == physical_model.project_id
                   and evidence_graph["physical_model_revision"] == physical_model.revision
                   and physical_model.calibration(gate_extraction.layer_id) == gate_extraction.calibration
                   and physical_model.calibration(diffusion_extraction.layer_id) == diffusion_extraction.calibration
                   and current_artifacts.get(gate_extraction.layer_id) == gate_extraction.mask
                   and current_artifacts.get(diffusion_extraction.layer_id) == diffusion_extraction.mask)
    except DomainValidationError:
        current = False
    results = []
    for gate in gate_extraction.components:
        registration_key = f"{reg.id}:{reg.revision}" if reg else "no-current-registration"
        identifier = str(uuid5(UUID(gate.id), f"mos-candidate:{diffusion_extraction.id}:{registration_key}:{physical_model.revision}"))
        values = dict(id=identifier, project_id=physical_model.project_id, physical_model_revision=physical_model.revision,
            gate_component_id=gate.id, gate_layer_id=gate_extraction.layer_id, diffusion_layer_id=diffusion_extraction.layer_id,
            gate_mask=gate_extraction.mask, diffusion_mask=diffusion_extraction.mask,
            gate_calibration_revision=gate_extraction.calibration.revision,
            diffusion_calibration_revision=diffusion_extraction.calibration.revision,
            registration_id=reg.id if reg else None, registration_revision=reg.revision if reg else None,
            state="unresolved", reason="unreviewed_or_stale_material")
        if not current:
            results.append(MosCandidate(**values))
            continue
        if reg is None or reg.fit.status != "fitted":
            values["reason"] = "current_directed_gate_to_diffusion_registration_required"
            results.append(MosCandidate(**values))
            continue
        bounds = _rectangle(gate)
        if bounds is None:
            # Even an unsupported shared gate can invalidate an unconditional
            # diffusion conductor: detect exact observed overlap, without
            # cutting channels or inventing transistor terminals.
            overlaps = set()
            for diffusion in diffusion_extraction.components:
                for y, start, stop in diffusion.runs:
                    polygon = []
                    for p in ((start - 0.5, y - 0.5), (stop - 0.5, y - 0.5), (stop - 0.5, y + 0.5), (start - 0.5, y + 0.5)):
                        physical = _inverse_affine(reg.fit, diffusion_extraction.calibration.to_physical(p))
                        calibration = gate_extraction.calibration
                        polygon.append(((physical[0] - calibration.origin_m[0]) / calibration.pixel_size_m[0],
                                        (physical[1] - calibration.origin_m[1]) / calibration.pixel_size_m[1]
                                        * (1 if calibration.y_axis == "up" else -1)))
                    epsilon = float_info.epsilon * max(1, *(abs(v) for p in polygon for v in p)) * 256
                    if any(abs(_area(_clip(polygon, (a - 0.5, gy - 0.5, b - 0.5, gy + 0.5)))) > epsilon
                           for gy, a, b in gate.runs):
                        overlaps.add(diffusion.id)
                        break
            values["channel_overlap_components"] = tuple(sorted(overlaps))
            values["reason"] = "nonrectangular_shared_gate_geometry_not_supported"
            results.append(MosCandidate(**values))
            continue
        left, bottom, right, top = bounds
        corners = [(left, bottom), (right, bottom), (right, top), (left, top)]
        projected = [reg.fit.apply(gate_extraction.calibration.to_physical(p)) for p in corners]
        values["gate_polygon_target_m"] = tuple(projected)
        epsilon = float_info.epsilon * max(1, *(abs(v) for v in bounds)) * 128
        sides = [set(), set(), set(), set()]
        overlaps, unresolved_diffusion = set(), set()
        for diffusion in diffusion_extraction.components:
            reviewed = nodes.get(diffusion.id, {}).get("state") == "reviewed_conductor"
            component_overlap, touched = 0.0, set()
            for y, start, stop in diffusion.runs:
                pixels = [(start - 0.5, y - 0.5), (stop - 0.5, y - 0.5),
                          (stop - 0.5, y + 0.5), (start - 0.5, y + 0.5)]
                polygon = []
                for p in pixels:
                    physical = _inverse_affine(reg.fit, diffusion_extraction.calibration.to_physical(p))
                    calibration = gate_extraction.calibration
                    polygon.append(((physical[0] - calibration.origin_m[0]) / calibration.pixel_size_m[0],
                                    (physical[1] - calibration.origin_m[1]) / calibration.pixel_size_m[1]
                                    * (1 if calibration.y_axis == "up" else -1)))
                component_overlap += abs(_area(_clip(polygon, bounds)))
                for edge, (a, b) in enumerate(zip(corners, corners[1:] + corners[:1])):
                    if _boundary_span(a, b, polygon, epsilon) * hypot(b[0] - a[0], b[1] - a[1]) > epsilon * 16:
                        touched.add(edge)
            if component_overlap > epsilon * max(right - left, top - bottom, 1) * 16:
                overlaps.add(diffusion.id)
            if not reviewed and (touched or diffusion.id in overlaps):
                unresolved_diffusion.add(diffusion.id)
            if reviewed:
                for edge in touched:
                    sides[edge].add(diffusion.id)
        values["channel_overlap_components"] = tuple(sorted(overlaps))
        values["boundary_component_candidates"] = tuple(tuple(sorted(side)) for side in sides)
        if gate.touches_nodata or any(c.touches_nodata for c in diffusion_extraction.components if c.id in set.union(*sides)):
            values["reason"] = "unobserved_pixels_touch_device_geometry"
        elif overlaps:
            values["reason"] = "continuous_diffusion_under_gate_requires_reviewed_channel_partition"
        elif nodes.get(gate.id, {}).get("state") != "reviewed_conductor" or unresolved_diffusion:
            values["reason"] = "unreviewed_diffusion_terminal_geometry"
        else:
            pairs = []
            ambiguous = False
            bypass = False
            for first, second in ((0, 2), (1, 3)):
                if sides[first] and sides[second]:
                    if len(sides[first]) != 1 or len(sides[second]) != 1:
                        ambiguous = True
                    elif sides[first] == sides[second]:
                        bypass = True
                    else:
                        pairs.append(tuple(sorted(sides[first] | sides[second])))
            if bypass:
                values["reason"] = "same_diffusion_component_bypasses_channel"
            elif ambiguous or len(set(pairs)) > 1:
                values["reason"] = "multiple_channels_or_ambiguous_diffusion_terminals"
            elif len(pairs) == 1:
                values.update(state="candidate", reason="two_distinct_reviewed_regions_meet_opposite_nominal_gate_boundaries",
                              diffusion_terminal_candidates=pairs[0])
            else:
                values["reason"] = "two_opposite_observed_terminal_boundaries_not_established"
        results.append(MosCandidate(**values))
    return tuple(results)


def realize_mos_candidate(*, candidate, physical_model, evidence_graph, evidence_catalog, instance, kind,
                          source_component_id, drain_component_id, reviewed_by, kind_evidence, evidence_kind, note):
    if (candidate.physical_model_revision != physical_model.revision
            or evidence_graph["physical_model_revision"] != physical_model.revision):
        raise DomainValidationError("device candidate coordinate revision is stale")
    for anchor in kind_evidence:
        evidence = evidence_catalog.get(anchor.version_id)
        if evidence is None or not evidence.current or evidence.project_id != candidate.project_id or evidence.artifact != anchor:
            raise DomainValidationError("independent device-kind evidence is unresolved or inactive")
    nodes = {node["component_id"]: node for node in evidence_graph["nodes"]}
    material = {r["component_id"]: r for net in evidence_graph["nets"] for r in net["material_reviews"]}
    terminals = []
    for port, identifier in (("GATE", candidate.gate_component_id), ("SOURCE", source_component_id), ("DRAIN", drain_component_id)):
        if nodes.get(identifier, {}).get("state") != "reviewed_conductor" or identifier not in material:
            raise DomainValidationError("device ports require current independently reviewed conductor components")
        proof = tuple(ArtifactAnchor(**a) for a in material[identifier]["evidence"])
        terminals.append(TerminalBinding(identifier, port, "input" if port == "GATE" else "inout", reviewed_by, proof,
            f"Device candidate {candidate.id}; explicit {port} assignment: {note}", instance, kind))
    return MosRealization(candidate, instance, kind, source_component_id, drain_component_id, reviewed_by,
                          tuple(kind_evidence), evidence_kind, note, tuple(terminals))
