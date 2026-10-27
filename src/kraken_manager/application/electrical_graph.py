"""Build a partial electrical graph from authoritative reviewed evidence records.

This pure builder has no database authority: adapters must obtain review records
and ProjectEvidence from authenticated, versioned project events/artifacts. It
never creates connections from distance, overlap or equal frame numbers.
"""
from __future__ import annotations

from collections import defaultdict
import base64
from dataclasses import asdict
from hashlib import sha256
import json
from math import hypot

from kraken_manager.domain.common import DomainValidationError


def _plain(value):
    return json.loads(json.dumps(value))


def _observation_intersects(component, calibration, point, radius, transform=None):
    """Reject an inconsistent explicit assignment; this never selects a region.

    Pixel coordinates denote centres. Each RLE run has a rectangular pixel-cell
    support; source supports are mapped by the recorded affine transform.
    """
    for y, start, stop in component.runs:
        polygon = [calibration.to_physical(p) for p in (
            (start - 0.5, y - 0.5), (stop - 0.5, y - 0.5),
            (stop - 0.5, y + 0.5), (start - 0.5, y + 0.5),
        )]
        if transform is not None:
            polygon = [transform.apply(p) for p in polygon]
        signs, distances = [], []
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            signs.append(dx * (point[1] - a[1]) - dy * (point[0] - a[0]))
            length2 = dx * dx + dy * dy
            fraction = max(0, min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2)) if length2 else 0
            distances.append(hypot(point[0] - a[0] - fraction * dx, point[1] - a[1] - fraction * dy))
        if all(s >= 0 for s in signs) or all(s <= 0 for s in signs) or min(distances) <= radius:
            return True
    return False


def _build_electrical_graph(*, physical_model, extractions, component_reviews, contact_bindings,
                            terminal_bindings, current_artifacts, evidence_catalog, device_realizations=(), component_exclusions=None):
    extractions, terminal_bindings = tuple(extractions), tuple(terminal_bindings)
    project_id = physical_model.project_id
    unknowns, conflicts, nodes, approved = [], [], [], {}
    components, extraction_for = {}, {}

    def evidence_valid(anchor, layer_id):
        record = evidence_catalog.get(anchor.version_id)
        return bool(record and record.current and record.project_id == project_id
                    and record.layer_id == layer_id and record.artifact == anchor)

    def independent(evidence, extraction):
        return bool(evidence) and all(
            evidence_valid(a, extraction.layer_id) and a.version_id != extraction.mask.version_id
            and a.sha256 != extraction.mask.sha256 for a in evidence
        )

    reviews = defaultdict(list)
    for review in component_reviews:
        reviews[review.component_id].append(review)
    layers = {layer.layer_id: layer for layer in physical_model.layers}
    for extraction in extractions:
        if extraction.project_id != project_id:
            raise DomainValidationError("conductor extraction belongs to a different project")
        try:
            current = (physical_model.calibration(extraction.layer_id) == extraction.calibration
                       and current_artifacts.get(extraction.layer_id) == extraction.mask
                       and evidence_valid(extraction.mask, extraction.layer_id))
        except DomainValidationError:
            current = False
        conductive = extraction.layer_id in layers and layers[extraction.layer_id].conductive
        if extraction.nodata_runs:
            unknowns.append({"code": "unobserved_mask_pixels", "extraction_id": extraction.id,
                             "runs": extraction.nodata_runs})
        for component in extraction.components:
            if component.id in components:
                raise DomainValidationError("component appears in multiple extraction inputs")
            components[component.id], extraction_for[component.id] = component, extraction
            candidates = list(dict.fromkeys(reviews.get(component.id, ())))
            state = "unreviewed"
            if len(candidates) > 1:
                conflicts.append({"code": "contradictory_component_reviews", "component_id": component.id})
                state = "conflicted"
            elif component.id in (component_exclusions or {}):
                state = component_exclusions[component.id]
            elif not current:
                state = "stale_or_unresolved_artifact_calibration"
            elif not conductive:
                state = "material_not_designated_conductive"
            elif candidates:
                review = candidates[0]
                if review.decision == "rejected":
                    state = "rejected"
                elif not independent(review.evidence, extraction):
                    state = "independent_material_evidence_missing"
                elif component.diagonal_bridge_count and not review.accepts_diagonal_bridges:
                    state = "diagonal_connectivity_not_reviewed"
                else:
                    state = "reviewed_conductor"
                    approved[component.id] = review
            nodes.append({"component_id": component.id, "layer_id": extraction.layer_id,
                          "mask": asdict(extraction.mask), "calibration_revision": extraction.calibration.revision,
                          "pixel_count": component.pixel_count,
                          "area_m2": component.pixel_count * extraction.calibration.pixel_size_m[0] * extraction.calibration.pixel_size_m[1],
                          "geometry_sha256": component.geometry_sha256, "state": state})
            if state != "reviewed_conductor":
                unknowns.append({"code": state, "component_id": component.id})
            if component.touches_nodata or component.touches_image_boundary:
                unknowns.append({"code": "unobserved_boundary_continuation", "component_id": component.id,
                                 "nodata": component.touches_nodata, "image_boundary": component.touches_image_boundary})
    for identifier in reviews.keys() - components.keys():
        unknowns.append({"code": "review_references_unknown_component", "component_id": identifier})

    contacts = {contact.id: contact for contact in physical_model.contacts}
    registrations = {(r.id, r.revision): r for r in physical_model.registrations}
    bindings_by_contact = defaultdict(list)
    for binding in contact_bindings:
        bindings_by_contact[binding.contact_id].append(binding)
    eligible = []
    for identifier, bindings in bindings_by_contact.items():
        pairs = {(b.source_component_id, b.target_component_id) for b in bindings}
        if len(pairs) > 1:
            conflicts.append({"code": "contradictory_contact_bindings", "contact_id": identifier,
                              "component_pairs": sorted(pairs)})
            continue
        for binding in dict.fromkeys(bindings):
            contact = contacts.get(identifier)
            source, target = binding.source_component_id, binding.target_component_id
            if contact is None or contact.state != "confirmed":
                unknowns.append({"code": "contact_not_engineer_confirmed", "contact_id": identifier})
                continue
            reg = registrations[contact.registration_id, contact.registration_revision]
            current = physical_model.registration_current(reg) and all(
                current_artifacts.get(c.layer_id) == c.artifact for c in (reg.source, reg.target)
            )
            if not current or reg.fit.status != "fitted":
                unknowns.append({"code": "stale_or_undetermined_contact_registration", "contact_id": identifier})
                continue
            if source not in approved or target not in approved:
                unknowns.append({"code": "contact_component_not_reviewed", "contact_id": identifier})
                continue
            sx, tx = extraction_for[source], extraction_for[target]
            if sx.calibration != reg.source or tx.calibration != reg.target:
                conflicts.append({"code": "contact_component_coordinate_revision_mismatch", "contact_id": identifier})
                continue
            if not (_observation_intersects(components[source], sx.calibration, contact.position_target_m, contact.radius_m, reg.fit)
                    and _observation_intersects(components[target], tx.calibration, contact.position_target_m, contact.radius_m)):
                conflicts.append({"code": "contact_component_observation_mismatch", "contact_id": identifier,
                                  "source_component_id": source, "target_component_id": target})
                continue
            if (not binding.evidence or not set(binding.evidence) <= set(contact.evidence)
                    or not all(evidence_valid(a, contact.contact_layer_id) for a in contact.evidence)):
                unknowns.append({"code": "contact_binding_evidence_unresolved", "contact_id": identifier})
                continue
            eligible.append({"contact_id": identifier, "source": source, "target": target,
                             "reason": binding.note, "reviewed_by": binding.reviewed_by,
                             "contact_reviewed_by": contact.reviewed_by,
                             "evidence": [asdict(a) for a in binding.evidence],
                             "registration_id": reg.id, "registration_revision": reg.revision,
                             "coordinate_check": "selected_components_intersect_observation_footprint",
                             "source_calibration_revision": sx.calibration.revision,
                             "target_calibration_revision": tx.calibration.revision})

    def grouping(edges):
        parent = {identifier: identifier for identifier in approved}

        def find(identifier):
            while parent[identifier] != identifier:
                parent[identifier] = parent[parent[identifier]]
                identifier = parent[identifier]
            return identifier

        for edge in edges:
            left, right = find(edge["source"]), find(edge["target"])
            parent[max(left, right)] = min(left, right)
        return {identifier: find(identifier) for identifier in parent}

    tentative = grouping(eligible)
    named_groups = defaultdict(set)
    for identifier, review in approved.items():
        if review.declared_net_name:
            named_groups[tentative[identifier]].add(review.declared_net_name)
    blocked = {group for group, names in named_groups.items() if len(names) > 1}
    for group in sorted(blocked):
        conflicts.append({"code": "contradictory_confirmed_net_identities", "net_names": sorted(named_groups[group]),
                          "components": sorted(identifier for identifier in approved if tentative[identifier] == group),
                          "contact_ids": sorted({e["contact_id"] for e in eligible if tentative[e["source"]] == group})})
    connections = [edge for edge in eligible if tentative[edge["source"]] not in blocked]
    groups = grouping(connections)
    net_members = defaultdict(list)
    for identifier, group in groups.items():
        net_members[group].append(identifier)
    nets, net_for_component = [], {}
    for group, members in sorted(net_members.items()):
        members = sorted(members)
        net_id = "N_" + sha256("|".join(members).encode()).hexdigest()[:24]
        for identifier in members:
            net_for_component[identifier] = net_id
        identities = {approved[c].declared_net_name for c in members if approved[c].declared_net_name}
        nets.append({"id": net_id, "declared_name": next(iter(identities)) if len(identities) == 1 else None,
                     "component_ids": members, "material_reviews": [asdict(approved[c]) for c in members],
                     "connections": [edge for edge in connections if edge["source"] in members]})
    # Equal labels on disconnected islands never create an electrical edge.
    name_nets = defaultdict(list)
    for net in nets:
        if net["declared_name"]:
            name_nets[net["declared_name"]].append(net["id"])
    for name, ids in name_nets.items():
        if len(ids) > 1:
            unknowns.append({"code": "same_name_disconnected_regions", "declared_name": name, "net_ids": ids})

    devices = []
    for device in device_realizations:
        candidate = device.candidate
        identifiers = {candidate.gate_component_id, device.source_component_id, device.drain_component_id}
        if (candidate.project_id != project_id or candidate.physical_model_revision != physical_model.revision
                or not identifiers <= approved.keys()):
            unknowns.append({"code": "device_candidate_or_material_review_stale", "instance": device.instance,
                             "candidate_id": candidate.id})
            continue
        if not all(evidence_catalog.get(a.version_id) and evidence_catalog[a.version_id].current
                   and evidence_catalog[a.version_id].project_id == project_id
                   and evidence_catalog[a.version_id].artifact == a for a in device.kind_evidence):
            unknowns.append({"code": "device_kind_evidence_unresolved", "instance": device.instance})
            continue
        if not all(independent(t.evidence, extraction_for[t.component_id]) for t in device.terminals):
            unknowns.append({"code": "device_terminal_material_evidence_unresolved", "instance": device.instance})
            continue
        terminal_bindings += device.terminals
        devices.append(device.to_dict())
        unknowns.append({"code": "device_bulk_unresolved", "instance": device.instance,
                         "candidate_id": candidate.id})
    terminals, pins = [], defaultdict(list)
    for terminal in terminal_bindings:
        pins[terminal.instance, terminal.port].append(terminal)
    instance_cells = defaultdict(set)
    for terminal in terminal_bindings:
        if terminal.instance:
            instance_cells[terminal.instance].add(terminal.cell)
    conflicted_instances = {instance for instance, cells in instance_cells.items() if len(cells) > 1}
    for instance in sorted(conflicted_instances):
        conflicts.append({"code": "conflicting_instance_cell_declarations", "instance": instance})
    for (instance, port), bindings in sorted(pins.items(), key=lambda item: (item[0][0] or "", item[0][1])):
        candidate_nets = {net_for_component.get(b.component_id) for b in bindings}
        if instance in conflicted_instances or len(candidate_nets) > 1 or len({b.direction for b in bindings}) > 1:
            conflicts.append({"code": "conflicting_terminal_bindings", "instance": instance, "port": port})
            continue
        for terminal in dict.fromkeys(bindings):
            if terminal.component_id not in approved or not independent(terminal.evidence, extraction_for[terminal.component_id]):
                unknowns.append({"code": "terminal_evidence_or_component_unresolved", "instance": instance, "port": port})
                continue
            terminals.append({**asdict(terminal), "net_id": net_for_component[terminal.component_id]})
    return _plain({
        "schema_version": 1, "project_id": project_id, "physical_model_revision": physical_model.revision,
        "unit": "m", "electrical_completeness": "partial_reviewed_subgraph",
        "extractions": [{"id": e.id, "mask": asdict(e.mask), "layer_id": e.layer_id,
                         "calibration": asdict(e.calibration), "connectivity": e.connectivity,
                         "foreground_value": e.foreground_value, "nodata_value": e.nodata_value,
                         "algorithm": e.algorithm, "backend": "scipy.ndimage.label",
                         "backend_version": e.backend_version} for e in extractions],
        "nodes": nodes, "connections": connections, "nets": nets, "terminals": terminals, "devices": devices,
        "conflicts": conflicts, "unknowns": unknowns,
    })


def build_electrical_graph(*, physical_model, extractions, component_reviews, contact_bindings,
                           terminal_bindings, current_artifacts, evidence_catalog, device_realizations=()):
    """Apply device-channel uncertainty before publishing conductor continuity."""
    from kraken_manager.application.device_candidates import detect_mos_candidates
    from kraken_manager.domain.project import LayerType

    arguments = dict(physical_model=physical_model, extractions=tuple(extractions), component_reviews=tuple(component_reviews),
                     contact_bindings=tuple(contact_bindings), terminal_bindings=tuple(terminal_bindings),
                     current_artifacts=current_artifacts, evidence_catalog=evidence_catalog,
                     device_realizations=tuple(device_realizations))
    graph = _build_electrical_graph(**arguments)
    types = {layer.layer_id: layer.layer_type for layer in physical_model.layers}
    candidates = []
    for gate in arguments["extractions"]:
        if types.get(gate.layer_id) is not LayerType.GATE:
            continue
        for diffusion in arguments["extractions"]:
            if types.get(diffusion.layer_id) is LayerType.DIFFUSION:
                candidates.extend(detect_mos_candidates(physical_model=physical_model, gate_extraction=gate,
                    diffusion_extraction=diffusion, evidence_graph=graph, current_artifacts=current_artifacts))
    exclusions = {identifier: "device_channel_partition_required" for candidate in candidates for identifier in candidate.channel_overlap_components}
    if exclusions:
        graph = _build_electrical_graph(**arguments, component_exclusions=exclusions)
    realized = {device["candidate"]["id"] for device in graph["devices"]}
    for candidate in candidates:
        if candidate.state == "unresolved" or candidate.id not in realized:
            graph["unknowns"].append({"code": "device_geometry_unresolved" if candidate.state == "unresolved" else "device_candidate_requires_independent_review",
                                     "candidate_id": candidate.id, "gate_component_id": candidate.gate_component_id,
                                     "reason": candidate.reason})
    graph["device_candidates"] = [candidate.to_dict() for candidate in candidates]
    return graph


def export_reviewed_edif(graph):
    """Export only explicit reviewed endpoints; preserve partial scope and refuse contradictions."""
    if graph.get("schema_version") != 1 or graph.get("conflicts"):
        raise DomainValidationError("Electrical graph contains unresolved contradictions")
    from kraken_manager.domain.electrical import _identifier

    terminals = graph["terminals"]
    instances, top_ports, cell_ports = {}, {}, defaultdict(dict)
    joined = defaultdict(set)
    for terminal in terminals:
        port = _identifier(terminal["port"])
        direction = terminal["direction"].upper()
        if direction not in {"INPUT", "OUTPUT", "INOUT"}:
            raise DomainValidationError("invalid terminal direction in export")
        if terminal["instance"]:
            name, cell = _identifier(terminal["instance"]), _identifier(terminal["cell"])
            instances[name] = cell
            cell_ports[cell][port] = direction
            joined[terminal["net_id"]].add(f"(portRef {port} (instanceRef {name}))")
        else:
            top_ports[port] = direction
            joined[terminal["net_id"]].add(f"(portRef {port})")
    lines = ["(edif KRAKEN_REVIEWED_PARTIAL", " (edifVersion 2 0 0) (edifLevel 0) (keywordMap (keywordLevel 0))",
             " (library REVIEWED_CELLS"]
    for cell, ports in sorted(cell_ports.items()):
        interface = " ".join(f"(port {port} (direction {direction}))" for port, direction in sorted(ports.items()))
        lines.append(f"  (cell {cell} (view NETLIST (viewType NETLIST) (interface {interface})))")
    lines.extend([" )", " (library REVIEWED_DESIGN (cell PARTIAL (view NETLIST (viewType NETLIST)",
                  "  (interface " + " ".join(f"(port {port} (direction {direction}))" for port, direction in sorted(top_ports.items())) + ")",
                  '  (property KRAKEN_SCOPE (string "partial_reviewed_subgraph"))',
                  f'  (property KRAKEN_PHYSICAL_REVISION (integer {int(graph["physical_model_revision"])}))',
                  f'  (property KRAKEN_UNRESOLVED_COUNT (integer {len(graph["unknowns"])}))', "  (contents"])
    for name, cell in sorted(instances.items()):
        lines.append(f"   (instance {name} (viewRef NETLIST (cellRef {cell} (libraryRef REVIEWED_CELLS))))")
    for net in graph["nets"]:
        identifier = _identifier(net["id"])
        proof = json.dumps({"material_reviews": net["material_reviews"], "connections": net["connections"],
                            "terminal_bindings": [t for t in terminals if t["net_id"] == identifier],
                            "devices": [d for d in graph.get("devices", []) if any(
                                t["component_id"] in net["component_ids"] for t in d["terminals"])]}, sort_keys=True)
        # ASCII base64 keeps arbitrary review notes out of EDIF syntax and is
        # portable to readers that use different string-escaping conventions.
        encoded = base64.b64encode(proof.encode()).decode("ascii")
        lines.append(f"   (net {identifier} (joined {' '.join(sorted(joined[identifier]))})"
                     f' (property KRAKEN_EVIDENCE_JSON_BASE64 (string "{encoded}")))')
    lines.extend(["  ))))", " (design KRAKEN_REVIEWED_PARTIAL (cellRef PARTIAL (libraryRef REVIEWED_DESIGN))))"])
    return "\n".join(lines) + "\n"
