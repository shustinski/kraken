"""Authenticated electrical evidence commands on the existing event/blob ports."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from typing import Protocol

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.device_candidates import realize_mos_candidate
from kraken_manager.application.electrical_graph import build_electrical_graph, export_reviewed_edif
from kraken_manager.application.errors import ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.application.multilayer import MultilayerService
from kraken_manager.domain.common import DomainValidationError, require_non_empty, validate_uuid
from kraken_manager.domain.device_candidates import MosCandidate
from kraken_manager.domain.electrical import (
    ComponentReview, ConductorExtraction, ContactBinding, ElectricalModel, ProjectEvidence, TerminalBinding,
)
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import ProjectState


class ConductorExtractor(Protocol):
    def __call__(self, data: bytes, *, project_id, mask, calibration, connectivity,
                 foreground_value, nodata_value=None) -> ConductorExtraction: ...


class ElectricalService:
    def __init__(self, events, projections, *, extractor: ConductorExtractor | None = None, blob_store=None):
        self.events, self.projections, self.extractor, self.blob_store = events, projections, extractor, blob_store
        self.physical = MultilayerService(events, projections)

    @staticmethod
    def stream(project_id):
        return f"electrical:{validate_uuid(str(project_id))}"

    def load(self, project_id):
        self.physical._project(project_id)
        events = self.events.load_stream(self.stream(project_id))
        if not events:
            return ElectricalModel(str(project_id))
        model = ElectricalModel.from_dict(events[-1].payload)
        if model.project_id != str(project_id) or model.revision != events[-1].revision:
            raise DomainValidationError("electrical stream identity/revision mismatch")
        return model

    def describe(self, model):
        original = model
        model = self.load(model.project_id)
        physical = self.physical.load(model.project_id)
        catalog, active = {}, {}
        component_layer = {c.id: e.layer_id for e in model.extractions for c in e.components}

        def evidence(anchor, layer_id):
            try:
                observed = self.physical._anchor(model.project_id, anchor.version_id, layer_id, active=False)
                if observed == anchor:
                    try:
                        self.physical._anchor(model.project_id, anchor.version_id, layer_id, active=True)
                        current = True
                    except (DomainValidationError, ConflictError):
                        current = False
                    catalog[anchor.version_id] = ProjectEvidence(model.project_id, layer_id, observed, current)
            except (DomainValidationError, ConflictError):
                pass  # Unresolvable historical evidence remains explicitly unknown in the graph.

        for calibration in physical.calibrations:
            try:
                active[calibration.layer_id] = self.physical._anchor(model.project_id, calibration.artifact.version_id, calibration.layer_id)
            except (DomainValidationError, ConflictError):
                pass
        for extraction in model.extractions:
            evidence(extraction.mask, extraction.layer_id)
        for review in (*model.component_reviews, *model.terminal_bindings):
            layer_id = component_layer.get(review.component_id)
            if layer_id:
                for anchor in review.evidence:
                    evidence(anchor, layer_id)
        contacts = {c.id: c for c in physical.contacts}
        for binding in model.contact_bindings:
            if binding.contact_id in contacts:
                for anchor in binding.evidence:
                    evidence(anchor, contacts[binding.contact_id].contact_layer_id)
        for device in model.device_realizations:
            for anchor in device.kind_evidence:
                version = self.projections.get_artifact_version(anchor.version_id)
                series = None if version is None else self.projections.get_artifact_series(version.series_id)
                if series is not None and series.layer_id is not None:
                    evidence(anchor, str(series.layer_id))
            for terminal in device.terminals:
                layer_id = component_layer.get(terminal.component_id)
                if layer_id:
                    for anchor in terminal.evidence:
                        evidence(anchor, layer_id)
        graph = build_electrical_graph(
            physical_model=physical, extractions=model.extractions, component_reviews=model.component_reviews,
            contact_bindings=model.contact_bindings, terminal_bindings=model.terminal_bindings,
            current_artifacts=active, evidence_catalog=catalog,
            device_realizations=model.device_realizations,
        )
        return {**original.to_dict(), "physical_model_revision": physical.revision,
                "graph_electrical_revision": model.revision, "graph": graph,
                "device_candidates": graph["device_candidates"]}

    def export(self, project_id):
        snapshot = self.describe(self.load(project_id))
        data = export_reviewed_edif(snapshot["graph"])
        return {"format": "edif", "content": data, "sha256": sha256(data.encode()).hexdigest(),
                "electrical_revision": snapshot["electrical"]["revision"],
                "physical_model_revision": snapshot["physical_model_revision"],
                "scope": "partial_reviewed_subgraph", "unknowns": snapshot["graph"]["unknowns"]}

    def mutate(self, *, project_id, principal, storage, roles, expected_revision, expected_physical_revision,
               operation, arguments, idempotency_key, gitlab_identity_verified=False):
        allowed = {"extract_conductors", "review_components", "bind_contact", "retract_contact_binding",
                   "bind_terminal", "retract_terminal", "realize_device", "retract_device"}
        if operation not in allowed:
            raise DomainValidationError("unknown electrical evidence operation")
        project = self.physical._project(project_id)
        if project.state is not ProjectState.ACTIVE:
            raise ConflictError("archived projects cannot change electrical evidence")
        AuthorizationPolicy().require(
            principal=principal, storage=storage, roles=roles, gitlab_identity_verified=gitlab_identity_verified,
            permission=Permission.MANAGE_STRUCTURE if operation == "extract_conductors" else Permission.ACCEPT_REVIEW,
        )
        require_non_empty(idempotency_key, field="idempotency_key")
        try:
            fingerprint = sha256(json.dumps([operation, arguments, expected_physical_revision, str(principal.id)],
                                           sort_keys=True, allow_nan=False).encode()).hexdigest()
        except (TypeError, ValueError) as exc:
            raise DomainValidationError("electrical evidence commands require finite JSON") from exc
        prior = self.events.find_by_idempotency_key(project.id, idempotency_key)
        if prior:
            event = prior[-1]
            if event.stream_id != self.stream(project.id) or event.payload.get("command_sha256") != fingerprint:
                raise ConflictError("idempotency key belongs to another electrical evidence command")
            return ElectricalModel.from_dict(event.payload)
        model = self.load(project.id)
        physical = self.physical.load(project.id)
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision != model.revision:
            raise ConcurrencyError("electrical model revision changed")
        if (isinstance(expected_physical_revision, bool) or not isinstance(expected_physical_revision, int)
                or expected_physical_revision != physical.revision):
            raise ConcurrencyError("physical model revision changed")
        components = {c.id: (c, e) for e in model.extractions for c in e.components}

        def component(identifier):
            if identifier not in components:
                raise NotFoundError("conductor component not found")
            c, e = components[identifier]
            if physical.calibration(e.layer_id) != e.calibration:
                raise ConflictError("conductor extraction calibration is stale")
            self.physical._anchor(project.id, e.mask.version_id, e.layer_id)
            return c, e

        fields = MultilayerService._fields
        if operation == "extract_conductors":
            args = fields(arguments, {"layer_id", "mask_version_id", "connectivity", "foreground_value", "nodata_value"})
            calibration = physical.calibration(args["layer_id"])
            mask = self.physical._anchor(project.id, args["mask_version_id"], calibration.layer_id)
            if calibration.artifact != mask:
                raise DomainValidationError("binary mask requires its own version-specific calibration")
            if self.extractor is None or self.blob_store is None:
                raise DomainValidationError("conductor extraction runtime is unavailable")
            version = self.projections.get_artifact_version(mask.version_id)
            if version.size_bytes > 64 * 1024 * 1024:
                raise DomainValidationError("binary mask exceeds the 64 MiB input limit")
            chunks, size = [], 0
            for chunk in self.blob_store.iter_bytes(version.blob):
                size += len(chunk)
                if size > 64 * 1024 * 1024:
                    raise DomainValidationError("binary mask bytes exceed the 64 MiB input limit")
                chunks.append(chunk)
            data = b"".join(chunks)
            if len(data) != version.size_bytes:
                raise DomainValidationError("binary mask blob length mismatch")
            extraction = self.extractor(data, project_id=project.id, mask=mask, calibration=calibration,
                connectivity=args["connectivity"], foreground_value=args["foreground_value"], nodata_value=args.get("nodata_value"))
            changed = replace(model, extractions=tuple(e for e in model.extractions if e.layer_id != extraction.layer_id) + (extraction,))
        elif operation == "review_components":
            args = fields(arguments, {"component_ids", "decision", "evidence_version_ids", "note", "declared_net_name", "accepts_diagonal_bridges"})
            identifiers = tuple(args["component_ids"])
            if not identifiers or len(set(identifiers)) != len(identifiers):
                raise DomainValidationError("material review requires unique component identities")
            reviews = []
            for identifier in identifiers:
                _c, extraction = component(identifier)
                evidence = tuple(self.physical._anchor(project.id, v, extraction.layer_id, active=args["decision"] == "confirmed")
                                 for v in args.get("evidence_version_ids", ()))
                review = ComponentReview(identifier, str(principal.id), evidence, args["note"],
                    args["decision"], args.get("declared_net_name"), args.get("accepts_diagonal_bridges", False))
                if review.decision == "confirmed" and any(a.sha256 == extraction.mask.sha256 for a in evidence):
                    raise DomainValidationError("material evidence cannot reuse binary mask contents")
                reviews.append(review)
            changed = replace(model, component_reviews=tuple(r for r in model.component_reviews if r.component_id not in identifiers) + tuple(reviews))
        elif operation == "bind_contact":
            args = fields(arguments, {"contact_id", "source_component_id", "target_component_id", "note"})
            contact = next((c for c in physical.contacts if c.id == args["contact_id"]), None)
            if contact is None or contact.state != "confirmed":
                raise DomainValidationError("contact has not been independently confirmed")
            reg = next(r for r in physical.registrations if (r.id, r.revision) == (contact.registration_id, contact.registration_revision))
            if not self.physical.registration_current(physical, reg):
                raise ConflictError("contact physical registration is stale")
            for anchor in contact.evidence:
                self.physical._anchor(project.id, anchor.version_id, contact.contact_layer_id, active=True)
            _sc, sx = component(args["source_component_id"])
            _tc, tx = component(args["target_component_id"])
            if sx.calibration != reg.source or tx.calibration != reg.target:
                raise DomainValidationError("contact endpoints do not match registered physical coordinates")
            binding = ContactBinding(contact.id, args["source_component_id"], args["target_component_id"],
                                     str(principal.id), contact.evidence, args["note"])
            changed = replace(model, contact_bindings=(*model.contact_bindings, binding))
        elif operation == "bind_terminal":
            args = fields(arguments, {"component_id", "port", "direction", "instance", "cell", "evidence_version_ids", "note"})
            _c, extraction = component(args["component_id"])
            ids = args.pop("evidence_version_ids")
            evidence = tuple(self.physical._anchor(project.id, v, extraction.layer_id, active=True) for v in ids)
            if any(a.sha256 == extraction.mask.sha256 for a in evidence):
                raise DomainValidationError("terminal evidence cannot reuse binary mask contents")
            terminal = TerminalBinding(**args, reviewed_by=str(principal.id), evidence=evidence)
            changed = replace(model, terminal_bindings=(*model.terminal_bindings, terminal))
        elif operation == "realize_device":
            args = fields(arguments, {"candidate_id", "instance", "kind", "source_component_id", "drain_component_id",
                                       "evidence_version_ids", "evidence_kind", "note"})
            if any(device.instance == args["instance"] for device in model.device_realizations):
                raise ConflictError("device instance is already realized; explicitly retract it before replacing its review")
            if any(device.candidate.id == args["candidate_id"] for device in model.device_realizations):
                raise ConflictError("physical candidate already has a device realization; retract it before replacing its identity")
            description = self.describe(model)
            candidate_data = next((c for c in description["device_candidates"] if c["id"] == args["candidate_id"]), None)
            if candidate_data is None:
                raise NotFoundError("current device geometry candidate not found")
            candidate = MosCandidate.from_dict(candidate_data)
            kind_evidence, catalog = [], {}
            for identifier in args["evidence_version_ids"]:
                version = self.projections.get_artifact_version(validate_uuid(str(identifier)))
                series = None if version is None else self.projections.get_artifact_series(version.series_id)
                if series is None or series.layer_id not in {candidate.gate_layer_id, candidate.diffusion_layer_id}:
                    raise DomainValidationError("device-kind evidence must be on the candidate gate or diffusion project layer")
                anchor = self.physical._anchor(project.id, identifier, str(series.layer_id), active=True)
                kind_evidence.append(anchor)
                catalog[anchor.version_id] = ProjectEvidence(str(project.id), str(series.layer_id), anchor, True)
            device = realize_mos_candidate(candidate=candidate, physical_model=physical, evidence_graph=description["graph"],
                evidence_catalog=catalog, instance=args["instance"], kind=args["kind"], source_component_id=args["source_component_id"],
                drain_component_id=args["drain_component_id"], reviewed_by=str(principal.id), kind_evidence=tuple(kind_evidence),
                evidence_kind=args["evidence_kind"], note=args["note"])
            changed = replace(model, device_realizations=(*model.device_realizations, device))
        elif operation == "retract_device":
            args = fields(arguments, {"instance", "note"})
            require_non_empty(args["note"], field="device retraction reason")
            if not any(device.instance == args["instance"] for device in model.device_realizations):
                raise NotFoundError("reviewed device instance not found")
            changed = replace(model, device_realizations=tuple(device for device in model.device_realizations if device.instance != args["instance"]))
        elif operation == "retract_contact_binding":
            args = fields(arguments, {"contact_id", "note"})
            require_non_empty(args["note"], field="retraction reason")
            if not any(b.contact_id == args["contact_id"] for b in model.contact_bindings):
                raise NotFoundError("contact binding not found")
            changed = replace(model, contact_bindings=tuple(b for b in model.contact_bindings if b.contact_id != args["contact_id"]))
        else:
            args = fields(arguments, {"instance", "port", "note"})
            require_non_empty(args["note"], field="retraction reason")
            if not any((b.instance, b.port) == (args.get("instance"), args["port"]) for b in model.terminal_bindings):
                raise NotFoundError("terminal binding not found")
            changed = replace(model, terminal_bindings=tuple(b for b in model.terminal_bindings
                if (b.instance, b.port) != (args.get("instance"), args["port"])))
        changed = replace(changed, revision=model.revision + 1)
        payload = changed.to_dict()
        payload.update(operation=operation, arguments=arguments, physical_model_revision=physical.revision, command_sha256=fingerprint)
        event = EventEnvelope.create(stream_id=self.stream(project.id), project_id=project.id, revision=changed.revision,
            event_type="ElectricalModelChangedV1", payload=payload, actor=ActorSnapshot.from_principal(principal),
            idempotency_key=idempotency_key)
        self.events.append(event.stream_id, expected_revision=model.revision, events=(event,))
        return changed
