"""Physical models in the existing event store, with project-owned artifact anchors."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256
import json
from uuid import UUID, uuid5

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.errors import ConcurrencyError, ConflictError, NotFoundError
from kraken_manager.domain.common import DomainValidationError, require_non_empty, validate_uuid
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.multilayer import (
    ArtifactAnchor, ControlPoint, CoordinateCalibration, InterlayerContact,
    MultilayerModel, PhysicalLayer, RegistrationRevision,
)
from kraken_manager.domain.project import ProjectState, StructureState


class MultilayerService:
    def __init__(self, events, projections, fit_affine=None):
        self.events, self.projections, self.fit_affine = events, projections, fit_affine

    @staticmethod
    def stream(project_id):
        return f"multilayer:{validate_uuid(str(project_id))}"

    def load(self, project_id):
        self._project(project_id)
        events = self.events.load_stream(self.stream(project_id))
        if not events:
            return MultilayerModel(str(project_id))
        model = MultilayerModel.from_dict(events[-1].payload)
        if model.project_id != str(project_id) or model.revision != events[-1].revision:
            raise DomainValidationError("multilayer stream identity/revision mismatch")
        return model

    def _project(self, project_id):
        project = self.projections.get_project(validate_uuid(str(project_id)))
        if project is None:
            raise NotFoundError("Project not found")
        return project

    def _anchor(self, project_id, version_id, layer_id, *, active=True):
        version = self.projections.get_artifact_version(validate_uuid(str(version_id)))
        series = None if version is None else self.projections.get_artifact_series(version.series_id)
        if (series is None or series.project_id != str(project_id) or series.layer_id != layer_id
                or series.archived or version.blob is None):
            raise DomainValidationError("Physical evidence must be a managed artifact on the specified project layer")
        if active:
            current = self.projections.get_active_artifact_version(series.id)
            if current is None or current.id != version.id:
                raise ConflictError("Physical input artifact revision is stale")
        return ArtifactAnchor(str(version.id), version.sha256)

    def registration_current(self, model, registration):
        if not model.registration_current(registration):
            return False
        try:
            return all(self._anchor(model.project_id, c.artifact.version_id, c.layer_id) == c.artifact
                       for c in (registration.source, registration.target))
        except (DomainValidationError, ConflictError):
            return False

    def describe(self, model):
        result = model.to_dict()
        # An idempotent replay returns its original snapshot, but its present
        # validity must still be checked against the latest physical revision.
        basis = self.load(model.project_id)
        current = {(r.id, r.revision): self.registration_current(basis, r) for r in model.registrations}
        result["assessment"] = {
            "registrations": [
                {"id": r.id, "revision": r.revision, "current": current[r.id, r.revision], "fit_status": r.fit.status}
                for r in model.registrations
            ],
            "contacts": [
                {"id": c.id, "effective_state": c.state if current[c.registration_id, c.registration_revision] else "stale",
                 "electrical_connectivity": "engineer_confirmed" if c.state == "confirmed"
                 and current[c.registration_id, c.registration_revision] else "unknown"}
                for c in model.contacts
            ],
            "automatic_recalculation": False,
            "stitches": [
                {"id": s.id, "revision": s.revision, "current": self.stitch_current(basis, s),
                 "geometry_status": "determined" if s.ready else "unresolved", "render_status":
                    next((r.state for r in reversed(basis.stitch_renders) if (r.stitch_id, r.stitch_revision) == (s.id, s.revision)), "not_computed")}
                for s in model.stitches
            ],
        }
        return result

    def stitch_current(self, model, stitch):
        layer = self.projections.get_layer(stitch.layer_id)
        if layer is None or layer.state is StructureState.ARCHIVED:
            return False
        if stitch.stack_revision != model.stack_revision or any(
                s.id == stitch.id and s.revision > stitch.revision for s in model.stitches):
            return False
        try:
            anchors = [(stitch.plan, stitch.layer_id)] + [
                (p.calibration.artifact, p.calibration.layer_id) for p in stitch.tiles]
            return all(self._anchor(model.project_id, anchor.version_id, layer_id) == anchor for anchor, layer_id in anchors)
        except (DomainValidationError, ConflictError):
            return False

    @staticmethod
    def _fields(value, allowed):
        if not isinstance(value, Mapping) or set(value) - set(allowed):
            raise DomainValidationError("Unknown physical model command fields")
        return dict(value)

    def mutate(self, *, project_id, principal, storage, roles, expected_revision, operation,
               arguments, idempotency_key, gitlab_identity_verified=False):
        allowed = {"set_stack", "calibrate", "register_affine", "propose_contact", "review_contact", "plan_layer_stitch",
                   "request_layer_stitch", "cancel_layer_stitch"}
        if operation not in allowed:
            raise DomainValidationError("Unknown multilayer operation")
        project = self._project(project_id)
        if project.state is not ProjectState.ACTIVE:
            raise ConflictError("Archived projects cannot change physical models")
        AuthorizationPolicy().require(
            principal=principal, storage=storage, roles=roles,
            permission=(Permission.ACCEPT_REVIEW if operation == "review_contact" else Permission.RUN_PLUGIN
                        if operation in {"request_layer_stitch", "cancel_layer_stitch"} else Permission.MANAGE_STRUCTURE),
            gitlab_identity_verified=gitlab_identity_verified,
        )
        require_non_empty(idempotency_key, field="idempotency_key")
        try:
            command_hash = sha256(json.dumps(
                [operation, arguments, str(principal.id)], sort_keys=True, allow_nan=False,
            ).encode()).hexdigest()
        except (TypeError, ValueError) as exc:
            raise DomainValidationError("Physical commands must contain finite JSON values") from exc
        prior = self.events.find_by_idempotency_key(project.id, idempotency_key)
        if prior:
            event = prior[-1]
            if event.stream_id != self.stream(project.id) or event.payload.get("command_sha256") != command_hash:
                raise ConflictError("Idempotency key belongs to another physical model command")
            return MultilayerModel.from_dict(event.payload)
        model = self.load(project.id)
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision != model.revision:
            raise ConcurrencyError(f"Expected physical model revision {expected_revision}, found {model.revision}")

        def identity(suffix):
            return str(uuid5(UUID(str(project.id)), f"physical:{idempotency_key}:{suffix}"))

        if operation == "set_stack":
            args = self._fields(arguments, {"layers"})
            layers = tuple(PhysicalLayer(**self._fields(
                layer, {"layer_id", "layer_type", "z_m", "thickness_m", "conductive"},
            )) for layer in args["layers"])
            if not layers:
                raise DomainValidationError("Physical stack cannot be empty")
            for layer in layers:
                actual = self.projections.get_layer(layer.layer_id)
                if (actual is None or actual.project_id != project.id or actual.state is StructureState.ARCHIVED
                        or actual.type != layer.layer_type):
                    raise DomainValidationError("Physical stack layer/type is outside active project structure")
            changed = replace(model, layers=layers, stack_revision=model.stack_revision + 1)
        elif operation == "calibrate":
            args = self._fields(arguments, {"layer_id", "artifact_version_id", "pixel_size_m", "origin_m", "y_axis", "uncertainty_m"})
            version_id = args.pop("artifact_version_id")
            layer_id = validate_uuid(args["layer_id"])
            anchor = self._anchor(project.id, version_id, layer_id)
            previous = next((c for c in model.calibrations if c.layer_id == layer_id), None)
            calibration = CoordinateCalibration(artifact=anchor, revision=0 if previous is None else previous.revision + 1, **args)
            changed = replace(model, calibrations=tuple(c for c in model.calibrations if c.layer_id != layer_id) + (calibration,))
        elif operation == "register_affine":
            args = self._fields(arguments, {"source_layer_id", "target_layer_id", "control_points", "max_error_m", "max_condition"})
            source, target = model.calibration(args["source_layer_id"]), model.calibration(args["target_layer_id"])
            for calibration in (source, target):
                if self._anchor(project.id, calibration.artifact.version_id, calibration.layer_id) != calibration.artifact:
                    raise ConflictError("Physical calibration artifact changed")
            points = tuple(ControlPoint(**self._fields(p, {"source_xy", "target_xy", "uncertainty_m"})) for p in args["control_points"])
            if self.fit_affine is None:
                raise DomainValidationError("Affine registration solver is unavailable")
            fit = self.fit_affine(source, target, points, max_error_m=args["max_error_m"], max_condition=args.get("max_condition", 1e6))
            registration_id = str(uuid5(UUID(str(project.id)), f"registration:{source.layer_id}:{target.layer_id}"))
            previous = [r.revision for r in model.registrations if r.id == registration_id]
            registration = RegistrationRevision(registration_id, max(previous, default=-1) + 1, model.stack_revision,
                                                source, target, points, fit)
            changed = replace(model, registrations=(*model.registrations, registration))
        elif operation == "plan_layer_stitch":
            from .stitch_plans import plan_layer_stitch
            changed = plan_layer_stitch(self, model, arguments)
        elif operation == "request_layer_stitch":
            from .stitch_requests import request_stitch
            changed = request_stitch(self, model, arguments, identity("stitch-render"), str(principal.id))
        elif operation == "cancel_layer_stitch":
            from .stitch_requests import cancel_stitch
            changed = cancel_stitch(self, model, arguments)
        elif operation == "propose_contact":
            args = self._fields(arguments, {"source_layer_id", "target_layer_id", "contact_layer_id", "registration_id",
                                           "registration_revision", "position_target_m", "radius_m"})
            changed = replace(model, contacts=(*model.contacts, InterlayerContact(id=identity("contact"), **args)))
        else:
            args = self._fields(arguments, {"contact_id", "decision", "evidence_version_ids", "review_note"})
            contact = next((c for c in model.contacts if c.id == args["contact_id"]), None)
            if contact is None:
                raise NotFoundError("Interlayer contact not found")
            if args["decision"] not in {"confirmed", "rejected"}:
                raise DomainValidationError("Contact review decision must be confirmed or rejected")
            reg = next(r for r in model.registrations if (r.id, r.revision) == (contact.registration_id, contact.registration_revision))
            if args["decision"] == "confirmed" and not self.registration_current(model, reg):
                raise ConflictError("Stale physical registration cannot confirm a contact")
            evidence = tuple(self._anchor(project.id, v, contact.contact_layer_id, active=False)
                             for v in args.get("evidence_version_ids", ()))
            updated = replace(contact, state=args["decision"], evidence=evidence,
                              reviewed_by=str(principal.id), review_note=args["review_note"])
            changed = replace(model, contacts=tuple(updated if c.id == updated.id else c for c in model.contacts))
        changed = replace(changed, revision=model.revision + 1)
        payload = changed.to_dict()
        payload.update(operation=operation, command_sha256=command_hash)
        event = EventEnvelope.create(
            stream_id=self.stream(project.id), project_id=project.id, revision=changed.revision,
            event_type="MultilayerModelChangedV1", payload=payload,
            actor=ActorSnapshot.from_principal(principal), idempotency_key=idempotency_key,
        )
        self.events.append(event.stream_id, expected_revision=model.revision, events=(event,))
        return changed
