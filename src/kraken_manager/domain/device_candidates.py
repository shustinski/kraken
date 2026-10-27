"""Provisional geometry candidates and explicit, evidence-reviewed MOS roles."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from .common import DomainValidationError, require_non_empty, validate_uuid
from .electrical import TerminalBinding, _identifier
from .multilayer import ArtifactAnchor, xy


@dataclass(frozen=True, slots=True)
class MosCandidate:
    id: str
    project_id: str
    physical_model_revision: int
    gate_component_id: str
    gate_layer_id: str
    diffusion_layer_id: str
    gate_mask: ArtifactAnchor
    diffusion_mask: ArtifactAnchor
    gate_calibration_revision: int
    diffusion_calibration_revision: int
    registration_id: str | None
    registration_revision: int | None
    state: str
    reason: str
    diffusion_terminal_candidates: tuple[str, ...] = ()
    channel_overlap_components: tuple[str, ...] = ()
    gate_polygon_target_m: tuple[tuple[float, float], ...] = ()
    boundary_component_candidates: tuple[tuple[str, ...], ...] = ()
    uncertainties: tuple[str, ...] = (
        "polarity_not_inferred", "source_drain_roles_unassigned", "bulk_unresolved",
        "nominal_geometry_is_not_electrical_connectivity",
        "registration_residuals_apply_only_at_supplied_controls",
    )

    def __post_init__(self):
        for name in ("id", "project_id", "gate_component_id", "gate_layer_id", "diffusion_layer_id"):
            object.__setattr__(self, name, validate_uuid(getattr(self, name)))
        for value in (self.physical_model_revision, self.gate_calibration_revision, self.diffusion_calibration_revision):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DomainValidationError("device coordinate revision must be nonnegative integer")
        if self.registration_id is not None:
            object.__setattr__(self, "registration_id", validate_uuid(self.registration_id))
            if isinstance(self.registration_revision, bool) or not isinstance(self.registration_revision, int) or self.registration_revision < 0:
                raise DomainValidationError("registration revision is required")
        elif self.registration_revision is not None:
            raise DomainValidationError("registration identity is required")
        if self.state not in {"candidate", "unresolved"}:
            raise DomainValidationError("geometry may propose a device, never confirm one")
        object.__setattr__(self, "reason", require_non_empty(self.reason, field="candidate reason"))
        for name in ("diffusion_terminal_candidates", "channel_overlap_components"):
            values = tuple(validate_uuid(v) for v in getattr(self, name))
            if len(values) != len(set(values)):
                raise DomainValidationError("device candidate components must be unique")
            object.__setattr__(self, name, values)
        if self.state == "candidate" and len(self.diffusion_terminal_candidates) != 2:
            raise DomainValidationError("single-channel candidate requires two distinct unassigned diffusion regions")
        object.__setattr__(self, "gate_polygon_target_m", tuple(xy(p) for p in self.gate_polygon_target_m))
        boundaries = tuple(tuple(validate_uuid(identifier) for identifier in side) for side in self.boundary_component_candidates)
        if boundaries and len(boundaries) != 4:
            raise DomainValidationError("gate boundary candidates require four explicitly indexed sides")
        object.__setattr__(self, "boundary_component_candidates", boundaries)
        object.__setattr__(self, "uncertainties", tuple(self.uncertainties))

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        for name in ("gate_mask", "diffusion_mask"):
            values[name] = ArtifactAnchor(**values[name])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class MosRealization:
    candidate: MosCandidate
    instance: str
    kind: str
    source_component_id: str
    drain_component_id: str
    reviewed_by: str
    kind_evidence: tuple[ArtifactAnchor, ...]
    evidence_kind: str
    note: str
    terminals: tuple[TerminalBinding, ...]
    bulk_status: str = "unresolved"

    def __post_init__(self):
        object.__setattr__(self, "instance", _identifier(self.instance))
        if self.kind not in {"NMOS", "PMOS"}:
            raise DomainValidationError("NMOS/PMOS must be explicitly reviewed, not inferred from layer names")
        for name in ("source_component_id", "drain_component_id", "reviewed_by"):
            object.__setattr__(self, name, validate_uuid(getattr(self, name)))
        if (self.source_component_id == self.drain_component_id or
                {self.source_component_id, self.drain_component_id} != set(self.candidate.diffusion_terminal_candidates)):
            raise DomainValidationError("source/drain assignment must select the two distinct observed terminal candidates")
        if self.candidate.state != "candidate":
            raise DomainValidationError("unresolved geometry cannot realize a device")
        if self.evidence_kind not in {"implant_observation", "process_documentation", "engineer_device_annotation"}:
            raise DomainValidationError("device kind requires an explicit independent evidence category")
        object.__setattr__(self, "kind_evidence", tuple(self.kind_evidence))
        if not self.kind_evidence or any(a.sha256 in {self.candidate.gate_mask.sha256, self.candidate.diffusion_mask.sha256}
                                         for a in self.kind_evidence):
            raise DomainValidationError("device-kind review requires independent evidence, not conductor mask bytes")
        object.__setattr__(self, "note", require_non_empty(self.note, field="device review rationale"))
        object.__setattr__(self, "terminals", tuple(self.terminals))
        expected = {"GATE": self.candidate.gate_component_id, "SOURCE": self.source_component_id, "DRAIN": self.drain_component_id}
        if (len(self.terminals) != 3 or {t.port: t.component_id for t in self.terminals} != expected
                or any(t.instance != self.instance or t.cell != self.kind or t.reviewed_by != self.reviewed_by for t in self.terminals)):
            raise DomainValidationError("device realization requires three explicit reviewed terminal bindings")
        if self.bulk_status != "unresolved":
            raise DomainValidationError("bulk connectivity is not established by this supported detector")

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        values["candidate"] = MosCandidate.from_dict(values["candidate"])
        values["kind_evidence"] = tuple(ArtifactAnchor(**a) for a in values["kind_evidence"])
        values["terminals"] = tuple(TerminalBinding(**{**t, "evidence": tuple(ArtifactAnchor(**a) for a in t["evidence"])})
                                    for t in values["terminals"])
        return cls(**values)
