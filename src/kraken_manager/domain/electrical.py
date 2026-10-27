"""Evidence-bearing conductor candidates and explicit reviewed connections.

Pixel adjacency describes a segmentation, not an electrical proof. These values
keep the calibration/version anchors and review evidence needed by graph builders.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re

from .artifacts import validate_sha256
from .common import DomainValidationError, require_non_empty, validate_uuid
from .multilayer import ArtifactAnchor, CoordinateCalibration


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise DomainValidationError("netlist names must be EDIF identifiers")
    return value


def runs_digest(runs):
    return sha256(json.dumps(runs, separators=(",", ":")).encode()).hexdigest()


def _runs(values):
    rows = tuple(tuple(run) for run in values)
    previous = None
    for run in rows:
        if (len(run) != 3 or any(isinstance(v, bool) or not isinstance(v, int) for v in run)
                or run[0] < 0 or run[1] < 0 or run[2] <= run[1]):
            raise DomainValidationError("pixel runs require row, inclusive start, exclusive end")
        if previous is not None and (run <= previous or (run[0] == previous[0] and run[1] < previous[2])):
            raise DomainValidationError("pixel runs must be ordered and disjoint")
        previous = run
    return rows


@dataclass(frozen=True, slots=True)
class ConductorComponent:
    id: str
    runs: tuple[tuple[int, int, int], ...]
    pixel_count: int
    geometry_sha256: str
    touches_nodata: bool
    touches_image_boundary: bool
    diagonal_bridge_count: int = 0

    def __post_init__(self):
        object.__setattr__(self, "id", validate_uuid(self.id))
        object.__setattr__(self, "runs", _runs(self.runs))
        if not self.runs or isinstance(self.pixel_count, bool) or self.pixel_count != sum(b - a for _y, a, b in self.runs):
            raise DomainValidationError("component pixel count must match nonempty geometry")
        if validate_sha256(self.geometry_sha256) != runs_digest(self.runs):
            raise DomainValidationError("component geometry fingerprint mismatch")
        if not isinstance(self.touches_nodata, bool) or not isinstance(self.touches_image_boundary, bool):
            raise DomainValidationError("unknown-boundary flags must be boolean")
        if isinstance(self.diagonal_bridge_count, bool) or not isinstance(self.diagonal_bridge_count, int) or self.diagonal_bridge_count < 0:
            raise DomainValidationError("diagonal bridge count must be nonnegative")


@dataclass(frozen=True, slots=True)
class ConductorExtraction:
    id: str
    project_id: str
    layer_id: str
    mask: ArtifactAnchor
    calibration: CoordinateCalibration
    width: int
    height: int
    connectivity: int
    foreground_value: int
    components: tuple[ConductorComponent, ...]
    nodata_runs: tuple[tuple[int, int, int], ...] = ()
    nodata_value: int | None = None
    algorithm: str = "explicit-binary-components-v1"
    backend_version: str | None = None

    def __post_init__(self):
        for name in ("id", "project_id", "layer_id"):
            object.__setattr__(self, name, validate_uuid(getattr(self, name)))
        if self.calibration.layer_id != self.layer_id or self.calibration.artifact != self.mask:
            raise DomainValidationError("binary mask requires its own version-specific pixel calibration")
        for size in (self.width, self.height):
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise DomainValidationError("mask dimensions must be positive integers")
        if isinstance(self.connectivity, bool) or self.connectivity not in {4, 8}:
            raise DomainValidationError("pixel adjacency must explicitly be 4 or 8")
        if isinstance(self.foreground_value, bool) or self.foreground_value not in {0, 255}:
            raise DomainValidationError("binary foreground must explicitly be 0 or 255")
        if self.nodata_value is not None and (isinstance(self.nodata_value, bool) or not isinstance(self.nodata_value, int)
                                            or self.nodata_value not in range(1, 255)):
            raise DomainValidationError("nodata marker must be a distinct grayscale value in 1..254")
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "nodata_runs", _runs(self.nodata_runs))
        if len({component.id for component in self.components}) != len(self.components):
            raise DomainValidationError("component identities must be unique")
        all_runs = sorted([run for c in self.components for run in c.runs] + list(self.nodata_runs))
        previous = None
        for run in all_runs:
            if run[0] >= self.height or run[2] > self.width:
                raise DomainValidationError("component geometry is outside its mask")
            if previous is not None and run[0] == previous[0] and run[1] < previous[2]:
                raise DomainValidationError("components and nodata must not overlap")
            previous = run
        if self.algorithm != "explicit-binary-components-v1":
            raise DomainValidationError("unsupported conductor extraction algorithm")

    def to_dict(self):
        return {"schema_version": 1, "unit": "m", "extraction": json.loads(json.dumps(asdict(self)))}

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema_version") != 1 or payload.get("unit") != "m":
            raise DomainValidationError("unsupported conductor extraction schema")
        values = dict(payload["extraction"])
        calibration = dict(values["calibration"])
        calibration["artifact"] = ArtifactAnchor(**calibration["artifact"])
        values["calibration"] = CoordinateCalibration(**calibration)
        values["mask"] = ArtifactAnchor(**values["mask"])
        values["components"] = tuple(ConductorComponent(**c) for c in values["components"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ProjectEvidence:
    project_id: str
    layer_id: str
    artifact: ArtifactAnchor
    current: bool = False

    def __post_init__(self):
        object.__setattr__(self, "project_id", validate_uuid(self.project_id))
        object.__setattr__(self, "layer_id", validate_uuid(self.layer_id))
        if not isinstance(self.current, bool):
            raise DomainValidationError("evidence currentness must be explicitly boolean")


@dataclass(frozen=True, slots=True)
class ComponentReview:
    component_id: str
    reviewed_by: str
    evidence: tuple[ArtifactAnchor, ...]
    note: str
    decision: str = "confirmed"
    declared_net_name: str | None = None
    accepts_diagonal_bridges: bool = False

    def __post_init__(self):
        object.__setattr__(self, "component_id", validate_uuid(self.component_id))
        object.__setattr__(self, "reviewed_by", validate_uuid(self.reviewed_by))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "note", require_non_empty(self.note, field="component review note"))
        if self.decision not in {"confirmed", "rejected"}:
            raise DomainValidationError("component review must confirm or reject")
        if self.decision == "confirmed" and not self.evidence:
            raise DomainValidationError("material confirmation requires independent evidence")
        if not isinstance(self.accepts_diagonal_bridges, bool):
            raise DomainValidationError("diagonal bridge acceptance must be explicit boolean")
        if self.declared_net_name is not None:
            object.__setattr__(self, "declared_net_name", require_non_empty(self.declared_net_name, field="net name", maximum=256))


@dataclass(frozen=True, slots=True)
class ContactBinding:
    contact_id: str
    source_component_id: str
    target_component_id: str
    reviewed_by: str
    evidence: tuple[ArtifactAnchor, ...]
    note: str

    def __post_init__(self):
        for name in ("contact_id", "source_component_id", "target_component_id", "reviewed_by"):
            object.__setattr__(self, name, validate_uuid(getattr(self, name)))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "note", require_non_empty(self.note, field="contact binding rationale"))
        if not self.evidence or self.source_component_id == self.target_component_id:
            raise DomainValidationError("contact binding requires independent evidence and distinct components")


@dataclass(frozen=True, slots=True)
class TerminalBinding:
    component_id: str
    port: str
    direction: str
    reviewed_by: str
    evidence: tuple[ArtifactAnchor, ...]
    note: str
    instance: str | None = None
    cell: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "component_id", validate_uuid(self.component_id))
        object.__setattr__(self, "reviewed_by", validate_uuid(self.reviewed_by))
        object.__setattr__(self, "port", _identifier(self.port))
        if self.instance is not None:
            object.__setattr__(self, "instance", _identifier(self.instance))
            object.__setattr__(self, "cell", _identifier(self.cell))
        elif self.cell is not None:
            raise DomainValidationError("top-level ports cannot declare an instance cell")
        if self.direction not in {"input", "output", "inout"}:
            raise DomainValidationError("terminal direction must be explicit")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "note", require_non_empty(self.note, field="terminal rationale"))
        if not self.evidence:
            raise DomainValidationError("terminal bindings require reviewed evidence")


@dataclass(frozen=True, slots=True)
class ElectricalModel:
    project_id: str
    revision: int = 0
    extractions: tuple[ConductorExtraction, ...] = ()
    component_reviews: tuple[ComponentReview, ...] = ()
    contact_bindings: tuple[ContactBinding, ...] = ()
    terminal_bindings: tuple[TerminalBinding, ...] = ()
    device_realizations: tuple[object, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "project_id", validate_uuid(self.project_id))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise DomainValidationError("electrical revision must be a nonnegative integer")
        for name in ("extractions", "component_reviews", "contact_bindings", "terminal_bindings", "device_realizations"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len({e.layer_id for e in self.extractions}) != len(self.extractions):
            raise DomainValidationError("only one current conductor extraction per layer")
        if any(e.project_id != self.project_id for e in self.extractions):
            raise DomainValidationError("foreign project conductor extraction")
        if len({r.component_id for r in self.component_reviews}) != len(self.component_reviews):
            raise DomainValidationError("only one current material review per component")
        if len({d.instance for d in self.device_realizations}) != len(self.device_realizations):
            raise DomainValidationError("only one reviewed device realization per instance")
        if len({d.candidate.id for d in self.device_realizations}) != len(self.device_realizations):
            raise DomainValidationError("a physical geometry candidate cannot be duplicated as multiple devices")

    def to_dict(self):
        return {"schema_version": 1, "electrical": json.loads(json.dumps(asdict(self)))}

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema_version") != 1:
            raise DomainValidationError("unsupported electrical model schema")
        values = dict(payload["electrical"])
        values["extractions"] = tuple(ConductorExtraction.from_dict({"schema_version": 1, "unit": "m", "extraction": e})
                                      for e in values["extractions"])
        for name, kind in (("component_reviews", ComponentReview), ("contact_bindings", ContactBinding),
                           ("terminal_bindings", TerminalBinding)):
            values[name] = tuple(kind(**{**record, "evidence": tuple(ArtifactAnchor(**a) for a in record["evidence"])})
                                 for record in values[name])
        from .device_candidates import MosRealization

        values["device_realizations"] = tuple(MosRealization.from_dict(d) for d in values.get("device_realizations", ()))
        return cls(**values)
