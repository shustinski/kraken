"""Versioned physical coordinates and reviewed contacts, independent of frame IDs.

Lengths are SI metres. Registration residuals describe supplied control points;
they do not certify accuracy everywhere or establish electrical connectivity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite

from .artifacts import validate_sha256
from .common import DomainValidationError, require_non_empty, validate_uuid
from .project import LayerType


def number(value, name, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise DomainValidationError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise DomainValidationError(f"{name} must be >= {minimum}")


def revision(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DomainValidationError("revision must be a nonnegative integer")


def xy(value):
    if len(value) != 2:
        raise DomainValidationError("coordinates require exactly two values")
    for coordinate in value:
        number(coordinate, "coordinate")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class ArtifactAnchor:
    version_id: str
    sha256: str

    def __post_init__(self):
        object.__setattr__(self, "version_id", validate_uuid(self.version_id))
        object.__setattr__(self, "sha256", validate_sha256(self.sha256))


@dataclass(frozen=True, slots=True)
class PhysicalLayer:
    layer_id: str
    layer_type: LayerType
    z_m: float
    thickness_m: float
    conductive: bool = False

    def __post_init__(self):
        object.__setattr__(self, "layer_id", validate_uuid(self.layer_id))
        object.__setattr__(self, "layer_type", LayerType(self.layer_type))
        number(self.z_m, "z_m")
        number(self.thickness_m, "thickness_m", minimum=0)
        if not isinstance(self.conductive, bool):
            raise DomainValidationError("conductive must be explicit boolean")


@dataclass(frozen=True, slots=True)
class CoordinateCalibration:
    layer_id: str
    revision: int
    artifact: ArtifactAnchor
    pixel_size_m: tuple[float, float]
    origin_m: tuple[float, float] = (0, 0)
    y_axis: str = "down"
    uncertainty_m: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "layer_id", validate_uuid(self.layer_id))
        revision(self.revision)
        if not isinstance(self.artifact, ArtifactAnchor):
            raise DomainValidationError("calibration requires an artifact revision")
        scale = xy(self.pixel_size_m)
        if min(scale) <= 0:
            raise DomainValidationError("pixel size must be positive in metres")
        object.__setattr__(self, "pixel_size_m", scale)
        object.__setattr__(self, "origin_m", xy(self.origin_m))
        if self.y_axis not in {"up", "down"}:
            raise DomainValidationError("y_axis must be up or down")
        if self.uncertainty_m is not None:
            number(self.uncertainty_m, "uncertainty_m", minimum=0)

    def to_physical(self, pixel):
        x, y = xy(pixel)
        return (self.origin_m[0] + x * self.pixel_size_m[0],
                self.origin_m[1] + y * self.pixel_size_m[1] * (1 if self.y_axis == "up" else -1))


@dataclass(frozen=True, slots=True)
class ControlPoint:
    source_xy: tuple[float, float]
    target_xy: tuple[float, float]
    uncertainty_m: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "source_xy", xy(self.source_xy))
        object.__setattr__(self, "target_xy", xy(self.target_xy))
        if self.uncertainty_m is not None:
            number(self.uncertainty_m, "control uncertainty", minimum=0)


@dataclass(frozen=True, slots=True)
class RegistrationFit:
    status: str
    reason: str
    affine: tuple[float, ...] | None = None
    residuals_m: tuple[float, ...] = ()
    rms_m: float | None = None
    max_residual_m: float | None = None
    design_condition: float | None = None
    transform_condition: float | None = None
    control_error_bound_m: float | None = None
    bound_scope: str = "supplied_control_points_only"

    def __post_init__(self):
        if self.status not in {"fitted", "insufficient", "rejected"}:
            raise DomainValidationError("invalid registration status")
        object.__setattr__(self, "reason", require_non_empty(self.reason, field="registration.reason"))
        if (self.status == "fitted") != (self.affine is not None):
            raise DomainValidationError("only fitted registrations carry a transform")
        if self.affine is not None:
            affine = tuple(self.affine)
            if len(affine) != 6:
                raise DomainValidationError("affine transform requires six coefficients")
            for value in affine:
                number(value, "affine coefficient")
            object.__setattr__(self, "affine", affine)
            if affine[0] * affine[4] == affine[1] * affine[3]:
                raise DomainValidationError("affine transform must be invertible")
        residuals = tuple(self.residuals_m)
        for value in residuals:
            number(value, "residual", minimum=0)
        object.__setattr__(self, "residuals_m", residuals)
        for name in ("rms_m", "max_residual_m", "design_condition", "transform_condition", "control_error_bound_m"):
            value = getattr(self, name)
            if value is not None:
                number(value, name, minimum=0)
        if self.bound_scope != "supplied_control_points_only":
            raise DomainValidationError("a global registration error bound is not established")

    def apply(self, physical_xy):
        if self.affine is None:
            raise DomainValidationError("registration has no determined transform")
        x, y = xy(physical_xy)
        a, b, tx, c, d, ty = self.affine
        return (a * x + b * y + tx, c * x + d * y + ty)


@dataclass(frozen=True, slots=True)
class RegistrationRevision:
    id: str
    revision: int
    stack_revision: int
    source: CoordinateCalibration
    target: CoordinateCalibration
    control_points: tuple[ControlPoint, ...]
    fit: RegistrationFit

    def __post_init__(self):
        object.__setattr__(self, "id", validate_uuid(self.id))
        revision(self.revision)
        revision(self.stack_revision)
        if self.source.layer_id == self.target.layer_id:
            raise DomainValidationError("interlayer registration requires distinct layers")
        object.__setattr__(self, "control_points", tuple(self.control_points))
        if self.fit.status == "fitted" and len(self.control_points) < 3:
            raise DomainValidationError("affine registration requires three noncollinear points")
        if self.fit.status == "fitted" and (len(self.fit.residuals_m) != len(self.control_points)
                                            or self.fit.rms_m is None or self.fit.max_residual_m is None):
            raise DomainValidationError("fitted registration requires residuals for every control point")


@dataclass(frozen=True, slots=True)
class InterlayerContact:
    id: str
    source_layer_id: str
    target_layer_id: str
    contact_layer_id: str
    registration_id: str
    registration_revision: int
    position_target_m: tuple[float, float]
    radius_m: float
    state: str = "candidate"
    evidence: tuple[ArtifactAnchor, ...] = ()
    reviewed_by: str | None = None
    review_note: str = ""

    def __post_init__(self):
        for name in ("id", "source_layer_id", "target_layer_id", "contact_layer_id", "registration_id"):
            object.__setattr__(self, name, validate_uuid(getattr(self, name)))
        revision(self.registration_revision)
        if len({self.source_layer_id, self.target_layer_id, self.contact_layer_id}) != 3:
            raise DomainValidationError("contact requires three distinct physical layers")
        object.__setattr__(self, "position_target_m", xy(self.position_target_m))
        number(self.radius_m, "radius_m", minimum=0)
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if self.state not in {"candidate", "confirmed", "rejected"}:
            raise DomainValidationError("invalid contact state")
        if self.state == "candidate" and (self.evidence or self.reviewed_by or self.review_note):
            raise DomainValidationError("unreviewed contact cannot carry a review")
        if self.state != "candidate":
            object.__setattr__(self, "reviewed_by", validate_uuid(self.reviewed_by or ""))
            object.__setattr__(self, "review_note", require_non_empty(self.review_note, field="review_note"))
        if self.state == "confirmed" and not self.evidence:
            raise DomainValidationError("confirmation requires independent versioned evidence")


@dataclass(frozen=True, slots=True)
class MultilayerModel:
    project_id: str
    revision: int = 0
    stack_revision: int = 0
    layers: tuple[PhysicalLayer, ...] = ()
    calibrations: tuple[CoordinateCalibration, ...] = ()
    registrations: tuple[RegistrationRevision, ...] = ()
    contacts: tuple[InterlayerContact, ...] = ()
    stitches: tuple = ()
    stitch_renders: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "project_id", validate_uuid(self.project_id))
        revision(self.revision)
        revision(self.stack_revision)
        for name in ("layers", "calibrations", "registrations", "contacts", "stitches", "stitch_renders"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        layers = {layer.layer_id: layer for layer in self.layers}
        if len(layers) != len(self.layers) or any(a.z_m >= b.z_m for a, b in zip(self.layers, self.layers[1:])):
            raise DomainValidationError("physical stack requires unique layers in increasing z order")
        if any(a.z_m + a.thickness_m / 2 > b.z_m - b.thickness_m / 2 for a, b in zip(self.layers, self.layers[1:])):
            raise DomainValidationError("physical layer thickness intervals overlap")
        if len({c.layer_id for c in self.calibrations}) != len(self.calibrations):
            raise DomainValidationError("only one current calibration per layer")
        if any(c.layer_id not in layers for c in self.calibrations):
            raise DomainValidationError("calibration layer is outside physical stack")
        from .stitch_plan import LayerStitchRevision
        if any(not isinstance(s, LayerStitchRevision) or s.layer_id not in layers for s in self.stitches):
            raise DomainValidationError("stitch revision must belong to a physical layer")
        if len({(s.id, s.revision) for s in self.stitches}) != len(self.stitches):
            raise DomainValidationError("stitch revisions must be unique")
        from .stitch_plan import StitchRender
        if len({r.job_id for r in self.stitch_renders}) != len(self.stitch_renders) or any(
                not isinstance(r, StitchRender) or (r.stitch_id, r.stitch_revision) not in {
                    (s.id, s.revision) for s in self.stitches} for r in self.stitch_renders):
            raise DomainValidationError("stitch renders require unique jobs bound to a geometry revision")
        registrations = {(r.id, r.revision): r for r in self.registrations}
        if len(registrations) != len(self.registrations):
            raise DomainValidationError("registration revisions must be unique")
        if len({c.id for c in self.contacts}) != len(self.contacts):
            raise DomainValidationError("contact identities must be unique")
        for registration in self.registrations:
            if registration.source.layer_id not in layers or registration.target.layer_id not in layers:
                raise DomainValidationError("registration layer is outside physical stack")
        for contact in self.contacts:
            reg = registrations.get((contact.registration_id, contact.registration_revision))
            if reg is None or (reg.source.layer_id, reg.target.layer_id) != (contact.source_layer_id, contact.target_layer_id):
                raise DomainValidationError("contact registration does not match its layers")
            lower, upper = sorted([layers[contact.source_layer_id].z_m, layers[contact.target_layer_id].z_m])
            via = layers.get(contact.contact_layer_id)
            if via is None or via.layer_type is not LayerType.CONTACT or not lower < via.z_m < upper:
                raise DomainValidationError("contact layer must lie between its conductors")
            if not layers[contact.source_layer_id].conductive or not layers[contact.target_layer_id].conductive:
                raise DomainValidationError("contact endpoints require explicitly designated conductors")
            if contact.state == "confirmed":
                if reg.fit.status != "fitted":
                    raise DomainValidationError("undetermined registration cannot confirm a contact")
                inputs = {reg.source.artifact.version_id, reg.target.artifact.version_id}
                input_hashes = {reg.source.artifact.sha256, reg.target.artifact.sha256}
                if any(anchor.version_id in inputs or anchor.sha256 in input_hashes for anchor in contact.evidence):
                    raise DomainValidationError("registration input is not independent contact evidence")

    def calibration(self, layer_id):
        result = next((c for c in self.calibrations if c.layer_id == layer_id), None)
        if result is None:
            raise DomainValidationError("layer has no physical calibration")
        return result

    def registration_current(self, registration):
        try:
            return (registration.stack_revision == self.stack_revision
                    and not any(r.id == registration.id and r.revision > registration.revision for r in self.registrations)
                    and registration.source == self.calibration(registration.source.layer_id)
                    and registration.target == self.calibration(registration.target.layer_id))
        except DomainValidationError:
            return False

    def to_dict(self):
        # The public snapshot has one canonical JSON shape in local and HTTP
        # modes; dataclasses.asdict alone preserves tuples from domain values.
        return {"schema_version": 1, "unit": "m", "model": json.loads(json.dumps(asdict(self)))}

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema_version") != 1 or payload.get("unit") != "m":
            raise DomainValidationError("unsupported multilayer schema or physical unit")
        values = dict(payload["model"])

        def calibration(data):
            return CoordinateCalibration(**{**data, "artifact": ArtifactAnchor(**data["artifact"])})

        values["layers"] = tuple(PhysicalLayer(**layer) for layer in values["layers"])
        values["calibrations"] = tuple(calibration(c) for c in values["calibrations"])
        values["registrations"] = tuple(RegistrationRevision(**{
            **r, "source": calibration(r["source"]), "target": calibration(r["target"]),
            "control_points": tuple(ControlPoint(**p) for p in r["control_points"]), "fit": RegistrationFit(**r["fit"]),
        }) for r in values["registrations"])
        values["contacts"] = tuple(InterlayerContact(**{
            **c, "evidence": tuple(ArtifactAnchor(**a) for a in c["evidence"]),
        }) for c in values["contacts"])
        from .stitch_plan import LayerStitchRevision
        values["stitches"] = tuple(LayerStitchRevision.from_dict(s) for s in values.get("stitches", ()))
        from .stitch_plan import StitchRender
        values["stitch_renders"] = tuple(StitchRender(**{**r,
            "output": None if r["output"] is None else ArtifactAnchor(**r["output"]),
        }) for r in values.get("stitch_renders", ()))
        return cls(**values)
