"""Immutable intra-layer geometry; coordinates never imply electrical contacts."""
from __future__ import annotations

from dataclasses import dataclass

from .common import DomainValidationError, require_non_empty, validate_uuid
from .artifacts import validate_sha256
from .multilayer import ArtifactAnchor, ControlPoint, CoordinateCalibration, RegistrationFit, revision


@dataclass(frozen=True, slots=True)
class TilePlacement:
    frame_id: str
    calibration: CoordinateCalibration
    width: int
    height: int
    provenance: str
    # Source physical metres -> reference physical metres.
    affine: tuple[float, ...] | None = None
    control_points: tuple[ControlPoint, ...] = ()
    fit: RegistrationFit | None = None
    note: str = ""
    catalog_xy: tuple[int, int] | None = None

    def __post_init__(self):
        object.__setattr__(self, "frame_id", validate_uuid(self.frame_id))
        if self.catalog_xy is not None:
            coordinates = tuple(self.catalog_xy)
            if len(coordinates) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in coordinates):
                raise DomainValidationError("catalog coordinates must be positive integer pairs")
            object.__setattr__(self, "catalog_xy", coordinates)
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 1_000_000:
                raise DomainValidationError("tile dimensions must be bounded positive integers")
        if not isinstance(self.calibration, CoordinateCalibration):
            raise DomainValidationError("tile requires versioned physical calibration")
        object.__setattr__(self, "control_points", tuple(self.control_points))
        if self.provenance not in {"reference", "declared_transform", "control_points"}:
            raise DomainValidationError("unknown tile placement provenance")
        if self.affine is not None:
            affine = RegistrationFit("fitted", "transform_validation", tuple(self.affine)).affine
            object.__setattr__(self, "affine", affine)
        if self.provenance == "control_points":
            if not isinstance(self.fit, RegistrationFit) or self.affine != self.fit.affine:
                raise DomainValidationError("computed tile transform must match its measured fit")
            if self.fit.status == "fitted" and (len(self.control_points) < 3
                    or len(self.fit.residuals_m) != len(self.control_points)
                    or self.fit.rms_m is None or self.fit.max_residual_m is None):
                raise DomainValidationError("computed placement requires every control residual")
        else:
            if self.fit is not None or self.control_points or self.affine is None:
                raise DomainValidationError("declared transforms cannot claim measured residuals")
            if self.provenance == "reference" and self.affine != (1, 0, 0, 0, 1, 0):
                raise DomainValidationError("reference transform must be identity")
            if self.provenance == "declared_transform":
                require_non_empty(self.note, field="declared transform evidence")


@dataclass(frozen=True, slots=True)
class LayerStitchRevision:
    id: str
    revision: int
    stack_revision: int
    layer_id: str
    plan: ArtifactAnchor
    reference_frame_id: str
    tiles: tuple[TilePlacement, ...]

    def __post_init__(self):
        for name in ("id", "layer_id", "reference_frame_id"):
            object.__setattr__(self, name, validate_uuid(getattr(self, name)))
        revision(self.revision)
        revision(self.stack_revision)
        if not isinstance(self.plan, ArtifactAnchor):
            raise DomainValidationError("stitch plan requires a managed artifact anchor")
        object.__setattr__(self, "tiles", tuple(self.tiles))
        if not self.tiles or len({p.frame_id for p in self.tiles}) != len(self.tiles):
            raise DomainValidationError("stitch plan requires unique nonempty frame placements")
        if len({p.calibration.artifact.version_id for p in self.tiles}) != len(self.tiles):
            raise DomainValidationError("stitch inputs must be distinct artifact versions")
        if any(p.calibration.layer_id != self.layer_id for p in self.tiles):
            raise DomainValidationError("all tiles must belong to the stitch layer")
        references = [p for p in self.tiles if p.provenance == "reference"]
        if len(references) != 1 or references[0].frame_id != self.reference_frame_id:
            raise DomainValidationError("stitch plan requires exactly one declared reference frame")
        reference = references[0]
        for tile in self.tiles:
            for point in tile.control_points:
                if not (0 <= point.source_xy[0] < tile.width and 0 <= point.source_xy[1] < tile.height
                        and 0 <= point.target_xy[0] < reference.width and 0 <= point.target_xy[1] < reference.height):
                    raise DomainValidationError("stitch control point lies outside its source or reference image")

    @property
    def ready(self):
        return all(tile.affine is not None for tile in self.tiles)

    @classmethod
    def from_dict(cls, data):
        def tile(value):
            values = dict(value)
            calibration = values["calibration"]
            values["calibration"] = CoordinateCalibration(**{
                **calibration, "artifact": ArtifactAnchor(**calibration["artifact"]),
            })
            values["control_points"] = tuple(ControlPoint(**p) for p in values.get("control_points", ()))
            values["fit"] = None if values.get("fit") is None else RegistrationFit(**values["fit"])
            return TilePlacement(**values)
        return cls(**{**data, "plan": ArtifactAnchor(**data["plan"]), "tiles": tuple(tile(p) for p in data["tiles"])})


@dataclass(frozen=True, slots=True)
class StitchRender:
    job_id: str
    stitch_id: str
    stitch_revision: int
    manifest_json: str
    state: str = "requested"
    output: ArtifactAnchor | None = None
    receipt_sha256: str | None = None
    error: str = ""

    def __post_init__(self):
        for name in ("job_id", "stitch_id"):
            object.__setattr__(self, name, validate_uuid(getattr(self, name)))
        revision(self.stitch_revision)
        if not isinstance(self.manifest_json, str) or not 0 < len(self.manifest_json.encode()) <= 16 * 1024**2:
            raise DomainValidationError("stitch request requires a bounded canonical manifest")
        if self.state not in {"requested", "imported", "failed", "cancelled"}:
            raise DomainValidationError("invalid stitch render state")
        if self.state == "imported":
            if not isinstance(self.output, ArtifactAnchor) or not self.receipt_sha256:
                raise DomainValidationError("imported mosaic requires output and publication receipt")
            object.__setattr__(self, "receipt_sha256", validate_sha256(self.receipt_sha256))
        elif self.output is not None or self.receipt_sha256 is not None:
            raise DomainValidationError("unfinished mosaic cannot claim a published result")
