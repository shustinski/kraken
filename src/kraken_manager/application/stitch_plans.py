"""Build intra-layer poses inside the existing physical-model command stream."""
from __future__ import annotations

from dataclasses import replace
from uuid import UUID, uuid5

from kraken_manager.domain.artifacts import ArtifactScope
from kraken_manager.domain.common import DomainValidationError, validate_uuid
from kraken_manager.domain.multilayer import ControlPoint, CoordinateCalibration
from kraken_manager.domain.stitch_plan import LayerStitchRevision, TilePlacement


def plan_layer_stitch(service, model, arguments):
    args = service._fields(arguments, {"layer_id", "plan_version_id", "reference_frame_id", "tiles",
                                        "max_error_m", "max_condition"})
    layer_id = validate_uuid(args["layer_id"])
    plan = service._anchor(model.project_id, args["plan_version_id"], layer_id)
    version = service.projections.get_artifact_version(plan.version_id)
    series = service.projections.get_artifact_series(version.series_id)
    if series.scope is not ArtifactScope.LAYER_ATTACHMENT:
        raise DomainValidationError("stitch plan must be a layer attachment")
    reference_id = validate_uuid(args["reference_frame_id"])
    tiles = []
    for data in args["tiles"]:
        values = service._fields(data, {"frame_id", "artifact_version_id", "width", "height", "pixel_size_m",
            "origin_m", "y_axis", "uncertainty_m", "control_points", "declared_affine", "note", "catalog_xy"})
        frame_id = validate_uuid(values.pop("frame_id"))
        anchor = service._anchor(model.project_id, values.pop("artifact_version_id"), layer_id)
        version = service.projections.get_artifact_version(anchor.version_id)
        series = service.projections.get_artifact_series(version.series_id)
        if series.scope is not ArtifactScope.FRAME_REPRESENTATION or str(series.frame_id) != frame_id:
            raise DomainValidationError("stitch tile must reference its managed frame artifact")
        width, height = values.pop("width"), values.pop("height")
        catalog_xy = values.pop("catalog_xy", None)
        if catalog_xy is not None:
            if (not isinstance(catalog_xy, (list, tuple)) or len(catalog_xy) != 2
                    or service._project(model.project_id).frame_id_at(*catalog_xy) != frame_id):
                raise DomainValidationError("stitch catalog coordinates do not match the managed frame")
        points, affine, note = values.pop("control_points", ()), values.pop("declared_affine", None), values.pop("note", "")
        calibration = CoordinateCalibration(layer_id, model.revision + 1, anchor, **values)
        if frame_id == reference_id:
            if points or affine is not None:
                raise DomainValidationError("reference frame cannot have an additional transform")
            tile = TilePlacement(frame_id, calibration, width, height, "reference", (1, 0, 0, 0, 1, 0), catalog_xy=catalog_xy)
        elif affine is not None:
            if points:
                raise DomainValidationError("choose observed control points or a declared transform")
            tile = TilePlacement(frame_id, calibration, width, height, "declared_transform", affine, note=note, catalog_xy=catalog_xy)
        else:
            # Resolve all fits after the reference calibration is known.
            controls = tuple(ControlPoint(**service._fields(p, {"source_xy", "target_xy", "uncertainty_m"})) for p in points)
            tile = (frame_id, calibration, width, height, controls, catalog_xy)
        tiles.append(tile)
    references = [p for p in tiles if isinstance(p, TilePlacement) and p.provenance == "reference"]
    if len(references) != 1:
        raise DomainValidationError("stitch plan requires exactly one reference frame")
    reference = references[0]
    solved = []
    for tile in tiles:
        if isinstance(tile, TilePlacement):
            solved.append(tile)
            continue
        frame_id, calibration, width, height, controls, catalog_xy = tile
        if service.fit_affine is None:
            raise DomainValidationError("Affine registration solver is unavailable")
        fit = service.fit_affine(calibration, reference.calibration, controls,
            max_error_m=args["max_error_m"], max_condition=args.get("max_condition", 1e6))
        solved.append(TilePlacement(frame_id, calibration, width, height, "control_points", fit.affine, controls, fit, catalog_xy=catalog_xy))
    identity = str(uuid5(UUID(model.project_id), f"layer-stitch:{layer_id}"))
    latest = max((s.revision for s in model.stitches if s.id == identity), default=-1)
    stitch = LayerStitchRevision(identity, latest + 1, model.stack_revision, layer_id, plan, reference_id, tuple(solved))
    return replace(model, stitches=(*model.stitches, stitch))
