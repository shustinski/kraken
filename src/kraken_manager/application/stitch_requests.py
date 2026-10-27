"""Durable stitch requests bound to physical revisions; Agent owns execution."""
from dataclasses import asdict, replace

from kraken_core.plugin_protocol import PluginAsset, PluginJobManifestV2
from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.domain.common import DomainValidationError, utc_now
from kraken_manager.domain.multilayer import revision
from kraken_manager.domain.stitch_plan import StitchRender


def request_stitch(service, model, arguments, job_id, actor_id, *, source_versions=None, pipeline_owner=None, created_at=None, max_file_bytes=None, content_kind="image"):
    args = service._fields(arguments, {"stitch_id", "stitch_revision", "mode", "block_size", "frame_coordinates"})
    revision(args["stitch_revision"])
    stitch = next((s for s in model.stitches if (s.id, s.revision) == (args["stitch_id"], args["stitch_revision"])), None)
    if stitch is None:
        raise NotFoundError("Stitch geometry not found")
    if not service.stitch_current(model, stitch) or not stitch.ready:
        raise ConflictError("Stale or unresolved stitch geometry cannot be computed")
    mode, block_size = args.get("mode", "rgb"), args.get("block_size", 256)
    if mode not in {"rgb", "gray", "labels"} or isinstance(block_size, bool) or not isinstance(block_size, int) or not 0 < block_size <= 1024:
        raise DomainValidationError("invalid stitch sampling mode or block size")
    project = service._project(model.project_id)
    supplied = args.get("frame_coordinates", {})
    frames = {tile.frame_id for tile in stitch.tiles}
    if not isinstance(supplied, dict) or set(supplied) - frames:
        raise DomainValidationError("stitch catalog coordinates refer to an unknown frame")
    from .stitch_content import content_versions
    if content_kind not in {"image", "vector"} or content_kind == "vector" and source_versions is None:
        raise DomainValidationError("vector stitching requires explicit typed content bindings")
    content = None if source_versions is None else content_versions(service, model, stitch, source_versions, kind=content_kind)
    assets = []
    samples = []
    for tile in stitch.tiles:
        anchor = tile.calibration.artifact
        version = service.projections.get_artifact_version(anchor.version_id)
        coords = supplied.get(tile.frame_id, tile.catalog_xy or (version.parameters.get("x"), version.parameters.get("y")))
        if not isinstance(coords, (list, tuple)) or len(coords) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in coords):
            raise DomainValidationError("stitch inputs require their actual catalog frame coordinates")
        if project.frame_id_at(*coords) != tile.frame_id:
            raise DomainValidationError("catalog coordinates do not match the immutable frame identity")
        sampled = version if content is None else content[tile.frame_id]
        changed = str(sampled.id) != anchor.version_id
        assets.append(PluginAsset(anchor.version_id, "geometry-source" if changed else "source", "frame", f"inputs/{anchor.version_id}.image",
            anchor.sha256, version.media_type, frame_id=tile.frame_id, x=coords[0], y=coords[1]))
        if content is not None:
            samples.append({"frame_id": tile.frame_id, "artifact_version_id": str(sampled.id), "sha256": sampled.sha256})
            if changed:
                assets.append(PluginAsset(str(sampled.id), "source", "frame", f"inputs/{sampled.id}.image", sampled.sha256,
                    sampled.media_type, frame_id=tile.frame_id, x=coords[0], y=coords[1]))
    version = service.projections.get_artifact_version(stitch.plan.version_id)
    assets.append(PluginAsset(stitch.plan.version_id, "stitch-plan", "layer", "inputs/stitch-plan.json",
                              stitch.plan.sha256, version.media_type))
    manifest = PluginJobManifestV2(job_id, "layer.stitch.v1", model.project_id, stitch.layer_id, actor_id, tuple(assets),
        parameters={"stitch_plan": asdict(stitch), "mode": mode, "block_size": block_size,
            **({"content_inputs": samples} if samples else {}), **({"_pipeline": pipeline_owner} if pipeline_owner is not None else {})},
        created_at=created_at or utc_now().isoformat())
    if max_file_bytes is not None:
        manifest = replace(manifest, parameters={**manifest.parameters, "max_file_bytes": max_file_bytes})
    if content_kind == "vector":
        manifest = replace(manifest, parameters={**manifest.parameters, "content_kind": "vector"})
    request = StitchRender(job_id, stitch.id, stitch.revision, manifest.to_json())
    return replace(model, stitch_renders=(*model.stitch_renders, request))


def cancel_stitch(service, model, arguments):
    args = service._fields(arguments, {"job_id"})
    request = next((r for r in model.stitch_renders if r.job_id == args["job_id"]), None)
    if request is None:
        raise NotFoundError("Stitch computation not found")
    if request.state == "imported":
        raise ConflictError("A published mosaic cannot be cancelled")
    return replace(model, stitch_renders=tuple(replace(r, state="cancelled") if r.job_id == request.job_id else r
                                               for r in model.stitch_renders))
