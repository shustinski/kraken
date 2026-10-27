"""Bind raster content separately from immutable physical geometry evidence."""
from .errors import ConflictError
from kraken_manager.domain.project import RepresentationKind, StructureState

GRID_PRESERVING_CAPABILITIES = frozenset({"frames.binary-segment.v1", "frames.binary-segment.v2", "frames.confidence.v1"})


def content_versions(service, model, stitch, requested, *, kind="image"):
    if not isinstance(requested, dict) or set(requested) != {p.frame_id for p in stitch.tiles}:
        raise ConflictError("Stitch content must bind every geometry frame exactly once")
    result = {}
    for tile in stitch.tiles:
        anchor = service._anchor(model.project_id, requested[tile.frame_id], stitch.layer_id)
        version = service.projections.get_artifact_version(anchor.version_id)
        series = service.projections.get_artifact_series(version.series_id)
        representation = service.projections.get_representation(series.representation_id) if series.representation_id else None
        vector = kind == "vector"
        expected_kind = RepresentationKind.VECTOR if vector else RepresentationKind.IMAGE
        if (str(series.frame_id) != tile.frame_id or representation is None or representation.kind is not expected_kind
                or representation.state is StructureState.ARCHIVED
                or (version.media_type != "application/x-cif" if vector else not version.media_type.startswith("image/"))):
            raise ConflictError("Stitch content must have its typed format on the same frame and layer")
        if anchor != tile.calibration.artifact:
            # An arbitrary same-sized picture is not evidence of the same pixel
            # grid. Accept only existing native grid-preserving jobs and a
            # complete immutable lineage back to the geometry image.
            pending, visited = [version], set()
            found = False
            while pending:
                item = pending.pop()
                if str(item.id) == tile.calibration.artifact.version_id:
                    found = item.sha256 == tile.calibration.artifact.sha256
                    continue
                if str(item.id) in visited:
                    continue
                visited.add(str(item.id))
                if len(visited) > 64:
                    raise ConflictError("Stitch raster lineage exceeds its bounded depth")
                event = next((e for e in service.events.load_stream(f"artifact-series:{item.series_id}")
                    if e.event_type == "ArtifactVersionCreated" and str(e.payload["artifact_version_id"]) == str(item.id)), None)
                job_id = None if event is None else event.payload.get("plugin_job_id")
                job = service.projections.get_plugin_job(job_id) if job_id else None
                allowed = {"frames.vectorize.v1"} if item.media_type == "application/x-cif" and vector else GRID_PRESERVING_CAPABILITIES
                if job is None or job.capability not in allowed:
                    raise ConflictError("Derived stitch content lacks a verified grid-preserving producer")
                parents = [service.projections.get_artifact_version(v) for v in item.input_version_ids]
                # Model inputs are auxiliary, not alternate image geometry.
                parents = [p for p in parents if p is not None and p.media_type.startswith("image/")]
                if not parents:
                    raise ConflictError("Derived stitch content has no image lineage")
                pending.extend(parents)
            if not found:
                raise ConflictError("Derived stitch content is not descended from its geometry image")
        result[tile.frame_id] = version
    return result
