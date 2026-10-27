"""Read an accepted layer result with canonical project/scope/manifest checks."""
from contextlib import contextmanager

from kraken_core.plugin_protocol import PluginJobManifestV2
from kraken_manager.application.errors import ConflictError, NotFoundError
from kraken_manager.application.multilayer import MultilayerService
from kraken_manager.domain.common import DomainValidationError, validate_uuid
from .mosaic import MosaicReader
from .vector_exchange import header_from_mosaic


@contextmanager
def published_mosaic(events, projections, path_for_blob, project_id, job_id):
    model = MultilayerService(events, projections).load(str(project_id))
    identifier = validate_uuid(str(job_id))
    request = next((r for r in model.stitch_renders if r.job_id == identifier), None)
    if request is None or request.state != "imported":
        raise NotFoundError("Published mosaic was not found in this project")
    plan = next(s for s in model.stitches if (s.id, s.revision) == (request.stitch_id, request.stitch_revision))
    version = projections.get_artifact_version(request.output.version_id)
    if version is None or version.blob is None or version.sha256 != request.output.sha256:
        raise ConflictError("Published mosaic artifact is missing or inconsistent")
    series = projections.get_artifact_series(version.series_id)
    if (series is None or str(series.project_id) != str(project_id) or str(series.layer_id) != plan.layer_id
            or series.scope.value != "layer_attachment" or series.frame_id is not None or series.representation_id is not None):
        raise ConflictError("Mosaic ownership differs from its requested layer")
    manifest = PluginJobManifestV2.from_json(request.manifest_json)
    with MosaicReader(path_for_blob(version.blob)) as reader:
        reader.check_request(expected_plan=plan, expected_job_id=identifier, mode=manifest.parameters["mode"], block_size=manifest.parameters["block_size"])
        if reader.metadata.get("content_inputs", []) != manifest.parameters.get("content_inputs", []):
            raise ConflictError("Mosaic content differs from the canonical request")
        if reader.metadata.get("vector_stitch") != version.parameters.get("vector_stitch"):
            raise ConflictError("Mosaic vector policy differs from the accepted artifact")
        yield reader, version


def vector_header(reader, version, project_id):
    return header_from_mosaic(reader, project_id=project_id, output_version_id=version.id, output_sha256=version.sha256)


def vector_block(reader, version, column, row):
    import base64
    import hashlib
    import zlib
    if "vector_stitch" not in reader.metadata:
        raise DomainValidationError("This mosaic contains no vector content")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (column, row)):
        raise DomainValidationError("mosaic block indices must be nonnegative integers")
    left, top, right, bottom = reader.renderer.bounds
    step = reader.renderer.limits.block_size
    x, y = left+column*step, top+row*step
    if x >= right or y >= bottom:
        raise DomainValidationError("mosaic vector block is outside its grid")
    block = reader.read_block(x, y)
    coverage = block.coverage.astype("<u4", copy=False).tobytes()
    return {"job_id": reader.metadata["job_id"], "output_version_id": str(version.id), "output_sha256": version.sha256,
        "coverage": {"encoding": "zlib-uint32-le", "data_base64": base64.b64encode(zlib.compress(coverage)).decode("ascii"),
            "sha256": hashlib.sha256(coverage).hexdigest()}, **reader.read_vectors(x, y)}
