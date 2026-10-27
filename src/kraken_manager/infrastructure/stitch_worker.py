"""Headless intra-layer stitching on the existing Agent V2 transport."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from contextlib import ExitStack
import os
from pathlib import Path
import shutil
from uuid import NAMESPACE_URL, uuid4, uuid5

from kraken_core.plugin_protocol import PluginAsset, PluginAssetScope, PluginJobManifestV2, PluginResultPublicationV2
from kraken_core.safe_files import contained_path, make_contained_directories, open_exclusive_write, open_regular_read, replace_atomic_with_retry
from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.stitch_plan import LayerStitchRevision
from .mosaic import MOSAIC_MEDIA_TYPE, STITCH_OPERATION, canonical, write_mosaic
from .stitching import StitchCancelled, StitchLimits, StitchRenderer, VerifiedTileSources


def _write_json(path, payload):
    temporary = Path(str(path) + ".part-" + uuid4().hex)
    identity = None
    try:
        with open_exclusive_write(temporary) as stream:
            stat = os.fstat(stream.fileno())
            identity = stat.st_dev, stat.st_ino
            stream.write(canonical(payload).encode())
            stream.flush()
            os.fsync(stream.fileno())
        replace_atomic_with_retry(temporary, path)
    finally:
        if identity is not None:
            try:
                stat = temporary.stat(follow_symlinks=False)
                if (stat.st_dev, stat.st_ino) == identity:
                    temporary.unlink()
            except FileNotFoundError:
                pass


def execute(manifest, staging_root, *, cancelled=lambda: False, progress=lambda *_: None):
    if manifest.operation != STITCH_OPERATION:
        raise DomainValidationError("unsupported stitch operation")
    if set(manifest.parameters) - {"stitch_plan", "mode", "block_size", "max_file_bytes", "content_inputs", "content_kind", "_pipeline"}:
        raise DomainValidationError("unknown stitch worker parameters")
    plan = LayerStitchRevision.from_dict(manifest.parameters["stitch_plan"])
    if plan.layer_id != manifest.layer_id or manifest.target_representation_ids:
        raise DomainValidationError("stitch result must target its layer, without a frame representation")
    inputs = {asset.asset_id: asset for asset in manifest.inputs}
    samples = manifest.parameters.get("content_inputs", [])
    bindings = {p["frame_id"]: p for p in samples}
    vector_content = manifest.parameters.get("content_kind", "image") == "vector"
    if manifest.parameters.get("content_kind", "image") not in {"image", "vector"} or vector_content and not samples:
        raise DomainValidationError("unsupported or missing stitch content binding")
    if samples and (len(bindings) != len(samples) or set(bindings) != {tile.frame_id for tile in plan.tiles}):
        raise DomainValidationError("sample bindings must cover the geometry frames exactly")
    owner = manifest.parameters.get("_pipeline")
    if owner is not None and owner.get("algorithm_version") != "1.0.0":
        raise DomainValidationError("stitch worker version differs from the pinned pipeline algorithm")
    expected = {tile.calibration.artifact.version_id for tile in plan.tiles} | {plan.plan.version_id} | {p["artifact_version_id"] for p in samples}
    if len(inputs) != len(manifest.inputs) or set(inputs) != expected:
        raise DomainValidationError("stitch job inputs do not match the pinned plan")
    paths, vector_paths = {}, {}
    if any(not asset.relative_path.startswith("inputs/") for asset in inputs.values()):
        raise DomainValidationError("stitch inputs must be staged below inputs/")
    for tile in plan.tiles:
        anchor = tile.calibration.artifact
        asset = inputs[anchor.version_id]
        if (asset.scope is not PluginAssetScope.FRAME or asset.frame_id != tile.frame_id
                or asset.role not in {"source", "geometry-source"} or asset.sha256 != anchor.sha256):
            raise DomainValidationError("stitch source identity mismatch")
        path = contained_path(staging_root, tuple(asset.relative_path.split("/")))
        if tile.frame_id in bindings:
            sampled = bindings[tile.frame_id]
            target = inputs[sampled["artifact_version_id"]]
            if target.role != "source" or target.scope is not PluginAssetScope.FRAME or target.frame_id != tile.frame_id or target.sha256 != sampled["sha256"]:
                raise DomainValidationError("sample source identity mismatch")
            if target.asset_id != anchor.version_id:
                with open_regular_read(path, root=staging_root) as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != anchor.sha256:
                        raise DomainValidationError("geometry source SHA-256 mismatch")
            if vector_content:
                if target.media_type != "application/x-cif":
                    raise DomainValidationError("vector stitching requires native Contour CIF content")
                vector_paths[tile.frame_id] = contained_path(staging_root, tuple(target.relative_path.split("/")))
                paths[anchor.version_id] = path
            else:
                path = contained_path(staging_root, tuple(target.relative_path.split("/")))
                paths[target.asset_id] = path
        else:
            paths[anchor.version_id] = path
    evidence = inputs[plan.plan.version_id]
    if evidence.scope is not PluginAssetScope.LAYER or evidence.role != "stitch-plan" or evidence.sha256 != plan.plan.sha256:
        raise DomainValidationError("stitch evidence identity mismatch")
    evidence_path = contained_path(staging_root, tuple(evidence.relative_path.split("/")))
    with open_regular_read(evidence_path, root=staging_root) as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != plan.plan.sha256:
            raise DomainValidationError("stitch evidence SHA-256 mismatch")
    limits = StitchLimits(block_size=manifest.parameters.get("block_size", 256))
    mode = manifest.parameters.get("mode", "rgb")
    renderer = StitchRenderer(plan, mode=mode, limits=limits)
    budget = manifest.parameters.get("max_file_bytes", 128 * 1024**3)
    if isinstance(budget, bool) or not isinstance(budget, int) or not 0 < budget <= 512 * 1024**3:
        raise DomainValidationError("invalid mosaic output budget")
    output_dir = make_contained_directories(staging_root, ("outputs", "stitch-" + uuid4().hex))
    if renderer.preflight()["raw_output_bytes"] + 16 * 1024**2 > shutil.disk_usage(output_dir).free:
        raise DomainValidationError("insufficient disk space for conservative mosaic preflight")
    path = output_dir / "mosaic.sqlite3"
    source = VerifiedTileSources(paths, mode=mode, limits=limits, cancelled=cancelled)
    def load_sample(tile):
        if tile.frame_id not in bindings or vector_content:
            return source(tile)
        from kraken_manager.domain.multilayer import ArtifactAnchor
        sampled = bindings[tile.frame_id]
        # This temporary decode descriptor is not persisted as a calibration;
        # the mosaic retains the original geometry and a separate content port.
        return source(replace(tile, calibration=replace(tile.calibration,
            artifact=ArtifactAnchor(sampled["artifact_version_id"], sampled["sha256"]))))
    with ExitStack() as stack:
        vectors = None
        if vector_content:
            from .stitch_vectors import VectorStitchIndex
            from contour.infrastructure.stitch_cif import load_stitch_cif
            vectors = stack.enter_context(VectorStitchIndex(output_dir / "vector-index.sqlite3", renderer, cancelled=cancelled))
            for tile in plan.tiles:
                try:
                    geometries = load_stitch_cif(vector_paths[tile.frame_id], expected_sha256=bindings[tile.frame_id]["sha256"],
                        image_size=(tile.width, tile.height))
                except ValueError as exc:
                    raise DomainValidationError(f"Vector frame {tile.frame_id}, version {bindings[tile.frame_id]['artifact_version_id']}: {exc}") from exc
                vectors.add(tile, bindings[tile.frame_id], geometries)
        metadata = write_mosaic(path, renderer, load_sample, job_id=manifest.job_id, cancelled=cancelled, progress=progress,
                                max_file_bytes=budget, content_inputs=samples, vector_index=vectors)
    with open_regular_read(path, root=staging_root) as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    asset = PluginAsset(str(uuid5(NAMESPACE_URL, "kraken:mosaic:" + manifest.job_id)), "mosaic", PluginAssetScope.LAYER,
        path.relative_to(staging_root).as_posix(), digest, MOSAIC_MEDIA_TYPE,
        metadata={"size_bytes": path.stat().st_size, "plan_sha256": metadata["plan_sha256"]})
    return PluginResultPublicationV2(manifest.job_id,
        str(uuid5(NAMESPACE_URL, "kraken:mosaic-publication:" + manifest.job_id)), 1, "kraken-stitch", "1.0.0", (asset,),
        report={"render": metadata["render"], "plan_sha256": metadata["plan_sha256"]}, final=True)


def main():
    root = Path(os.environ["KRAKEN_STAGING_ROOT"])
    with open_regular_read(Path(os.environ["KRAKEN_JOB_MANIFEST"]), root=root) as stream:
        manifest = PluginJobManifestV2.from_json(stream.read(16 * 1024**2 + 1).decode())
    cancel_path = Path(os.environ.get("KRAKEN_CANCEL_PATH", root / "cancel.request"))

    def progress(done, total):
        if os.environ.get("KRAKEN_PROGRESS_PATH"):
            _write_json(Path(os.environ["KRAKEN_PROGRESS_PATH"]), {"job_id": manifest.job_id, "completed": done, "total": total})

    try:
        result = execute(manifest, root, cancelled=cancel_path.exists, progress=progress)
    except Exception as exc:
        result = PluginResultPublicationV2(manifest.job_id,
            str(uuid5(NAMESPACE_URL, "kraken:mosaic-publication:" + manifest.job_id)), 1, "kraken-stitch", "1.0.0", (),
            outcome="cancelled" if isinstance(exc, StitchCancelled) else "failed", report={"error": str(exc)}, final=True)
    _write_json(Path(os.environ["KRAKEN_RESULT_MANIFEST"]), result.to_dict())


if __name__ == "__main__":
    from kraken_core.process_lifetime import bind_child_process_lifetime
    bind_child_process_lifetime()
    main()
