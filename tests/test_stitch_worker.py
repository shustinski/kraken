from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
import sqlite3
import sys
from uuid import uuid4
import zlib

import numpy as np
from PIL import Image
import pytest

from kraken_agent.jobs import AgentJobState, DurableJobStore, StagingWorkspace
from kraken_agent.runner import PluginProcessSpec, PluginRegistry, SubprocessPluginRunner
from kraken_core.plugin_protocol import PluginAsset, PluginJobManifestV2
from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.multilayer import ArtifactAnchor
from kraken_manager.infrastructure.mosaic import MosaicReader, STITCH_OPERATION, write_mosaic
from kraken_manager.infrastructure.stitch_worker import execute
from kraken_manager.infrastructure.stitching import StitchCancelled, StitchLimits, StitchRenderer, VerifiedTileSources
from tests.test_computed_stitching import plan


def prepare(tmp_path, *, mode="rgb", offset=(2, 1)):
    geometry = plan(offset=offset)
    root = tmp_path / "staging"
    identifier = str(uuid4())
    workspace = StagingWorkspace(root, identifier)
    workspace.path.mkdir(parents=True)
    assets, tiles, paths = [], [], {}
    for i, tile in enumerate(geometry.tiles):
        source = tmp_path / f"{i}.png"
        if mode == "labels":
            Image.fromarray(np.full((2, 3), (i+1) * 1000, np.uint16)).save(source)
        else:
            Image.new("RGB" if mode == "rgb" else "L", (3, 2), (255, 0, 0) if mode == "rgb" else 100).save(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        tile = replace(tile, calibration=replace(tile.calibration,
            artifact=ArtifactAnchor(tile.calibration.artifact.version_id, digest)))
        tiles.append(tile)
        relative = f"inputs/{i}.png"
        paths[tile.calibration.artifact.version_id] = workspace.stage_file(source, relative, expected_sha256=digest)
        assets.append(PluginAsset(tile.calibration.artifact.version_id, "source", "frame", relative, digest,
            "image/png", frame_id=tile.frame_id, x=i+1, y=1))
    evidence = tmp_path / "plan.json"
    evidence.write_text('{"fixture":"controlled geometry, not real physical layout"}', encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    geometry = replace(geometry, tiles=tuple(tiles), plan=ArtifactAnchor(geometry.plan.version_id, digest))
    workspace.stage_file(evidence, "inputs/plan.json", expected_sha256=digest)
    assets.append(PluginAsset(geometry.plan.version_id, "stitch-plan", "layer", "inputs/plan.json", digest, "application/json"))
    manifest = PluginJobManifestV2(identifier, STITCH_OPERATION, str(uuid4()), geometry.layer_id, str(uuid4()),
        tuple(assets), parameters={"stitch_plan": asdict(geometry), "mode": mode, "block_size": 2})
    workspace.write_manifest(manifest)
    return geometry, workspace, manifest, paths


@pytest.mark.parametrize("mode", ["rgb", "gray", "labels"])
def test_real_agent_subprocess_writes_complete_indexed_layer_mosaic(tmp_path, mode):
    geometry, workspace, manifest, _ = prepare(tmp_path, mode=mode)
    store = DurableJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue(manifest)
    root = Path(__file__).resolve().parents[1]
    registry = PluginRegistry({STITCH_OPERATION: PluginProcessSpec(STITCH_OPERATION,
        (sys.executable, "-m", "kraken_manager.infrastructure.stitch_worker"))})
    runner = SubprocessPluginRunner(store, workspace.root, registry,
        environment={"PYTHONPATH": str(root / "src")})
    assert runner.run_once()
    job = store.get(manifest.job_id)
    assert job.state is AgentJobState.IMPORTING, job.error
    publication = store.list_publications(manifest.job_id)[0]
    assert publication.final and len(publication.outputs) == 1
    asset = publication.outputs[0]
    assert asset.scope.value == "layer" and asset.frame_id is None
    with MosaicReader(workspace.resolve(asset.relative_path)) as reader:
        metadata = reader.validate(expected_plan=geometry, expected_job_id=manifest.job_id)
        assert metadata["render"]["block_count"] == 6
        block = reader.read_block(0, 0)
        assert block.coverage[0, 0] == 1
        assert block.pixels[0, 0].tolist() == ([255, 0, 0] if mode == "rgb" else 100 if mode == "gray" else 1000)
        overlap = reader.read_block(2, 0)
        assert overlap.coverage[1, 0] == 2
    workspace.verify_result(publication, manifest)
    before = os.stat(workspace.resolve(asset.relative_path)).st_size
    assert asset.metadata["size_bytes"] == before
    assert store.get(manifest.job_id).progress["completed"] == 6


@pytest.mark.parametrize("mutation", ["missing_block", "bad_pixels", "bad_coverage", "extra_block", "different_job"])
def test_complete_validation_refuses_tampered_or_incomplete_mosaic(tmp_path, mutation):
    geometry, workspace, manifest, _ = prepare(tmp_path)
    publication = execute(manifest, workspace.path)
    path = workspace.resolve(publication.outputs[0].relative_path)
    with sqlite3.connect(path) as db:
        if mutation == "missing_block":
            db.execute("DELETE FROM blocks WHERE x=0 AND y=0")
        elif mutation == "bad_pixels":
            db.execute("UPDATE blocks SET pixels=? WHERE x=0 AND y=0", (b"corrupt",))
        elif mutation == "bad_coverage":
            db.execute("UPDATE blocks SET coverage=? WHERE x=0 AND y=0", (b"corrupt",))
        elif mutation == "extra_block":
            db.execute("INSERT INTO blocks SELECT 999,999,width,height,pixels,pixels_sha,coverage,coverage_sha FROM blocks LIMIT 1")
    with pytest.raises((DomainValidationError, ValueError, zlib.error)):
        with MosaicReader(path) as reader:
            reader.validate(expected_plan=geometry, expected_job_id="different" if mutation == "different_job" else manifest.job_id)


def test_cancel_and_disk_budget_never_return_partial_artifact(tmp_path):
    geometry, _, _, paths = prepare(tmp_path)
    renderer = StitchRenderer(geometry, limits=StitchLimits(block_size=2))
    source = VerifiedTileSources(paths)
    path = tmp_path / "cancelled.sqlite3"
    progress = []
    with pytest.raises(StitchCancelled):
        write_mosaic(path, renderer, source, job_id="fixture", cancelled=lambda: len(progress) >= 1,
            progress=lambda done, total: progress.append(done))
    assert progress == [1] and not path.exists()
    with pytest.raises(DomainValidationError, match="budget"):
        write_mosaic(path, renderer, source, job_id="fixture", max_file_bytes=1)
    assert not path.exists()


def test_worker_rejects_unpinned_input_roles_and_corrupt_sources(tmp_path):
    _, workspace, manifest, paths = prepare(tmp_path)
    with pytest.raises(DomainValidationError, match="identity"):
        execute(replace(manifest, inputs=(replace(manifest.inputs[0], frame_id=str(uuid4())), *manifest.inputs[1:])), workspace.path)
    next(iter(paths.values())).write_bytes(b"wrong source")
    with pytest.raises(DomainValidationError, match="SHA-256"):
        execute(manifest, workspace.path)
    assert not tuple((workspace.path / "outputs").rglob("mosaic.sqlite3"))


def test_existing_v1_container_fingerprint_survives_new_optional_catalog_field(tmp_path):
    import json
    from kraken_manager.infrastructure.mosaic import canonical

    geometry, workspace, manifest, _ = prepare(tmp_path)
    publication = execute(manifest, workspace.path)
    path = workspace.resolve(publication.outputs[0].relative_path)
    with sqlite3.connect(path) as db:
        payload = json.loads(db.execute("SELECT payload FROM metadata").fetchone()[0])
        for tile in payload["plan"]["tiles"]:
            tile.pop("catalog_xy")
        payload["plan_sha256"] = hashlib.sha256(canonical(payload["plan"]).encode()).hexdigest()
        db.execute("UPDATE metadata SET payload=?", (canonical(payload),))
    with MosaicReader(path) as reader:
        reader.validate(expected_plan=geometry, expected_job_id=manifest.job_id)
