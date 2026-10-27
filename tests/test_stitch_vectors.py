"""Analytic geometry/uncertainty checks independent of model accuracy."""
from dataclasses import replace
import hashlib
import json

import pytest
from shapely import Polygon, box, union_all
from shapely.geometry import shape

from kraken_manager.domain.common import DomainValidationError
from kraken_manager.infrastructure.stitch_vectors import VectorLimits, VectorStitchIndex
from kraken_manager.infrastructure.stitching import StitchCancelled, StitchLimits, StitchRenderer
from tests.test_computed_stitching import plan


def index(tmp_path, geometry, *, limits=None, cancelled=lambda: False):
    renderer = StitchRenderer(geometry, limits=StitchLimits(block_size=2))
    return VectorStitchIndex(tmp_path / "vector-index.sqlite", renderer, limits=limits, cancelled=cancelled)


def source(tile):
    return {"artifact_version_id": tile.calibration.artifact.version_id}


def test_physical_projection_union_and_explicit_overlap_disagreement(tmp_path):
    geometry = plan((2, 0), size=(6, 6))
    a, b = geometry.tiles
    with index(tmp_path, geometry) as output:
        output.add(a, source(a), [("conductor", box(1, 1, 4, 4))])
        output.add(b, source(b), [("conductor", box(0, 1, 3, 4))])
        payload, digest = output.block(0, 0, 8, 6)
        assert hashlib.sha256(payload).hexdigest() == digest
        value = json.loads(payload)
        merged = shape(value["features"][0]["geometry"])
        assert merged.equals(box(1, 1, 5, 4))
        assert set(value["features"][0]["source_version_ids"]) == {source(a)["artifact_version_id"], source(b)["artifact_version_id"]}
        conflict = union_all([shape(f["geometry"]) for f in value["conflicts"]])
        expected = box(1, 1, 4, 4).symmetric_difference(box(2, 1, 5, 4)).intersection(box(1.5, -.5, 5.5, 5.5))
        assert conflict.equals(expected)
    assert not (tmp_path / "vector-index.sqlite").exists()


def test_block_partition_preserves_holes_categories_and_exact_global_geometry(tmp_path):
    geometry = plan((6, 0), size=(6, 6))
    a, b = geometry.tiles
    ring = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)], [[(1, 1), (4, 1), (4, 4), (1, 4)]])
    with index(tmp_path, geometry) as output:
        output.add(a, source(a), [("conductor", ring), ("via", box(2, 2, 3, 3))])
        output.add(b, source(b), [])
        blocks = [json.loads(output.block(x, y, 2, 2)[0]) for y in range(0, 6, 2) for x in range(0, 12, 2)]
        actual = union_all([shape(f["geometry"]) for v in blocks for f in v["features"] if f["category"] == "conductor"])
        assert actual.equals(ring) and len(actual.interiors) == 1
        assert all(not value["conflicts"] for value in blocks)
        assert {f["category"] for value in blocks for f in value["features"]} == {"conductor", "via"}


def test_derived_vectors_use_anisotropic_physical_calibration_not_catalog_positions(tmp_path):
    geometry = plan((0, 0), size=(8, 8))
    a, b = geometry.tiles
    b = replace(b, calibration=replace(b.calibration, pixel_size_m=(2, 3), origin_m=(1, 20), y_axis="down"))
    geometry = replace(geometry, tiles=(a, b))
    with index(tmp_path, geometry) as output:
        output.add(a, source(a), [])
        output.add(b, source(b), [("conductor", box(1, 1, 2, 2))])
        payload = json.loads(output.block(0, 0, 18, 22)[0])
        assert shape(payload["features"][0]["geometry"]).equals(box(3, 14, 5, 17))


def test_limits_invalid_geometry_missing_frames_and_cancel_fail_explicitly(tmp_path):
    geometry = plan((0, 0), size=(6, 6))
    a, b = geometry.tiles
    with index(tmp_path, geometry, limits=VectorLimits(max_features=1)) as output:
        with pytest.raises(DomainValidationError, match="topology"):
            output.add(a, source(a), [("conductor", Polygon([(1, 1), (4, 4), (1, 4), (4, 1)]))])
    with index(tmp_path, geometry) as output:
        output.add(a, source(a), [])
        with pytest.raises(DomainValidationError, match="every geometry"):
            output.block(0, 0, 6, 6)
        output.add(b, source(b), [])
        output.cancelled = lambda: True
        with pytest.raises(StitchCancelled):
            output.block(0, 0, 6, 6)


def test_strict_contour_bridge_uses_existing_hole_codec_and_rejects_partial_cif(tmp_path):
    from contour.domain import PolygonData
    from contour.serializers import save_polygons_cif
    from contour.infrastructure.stitch_cif import load_stitch_cif
    path = tmp_path / "hole.cif"
    save_polygons_cif(path, "image.png", [PolygonData(1, [(1, 1), (9, 1), (9, 9), (1, 9)]),
        PolygonData(2, [(3, 3), (7, 3), (7, 7), (3, 7)], is_hole=True, parent_id=1)], (12, 12))
    def load():
        return load_stitch_cif(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), image_size=(12, 12))
    shapes = load()
    assert len(shapes) == 1 and shapes[0][1].area == 48 and len(shapes[0][1].interiors) == 1
    path.write_text(path.read_text().replace("DF;", "W 2 0 0 5 5;\nDF;"))
    with pytest.raises(ValueError, match="Unsupported CIF"):
        load()


def test_mosaic_vector_extension_validates_block_count_digest_and_provenance(tmp_path):
    import sqlite3
    import numpy as np
    from kraken_manager.infrastructure.mosaic import MosaicReader, write_mosaic
    geometry = plan((2, 0), size=(6, 6))
    renderer = StitchRenderer(geometry, limits=StitchLimits(block_size=3))
    a, b = geometry.tiles
    bindings = [{**source(t), "frame_id": t.frame_id, "sha256": t.calibration.artifact.sha256} for t in geometry.tiles]
    path = tmp_path / "mosaic.sqlite3"
    with VectorStitchIndex(tmp_path / "index.sqlite3", renderer) as vectors:
        vectors.add(a, source(a), [("conductor", box(1, 1, 4, 4))])
        vectors.add(b, source(b), [("conductor", box(0, 1, 3, 4))])
        write_mosaic(path, renderer, lambda tile: np.zeros((6, 6, 3), dtype=np.uint8), job_id="vector-test",
            content_inputs=bindings, vector_index=vectors)
    with MosaicReader(path) as reader:
        reader.validate(expected_plan=geometry)
        assert reader.metadata["vector_stitch"]["conflict_blocks"] > 0
    with sqlite3.connect(path) as db:
        payload, = db.execute("SELECT payload FROM vector_blocks WHERE x=0 AND y=0").fetchone()
        value = json.loads(payload)
        value["features"][0]["source_version_ids"] = ["another-project"]
        encoded = json.dumps(value).encode()
        db.execute("UPDATE vector_blocks SET payload=?,sha256=? WHERE x=0 AND y=0", (encoded, hashlib.sha256(encoded).hexdigest()))
    with MosaicReader(path) as reader, pytest.raises(DomainValidationError, match="provenance"):
        reader.validate()
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM vector_blocks WHERE x=0 AND y=0")
    with MosaicReader(path) as reader, pytest.raises(DomainValidationError, match="block count"):
        reader.validate()


@pytest.mark.parametrize("body,code", [("P 1 1 2 1 1 1;", "degenerate_polygon"), ("P 1 1 2 1;", "malformed_polygon"),
    ("L ;", "empty_layer"), ("P 1 1 4 4 1 4 4 1;", "ambiguous_topology")])
def test_cif_refusal_identifies_the_problem_and_never_omits_a_degenerate_primitive(tmp_path, body, code):
    from contour.infrastructure.stitch_cif import CifStitchError, load_stitch_cif
    path = tmp_path / "invalid.cif"
    path.write_text("DS 1 1 1;\nL NM;\n( S 6 6 );\nP 0 6 1 6 0 5;\n" + body + "\nDF;\nE\n")
    with pytest.raises(CifStitchError) as caught:
        load_stitch_cif(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), image_size=(6, 6))
    assert caught.value.code == code and caught.value.command_index == 5
    if code != "empty_layer":
        assert "command 5" in str(caught.value)


def test_actual_native_vector_failure_report_names_frame_version_polygon_and_command(tmp_path):
    from dataclasses import replace
    from pathlib import Path
    import sys
    from uuid import uuid4
    from kraken_agent.jobs import AgentJobState, DurableJobStore
    from kraken_agent.runner import PluginProcessSpec, PluginRegistry, SubprocessPluginRunner
    from kraken_core.plugin_protocol import PluginAsset
    from kraken_manager.infrastructure.mosaic import STITCH_OPERATION
    from tests.test_stitch_worker import prepare
    geometry, workspace, manifest, _ = prepare(tmp_path)
    assets = [replace(a, role="geometry-source") if a.role == "source" else a for a in manifest.inputs]
    bindings = []
    for i, tile in enumerate(geometry.tiles):
        version = str(uuid4())
        content = b"DS 1 1 1;\nL NM;\n( S 3 2 );\nP 0 2 2 2 0 1;\nP 1 1 2 1 1 1;\nDF;\nE\n"
        digest = hashlib.sha256(content).hexdigest()
        source_path = tmp_path / f"vector-{i}.cif"
        source_path.write_bytes(content)
        relative = f"inputs/vector-{i}.cif"
        workspace.stage_file(source_path, relative, expected_sha256=digest)
        assets.append(PluginAsset(version, "source", "frame", relative, digest, "application/x-cif", frame_id=tile.frame_id, x=i+1, y=1))
        bindings.append({"frame_id": tile.frame_id, "artifact_version_id": version, "sha256": digest})
    manifest = replace(manifest, inputs=tuple(assets), parameters={**manifest.parameters, "content_kind": "vector", "content_inputs": bindings})
    workspace.write_manifest(manifest)
    store = DurableJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue(manifest)
    registry = PluginRegistry({STITCH_OPERATION: PluginProcessSpec(STITCH_OPERATION,
        (sys.executable, "-m", "kraken_manager.infrastructure.stitch_worker"))})
    runner = SubprocessPluginRunner(store, workspace.root, registry,
        environment={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    assert runner.run_once()
    job = store.get(manifest.job_id)
    assert job.state is AgentJobState.FAILED and not job.result.outputs
    error = job.result.report["error"]
    assert geometry.tiles[0].frame_id in error and bindings[0]["artifact_version_id"] in error
    assert "polygon 2" in error and "command 5" in error and "degenerate_polygon" in error
