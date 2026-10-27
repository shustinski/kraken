"""Verified SI exchange, exact pixel geometry, truncation and output safety."""
from dataclasses import replace
import hashlib
import json
from uuid import uuid4

import numpy as np
import pytest
from shapely import Polygon, box, union_all
from shapely.geometry import shape

from kraken_manager.domain.common import DomainValidationError
from kraken_manager.infrastructure.mosaic import MosaicReader, canonical, write_mosaic
from kraken_manager.infrastructure.stitch_reads import vector_block
from kraken_manager.infrastructure.stitch_vectors import VectorStitchIndex
from kraken_manager.infrastructure.stitching import StitchCancelled, StitchLimits, StitchRenderer
from kraken_manager.infrastructure.vector_exchange import VectorExchangeReader, header_from_mosaic, write_vector_exchange
from tests.test_computed_stitching import plan


@pytest.fixture
def vector_source(tmp_path):
    geometry = plan((2, 0), size=(8, 8))
    # Actual SI conversion must preserve anisotropy, origin and downwards Y.
    geometry = replace(geometry, tiles=tuple(replace(t, calibration=replace(t.calibration,
        pixel_size_m=(2e-6, 3e-6), origin_m=(10e-6, -5e-6), y_axis="down")) for t in geometry.tiles))
    # The second tile is colocated here: its offset is explicitly SI metres.
    geometry = replace(geometry, tiles=(geometry.tiles[0], replace(geometry.tiles[1], affine=(1, 0, 0, 0, 1, 0))))
    renderer = StitchRenderer(geometry, limits=StitchLimits(block_size=3))
    bindings = [{"frame_id": t.frame_id, "artifact_version_id": t.calibration.artifact.version_id,
        "sha256": t.calibration.artifact.sha256} for t in geometry.tiles]
    ring = Polygon([(1, 1), (6, 1), (6, 6), (1, 6)], [[(2, 2), (4, 2), (4, 4), (2, 4)]])
    path = tmp_path / "mosaic.sqlite3"
    with VectorStitchIndex(tmp_path / "index.sqlite3", renderer) as index:
        index.add(geometry.tiles[0], bindings[0], [("conductor", ring)])
        index.add(geometry.tiles[1], bindings[1], [("conductor", box(4, 1, 7, 6))])
        write_mosaic(path, renderer, lambda tile: np.zeros((8, 8, 3), np.uint8), job_id=str(uuid4()),
            content_inputs=bindings, vector_index=index)
    from types import SimpleNamespace
    version = SimpleNamespace(id=str(uuid4()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with MosaicReader(path) as reader:
        header = header_from_mosaic(reader, project_id=str(uuid4()), output_version_id=version.id, output_sha256=version.sha256)
    return path, header, version, ring.union(box(4, 1, 7, 6))


def export(vector_source, destination, **kwargs):
    path, header, version, _ = vector_source
    with MosaicReader(path) as reader:
        return write_vector_exchange(destination, header, lambda c, r: vector_block(reader, version, c, r), **kwargs)


def test_lossless_pixel_and_si_polygons_provenance_conflicts_and_complete_seal(vector_source, tmp_path):
    destination = tmp_path / "vectors.jsonl"
    result = export(vector_source, destination)
    with VectorExchangeReader(destination) as reader:
        assert not reader.verified
        records = list(reader.blocks())
        assert reader.verified and len(records) == result["blocks"] == 9
        assert reader.header["coverage_semantics"] == "sampled_reference_pixel_centres"
        assert np.frombuffer(reader.coverage(records[0]), dtype="<u4").max() == 2
    actual = union_all([shape(f["geometry"]) for r in records for f in r["features"]])
    assert actual.equals(vector_source[-1])
    from shapely.affinity import affine_transform
    physical = union_all([shape(f["geometry_m"]) for r in records for f in r["features"]])
    expected = affine_transform(actual, [2e-6, 0, 0, -3e-6, 10e-6, -5e-6])
    assert physical.symmetric_difference(expected).area < 1e-25
    assert any(r["conflicts"] for r in records)
    assert not list(tmp_path.glob(".*.part-*"))


@pytest.mark.parametrize("fault", ["truncated", "wrong_physical", "wrong_source", "duplicate", "extra"])
def test_exchange_consumer_refuses_incomplete_or_inconsistent_stream_even_with_resealed_bytes(vector_source, tmp_path, fault):
    destination = tmp_path / "vectors.jsonl"
    export(vector_source, destination)
    values = [json.loads(line) for line in destination.read_text().splitlines()]
    if fault == "truncated":
        destination.write_bytes(destination.read_bytes().rsplit(b"\n", 2)[0]+b"\n")
    else:
        if fault == "wrong_physical":
            item = next(r for r in values if r.get("features"))
            item["features"][0]["geometry_m"] = item["features"][0]["geometry"]
        if fault == "wrong_source":
            item = next(r for r in values if r.get("features"))
            item["features"][0]["source_version_ids"] = [str(uuid4())]
        if fault == "duplicate":
            values[2] = values[1]
        lines = [(canonical(v)+"\n").encode() for v in values[:-1]]
        values[-1]["sha256"] = hashlib.sha256(b"".join(lines)).hexdigest()
        destination.write_bytes(b"".join(lines)+(canonical(values[-1])+"\n").encode()+(b"junk" if fault == "extra" else b""))
    with VectorExchangeReader(destination) as reader, pytest.raises((DomainValidationError, ValueError)):
        list(reader.blocks())
    assert not reader.verified


def test_export_failure_cancel_or_existing_destination_never_destroys_user_file(vector_source, tmp_path):
    destination = tmp_path / "vectors.jsonl"
    destination.write_bytes(b"existing engineering document")
    with pytest.raises(FileExistsError):
        export(vector_source, destination)
    assert destination.read_bytes() == b"existing engineering document"
    with pytest.raises(StitchCancelled):
        export(vector_source, tmp_path / "cancelled.jsonl", cancelled=lambda: True)
    with pytest.raises(DomainValidationError, match="budget"):
        export(vector_source, tmp_path / "budget.jsonl", max_bytes=10)
    assert not (tmp_path / "cancelled.jsonl").exists() and not (tmp_path / "budget.jsonl").exists()
    assert not list(tmp_path.glob(".*.part-*"))


def test_source_identity_change_mid_export_refuses_publication(vector_source, tmp_path):
    path, header, version, _ = vector_source
    destination = tmp_path / "changed.jsonl"
    with MosaicReader(path) as reader:
        def load(c, r):
            payload = vector_block(reader, version, c, r)
            if c == 1:
                payload["output_version_id"] = str(uuid4())
            return payload
        with pytest.raises(DomainValidationError, match="changed"):
            write_vector_exchange(destination, header, load)
    assert not destination.exists() and not list(tmp_path.glob(".*.part-*"))


def test_legacy_exchange_without_coverage_is_readable_as_unknown(vector_source, tmp_path):
    path, header, version, _ = vector_source
    header = {k: v for k, v in header.items() if k != "coverage_semantics"}
    destination = tmp_path / "legacy.jsonl"
    with MosaicReader(path) as reader:
        write_vector_exchange(destination, header, lambda c, r: vector_block(reader, version, c, r))
    with VectorExchangeReader(destination) as reader:
        records = list(reader.blocks())
        assert reader.verified and reader.coverage(records[0]) is None


def test_vector_exchange_preserves_nodata_and_detects_corrupted_coverage(tmp_path):
    import base64
    from kraken_manager.infrastructure.vector_exchange import decode_coverage
    geometry = plan((6, 0), size=(3, 2))
    renderer = StitchRenderer(geometry)
    path = tmp_path / "gapped.sqlite3"
    bindings = [{"frame_id": t.frame_id, "artifact_version_id": t.calibration.artifact.version_id,
        "sha256": t.calibration.artifact.sha256} for t in geometry.tiles]
    with VectorStitchIndex(tmp_path / "gapped-index.sqlite3", renderer) as index:
        for tile, binding in zip(geometry.tiles, bindings):
            index.add(tile, binding, [])
        write_mosaic(path, renderer, lambda tile: np.zeros((2, 3, 3), np.uint8), job_id=str(uuid4()),
            content_inputs=bindings, vector_index=index)
    from types import SimpleNamespace
    version = SimpleNamespace(id=str(uuid4()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    destination = tmp_path / "gapped.jsonl"
    with MosaicReader(path) as reader:
        header = header_from_mosaic(reader, project_id=str(uuid4()), output_version_id=version.id, output_sha256=version.sha256)
        expected = reader.read_block(0, 0).coverage
        write_vector_exchange(destination, header, lambda c, r: vector_block(reader, version, c, r))
    with VectorExchangeReader(destination) as reader:
        record, = list(reader.blocks())
        actual = np.frombuffer(reader.coverage(record), dtype="<u4").reshape(expected.shape)
        np.testing.assert_array_equal(actual, expected)
        assert set(actual.flatten()) == {0, 1}
    record["coverage"]["data_base64"] = base64.b64encode(b"damaged").decode()
    with pytest.raises(DomainValidationError, match="coverage"):
        decode_coverage(record, max_sources=2)
