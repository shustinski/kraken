"""Lossless partitioned engineering vectors, not geographic GeoJSON.

JSON Lines carries both reference-pixel polygons and calibrated SI polygons.
Every block retains conflicts and provenance. The final seal covers the whole
stream; consumers must exhaust/verify it before accepting a complete result.
"""
from dataclasses import asdict
from contextlib import ExitStack
import base64
import hashlib
import json
import os
import struct
import zlib
from pathlib import Path
from uuid import uuid4

from kraken_core.safe_files import open_exclusive_write, open_regular_read
from kraken_manager.domain.common import DomainValidationError, validate_uuid
from kraken_manager.domain.artifacts import validate_sha256
from kraken_manager.domain.multilayer import ArtifactAnchor, CoordinateCalibration
from .mosaic import canonical
from .stitch_vectors import _valid, _vertices, VectorLimits
from .stitching import StitchCancelled

SCHEMA = "kraken.layer-vectors.v1"
MAX_LINE = 32 * 1024**2


def header_from_mosaic(reader, *, project_id, output_version_id, output_sha256):
    if "vector_stitch" not in reader.metadata:
        raise DomainValidationError("This mosaic contains no vector content")
    reference = next(t for t in reader.plan.tiles if t.frame_id == reader.plan.reference_frame_id)
    return {"type": "header", "schema": SCHEMA, "unit": "m", "project_id": str(project_id),
        "layer_id": reader.plan.layer_id, "job_id": reader.metadata["job_id"],
        "output_version_id": str(output_version_id), "output_sha256": output_sha256,
        "plan_sha256": reader.metadata["plan_sha256"], "render": reader.metadata["render"],
        "reference_calibration": asdict(reference.calibration), "content_inputs": reader.metadata["content_inputs"],
        "vector_stitch": reader.metadata["vector_stitch"], "physical_accuracy": "not_established",
        "coverage_semantics": "sampled_reference_pixel_centres",
        "partition": "closed_block_fragments_in_common_coordinates"}


def _header(value):
    if value.get("type") != "header" or value.get("schema") != SCHEMA or value.get("unit") != "m":
        raise DomainValidationError("unsupported engineering vector exchange")
    for name in ("project_id", "layer_id", "job_id", "output_version_id"):
        validate_uuid(value[name])
    for name in ("output_sha256", "plan_sha256"):
        validate_sha256(value[name])
    calibration = dict(value["reference_calibration"])
    calibration["artifact"] = ArtifactAnchor(**calibration["artifact"])
    calibration = CoordinateCalibration(**calibration)
    if calibration.layer_id != value["layer_id"]:
        raise DomainValidationError("exchange calibration belongs to another layer")
    inputs = value["content_inputs"]
    if not isinstance(inputs, list) or not inputs or len(inputs) > 100000:
        raise DomainValidationError("vector exchange requires source provenance")
    for source in inputs:
        validate_uuid(source["frame_id"])
        ArtifactAnchor(source["artifact_version_id"], source["sha256"])
    if len({s["frame_id"] for s in inputs}) != len(inputs) or len({s["artifact_version_id"] for s in inputs}) != len(inputs):
        raise DomainValidationError("exchange sources must be unique")
    render = value["render"]
    bounds, step = render["bounds"], render["block_size"]
    if (not isinstance(bounds, list) or len(bounds) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in bounds)
            or isinstance(step, bool) or not isinstance(step, int) or not 0 < step <= 1024):
        raise DomainValidationError("invalid exchange block grid")
    left, top, right, bottom = bounds
    columns, rows = (right-left+step-1)//step, (bottom-top+step-1)//step
    if (min(right-left, bottom-top) <= 0 or columns*rows > 1000000 or render["block_count"] != columns*rows
            or render["width"] != right-left or render["height"] != bottom-top):
        raise DomainValidationError("exchange grid exceeds its bounds or count")
    policy = value["vector_stitch"]
    if value.get("coverage_semantics", "unavailable") not in {"unavailable", "sampled_reference_pixel_centres"}:
        raise DomainValidationError("unsupported vector coverage semantics")
    if (policy.get("electrical_connectivity") != "unverified" or policy.get("coordinates") != "reference_pixel_centres"
            or value.get("physical_accuracy") != "not_established"
            or value.get("partition") != "closed_block_fragments_in_common_coordinates"):
        raise DomainValidationError("exchange cannot claim geometric or electrical acceptance")
    if policy.get("schema") != "kraken.vector-blocks.v1" or policy.get("policy") != "category_union_with_overlap_disagreement":
        raise DomainValidationError("unsupported vector exchange policy")
    for name in ("feature_blocks", "conflict_blocks", "source_features", "source_vertices"):
        if isinstance(policy.get(name), bool) or not isinstance(policy.get(name), int) or policy[name] < 0:
            raise DomainValidationError("invalid vector exchange counts")
    if max(policy["feature_blocks"], policy["conflict_blocks"]) > columns*rows:
        raise DomainValidationError("vector exchange counts exceed its grid")
    return calibration, columns, rows


def _si_geometry(geometry, calibration):
    from shapely.affinity import affine_transform
    from shapely.geometry import mapping, shape
    sx, sy = calibration.pixel_size_m
    ox, oy = calibration.origin_m
    return json.loads(canonical(mapping(affine_transform(shape(geometry), [sx, 0, 0, sy if calibration.y_axis == "up" else -sy, ox, oy]))))


def exchange_block(header, column, row, payload):
    calibration, columns, rows = _header(header)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (column, row)) or not 0 <= column < columns or not 0 <= row < rows:
        raise DomainValidationError("vector exchange block index is outside its grid")
    if any(payload.get(k) != header[k] for k in ("job_id", "output_version_id", "output_sha256")):
        raise DomainValidationError("vector source changed during export")
    step = header["render"]["block_size"]
    left, top, right, bottom = header["render"]["bounds"]
    x, y = left+column*step, top+row*step
    record = {"type": "block", "column": column, "row": row, "x": x, "y": y,
        "width": min(step, right-x), "height": min(step, bottom-y)}
    if header.get("coverage_semantics") == "sampled_reference_pixel_centres":
        record["coverage"] = payload["coverage"]
    for key in ("features", "conflicts"):
        if not isinstance(payload[key], list) or len(payload[key]) > VectorLimits().max_block_features:
            raise DomainValidationError("exchange exceeds its block feature budget")
        record[key] = [{**feature, "geometry_m": _si_geometry(feature["geometry"], calibration)} for feature in payload[key]]
    _block(header, record, column, row, calibration)
    return record


def _block(header, record, column, row, calibration):
    from shapely.geometry import shape
    sources = {s["artifact_version_id"]: s["frame_id"] for s in header["content_inputs"]}
    step = header["render"]["block_size"]
    left, top, right, bottom = header["render"]["bounds"]
    x, y = left+column*step, top+row*step
    expected = {"type": "block", "column": column, "row": row, "x": x, "y": y,
        "width": min(step, right-x), "height": min(step, bottom-y)}
    if any(record.get(k) != v for k, v in expected.items()):
        raise DomainValidationError("exchange has a missing, duplicate or out-of-order block")
    if any(isinstance(record.get(k), bool) or not isinstance(record.get(k), int) for k in expected if k != "type"):
        raise DomainValidationError("exchange block coordinates must be integers")
    if header.get("coverage_semantics") == "sampled_reference_pixel_centres":
        decode_coverage(record, max_sources=len(sources))
    vertices = 0
    for key in ("features", "conflicts"):
        if not isinstance(record[key], list) or len(record[key]) > VectorLimits().max_block_features:
            raise DomainValidationError("exchange exceeds its block feature budget")
        for feature in record[key]:
            if not isinstance(feature.get("category"), str) or not 0 < len(feature["category"]) <= 64:
                raise DomainValidationError("invalid vector category")
            identifiers = feature["source_version_ids"]
            if (not isinstance(identifiers, list) or not identifiers or not all(isinstance(v, str) for v in identifiers)
                    or len(set(identifiers)) != len(identifiers) or set(identifiers)-sources.keys()):
                raise DomainValidationError("exchange vector provenance is inconsistent")
            if key == "conflicts" and (len(identifiers) != 2 or feature["frame_ids"] != [sources[v] for v in identifiers]
                    or len(set(feature["frame_ids"])) != 2):
                raise DomainValidationError("invalid disagreement provenance")
            geometry = shape(feature["geometry"])
            _valid(geometry, VectorLimits().max_block_vertices)
            physical = shape(feature["geometry_m"])
            _valid(physical, VectorLimits().max_block_vertices)
            if physical.is_empty or geometry.is_empty or feature["geometry_m"] != json.loads(canonical(_si_geometry(feature["geometry"], calibration))):
                raise DomainValidationError("exchange physical geometry differs from its calibrated pixel geometry")
            vertices += _vertices(geometry)
            if vertices > VectorLimits().max_block_vertices:
                raise DomainValidationError("exchange exceeds its cumulative block vertex budget")
            a, b, c, d = geometry.bounds
            if a < x-.5-1e-7 or b < y-.5-1e-7 or c > x+expected["width"]-.5+1e-7 or d > y+expected["height"]-.5+1e-7:
                raise DomainValidationError("exchange geometry extends outside its block")


def decode_coverage(record, *, max_sources):
    """Return validated uint32-le bytes. Zero denotes unknown/uncovered pixels."""
    if (any(isinstance(record.get(k), bool) or not isinstance(record.get(k), int) or not 0 < record[k] <= 1024 for k in ("width", "height"))
            or isinstance(max_sources, bool) or not isinstance(max_sources, int) or not 0 < max_sources <= 100000):
        raise DomainValidationError("invalid vector coverage grid or source count")
    value = record["coverage"]
    if value.get("encoding") != "zlib-uint32-le" or len(value["data_base64"]) > 8*1024**2:
        raise DomainValidationError("invalid or excessive vector coverage encoding")
    compressed = base64.b64decode(value["data_base64"], validate=True)
    size = record["width"]*record["height"]*4
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(compressed, size+1)
    except zlib.error as exc:
        raise DomainValidationError("invalid vector coverage compression") from exc
    if (len(raw) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail
            or hashlib.sha256(raw).hexdigest() != value["sha256"]):
        raise DomainValidationError("vector coverage size or checksum mismatch")
    if any(v > max_sources for (v,) in struct.iter_unpack("<I", raw)):
        raise DomainValidationError("vector coverage count exceeds its source frames")
    return raw


class VectorExchangeReader:
    def __init__(self, path):
        self._stack = ExitStack()
        self.stream = self._stack.enter_context(open_regular_read(path))
        self.verified = False
        self.digest = hashlib.sha256()
        try:
            self.header, raw = self._line()
            self.digest.update(raw)
            self.calibration, self.columns, self.rows = _header(self.header)
        except BaseException:
            self._stack.close()
            raise

    def _line(self):
        raw = self.stream.readline(MAX_LINE+1)
        if not raw or len(raw) > MAX_LINE or not raw.endswith(b"\n"):
            raise DomainValidationError("vector exchange is truncated or exceeds its line budget")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise DomainValidationError("exchange record must be an object")
        return value, raw

    def blocks(self):
        if getattr(self, "_started", False):
            raise DomainValidationError("vector exchange consumer can be exhausted only once")
        self._started = True
        counts = {"feature_blocks": 0, "conflict_blocks": 0}
        for row in range(self.rows):
            for column in range(self.columns):
                record, raw = self._line()
                _block(self.header, record, column, row, self.calibration)
                self.digest.update(raw)
                for name, key in (("feature_blocks", "features"), ("conflict_blocks", "conflicts")):
                    counts[name] += bool(record[key])
                yield record
        end, _ = self._line()
        if (end != {"type": "seal", "blocks": self.columns*self.rows, "sha256": self.digest.hexdigest()}
                or self.stream.read(1) or any(self.header["vector_stitch"][name] != value for name, value in counts.items())):
            raise DomainValidationError("exchange final seal or complete-content counts mismatch")
        self.verified = True

    def __enter__(self):
        return self

    def coverage(self, record):
        if self.header.get("coverage_semantics") != "sampled_reference_pixel_centres":
            return None
        return decode_coverage(record, max_sources=len(self.header["content_inputs"]))

    def __exit__(self, *_):
        self._stack.close()


def write_vector_exchange(destination, header, load_block, *, cancelled=lambda: False, max_bytes=128*1024**3):
    """Publish a new local file only after complete validation; never overwrite."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 0 < max_bytes <= 512*1024**3:
        raise DomainValidationError("invalid vector export disk budget")
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(str(destination))
    temporary = destination.with_name("."+destination.name+".part-"+uuid4().hex)
    _, columns, rows = _header(header)
    digest, size = hashlib.sha256(), 0
    identity = None
    try:
        with open_exclusive_write(temporary) as stream:
            stat = os.fstat(stream.fileno())
            identity = stat.st_dev, stat.st_ino
            def emit(value, *, sealed=False):
                nonlocal size
                data = (canonical(value)+"\n").encode("utf-8")
                size += len(data)
                if len(data) > MAX_LINE or size > max_bytes:
                    raise DomainValidationError("vector export exceeds its output budget")
                stream.write(data)
                if not sealed:
                    digest.update(data)
            emit(header)
            for row in range(rows):
                for column in range(columns):
                    if cancelled():
                        raise StitchCancelled("vector export cancelled")
                    emit(exchange_block(header, column, row, load_block(column, row)))
            emit({"type": "seal", "blocks": columns*rows, "sha256": digest.hexdigest()}, sealed=True)
            stream.flush()
            os.fsync(stream.fileno())
        with VectorExchangeReader(temporary) as reader:
            for _ in reader.blocks():
                if cancelled():
                    raise StitchCancelled("vector export cancelled")
            assert reader.verified
        if cancelled():
            raise StitchCancelled("vector export cancelled")
        stat = temporary.stat(follow_symlinks=False)
        if (stat.st_dev, stat.st_ino) != identity:
            raise DomainValidationError("vector export staging file was replaced")
        if os.name == "nt":
            # Windows rename refuses an existing destination, including on FAT
            # volumes where hard links are unavailable. POSIX rename overwrites.
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
        return {"path": str(destination), "blocks": columns*rows, "size_bytes": size,
            "content_sha256": digest.hexdigest(), "output_version_id": header["output_version_id"]}
    finally:
        if identity is not None:
            try:
                stat = temporary.stat(follow_symlinks=False)
                if (stat.st_dev, stat.st_ino) == identity:
                    temporary.unlink()
            except FileNotFoundError:
                pass
