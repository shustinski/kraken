"""Immutable, indexed mosaic artifact; decode and validate one bounded block.

SQLite is the block container, not a second project database. Its closed bytes
are imported as one content-addressed layer attachment. No canvas-sized array
or in-memory archive index is needed for reading/writing a large mosaic.
"""
from __future__ import annotations

from dataclasses import asdict
from contextlib import closing
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import zlib

from kraken_manager.domain.common import DomainValidationError
from kraken_core.plugin_protocol import MOSAIC_MEDIA_TYPE as MOSAIC_MEDIA_TYPE
from kraken_manager.domain.stitch_plan import LayerStitchRevision
from .stitching import StitchBlock, StitchCancelled, StitchLimits, StitchRenderer

MOSAIC_SCHEMA = "kraken.layer-mosaic.v1"
STITCH_OPERATION = "layer.stitch.v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(plan):
    return hashlib.sha256(canonical(asdict(plan)).encode()).hexdigest()


def _cancelled(callback):
    if callback():
        raise StitchCancelled("stitch computation cancelled")


def write_mosaic(path, renderer, load_source, *, job_id, cancelled=lambda: False, progress=lambda *_: None,
                 max_file_bytes=128 * 1024**3, content_inputs=None, vector_index=None):
    """Use a caller-owned new path; a failed/cancelled artifact is never returned."""
    from PIL import Image
    from kraken_core.safe_files import open_exclusive_write

    path = Path(path)
    with open_exclusive_write(path):
        pass
    try:
        with closing(sqlite3.connect(path)) as db:
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("PRAGMA cache_size=-8192")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("CREATE TABLE metadata (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)")
            db.execute("CREATE TABLE blocks (x INTEGER NOT NULL,y INTEGER NOT NULL,width INTEGER NOT NULL,height INTEGER NOT NULL,"
                       "pixels BLOB NOT NULL,pixels_sha TEXT NOT NULL,coverage BLOB NOT NULL,coverage_sha TEXT NOT NULL,PRIMARY KEY(x,y)) WITHOUT ROWID")
            metadata = {"schema": MOSAIC_SCHEMA, "job_id": job_id, "plan": asdict(renderer.plan),
                "plan_sha256": fingerprint(renderer.plan), "render": renderer.preflight(),
                "sampling": "nearest" if renderer.mode == "labels" else "bilinear",
                "overlap": "first_frame_id" if renderer.mode == "labels" else "equal_weight",
                "coverage_encoding": "zlib-uint32-le", "algorithm_version": "1"}
            if content_inputs:
                metadata["content_inputs"] = content_inputs
            if vector_index is not None:
                db.execute("CREATE TABLE vector_blocks (x INTEGER NOT NULL,y INTEGER NOT NULL,payload BLOB NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(x,y)) WITHOUT ROWID")
                metadata["vector_stitch"] = {"schema": "kraken.vector-blocks.v1", "coordinates": "reference_pixel_centres",
                    "policy": "category_union_with_overlap_disagreement", "electrical_connectivity": "unverified",
                    "source_features": vector_index.features, "source_vertices": vector_index.vertices,
                    "conflict_blocks": 0, "feature_blocks": 0}
            db.execute("INSERT INTO metadata VALUES(1,?)", (canonical(metadata),))
            for index, block in enumerate(renderer.blocks(load_source, cancelled=cancelled), 1):
                _cancelled(cancelled)
                image = BytesIO()
                Image.fromarray(block.pixels).save(image, format="PNG")
                data = image.getvalue()
                coverage = block.coverage.astype("<u4", copy=False).tobytes()
                height, width = block.coverage.shape
                db.execute("INSERT INTO blocks VALUES(?,?,?,?,?,?,?,?)", (block.x, block.y, width, height,
                    data, hashlib.sha256(data).hexdigest(), zlib.compress(coverage), hashlib.sha256(coverage).hexdigest()))
                if vector_index is not None:
                    payload, digest = vector_index.block(block.x, block.y, width, height)
                    value = json.loads(payload)
                    metadata["vector_stitch"]["conflict_blocks"] += bool(value["conflicts"])
                    metadata["vector_stitch"]["feature_blocks"] += bool(value["features"])
                    db.execute("INSERT INTO vector_blocks VALUES(?,?,?,?)", (block.x, block.y, payload, digest))
                # Commit by block: bounds the rollback journal. This file is not
                # authoritative until final close + checksum + project import.
                db.commit()
                if path.stat().st_size > max_file_bytes:
                    raise DomainValidationError("mosaic exceeds output disk budget")
                progress(index, renderer.block_count)
            _cancelled(cancelled)
            if vector_index is not None:
                db.execute("UPDATE metadata SET payload=? WHERE id=1", (canonical(metadata),))
                db.commit()
            db.execute("PRAGMA optimize")
        return metadata
    except BaseException:
        path.unlink(missing_ok=True)
        Path(str(path) + "-journal").unlink(missing_ok=True)
        raise


class MosaicReader:
    def __init__(self, path):
        self.path = Path(path)
        self.db = sqlite3.connect(self.path.absolute().as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            self.db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 16 * 1024**2)
            self.db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 128 * 1024)
            self.db.execute("PRAGMA trusted_schema=OFF")
            self.db.execute("PRAGMA cache_size=-8192")
            objects = self.db.execute("SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall()
            base = {("table", "metadata"), ("table", "blocks")}
            if set(objects) not in (base, base | {("table", "vector_blocks")}):
                raise DomainValidationError("mosaic container has an unexpected schema")
            rows = self.db.execute("SELECT payload FROM metadata WHERE id=1").fetchall()
            if len(rows) != 1 or self.db.execute("SELECT count(*) FROM metadata").fetchone()[0] != 1:
                raise DomainValidationError("mosaic requires exactly one metadata record")
            self.metadata = json.loads(rows[0][0])
            if ("vector_stitch" in self.metadata) != (("table", "vector_blocks") in set(objects)):
                raise DomainValidationError("mosaic vector extension metadata and storage disagree")
            if "vector_stitch" in self.metadata:
                value = self.metadata["vector_stitch"]
                expected = {"schema": "kraken.vector-blocks.v1", "coordinates": "reference_pixel_centres",
                    "policy": "category_union_with_overlap_disagreement", "electrical_connectivity": "unverified"}
                if any(value.get(k) != v for k, v in expected.items()):
                    raise DomainValidationError("unsupported mosaic vector extension policy")
                for name in ("source_features", "source_vertices", "conflict_blocks", "feature_blocks"):
                    if isinstance(value.get(name), bool) or not isinstance(value.get(name), int) or value[name] < 0:
                        raise DomainValidationError("invalid mosaic vector extension counts")
            if self.metadata["schema"] != MOSAIC_SCHEMA or self.metadata["algorithm_version"] != "1":
                raise DomainValidationError("unsupported mosaic format")
            self.plan = LayerStitchRevision.from_dict(self.metadata["plan"])
            # Hash the serialized plan, before adding future optional domain
            # defaults. Existing v1 containers remain immutable and readable.
            if self.metadata["plan_sha256"] != hashlib.sha256(canonical(self.metadata["plan"]).encode()).hexdigest():
                raise DomainValidationError("mosaic plan fingerprint mismatch")
            render = self.metadata["render"]
            self.renderer = StitchRenderer(self.plan, mode=render["mode"], limits=StitchLimits(block_size=render["block_size"]))
            if self.renderer.preflight() != render:
                raise DomainValidationError("mosaic render grid disagrees with its geometry")
            expected_sampling = "nearest" if self.renderer.mode == "labels" else "bilinear"
            expected_overlap = "first_frame_id" if self.renderer.mode == "labels" else "equal_weight"
            if (self.metadata["sampling"] != expected_sampling or self.metadata["overlap"] != expected_overlap
                    or self.metadata["coverage_encoding"] != "zlib-uint32-le"):
                raise DomainValidationError("mosaic sampling policy mismatch")
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def read_block(self, x, y):
        import numpy as np
        from PIL import Image

        row = self.db.execute("SELECT width,height,pixels,pixels_sha,coverage,coverage_sha FROM blocks WHERE x=? AND y=?", (x, y)).fetchone()
        if row is None:
            raise DomainValidationError("mosaic block is missing")
        width, height, data, digest, compressed, coverage_sha = row
        left, top, right, bottom = self.renderer.bounds
        step = self.renderer.limits.block_size
        if (not left <= x < right or not top <= y < bottom or (x-left) % step or (y-top) % step
                or width != min(step, right-x) or height != min(step, bottom-y)):
            raise DomainValidationError("mosaic block has invalid grid dimensions")
        if hashlib.sha256(data).hexdigest() != digest:
            raise DomainValidationError("mosaic pixel checksum mismatch")
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or image.size != (width, height):
                raise DomainValidationError("mosaic block PNG dimensions mismatch")
            expected = {"rgb": {"RGB"}, "gray": {"L"}, "labels": {"I;16", "I;16L", "I;16B", "I"}}[self.renderer.mode]
            if image.mode not in expected:
                raise DomainValidationError("mosaic block pixel mode mismatch")
            pixels = np.asarray(image).copy()
        size = width * height * 4
        decoder = zlib.decompressobj()
        try:
            raw = decoder.decompress(compressed, size + 1)
        except zlib.error as exc:
            raise DomainValidationError("mosaic coverage compression is invalid") from exc
        if len(raw) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise DomainValidationError("mosaic coverage exceeds its bounded grid")
        if hashlib.sha256(raw).hexdigest() != coverage_sha:
            raise DomainValidationError("mosaic coverage checksum mismatch")
        coverage = np.frombuffer(raw, dtype="<u4").reshape(height, width).copy()
        if coverage.max() > len(self.plan.tiles):
            raise DomainValidationError("mosaic coverage count exceeds input tiles")
        return StitchBlock(x, y, pixels, coverage)

    def check_request(self, *, expected_plan=None, expected_job_id=None, mode=None, block_size=None):
        if expected_plan is not None and self.plan != expected_plan:
            raise DomainValidationError("mosaic belongs to another stitch revision")
        if expected_job_id is not None and self.metadata["job_id"] != expected_job_id:
            raise DomainValidationError("mosaic belongs to another job")
        if ((mode is not None and self.renderer.mode != mode)
                or (block_size is not None and self.renderer.limits.block_size != block_size)):
            raise DomainValidationError("mosaic policy differs from the requested policy")

    def read_vectors(self, x, y):
        if "vector_stitch" not in self.metadata:
            return {"features": [], "conflicts": []}
        from shapely.geometry import shape
        from shapely.errors import ShapelyError
        from .stitch_vectors import _valid, _vertices, VectorLimits
        row = self.db.execute("SELECT payload,sha256 FROM vector_blocks WHERE x=? AND y=?", (x, y)).fetchone()
        if row is None or len(row[0]) > VectorLimits().max_payload_bytes or hashlib.sha256(row[0]).hexdigest() != row[1]:
            raise DomainValidationError("mosaic vector block missing, excessive or checksum mismatch")
        value = json.loads(row[0])
        if not isinstance(value, dict) or set(value) != {"features", "conflicts"}:
            raise DomainValidationError("invalid mosaic vector block structure")
        sources = {p["artifact_version_id"]: p["frame_id"] for p in self.metadata.get("content_inputs", ())}
        if (len(sources) != len(self.plan.tiles) or set(sources.values()) != {t.frame_id for t in self.plan.tiles}):
            raise DomainValidationError("mosaic vector sources do not cover its calibrated frames")
        step = self.renderer.limits.block_size
        left, top, right, bottom = self.renderer.bounds
        if not left <= x < right or not top <= y < bottom or (x-left) % step or (y-top) % step:
            raise DomainValidationError("vector block is outside the mosaic grid")
        bounds = (x-.5, y-.5, min(x+step, right)-.5, min(y+step, bottom)-.5)
        vertices = 0
        for key in ("features", "conflicts"):
            if not isinstance(value[key], list) or len(value[key]) > VectorLimits().max_block_features:
                raise DomainValidationError("mosaic vector block exceeds its feature budget")
            for feature in value[key]:
                if not isinstance(feature, dict) or not isinstance(feature.get("category"), str) or not 0 < len(feature["category"]) <= 64:
                    raise DomainValidationError("invalid mosaic vector category")
                identifiers = feature.get("source_version_ids")
                if (not isinstance(identifiers, list) or not identifiers or not all(isinstance(v, str) for v in identifiers)
                        or len(set(identifiers)) != len(identifiers) or set(identifiers) - sources.keys()):
                    raise DomainValidationError("mosaic vector provenance differs from its content inputs")
                if key == "conflicts" and (len(identifiers) != 2 or feature.get("frame_ids") != [sources[v] for v in identifiers]
                        or len(set(feature["frame_ids"])) != 2):
                    raise DomainValidationError("invalid mosaic vector disagreement sources")
                try:
                    geometry = shape(feature["geometry"])
                    _valid(geometry, VectorLimits().max_block_vertices)
                except (ValueError, TypeError, KeyError, AttributeError, ShapelyError) as exc:
                    raise DomainValidationError("mosaic vector geometry is invalid") from exc
                vertices += _vertices(geometry)
                if vertices > VectorLimits().max_block_vertices:
                    raise DomainValidationError("mosaic vector block exceeds its cumulative vertex budget")
                if geometry.is_empty or any(a < b-1e-7 for a, b in zip(geometry.bounds[:2], bounds[:2])) or any(a > b+1e-7 for a, b in zip(geometry.bounds[2:], bounds[2:])):
                    raise DomainValidationError("mosaic vector geometry extends outside its block")
        return value

    def validate(self, *, expected_plan=None, expected_job_id=None, cancelled=lambda: False):
        self.check_request(expected_plan=expected_plan, expected_job_id=expected_job_id)
        if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise DomainValidationError("mosaic database integrity check failed")
        if self.db.execute("SELECT count(*) FROM blocks").fetchone()[0] != self.renderer.block_count:
            raise DomainValidationError("mosaic block count is incomplete or excessive")
        left, top, right, bottom = self.renderer.bounds
        step = self.renderer.limits.block_size
        vector_counts = {"conflict_blocks": 0, "feature_blocks": 0}
        if "vector_stitch" in self.metadata and self.db.execute("SELECT count(*) FROM vector_blocks").fetchone()[0] != self.renderer.block_count:
            raise DomainValidationError("mosaic vector block count is incomplete or excessive")
        for y in range(top, bottom, step):
            for x in range(left, right, step):
                _cancelled(cancelled)
                self.read_block(x, y)
                if "vector_stitch" in self.metadata:
                    vectors = self.read_vectors(x, y)
                    vector_counts["conflict_blocks"] += bool(vectors["conflicts"])
                    vector_counts["feature_blocks"] += bool(vectors["features"])
        if "vector_stitch" in self.metadata and any(self.metadata["vector_stitch"][key] != count for key, count in vector_counts.items()):
            raise DomainValidationError("mosaic vector counts differ from its complete contents")
        return self.metadata


def block_preview(reader, column, row, *, coverage=False):
    """One bounded PNG and a small header; never serializes the full tile plan."""
    import numpy as np
    from PIL import Image

    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (column, row)):
        raise DomainValidationError("mosaic block indices must be nonnegative integers")
    left, top, right, bottom = reader.renderer.bounds
    step = reader.renderer.limits.block_size
    columns, rows = (right-left+step-1)//step, (bottom-top+step-1)//step
    if column >= columns or row >= rows:
        raise DomainValidationError("mosaic block index is outside the render grid")
    x, y = left + column * step, top + row * step
    block = reader.read_block(x, y)
    pixels = block.pixels
    if coverage:
        pixels = np.zeros((*block.coverage.shape, 3), dtype=np.uint8)
        pixels[block.coverage == 1] = (255, 255, 255)
        pixels[block.coverage > 1] = (0, 220, 255)
    preview = Image.fromarray(pixels)
    if "vector_stitch" in reader.metadata and not coverage:
        from PIL import ImageDraw
        from shapely.geometry import shape
        preview = preview.convert("RGB")
        draw = ImageDraw.Draw(preview)
        vectors = reader.read_vectors(x, y)
        for key, color in (("features", (0, 255, 80)), ("conflicts", (255, 45, 45))):
            for feature in vectors[key]:
                geometry = shape(feature["geometry"])
                for polygon in ([geometry] if geometry.geom_type == "Polygon" else geometry.geoms):
                    for ring in (polygon.exterior, *polygon.interiors):
                        draw.line([(px-x, py-y) for px, py in ring.coords], fill=color, width=1)
    data = BytesIO()
    preview.save(data, format="PNG")
    header = {k: reader.metadata[k] for k in ("schema", "job_id", "plan_sha256", "render", "sampling", "overlap")}
    if "vector_stitch" in reader.metadata:
        header["vector_stitch"] = reader.metadata["vector_stitch"]
    return data.getvalue(), header, x, y, columns, rows
