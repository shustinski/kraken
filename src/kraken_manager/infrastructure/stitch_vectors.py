"""Spatially partitioned vector union and explicit overlap disagreement.

All coordinates are reference pixel centres, using the same measured physical
transforms as the raster renderer. Holes, categories and source provenance are
preserved. Neither polygon union nor seam agreement proves electrical contact.
"""
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import sqlite3

from kraken_manager.domain.common import DomainValidationError
from .stitching import StitchCancelled, pixel_transform


@dataclass(frozen=True, slots=True)
class VectorLimits:
    max_features: int = 1000000
    max_vertices: int = 10000000
    max_block_features: int = 10000
    max_block_vertices: int = 1000000
    max_payload_bytes: int = 8 * 1024**2
    max_overlap_pairs: int = 10000

    def __post_init__(self):
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (getattr(self, name) for name in self.__dataclass_fields__)):
            raise DomainValidationError("vector budgets must be positive integers")


def _vertices(geometry):
    if geometry.is_empty:
        return 0
    if geometry.geom_type == "Polygon":
        return len(geometry.exterior.coords) + sum(len(r.coords) for r in geometry.interiors)
    return sum(_vertices(g) for g in geometry.geoms)


def _polygonal(geometry):
    """Clipping can produce zero-area lines at partition boundaries."""
    from shapely import MultiPolygon
    if geometry.is_empty:
        return MultiPolygon([])
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    from shapely import union_all
    return union_all([g for g in geometry.geoms if g.geom_type in {"Polygon", "MultiPolygon"}])


def _valid(geometry, limit):
    if geometry.geom_type not in {"Polygon", "MultiPolygon"} or not geometry.is_valid:
        raise DomainValidationError("stitch vectors require valid polygon topology, without automatic repair")
    if not geometry.is_empty and (not all(math.isfinite(v) and abs(v) <= 2**40 for v in geometry.bounds) or _vertices(geometry) > limit):
        raise DomainValidationError("stitch vector geometry exceeds its bounded coordinate or vertex limit")


class VectorStitchIndex:
    """Disposable disk index; only bounded query results enter GEOS memory."""
    def __init__(self, path, renderer, *, limits=None, cancelled=lambda: False):
        from kraken_core.safe_files import open_exclusive_write
        self.path, self.renderer = path, renderer
        self.limits, self.cancelled = limits or VectorLimits(), cancelled
        with open_exclusive_write(path):
            pass
        self.db = sqlite3.connect(path)
        self.features = self.vertices = 0
        self.sources = {}
        try:
            self.db.executescript("PRAGMA temp_store=FILE; PRAGMA cache_size=-8192;"
                "CREATE TABLE features(id INTEGER PRIMARY KEY, frame TEXT, category TEXT, wkb BLOB);"
                "CREATE VIRTUAL TABLE spatial USING rtree(id,left,right,top,bottom);"
                "CREATE TABLE footprints(id INTEGER PRIMARY KEY, frame TEXT, wkb BLOB);"
                "CREATE VIRTUAL TABLE tiles USING rtree(id,left,right,top,bottom);")
            reference = next(t for t in renderer.plan.tiles if t.frame_id == renderer.plan.reference_frame_id)
            from shapely import Polygon
            from shapely.affinity import affine_transform
            for index, tile in enumerate(renderer.plan.tiles, 1):
                matrix = pixel_transform(tile, reference)
                transform = [matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]]
                footprint = affine_transform(Polygon([(-.5, -.5), (tile.width-.5, -.5),
                    (tile.width-.5, tile.height-.5), (-.5, tile.height-.5)]), transform)
                left, top, right, bottom = footprint.bounds
                self.db.execute("INSERT INTO footprints VALUES(?,?,?)", (index, tile.frame_id, footprint.wkb))
                self.db.execute("INSERT INTO tiles VALUES(?,?,?,?,?)", (index, left, right, top, bottom))
            self.db.commit()
        except BaseException:
            self.close()
            raise

    def close(self):
        self.db.close()
        self.path.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def check_cancel(self):
        if self.cancelled():
            raise StitchCancelled("vector stitching cancelled")

    def add(self, tile, source, geometries):
        """geometries yields (category, polygon) in the tile's image pixel grid."""
        from shapely.affinity import affine_transform
        if tile.frame_id in self.sources:
            raise DomainValidationError("duplicate vector source for a stitch frame")
        self.sources[tile.frame_id] = source
        reference = next(t for t in self.renderer.plan.tiles if t.frame_id == self.renderer.plan.reference_frame_id)
        matrix = pixel_transform(tile, reference)
        transform = [matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]]
        for category, polygon in geometries:
            self.check_cancel()
            if not isinstance(category, str) or not 0 < len(category) <= 64:
                raise DomainValidationError("invalid stitched vector category")
            _valid(polygon, self.limits.max_block_vertices)
            if polygon.is_empty:
                continue
            left, top, right, bottom = polygon.bounds
            if left < -.5 or top < -.5 or right > tile.width-.5 or bottom > tile.height-.5:
                raise DomainValidationError("vector source extends outside its calibrated image footprint")
            self.features += 1
            self.vertices += _vertices(polygon)
            if self.features > self.limits.max_features or self.vertices > self.limits.max_vertices:
                raise DomainValidationError("vector stitch source exceeds the configured job budget")
            projected = affine_transform(polygon, transform)
            _valid(projected, self.limits.max_block_vertices)
            left, top, right, bottom = projected.bounds
            self.db.execute("INSERT INTO features VALUES(?,?,?,?)", (self.features, tile.frame_id, category, projected.wkb))
            self.db.execute("INSERT INTO spatial VALUES(?,?,?,?,?)", (self.features, left, right, top, bottom))
        self.db.commit()

    def block(self, x, y, width, height):
        from shapely import box, from_wkb, union_all
        from shapely.geometry import mapping
        self.check_cancel()
        if set(self.sources) != {t.frame_id for t in self.renderer.plan.tiles}:
            raise DomainValidationError("vector content must cover every geometry tile, including empty frames")
        # Partition edges lie half a pixel outside the raster block centres.
        bounds = (x-.5, y-.5, x+width-.5, y+height-.5)
        region = box(*bounds)
        arguments = (bounds[2], bounds[0], bounds[3], bounds[1])
        query = "SELECT f.frame,f.category,f.wkb FROM features f JOIN spatial s ON s.id=f.id WHERE s.left<=? AND s.right>=? AND s.top<=? AND s.bottom>=? ORDER BY f.id LIMIT ?"
        rows = self.db.execute(query, (*arguments, self.limits.max_block_features+1)).fetchall()
        if len(rows) > self.limits.max_block_features:
            raise DomainValidationError("vector block exceeds the configured feature budget")
        by_frame = defaultdict(lambda: defaultdict(list))
        count = 0
        for frame, category, data in rows:
            self.check_cancel()
            geometry = from_wkb(data)
            count += _vertices(geometry)
            if count > self.limits.max_block_vertices:
                raise DomainValidationError("vector block exceeds the configured vertex budget")
            clipped = _polygonal(geometry.intersection(region))
            if not clipped.is_empty:
                by_frame[frame][category].append(clipped)
        merged = {frame: {category: union_all(parts) for category, parts in categories.items()} for frame, categories in by_frame.items()}
        categories = sorted({category for value in merged.values() for category in value})
        features = []
        output_vertices = 0
        for category in categories:
            self.check_cancel()
            frames = sorted(frame for frame in merged if category in merged[frame])
            geometry = union_all([merged[frame][category] for frame in frames])
            _valid(geometry, self.limits.max_block_vertices)
            output_vertices += _vertices(geometry)
            if output_vertices > self.limits.max_block_vertices:
                raise DomainValidationError("vector block output exceeds its vertex budget")
            features.append({"category": category, "geometry": mapping(geometry),
                "source_version_ids": [self.sources[frame]["artifact_version_id"] for frame in frames]})
        tile_rows = self.db.execute("SELECT f.frame,f.wkb FROM footprints f JOIN tiles t ON t.id=f.id WHERE t.left<=? AND t.right>=? AND t.top<=? AND t.bottom>=? ORDER BY f.frame LIMIT ?",
            (*arguments, self.limits.max_block_features+1)).fetchall()
        if len(tile_rows) > self.limits.max_block_features:
            raise DomainValidationError("vector block exceeds the overlap tile budget")
        footprints = [(frame, from_wkb(data).intersection(region)) for frame, data in tile_rows]
        if len(footprints) * (len(footprints)-1) // 2 > self.limits.max_overlap_pairs:
            raise DomainValidationError("vector block exceeds the pairwise comparison budget")
        conflicts = []
        for index, (a, footprint_a) in enumerate(footprints):
            for b, footprint_b in footprints[index+1:]:
                self.check_cancel()
                overlap = footprint_a.intersection(footprint_b)
                if overlap.area == 0:
                    continue
                for category in categories:
                    left = merged.get(a, {}).get(category, box(0, 0, 0, 0))
                    right = merged.get(b, {}).get(category, box(0, 0, 0, 0))
                    disagreement = _polygonal(left.symmetric_difference(right).intersection(overlap))
                    if not disagreement.is_empty:
                        _valid(disagreement, self.limits.max_block_vertices)
                        output_vertices += _vertices(disagreement)
                        if output_vertices > self.limits.max_block_vertices:
                            raise DomainValidationError("vector block output exceeds its vertex budget")
                        conflicts.append({"category": category, "geometry": mapping(disagreement), "frame_ids": [a, b],
                            "source_version_ids": [self.sources[f]["artifact_version_id"] for f in (a, b)]})
                        if len(conflicts) > self.limits.max_block_features:
                            raise DomainValidationError("vector block exceeds the disagreement budget")
        payload = {"features": features, "conflicts": conflicts}
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
        if len(encoded) > self.limits.max_payload_bytes:
            raise DomainValidationError("vector block exceeds the serialized payload budget")
        return encoded, hashlib.sha256(encoded).hexdigest()
