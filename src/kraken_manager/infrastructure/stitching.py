"""Bounded block renderer for explicit intra-layer poses, without a dense canvas.

Pixel centres are integers; the reference calibration defines the output grid.
RGB/gray uses bilinear sampling and equal-weight overlap. Labels use nearest
sampling and the first frame in stable ID order. Coverage is separate from data.
This module computes blocks only; it does not publish or approve a layer.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import hashlib
from math import ceil

from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.stitch_plan import LayerStitchRevision


class StitchCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StitchLimits:
    block_size: int = 256
    max_source_pixels: int = 16_777_216
    max_canvas_pixels: int = 1_000_000_000_000
    max_blocks: int = 1_000_000
    max_tiles: int = 100_000

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DomainValidationError("stitch resource limits must be positive integers")
        if self.block_size > 1024:
            raise DomainValidationError("stitch blocks exceed allocation limit")


@dataclass(frozen=True, slots=True)
class StitchBlock:
    x: int
    y: int
    pixels: object
    coverage: object


class VerifiedTileSources:
    """Decode the same open file that passed SHA validation, with explicit limits."""
    def __init__(self, paths, *, mode="rgb", limits=None, cancelled=lambda: False):
        self.paths, self.mode, self.limits, self.cancelled = paths, mode, limits or StitchLimits(), cancelled

    def __call__(self, tile):
        import numpy as np
        from PIL import Image
        from kraken_core.safe_files import open_regular_read

        if tile.width * tile.height > self.limits.max_source_pixels:
            raise DomainValidationError("stitch source exceeds decoded pixel limit")
        with open_regular_read(self.paths[tile.calibration.artifact.version_id]) as stream:
            digest = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                if self.cancelled():
                    raise StitchCancelled("stitch computation cancelled")
                digest.update(chunk)
            if digest.hexdigest() != tile.calibration.artifact.sha256:
                raise DomainValidationError("stitch source SHA-256 mismatch")
            stream.seek(0)
            with Image.open(stream) as image:
                if image.size != (tile.width, tile.height):
                    raise DomainValidationError("decoded stitch source dimensions mismatch")
                if self.mode == "rgb":
                    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
                if self.mode == "gray":
                    if image.mode != "L":
                        raise DomainValidationError("gray stitching requires an explicit 8-bit grayscale image")
                    return np.asarray(image, dtype=np.uint8).copy()
                if self.mode == "labels":
                    if image.mode not in {"L", "I;16", "I;16L", "I;16B", "I"}:
                        raise DomainValidationError("label stitching requires integer labels, without color conversion")
                    labels = np.asarray(image)
                    if labels.min() < 0 or labels.max() > 65535:
                        raise DomainValidationError("label values exceed uint16 range")
                    return labels.astype(np.uint16)
                raise DomainValidationError("unsupported stitch decode mode")


def pixel_transform(tile, reference):
    import numpy as np

    def calibration(value):
        sx, sy = value.pixel_size_m
        ox, oy = value.origin_m
        return np.array([[sx, 0, ox], [0, sy * (1 if value.y_axis == "up" else -1), oy], [0, 0, 1]])
    a, b, tx, c, d, ty = tile.affine
    return np.linalg.solve(calibration(reference.calibration),
        np.array([[a, b, tx], [c, d, ty], [0, 0, 1]]) @ calibration(tile.calibration))


class StitchRenderer:
    def __init__(self, plan: LayerStitchRevision, *, mode="rgb", limits=None):
        import numpy as np

        if not plan.ready:
            raise DomainValidationError("unresolved tile placement cannot render a mosaic")
        if mode not in {"rgb", "gray", "labels"}:
            raise DomainValidationError("stitch mode must be rgb, gray or labels")
        self.plan, self.mode, self.limits = plan, mode, limits or StitchLimits()
        if len(plan.tiles) > self.limits.max_tiles:
            raise DomainValidationError("stitch plan exceeds tile limit")
        reference = next(p for p in plan.tiles if p.frame_id == plan.reference_frame_id)
        footprints = []
        for tile in plan.tiles:
            if tile.width * tile.height > self.limits.max_source_pixels:
                raise DomainValidationError("stitch source exceeds decoded pixel limit")
            transform = pixel_transform(tile, reference)
            if not np.isfinite(transform).all() or np.linalg.cond(transform[:2, :2]) > 1e6:
                raise DomainValidationError("stitch transform is nonfinite or ill conditioned")
            corners = np.array([[-.5, -.5, 1], [tile.width - .5, -.5, 1],
                                [-.5, tile.height - .5, 1], [tile.width - .5, tile.height - .5, 1]])
            points = corners @ transform.T
            left, top = points[:, :2].min(axis=0)
            right, bottom = points[:, :2].max(axis=0)
            if max(abs(v) for v in (left, top, right, bottom)) > 2**40:
                raise DomainValidationError("stitch coordinates exceed numeric range")
            footprints.append((float(left), float(top), float(right), float(bottom), tile, np.linalg.inv(transform)))
        self._footprints = sorted(footprints, key=lambda f: (f[0], f[4].frame_id))
        self._lefts = [f[0] for f in self._footprints]
        self.bounds = (ceil(min(f[0] for f in footprints)), ceil(min(f[1] for f in footprints)),
                       ceil(max(f[2] for f in footprints)), ceil(max(f[3] for f in footprints)))
        left, top, right, bottom = self.bounds
        width, height = right - left, bottom - top
        block = self.limits.block_size
        count = ((width + block - 1) // block) * ((height + block - 1) // block)
        if width <= 0 or height <= 0 or width * height > self.limits.max_canvas_pixels or count > self.limits.max_blocks:
            raise DomainValidationError("stitch canvas exceeds configured limits")
        self.block_count = count

    def preflight(self):
        left, top, right, bottom = self.bounds
        # Encoded PNG size is content dependent; this is a raw output estimate.
        return {"bounds": list(self.bounds), "width": right - left, "height": bottom - top,
                "block_count": self.block_count, "block_size": self.limits.block_size,
                "raw_output_bytes": (right - left) * (bottom - top) * (7 if self.mode == "rgb" else 6),
                "mode": self.mode, "physical_accuracy": "not_established",
                "bound_scope": "supplied_control_points_only"}

    def blocks(self, load_source, *, cancelled=lambda: False):
        """Load one bounded source at a time. Caller verifies its immutable anchor.

        The callback receives TilePlacement and must return uint8 RGB/gray or
        uint16 labels. No result is a durable publication until its caller has
        committed every block, coverage and manifest in the existing job flow.
        """
        import numpy as np

        def check_cancel():
            if cancelled():
                raise StitchCancelled("stitch computation cancelled")

        left, top, right, bottom = self.bounds
        step = self.limits.block_size
        for y in range(top, bottom, step):
            for x in range(left, right, step):
                check_cancel()
                height, width = min(step, bottom - y), min(step, right - x)
                shape = (height, width, 3) if self.mode == "rgb" else (height, width)
                values = np.zeros(shape, dtype=np.float64 if self.mode != "labels" else np.uint16)
                coverage = np.zeros((height, width), dtype=np.uint32)
                yy, xx = np.mgrid[y:y + height, x:x + width]
                stop = bisect_left(self._lefts, x + width)
                candidates = (f for f in self._footprints[:stop] if f[2] > x and f[1] < y + height and f[3] > y)
                for _, _, _, _, tile, inverse in sorted(candidates, key=lambda f: f[4].frame_id):
                    check_cancel()
                    sx = inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2]
                    sy = inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2]
                    inside = (sx >= -.5) & (sx < tile.width - .5) & (sy >= -.5) & (sy < tile.height - .5)
                    if not inside.any():
                        continue
                    source = load_source(tile)
                    expected = (tile.height, tile.width, 3) if self.mode == "rgb" else (tile.height, tile.width)
                    dtype = np.uint16 if self.mode == "labels" else np.uint8
                    if not isinstance(source, np.ndarray) or source.shape != expected or source.dtype != dtype:
                        raise DomainValidationError("decoded stitch source dimensions, dtype or mode mismatch")
                    check_cancel()
                    px, py = np.clip(sx[inside], 0, tile.width - 1), np.clip(sy[inside], 0, tile.height - 1)
                    if self.mode == "labels":
                        samples = source[np.floor(py + .5).astype(int), np.floor(px + .5).astype(int)]
                        prior = coverage[inside]
                        values[inside] = np.where(prior == 0, samples, values[inside])
                    else:
                        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
                        x1, y1 = np.minimum(x0 + 1, tile.width - 1), np.minimum(y0 + 1, tile.height - 1)
                        wx, wy = px - x0, py - y0
                        if self.mode == "rgb":
                            wx, wy = wx[:, None], wy[:, None]
                        samples = ((1 - wx) * (1 - wy) * source[y0, x0] + wx * (1 - wy) * source[y0, x1]
                                   + (1 - wx) * wy * source[y1, x0] + wx * wy * source[y1, x1])
                        values[inside] += samples
                    coverage[inside] += 1
                    del source
                if self.mode != "labels":
                    divisor = np.maximum(coverage, 1)
                    if self.mode == "rgb":
                        divisor = divisor[:, :, None]
                    values = np.clip(np.rint(values / divisor), 0, 255).astype(np.uint8)
                check_cancel()
                yield StitchBlock(x, y, values, coverage)
