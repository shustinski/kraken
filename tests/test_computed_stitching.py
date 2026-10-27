from dataclasses import asdict, replace
import json
from uuid import uuid4

import numpy as np
import pytest

from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.multilayer import ArtifactAnchor, CoordinateCalibration, MultilayerModel, PhysicalLayer
from kraken_manager.domain.project import LayerType
from kraken_manager.domain.stitch_plan import LayerStitchRevision, TilePlacement
from kraken_manager.infrastructure.stitching import StitchCancelled, StitchLimits, StitchRenderer, pixel_transform


def uid():
    return str(uuid4())


def plan(offset=(3, 0), *, size=(3, 2), affine=None):
    layer = uid()
    calibration = CoordinateCalibration(layer, 0, ArtifactAnchor(uid(), "a" * 64), (1, 1), y_axis="up")
    reference = TilePlacement(uid(), calibration, *size, "reference", (1, 0, 0, 0, 1, 0))
    other = TilePlacement(uid(), replace(calibration, artifact=ArtifactAnchor(uid(), "b" * 64)), *size,
        "declared_transform", affine or (1, 0, offset[0], 0, 1, offset[1]), note="controlled functional fixture")
    return LayerStitchRevision(uid(), 0, 0, layer, ArtifactAnchor(uid(), "c" * 64), reference.frame_id, (reference, other))


def assemble(renderer, arrays):
    left, top, right, bottom = renderer.bounds
    dtype = np.uint16 if renderer.mode == "labels" else np.uint8
    shape = (bottom - top, right - left, 3) if renderer.mode == "rgb" else (bottom - top, right - left)
    pixels, coverage = np.zeros(shape, dtype=dtype), np.zeros(shape[:2], dtype=np.uint32)
    for block in renderer.blocks(lambda tile: arrays[tile.frame_id]):
        h, w = block.coverage.shape
        y, x = block.y - top, block.x - left
        pixels[y:y+h, x:x+w] = block.pixels
        coverage[y:y+h, x:x+w] = block.coverage
    return pixels, coverage


@pytest.mark.parametrize("offset", [(3, 0), (-3, 0), (1, 0), (5, 3), (-5, -3)])
def test_translation_overlap_and_uncovered_holes_are_exact(offset):
    geometry = plan(offset)
    arrays = {p.frame_id: np.full((2, 3), v, dtype=np.uint8) for p, v in zip(geometry.tiles, (0, 100))}
    renderer = StitchRenderer(geometry, mode="gray", limits=StitchLimits(block_size=2))
    pixels, coverage = assemble(renderer, arrays)
    left, top, right, bottom = renderer.bounds
    expected, counts = np.zeros_like(pixels, dtype=float), np.zeros_like(coverage)
    for dx, dy, value in ((0, 0, 0), (*offset, 100)):
        expected[dy-top:dy-top+2, dx-left:dx-left+3] += value
        counts[dy-top:dy-top+2, dx-left:dx-left+3] += 1
    assert np.array_equal(coverage, counts)
    assert np.array_equal(pixels, np.rint(expected / np.maximum(counts, 1)))
    assert renderer.preflight()["width"] == right - left
    assert renderer.preflight()["height"] == bottom - top


@pytest.mark.parametrize("affine", [(0, -1, 4, 1, 0, -2), (1, .4, -.25, .2, 1, .5), (2, 0, -3, 0, .5, 1)])
def test_rotation_shear_and_subpixel_sampling_independent_of_block_boundaries(affine):
    geometry = plan(affine=affine)
    arrays = {p.frame_id: np.arange(18, dtype=np.uint8).reshape(2, 3, 3) for p in geometry.tiles}
    small = assemble(StitchRenderer(geometry, limits=StitchLimits(block_size=1)), arrays)
    large = assemble(StitchRenderer(geometry, limits=StitchLimits(block_size=20)), arrays)
    for actual, expected in zip(small, large):
        np.testing.assert_array_equal(actual, expected)
    assert small[1].max() >= 1


def test_labels_nearest_preserve_classes_and_overlap_is_stable_by_frame_id():
    geometry = plan((.5, 0))
    arrays = {p.frame_id: np.full((2, 3), value, np.uint16) for p, value in zip(geometry.tiles, (7, 1000))}
    renderer = StitchRenderer(geometry, mode="labels")
    pixels, coverage = assemble(renderer, arrays)
    assert set(np.unique(pixels)) <= {0, 7, 1000}
    first = min(geometry.tiles, key=lambda p: p.frame_id)
    assert np.all(pixels[coverage == 2] == arrays[first.frame_id][0, 0])
    reversed_result = assemble(StitchRenderer(replace(geometry, tiles=tuple(reversed(geometry.tiles))), mode="labels"), arrays)
    np.testing.assert_array_equal(pixels, reversed_result[0])


def test_anisotropic_calibration_and_y_orientation_are_composed_in_physical_units():
    geometry = plan()
    source, reference = geometry.tiles[1], geometry.tiles[0]
    source = replace(source, calibration=replace(source.calibration, pixel_size_m=(2, 3), origin_m=(4, 5), y_axis="down"))
    transform = pixel_transform(source, reference)
    np.testing.assert_allclose(transform @ [1, 2, 1], [9, -1, 1])


def test_snapshot_roundtrip_old_snapshot_and_no_fabricated_residuals():
    geometry = plan()
    model = MultilayerModel(uid(), layers=(PhysicalLayer(geometry.layer_id, LayerType.METAL, 0, 0),), stitches=(geometry,))
    assert MultilayerModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    old = model.to_dict()
    old["model"].pop("stitches")
    assert MultilayerModel.from_dict(old).stitches == ()
    assert geometry.tiles[1].fit is None
    assert LayerStitchRevision.from_dict(json.loads(json.dumps(asdict(geometry)))) == geometry


def test_preflight_limits_cancel_and_wrong_decode_fail_before_publication():
    geometry = plan((1_000_000, 1_000_000))
    for limits in (StitchLimits(max_canvas_pixels=100), StitchLimits(max_blocks=1), StitchLimits(max_source_pixels=2)):
        with pytest.raises(DomainValidationError, match="limit"):
            StitchRenderer(geometry, limits=limits)
    renderer = StitchRenderer(plan(), mode="gray")
    with pytest.raises(StitchCancelled):
        next(renderer.blocks(lambda _: pytest.fail("must cancel before decode"), cancelled=lambda: True))
    with pytest.raises(DomainValidationError, match="dimensions"):
        next(renderer.blocks(lambda _: np.zeros((1, 1), np.uint8)))


def test_unresolved_fit_and_duplicate_frame_never_render():
    from kraken_manager.domain.multilayer import RegistrationFit
    geometry = plan()
    unresolved = replace(geometry.tiles[1], provenance="control_points", affine=None,
                         fit=RegistrationFit("insufficient", "no observations"))
    geometry = replace(geometry, tiles=(geometry.tiles[0], unresolved))
    assert not geometry.ready
    with pytest.raises(DomainValidationError, match="unresolved"):
        StitchRenderer(geometry)
    with pytest.raises(DomainValidationError, match="unique"):
        replace(geometry, tiles=(geometry.tiles[0], geometry.tiles[0]))
