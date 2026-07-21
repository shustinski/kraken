import numpy as np
import pytest

pytest.importorskip('PIL')

from PIL import Image

from neuralimage.lib.image_processing import (
    cut_image,
    get_names_from_file,
    img_crop_border,
    reshape_imgs,
    sew_image,
    stitch_probability_map,
)
from neuralimage.lib.tiling import build_tile_plan
from tests.helpers import make_test_dir


def test_get_names_from_file():
    tmp_path = make_test_dir("image_processing")
    names_file = tmp_path / 'names.txt'
    names_file.write_text('a\nb\n', encoding='utf-8')

    assert list(get_names_from_file(names_file)) == ['a', 'b']


def test_reshape_imgs():
    imgs = np.arange(2 * 3 * 4).reshape(1, 2, 3, 4)
    reshaped = reshape_imgs(imgs)
    assert reshaped.shape == (1, 2, 3, 1)


def test_cut_and_sew_image_smoke():
    base = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    parts = cut_image(base, (1, 2, 2), overlap=0)
    assert parts.shape[1:] == (1, 2, 2)

    pred = np.ones((parts.shape[0], 1, 2, 2), dtype=np.float32)
    res = sew_image((4, 4), pred, overlap=0)
    assert res.size == (4, 4)


def test_cut_and_sew_image_non_square_segments():
    base = np.arange(1 * 6 * 4, dtype=np.float32).reshape(1, 6, 4)
    parts = cut_image(base, (1, 2, 3), overlap=0)
    assert parts.shape[1:] == (1, 3, 2)

    pred = np.ones((parts.shape[0], 1, 3, 2), dtype=np.float32)
    res = sew_image((4, 6), pred, overlap=0)
    arr = np.array(res)
    assert arr.shape == (6, 4)
    assert int(arr.min()) == 255
    assert int(arr.max()) == 255


def test_cut_image_small_source_with_large_segment():
    base = np.full((1, 64, 64), 255, dtype=np.float32)
    parts = cut_image(base, (1, 512, 512), overlap=0)

    assert parts.shape == (1, 1, 512, 512)
    assert np.all(parts[0, 0, :64, :64] == 1.0)
    assert np.all(parts[0, 0, 64:, :] == 0.0)
    assert np.all(parts[0, 0, :, 64:] == 0.0)


def test_sew_image_small_source_with_large_segment():
    pred = np.ones((1, 1, 512, 512), dtype=np.float32)
    res = sew_image((64, 64), pred, overlap=32)
    arr = np.array(res)

    assert arr.shape == (64, 64)
    assert int(arr.min()) == 255
    assert int(arr.max()) == 255


def test_sew_image_applies_threshold_and_postprocess():
    pred = np.array(
        [
            [
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ]
        ],
        dtype=np.float32,
    )

    raw = np.array(sew_image((3, 3), pred, overlap=0, threshold=0.5))
    processed = np.array(sew_image((3, 3), pred, overlap=0, threshold=0.5, postprocess_kernel_size=3))

    assert raw[1, 1] == 0
    assert processed[1, 1] == 255


def test_img_crop_border_crops_to_black_frame():
    img = Image.fromarray(np.full((5, 5), 255, dtype=np.uint8), mode='L')
    cropped = img_crop_border(1, img)
    arr = np.array(cropped)
    assert arr[0, 0] == 0
    assert arr[2, 2] == 255


@pytest.mark.parametrize(
    ('base_size', 'tile_size', 'overlap'),
    [
        ((128, 128), (64, 64), 1),
        ((101, 79), (32, 24), 7),
        ((40, 40), (16, 16), 15),
        ((17, 19), (64, 32), 3),
        ((128, 64), (64, 32), 0),
    ],
)
def test_tile_plan_roundtrip_has_complete_coverage(base_size, tile_size, overlap):
    width, height = base_size
    tile_width, tile_height = tile_size
    base = np.full((1, height, width), 255, dtype=np.float32)
    plan = build_tile_plan((height, width), (tile_width, tile_height), overlap)

    parts = cut_image(base, (1, tile_width, tile_height), overlap, tile_plan=plan)
    stitched = stitch_probability_map(base_size, parts, overlap, tile_plan=plan)

    assert plan.tile_count == len(set((window.left, window.top) for window in plan.windows))
    assert stitched.tile_count == plan.tile_count
    assert stitched.min_weight > 0.0
    assert np.isfinite(stitched.probabilities).all()
    assert np.allclose(stitched.probabilities, 1.0, atol=1e-6)


def test_tile_plan_roundtrip_preserves_coordinate_gradient():
    width, height = 101, 79
    x = np.linspace(0.0, 255.0, width, dtype=np.float32)
    y = np.linspace(0.0, 255.0, height, dtype=np.float32)[:, None]
    base = ((x[None, :] + y) * 0.5)[None, :, :]
    plan = build_tile_plan((height, width), (32, 24), overlap=7)

    parts = cut_image(base, (1, 32, 24), 7, tile_plan=plan)
    stitched = stitch_probability_map((width, height), parts, 7, tile_plan=plan)

    assert np.allclose(stitched.probabilities, base[0] / 255.0, atol=1e-6)


def test_stitch_probability_map_blends_conflicting_tiles_smoothly():
    plan = build_tile_plan((8, 12), (8, 8), overlap=4)
    predictions = np.zeros((2, 1, 8, 8), dtype=np.float32)
    predictions[1] = 1.0

    stitched = stitch_probability_map((12, 8), predictions, 4, tile_plan=plan).probabilities
    overlap_profile = stitched[4, 4:8]

    assert np.all(np.diff(overlap_profile) > 0.0)
    assert np.all(stitched[:, :4] == 0.0)
    assert np.all(stitched[:, 8:] == 1.0)


def test_stitch_probability_map_uses_actual_overlap_for_edge_aligned_tile():
    plan = build_tile_plan((4, 10), (4, 4), overlap=0)
    predictions = np.zeros((3, 1, 4, 4), dtype=np.float32)
    predictions[1] = 0.5
    predictions[2] = 1.0

    stitched = stitch_probability_map((10, 4), predictions, 0, tile_plan=plan)

    assert stitched.min_overlap == 2
    assert stitched.max_overlap == 2
    assert np.all(np.diff(stitched.probabilities[2, 5:9]) > 0.0)


def test_stitch_probability_map_rejects_prediction_count_mismatch():
    plan = build_tile_plan((32, 32), (16, 16), overlap=3)
    predictions = np.zeros((plan.tile_count - 1, 1, 16, 16), dtype=np.float32)

    with pytest.raises(ValueError, match='Prediction count'):
        stitch_probability_map((32, 32), predictions, 3, tile_plan=plan)


def test_tile_plan_rejects_invalid_overlap():
    with pytest.raises(ValueError, match='overlap must satisfy'):
        build_tile_plan((32, 32), (16, 16), overlap=16)

