"""Faster skeleton, point extraction, photo mask and point matching give what the old code gave."""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from karakal.core import mask_primitives
from karakal.core import source_mask_from_gray as source_mask
from karakal.core.image_io import _extract_patch, build_prediction_view_from_gray
from karakal.core.mask_metrics import _match_point_sets, _point_match_threshold
from karakal.core.mask_primitives import _boundary_mask, _label_components, skeletonize
from karakal.core.repository_shared import EPS, _PredictionPoint
from karakal.core.synthetic_chip import LayerStyle, PhotoStyle
from karakal.core.synthetic_validation import make_case

LAYERS = (LayerStyle(), LayerStyle(vias=True), LayerStyle(diagonals=True, mixed_widths=True))


def _dense_skeleton(mask: np.ndarray) -> np.ndarray:
    image = np.asarray(mask, dtype=np.uint8).copy()
    mask_primitives._skeletonize_binary_dense(image)
    return image.astype(bool)


@pytest.mark.parametrize("case", ["clean", "open", "bridge", "boundary_jitter", "extra_blob", "edge_shift"])
def test_boundary_thinning_equals_reading_every_pixel(case: str) -> None:
    for layer in LAYERS:
        try:
            frame = make_case(case, 3, layer)
        except ValueError:
            continue
        mask = frame.masks["a"] > 0
        assert np.array_equal(skeletonize(mask), _dense_skeleton(mask))


def test_boundary_thinning_on_random_shapes() -> None:
    rng = np.random.default_rng(0)
    for _index in range(25):
        noise = cv2.GaussianBlur(rng.normal(size=(96, 128)).astype(np.float32), (0, 0), float(rng.uniform(1.0, 4.0)))
        mask = noise > float(rng.uniform(-0.2, 0.3))
        assert np.array_equal(skeletonize(mask), _dense_skeleton(mask))
    assert not skeletonize(np.zeros((5, 5), bool)).any()
    assert skeletonize(np.ones((1, 1), bool)).all()


def _reference_view_points(gray: np.ndarray, threshold: int = 128) -> list[tuple[float, ...]]:
    """The old point extraction: one look at the whole frame per point."""

    pred_bin = gray >= threshold
    labels, count = _label_components(pred_bin)
    rows = []
    for label in range(1, count + 1):
        ys, xs = np.where(labels == label)
        area = int(xs.size)
        cx, cy = float(np.mean(xs, dtype=np.float64)), float(np.mean(ys, dtype=np.float64))
        px = int(np.clip(round(cx), 0, gray.shape[1] - 1))
        py = int(np.clip(round(cy), 0, gray.shape[0] - 1))
        patch = _extract_patch(gray, cx, cy)
        peak = float(gray[py, px] / 255.0)
        patch_mean = float(patch.mean(dtype=np.float32) / 255.0)
        patch_std = float(patch.std(dtype=np.float32) / 255.0)
        contrast = float(max(0.0, peak - patch_mean))
        perimeter = float(max(1, np.count_nonzero(_boundary_mask(labels == label))))
        compactness = float(np.clip((4.0 * np.pi * area) / max(EPS, perimeter * perimeter), 0.0, 1.0))
        rows.append((cx, cy, peak, contrast, compactness, contrast / max(1e-6, patch_std), float(np.sqrt(area / np.pi)), float(area)))
    return rows


@pytest.mark.parametrize("case", ["clean", "merged_points", "point_jitter"])
def test_point_extraction_per_box_equals_the_whole_frame_way(case: str) -> None:
    frame = make_case(case, 5, LayerStyle(vias=True))
    gray = frame.masks["a"]
    view = build_prediction_view_from_gray("a", gray)
    expected = _reference_view_points(gray)
    got = [(p.x, p.y, p.score, p.local_contrast, p.blob_score, p.local_snr, p.radius, p.spot_area) for p in view.points]
    assert len(got) == len(expected) > 50
    assert np.allclose(np.asarray(got), np.asarray(expected), rtol=0.0, atol=1e-9)


def _reference_source_mask(original: np.ndarray) -> np.ndarray:
    settings = source_mask.SourceMaskConfig()
    gray = source_mask._prepare_gray(original, target_shape=None)
    smooth = source_mask._smooth(source_mask._robust_normalize(gray), settings.smooth_sigma)
    wall = source_mask._gradient_barriers(smooth, edge_percentile=settings.edge_percentile)
    wall = source_mask._binary_dilate(wall, radius=settings.barrier_dilate_px)
    interior = ~wall
    labels, count = source_mask._label_components(interior)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    keep = np.zeros(count + 1, dtype=bool)
    mean = float(np.mean(smooth[interior], dtype=np.float64))
    for label in range(1, count + 1):
        if areas[label] >= settings.min_region_area and float(np.mean(smooth[labels == label], dtype=np.float64)) >= mean:
            keep[label] = True
    keep[0] = False
    return keep[labels]


@pytest.mark.parametrize(
    "photo", [PhotoStyle(), PhotoStyle(light_gradient=0.3, grain=5.0, edge_roughness=0.6, noise=5.0), PhotoStyle(metal_level=60.0, substrate_level=190.0)]
)
def test_photo_mask_by_region_means_equals_the_region_loop(photo: PhotoStyle) -> None:
    for layer in LAYERS:
        for case in ("clean", "residue", "particles", "hole"):
            frame = make_case(case, 2, layer, photo)
            assert np.array_equal(source_mask.build_source_mask(frame.photo), _reference_source_mask(frame.photo))


def _points(*coordinates: tuple[float, float], radius: float = 0.0) -> tuple[_PredictionPoint, ...]:
    return tuple(_PredictionPoint(x, y, 1.0, 255.0, 0.0, 1.0, 1.0, radius, 1.0) for x, y in coordinates)


def test_optimal_matching_keeps_a_pair_greedy_matching_loses() -> None:
    first = _points((0.0, 0.0), (2.0, 0.0))
    second = _points((1.0, 0.0), (3.1, 0.0))
    distances, matched_a, matched_b = _match_point_sets(first, second, 1.5)
    assert matched_a == {0, 1} and matched_b == {0, 1}
    assert sorted(distances) == pytest.approx([1.0, 1.1])


def test_bigger_points_match_farther() -> None:
    small = _points((0.0, 0.0), radius=1.0)
    big = _points((0.0, 0.0), radius=6.0)
    assert _point_match_threshold(small[0], small[0], 3.0) == 3.0
    assert _point_match_threshold(big[0], big[0], 3.0) == 6.0
    moved = _points((5.0, 0.0), radius=6.0)
    assert len(_match_point_sets(big, moved, 3.0)[0]) == 1
    assert len(_match_point_sets(small, _points((5.0, 0.0), radius=1.0), 3.0)[0]) == 0


def test_matching_is_one_to_one_and_symmetric() -> None:
    rng = np.random.default_rng(1)
    first = _points(*map(tuple, rng.uniform(0, 100, size=(60, 2))))
    second = _points(*map(tuple, rng.uniform(0, 100, size=(50, 2))))
    distances, matched_a, matched_b = _match_point_sets(first, second, 4.0)
    assert len(distances) == len(matched_a) == len(matched_b)
    back, back_a, back_b = _match_point_sets(second, first, 4.0)
    assert len(back) == len(distances) and sum(back) == pytest.approx(sum(distances))
    assert _match_point_sets((), second, 4.0) == ([], set(), set())
