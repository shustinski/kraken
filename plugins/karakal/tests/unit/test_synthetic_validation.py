"""The synthetic validation bench: network outputs with planted errors over synthetic chip photos."""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from karakal.core.synthetic_chip import LayerStyle, PhotoStyle
from karakal.core.synthetic_validation import (
    BENIGN_KINDS,
    CONDUCTOR_ERROR_CODES,
    DATA_DEFECT_KINDS,
    DIAG_BAD_DATA,
    DIAG_CONFIDENT_ERROR,
    DIAG_MODEL_ERROR,
    DIAG_NOT_ALIGNED,
    DIAG_OK,
    DIAG_UNCERTAIN,
    POINT_ERROR_CODES,
    ConfidenceStyle,
    ModelError,
    ModelPlan,
    cases_for,
    make_case,
    make_validation_frame,
)

VIAS = LayerStyle(vias=True)
STYLES = {
    "manhattan": (LayerStyle(), PhotoStyle()),
    "diagonals": (LayerStyle(diagonals=True), PhotoStyle()),
    "mixed_widths": (LayerStyle(mixed_widths=True), PhotoStyle()),
    "dark_metal": (LayerStyle(), PhotoStyle(metal_level=60.0, substrate_level=190.0)),
    "vias": (VIAS, PhotoStyle()),
}


def _count(mask: np.ndarray) -> int:
    return int(cv2.connectedComponents((np.asarray(mask) > 0).astype(np.uint8), connectivity=8)[0]) - 1


def _model(frame, model_id: str = "a") -> np.ndarray:
    return (frame.masks[model_id] > 0).astype(np.uint8)


def _inside(box, points_y: np.ndarray, points_x: np.ndarray) -> bool:
    x, y, w, h = box
    return bool(np.all((points_x >= x) & (points_x < x + w) & (points_y >= y) & (points_y < y + h)))


@pytest.mark.parametrize("style", sorted(STYLES))
def test_every_case_of_every_style_builds(style: str) -> None:
    layer, photo = STYLES[style]
    for case in cases_for(layer):
        for seed in (0, 1):
            frame = make_case(case, seed, layer, photo)
            assert frame.photo.dtype == np.uint8 and frame.photo.shape == frame.truth.shape
            assert set(frame.masks) == {"a", "b"} == set(frame.confidences)
            assert frame.case == case


def test_same_seed_same_frame_other_seed_other_frame() -> None:
    first = make_case("open", 3)
    again = make_case("open", 3)
    other = make_case("open", 4)
    assert np.array_equal(first.photo, again.photo)
    assert np.array_equal(first.masks["a"], again.masks["a"])
    assert np.array_equal(first.confidences["a"], again.confidences["a"])
    assert first.items == again.items
    assert not np.array_equal(first.masks["a"], other.masks["a"])


def test_clean_frame_masks_are_the_truth_and_nothing_is_planted() -> None:
    frame = make_case("clean", 0)
    assert frame.items == ()
    for model_id in ("a", "b"):
        assert np.array_equal(_model(frame, model_id), frame.truth)


@pytest.mark.parametrize("kind", sorted(CONDUCTOR_ERROR_CODES))
def test_conductor_error_stays_in_its_box_of_model_a_only(kind: str) -> None:
    for seed in range(4):
        frame = make_case(kind, seed)
        (item,) = frame.items
        assert item.kind == kind and item.diagnosis == DIAG_MODEL_ERROR
        assert item.models == ("a",)
        assert item.codes == CONDUCTOR_ERROR_CODES[kind]
        assert np.array_equal(_model(frame, "b"), frame.truth)
        ys, xs = np.nonzero(_model(frame) != frame.truth)
        assert ys.size > 0
        assert _inside(item.box, ys, xs)


@pytest.mark.parametrize(
    ("kind", "delta"),
    [("open", 1), ("bridge", -1), ("missing_conductor", -1), ("extra_blob", 1), ("neck", 0), ("local_edge_shift", 0)],
)
def test_conductor_errors_change_the_object_count_as_planned(kind: str, delta: int) -> None:
    for seed in range(4):
        frame = make_case(kind, seed)
        assert _count(_model(frame)) - _count(frame.truth) == delta


@pytest.mark.parametrize(
    ("kind", "delta"),
    [("missing_point", -1), ("extra_point", 1), ("merged_points", -1), ("shifted_point", 0)],
)
def test_point_errors_change_the_point_count_as_planned(kind: str, delta: int) -> None:
    for seed in range(4):
        frame = make_case(kind, seed, VIAS)
        (item,) = frame.items
        assert item.codes == POINT_ERROR_CODES[kind]
        assert _count(_model(frame)) - _count(frame.truth) == delta
        ys, xs = np.nonzero(_model(frame) != frame.truth)
        assert _inside(item.box, ys, xs)


def test_shifted_point_moves_farther_than_the_match_radius() -> None:
    frame = make_validation_frame(seed=2, layer=VIAS, models=(ModelPlan("a", (ModelError("shifted_point", shift=9),)),))
    (item,) = frame.items
    assert item.param("shift_px") == 9.0
    moved = _model(frame) != frame.truth
    assert np.count_nonzero(moved & (frame.truth > 0)) > 0 and np.count_nonzero(moved & (frame.truth == 0)) > 0


@pytest.mark.parametrize("shift", [-2, -1, 1, 2])
def test_edge_shift_grows_or_shrinks_the_metal_without_changing_objects(shift: int) -> None:
    frame = make_validation_frame(seed=1, models=(ModelPlan("a", (ModelError("edge_shift", shift=shift),)),))
    (item,) = frame.items
    assert item.box is None and item.param("shift_px") == float(shift)
    area, truth_area = int(_model(frame).sum()), int(frame.truth.sum())
    assert (area > truth_area) if shift > 0 else (area < truth_area)
    assert _count(_model(frame)) == _count(frame.truth)


@pytest.mark.parametrize("kind", sorted(BENIGN_KINDS))
def test_harmless_jitter_keeps_every_object(kind: str) -> None:
    layer = VIAS if BENIGN_KINDS[kind] == "point" else LayerStyle()
    for seed in range(6):
        frame = make_case(kind, seed, layer)
        (item,) = frame.items
        assert item.diagnosis == DIAG_OK and not item.is_positive and item.codes == ()
        assert _count(_model(frame)) == _count(frame.truth)
        assert np.count_nonzero(_model(frame) != frame.truth) > 0
        # Never farther than one pixel from the truth.
        near = cv2.dilate(frame.truth, np.ones((3, 3), np.uint8))
        assert not np.any((_model(frame) > 0) & (near == 0))


@pytest.mark.parametrize("kind", DATA_DEFECT_KINDS)
def test_data_defect_spoils_only_the_photo(kind: str) -> None:
    frame = make_case(kind, 0)
    clean = make_case("clean", 0)
    assert all(item.diagnosis == DIAG_BAD_DATA and item.models == () for item in frame.items)
    assert np.array_equal(_model(frame), frame.truth)
    assert not np.array_equal(frame.photo, clean.photo)
    for item in frame.items:
        if item.box is None:
            continue
        x, y, w, h = item.box
        assert not np.array_equal(frame.photo[y : y + h, x : x + w], clean.photo[y : y + h, x : x + w])


def test_a_model_that_follows_the_photo_errs_only_inside_the_defect() -> None:
    frame = make_case("residue", 0, reaction="follows")
    (item,) = frame.items
    assert item.param("reaction") == "follows"
    ys, xs = np.nonzero(_model(frame) != frame.truth)
    assert ys.size > 0 and _inside(item.box, ys, xs)


def test_misaligned_photo_is_shifted_against_the_masks() -> None:
    frame = make_case("misaligned", 0)
    (item,) = frame.items
    assert item.diagnosis == DIAG_NOT_ALIGNED and item.box is None
    dx, dy = float(item.param("dx")), float(item.param("dy"))
    assert np.hypot(dx, dy) >= 3.0
    (found_dx, found_dy), _response = cv2.phaseCorrelate(
        frame.truth.astype(np.float32), frame.photo.astype(np.float32)
    )
    assert abs(found_dx - dx) <= 1.0 and abs(found_dy - dy) <= 1.0
    assert np.array_equal(_model(frame), frame.truth)


def _uncertainty(frame, model_id: str = "a") -> np.ndarray:
    probability = frame.confidences[model_id].astype(np.float32) / 255.0
    return 1.0 - np.abs(2.0 * probability - 1.0)


def _error_and_flat(frame) -> tuple[np.ndarray, np.ndarray]:
    wrong = (_model(frame) != frame.truth).astype(np.uint8)
    edges = cv2.morphologyEx(frame.truth, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    near = cv2.dilate(wrong | edges, np.ones((9, 9), np.uint8)) > 0
    return wrong > 0, ~near


def test_calibrated_model_is_unsure_where_it_is_wrong() -> None:
    frame = make_case("missing_conductor", 0)
    wrong, flat = _error_and_flat(frame)
    uncertainty = _uncertainty(frame)
    assert float(uncertainty[wrong].mean()) > 0.6
    assert float(uncertainty[flat].mean()) < 0.1


def test_overconfident_model_is_sure_of_its_errors() -> None:
    frame = make_case("missing_conductor", 0, behaviour="overconfident")
    (item,) = frame.items
    assert item.diagnosis == DIAG_CONFIDENT_ERROR
    wrong, _flat = _error_and_flat(frame)
    assert float(_uncertainty(frame)[wrong].mean()) < 0.1


def test_underconfident_model_is_never_sure() -> None:
    frame = make_case("clean", 0, behaviour="underconfident")
    probability = frame.confidences["a"].astype(np.float32) / 255.0
    assert float(np.abs(2.0 * probability - 1.0).max()) < 0.5


def test_calibrated_model_is_unsure_over_a_photo_defect() -> None:
    frame = make_case("residue", 0)
    (item,) = frame.items
    x, y, w, h = item.box
    uncertainty = _uncertainty(frame)
    _wrong, flat = _error_and_flat(make_case("clean", 0))
    assert float(uncertainty[y : y + h, x : x + w].mean()) > 0.6
    outside = flat.copy()
    outside[y : y + h, x : x + w] = False
    assert float(uncertainty[outside].mean()) < 0.1


def test_flat_noise_plants_unsure_places_where_the_model_is_right() -> None:
    frame = make_case("uncertain_flat", 0)
    flat_items = [item for item in frame.items if item.kind == "uncertain_flat"]
    assert flat_items and all(item.diagnosis == DIAG_UNCERTAIN and item.models == ("a",) for item in flat_items)
    assert np.array_equal(_model(frame), frame.truth)
    uncertainty = _uncertainty(frame)
    for item in flat_items:
        x, y, w, h = item.box
        assert float(uncertainty[y : y + h, x : x + w].max()) > 0.6
    # Model b keeps the calibrated behaviour: nothing planted on it.
    assert float(_uncertainty(frame, "b")[_error_and_flat(frame)[1]].mean()) < 0.1


def test_confidence_encoding_is_the_distance_from_one_half() -> None:
    probability = make_case("open", 0, encoding="probability").confidences["a"].astype(np.float32) / 255.0
    confidence = make_case("open", 0, encoding="confidence").confidences["a"].astype(np.float32) / 255.0
    assert np.allclose(confidence, np.abs(2.0 * probability - 1.0), atol=2.5 / 255.0)
    # Background away from the edges is sure in the confidence encoding (white), not in the probability
    # one (black).
    truth = make_case("clean", 0).truth
    background = cv2.dilate(truth, np.ones((13, 13), np.uint8)) == 0
    assert float(confidence[background].mean()) > 0.9 > 0.1 > float(probability[background].mean())


def test_shared_errors_make_one_item_for_both_models() -> None:
    frame = make_case("open", 0, shared=True)
    (item,) = frame.items
    assert item.models == ("a", "b")
    assert np.array_equal(frame.masks["a"], frame.masks["b"])
    assert not np.array_equal(_model(frame), frame.truth)


def test_independent_models_make_their_own_errors() -> None:
    plans = (ModelPlan("a", (ModelError("open"),)), ModelPlan("b", (ModelError("open"),)))
    frame = make_validation_frame(seed=0, models=plans)
    assert len(frame.items) == 2
    assert {item.models for item in frame.items} == {("a",), ("b",)}


def test_items_for_a_model_skip_the_other_models_errors() -> None:
    plans = (ModelPlan("a", (ModelError("open"),)), ModelPlan("b", (ModelError("bridge"),)))
    frame = make_validation_frame(seed=0, models=plans, data_defect="residue")
    assert {item.kind for item in frame.items_for("a")} == {"open", "residue"}
    assert {item.kind for item in frame.items_for("b")} == {"bridge", "residue"}
    assert len(frame.items_for(None)) == 3


def test_unknown_names_are_refused() -> None:
    with pytest.raises(ValueError):
        make_case("nonsense")
    with pytest.raises(ValueError):
        make_case("clean", behaviour="nonsense")
    with pytest.raises(ValueError):
        make_validation_frame(models=(ModelPlan("a", confidence=ConfidenceStyle(encoding="nonsense")),))
