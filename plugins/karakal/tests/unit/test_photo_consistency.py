"""The photo check: masks against the chip photo, the photo's quality, the models' confidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from karakal.core.analytics import compute_build_result_analytics
from karakal.core.confidence_maps import (
    ENCODING_CONFIDENCE,
    ENCODING_PROBABILITY,
    build_model_uncertainty,
    detect_confidence_encoding,
)
from karakal.core.domain import BuildOptions, BuildResult, FrameRecord, GeometryMode, ModelSpec
from karakal.core.findings import (
    DIAG_BAD_DATA,
    DIAG_CONFIDENT_ERROR,
    DIAG_MODEL_ERROR,
    DIAG_NOT_ALIGNED,
    DIAG_UNCERTAIN,
    frame_findings,
)
from karakal.core.mask_metrics import _configured_confidence_pair_metric_values
from karakal.core.photo_consistency import (
    BAD_PHOTO_KIND,
    DETECTOR_PHOTO,
    NOT_ALIGNED_KIND,
    UNCERTAIN_REGION_KIND,
    photo_findings,
)
from karakal.core.repository_shared import ComparisonPairSelection
from karakal.core.synthetic_chip import LayerStyle, PhotoStyle
from karakal.core.synthetic_scorecard import Scorecard, found_from_findings
from karakal.core.synthetic_validation import ModelError, ModelPlan, make_case, make_validation_frame

STYLES = {
    "manhattan": (LayerStyle(), PhotoStyle()),
    "realistic": (LayerStyle(), PhotoStyle(light_gradient=0.3, vignette=0.15, grain=5.0, edge_roughness=0.6, noise=5.0)),
    "dark_metal": (LayerStyle(), PhotoStyle(metal_level=60.0, substrate_level=190.0)),
    "diagonals": (LayerStyle(diagonals=True), PhotoStyle()),
    "vias": (LayerStyle(vias=True), PhotoStyle()),
}
SEEDS = (100, 101, 102)


def _check(frame, *, confidences: bool = False):
    found, metrics = photo_findings(
        frame.photo,
        {"a": frame.masks["a"] > 0},
        confidences={"a": frame.confidences["a"]} if confidences else None,
        point=frame.geometry == "point",
    )
    return found, metrics


def _card(cases, layer=None, photo=None, **case_options) -> Scorecard:
    card = Scorecard()
    for case in cases:
        for seed in SEEDS:
            frame = make_case(case, seed, layer, photo, model_count=1, **case_options)
            found, _metrics = _check(frame, confidences=True)
            card.add_frame(f"{case}{seed}", frame, found_from_findings(found, frame.shape, model_id="a"), model_id="a")
    return card


@pytest.mark.parametrize("style", sorted(STYLES))
def test_right_masks_on_clean_photos_give_nothing(style: str) -> None:
    layer, photo = STYLES[style]
    for seed in SEEDS:
        frame = make_case("clean", seed, layer, photo, model_count=1)
        found, metrics = _check(frame, confidences=True)
        assert found == (), (style, seed, found)
        assert metrics["photo_mismatch_risk::a"] == 0.0


@pytest.mark.parametrize("style", ["manhattan", "realistic", "dark_metal", "diagonals"])
def test_model_errors_on_a_good_photo_are_the_models(style: str) -> None:
    layer, photo = STYLES[style]
    kinds = ("open", "bridge", "missing_conductor", "neck")
    card = _card(kinds, layer, photo)
    for kind in kinds:
        assert card.recall(kind) >= 0.9, (style, kind, card.to_markdown())
    assert card.diagnosis_pairs.get((DIAG_MODEL_ERROR, DIAG_MODEL_ERROR), 0) > 0
    assert card.diagnosis_accuracy == 1.0


def test_point_errors_against_the_photo() -> None:
    kinds = ("missing_point", "extra_point", "merged_points", "shifted_point")
    card = _card(kinds, LayerStyle(vias=True), PhotoStyle())
    for kind in kinds:
        assert card.recall(kind) >= 0.9, (kind, card.to_markdown())
    assert card.false_findings == []


@pytest.mark.parametrize("defect", ["residue", "particles", "blur", "hair"])
def test_a_poor_photo_is_the_data_not_the_model(defect: str) -> None:
    for seed in SEEDS:
        frame = make_case(defect, seed, model_count=1)
        found, _metrics = _check(frame)
        assert any(item.kind == BAD_PHOTO_KIND and item.diagnosis == DIAG_BAD_DATA for item in found), (defect, seed)
        assert not [item for item in found if item.diagnosis == DIAG_MODEL_ERROR], (defect, seed, found)
    card = _card((defect,))
    assert card.recall(defect) >= 0.9


@pytest.mark.parametrize("defect", ["craters", "hole"])
def test_gross_damage_is_found_as_a_poor_photo(defect: str) -> None:
    card = _card((defect,))
    assert card.recall(defect) >= 0.6, card.to_markdown()


@pytest.mark.xfail(reason="a gentle stain (30% darker metal) stays above the mid-tones: not found yet", strict=False)
def test_a_stain_is_a_poor_photo() -> None:
    assert _card(("stain",)).recall("stain") >= 0.9


@pytest.mark.xfail(reason="a round particle on a via layer looks like a via: told apart by nothing yet", strict=False)
def test_particles_on_a_via_layer_are_not_missed_vias() -> None:
    card = _card(("particles",), LayerStyle(vias=True), PhotoStyle())
    assert not [item for _key, item in card.false_findings if item.code == "missing_point"]


@pytest.mark.parametrize("shift", [-2, 2])
def test_edges_shifted_against_the_photo_are_measured(shift: int) -> None:
    frame = make_validation_frame(seed=7, models=(ModelPlan("a", (ModelError("edge_shift", shift=shift),)),))
    found, _metrics = _check(frame)
    frame_wide = [item for item in found if item.kind == "boundary_shift" and item.box is None]
    assert len(frame_wide) == 1, found
    assert float(frame_wide[0].evidence_value("shift_px")) == pytest.approx(shift, abs=0.6)


def test_a_shifted_photo_is_only_said_to_be_shifted() -> None:
    for seed in SEEDS:
        frame = make_case("misaligned", seed, model_count=1)
        found, metrics = _check(frame)
        assert [(item.kind, item.diagnosis) for item in found] == [(NOT_ALIGNED_KIND, DIAG_NOT_ALIGNED)]
        (planted,) = frame.items
        assert float(found[0].evidence_value("dx")) == pytest.approx(float(planted.param("dx")), abs=1.0)
        assert metrics["photo_mismatch_risk::a"] > 0.0


def test_the_poor_place_is_where_the_film_is() -> None:
    frame = make_case("residue", 101, model_count=1)
    found, _metrics = _check(frame)
    (planted,) = frame.items
    poor = [item for item in found if item.kind == BAD_PHOTO_KIND and item.box is not None]
    x, y, w, h = planted.box
    film = np.zeros(frame.shape, bool)
    film[y : y + h, x : x + w] = True
    covered = np.zeros(frame.shape, bool)
    for item in poor:
        px, py, pw, ph = (int(round(v)) for v in item.pixel_box(frame.shape))
        covered[py : py + ph, px : px + pw] = True
    assert np.count_nonzero(film & covered) >= 0.5 * np.count_nonzero(film)


# --- Confidence -------------------------------------------------------------------------------------


def test_a_sure_model_that_is_wrong_makes_confident_errors() -> None:
    sure = _card(("open", "bridge", "missing_conductor"), behaviour="overconfident")
    calibrated = _card(("open", "bridge", "missing_conductor"), behaviour="calibrated")
    assert sure.diagnosis_pairs.get((DIAG_CONFIDENT_ERROR, DIAG_CONFIDENT_ERROR), 0) >= 8
    assert calibrated.diagnosis_pairs.get((DIAG_MODEL_ERROR, DIAG_MODEL_ERROR), 0) >= 8
    assert sure.diagnosis_accuracy == 1.0 and calibrated.diagnosis_accuracy == 1.0


def test_unsure_places_where_the_model_is_right() -> None:
    card = _card(("uncertain_flat",))
    assert card.recall("uncertain_flat") >= 0.9
    assert card.false_findings == []


def test_an_underconfident_model_is_one_finding_for_the_frame() -> None:
    frame = make_case("clean", 100, model_count=1, behaviour="underconfident")
    found, _metrics = _check(frame, confidences=True)
    assert [(item.kind, item.box, item.diagnosis) for item in found] == [(UNCERTAIN_REGION_KIND, None, DIAG_UNCERTAIN)]


@pytest.mark.parametrize("behaviour", ["calibrated", "overconfident"])
def test_both_encodings_give_the_same_findings(behaviour: str) -> None:
    for case in ("open", "uncertain_flat", "clean"):
        first, _m1 = _check(make_case(case, 100, model_count=1, behaviour=behaviour, encoding="probability"), confidences=True)
        second, _m2 = _check(make_case(case, 100, model_count=1, behaviour=behaviour, encoding="confidence"), confidences=True)
        assert [(item.kind, item.box, item.diagnosis) for item in first] == [(item.kind, item.box, item.diagnosis) for item in second]


@pytest.mark.parametrize("encoding", [ENCODING_PROBABILITY, ENCODING_CONFIDENCE])
def test_the_encoding_is_read_from_the_background(encoding: str) -> None:
    for case in ("clean", "open", "residue"):
        frame = make_case(case, 100, model_count=1, encoding=encoding)
        assert detect_confidence_encoding(frame.confidences["a"], frame.masks["a"] > 0) == encoding


def test_a_mid_gray_map_keeps_the_old_reading() -> None:
    frame = make_case("clean", 100, model_count=1)
    flat = np.full(frame.shape, 128, np.uint8)
    assert detect_confidence_encoding(flat, frame.masks["a"] > 0) == ENCODING_PROBABILITY
    assert detect_confidence_encoding(flat, None) == ENCODING_PROBABILITY
    assert np.allclose(build_model_uncertainty(np.zeros((4, 4)), ENCODING_CONFIDENCE), 1.0)
    assert np.allclose(build_model_uncertainty(np.zeros((4, 4)), ENCODING_PROBABILITY), 0.0)


def test_calibration_and_sureness_of_errors() -> None:
    def mean_metric(behaviour: str, key: str) -> float:
        values = []
        for case in ("open", "missing_conductor", "clean"):
            for seed in SEEDS:
                _found, metrics = _check(make_case(case, seed, model_count=1, behaviour=behaviour), confidences=True)
                if key in metrics:
                    values.append(metrics[key])
        return float(np.mean(values))

    assert mean_metric("underconfident", "confidence_ece::a") > mean_metric("calibrated", "confidence_ece::a")
    assert mean_metric("overconfident", "confident_error_share::a") > mean_metric("calibrated", "confident_error_share::a")


def test_pair_confidence_of_probability_maps_is_read_as_confidence() -> None:
    frame = make_case("clean", 100)
    pair = (ComparisonPairSelection("a", "b", ("xor",)),)
    probabilities = {m: frame.confidences[m].astype(np.float32) / 255.0 for m in ("a", "b")}
    masks = {m: frame.masks[m] > 0 for m in ("a", "b")}
    values = _configured_confidence_pair_metric_values(pair, probabilities, encoding="auto", masks_by_model=masks)
    # The two calibrated models are sure of the same background: their "low confidence" places
    # are the edges, which they share, not the whole background.
    low_iou = next(value for key, value in values.items() if key.endswith("low_iou"))
    disagreement = next(value for key, value in values.items() if key.endswith("disagreement"))
    assert low_iou > 0.8 and disagreement < 0.05


# --- Through the analytics ------------------------------------------------------------------------


def _write(path: Path, image) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return str(path)


def _run(tmp_path: Path, metric_key: str, photo_check: bool = False) -> BuildResult:
    frames = {"F_00": make_case("open", 100), "F_01": make_case("residue", 101), "F_02": make_case("clean", 102)}
    records = tuple(
        FrameRecord(
            key,
            key,
            original_path=_write(tmp_path / "photos" / f"{key}.png", frame.photo),
            model_mask_paths={m: _write(tmp_path / m / f"{key}.png", frame.masks[m]) for m in ("a", "b")},
            model_prob_paths={m: _write(tmp_path / m / f"{key}_confidence.png", frame.confidences[m]) for m in ("a", "b")},
        )
        for key, frame in frames.items()
    )
    source = BuildResult(
        records=records,
        model_specs=tuple(ModelSpec(m, m, tmp_path / m, prob_folder=tmp_path / m) for m in ("a", "b")),
        options=BuildOptions(cache_enabled=False, geometry_mode=GeometryMode.MASK, photo_check=photo_check),
    )
    return compute_build_result_analytics(source, metric_key=metric_key)


def test_choosing_a_photo_metric_runs_the_photo_check(tmp_path: Path) -> None:
    result = _run(tmp_path, "photo_mismatch_risk")
    assert result.selected_metric_key == "photo_mismatch_risk"
    by_key = {record.key: record.summary for record in result.records}
    broken = [item for item in frame_findings(by_key["F_00"]) if item.detector == DETECTOR_PHOTO]
    assert {(item.model_id, item.kind) for item in broken} >= {("a", "open")}
    assert by_key["F_00"].metric_values["photo_mismatch_risk"] > 0.0
    assert by_key["F_02"].metric_values["photo_mismatch_risk"] == 0.0
    film = [item for item in frame_findings(by_key["F_01"]) if item.kind == BAD_PHOTO_KIND]
    assert film and all(item.diagnosis == DIAG_BAD_DATA for item in film)
    assert by_key["F_01"].metric_values["photo_quality_risk"] > by_key["F_02"].metric_values["photo_quality_risk"]
    assert "confidence_ece::a" in by_key["F_00"].metric_values
    assert "photo_mismatch_risk::b" in result.available_metric_keys


def test_without_a_photo_metric_the_photo_is_not_read(tmp_path: Path) -> None:
    plain = _run(tmp_path, "topology_risk")
    assert all(item.detector != DETECTOR_PHOTO for record in plain.records for item in frame_findings(record.summary))
    switched_on = _run(tmp_path / "on", "topology_risk", photo_check=True)
    assert any(item.detector == DETECTOR_PHOTO for record in switched_on.records for item in frame_findings(record.summary))
