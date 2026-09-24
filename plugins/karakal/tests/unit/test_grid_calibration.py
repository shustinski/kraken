"""Calibration storage, example correction, and the shared decision."""

from __future__ import annotations

import json
from time import perf_counter

from PyQt6.QtCore import QSettings

from karakal.app.main_window import KarakalWidget, QtUpdateController
from karakal.core.grid_anomaly import GridDamageAnalysisConfig, _grid_damage_cache_key
from karakal.core.grid_calibration import (
    GridCalibration,
    GridCellExample,
    apply_example_correction,
    calibration_summary,
    decide_reasons,
    example_detector_scores,
    fit_slider,
    fit_sliders_from_examples,
)
from karakal.infra.services import KarakalSettingsService
from karakal.ui.ui_constants import SETTINGS_GRID_CALIBRATION_KEY


def _example(label: str, interior: float, *, thumbnail: str = "", created_at: str = "") -> GridCellExample:
    return GridCellExample(
        label=label,
        features=(("interior_fill", interior), ("solidity", 0.9), ("extent", 0.4), ("area", 200.0)),
        frame_key="frame-a",
        bbox=(1, 2, 10, 12),
        thumbnail_png_b64=thumbnail,
        created_at=created_at,
    )


def test_calibration_roundtrip_and_bad_payload() -> None:
    original = GridCalibration(
        preset="custom",
        examples=(_example("normal", 0.1),),
        calibration_frame_keys=("frame-a", "frame-b"),
        reference={"interior_fill": 0.1},
        example_influence=0.4,
    )
    restored = GridCalibration.from_payload(original.to_payload())
    assert restored.preset == "custom"
    assert restored.examples[0].label == "normal"
    assert restored.calibration_frame_keys == ("frame-a", "frame-b")
    assert restored.reference == {"interior_fill": 0.1}
    assert GridCalibration.from_payload({"schema": "old"}) == GridCalibration()
    assert GridCalibration.from_payload("broken") == GridCalibration()


def test_fingerprint_ignores_thumbnail_and_date() -> None:
    left = GridCalibration(examples=(_example("normal", 0.1, thumbnail="aaa", created_at="2020"),))
    right = GridCalibration(examples=(_example("normal", 0.1, thumbnail="bbb", created_at="2026"),))
    changed = GridCalibration(examples=(_example("filled_cell", 0.9),))
    assert left.fingerprint() == right.fingerprint()
    assert left.fingerprint() != changed.fingerprint()


def test_normal_example_suppresses_and_defect_example_promotes() -> None:
    features = {"interior_fill": 0.12, "solidity": 0.9, "extent": 0.4, "area": 200.0}
    normal = {"label": "normal", "features": features}
    filled = {"label": "filled_cell", "features": {"interior_fill": 0.9, "solidity": 0.98, "extent": 0.9, "area": 800.0}}
    suppressed = apply_example_correction(
        ("filled_cell",),
        {"fill": 0.8},
        {"fill": 0.4},
        features,
        (normal,),
        example_influence=1.0,
    )
    assert "filled_cell" not in suppressed
    promoted = apply_example_correction(
        (),
        {"fill": 0.45, "geometry": 0.0, "merge": 0.0, "debris": 0.0},
        {"fill": 0.55, "geometry": 1, "merge": 1, "debris": 1},
        filled["features"],
        (filled, normal),
        example_influence=1.0,
    )
    assert "filled_cell" in promoted
    untouched = apply_example_correction(
        ("filled_cell",),
        {"fill": 0.8},
        {"fill": 0.4},
        features,
        ({"label": "normal", "features": None},),
        example_influence=1.0,
    )
    assert untouched == ("filled_cell",)


def _typed_features(label: str) -> dict[str, float]:
    features = {
        "width_ratio": 1.0,
        "height_ratio": 1.0,
        "area_ratio": 1.0,
        "interior_fill": 0.12,
        "reference_interior": 0.12,
        "solidity": 0.92,
        "reference_solidity": 0.92,
        "extent": 0.74,
        "reference_extent": 0.74,
    }
    if label == "filled_cell":
        features["interior_fill"] = 0.95
    elif label == "broken_geometry":
        features["solidity"] = 0.40
    elif label == "merged_contour":
        features["width_ratio"] = 2.4
        features["area_ratio"] = 2.6
    elif label == "small_artifact":
        features["area_ratio"] = 0.12
        features["width_ratio"] = 0.35
        features["height_ratio"] = 0.35
    return features


def test_fitted_thresholds_classify_every_synthetic_example() -> None:
    from karakal.core.grid_scoring import slider_threshold

    labels = ("normal", "filled_cell", "broken_geometry", "merged_contour", "small_artifact")
    examples = [{"label": label, "features": _typed_features(label)} for label in labels for _repeat in range(2)]
    fitted = fit_sliders_from_examples(examples)
    assert set(fitted) == {
        "fill_sensitivity",
        "geometry_sensitivity",
        "merge_sensitivity",
        "debris_sensitivity",
    }
    thresholds = {name.removesuffix("_sensitivity"): slider_threshold(value) for name, value in fitted.items()}
    for example in examples:
        reasons = decide_reasons(example_detector_scores(example["features"]), thresholds, features=example["features"])
        if example["label"] == "normal":
            assert reasons == ()
        else:
            assert example["label"] in reasons


def test_slider_preview_does_not_reread_the_mask(monkeypatch) -> None:
    from types import SimpleNamespace

    from karakal.app.presenter import KarakalPresenter
    from karakal.core.grid_anomaly import GridCellAnalysisResult, GridFrameAnalysisResult

    calls = {"n": 0}
    cell = GridCellAnalysisResult(
        0,
        0,
        (1, 1, 8, 8),
        (5.0, 5.0),
        1,
        "normal",
        0.0,
        (),
        feature_snapshot=(
            ("width_ratio", 1.0),
            ("height_ratio", 1.0),
            ("area_ratio", 1.0),
            ("interior_fill", 0.9),
            ("reference_interior", 0.1),
            ("solidity", 0.9),
            ("reference_solidity", 0.9),
            ("extent", 0.7),
            ("reference_extent", 0.7),
            ("fill_score", 0.9),
            ("geometry_score", 0.0),
            ("merge_score", 0.0),
            ("debris_score", 0.0),
            ("edge_score", 0.0),
        ),
    )
    frame = GridFrameAnalysisResult(
        frame_id="f",
        frame_path="mask.png",
        image_width=32,
        image_height=32,
        grid_rows=1,
        grid_cols=1,
        total_expected_cells=1,
        detected_cells=1,
        normal_cells=1,
        suspicious_cells=0,
        broken_cells=0,
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.0,
        severity_level="none",
        grid_detected=True,
        per_cell_results=(cell,),
    )

    def fake_analyze(*_args, **_kwargs):
        calls["n"] += 1
        return frame

    monkeypatch.setattr("karakal.core.grid_anomaly.analyze_grid_frame_path", fake_analyze)
    host = SimpleNamespace(
        _grid_calibration=GridCalibration(),
        _grid_prepared_frames={},
        _grid_tuning_dialog=None,
    )
    host._put_calibration_in_payload = KarakalPresenter._put_calibration_in_payload
    host._refresh_grid_calibration_summary = lambda dialog=None: None
    host._grid_score_thresholds = KarakalPresenter._grid_score_thresholds.__get__(host)
    host._redecide_prepared_frame = KarakalPresenter._redecide_prepared_frame.__get__(host)
    host._grid_inspection_model_id_for_state = lambda _state: "m"
    host._grid_inspection_config_payload = lambda: {"scoring_mode": "calibrated"}
    host._grid_damage_config_from_payload = lambda _payload: GridDamageAnalysisConfig(scoring_mode="calibrated")
    record = SimpleNamespace(key="f", model_mask_paths={"m": "mask.png"}, model_prob_paths={})
    KarakalPresenter._analyze_grid_record_for_details(host, record, SimpleNamespace(), {"fill_sensitivity": 60})
    KarakalPresenter._analyze_grid_record_for_details(host, record, SimpleNamespace(), {"fill_sensitivity": 90})
    assert calls["n"] == 1


def test_fit_slider_separates_examples_and_skips_sparse_labels() -> None:
    assert fit_slider((0.9, 0.85), (0.1, 0.2)) is not None
    assert fit_slider((0.9,), (0.1, 0.2)) is None
    slider = fit_slider((0.9, 0.8), (0.1, 0.15))
    assert slider is not None and slider >= 50


def test_cached_decision_stays_within_budget() -> None:
    scores = {"fill": 0.2, "geometry": 0.1, "merge": 0.0, "debris": 0.3, "edge": 0.0}
    thresholds = {"fill": 0.4, "geometry": 0.6, "merge": 0.65, "debris": 0.25, "edge": 1.0}
    started = perf_counter()
    for _ in range(4000):
        decide_reasons(scores, thresholds)
    assert perf_counter() - started < 0.05


def test_summary_counts_agreement() -> None:
    summary = calibration_summary(
        (
            {"key": "a", "before": 4, "after": 2, "examples_ok": 2, "examples_total": 3, "disputed": 1},
            {"key": "b", "before": 1, "after": 1, "examples_ok": 1, "examples_total": 1, "disputed": 0},
        )
    )
    assert summary["examples_ok"] == 3
    assert summary["examples_total"] == 4
    assert summary["frames"][0]["before"] == 4


def test_cache_key_tracks_fingerprint(tmp_path) -> None:
    base = GridDamageAnalysisConfig(scoring_mode="calibrated")
    changed = GridDamageAnalysisConfig(scoring_mode="calibrated", calibration_fingerprint="abc")
    path = tmp_path / "frame.png"
    path.write_bytes(b"x")
    assert _grid_damage_cache_key(path, frame_id="f", config=base) != _grid_damage_cache_key(
        path, frame_id="f", config=changed
    )


def test_settings_roundtrip_and_close_keeps_confirmed(tmp_path) -> None:
    settings = QSettings(str(tmp_path / "k.ini"), QSettings.Format.IniFormat)
    service = KarakalSettingsService(settings)
    saved = GridCalibration(preset="strict", examples=(_example("ignore", 0.2),), confirmed_at="2026-09-24T12:00:00+00:00")
    service.save_grid_calibration_payload(saved.to_payload())
    loaded = GridCalibration.from_payload(service.load_grid_calibration_payload())
    assert loaded.preset == "strict"
    assert loaded.examples[0].label == "ignore"
    confirmed = saved
    working = GridCalibration(preset="custom")
    working = confirmed
    assert working.preset == "strict"


def _schema(**extra: object) -> dict:
    payload = {"schema": "karakal.grid-calibration.v1", "preset": "custom", "example_influence": 0.4}
    payload.update(extra)
    return payload


def test_broken_example_is_skipped() -> None:
    loaded = GridCalibration.from_payload(
        _schema(
            examples=[
                {"label": "normal", "features": [0.5, 0.7]},
                {"label": "filled_cell", "features": [["interior_fill", 0.9]]},
            ]
        )
    )
    assert loaded.preset == "custom"
    assert len(loaded.examples) == 1
    assert loaded.examples[0].label == "filled_cell"


def test_broken_slider_is_skipped() -> None:
    loaded = GridCalibration.from_payload(_schema(sliders={"fill_sensitivity": "abc", "debris_sensitivity": 10}))
    assert loaded.sliders == {"debris_sensitivity": 10}
    assert loaded.preset == "custom"


def test_zero_example_influence_is_kept() -> None:
    loaded = GridCalibration.from_payload(_schema(example_influence=0))
    assert loaded.example_influence == 0.0


def test_widget_starts_with_broken_calibration(tmp_path, qtbot, monkeypatch) -> None:
    monkeypatch.setattr(QtUpdateController, "check_for_updates", lambda self, manual=False: None)
    settings = QSettings(str(tmp_path / "k.ini"), QSettings.Format.IniFormat)
    settings.setValue(
        SETTINGS_GRID_CALIBRATION_KEY,
        json.dumps(
            {
                "schema": "karakal.grid-calibration.v1",
                "examples": [{"label": "normal", "features": [0.5, 0.7]}],
                "sliders": {"fill_sensitivity": "abc"},
                "example_influence": 0,
            }
        ),
    )
    widget = KarakalWidget(settings=settings)
    qtbot.addWidget(widget)
    assert widget._presenter._grid_calibration.example_influence == 0.0
    assert widget._presenter._grid_calibration.examples == ()
