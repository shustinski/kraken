from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from kraken_core.analysis_protocol import AnalysisProfileKind

from karakal.core.analysis_profiles import auto_validation_profile, build_standalone_preflight
from karakal.core.analytics import compute_build_result_analytics
from karakal.core.domain import BuildOptions, BuildResult, FrameAnalysisSummary, FrameRecord, ModelSpec
from karakal.core.heuristic_check import (
    HEURISTIC_CHECK_SCORE,
    VERDICT_DIRT,
    VERDICT_ETCH_DEVIATION,
    VERDICT_NORMAL,
    VERDICT_TOPOLOGY,
    VERDICT_UNREADABLE,
    HeuristicCheckSummary,
    analyze_frame,
    compute_heuristic_check,
    folder_holds_photos,
    looks_like_photo,
    verdict_counts,
)
from karakal.core.image_io import _grayscale_array_to_qimage
from karakal.core.single_result_risk import RiskReason

WIDTH, PITCH = 8, 24


def _segments(rng: np.random.Generator, start: int, stop: int) -> list[tuple[int, int]]:
    """Pieces of one track with designed cuts at least a pitch long."""

    runs: list[tuple[int, int]] = []
    position = start
    while True:
        end = min(stop, position + int(rng.integers(60, 160)))
        if end - position >= 30:
            runs.append((position, end))
        position = end + int(rng.integers(PITCH, 2 * PITCH))
        if position >= stop - 30:
            return runs


def _layout(seed: int = 0, size: int = 512) -> np.ndarray:
    """A clean metal layer: horizontal tracks, vertical tracks and a pour, every gap 16 px."""

    rng = np.random.default_rng(seed)
    metal = np.zeros((size, size), dtype=np.uint8)
    for y in range(16, 360, PITCH):
        for a, b in _segments(rng, 8, 248):
            metal[y : y + WIDTH, a:b] = 1
    for x in range(272, size - 8 - WIDTH, PITCH):
        for a, b in _segments(rng, 8, 368):
            metal[a:b, x : x + WIDTH] = 1
    metal[400:480, 16:240] = 1
    return metal


def _photo(metal: np.ndarray, *, seed: int = 0, blur: float = 1.2, noise: float = 4.0) -> np.ndarray:
    rng = np.random.default_rng(seed + 100)
    image = cv2.GaussianBlur(np.where(metal > 0, 180, 60).astype(np.float32), (0, 0), blur)
    return np.clip(image + rng.normal(0.0, noise, image.shape), 0, 255).astype(np.uint8)


def _long_run(metal: np.ndarray, y: int) -> tuple[int, int]:
    """Columns of the longest piece of the horizontal track at row ``y``."""

    columns = np.flatnonzero(metal[y + WIDTH // 2, :248])
    breaks = np.flatnonzero(np.diff(columns) > 1)
    starts = np.concatenate(([columns[0]], columns[breaks + 1]))
    ends = np.concatenate((columns[breaks], [columns[-1]]))
    index = int(np.argmax(ends - starts))
    return int(starts[index]), int(ends[index])


def _with_open(metal: np.ndarray, gap: int = 6) -> tuple[np.ndarray, int]:
    broken = metal.copy()
    start, end = _long_run(metal, 64)
    middle = (start + end) // 2
    broken[64 : 64 + WIDTH, middle : middle + gap] = 0
    return broken, middle


def _codes(findings) -> set[str]:
    return {reason.code for reason in findings.reasons}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_clean_layer_has_no_findings(seed: int) -> None:
    findings = analyze_frame(_photo(_layout(seed), seed=seed))

    assert findings.capture_issues == ()
    assert findings.topology_raw == 0.0
    assert findings.contamination_raw < 0.05
    assert findings.geometry_raw == 0.0
    assert findings.features["width_narrow"] == pytest.approx(7.0, abs=1.0)
    assert findings.features["gap_narrow"] == pytest.approx(15.0, abs=1.5)


def test_noisy_clean_layer_has_no_findings() -> None:
    findings = analyze_frame(_photo(_layout(), noise=10.0))

    assert findings.topology_raw == 0.0
    assert findings.contamination_raw < 0.05


@pytest.mark.parametrize("gap", [3, 6])
def test_open_is_found_between_facing_line_ends(gap: int) -> None:
    broken, middle = _with_open(_layout(), gap)

    findings = analyze_frame(_photo(broken))
    opens = [reason for reason in findings.reasons if reason.code == "open"]

    assert findings.extra["open_count"] == 1.0
    assert len(opens) == 1
    x, y, w, h = opens[0].bbox
    assert x * 512 <= middle <= (x + w) * 512
    assert y * 512 <= 68 <= (y + h) * 512
    assert findings.topology_raw >= 0.5


def test_neck_and_thin_bridge_are_thin_metal() -> None:
    metal = _layout()
    start, end = _long_run(metal, 64)
    middle = (start + end) // 2
    necked = metal.copy()
    necked[64:66, middle : middle + 10] = 0
    necked[70:72, middle : middle + 10] = 0
    bridged = metal.copy()
    bridged[72:88, middle : middle + 3] = 1

    assert "thin_metal" in _codes(analyze_frame(_photo(necked)))
    assert "thin_metal" in _codes(analyze_frame(_photo(bridged)))


def test_particles_on_metal_and_substrate_are_dirt() -> None:
    image = _photo(_layout())
    dark_on_pour = image.copy()
    cv2.circle(dark_on_pour, (120, 440), 4, 40, -1)
    bright_on_substrate = image.copy()
    cv2.circle(bright_on_substrate, (130, 382), 5, 200, -1)

    assert "pinhole" in _codes(analyze_frame(dark_on_pour))
    assert "debris" in _codes(analyze_frame(bright_on_substrate))


def test_residue_film_is_dirt_not_broken_topology() -> None:
    image = _photo(_layout())
    image[200:260, 100:200] = np.clip(image[200:260, 100:200].astype(int) + 55, 0, 255).astype(np.uint8)

    findings = analyze_frame(image)

    assert "residue" in _codes(findings)
    assert findings.contamination_raw == 1.0
    assert findings.topology_raw == 0.0


def test_poor_capture_is_not_rated() -> None:
    metal = _layout()
    flat = np.clip(_photo(metal).astype(np.float32) * 0.05 + 100, 0, 255).astype(np.uint8)

    assert analyze_frame(flat).capture_issues == ("low_contrast",)
    assert "blur" in analyze_frame(_photo(metal, blur=4.0)).capture_issues
    assert analyze_frame(np.full((128, 128), 120, dtype=np.uint8)).capture_issues == ("low_contrast",)


def test_large_frame_is_read_at_the_working_scale_and_reported_in_its_own_pixels() -> None:
    metal = cv2.resize(_layout(), (2048, 2048), interpolation=cv2.INTER_NEAREST)

    findings = analyze_frame(_photo(metal, blur=4.0))

    assert findings.extra["analysis_scale"] < 0.5
    assert findings.topology_raw == 0.0
    assert findings.features["width_narrow"] == pytest.approx(28.0, rel=0.15)
    assert findings.features["gap_narrow"] == pytest.approx(60.0, rel=0.15)


def test_photos_are_told_from_masks(tmp_path: Path) -> None:
    metal = _layout()
    soft = cv2.GaussianBlur(metal.astype(np.float32) * 255.0, (0, 0), 1.5).astype(np.uint8)

    assert looks_like_photo(_photo(metal))
    assert not looks_like_photo(metal * 255)
    assert not looks_like_photo(soft)

    photos = tmp_path / "photos"
    masks = tmp_path / "masks"
    for index in range(3):
        _write(photos / f"F_{index:04d}.png", _photo(_layout(index), seed=index))
        _write(masks / f"F_{index:04d}.png", _layout(index) * 255)
    assert folder_holds_photos(photos)
    assert not folder_holds_photos(masks)
    assert not folder_holds_photos(tmp_path / "missing")


def _write(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert _grayscale_array_to_qimage(np.ascontiguousarray(array, dtype=np.uint8)).save(str(path))
    return path


def _run(tmp_path: Path, images: list[np.ndarray]) -> BuildResult:
    folder = tmp_path / "photos"
    records = []
    for index, image in enumerate(images):
        key = f"F_{index:04d}"
        path = _write(folder / f"{key}.png", image)
        records.append(
            FrameRecord(key, key, first_path=str(path), second_path=str(path), model_mask_paths={"photos": str(path)})
        )
    return BuildResult(
        records=tuple(records),
        model_specs=(ModelSpec("photos", "Photos", folder),),
        options=BuildOptions(single_result_sensitivity="balanced"),
    )


def _check(record: FrameRecord) -> HeuristicCheckSummary:
    summary = getattr(record.summary, "heuristic_check", None)
    assert summary is not None, record.summary
    return summary


def test_run_rates_every_frame_and_names_the_leading_problem(tmp_path: Path) -> None:
    clean = [_photo(_layout(seed), seed=seed) for seed in range(10)]
    broken, _middle = _with_open(_layout(20))
    over_etched = cv2.erode(_layout(21), np.ones((3, 3), dtype=np.uint8))
    dirty = _photo(_layout(22), seed=22)
    dirty[200:260, 100:200] = np.clip(dirty[200:260, 100:200].astype(int) + 55, 0, 255).astype(np.uint8)
    flat = np.full((512, 512), 120, dtype=np.uint8)
    images = clean + [_photo(broken, seed=20), _photo(over_etched, seed=21), dirty, flat]

    result = compute_build_result_analytics(_run(tmp_path, images), metric_key=HEURISTIC_CHECK_SCORE)
    records = {record.key: record for record in result.records}
    checks = {key: _check(record) for key, record in records.items()}

    assert result.selected_metric_key == HEURISTIC_CHECK_SCORE
    assert all(checks[f"F_{index:04d}"].verdict == VERDICT_NORMAL for index in range(10))
    assert checks["F_0010"].verdict == VERDICT_TOPOLOGY
    assert checks["F_0011"].verdict == VERDICT_ETCH_DEVIATION
    assert any(reason.code == "etch_over" for reason in checks["F_0011"].reasons)
    assert checks["F_0012"].verdict == VERDICT_DIRT
    assert checks["F_0013"].verdict == VERDICT_UNREADABLE
    assert not records["F_0013"].score_ready
    assert records["F_0013"].absolute_score is None
    worst_clean = max(float(records[f"F_{index:04d}"].absolute_score) for index in range(10))
    for key in ("F_0010", "F_0011", "F_0012"):
        assert float(records[key].absolute_score) > worst_clean + 20.0
    assert sum(checks["F_0010"].component_contributions.values()) <= checks["F_0010"].total_risk + 1e-6
    assert verdict_counts(result) == {
        VERDICT_NORMAL: 10,
        VERDICT_TOPOLOGY: 1,
        VERDICT_DIRT: 1,
        VERDICT_ETCH_DEVIATION: 1,
        VERDICT_UNREADABLE: 1,
    }


def test_small_run_has_no_run_components(tmp_path: Path) -> None:
    result = compute_heuristic_check(_run(tmp_path, [_photo(_layout(seed), seed=seed) for seed in range(3)]))

    for record in result.records:
        check = _check(record)
        assert check.batch_outlier_risk is None
        assert check.width_risk is None
        assert record.score_ready


def test_unreadable_photo_file_is_an_error(tmp_path: Path) -> None:
    folder = tmp_path / "photos"
    folder.mkdir()
    broken = folder / "broken.png"
    broken.write_bytes(b"not-an-image")
    build_result = BuildResult(
        records=(FrameRecord("broken", "broken", model_mask_paths={"photos": str(broken)}),),
        model_specs=(ModelSpec("photos", "Photos", folder),),
    )

    result = compute_heuristic_check(build_result)

    assert not result.records[0].score_ready
    assert result.records[0].summary.notes == ("decode_error",)


def test_photo_folder_picks_the_heuristic_check(tmp_path: Path) -> None:
    assert auto_validation_profile(1, 0, photo_source=True) == AnalysisProfileKind.HEURISTIC_CHECK
    assert auto_validation_profile(1, 0) == AnalysisProfileKind.SINGLE_RESULT_RISK
    assert auto_validation_profile(1, 1, photo_source=True) == AnalysisProfileKind.CONFIDENCE_AUDIT
    assert auto_validation_profile(2, 0, photo_source=True) == AnalysisProfileKind.MODEL_COMPARISON

    photos = tmp_path / "photos"
    _write(photos / "F_0001.png", _photo(_layout()))
    report = build_standalone_preflight(
        AnalysisProfileKind.HEURISTIC_CHECK, None, (ModelSpec("photos", "Photos", photos),)
    )
    roles = {role.role.value: role for role in report.roles}
    assert report.can_run
    assert roles["original"].state == "ready"
    assert roles["model_output"].state == "missing"

    empty = build_standalone_preflight(AnalysisProfileKind.HEURISTIC_CHECK, None, ())
    assert not empty.can_run
    assert "photo_source_required" in {issue.code for issue in empty.issues}
    two = build_standalone_preflight(
        AnalysisProfileKind.HEURISTIC_CHECK,
        None,
        (ModelSpec("a", "A", photos), ModelSpec("b", "B", photos)),
    )
    assert "single_photo_source_required" in {issue.code for issue in two.issues}


def test_export_manifest_keeps_the_verdict_and_reasons(tmp_path: Path) -> None:
    import json

    from karakal.core.exports import export_ranked_frames

    broken, _middle = _with_open(_layout())
    result = compute_heuristic_check(_run(tmp_path, [_photo(broken), _photo(_layout(1), seed=1)]))

    exported = export_ranked_frames(result, tmp_path / "export", metric_key=HEURISTIC_CHECK_SCORE, top_k=2)
    manifest = json.loads(Path(exported["manifest_path"]).read_text(encoding="utf-8"))

    entry = manifest["heuristic_check"]["F_0000"]
    assert entry["verdict"] == VERDICT_TOPOLOGY
    assert any(reason["code"] == "open" for reason in entry["reasons"])


def test_details_show_verdict_findings_and_layers(tmp_path: Path, qtbot) -> None:
    from karakal.ui.details_dialog import HEURISTIC_RESULT_KINDS, ExtendFrameDetailsDialog

    path = _write(tmp_path / "frame.png", _photo(_layout()))
    check = HeuristicCheckSummary(
        total_risk=64.0,
        verdict=VERDICT_TOPOLOGY,
        topology_risk=80.0,
        contamination_risk=10.0,
        width_risk=None,
        geometry_risk=0.0,
        batch_outlier_risk=None,
        sensitivity="balanced",
        capture_issues=(),
        reasons=(
            RiskReason("open", "high", 0.8, (0.1, 0.1, 0.05, 0.05)),
            RiskReason("particle", "low", 0.1, (0.5, 0.5, 0.02, 0.02)),
        ),
        component_weights={"heuristic_topology_risk": 0.5},
        component_contributions={"heuristic_topology_risk": 40.0},
        feature_values={"width_narrow": 7.0, "gap_narrow": 15.0, "residue_fraction": 0.0},
    )
    record = FrameRecord(
        "frame",
        "frame",
        first_path=str(path),
        second_path=str(path),
        model_mask_paths={"photos": str(path)},
        score_ready=True,
        absolute_score=64.0,
        summary=FrameAnalysisSummary(0.0, 0.0, 0.8, 0.64, heuristic_check=check),
    )
    dialog = ExtendFrameDetailsDialog(
        record,
        BuildResult(records=(record,), options=BuildOptions(), scores_computed=True),
        allowed_result_kinds=HEURISTIC_RESULT_KINDS,
    )
    qtbot.addWidget(dialog)
    dialog._payload["heuristic_check"] = check
    dialog._refresh_result_kind_options("heuristic_all")
    dialog._refresh_comparison_panel()

    assert dialog.result_kind_combo.count() == len(HEURISTIC_RESULT_KINDS)
    assert dialog._t("heuristic.verdict.topology") in dialog.comparison_metrics_label.text()
    assert dialog._t("heuristic.reason.open") in dialog.comparison_events_label.text()
    for kind in HEURISTIC_RESULT_KINDS:
        assert not dialog._heuristic_check_pixmap(kind).isNull()


def test_photo_folder_switches_validation_to_the_heuristic_check(tmp_path: Path, qtbot) -> None:
    from PyQt6.QtCore import QSettings

    from karakal.app.main_window import KarakalWidget

    widget = KarakalWidget(settings=QSettings(str(tmp_path / "karakal.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    presenter = widget._presenter
    photos = tmp_path / "photos"
    for index in range(2):
        _write(photos / f"F_{index:04d}.png", _photo(_layout(index), seed=index))

    presenter._append_folder_item(photos, checked=True)
    presenter._refresh_folder_rows()
    presenter._sync_action_buttons()

    assert presenter._analysis_profile == AnalysisProfileKind.HEURISTIC_CHECK
    assert widget.app_mode_combo.currentData() == "validation"
    assert widget.metric_combo.currentData() == HEURISTIC_CHECK_SCORE
    assert widget._t("profile.heuristic_check.title") in widget.analysis_setup_panel.profile_description_label.text()
    assert not widget._matrix_single_result_sensitivity_row.isHidden()
    assert widget._matrix_geometry_row.isHidden()
