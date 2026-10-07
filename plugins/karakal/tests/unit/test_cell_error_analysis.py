"""Grid-free cell-error analysis: the acceptance list of the 0.3 specification (section 15)."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from karakal.core import error_classes as ec
from karakal.core.cell_components import extract_components
from karakal.core.cell_error_analysis import (
    MODE_MASK,
    MODE_MASK_CONF,
    MODE_MASK_CONF_REF,
    MODE_MASK_REF,
    ErrorAnalysisResult,
    analyze_cell_errors,
)
from karakal.core.error_maps import export_maps
from karakal.core.normal_bank import SOURCE_REFERENCE, STATUS_INSUFFICIENT, build_normal_bank
from karakal.core.synthetic_masks import SyntheticLayout, add_defect, normal_frame, synthetic_frame


@pytest.fixture(scope="module")
def bank():
    return build_normal_bank([extract_components(normal_frame(seed=seed).mask) for seed in range(100, 105)])


def _object_classes(mask: np.ndarray, class_map: np.ndarray) -> list[int]:
    count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return [int(np.bincount(class_map[labels == label]).argmax()) for label in range(1, count)]


def _defect_class(frame, result: ErrorAnalysisResult) -> int:
    pixels = frame.defects[-1].pixels
    return int(np.bincount(result.class_map()[pixels]).argmax())


def test_all_four_input_sets_give_the_same_result_type(bank) -> None:
    frame = synthetic_frame({"bite": 1, "debris": 1}, seed=3)
    reference = build_normal_bank(
        [extract_components(normal_frame(seed=seed).mask) for seed in range(110, 113)], source=SOURCE_REFERENCE
    )
    confidence = np.full(frame.mask.shape, 250, dtype=np.uint8)
    results = {
        MODE_MASK: analyze_cell_errors(frame.mask, bank),
        MODE_MASK_CONF: analyze_cell_errors(frame.mask, bank, confidence=confidence),
        MODE_MASK_REF: analyze_cell_errors(frame.mask, reference),
        MODE_MASK_CONF_REF: analyze_cell_errors(frame.mask, reference, confidence=confidence),
    }
    for mode, result in results.items():
        assert isinstance(result, ErrorAnalysisResult)
        assert result.mode == mode
        assert result.class_map().shape == frame.mask.shape
    assert results[MODE_MASK_CONF].inputs == {"mask": True, "confidence": True, "reference": False}


def test_normal_cells_anywhere_are_normal_without_a_grid(bank) -> None:
    for seed in range(4):
        frame = normal_frame(SyntheticLayout(arrangement="scatter"), seed=seed)
        result = analyze_cell_errors(frame.mask, bank)
        assert result.regions == ()
    lone = np.zeros((300, 300), dtype=np.uint8)
    lone[120:160, 140:158] = 255
    assert analyze_cell_errors(lone, bank).regions == ()


def test_a_removed_cell_leaves_no_error_in_the_empty_place(bank) -> None:
    frame = add_defect(normal_frame(seed=8), "removed", seed=2)
    result = analyze_cell_errors(frame.mask, bank)
    assert not result.class_map()[frame.defects[0].pixels].any()
    assert result.regions == ()


def test_moving_a_defective_cell_keeps_its_class_and_score(bank) -> None:
    frame = add_defect(normal_frame(seed=5), "bite", seed=4)
    pixels = frame.defects[0].pixels
    ys, xs = np.nonzero(pixels)
    piece = pixels[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    moved = np.zeros((400, 400), dtype=np.uint8)
    moved[200 : 200 + piece.shape[0], 30 : 30 + piece.shape[1]][piece] = 255
    first = analyze_cell_errors(frame.mask, bank)
    second = analyze_cell_errors(moved, bank)
    region_a = next(region for region in first.regions if region.error_class == ec.BAD_GEOMETRY)
    (region_b,) = second.regions
    assert region_b.error_class == ec.BAD_GEOMETRY
    assert region_b.score == pytest.approx(region_a.score, abs=1e-6)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("bite", ec.BAD_GEOMETRY),
        ("cut_corner", ec.BAD_GEOMETRY),
        ("short", ec.BAD_GEOMETRY),
        ("wrong_size", ec.BAD_GEOMETRY),
        ("debris", ec.DEBRIS),
        ("bridge", ec.MERGE),
        ("solid_merge", ec.MERGE),
        ("split", ec.SPLIT),
    ],
)
def test_each_defect_kind_gets_its_class(bank, kind: str, expected: int) -> None:
    hits = 0
    for seed in range(4):
        frame = add_defect(normal_frame(seed=seed), kind, seed=seed * 13 + 1)
        hits += _defect_class(frame, analyze_cell_errors(frame.mask, bank)) == expected
    assert hits >= 3


def test_jpeg_mask_gives_practically_the_same_classes(bank) -> None:
    frame = synthetic_frame({"bite": 2, "debris": 2, "bridge": 1, "split": 1}, seed=6)
    lossless = analyze_cell_errors(frame.mask, bank)
    compressed = analyze_cell_errors(frame.jpeg(80).mask, bank)
    a = _object_classes(frame.mask, lossless.class_map())
    b = _object_classes(frame.mask, compressed.class_map())
    assert np.mean(np.asarray(a) == np.asarray(b)) >= 0.95
    assert compressed.binarization["method"] == "fixed"


def test_conductor_is_ignored_kept_out_of_the_bank_and_spares_cells_next_to_it() -> None:
    frames = [add_defect(normal_frame(seed=seed), "conductor", seed=seed) for seed in range(4)]
    bank_with_grain = build_normal_bank([extract_components(frame.mask) for frame in frames])
    assert bank_with_grain.cell_width == pytest.approx(18, abs=1)
    assert bank_with_grain.cell_height == pytest.approx(40, abs=1)
    for frame in frames:
        result = analyze_cell_errors(frame.mask, bank_with_grain)
        class_map = result.class_map()
        grain = frame.ignore & (frame.mask > 0)
        assert np.mean(class_map[grain] == ec.IGNORE) > 0.9
        cells = (frame.truth > 0) & ~frame.ignore
        assert not np.any(class_map[cells] == ec.IGNORE)
        assert result.ignore_regions


def test_high_confidence_does_not_hide_a_merge_and_low_confidence_creates_nothing(bank) -> None:
    frame = add_defect(normal_frame(seed=2), "bridge", seed=9)
    certain = analyze_cell_errors(frame.mask, bank, confidence=np.full(frame.mask.shape, 255, np.uint8))
    assert _defect_class(frame, certain) == ec.MERGE

    clean = normal_frame(seed=3)
    doubtful = np.full(clean.mask.shape, 255, np.uint8)
    doubtful[clean.mask > 0] = 120
    result = analyze_cell_errors(clean.mask, bank, confidence=doubtful)
    assert result.regions == ()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_confidence_never_changes_the_class_map(bank, seed: int) -> None:
    frame = synthetic_frame({"bite": 2, "debris": 2, "bridge": 1, "split": 1, "short": 1}, seed=seed)
    rng = np.random.default_rng(seed)
    noisy = rng.integers(0, 256, size=frame.mask.shape, dtype=np.uint8)
    without = analyze_cell_errors(frame.mask, bank)
    with_confidence = analyze_cell_errors(frame.mask, bank, confidence=noisy)
    assert np.array_equal(without.class_map(), with_confidence.class_map())
    assert "confidence" in with_confidence.regions[0].evidence_sources


def test_maps_are_lossless_full_size_with_documented_dtypes(bank, tmp_path) -> None:
    frame = synthetic_frame({"bite": 1, "debris": 1, "conductor": 1}, seed=4)
    result = analyze_cell_errors(frame.mask, bank, frame_id="f1")
    assert result.class_map().dtype == np.uint8
    assert result.score_map().dtype == np.float32
    assert result.instance_map().dtype == np.uint16
    for image in (result.class_map(), result.score_map(), result.instance_map()):
        assert image.shape == frame.mask.shape
    written = export_maps(result, tmp_path, "f1", mask=frame.mask)
    class_back = cv2.imread(str(written["class_map"]), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(class_back, result.class_map())
    score_back = cv2.imread(str(written["score_map"]), cv2.IMREAD_UNCHANGED)
    assert score_back.dtype == np.uint16
    manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
    assert manifest["mode"] == MODE_MASK and manifest["binarization"]["method"]
    assert manifest["classes"]["255"] == "ignore"
    assert all(path.suffix in {".png", ".json"} for path in written.values())


def test_without_a_reliable_normal_cell_the_frame_is_not_guessed() -> None:
    rng = np.random.default_rng(0)
    crumbs = np.zeros((200, 200), dtype=np.uint8)
    for _ in range(200):
        x, y = rng.integers(2, 196, size=2)
        crumbs[y : y + 2, x : x + 2] = 255
    weak = build_normal_bank([extract_components(crumbs)])
    assert weak.status == STATUS_INSUFFICIENT
    result = analyze_cell_errors(synthetic_frame({"bite": 2}, seed=1).mask, weak)
    assert result.regions == () and result.normal_model_status == STATUS_INSUFFICIENT


def test_old_shape_of_the_result_still_feeds_the_matrix_and_old_export(bank) -> None:
    from karakal.core.cell_error_adapter import to_grid_frame_result
    from karakal.core.grid_error_export import render_grid_error_image

    frame = synthetic_frame({"bite": 2, "debris": 2, "bridge": 1, "split": 1, "conductor": 1}, seed=7)
    result = analyze_cell_errors(frame.mask, bank, frame_id="f7")
    legacy = to_grid_frame_result(result)
    reasons = {reason for cell in legacy.per_cell_results for reason in cell.reasons}
    assert {"broken_geometry", "small_artifact", "merged_contour", "split_cell", "conductor_zone"} <= reasons
    assert legacy.broken_cells == len(result.regions)
    assert legacy.cell_width == 18 and legacy.cell_height == 40
    for cell in legacy.per_cell_results:
        x, y, w, h = cell.bbox
        assert all(x - 1 <= px <= x + w and y - 1 <= py <= y + h for px, py in cell.outline)
    image = render_grid_error_image(legacy, ("broken_geometry", "small_artifact", "merged_contour"), color_mode="color")
    assert image.shape[:2] == frame.mask.shape and image.any()


def test_worker_builds_the_bank_itself_and_analyzes_frames_grid_free(tmp_path) -> None:
    from karakal.core.domain import FrameRecord
    from karakal.core.grid_anomaly import GridDamageAnalysisConfig, RunProfileRequest
    from karakal.core.performance import PerformanceConfig
    from karakal.core.workers import PairedGridInspectionWorker

    paths = []
    for seed in range(4):
        frame = synthetic_frame({"bite": 1, "debris": 1}, seed=seed)
        path = tmp_path / f"frame_{seed}.png"
        cv2.imwrite(str(path), frame.mask)
        paths.append(path)
    records = tuple(FrameRecord(f"f{index}", path.name, model_mask_paths={"m": str(path)}) for index, path in enumerate(paths))
    worker = PairedGridInspectionWorker(
        records,
        "m",
        GridDamageAnalysisConfig(scoring_mode="calibrated"),
        reference_profile=RunProfileRequest(tuple(str(path) for path in paths)),
        use_cache=False,
        performance_config=PerformanceConfig(parallel_enabled=False),
        requested_layers=("binary",),
    )
    profiles, finished = [], []
    worker.referenceProfileReady.connect(lambda model_id, profile: profiles.append((model_id, profile)))
    worker.finished.connect(finished.append)
    worker.run()
    assert profiles and profiles[0][0] == "m" and profiles[0][1].normal_bank.ready
    payloads = finished[0]
    reasons = {reason for layers in payloads.values() for cell in layers["binary"].per_cell_results for reason in cell.reasons}
    assert {"broken_geometry", "small_artifact"} <= reasons


def test_frame_set_export_writes_lossless_error_maps(tmp_path) -> None:
    from karakal.core.frame_set_export import (
        ITEM_ERROR_MAPS,
        FrameSetExportFrame,
        FrameSetExportModel,
        FrameSetExportPlan,
        export_frame_set,
        preview_frame_set_export,
    )
    from karakal.core.grid_anomaly import GridDamageAnalysisConfig, estimate_run_normal_profile

    paths = []
    for seed in range(3):
        path = tmp_path / f"frame_{seed}.png"
        cv2.imwrite(str(path), synthetic_frame({"bite": 1, "bridge": 1}, seed=seed).mask)
        paths.append(path)
    profile = estimate_run_normal_profile(paths)
    frames = tuple(FrameSetExportFrame(key=f"k{index}", name=path.name, mask_paths={"m": str(path)}) for index, path in enumerate(paths))
    plan = FrameSetExportPlan(
        set_name="set",
        layer_title="m",
        frames=frames,
        models=(FrameSetExportModel("m", "m"),),
        items=frozenset({ITEM_ERROR_MAPS}),
        normal_profiles={"m": profile},
        analysis_config=GridDamageAnalysisConfig(scoring_mode="calibrated"),
    )
    assert preview_frame_set_export(plan, tmp_path / "out").file_count == 15
    report = export_frame_set(plan, tmp_path / "out", run_name="run")
    folder = report.set_dir / "Карты ошибок"
    class_map = cv2.imdecode(np.fromfile(str(folder / "frame_0_class.png"), np.uint8), cv2.IMREAD_UNCHANGED)
    assert class_map.dtype == np.uint8 and class_map.shape == cv2.imread(str(paths[0]), 0).shape
    assert {ec.BAD_GEOMETRY, ec.MERGE} <= set(np.unique(class_map))
    manifest = json.loads((folder / "frame_0_manifest.json").read_text(encoding="utf-8"))
    assert manifest["normal_model_status"] == "ok"
    assert not report.missing and report.written == 15
