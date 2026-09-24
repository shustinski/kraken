import cv2
import numpy as np

from dataclasses import replace
from types import SimpleNamespace

from karakal.app.presenter import KarakalPresenter
from karakal.core.grid_anomaly import GridCellAnalysisResult, GridDamageAnalysisConfig, GridFrameAnalysisResult, detect_grid_cell_anomalies
from karakal.core.grid_calibration import GridCalibration, reference_from_normal_examples
from karakal.core.grid_scoring import (
    PARTIAL_FILL_GAP,
    CalibratedScores,
    score_debris,
    score_fill,
    score_geometry,
    score_merge,
    slider_threshold,
)


def test_slider_threshold_falls_as_sensitivity_rises() -> None:
    assert slider_threshold(0) == 1.0
    assert slider_threshold(100) == 0.0
    assert slider_threshold(60) < slider_threshold(30)


def test_scores_are_monotonic_in_the_measured_feature() -> None:
    assert score_fill(0.9, 0.2) > score_fill(0.4, 0.2)
    assert score_geometry(0.5, 0.4, 0.92, 0.74) > score_geometry(0.92, 0.74, 0.92, 0.74)
    assert score_merge(2.0, 2.0) > score_merge(1.0, 1.0)
    assert score_debris(0.1, 0.4) > score_debris(0.9, 1.2)


def test_raising_fill_slider_never_removes_a_fill() -> None:
    scores = CalibratedScores(fill=0.5, geometry=0.1, merge=0.1, debris=0.1, edge=0.0)
    low = slider_threshold(30)
    high = slider_threshold(90)
    assert "filled_cell" not in scores.reasons(fill=low, geometry=1, merge=1, debris=1, edge=1, edge_enabled=False)
    assert "filled_cell" in scores.reasons(fill=high, geometry=1, merge=1, debris=1, edge=1, edge_enabled=False)


def test_reference_cell_stays_clean_at_every_slider() -> None:
    fill = score_fill(0.20, 0.20)
    geometry = score_geometry(0.90, 0.70, 0.90, 0.70)
    merge = score_merge(1.0, 1.0)
    debris = score_debris(1.0, 1.0)
    assert max(fill, geometry, merge, debris) <= 0.02
    for slider in range(0, 101, 5):
        threshold = slider_threshold(slider)
        reasons = CalibratedScores(fill=fill, geometry=geometry, merge=merge, debris=debris, edge=0.0).reasons(
            fill=threshold,
            geometry=threshold,
            merge=threshold,
            debris=threshold,
            edge=threshold,
            edge_enabled=False,
        )
        assert reasons == ()


def test_partial_fill_sits_below_the_full_fill_threshold() -> None:
    threshold = 0.70
    partial = CalibratedScores(fill=threshold - PARTIAL_FILL_GAP / 2, geometry=0, merge=0, debris=0, edge=0)
    full = CalibratedScores(fill=threshold + 0.05, geometry=0, merge=0, debris=0, edge=0)
    assert partial.reasons(fill=threshold, geometry=1, merge=1, debris=1, edge=1, edge_enabled=False) == ("partial_filled_cell",)
    assert full.reasons(fill=threshold, geometry=1, merge=1, debris=1, edge=1, edge_enabled=False) == ("filled_cell",)


def _outline_grid() -> np.ndarray:
    image = np.zeros((360, 480), dtype=np.uint8)
    for row in range(4):
        for col in range(5):
            x, y = 24 + col * 90, 20 + row * 84
            cv2.rectangle(image, (x, y), (x + 48, y + 44), 255, 2)
    return image


def _calibrated(fill: int = 60, debris: int = 75, geometry: int = 40, merge: int = 35) -> GridDamageAnalysisConfig:
    return GridDamageAnalysisConfig(
        cell_representation="binary",
        scoring_mode="calibrated",
        blur_radius=1,
        min_contour_area=80.0,
        min_cell_size=8,
        fill_sensitivity=fill,
        debris_sensitivity=debris,
        geometry_sensitivity=geometry,
        merge_sensitivity=merge,
        min_grid_candidate_cells=4,
    )


def _reasons(image: np.ndarray, **sliders: int) -> set[str]:
    result = detect_grid_cell_anomalies(image, config=_calibrated(**sliders))
    return {reason for cell in result.cells for reason in cell.reasons}


def test_balanced_preset_finds_each_defect_and_leaves_normal_cells() -> None:
    normal = _outline_grid()
    assert _reasons(normal) == set()

    filled = _outline_grid()
    cv2.rectangle(filled, (26, 22), (70, 62), 255, -1)
    assert "filled_cell" in _reasons(filled) or "partial_filled_cell" in _reasons(filled)

    bitten = _outline_grid()
    cv2.rectangle(bitten, (22, 18), (76, 68), 0, -1)
    notch = np.array([[24, 20], [72, 20], [72, 64], [40, 64], [40, 36], [24, 36]], dtype=np.int32)
    cv2.polylines(bitten, [notch], True, 255, 2)
    assert "broken_geometry" in _reasons(bitten)

    merged = _outline_grid()
    cv2.rectangle(merged, (24, 20), (160, 64), 255, 2)
    assert "merged_contour" in _reasons(merged)

    debris = _outline_grid()
    cv2.rectangle(debris, (8, 8), (14, 14), 255, -1)
    assert "small_artifact" in _reasons(debris)


def _fill_count(image: np.ndarray, reference: dict[str, float] | None = None) -> int:
    config = _calibrated()
    if reference:
        config = GridDamageAnalysisConfig(
            cell_representation="binary",
            scoring_mode="calibrated",
            blur_radius=1,
            min_contour_area=80.0,
            min_cell_size=8,
            fill_sensitivity=60,
            debris_sensitivity=75,
            geometry_sensitivity=40,
            merge_sensitivity=35,
            min_grid_candidate_cells=4,
            calibration_reference=tuple(sorted((key, float(value)) for key, value in reference.items())),
        )
    result = detect_grid_cell_anomalies(image, config=config)
    return sum("filled_cell" in cell.reasons or "partial_filled_cell" in cell.reasons for cell in result.cells)


def test_normal_examples_find_fill_that_the_frame_median_hides() -> None:
    image = _outline_grid()
    slot = 0
    for row in range(4):
        for col in range(5):
            if slot < 12:
                x, y = 24 + col * 90, 20 + row * 84
                cv2.rectangle(image, (x + 2, y + 2), (x + 46, y + 42), 255, -1)
            slot += 1
    assert _fill_count(image) >= 8
    samples = (
        {"label": "normal", "interior_fill": 0.12, "solidity": 0.86, "extent": 0.34},
        {"label": "normal", "interior_fill": 0.10, "solidity": 0.84, "extent": 0.32},
        {"label": "filled_cell", "interior_fill": 0.95, "solidity": 0.98, "extent": 0.90},
    )
    calibration = GridCalibration().with_normal_examples(samples)
    assert calibration.reference is not None
    assert calibration.reference["interior_fill"] < 0.2
    assert _fill_count(image, calibration.reference) >= 12
    assert reference_from_normal_examples(samples) == calibration.reference


def test_majority_filled_frame_is_found_without_examples() -> None:
    image = np.zeros((420, 560), dtype=np.uint8)
    filled = 0
    for row in range(6):
        for col in range(8):
            x, y = 20 + col * 66, 16 + row * 66
            cv2.rectangle(image, (x, y), (x + 40, y + 40), 255, 2)
            if filled < 30:
                cv2.rectangle(image, (x + 3, y + 3), (x + 37, y + 37), 255, -1)
                filled += 1
    result = detect_grid_cell_anomalies(image, config=_calibrated())
    found = sum("filled_cell" in cell.reasons or "partial_filled_cell" in cell.reasons for cell in result.cells)
    assert found >= 20


def test_fill_slider_is_monotonic_on_a_frame() -> None:
    image = _outline_grid()
    cv2.rectangle(image, (26, 22), (70, 62), 255, -1)
    counts = []
    for slider in (0, 40, 70, 100):
        result = detect_grid_cell_anomalies(image, config=_calibrated(fill=slider))
        counts.append(sum("filled_cell" in cell.reasons or "partial_filled_cell" in cell.reasons for cell in result.cells))
    assert counts == sorted(counts)


def _defect_signature(image: np.ndarray, confidence: np.ndarray | None) -> tuple:
    result = detect_grid_cell_anomalies(image, config=_calibrated(), confidence_map=confidence)
    defects = tuple((cell.bbox, cell.reasons) for cell in result.cells if cell.reasons)
    return result.damage_score, defects, result


def test_confidence_does_not_change_calibrated_mask_defects() -> None:
    image = _outline_grid()
    cv2.rectangle(image, (26, 22), (70, 62), 255, -1)
    steady = np.full(image.shape, 0.95, dtype=np.float32)
    shaky = np.full(image.shape, 0.40, dtype=np.float32)
    bare_score, bare_defects, bare = _defect_signature(image, None)
    steady_score, steady_defects, _steady = _defect_signature(image, steady)
    shaky_score, shaky_defects, shaky_result = _defect_signature(image, shaky)
    assert bare_defects == steady_defects == shaky_defects
    assert bare_score == steady_score == shaky_score
    assert all(cell.mean_confidence is None for cell in bare.cells)
    assert all(cell.uncertain_pixel_ratio is None and cell.border_uncertainty is None for cell in bare.cells)
    assert any(cell.mean_confidence is not None for cell in shaky_result.cells)


def _frame(reasons: tuple[str, ...], score: float) -> GridFrameAnalysisResult:
    cell = GridCellAnalysisResult(0, 0, (1, 1, 8, 8), (5.0, 5.0), 1, "broken", score, reasons)
    return GridFrameAnalysisResult(
        frame_id="f",
        frame_path="",
        image_width=32,
        image_height=32,
        grid_rows=1,
        grid_cols=1,
        total_expected_cells=1,
        detected_cells=1,
        normal_cells=0,
        suspicious_cells=0,
        broken_cells=1,
        missing_cells=0,
        artifact_cells=0,
        damage_score=score,
        severity_level="high",
        grid_detected=True,
        per_cell_results=(cell,),
    )


def test_unified_layer_ignores_confidence_verdicts() -> None:
    host = SimpleNamespace(_grid_mismatch_enabled=False)
    merged = KarakalPresenter._merge_grid_layer_results(
        host,
        {
            "binary": _frame(("filled_cell",), 0.4),
            "confidence": _frame(("broken_geometry",), 0.9),
            "comparison": _frame(("geometry_mismatch",), 0.8),
        },
    )
    assert merged is not None
    assert merged.damage_score == 0.4
    assert {reason for cell in merged.cells for reason in cell.reasons} == {"filled_cell"}


def _growing_grid(factor: float, filled: tuple[tuple[int, int], ...] = ()) -> np.ndarray:
    image = np.zeros((420, 980), dtype=np.uint8)
    base = 28
    gap = 6
    widths = [int(round(base * (1.0 + (factor - 1.0) * col / 7))) for col in range(8)]
    height = base
    for row in range(6):
        x = 10
        y = 10 + row * (height + gap)
        for col, width in enumerate(widths):
            cv2.rectangle(image, (x, y), (x + width, y + height), 255, 2)
            if (row, col) in filled:
                cv2.rectangle(image, (x + 3, y + 3), (x + width - 3, y + height - 3), 255, -1)
            x += width + gap
    return image


def test_growing_grid_does_not_crash_and_stays_clean() -> None:
    presets = ((30, 40, 20, 15), (60, 75, 40, 35), (90, 95, 80, 85))
    for factor in (1.3, 1.6, 2.0):
        image = _growing_grid(factor)
        for fill, debris, geometry, merge in presets:
            result = detect_grid_cell_anomalies(
                image,
                config=_calibrated(fill=fill, debris=debris, geometry=geometry, merge=merge),
            )
            assert {reason for cell in result.cells for reason in cell.reasons} == set()
            assert len(result.cells) >= 48
            snapshot = dict(result.cells[0].feature_snapshot)
            assert "width" not in snapshot
            assert "width_ratio" in snapshot


def _notched_grid(scale: float = 1.0) -> np.ndarray:
    image = np.zeros((int(360 * scale) + 8, int(480 * scale) + 8), dtype=np.uint8)
    for row in range(4):
        for col in range(5):
            x = int((24 + col * 90) * scale)
            y = int((20 + row * 84) * scale)
            width = int(48 * scale)
            height = int(44 * scale)
            cv2.rectangle(image, (x, y), (x + width, y + height), 255, max(1, int(round(2 * scale))))
    if scale == 1.0:
        cv2.rectangle(image, (22, 18), (76, 68), 0, -1)
        notch = np.array([[24, 20], [72, 20], [72, 64], [40, 64], [40, 36], [24, 36]], dtype=np.int32)
    else:
        cv2.rectangle(image, (int(22 * scale), int(18 * scale)), (int(76 * scale), int(68 * scale)), 0, -1)
        notch = np.array(
            [[int(value * scale) for value in point] for point in ((24, 20), (72, 20), (72, 64), (40, 64), (40, 36), (24, 36))],
            dtype=np.int32,
        )
    cv2.polylines(image, [notch], True, 255, max(1, int(round(2 * scale))))
    return image


def test_neighbor_agreement_drops_geometry_only() -> None:
    image = _outline_grid()
    cv2.rectangle(image, (200, 110), (244, 150), 255, -1)
    cv2.rectangle(image, (22, 18), (76, 68), 0, -1)
    notch = np.array([[24, 20], [72, 20], [72, 64], [40, 64], [40, 36], [24, 36]], dtype=np.int32)
    cv2.polylines(image, [notch], True, 255, 2)
    result = detect_grid_cell_anomalies(
        image,
        config=replace(_calibrated(geometry=90), calibration_reference=(("extent", 0.20), ("solidity", 0.40))),
    )
    geometry_cells = [cell for cell in result.cells if "broken_geometry" in cell.reasons]
    assert len(geometry_cells) == 1
    assert any("filled_cell" in cell.reasons or "partial_filled_cell" in cell.reasons for cell in result.cells)


def test_normal_example_suppresses_the_same_cell_at_another_scale() -> None:
    small = detect_grid_cell_anomalies(_notched_grid(1.0), config=_calibrated(geometry=90))
    source = next(cell for cell in small.cells if "broken_geometry" in cell.reasons)
    large = _notched_grid(1.5)
    bare = detect_grid_cell_anomalies(large, config=_calibrated(geometry=90))
    assert any("broken_geometry" in cell.reasons for cell in bare.cells)
    corrected = detect_grid_cell_anomalies(
        large,
        config=replace(
            _calibrated(geometry=90),
            calibration_examples=(("normal", source.feature_snapshot),),
            example_influence=1.0,
        ),
    )
    assert not any("broken_geometry" in cell.reasons for cell in corrected.cells)


def test_ignore_example_suppresses_debris_on_another_frame() -> None:
    from karakal.app.presenter import KarakalPresenter
    from karakal.core.grid_calibration import GridCalibration, GridCellExample

    def speckled() -> np.ndarray:
        image = np.zeros((300, 900), dtype=np.uint8)
        origin_x = 20
        for width in (36, 42, 50, 60, 72, 86):
            for row in range(4):
                y = 20 + row * 60
                cv2.rectangle(image, (origin_x, y), (origin_x + width, y + 40), 255, 2)
            cv2.rectangle(image, (origin_x + width + 4, 34), (origin_x + width + 12, 42), 255, -1)
            origin_x += width + 20
        return image

    source = detect_grid_cell_anomalies(speckled(), config=_calibrated())
    speck = next(cell for cell in source.cells if "small_artifact" in cell.reasons)
    calibration = GridCalibration(
        examples=(GridCellExample(label="ignore", features=speck.feature_snapshot, frame_key="frame-1"),),
        example_influence=1.0,
    )
    payload: dict[str, object] = {}
    KarakalPresenter._put_calibration_in_payload(payload, calibration)
    from karakal.app.presenter import _example_pairs

    corrected = detect_grid_cell_anomalies(
        speckled(),
        config=replace(
            _calibrated(),
            calibration_examples=_example_pairs(payload["calibration_examples"]),
            example_influence=float(payload["example_influence"]),
        ),
    )
    assert not any("small_artifact" in cell.reasons for cell in corrected.cells)


def test_debris_between_large_cells_on_a_growing_grid() -> None:
    image = np.zeros((300, 900), dtype=np.uint8)
    origin_x = 20
    for width in (36, 42, 50, 60, 72, 86):
        for row in range(4):
            y = 20 + row * 60
            cv2.rectangle(image, (origin_x, y), (origin_x + width, y + 40), 255, 2)
        cv2.rectangle(image, (origin_x + width + 4, 34), (origin_x + width + 12, 42), 255, -1)
        origin_x += width + 20
    reasons = {
        reason
        for cell in detect_grid_cell_anomalies(image, config=_calibrated()).cells
        for reason in cell.reasons
    }
    assert "small_artifact" in reasons


def test_fill_is_found_in_small_and_large_cells() -> None:
    image = _growing_grid(1.6, filled=((0, 0), (0, 7)))
    result = detect_grid_cell_anomalies(image, config=_calibrated())
    marked = [cell for cell in result.cells if cell.reasons]
    assert len(marked) >= 2
    xs = sorted(cell.centroid[0] for cell in marked)
    assert xs[-1] - xs[0] > 200
