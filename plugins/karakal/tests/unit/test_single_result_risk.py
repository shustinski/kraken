from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kraken_core.analysis_protocol import AnalysisProfileKind, AnalysisScaleMode

from karakal.core.analytics import compute_build_result_analytics
from karakal.core.confidence_analysis import _model_output_confidence_metrics
from karakal.core.domain import BuildOptions, BuildResult, FrameRecord, ModelSpec
from karakal.core.exports import export_ranked_frames
from karakal.core.image_io import _grayscale_array_to_qimage
from karakal.core.single_result_risk import (
    BATCH_OUTLIER_RISK_METRIC,
    CONFIDENCE_RISK_METRIC,
    MASK_STRUCTURE_RISK_METRIC,
    SINGLE_RESULT_RISK_METRIC,
    SOURCE_ALIGNMENT_RISK_METRIC,
    compute_single_result_risk,
)
from karakal.plugin.result_adapter import build_analysis_result_manifest


def _write_gray(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _grayscale_array_to_qimage(np.ascontiguousarray(array, dtype=np.uint8))
    assert image.save(str(path))
    return path


def _normal_mask(size: int = 96, *, inset: int = 24) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[inset : size - inset, inset : size - inset] = 255
    return mask


def _noisy_mask(size: int = 96) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    for index in range(30):
        y = 2 + (index * 17) % (size - 5)
        x = 2 + (index * 29) % (size - 5)
        mask[y : y + 2, x : x + 2] = 255
    return mask


def _fragmented_mask(size: int = 96) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    for y, x in ((10, 10), (10, 58), (58, 10), (58, 58)):
        mask[y : y + 22, x : x + 22] = 255
    return mask


def _broken_mask(size: int = 96) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    for x in range(8, 80, 16):
        mask[44:51, x : x + 10] = 255
    return mask


def _build_result(
    tmp_path: Path,
    masks: list[np.ndarray],
    *,
    originals: list[np.ndarray] | None = None,
    confidences: list[np.ndarray] | None = None,
) -> BuildResult:
    mask_folder = tmp_path / "masks"
    original_folder = tmp_path / "originals"
    confidence_folder = tmp_path / "confidence"
    records: list[FrameRecord] = []
    for index, mask in enumerate(masks):
        key = f"frame_{index:03d}.png"
        mask_path = _write_gray(mask_folder / key, mask)
        original_path = None
        if originals is not None:
            original_path = _write_gray(original_folder / key, originals[index])
        confidence_path = None
        if confidences is not None:
            confidence_path = _write_gray(confidence_folder / key, confidences[index])
        records.append(
            FrameRecord(
                key=key,
                display_name=key,
                first_path=str(mask_path),
                second_path=str(mask_path),
                original_path=None if original_path is None else str(original_path),
                model_mask_paths={"model": str(mask_path)},
                model_prob_paths={"model": "" if confidence_path is None else str(confidence_path)},
            )
        )
    return BuildResult(
        records=tuple(records),
        model_specs=(
            ModelSpec(
                "model",
                "Model",
                mask_folder,
                prob_folder=confidence_folder if confidences is not None else None,
            ),
        ),
        options=BuildOptions(single_result_sensitivity="balanced"),
    )


def _risk(result: BuildResult, index: int = 0) -> float:
    summary = result.records[index].summary
    assert summary is not None
    return float(summary.metric_values[SINGLE_RESULT_RISK_METRIC])


def test_absolute_structure_checks_rank_abnormal_masks_higher(tmp_path: Path) -> None:
    normal = _normal_mask()
    empty = np.zeros_like(normal)
    full = np.full_like(normal, 255)
    fragmented = _fragmented_mask()
    noisy = _noisy_mask()
    broken = _broken_mask()

    result = compute_single_result_risk(
        _build_result(tmp_path, [normal, empty, full, fragmented, noisy, broken])
    )

    assert _risk(result, 1) == 100.0
    assert _risk(result, 2) == 100.0
    assert _risk(result, 3) > _risk(result, 0)
    assert _risk(result, 4) > _risk(result, 0)
    assert _risk(result, 5) > _risk(result, 0)
    assert result.records[1].summary.single_result_risk.reasons[0].severity == "critical"
    assert sum(result.records[1].summary.single_result_risk.component_contributions.values()) == pytest.approx(100.0)
    assert all(len(record.summary.single_result_risk.reasons) <= 8 for record in result.records)


def test_non_binary_mask_is_an_error_without_artificial_score(tmp_path: Path) -> None:
    gradient = np.arange(96, dtype=np.uint8)[None, :].repeat(96, axis=0)

    result = compute_single_result_risk(_build_result(tmp_path, [gradient]))
    record = result.records[0]

    assert not record.score_ready
    assert record.absolute_score is None
    assert record.summary is not None
    assert record.summary.notes == ("non_binary_mask",)


def test_zero_one_binary_mask_is_read_as_foreground(tmp_path: Path) -> None:
    mask = (_normal_mask() > 0).astype(np.uint8)

    result = compute_single_result_risk(_build_result(tmp_path, [mask]))

    assert result.records[0].score_ready
    assert result.records[0].summary.single_result_risk.feature_values["area_fraction"] > 0.0


def test_decode_failure_is_an_error_without_artificial_score(tmp_path: Path) -> None:
    mask_folder = tmp_path / "masks"
    mask_folder.mkdir()
    broken_path = mask_folder / "broken.png"
    broken_path.write_bytes(b"not-an-image")
    record = FrameRecord(
        "broken.png",
        "broken.png",
        model_mask_paths={"model": str(broken_path)},
    )
    build_result = BuildResult(
        records=(record,),
        model_specs=(ModelSpec("model", "Model", mask_folder),),
    )

    result = compute_single_result_risk(build_result)

    assert not result.records[0].score_ready
    assert result.records[0].absolute_score is None
    assert result.records[0].summary.notes == ("decode_error",)


def test_batch_baseline_uses_robust_statistics_and_flags_outlier(tmp_path: Path) -> None:
    masks = [_normal_mask(inset=18 + index) for index in range(10)]
    masks.append(_noisy_mask())

    result = compute_single_result_risk(_build_result(tmp_path, masks))
    normal_batch = [
        float(record.summary.metric_values[BATCH_OUTLIER_RISK_METRIC])
        for record in result.records[:-1]
        if record.summary is not None
    ]
    outlier_batch = float(result.records[-1].summary.metric_values[BATCH_OUTLIER_RISK_METRIC])

    assert outlier_batch > float(np.median(normal_batch))
    assert any(reason.code.startswith("batch_outlier.") for reason in result.records[-1].summary.single_result_risk.reasons)


def test_small_or_constant_batch_disables_batch_component_with_reason(tmp_path: Path) -> None:
    result = compute_single_result_risk(_build_result(tmp_path, [_normal_mask()] * 10))

    for record in result.records:
        assert BATCH_OUTLIER_RISK_METRIC not in record.summary.metric_values
        assert any(reason.code == "insufficient_baseline" for reason in record.summary.single_result_risk.reasons)


def test_batch_smaller_than_ten_frames_has_no_statistical_norm(tmp_path: Path) -> None:
    result = compute_single_result_risk(
        _build_result(tmp_path, [_normal_mask(inset=18 + index) for index in range(9)])
    )

    assert all(BATCH_OUTLIER_RISK_METRIC not in record.summary.metric_values for record in result.records)


def test_zero_mad_uses_iqr_when_the_feature_is_not_constant(tmp_path: Path) -> None:
    insets = [18, 20, 22, 24, 24, 24, 24, 24, 24, 30]
    result = compute_single_result_risk(
        _build_result(tmp_path, [_normal_mask(inset=inset) for inset in insets])
    )

    assert all(BATCH_OUTLIER_RISK_METRIC in record.summary.metric_values for record in result.records)


@pytest.mark.parametrize(
    ("with_original", "with_confidence", "expected_weights"),
    [
        (False, False, {MASK_STRUCTURE_RISK_METRIC: 1.0}),
        (True, False, {MASK_STRUCTURE_RISK_METRIC: 2.0 / 3.0, SOURCE_ALIGNMENT_RISK_METRIC: 1.0 / 3.0}),
        (False, True, {MASK_STRUCTURE_RISK_METRIC: 0.8, CONFIDENCE_RISK_METRIC: 0.2}),
        (
            True,
            True,
            {
                MASK_STRUCTURE_RISK_METRIC: 4.0 / 7.0,
                SOURCE_ALIGNMENT_RISK_METRIC: 2.0 / 7.0,
                CONFIDENCE_RISK_METRIC: 1.0 / 7.0,
            },
        ),
    ],
)
def test_missing_components_renormalize_weights(
    tmp_path: Path,
    with_original: bool,
    with_confidence: bool,
    expected_weights: dict[str, float],
) -> None:
    mask = _normal_mask()
    original = np.where(mask > 0, 230, 20).astype(np.uint8)
    confidence = np.full_like(mask, 245)
    result = compute_single_result_risk(
        _build_result(
            tmp_path,
            [mask],
            originals=[original] if with_original else None,
            confidences=[confidence] if with_confidence else None,
        )
    )
    summary = result.records[0].summary.single_result_risk

    assert summary is not None, result.records[0].summary.notes
    assert summary.component_weights == pytest.approx(expected_weights)
    assert sum(summary.component_weights.values()) == pytest.approx(1.0)
    assert sum(summary.component_contributions.values()) == pytest.approx(summary.total_risk)
    assert (SOURCE_ALIGNMENT_RISK_METRIC in result.available_metric_keys) is with_original
    assert (CONFIDENCE_RISK_METRIC in result.available_metric_keys) is with_confidence


def test_all_four_components_keep_nominal_weights(tmp_path: Path) -> None:
    masks = [_normal_mask(inset=inset) for inset in (18, 20, 22, 24, 24, 24, 24, 24, 24, 30)]
    originals = [np.where(mask > 0, 230, 20).astype(np.uint8) for mask in masks]
    confidences = [np.full_like(mask, 240) for mask in masks]

    result = compute_single_result_risk(
        _build_result(tmp_path, masks, originals=originals, confidences=confidences)
    )

    assert result.records[0].summary.single_result_risk.component_weights == pytest.approx(
        {
            MASK_STRUCTURE_RISK_METRIC: 0.4,
            BATCH_OUTLIER_RISK_METRIC: 0.3,
            SOURCE_ALIGNMENT_RISK_METRIC: 0.2,
            CONFIDENCE_RISK_METRIC: 0.1,
        }
    )


def test_sensitivity_is_monotonic(tmp_path: Path) -> None:
    build_result = _build_result(tmp_path, [_noisy_mask()])

    soft = _risk(compute_single_result_risk(build_result, sensitivity="soft"))
    balanced = _risk(compute_single_result_risk(build_result, sensitivity="balanced"))
    strict = _risk(compute_single_result_risk(build_result, sensitivity="strict"))

    assert soft <= balanced <= strict


def test_confidence_component_uses_frame_uncertainty_score_directly(tmp_path: Path) -> None:
    confidence = np.full((96, 96), 128, dtype=np.uint8)
    result = compute_single_result_risk(
        _build_result(tmp_path, [_normal_mask()], confidences=[confidence]),
        sensitivity="strict",
    )
    expected = 100.0 * _model_output_confidence_metrics(confidence.astype(np.float32) / 255.0).frame_uncertainty_score

    assert result.records[0].summary.metric_values[CONFIDENCE_RISK_METRIC] == pytest.approx(expected)


def test_analytics_router_uses_saved_single_result_sensitivity(tmp_path: Path) -> None:
    build_result = replace(
        _build_result(tmp_path, [_noisy_mask()]),
        options=BuildOptions(single_result_sensitivity="strict"),
    )

    result = compute_build_result_analytics(build_result, metric_key=MASK_STRUCTURE_RISK_METRIC)

    assert result.selected_metric_key == SINGLE_RESULT_RISK_METRIC
    assert result.records[0].summary.single_result_risk.sensitivity == "strict"


def test_result_contract_and_ranked_export_preserve_risk_explanations(tmp_path: Path) -> None:
    result = compute_single_result_risk(_build_result(tmp_path, [_noisy_mask()]))
    manifest = build_analysis_result_manifest(
        job_id="job-1",
        project_id="project-1",
        profile=AnalysisProfileKind.SINGLE_RESULT_RISK,
        build_result=result,
        metric_key=SINGLE_RESULT_RISK_METRIC,
        scale_mode=AnalysisScaleMode.ABSOLUTE,
    )
    metric = next(item for item in manifest.frames[0].metrics if item.key == SINGLE_RESULT_RISK_METRIC)

    assert metric.raw_value == pytest.approx(_risk(result))
    assert metric.goodness == pytest.approx(1.0 - metric.raw_value / 100.0)
    assert not metric.higher_is_better
    assert manifest.scales[0].low == 0.0
    assert manifest.scales[0].high == 100.0
    assert manifest.frames[0].anomalies

    export_result = export_ranked_frames(
        result,
        tmp_path / "export",
        top_k=1,
        neighbor_radius=0,
        metric_key=SINGLE_RESULT_RISK_METRIC,
    )
    payload = json.loads(Path(export_result["manifest_path"]).read_text(encoding="utf-8"))
    risk_payload = payload["single_result_risk"][result.records[0].key]
    assert risk_payload["risk"] == pytest.approx(_risk(result))
    assert risk_payload["component_contributions"]
    assert risk_payload["reasons"]


def test_single_result_engine_requires_exactly_one_model(tmp_path: Path) -> None:
    empty_result = BuildResult()
    two_models = replace(
        _build_result(tmp_path, [_normal_mask()]),
        model_specs=(
            ModelSpec("a", "A", tmp_path),
            ModelSpec("b", "B", tmp_path),
        ),
    )

    with pytest.raises(ValueError, match="exactly one"):
        compute_single_result_risk(empty_result)
    with pytest.raises(ValueError, match="exactly one"):
        compute_single_result_risk(two_models)
