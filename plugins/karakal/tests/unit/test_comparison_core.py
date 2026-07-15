from __future__ import annotations

import numpy as np

from karakal.core.domain import BuildResult, BuildOptions, FrameRecord, ModelSpec
from karakal.core.repository import (
    _grayscale_array_to_qimage,
    available_result_layer_exports,
    compute_build_result_analytics,
    export_result_layer_jpgs,
    export_result_layers_jpgs,
    metric_value_for_record,
)
from karakal.comparison import (
    EnsembleComparisonRequest,
    ModelFrameResult,
    PairwiseComparisonRequest,
    compare_ensemble,
    compare_pairwise,
)
from karakal.comparison.artifacts import PerModelArtifactCache
from karakal.comparison.cache import build_cache_key
from karakal.comparison.serialization import comparison_result_to_json_dict


def _model(model_id: str, mask: np.ndarray, probability: np.ndarray | None = None) -> ModelFrameResult:
    return ModelFrameResult(model_id=model_id, frame_id="f1", probability_map=probability, binary_mask=mask, metadata={})


def _metric_map(result) -> dict[str, float | int | None]:
    return {metric.name: metric.value for metric in result.frame.metrics}


def test_pairwise_identical_masks_are_perfect() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:5] = True

    result = compare_pairwise(PairwiseComparisonRequest("f1", _model("A", mask), _model("B", mask), profile="mixed"))
    metrics = _metric_map(result)

    assert metrics["dice_ab"] == 1.0
    assert metrics["iou_ab"] == 1.0
    assert metrics["disagreement_rate"] == 0.0
    assert metrics["foreground_disagreement_rate"] == 0.0
    assert result.frame.risk["total"] >= 0.0


def test_pairwise_empty_masks_do_not_nan() -> None:
    empty = np.zeros((5, 5), dtype=bool)

    result = compare_pairwise(PairwiseComparisonRequest("f1", _model("A", empty), _model("B", empty), profile="mixed"))
    metrics = _metric_map(result)

    assert metrics["dice_ab"] == 1.0
    assert metrics["iou_ab"] == 1.0
    for value in metrics.values():
        if isinstance(value, float):
            assert np.isfinite(value)


def test_pairwise_one_empty_mask_values() -> None:
    empty = np.zeros((5, 5), dtype=bool)
    filled = np.zeros((5, 5), dtype=bool)
    filled[1:3, 1:3] = True

    result = compare_pairwise(PairwiseComparisonRequest("f1", _model("A", empty), _model("B", filled), profile="polygon"))
    metrics = _metric_map(result)

    assert metrics["dice_ab"] == 0.0
    assert metrics["iou_ab"] == 0.0
    assert metrics["a_only_area"] == 0
    assert metrics["b_only_area"] == 4


def test_soft_probability_metrics_and_layers() -> None:
    mask_a = np.zeros((4, 4), dtype=bool)
    mask_b = np.zeros((4, 4), dtype=bool)
    probability_a = np.full((4, 4), 0.25, dtype=np.float32)
    probability_b = np.full((4, 4), 0.75, dtype=np.float32)

    result = compare_pairwise(
        PairwiseComparisonRequest("f1", _model("A", mask_a, probability_a), _model("B", mask_b, probability_b), profile="polygon", threshold=0.5)
    )
    metrics = _metric_map(result)
    layer_ids = {layer.layer_id for layer in result.frame.raster_layers}

    assert metrics["soft_mae_ab"] == 0.5
    assert metrics["threshold_crossing_rate"] == 1.0
    assert "soft_abs_difference" in layer_ids
    assert "threshold_crossing_map" in layer_ids


def test_component_split_event_is_created() -> None:
    mask_a = np.zeros((8, 8), dtype=bool)
    mask_a[3, 2:6] = True
    mask_b = np.zeros((8, 8), dtype=bool)
    mask_b[3, 2:3] = True
    mask_b[3, 5:6] = True

    result = compare_pairwise(PairwiseComparisonRequest("f1", _model("A", mask_a), _model("B", mask_b), profile="mixed"))
    event_types = {event.event_type for event in result.frame.events}

    assert "COMPONENT_SPLIT" in event_types or "PROBABLE_BREAK" in event_types


def test_shifted_line_skeleton_f1_at_radius_exceeds_exact_dice() -> None:
    mask_a = np.zeros((10, 10), dtype=bool)
    mask_a[5, 2:8] = True
    mask_b = np.zeros((10, 10), dtype=bool)
    mask_b[6, 2:8] = True

    result = compare_pairwise(PairwiseComparisonRequest("f1", _model("A", mask_a), _model("B", mask_b), profile="line_network"))
    metrics = _metric_map(result)

    assert metrics["skeleton_dice_ab"] < metrics["skeleton_f1_r1_ab"]
    assert metrics["skeleton_f1_r1_ab"] >= 0.9


def test_ensemble_consensus_and_outlier_score() -> None:
    base = np.zeros((6, 6), dtype=bool)
    base[2:4, 2:4] = True
    outlier = np.zeros((6, 6), dtype=bool)
    outlier[0:2, 0:2] = True

    result = compare_ensemble(
        EnsembleComparisonRequest(
            frame_id="f1",
            models=(
                _model("A", base),
                _model("B", base),
                _model("C", base),
                _model("D", outlier),
            ),
            profile="mixed",
            consensus_threshold=0.5,
        )
    )
    metrics = _metric_map(result)

    assert metrics["model_count"] == 4
    assert metrics["consensus_area"] == 4
    assert metrics["outlier_score::D"] > metrics["outlier_score::A"]
    assert result.frame.metadata["outlier_model_id"] == "D"


def test_cache_key_changes_with_threshold_and_serialization_omits_arrays() -> None:
    left = build_cache_key(
        frame_id="f1",
        model_ids=("A", "B"),
        comparison_mode="pairwise",
        profile="polygon",
        threshold=0.5,
        consensus_threshold=None,
        connectivity=8,
        pruning_threshold=5,
        evidence_provider_version="none",
    )
    right = build_cache_key(
        frame_id="f1",
        model_ids=("A", "B"),
        comparison_mode="pairwise",
        profile="polygon",
        threshold=0.6,
        consensus_threshold=None,
        connectivity=8,
        pruning_threshold=5,
        evidence_provider_version="none",
    )
    assert left != right

    mask = np.zeros((3, 3), dtype=bool)
    result = compare_pairwise(PairwiseComparisonRequest("f1", _model("A", mask), _model("B", mask), profile="polygon"))
    payload = comparison_result_to_json_dict(result.frame)

    assert payload["schema_version"] == 1
    assert payload["comparison"]["layers"]["raster"][0]["shape"] == (3, 3)
    assert "image" not in payload["comparison"]["layers"]["raster"][0]


def test_pairwise_reuses_prepared_components_and_skeleton() -> None:
    mask_a = np.zeros((12, 12), dtype=bool)
    mask_a[5, 2:10] = True
    mask_b = np.zeros((12, 12), dtype=bool)
    mask_b[6, 2:10] = True
    cache = PerModelArtifactCache()

    compare_pairwise(PairwiseComparisonRequest("f1", _model("A", mask_a), _model("B", mask_b), profile="line_network"), artifact_cache=cache)
    compare_pairwise(PairwiseComparisonRequest("f1", _model("A", mask_a), _model("B", mask_b), profile="line_network"), artifact_cache=cache)

    assert cache.hits >= 2
    assert cache.misses == 2


def test_fast_level_skips_heavy_layers() -> None:
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True

    result = compare_pairwise(
        PairwiseComparisonRequest("f1", _model("A", mask), _model("B", np.roll(mask, 1, axis=0)), profile="line_network", compute_level="fast")
    )
    layer_ids = {layer.layer_id for layer in result.frame.raster_layers}
    metric_names = {metric.name for metric in result.frame.metrics}

    assert "mask_xor" in layer_ids
    assert "skeleton_xor" not in layer_ids
    assert "skeleton_f1_r1_ab" not in metric_names


def test_ensemble_precomputes_each_model_once() -> None:
    base = np.zeros((10, 10), dtype=bool)
    base[3:7, 3:7] = True
    models = tuple(_model(f"M{index}", np.roll(base, shift=index % 2, axis=0)) for index in range(4))
    cache = PerModelArtifactCache()

    compare_ensemble(EnsembleComparisonRequest("f1", models, profile="line_network"), artifact_cache=cache)

    assert cache.misses == 4
    assert cache.hits >= 8


def test_export_result_layer_jpgs_writes_comparison_layer(tmp_path) -> None:
    model_a_dir = tmp_path / "model_a"
    model_b_dir = tmp_path / "model_b"
    model_a_dir.mkdir()
    model_b_dir.mkdir()
    mask_a = np.zeros((8, 8), dtype=np.uint8)
    mask_b = np.zeros((8, 8), dtype=np.uint8)
    mask_a[2:5, 2:5] = 255
    mask_b[3:6, 2:5] = 255
    confidence_a = np.full((8, 8), 64, dtype=np.uint8)
    confidence_b = np.full((8, 8), 192, dtype=np.uint8)
    confidence_a[2:5, 2:5] = 230
    confidence_b[3:6, 2:5] = 210
    path_a = model_a_dir / "frame_001.png"
    path_b = model_b_dir / "frame_001.png"
    confidence_path_a = model_a_dir / "frame_001_confidence.png"
    confidence_path_b = model_b_dir / "frame_001_confidence.png"
    assert _grayscale_array_to_qimage(mask_a).save(str(path_a), "PNG")
    assert _grayscale_array_to_qimage(mask_b).save(str(path_b), "PNG")
    assert _grayscale_array_to_qimage(confidence_a).save(str(confidence_path_a), "PNG")
    assert _grayscale_array_to_qimage(confidence_b).save(str(confidence_path_b), "PNG")

    build_result = BuildResult(
        records=(
            FrameRecord(
                key="sequence/frame_001",
                display_name="frame_001",
                model_mask_paths={"A": str(path_a), "B": str(path_b)},
                model_prob_paths={"A": str(confidence_path_a), "B": str(confidence_path_b)},
            ),
        ),
        model_specs=(
            ModelSpec(model_id="A", display_name="A", mask_folder=model_a_dir),
            ModelSpec(model_id="B", display_name="B", mask_folder=model_b_dir),
        ),
        options=BuildOptions(),
    )

    choices = available_result_layer_exports(build_result)
    keys = {choice["key"] for choice in choices}
    title_keys = {choice.get("title_key") for choice in choices}
    assert "result_kind::diff" in keys
    assert "result_kind::bce" in keys
    assert "result_kind::confidence_difference" in keys
    assert "result_kind::confidence_bce" in keys
    assert "result_kind::confidence_threshold_crossing" in keys
    assert "result_confidence::bad_inside::A" in keys
    assert "result_confidence::conflict::A" in keys
    assert "result_confidence::boundary_uncertainty::A" not in keys
    assert "result_confidence::transition_uncertainty::A" not in keys
    assert "details.comparison_difference" in title_keys
    assert "details.bce_heatmap" in title_keys
    assert "details.confidence_difference" in title_keys
    assert "details.result_confidence_bad_inside" in title_keys
    assert "model_mask::A" in keys
    assert "comparison::mask_xor" in keys

    progress_updates = []
    result = export_result_layer_jpgs(
        build_result,
        tmp_path / "export",
        layer_key="result_kind::bce",
        map_color=(0, 255, 0),
        progress_callback=lambda current, total, frame: progress_updates.append((current, total, frame)),
    )

    assert result["exported_count"] == 1
    assert result["skipped_count"] == 0
    assert progress_updates[0][0] == 0
    assert progress_updates[-1][0] == 1
    exported_path = next(iter(result["files"]))["destination"]
    assert exported_path.endswith(".jpg")
    assert (tmp_path / "export" / "result_layer_bce_heatmap" / "export_manifest.json").is_file()

    multi_progress_updates = []
    multi_result = export_result_layers_jpgs(
        build_result,
        tmp_path / "multi_export",
        layer_keys=(
            "result_kind::dice",
            "result_kind::diff",
            "result_kind::confidence_difference",
            "result_confidence::bad_inside::A",
        ),
        map_color=(255, 255, 255),
        progress_callback=lambda current, total, frame: multi_progress_updates.append((current, total, frame)),
    )

    assert multi_result["layer_count"] == 4
    assert multi_result["exported_count"] == 4
    assert multi_progress_updates[0][0] == 0
    assert multi_progress_updates[-1][0] == 4
    assert (tmp_path / "multi_export" / "result_layer_dice_overlap" / "export_manifest.json").is_file()
    assert (tmp_path / "multi_export" / "result_layer_comparison_difference" / "export_manifest.json").is_file()
    assert (tmp_path / "multi_export" / "result_layer_confidence_difference" / "export_manifest.json").is_file()
    assert (tmp_path / "multi_export" / "result_layer_result_confidence_bad_inside_A" / "export_manifest.json").is_file()


def test_compute_build_result_analytics_scores_confidence_folder_comparison(tmp_path) -> None:
    mask_a_dir = tmp_path / "mask_a"
    mask_b_dir = tmp_path / "mask_b"
    confidence_a_dir = tmp_path / "confidence_a"
    confidence_b_dir = tmp_path / "confidence_b"
    for folder in (mask_a_dir, mask_b_dir, confidence_a_dir, confidence_b_dir):
        folder.mkdir()

    full = np.full((32, 32), 255, dtype=np.uint8)
    low = np.full((32, 32), 32, dtype=np.uint8)
    for name in ("same.png", "diff.png"):
        assert _grayscale_array_to_qimage(full).save(str(mask_a_dir / name), "PNG")
        assert _grayscale_array_to_qimage(full).save(str(mask_b_dir / name), "PNG")
        assert _grayscale_array_to_qimage(full).save(str(confidence_a_dir / name), "PNG")
    assert _grayscale_array_to_qimage(full).save(str(confidence_b_dir / "same.png"), "PNG")
    assert _grayscale_array_to_qimage(low).save(str(confidence_b_dir / "diff.png"), "PNG")

    build_result = BuildResult(
        records=(
            FrameRecord(
                key="same",
                display_name="same.png",
                model_mask_paths={"A": str(mask_a_dir / "same.png"), "B": str(mask_b_dir / "same.png")},
                model_prob_paths={"A": str(confidence_a_dir / "same.png"), "B": str(confidence_b_dir / "same.png")},
            ),
            FrameRecord(
                key="diff",
                display_name="diff.png",
                model_mask_paths={"A": str(mask_a_dir / "diff.png"), "B": str(mask_b_dir / "diff.png")},
                model_prob_paths={"A": str(confidence_a_dir / "diff.png"), "B": str(confidence_b_dir / "diff.png")},
            ),
        ),
        model_specs=(
            ModelSpec(model_id="A", display_name="A", mask_folder=mask_a_dir, prob_folder=confidence_a_dir),
            ModelSpec(model_id="B", display_name="B", mask_folder=mask_b_dir, prob_folder=confidence_b_dir),
        ),
        options=BuildOptions(),
    )

    result = compute_build_result_analytics(build_result, metric_key="confidence_model_score")
    values = {record.key: metric_value_for_record(record, "confidence_model_score") for record in result.records}

    assert "confidence_model_score" in result.available_metric_keys
    assert values["same"] is not None and values["same"] > 99.0
    assert values["diff"] is not None and values["diff"] < values["same"]
