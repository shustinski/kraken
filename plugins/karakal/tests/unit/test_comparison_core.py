from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import uuid

import numpy as np
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication

from karakal.core.domain import BuildResult, BuildOptions, ComparisonPairSelection, FrameRecord, ModelSpec
from karakal.core.repository import (
    _grayscale_array_to_qimage,
    _confidence_pair_metrics,
    _configured_combined_pair_metric_values,
    _load_cached_record_payload,
    _record_payload_cache_path,
    _store_cached_record_payload,
    available_result_layer_exports,
    combined_pair_metric_key,
    collect_frame_records,
    compute_build_result_analytics,
    export_grid_cell_defect_canvas,
    export_grid_cell_defect_bmps,
    export_result_layer_jpgs,
    export_result_layers_jpgs,
    confidence_pair_metric_key,
    load_grayscale_image,
    metric_value_for_record,
    pair_metric_key,
)
from karakal.core.grid_anomaly import GridCellAnalysisResult, GridFrameAnalysisResult, analyze_grid_frame_path
from karakal.app.main_window import _NoWheelSpinBox
from karakal.ui.matrix_view import MatrixLayoutConfig, MatrixListWidget
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


def test_record_payload_cache_concurrent_writes_do_not_corrupt_payload() -> None:
    cache_key = f"test_concurrent_{uuid.uuid4().hex}"
    cache_path = _record_payload_cache_path(cache_key)
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(_store_cached_record_payload, cache_key, {"writer": index, "values": tuple(range(index + 1))})
                for index in range(16)
            ]
            for future in futures:
                future.result()

        payload = _load_cached_record_payload(cache_key)
        assert isinstance(payload, dict)
        assert isinstance(payload.get("writer"), int)
        assert isinstance(payload.get("values"), tuple)
        assert not list(cache_path.parent.glob(f"{cache_path.name}.*.tmp"))
    finally:
        cache_path.unlink(missing_ok=True)
        for tmp_path in cache_path.parent.glob(f"{cache_path.name}.*.tmp"):
            tmp_path.unlink(missing_ok=True)


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


def test_export_grid_cell_defect_bmps_writes_white_check_mask(tmp_path) -> None:
    model_dir = tmp_path / "imported"
    export_dir = tmp_path / "export"
    model_dir.mkdir()
    export_dir.mkdir()
    source = model_dir / "frame_001.png"
    assert _grayscale_array_to_qimage(np.zeros((8, 8), dtype=np.uint8)).save(str(source), "PNG")
    record = FrameRecord(
        key="frame_001",
        display_name="frame_001",
        model_mask_paths={"A": str(source)},
    )
    build_result = BuildResult(
        records=(record,),
        model_specs=(ModelSpec(model_id="A", display_name="A", mask_folder=model_dir),),
        options=BuildOptions(),
    )
    result = GridFrameAnalysisResult(
        frame_id="frame_001",
        frame_path=str(source),
        image_width=8,
        image_height=8,
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=2,
        detected_cells=2,
        normal_cells=1,
        suspicious_cells=0,
        broken_cells=1,
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.5,
        severity_level="HIGH",
        grid_detected=True,
        per_cell_results=(
            GridCellAnalysisResult(0, 0, (0, 0, 2, 2), (1.0, 1.0), None, "normal", 0.0, ()),
            GridCellAnalysisResult(1, 0, (2, 3, 3, 2), (3.5, 4.0), None, "broken", 0.9, ("broken_geometry",)),
        ),
    )

    export = export_grid_cell_defect_bmps(build_result, {"frame_001": result}, export_dir)

    assert export["exported_count"] == 1
    assert export["extension"] == "bmp"
    exported_path = export_dir / "check_frame_bmp" / "frame_001.bmp"
    assert exported_path.is_file()
    assert not (model_dir / "check_frame_bmp" / "frame_001.bmp").exists()
    mask = load_grayscale_image(exported_path)
    assert mask.shape == (8, 8)
    assert np.all(mask[3:5, 2:5] == 255)
    assert np.count_nonzero(mask) == 6
    assert not (model_dir / "check_frame_bmp" / "export_manifest.json").exists()

    jpg_export = export_grid_cell_defect_bmps(
        build_result,
        {"frame_001": result},
        tmp_path / "export_jpg",
        image_format="jpg",
    )
    assert jpg_export["extension"] == "jpg"
    assert jpg_export["format"] == "JPG"
    assert (tmp_path / "export_jpg" / "check_frame_jpg" / "frame_001.jpg").is_file()


def test_export_grid_cell_defect_bmps_blacks_unselected_records(tmp_path) -> None:
    model_dir = tmp_path / "imported"
    export_dir = tmp_path / "export"
    model_dir.mkdir()
    export_dir.mkdir()
    source_a = model_dir / "frame_001.png"
    source_b = model_dir / "frame_002.png"
    assert _grayscale_array_to_qimage(np.zeros((8, 8), dtype=np.uint8)).save(str(source_a), "PNG")
    assert _grayscale_array_to_qimage(np.zeros((8, 8), dtype=np.uint8)).save(str(source_b), "PNG")
    record_a = FrameRecord(
        key="frame_001",
        display_name="frame_001",
        model_mask_paths={"A": str(source_a)},
    )
    record_b = FrameRecord(
        key="frame_002",
        display_name="frame_002",
        model_mask_paths={"A": str(source_b)},
    )
    build_result = BuildResult(
        records=(record_a, record_b),
        model_specs=(ModelSpec(model_id="A", display_name="A", mask_folder=model_dir),),
        options=BuildOptions(),
    )
    result_a = GridFrameAnalysisResult(
        frame_id="frame_001",
        frame_path=str(source_a),
        image_width=8,
        image_height=8,
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=1,
        detected_cells=1,
        normal_cells=0,
        suspicious_cells=0,
        broken_cells=1,
        missing_cells=0,
        artifact_cells=0,
        damage_score=1.0,
        severity_level="HIGH",
        grid_detected=True,
        per_cell_results=(GridCellAnalysisResult(0, 0, (2, 3, 3, 2), (3.5, 4.0), None, "broken", 0.9, ("broken_geometry",)),),
    )
    result_b = GridFrameAnalysisResult(
        frame_id="frame_002",
        frame_path=str(source_b),
        image_width=8,
        image_height=8,
        grid_rows=0,
        grid_cols=0,
        total_expected_cells=1,
        detected_cells=1,
        normal_cells=0,
        suspicious_cells=0,
        broken_cells=1,
        missing_cells=0,
        artifact_cells=0,
        damage_score=1.0,
        severity_level="HIGH",
        grid_detected=True,
        per_cell_results=(GridCellAnalysisResult(0, 0, (1, 1, 4, 3), (3.0, 2.5), None, "broken", 0.9, ("broken_geometry",)),),
    )

    export = export_grid_cell_defect_bmps(
        build_result,
        {"frame_001": result_a, "frame_002": result_b},
        export_dir,
        render_record_keys=("frame_001",),
        image_format="png",
    )

    assert export["exported_count"] == 2
    assert export["format"] == "PNG"
    assert export["extension"] == "png"
    selected_mask = load_grayscale_image(export_dir / "check_frame_png" / "frame_001.png")
    unselected_mask = load_grayscale_image(export_dir / "check_frame_png" / "frame_002.png")
    assert np.count_nonzero(selected_mask) == 6
    assert np.count_nonzero(unselected_mask) == 0

    jpg_export = export_grid_cell_defect_bmps(
        build_result,
        {"frame_001": result_a, "frame_002": result_b},
        tmp_path / "export_jpg_selected",
        render_record_keys=("frame_001",),
        image_format="jpg",
    )

    assert jpg_export["exported_count"] == 2
    assert jpg_export["format"] == "JPG"
    assert jpg_export["extension"] == "jpg"
    selected_jpg_mask = load_grayscale_image(tmp_path / "export_jpg_selected" / "check_frame_jpg" / "frame_001.jpg")
    unselected_jpg_mask = load_grayscale_image(tmp_path / "export_jpg_selected" / "check_frame_jpg" / "frame_002.jpg")
    assert np.count_nonzero(selected_jpg_mask) > 0
    assert np.count_nonzero(unselected_jpg_mask) == 0


def test_export_grid_cell_defect_canvas_writes_one_exact_binary_bmp(tmp_path) -> None:
    records = tuple(
        FrameRecord(key=f"frame_{index:03d}", display_name=f"frame_{index:03d}")
        for index in range(1, 4)
    )
    build_result = BuildResult(records=records, options=BuildOptions())

    def make_result(record: FrameRecord) -> GridFrameAnalysisResult:
        return GridFrameAnalysisResult(
            frame_id=record.key,
            frame_path="",
            image_width=4,
            image_height=4,
            grid_rows=1,
            grid_cols=1,
            total_expected_cells=1,
            detected_cells=1,
            normal_cells=0,
            suspicious_cells=0,
            broken_cells=1,
            missing_cells=0,
            artifact_cells=0,
            damage_score=1.0,
            severity_level="HIGH",
            grid_detected=True,
            per_cell_results=(
                GridCellAnalysisResult(0, 0, (0, 0, 4, 4), (2.0, 2.0), None, "broken", 1.0, ("broken_geometry",)),
            ),
        )

    results = {record.key: make_result(record) for record in records}
    export = export_grid_cell_defect_canvas(
        build_result,
        results,
        tmp_path,
        canvas_width=10,
        canvas_height=8,
        frames_per_row=2,
        records=records,
        render_record_keys=("frame_001", "frame_003"),
        preserve_aspect_ratio=False,
    )

    canvas_path = tmp_path / "check_matrix.bmp"
    assert export["exported_count"] == 1
    assert export["rendered_count"] == 2
    assert export["canvas_size"] == (10, 8)
    assert export["destination"] == str(canvas_path)
    canvas = load_grayscale_image(canvas_path)
    assert canvas.shape == (8, 10)
    assert set(np.unique(canvas).tolist()) == {0, 255}
    assert np.all(canvas[:4, :5] == 255)
    assert np.all(canvas[:4, 5:] == 0)
    assert np.all(canvas[4:, :5] == 255)
    assert np.all(canvas[4:, 5:] == 0)


def test_export_grid_cell_defect_canvas_preserves_frame_aspect_ratio(tmp_path) -> None:
    record = FrameRecord(key="frame_001", display_name="frame_001")
    build_result = BuildResult(records=(record,), options=BuildOptions())
    result = GridFrameAnalysisResult(
        frame_id=record.key,
        frame_path="",
        image_width=4,
        image_height=2,
        grid_rows=1,
        grid_cols=1,
        total_expected_cells=1,
        detected_cells=1,
        normal_cells=0,
        suspicious_cells=0,
        broken_cells=1,
        missing_cells=0,
        artifact_cells=0,
        damage_score=1.0,
        severity_level="HIGH",
        grid_detected=True,
        per_cell_results=(
            GridCellAnalysisResult(0, 0, (0, 0, 4, 2), (2.0, 1.0), None, "broken", 1.0, ("broken_geometry",)),
        ),
    )

    export_grid_cell_defect_canvas(
        build_result,
        {record.key: result},
        tmp_path,
        canvas_width=4,
        canvas_height=4,
        frames_per_row=1,
    )

    canvas = load_grayscale_image(tmp_path / "check_matrix.bmp")
    assert np.all(canvas[:1, :] == 0)
    assert np.all(canvas[1:3, :] == 255)
    assert np.all(canvas[3:, :] == 0)


def test_export_grid_cell_defect_canvas_renders_white_cells_with_red_errors(tmp_path) -> None:
    record = FrameRecord(key="frame_001", display_name="frame_001")
    build_result = BuildResult(records=(record,), options=BuildOptions())
    result = GridFrameAnalysisResult(
        frame_id=record.key,
        frame_path="",
        image_width=16,
        image_height=4,
        grid_rows=1,
        grid_cols=2,
        total_expected_cells=2,
        detected_cells=2,
        normal_cells=1,
        suspicious_cells=0,
        broken_cells=1,
        missing_cells=0,
        artifact_cells=0,
        damage_score=0.5,
        severity_level="HIGH",
        grid_detected=True,
        per_cell_results=(
            GridCellAnalysisResult(0, 0, (1, 0, 5, 4), (3.5, 2.0), None, "normal", 0.0, ()),
            GridCellAnalysisResult(0, 1, (10, 0, 5, 4), (12.5, 2.0), None, "broken", 1.0, ("broken_geometry",)),
        ),
    )

    export = export_grid_cell_defect_canvas(
        build_result,
        {record.key: result},
        tmp_path,
        canvas_width=16,
        canvas_height=16,
        frames_per_row=1,
        overlay_errors_on_source_mask=True,
        file_name="check_matrix_errors.bmp",
    )

    canvas_path = tmp_path / "check_matrix_errors.bmp"
    assert export["destination"] == str(canvas_path)
    canvas = QImage(str(canvas_path)).convertToFormat(QImage.Format.Format_RGB888)
    assert not canvas.isNull()
    assert canvas.pixelColor(0, 0).getRgb()[:3] == (0, 0, 0)
    assert canvas.pixelColor(1, 0).getRgb()[:3] == (255, 255, 255)
    assert canvas.pixelColor(1, 15).getRgb()[:3] == (255, 255, 255)
    assert canvas.pixelColor(10, 0).getRgb()[:3] == (255, 0, 0)
    assert canvas.pixelColor(10, 15).getRgb()[:3] == (255, 0, 0)
    assert canvas.pixelColor(15, 15).getRgb()[:3] == (0, 0, 0)


def test_collect_frame_records_ignores_nested_frames_by_default(tmp_path) -> None:
    model_a_dir = tmp_path / "model_a"
    model_b_dir = tmp_path / "model_b"
    nested_a_dir = model_a_dir / "nested"
    nested_b_dir = model_b_dir / "nested"
    nested_a_dir.mkdir(parents=True)
    nested_b_dir.mkdir(parents=True)
    frame = np.full((4, 4), 255, dtype=np.uint8)
    for folder in (model_a_dir, model_b_dir):
        assert _grayscale_array_to_qimage(frame).save(str(folder / "frame_001.png"), "PNG")
    assert _grayscale_array_to_qimage(frame).save(str(nested_a_dir / "frame_002.png"), "PNG")
    assert _grayscale_array_to_qimage(frame).save(str(nested_b_dir / "frame_002.png"), "PNG")

    result = collect_frame_records(
        (
            ModelSpec(model_id="A", display_name="A", mask_folder=model_a_dir),
            ModelSpec(model_id="B", display_name="B", mask_folder=model_b_dir),
        ),
        BuildOptions(),
    )

    assert [record.display_name for record in result.records] == ["frame_001.png"]
    assert all("nested" not in record.key for record in result.records)


def test_matrix_ctrl_range_selection_adds_to_existing_selection() -> None:
    _app = QApplication.instance() or QApplication([])
    view = MatrixListWidget()
    view.set_layout_config(MatrixLayoutConfig(total_frames=4, frames_per_row=2))
    records = tuple(FrameRecord(f"frame_{index}", f"frame_{index}.png") for index in range(4))
    view.set_records(list(records), reset_view=True)

    view._set_range_selected_records((records[0], records[1]))
    view._add_range_selected_records((records[3],))

    assert tuple(record.key for record in view.selected_records()) == ("frame_0", "frame_1", "frame_3")
    view.close()


def test_correlation_limit_spin_clamps_manual_text_to_maximum() -> None:
    _app = QApplication.instance() or QApplication([])
    spinbox = _NoWheelSpinBox()
    spinbox.setRange(1, 50)
    spinbox.setValue(10)
    spinbox.set_clamp_to_max_on_edit(True)

    spinbox.lineEdit().textEdited.emit("500")

    assert spinbox.value() == 50
    spinbox.close()


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
        options=BuildOptions(comparison_pairs=(ComparisonPairSelection("A", "B", ("xor",)),)),
    )

    result = compute_build_result_analytics(build_result, metric_key="confidence_model_score")
    values = {record.key: metric_value_for_record(record, "confidence_model_score") for record in result.records}
    confidence_pair_key = confidence_pair_metric_key("A", "B", "disagreement")
    combined_pair_key = combined_pair_metric_key("A", "B")
    confidence_pair_values = {record.key: metric_value_for_record(record, confidence_pair_key) for record in result.records}

    assert "confidence_model_score" in result.available_metric_keys
    assert confidence_pair_key in result.available_metric_keys
    assert combined_pair_key in result.available_metric_keys
    assert values["same"] is not None and values["same"] > 99.0
    assert values["diff"] is not None and values["diff"] < values["same"]
    assert confidence_pair_values["same"] == 0.0
    assert confidence_pair_values["diff"] == 1.0


def test_confidence_pair_metrics_compare_model_output_maps() -> None:
    confidence_a = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
    confidence_b = np.array([[0.1, 0.6], [0.7, 0.8]], dtype=np.float32)

    metrics = _confidence_pair_metrics(confidence_a, confidence_b)

    assert np.isclose(metrics["mae"], np.mean(np.abs(confidence_a - confidence_b)))
    assert np.isclose(metrics["rmse"], np.sqrt(np.mean(np.square(confidence_a - confidence_b))))
    assert np.isclose(metrics["mean_delta"], np.mean(confidence_a - confidence_b))
    assert metrics["disagreement"] == 0.5
    assert np.isfinite(metrics["correlation"])


def test_confidence_pair_metrics_handle_invalid_and_constant_maps() -> None:
    confidence_a = np.array([[0.8, np.nan], [np.inf, 0.8]], dtype=np.float32)
    confidence_b = np.array([[0.8, 0.1], [0.2, 0.8]], dtype=np.float32)

    metrics = _confidence_pair_metrics(confidence_a, confidence_b)

    assert metrics["mae"] == 0.0
    assert metrics["low_iou"] == 0.0
    assert np.isnan(metrics["correlation"])


def test_combined_pair_metric_blends_output_and_confidence_disagreement() -> None:
    pair = ComparisonPairSelection("A", "B", ("xor",))
    output_values = {pair_metric_key("A", "B", "xor"): 0.2}
    confidence_values = {confidence_pair_metric_key("A", "B", "disagreement"): 0.8}

    values = _configured_combined_pair_metric_values((pair,), output_values, confidence_values)

    assert values[combined_pair_metric_key("A", "B")] == 0.38


def test_combined_pair_metric_is_unavailable_without_confidence_disagreement() -> None:
    pair = ComparisonPairSelection("A", "B", ("xor",))
    output_values = {pair_metric_key("A", "B", "xor"): 0.2}

    values = _configured_combined_pair_metric_values((pair,), output_values, {})

    assert combined_pair_metric_key("A", "B") not in values


def test_grid_frame_path_loader_handles_non_ascii_paths(tmp_path) -> None:
    folder = tmp_path / "\u043a\u0430\u0434\u0440\u044b"
    folder.mkdir()
    path = folder / "frame_001.png"
    image = np.zeros((24, 32), dtype=np.uint8)
    image[4:20, 6:26] = 255
    assert _grayscale_array_to_qimage(image).save(str(path), "PNG")

    result = analyze_grid_frame_path(path, frame_id="frame_001", use_cache=False)

    assert result is not None
    assert result.image_width == 32
    assert result.image_height == 24
