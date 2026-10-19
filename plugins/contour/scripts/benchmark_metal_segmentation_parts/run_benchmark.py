"""benchmark_metal_segmentation: run benchmark."""
from __future__ import annotations
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import numpy as np
from contour.application.processing import ContourExtractionSettings  # noqa: E402
from contour.vision.metal_recovery import (  # noqa: E402
    detect_metalization,
)
from contour.vision.metal_recovery.detector import _segmentation_config_from_recovery  # noqa: E402
from contour.vision.metal_recovery.gradient_watershed import (  # noqa: E402
    GradientWatershedConfig,
    build_conductor_seeds,
    gradient_watershed_mask_from_seeds,
    selective_conductor_recovery,
)
from contour.vision.metal_recovery.pipeline_stages import (  # noqa: E402
    IMPLEMENTED_NEW_STRATEGIES,
    _segment,
    build_metal_segmentation_mask_staged,
)
from contour.vision.preprocessing import (  # noqa: E402
    metal_preprocess_config_from_settings,
)

from .module_state import (
    BORDER_AUDIT_FRAMES,
    STRATEGIES,
)

from .aggregate_strategy_metrics import _aggregate_strategy_metrics

from .build_benchmark_cases import build_benchmark_cases

from .evaluation_record import _evaluation_record

from .measure_segmentation import measure_segmentation

from .rasterize_polygon_labels import (
    components_removed_by_crop,
    evaluation_crop_slices,
    labels_for_component_audit,
    prepare_evaluation_masks,
)

from .seed_diagnostics import seed_diagnostics

from .ui_recovery_config import (
    _rasterize_detected_polygons,
    _ui_recovery_config,
    _write_diagnostic_overlay,
    resolution_probe,
)

def run_benchmark(
    strategies: Iterable[str] = STRATEGIES,
    *,
    cases: Iterable[BenchmarkCase] | None = None,
    suite: str = "synthetic",
    diagnostics_dir: Path | None = None,
    check_presence: bool = True,
    evaluation_stage: str = "ui",
    evaluation_border_crop_px: int = 0,
    strategy_parameter_overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    config = GradientWatershedConfig()
    case_results: dict[str, dict[str, dict[str, object]]] = {}
    full_frame_case_results: dict[str, dict[str, dict[str, object]]] = {}
    case_categories: dict[str, str] = {}
    seed_results: dict[str, dict[str, float | int | bool]] = {}
    border_audit: dict[str, dict[str, object]] = {}
    strategy_names = tuple(strategies)
    selected_cases = tuple(build_benchmark_cases() if cases is None else cases)
    crop_px = int(evaluation_border_crop_px)
    for case in selected_cases:
        evaluation_crop_slices(case.image.shape[:2], crop_px, frame_id=case.name)
        case_categories[case.name] = case.category
        seed_results[case.name] = asdict(seed_diagnostics(case.image, config, check_presence=check_presence))
        per_strategy: dict[str, dict[str, object]] = {}
        per_strategy_full: dict[str, dict[str, object]] = {}
        for strategy in strategy_names:
            started = perf_counter()
            predicted_labels: np.ndarray | None = None
            stage_timings_ms: dict[str, float] = {}
            if evaluation_stage == "ui":
                detection = detect_metalization(
                    case.image,
                    _ui_recovery_config(
                        strategy,
                        config,
                        None if strategy_parameter_overrides is None else strategy_parameter_overrides.get(strategy),
                    ),
                    source_image=case.source_image,
                )
                raw_stage_timings = detection.params_snapshot.get("strategy_timings_ms", {})
                if isinstance(raw_stage_timings, dict):
                    stage_timings_ms = {
                        str(name): float(value)
                        for name, value in raw_stage_timings.items()
                        if isinstance(value, (int, float))
                    }
                predicted_labels = _rasterize_detected_polygons(
                    detection.accepted,
                    case.image.shape[:2],
                )
                mask = np.where(predicted_labels > 0, 255, 0).astype(np.uint8)
                segmentation_ms = (perf_counter() - started) * 1000.0
                refinement_ms = 0.0
            elif strategy in IMPLEMENTED_NEW_STRATEGIES:
                segmentation = build_metal_segmentation_mask_staged(
                    case.image,
                    _segmentation_config_from_recovery(
                        _ui_recovery_config(
                            strategy,
                            config,
                            None
                            if strategy_parameter_overrides is None
                            else strategy_parameter_overrides.get(strategy),
                        )
                    ),
                )
                mask = segmentation.mask
                predicted_labels = segmentation.instance_labels
                stage_timings_ms = dict(segmentation.timings_ms)
                segmentation_ms = (perf_counter() - started) * 1000.0
                refinement_ms = 0.0
            elif strategy == "gradient_watershed":
                seeds = build_conductor_seeds(
                    case.image,
                    config,
                    check_presence=check_presence,
                )
                if seeds is None:
                    baseline_mask = np.zeros(case.image.shape, dtype=np.uint8)
                else:
                    baseline_mask = gradient_watershed_mask_from_seeds(case.image, seeds)
                segmentation_ms = (perf_counter() - started) * 1000.0
                refinement_started = perf_counter()
                mask = (
                    selective_conductor_recovery(baseline_mask, seeds, config)
                    if check_presence and seeds is not None
                    else baseline_mask
                )
                refinement_ms = (perf_counter() - refinement_started) * 1000.0
            else:
                mask, _polarity, _selected = _segment(
                    case.image,
                    strategy=strategy,
                    watershed_config=config,
                )
                segmentation_ms = (perf_counter() - started) * 1000.0
                refinement_ms = 0.0
            elapsed_ms = (perf_counter() - started) * 1000.0
            full_metrics = measure_segmentation(
                mask,
                case.labels,
                elapsed_ms=elapsed_ms,
                segmentation_ms=segmentation_ms,
                refinement_ms=refinement_ms,
                predicted_labels=predicted_labels,
            )
            predicted_eval, expected_eval, predicted_labels_eval = prepare_evaluation_masks(
                mask,
                case.labels,
                crop_px=crop_px,
                predicted_labels=predicted_labels,
                frame_id=case.name,
            )
            eval_metrics = (
                full_metrics
                if crop_px == 0
                else measure_segmentation(
                    predicted_eval,
                    expected_eval,
                    elapsed_ms=elapsed_ms,
                    segmentation_ms=segmentation_ms,
                    refinement_ms=refinement_ms,
                    predicted_labels=predicted_labels_eval,
                )
            )
            predicted_audit_labels = labels_for_component_audit(mask, predicted_labels)
            removed_gt = components_removed_by_crop(
                case.labels,
                crop_px,
                frame_id=case.name,
            )
            removed_predicted = components_removed_by_crop(
                predicted_audit_labels,
                crop_px,
                frame_id=case.name,
            )
            per_strategy[strategy] = _evaluation_record(
                eval_metrics,
                original_shape=case.image.shape[:2],
                evaluation_shape=predicted_eval.shape[:2],
                full_metrics=full_metrics,
                removed_gt_component_count=len(removed_gt),
                removed_predicted_component_count=len(removed_predicted),
            )
            per_strategy[strategy]["stage_timings_ms"] = stage_timings_ms
            per_strategy_full[strategy] = asdict(full_metrics)
            if case.name in BORDER_AUDIT_FRAMES:
                border_audit.setdefault(case.name, {})[strategy] = {
                    "removed_gt_components": removed_gt,
                    "removed_predicted_components": removed_predicted,
                    "remaining": {
                        "false_positive_components": eval_metrics.false_positive_components,
                        "missed_expected_components": eval_metrics.missed_expected_components,
                        "false_merges": eval_metrics.false_merges,
                        "false_splits": eval_metrics.false_splits,
                        "component_count_absolute_error": eval_metrics.component_count_absolute_error,
                        "expected_components": eval_metrics.expected_components,
                        "predicted_components": eval_metrics.predicted_components,
                        "topology_exact_match": eval_metrics.topology_exact_match,
                    },
                }
            if diagnostics_dir is not None and strategy == "gradient_watershed":
                _write_diagnostic_overlay(
                    diagnostics_dir / f"{case.name}.png",
                    case,
                    mask,
                    config,
                    check_presence=check_presence,
                )
        case_results[case.name] = per_strategy
        full_frame_case_results[case.name] = per_strategy_full

    aggregates = _aggregate_strategy_metrics(case_results, case_categories, strategy_names)
    exact_topology_frames = {
        strategy: [
            name for name, per_strategy in case_results.items() if bool(per_strategy[strategy]["topology_exact_match"])
        ]
        for strategy in strategy_names
    }
    return {
        "schema_version": 4,
        "suite": suite,
        "evaluation_stage": evaluation_stage,
        "evaluation_border_crop_px": crop_px,
        "case_count": len(case_results),
        "positive_case_count": sum(category != "empty" for category in case_categories.values()),
        "empty_case_count": sum(category == "empty" for category in case_categories.values()),
        "strategies": list(strategy_names),
        "config": asdict(config),
        "recovery_configs": {
            strategy: asdict(
                _ui_recovery_config(
                    strategy,
                    config,
                    None if strategy_parameter_overrides is None else strategy_parameter_overrides.get(strategy),
                )
            )
            for strategy in strategy_names
        },
        "preprocess_config": asdict(metal_preprocess_config_from_settings(ContourExtractionSettings())),
        "metal_presence_check": check_presence,
        "aggregates": aggregates,
        "full_frame_aggregates": _aggregate_strategy_metrics(
            full_frame_case_results,
            case_categories,
            strategy_names,
        ),
        "exact_topology_frames": exact_topology_frames,
        "border_audit": border_audit,
        "case_categories": case_categories,
        "seed_diagnostics": seed_results,
        "cases": case_results,
        "resolution_probe": resolution_probe(),
    }
