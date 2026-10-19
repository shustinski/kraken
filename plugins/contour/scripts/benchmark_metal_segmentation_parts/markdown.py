"""benchmark_metal_segmentation: markdown."""
from __future__ import annotations

from .module_state import BORDER_AUDIT_FRAMES

from .write_csv import _format_metric

def _markdown(report: dict[str, object]) -> str:
    aggregates = report["aggregates"]
    assert isinstance(aggregates, dict)
    crop_px = int(report.get("evaluation_border_crop_px", 0) or 0)
    lines = [
        f"evaluation_border_crop_px = {crop_px}",
        "",
        "| strategy | mean IoU | boundary F1 | component F1 | false components | missed objects | count error | merges | splits | exact topology | empty false metal | mean ms/frame |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, raw_metrics in aggregates.items():
        assert isinstance(raw_metrics, dict)
        lines.append(
            f"| {strategy} | {raw_metrics['positive_mean_iou']:.3f} | "
            f"{raw_metrics['positive_mean_boundary_f1']:.3f} | "
            f"{raw_metrics['positive_mean_component_f1']:.3f} | "
            f"{raw_metrics['false_positive_components']} | "
            f"{raw_metrics['missed_expected_components']} | "
            f"{raw_metrics['component_count_absolute_error']} | "
            f"{raw_metrics['false_merges']} | {raw_metrics['false_splits']} | "
            f"{raw_metrics['exact_topology_frames']} | "
            f"{raw_metrics['empty_false_metal_fraction']:.4f} | "
            f"{raw_metrics['mean_elapsed_ms']:.1f} |"
        )

    cases = report["cases"]
    exact_frames = report.get("exact_topology_frames", {})
    assert isinstance(cases, dict)
    strategy_names = [str(name) for name in report.get("strategies", ())]
    if len(strategy_names) == 1:
        strategy = strategy_names[0]
        lines.extend(
            [
                "",
                f"Per-frame evaluation ({strategy}):",
                "",
                "| frame | original | eval size | GT full | GT crop | pred crop | IoU | precision | recall | Boundary F1 | false | misses | merges | splits | count err | exact |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for frame_id, per_strategy in cases.items():
            assert isinstance(per_strategy, dict)
            raw_metrics = per_strategy[strategy]
            assert isinstance(raw_metrics, dict)
            original = f"{raw_metrics['original_height']}x{raw_metrics['original_width']}"
            evaluation = f"{raw_metrics['evaluation_height']}x{raw_metrics['evaluation_width']}"
            lines.append(
                f"| {frame_id} | {original} | {evaluation} | "
                f"{raw_metrics['expected_components_full_frame']} | "
                f"{raw_metrics['expected_components']} | "
                f"{raw_metrics['predicted_components']} | "
                f"{raw_metrics['iou']:.3f} | {raw_metrics['precision']:.3f} | "
                f"{raw_metrics['recall']:.3f} | {raw_metrics['boundary_f1']:.3f} | "
                f"{raw_metrics['false_positive_components']} | "
                f"{raw_metrics['missed_expected_components']} | "
                f"{raw_metrics['false_merges']} | {raw_metrics['false_splits']} | "
                f"{raw_metrics['component_count_absolute_error']} | "
                f"{'yes' if raw_metrics['topology_exact_match'] else 'no'} |"
            )
        matched = exact_frames.get(strategy, []) if isinstance(exact_frames, dict) else []
        lines.extend(["", f"Exact topology frames: {', '.join(str(item) for item in matched) or '(none)'}"])

    full_frame_aggregates = report.get("full_frame_aggregates")
    if crop_px > 0 and isinstance(full_frame_aggregates, dict) and len(strategy_names) == 1:
        strategy = strategy_names[0]
        old_metrics = full_frame_aggregates[strategy]
        new_metrics = aggregates[strategy]
        assert isinstance(old_metrics, dict)
        assert isinstance(new_metrics, dict)
        rows = (
            ("Mean IoU", "positive_mean_iou", 3),
            ("Median IoU", "positive_median_iou", 3),
            ("Boundary F1", "positive_mean_boundary_f1", 3),
            ("False components", "false_positive_components", 0),
            ("Misses", "missed_expected_components", 0),
            ("Count error", "component_count_absolute_error", 0),
            ("Merges", "false_merges", 0),
            ("Splits", "false_splits", 0),
            ("Exact topology frames", "exact_topology_frames", 0),
        )
        lines.extend(
            [
                "",
                f"Metric                     old full-frame    new crop-{crop_px}",
                "--------------------------------------------------------",
            ]
        )
        for label, key, digits in rows:
            old_value = _format_metric(old_metrics[key], digits)
            new_value = _format_metric(new_metrics[key], digits)
            lines.append(f"{label:<26} {old_value:>16} {new_value:>14}")

    border_audit = report.get("border_audit", {})
    if isinstance(border_audit, dict) and border_audit:
        lines.extend(["", "Border-frame audit:"])
        for frame_id in BORDER_AUDIT_FRAMES:
            frame_audit = border_audit.get(frame_id)
            if not isinstance(frame_audit, dict):
                continue
            for strategy, payload in frame_audit.items():
                assert isinstance(payload, dict)
                removed_gt = payload["removed_gt_components"]
                removed_pred = payload["removed_predicted_components"]
                remaining = payload["remaining"]
                assert isinstance(removed_gt, list)
                assert isinstance(removed_pred, list)
                assert isinstance(remaining, dict)
                lines.extend(
                    [
                        "",
                        f"### {frame_id} ({strategy})",
                        f"GT components removed by crop: {len(removed_gt)}",
                    ]
                )
                for record in removed_gt:
                    lines.append(f"- GT id={record['id']} area={record['area']} bbox={record['bbox_xyxy']}")
                lines.append(f"Predicted components removed by crop: {len(removed_pred)}")
                for record in removed_pred:
                    lines.append(f"- pred id={record['id']} area={record['area']} bbox={record['bbox_xyxy']}")
                lines.append(
                    "Remaining topology: "
                    f"false={remaining['false_positive_components']}, "
                    f"misses={remaining['missed_expected_components']}, "
                    f"merges={remaining['false_merges']}, "
                    f"splits={remaining['false_splits']}, "
                    f"count_error={remaining['component_count_absolute_error']}, "
                    f"pred/gt={remaining['predicted_components']}/{remaining['expected_components']}, "
                    f"exact={'yes' if remaining['topology_exact_match'] else 'no'}"
                )
    return "\n".join(lines)
