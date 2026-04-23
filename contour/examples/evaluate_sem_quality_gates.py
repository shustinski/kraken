from __future__ import annotations

import argparse
import json
from pathlib import Path

from polygon_widget.application.services.quality_gates import (
    SemQualityGateThresholds,
    SemQualityMetrics,
    evaluate_sem_quality_gates,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PCB/IC SEM quality gates from metrics JSON.")
    parser.add_argument("metrics_json", type=Path, help="Path to JSON with aggregated SEM metrics.")
    args = parser.parse_args()
    payload = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    metrics = SemQualityMetrics(
        via_precision=float(payload.get("via_precision", 0.0)),
        via_recall=float(payload.get("via_recall", 0.0)),
        conductor_recall=float(payload.get("conductor_recall", 0.0)),
        via_false_positives_per_frame=float(payload.get("via_false_positives_per_frame", 999.0)),
        via_center_shift_px=float(payload.get("via_center_shift_px", 999.0)),
    )
    thresholds = SemQualityGateThresholds(
        via_precision_min=float(payload.get("via_precision_min", 0.88)),
        via_recall_min=float(payload.get("via_recall_min", 0.82)),
        conductor_recall_min=float(payload.get("conductor_recall_min", 0.90)),
        via_false_positives_per_frame_max=float(payload.get("via_false_positives_per_frame_max", 1.2)),
        via_center_shift_px_max=float(payload.get("via_center_shift_px_max", 2.0)),
    )
    passed, failures = evaluate_sem_quality_gates(metrics, thresholds)
    print(f"passed={passed}")
    if failures:
        print("failed_metrics=" + ",".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
