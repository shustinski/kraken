"""benchmark_metal_segmentation: write csv."""
from __future__ import annotations
import csv
from pathlib import Path

def _write_csv(report: dict[str, object], output_path: Path) -> None:
    cases = report["cases"]
    diagnostics = report["seed_diagnostics"]
    categories = report["case_categories"]
    assert isinstance(cases, dict)
    assert isinstance(diagnostics, dict)
    assert isinstance(categories, dict)
    rows: list[dict[str, object]] = []
    for case_name, raw_strategies in cases.items():
        assert isinstance(raw_strategies, dict)
        raw_diagnostics = diagnostics[case_name]
        assert isinstance(raw_diagnostics, dict)
        for strategy, raw_metrics in raw_strategies.items():
            assert isinstance(raw_metrics, dict)
            scalar_metrics = {
                key: value for key, value in raw_metrics.items() if not isinstance(value, (dict, list, tuple))
            }
            rows.append(
                {
                    "case": case_name,
                    "category": categories[case_name],
                    "strategy": strategy,
                    "evaluation_border_crop_px": report.get("evaluation_border_crop_px", 0),
                    **scalar_metrics,
                    **raw_diagnostics,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def _format_metric(value: object, digits: int = 3) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)
