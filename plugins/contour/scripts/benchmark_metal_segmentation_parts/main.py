"""benchmark_metal_segmentation: main."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .module_state import (
    EVALUATION_BORDER_CROP_PX,
    REAL_DATASET_ROOT,
    STRATEGIES,
)

from .build_benchmark_cases import build_benchmark_cases

from .default_evaluation_border_crop import _default_evaluation_border_crop_px

from .markdown import _markdown

from .rasterize_polygon_labels import build_real_benchmark_cases

from .run_benchmark import run_benchmark

from .write_csv import _write_csv

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    parser.add_argument("--output", type=Path, help="write the full JSON report to this path")
    parser.add_argument("--csv", type=Path, help="write one flattened row per frame and strategy")
    parser.add_argument(
        "--suite",
        choices=("synthetic", "real"),
        default="synthetic",
        help="benchmark deterministic synthetic scenes or tests/test_metal JPG/CIF pairs",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REAL_DATASET_ROOT,
        help="root containing images/ and cif/ for --suite real",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        help="write gradient-watershed GT/prediction/seed/error overlays",
    )
    parser.add_argument(
        "--skip-presence-check",
        action="store_true",
        help="benchmark the pre-presence-rejection gradient-watershed baseline",
    )
    parser.add_argument(
        "--evaluation-stage",
        choices=("ui", "segmentation"),
        default="ui",
        help="score final UI polygons by default, or the raw segmentation mask for diagnostics",
    )
    parser.add_argument(
        "--evaluation-border-crop-px",
        type=int,
        default=None,
        help=(
            "Discard this many pixels on each side after recognition, before metrics. "
            f"Default: {EVALUATION_BORDER_CROP_PX} for --suite real, 0 for synthetic. "
            "Use 0 to restore full-frame scoring."
        ),
    )
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument(
        "--frames",
        nargs="+",
        help="benchmark only these frame ids (for example: 0175 0580 3242)",
    )
    args = parser.parse_args()
    cases = build_real_benchmark_cases(args.dataset_root) if args.suite == "real" else build_benchmark_cases()
    if args.frames:
        requested_frames = set(args.frames)
        available_frames = {case.name for case in cases}
        missing_frames = sorted(requested_frames - available_frames)
        if missing_frames:
            parser.error(f"unknown frame ids: {', '.join(missing_frames)}")
        cases = tuple(case for case in cases if case.name in requested_frames)
    report = run_benchmark(
        args.strategies,
        cases=cases,
        suite=args.suite,
        diagnostics_dir=args.diagnostics_dir,
        check_presence=not args.skip_presence_check,
        evaluation_stage=args.evaluation_stage,
        evaluation_border_crop_px=_default_evaluation_border_crop_px(
            args.suite,
            args.evaluation_border_crop_px,
        ),
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.csv is not None:
        _write_csv(report, args.csv)
    print(json.dumps(report, indent=2) if args.json else _markdown(report))
    return 0
