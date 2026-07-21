from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path


def _add_source_paths() -> None:
    script_path = Path(__file__).resolve()
    plugin_root = script_path.parents[1]
    workspace_root = plugin_root.parent.parent
    for path in (workspace_root / "src", plugin_root / "src"):
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)


def main() -> int:
    _add_source_paths()
    from contour.application.processing import ContourExtractionSettings, DisplaySettings, SaveOptions
    from contour.batch_processor import configure_batch_runtime, run_batch_benchmark, run_sequential_benchmark

    parser = argparse.ArgumentParser(description="Benchmark Contour batch multiprocessing throughput.")
    parser.add_argument("image", help="Image path to process repeatedly.")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()

    configure_batch_runtime()
    common = dict(
        pipeline_config={"steps": []},
        contour_settings=ContourExtractionSettings(min_area=1.0, min_perimeter=1.0),
        output_directory=None,
        save_options=SaveOptions(),
        display_settings=DisplaySettings(),
    )
    sequential = run_sequential_benchmark(
        args.image,
        repeats=args.repeats,
        **common,
    )
    multiprocessing = run_batch_benchmark(
        args.image,
        repeats=args.repeats,
        max_workers=args.workers,
        chunk_size=args.chunk_size,
        **common,
    )
    print(
        json.dumps(
            {
                "sequential_baseline": sequential,
                "multiprocessing": multiprocessing,
                "speedup": multiprocessing["throughput_fps"] / max(1e-6, sequential["throughput_fps"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
