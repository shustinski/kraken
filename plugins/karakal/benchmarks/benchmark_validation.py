"""Reproducible benchmark of Karakal's validation analytics on synthetic frames (the baseline of stage 0).

    python benchmarks/benchmark_validation.py [--sizes 1024,2048,4096] [--models 2,4,8] [--frames 100]
        [--methods inter_model,confidence,internal_confidence,single_risk] [--workers 1,8]
        [--max-side 1024,0] [--layer conductors|vias] [--output build/validation-benchmark/latest.json]
        [--compare BASELINE.json]

Frames come from ``core/synthetic_validation``: a chip photo, model masks with planted errors and
confidence maps, written once per frame size into ``--data-dir`` and reused (generation is not
timed). Every run is one ``compute_build_result_analytics`` call under the app's profiler in
DETAILED mode, with the disk cache off, so each run is cold.

The JSON has the stages of every run (time, self time, calls, p95 per stage), its wall and CPU
time, ms per frame and the peak working set; the top level carries ``wall_time_seconds`` and
``peak_rss_mb`` so ``performance_regression.py`` can compare two files. ``--compare`` prints the change
of every run against a baseline file. A Markdown summary is written next to the JSON.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
for path in (PLUGIN_ROOT / "src", PLUGIN_ROOT.parents[1] / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cv2  # noqa: E402

from karakal.core.analysis_modes import model_output_confidence_metric_key  # noqa: E402
from karakal.core.analytics import compute_build_result_analytics  # noqa: E402
from karakal.core.domain import BuildOptions, BuildResult, FrameRecord, GeometryMode, ModelSpec  # noqa: E402
from karakal.core.performance import PerformanceConfig, ProfilingMode  # noqa: E402
from karakal.core.profiling import ProfilerRun, activate_profiler  # noqa: E402
from karakal.core.synthetic_chip import LayerStyle, PhotoStyle  # noqa: E402
from karakal.core.synthetic_validation import (  # noqa: E402
    CONDUCTOR_ERROR_CODES,
    DATA_DEFECT_KINDS,
    POINT_ERROR_CODES,
    ConfidenceStyle,
    ModelError,
    ModelPlan,
    make_validation_frame,
)

CONFIDENCE_SUFFIX = "_confidence"
METHODS: dict[str, tuple[str, bool]] = {
    # name: (metric key, needs two or more models)
    "inter_model": ("overall_frame_score", True),
    "confidence": (model_output_confidence_metric_key("a"), False),
    "internal_confidence": ("model_confidence::a", False),
    "single_risk": ("single_result_risk_score", False),
    # The photo against model a's mask (and its confidence): core/photo_consistency.py.
    "photo": ("photo_mismatch_risk", False),
}
MODEL_IDS = "abcdefgh"


# --- Memory ----------------------------------------------------------------------------------------


def _current_rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    if os.name == "nt":

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            query = kernel32.K32GetProcessMemoryInfo
            query.argtypes = (ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong)
            query.restype = ctypes.c_int
            if query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except Exception:
            return None
        return None
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


class PeakMemory:
    """The highest working set of this process while a run lasts (sampled)."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_bytes: int | None = None
        self.peak_bytes: int | None = None

    def _sample(self) -> None:
        value = _current_rss_bytes()
        if value is not None:
            self.peak_bytes = value if self.peak_bytes is None else max(self.peak_bytes, value)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def __enter__(self) -> "PeakMemory":
        self.start_bytes = _current_rss_bytes()
        self._sample()
        self._thread = threading.Thread(target=self._run, name="validation-benchmark-memory", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()


# --- Data ------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Dataset:
    folder: Path
    size: int
    frames: tuple[str, ...]
    model_count: int
    geometry: str


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", np.ascontiguousarray(image, dtype=np.uint8))
    if not ok:
        raise OSError(f"cannot encode {path}")
    encoded.tofile(str(path))


def _frame_plans(rng: np.random.Generator, model_count: int, errors: tuple[str, ...]) -> tuple[ModelPlan, ...]:
    plans = []
    for index in range(model_count):
        planted = tuple(ModelError(str(kind)) for kind in errors if rng.random() < 0.15)
        plans.append(ModelPlan(MODEL_IDS[index], planted, ConfidenceStyle()))
    return tuple(plans)


def prepare_dataset(data_dir: Path, size: int, frames: int, model_count: int, layer_kind: str, seed: int = 0) -> Dataset:
    """Frames of one size, written once and reused while the parameters match."""

    vias = layer_kind == "vias"
    folder = data_dir / f"{layer_kind}_s{size}_m{model_count}_n{frames}_seed{seed}"
    marker = folder / "dataset.json"
    keys = tuple(f"F_{index:04d}" for index in range(frames))
    if marker.is_file():
        info = json.loads(marker.read_text(encoding="utf-8"))
        if info.get("complete"):
            return Dataset(folder, size, keys, model_count, info["geometry"])
    layer = LayerStyle(size=size, vias=vias)
    photo = PhotoStyle(light_gradient=0.2, grain=4.0, noise=5.0)
    errors = tuple(POINT_ERROR_CODES) if vias else tuple(CONDUCTOR_ERROR_CODES)
    rng = np.random.default_rng(seed)
    for sub in ("photos", *(f"model_{MODEL_IDS[index]}" for index in range(model_count))):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    for index, key in enumerate(keys):
        data_defect = str(rng.choice(DATA_DEFECT_KINDS)) if rng.random() < 0.2 else None
        try:
            frame = make_validation_frame(
                seed=seed * 100_000 + index,
                layer=layer,
                photo=photo,
                models=_frame_plans(rng, model_count, errors),
                data_defect=data_defect,
            )
        except ValueError:
            # A planted error found no place on this layout: the frame stays clean.
            frame = make_validation_frame(
                seed=seed * 100_000 + index,
                layer=layer,
                photo=photo,
                models=tuple(ModelPlan(MODEL_IDS[model]) for model in range(model_count)),
            )
        _write_png(folder / "photos" / f"{key}.png", frame.photo)
        for model_id, mask in frame.masks.items():
            _write_png(folder / f"model_{model_id}" / f"{key}.png", mask)
            _write_png(folder / f"model_{model_id}" / f"{key}{CONFIDENCE_SUFFIX}.png", frame.confidences[model_id])
    geometry = "point" if vias else "polygon"
    marker.write_text(json.dumps({"complete": True, "geometry": geometry, "size": size, "frames": frames}), encoding="utf-8")
    return Dataset(folder, size, keys, model_count, geometry)


def build_result(dataset: Dataset, model_count: int, *, workers: int, max_side: int) -> BuildResult:
    model_ids = tuple(MODEL_IDS[index] for index in range(model_count))
    records = tuple(
        FrameRecord(
            key,
            key,
            original_path=str(dataset.folder / "photos" / f"{key}.png"),
            model_mask_paths={model_id: str(dataset.folder / f"model_{model_id}" / f"{key}.png") for model_id in model_ids},
            model_prob_paths={
                model_id: str(dataset.folder / f"model_{model_id}" / f"{key}{CONFIDENCE_SUFFIX}.png")
                for model_id in model_ids
            },
        )
        for key in dataset.frames
    )
    specs = tuple(
        ModelSpec(model_id, f"Model {model_id}", dataset.folder / f"model_{model_id}", prob_folder=dataset.folder / f"model_{model_id}")
        for model_id in model_ids
    )
    options = BuildOptions(
        geometry_mode=GeometryMode.POINT if dataset.geometry == "point" else GeometryMode.MASK,
        cache_enabled=False,
        max_workers=max(1, workers),
        analysis_max_side=max(0, int(max_side)),
    )
    return BuildResult(records=records, model_specs=specs, options=options)


# --- Runs ------------------------------------------------------------------------------------------


def run_once(dataset: Dataset, method: str, model_count: int, *, workers: int, max_side: int, top_stages: int) -> dict[str, object]:
    metric_key, _needs_pair = METHODS[method]
    source = build_result(dataset, model_count, workers=workers, max_side=max_side)
    config = PerformanceConfig(profiling_mode=ProfilingMode.DETAILED)
    profiler = ProfilerRun(f"benchmark_{method}", config)
    with PeakMemory() as memory, activate_profiler(profiler):
        started_wall, started_cpu = perf_counter(), process_time()
        result = compute_build_result_analytics(source, metric_key=metric_key)
        wall, cpu = perf_counter() - started_wall, process_time() - started_cpu
    snapshot = profiler.snapshot()
    ready = sum(1 for record in result.records if record.score_ready)
    stages = [
        {
            key: stage.get(key)
            for key in ("name", "calls", "total_ms", "self_ms", "average_ms", "p95_ms", "share")
        }
        for stage in snapshot.stages[: max(1, top_stages)]
    ]
    frames = len(dataset.frames)
    return {
        "key": run_key(dataset.geometry, dataset.size, model_count, method, workers, max_side),
        "layer": dataset.geometry,
        "size": dataset.size,
        "models": model_count,
        "method": method,
        "metric_key": metric_key,
        "metric_used": result.selected_metric_key,
        "frames": frames,
        "frames_scored": ready,
        "workers": workers,
        "max_side": max_side,
        "wall_time_seconds": wall,
        "cpu_time_seconds": cpu,
        "ms_per_frame": 1000.0 * wall / max(1, frames),
        "peak_rss_mb": None if memory.peak_bytes is None else memory.peak_bytes / 1024.0**2,
        "start_rss_mb": None if memory.start_bytes is None else memory.start_bytes / 1024.0**2,
        "stages": stages,
        "counters": dict(snapshot.counters),
    }


def run_key(layer: str, size: int, models: int, method: str, workers: int, max_side: int) -> str:
    return f"{layer}/s{size}/m{models}/{method}/w{workers}/side{max_side}"


def _ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _environment() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }


def summary_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Validation benchmark",
        "",
        f"{payload['environment']}",
        "",
        "| run | frames | ms/frame | wall s | CPU s | peak MB | top stage (self ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in payload["runs"]:
        stages = run.get("stages") or []
        top = max(stages, key=lambda stage: float(stage.get("self_ms") or 0.0), default=None)
        top_text = "" if top is None else f"{top['name']} ({float(top['self_ms'] or 0.0):.0f})"
        peak = run.get("peak_rss_mb")
        lines.append(
            f"| {run['key']} | {run['frames_scored']}/{run['frames']} | {run['ms_per_frame']:.1f} | "
            f"{run['wall_time_seconds']:.2f} | {run['cpu_time_seconds']:.2f} | "
            f"{'' if peak is None else f'{peak:.0f}'} | {top_text} |"
        )
    return "\n".join(lines) + "\n"


def compare_runs(baseline: dict[str, object], candidate: dict[str, object]) -> list[str]:
    base = {run["key"]: run for run in baseline.get("runs", [])}
    lines = []
    for run in candidate.get("runs", []):
        old = base.get(run["key"])
        if old is None:
            lines.append(f"{run['key']}: new run")
            continue
        change = 100.0 * (float(run["wall_time_seconds"]) / max(1e-9, float(old["wall_time_seconds"])) - 1.0)
        lines.append(f"{run['key']}: {old['ms_per_frame']:.1f} -> {run['ms_per_frame']:.1f} ms/frame ({change:+.1f}%)")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="1024,2048,4096", help="frame sides in px")
    parser.add_argument("--models", default="2,4,8", help="model counts for the model comparison")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--methods", default=",".join(METHODS), help=f"comma-separated: {', '.join(METHODS)}")
    parser.add_argument("--workers", default=f"1,{os.cpu_count() or 4}", help="analytics workers, comma-separated")
    parser.add_argument("--max-side", default="1024", help="analysis_max_side values; 0 = full resolution")
    parser.add_argument("--layer", choices=("conductors", "vias"), default="conductors")
    parser.add_argument("--top-stages", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=1, help="runs of every setting; the one with the median time is kept")
    parser.add_argument("--data-dir", type=Path, default=PLUGIN_ROOT / "build" / "validation-benchmark" / "data")
    parser.add_argument("--output", type=Path, default=PLUGIN_ROOT / "build" / "validation-benchmark" / "latest.json")
    parser.add_argument("--compare", type=Path, help="baseline JSON to compare every run with")
    args = parser.parse_args(argv)

    methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    unknown = [name for name in methods if name not in METHODS]
    if unknown:
        parser.error(f"unknown methods: {', '.join(unknown)}")
    model_counts = sorted(set(_ints(args.models)))
    most_models = max([*model_counts, 1])
    runs: list[dict[str, object]] = []
    for size in _ints(args.sizes):
        started = perf_counter()
        dataset = prepare_dataset(args.data_dir, size, args.frames, most_models, args.layer)
        print(f"data s{size}: {dataset.folder} ({perf_counter() - started:.1f}s)", flush=True)
        for method in methods:
            _metric, needs_pair = METHODS[method]
            counts = [count for count in model_counts if count >= 2] if needs_pair else [1]
            for model_count in counts:
                for max_side in _ints(args.max_side):
                    for workers in _ints(args.workers):
                        tries = [
                            run_once(dataset, method, model_count, workers=workers, max_side=max_side, top_stages=args.top_stages)
                            for _repeat in range(max(1, args.repeats))
                        ]
                        tries.sort(key=lambda item: float(item["wall_time_seconds"]))
                        run = tries[len(tries) // 2]
                        run["repeats"] = len(tries)
                        run["wall_times_seconds"] = [float(item["wall_time_seconds"]) for item in tries]
                        runs.append(run)
                        print(
                            f"{run['key']}: {run['ms_per_frame']:.1f} ms/frame, wall {run['wall_time_seconds']:.2f}s, "
                            f"peak {run['peak_rss_mb'] or 0:.0f} MB, scored {run['frames_scored']}/{run['frames']}",
                            flush=True,
                        )
    peaks = [float(run["peak_rss_mb"]) for run in runs if run.get("peak_rss_mb") is not None]
    payload = {
        "schema_version": 1,
        "benchmark": "validation",
        "environment": _environment(),
        "settings": {
            "sizes": _ints(args.sizes),
            "models": model_counts,
            "frames": args.frames,
            "methods": methods,
            "workers": _ints(args.workers),
            "max_side": _ints(args.max_side),
            "layer": args.layer,
        },
        "runs": runs,
        "wall_time_seconds": float(sum(float(run["wall_time_seconds"]) for run in runs)),
        "peak_rss_mb": max(peaks) if peaks else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(summary_markdown(payload), encoding="utf-8")
    print(f"written {args.output}")
    if args.compare is not None:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        print("\n".join(compare_runs(baseline, payload)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
