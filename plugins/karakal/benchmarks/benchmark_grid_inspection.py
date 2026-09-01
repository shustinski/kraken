"""Reproducible benchmark for Karakal grid inspection."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from time import perf_counter, process_time
from typing import Iterable

import numpy as np

try:
    import cv2
except Exception as error:  # pragma: no cover - benchmark dependency failure
    raise SystemExit(f"OpenCV is required: {error}") from error


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PLUGIN_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from karakal.core.grid_anomaly import (  # noqa: E402
    GRID_DAMAGE_CACHE_DIR,
    GridDamageAnalysisConfig,
    GridFrameAnalysisResult,
    _load_cached_grid_result,
    _store_cached_grid_result,
    detect_grid_cell_anomalies,
)


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(slots=True)
class FrameTiming:
    frame_id: str
    path: str
    cache_hit: bool = False
    file_read_ms: float = 0.0
    decode_ms: float = 0.0
    analysis_ms: float = 0.0
    cache_read_ms: float = 0.0
    cache_write_ms: float = 0.0
    total_ms: float = 0.0
    status: str = "ok"
    damage_score: float | None = None
    defect_count: int = 0


@dataclass(slots=True)
class ResourceSample:
    timestamp: float
    rss_mb: float
    child_rss_mb: float
    cpu_percent: float


@dataclass(slots=True)
class BenchmarkResult:
    schema_version: int
    dataset: str
    implementation: str
    cache_mode: str
    frame_count: int
    unique_path_count: int
    workers: int
    chunk_size: int
    opencv_threads: int
    logical_cpu_count: int
    wall_time_seconds: float
    frames_per_second: float
    mean_frame_ms: float
    process_cpu_percent: float
    peak_rss_mb: float | None
    peak_child_rss_mb: float | None
    cache_hits: int
    failures: int
    warmup_runs: int
    measurement_runs: int
    measurement_seconds: list[float]
    stages: dict[str, dict[str, float]] = field(default_factory=dict)
    resources: list[ResourceSample] = field(default_factory=list)


class ResourceMonitor:
    def __init__(self, interval_seconds: float = 0.10) -> None:
        self._interval_seconds = max(0.02, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[ResourceSample] = []
        self._known_children: dict[int, object] = {}
        try:
            import psutil

            self._psutil = psutil
            self._process = psutil.Process()
        except Exception:
            self._psutil = None
            self._process = None

    def start(self) -> None:
        if self._process is None:
            return
        self._process.cpu_percent(None)
        self._thread = threading.Thread(target=self._run, name="grid-benchmark-resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                children = self._process.children(recursive=True)
                child_cpu = 0.0
                current_pids = set()
                for child in children:
                    pid = int(child.pid)
                    current_pids.add(pid)
                    known_child = self._known_children.get(pid)
                    if known_child is None:
                        child.cpu_percent(None)
                        self._known_children[pid] = child
                    else:
                        child_cpu += float(known_child.cpu_percent(None))
                self._known_children = {pid: child for pid, child in self._known_children.items() if pid in current_pids}
                child_rss = sum(child.memory_info().rss for child in children if child.is_running())
                self.samples.append(
                    ResourceSample(
                        timestamp=time.time(),
                        rss_mb=float(self._process.memory_info().rss / (1024.0 * 1024.0)),
                        child_rss_mb=float(child_rss / (1024.0 * 1024.0)),
                        cpu_percent=float(self._process.cpu_percent(None)) + child_cpu,
                    )
                )
            except Exception:
                continue


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _stage_summary(values: Iterable[float], total_stage_time: float) -> dict[str, float]:
    rows = [float(value) for value in values]
    stage_total = float(sum(rows))
    return {
        "count": float(len(rows)),
        "mean_ms": float(mean(rows)) if rows else 0.0,
        "median_ms": float(median(rows)) if rows else 0.0,
        "p90_ms": _percentile(rows, 90.0),
        "p95_ms": _percentile(rows, 95.0),
        "p99_ms": _percentile(rows, 99.0),
        "min_ms": float(min(rows)) if rows else 0.0,
        "max_ms": float(max(rows)) if rows else 0.0,
        "total_ms": stage_total,
        "share_percent": 100.0 * stage_total / max(1e-9, float(total_stage_time)),
    }


def _profile_frame(
    item: tuple[int, str],
    *,
    config: GridDamageAnalysisConfig,
    read_cache: bool,
    write_cache: bool,
) -> FrameTiming:
    index, path_text = item
    frame_id = f"benchmark_{index:08d}"
    path = Path(path_text)
    started = perf_counter()
    row = FrameTiming(frame_id=frame_id, path=str(path))

    if read_cache:
        stage_started = perf_counter()
        cached = _load_cached_grid_result(path, frame_id=frame_id, config=config)
        row.cache_read_ms = 1000.0 * (perf_counter() - stage_started)
        if isinstance(cached, GridFrameAnalysisResult):
            row.cache_hit = True
            row.damage_score = float(cached.damage_score)
            row.defect_count = int(cached.bad_cells)
            row.total_ms = 1000.0 * (perf_counter() - started)
            return row

    try:
        stage_started = perf_counter()
        encoded = np.fromfile(str(path), dtype=np.uint8)
        row.file_read_ms = 1000.0 * (perf_counter() - stage_started)

        stage_started = perf_counter()
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE) if encoded.size else None
        row.decode_ms = 1000.0 * (perf_counter() - stage_started)
        if image is None:
            row.status = "decode_error"
            row.total_ms = 1000.0 * (perf_counter() - started)
            return row

        stage_started = perf_counter()
        result = detect_grid_cell_anomalies(image, frame_id=frame_id, frame_path=str(path), config=config)
        row.analysis_ms = 1000.0 * (perf_counter() - stage_started)
        row.damage_score = float(result.damage_score)
        row.defect_count = int(result.bad_cells)

        if write_cache:
            stage_started = perf_counter()
            _store_cached_grid_result(path, frame_id=frame_id, config=config, result=result)
            row.cache_write_ms = 1000.0 * (perf_counter() - stage_started)
    except Exception as error:
        row.status = f"error:{type(error).__name__}"
    row.total_ms = 1000.0 * (perf_counter() - started)
    return row


def _process_initializer(opencv_threads: int) -> None:
    cv2.setNumThreads(max(1, int(opencv_threads)))


def _profile_chunk_task(
    task: tuple[list[tuple[int, str]], GridDamageAnalysisConfig, bool, bool],
) -> list[FrameTiming]:
    items, config, read_cache, write_cache = task
    return [_profile_frame(item, config=config, read_cache=read_cache, write_cache=write_cache) for item in items]


def _chunks(items: list[tuple[int, str]], chunk_size: int) -> list[list[tuple[int, str]]]:
    size = max(1, int(chunk_size))
    return [items[offset : offset + size] for offset in range(0, len(items), size)]


def _run_frames(
    items: list[tuple[int, str]],
    *,
    implementation: str,
    workers: int,
    chunk_size: int,
    opencv_threads: int,
    config: GridDamageAnalysisConfig,
    use_cache: bool,
) -> list[FrameTiming]:
    def action(item: tuple[int, str]) -> FrameTiming:
        return _profile_frame(item, config=config, read_cache=use_cache, write_cache=use_cache)

    if implementation == "sequential" or workers <= 1:
        return [action(item) for item in items]
    if implementation == "process":
        cached_rows: list[FrameTiming] = []
        cache_miss_timings: dict[str, float] = {}
        process_items = items
        if use_cache:
            process_items = []
            for index, path_text in items:
                frame_id = f"benchmark_{index:08d}"
                stage_started = perf_counter()
                cached = _load_cached_grid_result(Path(path_text), frame_id=frame_id, config=config)
                elapsed_ms = 1000.0 * (perf_counter() - stage_started)
                if isinstance(cached, GridFrameAnalysisResult):
                    cached_rows.append(
                        FrameTiming(
                            frame_id=frame_id,
                            path=path_text,
                            cache_hit=True,
                            cache_read_ms=elapsed_ms,
                            total_ms=elapsed_ms,
                            damage_score=float(cached.damage_score),
                            defect_count=int(cached.bad_cells),
                        )
                    )
                else:
                    process_items.append((index, path_text))
                    cache_miss_timings[frame_id] = elapsed_ms
        tasks = [(chunk, config, False, use_cache) for chunk in _chunks(process_items, chunk_size)]
        if not tasks:
            return cached_rows
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_process_initializer,
            initargs=(opencv_threads,),
        ) as executor:
            nested = executor.map(_profile_chunk_task, tasks)
            processed_rows = [row for rows in nested for row in rows]
        for row in processed_rows:
            cache_read_ms = cache_miss_timings.get(row.frame_id, 0.0)
            row.cache_read_ms += cache_read_ms
            row.total_ms += cache_read_ms
        return sorted(cached_rows + processed_rows, key=lambda row: row.frame_id)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="grid-benchmark") as executor:
        return list(executor.map(action, items))


def _discover_images(dataset: Path) -> list[Path]:
    return sorted(
        (path for path in dataset.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: str(path).lower(),
    )


def _generate_synthetic_dataset(directory: Path, count: int, shape: tuple[int, int]) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    height, width = shape
    rows, cols = 12, 16
    cell_w = max(8, width // (cols + 2))
    cell_h = max(8, height // (rows + 2))
    margin_x = max(4, (width - cols * cell_w) // 2)
    margin_y = max(4, (height - rows * cell_h) // 2)
    paths: list[Path] = []
    unique_count = min(max(1, count), 64)
    for index in range(unique_count):
        path = directory / f"grid_{index:03d}.png"
        if not path.is_file():
            image = np.zeros((height, width), dtype=np.uint8)
            for row in range(rows):
                for col in range(cols):
                    x = margin_x + col * cell_w
                    y = margin_y + row * cell_h
                    cv2.rectangle(image, (x + 2, y + 2), (x + cell_w - 3, y + cell_h - 3), 255, 2)
            defect = index % 8
            x = margin_x + (index * 5 % cols) * cell_w
            y = margin_y + (index * 3 % rows) * cell_h
            if defect == 1:
                cv2.rectangle(image, (x + 4, y + 4), (x + cell_w - 5, y + cell_h - 5), 255, -1)
            elif defect == 2:
                cv2.rectangle(image, (x + 4, y + cell_h // 2), (x + cell_w - 5, y + cell_h - 5), 255, -1)
            elif defect == 3:
                cv2.circle(image, (min(width - 2, x + cell_w + 5), min(height - 2, y + 5)), 3, 255, -1)
            elif defect == 4:
                cv2.line(image, (x + 2, y + 2), (x + cell_w - 3, y + cell_h - 3), 0, 3)
            elif defect == 5:
                cv2.rectangle(image, (x + cell_w - 5, y + 4), (min(width - 1, x + cell_w + 8), y + cell_h - 5), 255, -1)
            elif defect == 6:
                image[:, :] = cv2.add(image, np.full_like(image, 20))
            elif defect == 7:
                noise = np.random.default_rng(index).normal(0.0, 7.0, image.shape)
                image = np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"Unable to create synthetic frame: {path}")
        paths.append(path)
    return paths


def _build_items(paths: list[Path], limit: int) -> list[tuple[int, str]]:
    count = len(paths) if limit <= 0 else int(limit)
    return [(index, str(paths[index % len(paths)])) for index in range(count)]


def _clear_cache() -> None:
    resolved = GRID_DAMAGE_CACHE_DIR.resolve()
    if resolved.name != "grid_damage":
        raise RuntimeError(f"Refusing to clear unexpected cache directory: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)


def _write_csv(path: Path, rows: list[FrameTiming]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else list(FrameTiming.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _result_payload(result: BenchmarkResult) -> dict[str, object]:
    payload = asdict(result)
    payload["resources"] = [asdict(sample) for sample in result.resources]
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Folder containing grid frames; synthetic frames are used when omitted.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--shape", default="768x1024", help="Synthetic frame shape as HEIGHTxWIDTH.")
    parser.add_argument("--implementation", choices=("sequential", "thread", "process"), default="sequential")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=1, help="Recorded for later chunked implementation comparisons.")
    parser.add_argument("--opencv-threads", type=int, default=1)
    parser.add_argument("--cache", choices=("off", "cold", "warm"), default="off")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--clear-cache-only", action="store_true")
    parser.add_argument("--output", type=Path, default=PLUGIN_ROOT / "build" / "grid-inspection-benchmark" / "latest.json")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)

    if args.clear_cache or args.clear_cache_only or args.cache == "cold":
        _clear_cache()
    if args.clear_cache_only:
        print(f"Cleared grid inspection cache: {GRID_DAMAGE_CACHE_DIR}")
        return 0
    cv2.setNumThreads(max(1, int(args.opencv_threads)))
    try:
        height_text, width_text = str(args.shape).lower().split("x", 1)
        shape = (max(32, int(height_text)), max(32, int(width_text)))
    except Exception as error:
        raise SystemExit(f"Invalid --shape value: {args.shape}") from error

    if args.dataset is not None:
        paths = _discover_images(args.dataset)
        dataset_label = str(args.dataset.resolve())
    else:
        synthetic_dir = PLUGIN_ROOT / "build" / "grid-inspection-benchmark" / "synthetic" / f"{shape[0]}x{shape[1]}"
        paths = _generate_synthetic_dataset(synthetic_dir, max(1, int(args.limit)), shape)
        dataset_label = f"synthetic:{shape[0]}x{shape[1]}"
    if not paths:
        raise SystemExit("No supported image files found.")
    items = _build_items(paths, int(args.limit))
    config = GridDamageAnalysisConfig().normalized()

    run_kwargs = {
        "implementation": str(args.implementation),
        "workers": max(1, int(args.workers)),
        "chunk_size": max(1, int(args.chunk_size)),
        "opencv_threads": max(1, int(args.opencv_threads)),
        "config": config,
        "use_cache": args.cache != "off",
    }
    warmup_runs = max(0, int(args.warmup))
    for _run_index in range(warmup_runs):
        if args.cache == "cold":
            _clear_cache()
        _run_frames(items, **run_kwargs)

    measurements: list[tuple[float, float, list[FrameTiming], list[ResourceSample]]] = []
    for _run_index in range(max(3, int(args.repeats))):
        if args.cache == "cold":
            _clear_cache()
        monitor = ResourceMonitor()
        monitor.start()
        cpu_started = process_time()
        started = perf_counter()
        measured_rows = _run_frames(items, **run_kwargs)
        elapsed = perf_counter() - started
        cpu_elapsed = process_time() - cpu_started
        monitor.stop()
        measurements.append((elapsed, cpu_elapsed, measured_rows, monitor.samples))

    ordered_measurements = sorted(measurements, key=lambda item: item[0])
    wall_time, process_cpu_time, rows, resource_samples = ordered_measurements[len(ordered_measurements) // 2]

    total_stage_time = sum(row.total_ms for row in rows)
    stage_names = ("cache_read_ms", "file_read_ms", "decode_ms", "analysis_ms", "cache_write_ms", "total_ms")
    stages = {name.removesuffix("_ms"): _stage_summary([getattr(row, name) for row in rows], total_stage_time) for name in stage_names}
    rss_values = [sample.rss_mb for sample in resource_samples]
    child_rss_values = [sample.child_rss_mb for sample in resource_samples]
    logical_cpu_count = max(1, int(os.cpu_count() or 1))
    sampled_cpu_percent = [sample.cpu_percent / logical_cpu_count for sample in resource_samples]
    result = BenchmarkResult(
        schema_version=1,
        dataset=dataset_label,
        implementation=str(args.implementation),
        cache_mode=str(args.cache),
        frame_count=len(rows),
        unique_path_count=len({row.path for row in rows}),
        workers=max(1, int(args.workers)),
        chunk_size=max(1, int(args.chunk_size)),
        opencv_threads=int(cv2.getNumThreads()),
        logical_cpu_count=logical_cpu_count,
        wall_time_seconds=float(wall_time),
        frames_per_second=float(len(rows) / max(1e-9, wall_time)),
        mean_frame_ms=float(mean([row.total_ms for row in rows])) if rows else 0.0,
        process_cpu_percent=(
            float(mean(sampled_cpu_percent))
            if sampled_cpu_percent
            else float(100.0 * process_cpu_time / max(1e-9, wall_time * logical_cpu_count))
        ),
        peak_rss_mb=max(rss_values) if rss_values else None,
        peak_child_rss_mb=max(child_rss_values) if child_rss_values else None,
        cache_hits=sum(1 for row in rows if row.cache_hit),
        failures=sum(1 for row in rows if row.status != "ok"),
        warmup_runs=warmup_runs,
        measurement_runs=len(measurements),
        measurement_seconds=[float(item[0]) for item in measurements],
        stages=stages,
        resources=resource_samples,
    )
    payload = _result_payload(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = args.csv or args.output.with_suffix(".csv")
    _write_csv(csv_path, rows)
    print(json.dumps({key: value for key, value in payload.items() if key != "resources"}, indent=2))
    print(f"Saved JSON: {args.output}")
    print(f"Saved CSV: {csv_path}")
    return 0 if result.failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
