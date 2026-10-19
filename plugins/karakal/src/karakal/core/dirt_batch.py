"""Dirty regions of many frames: in worker processes, each frame kept in a cache. Qt-free.

A frame's regions change only with its mask and confidence files (path, size, time) or the
settings, so a second run over the same frames reads none of them.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from .dirt_regions import DirtConfig, FrameDirtResult, analyze_dirt_frame

DIRT_CACHE_MAX_BYTES = 256 * 1024 * 1024


class DirtCancelled(Exception):
    pass


@dataclass(frozen=True, slots=True)
class DirtFrame:
    key: str
    mask_path: str
    confidence_path: str = ""


def _file_stamp(path: str) -> list:
    if not path:
        return []
    try:
        stat = Path(path).stat()
    except OSError:
        return [str(path)]
    return [str(Path(path).resolve()), int(stat.st_size), int(stat.st_mtime_ns)]


def dirt_cache_key(frame: DirtFrame, config: DirtConfig) -> str:
    payload = {
        "config": list(config.signature()),
        "mask": _file_stamp(frame.mask_path),
        "confidence": _file_stamp(frame.confidence_path),
        "key": frame.key,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _analyze_one(frame: DirtFrame, config: DirtConfig) -> FrameDirtResult:
    return analyze_dirt_frame(
        frame.mask_path, frame_id=frame.key, confidence_path=frame.confidence_path or None, config=config
    )


def _single_threaded_opencv() -> None:
    """Pool worker start: one OpenCV thread per process, so the processes do not fight for cores."""

    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
        pass


def _analyze_batch(batch: list[tuple[int, DirtFrame]], config: DirtConfig) -> list[tuple[int, FrameDirtResult]]:
    """Process-pool job: a few frames at once (one pickling round trip for several frames)."""

    return [(index, _analyze_one(frame, config)) for index, frame in batch]


def default_cache_dir() -> Path:
    from .backend_constants import CACHE_DIR

    return CACHE_DIR / "dirt_regions"


def analyze_dirt_frames(
    frames: Sequence[DirtFrame],
    config: DirtConfig | None = None,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    workers: int | None = None,
    use_processes: bool = True,
    cache_dir: Path | str | None = "default",
    startup_timeout: float = 120.0,
) -> dict[str, FrameDirtResult]:
    """Regions of every frame by key. progress(done, total) is called as frames finish; cancelled()
    returning True stops the run (DirtCancelled)."""

    cfg = (config or DirtConfig()).normalized()
    total = len(frames)
    worker_count = max(1, int(workers or (os.cpu_count() or 4)))

    def report(done: int) -> None:
        if progress is not None:
            progress(done, total)
        if cancelled is not None and cancelled():
            raise DirtCancelled()

    cache_path = default_cache_dir() if cache_dir == "default" else (Path(cache_dir) if cache_dir else None)
    results: list[FrameDirtResult | None] = [None] * total
    keys = [""] * total
    if cache_path is not None:
        for index, frame in enumerate(frames):
            keys[index] = dirt_cache_key(frame, cfg)
            entry = cache_path / f"{keys[index]}.pickle"
            if entry.is_file():
                try:
                    with entry.open("rb") as handle:
                        cached = pickle.load(handle)
                    if isinstance(cached, FrameDirtResult):
                        results[index] = cached
                except Exception:
                    results[index] = None
    todo = [(index, frame) for index, frame in enumerate(frames) if results[index] is None]
    done = total - len(todo)
    report(done)

    def store(index: int, result: FrameDirtResult) -> None:
        results[index] = result
        if cache_path is not None and keys[index] and not result.error:
            from .cache_utils import atomic_pickle_dump

            try:
                atomic_pickle_dump(cache_path / f"{keys[index]}.pickle", result)
            except OSError:
                pass

    if todo:
        size = max(2, min(16, -(-len(todo) // (worker_count * 4))))
        batches = [todo[i : i + size] for i in range(0, len(todo), size)]
        remaining = list(batches)
        if use_processes and worker_count > 1 and len(todo) > size:
            # Processes skip the interpreter lock; a pool that cannot start (or breaks) hands the
            # frames it did not finish to the threads below.
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            pool = None
            try:
                pool = ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_single_threaded_opencv,
                )
                futures = {pool.submit(_analyze_batch, batch, cfg): batch for batch in batches}
                pending = set(futures)
                first = True
                while pending:
                    finished, pending = wait(pending, timeout=startup_timeout if first else None, return_when=FIRST_COMPLETED)
                    if not finished:
                        raise TimeoutError("dirt workers did not start")
                    first = False
                    for future in finished:
                        for index, result in future.result():
                            store(index, result)
                        remaining.remove(futures[future])
                        done += len(futures[future])
                        report(done)
            except DirtCancelled:
                if pool is not None:
                    pool.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception:
                pass
            finally:
                if pool is not None:
                    pool.shutdown(wait=False, cancel_futures=True)
        if remaining:
            with ThreadPoolExecutor(max_workers=worker_count) as threads:
                for batch_result in threads.map(lambda batch: _analyze_batch(batch, cfg), remaining):
                    for index, result in batch_result:
                        store(index, result)
                        done += 1
                    report(done)
        if cache_path is not None:
            from .cache_utils import trim_directory_by_bytes

            trim_directory_by_bytes(cache_path, max_bytes=DIRT_CACHE_MAX_BYTES)
    return {frame.key: result for frame, result in zip(frames, results) if result is not None}
