from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import cv2

from .application.processing import (
    BatchFrameTiming,
    BatchImageMetadata,
    ContourExtractionSettings,
    DisplaySettings,
    SaveOptions,
)
from .application.use_cases.processing import process_image_path_timed
from .pipeline import PreprocessingPipeline


@dataclass(frozen=True, slots=True)
class BatchChunkRequest:
    chunk_id: int
    image_paths: tuple[str, ...]
    pipeline_config: dict[str, Any]
    contour_settings: dict[str, Any]
    output_directory: str | None
    save_options: dict[str, Any]
    display_settings: dict[str, Any]


@dataclass(slots=True)
class BatchChunkDiagnostics:
    chunk_id: int
    worker_pid: int
    frame_count: int
    wall_ms: float
    busy_ms: float
    utilization: float
    rss_mb: float | None = None
    cpu_user_seconds: float | None = None
    cpu_system_seconds: float | None = None


@dataclass(slots=True)
class BatchChunkResult:
    chunk_id: int
    metadata: list[BatchImageMetadata] = field(default_factory=list)
    diagnostics: BatchChunkDiagnostics | None = None


def configure_worker_runtime() -> None:
    """Keep each process single-threaded internally.

    Multiprocessing already provides process-level parallelism; allowing OpenCV
    to spawn threads inside every worker causes CPU oversubscription and long-run
    throughput collapse on Windows.
    """
    cv2.setNumThreads(1)
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def process_batch_chunk(request: BatchChunkRequest, cancel_event: Any | None = None) -> BatchChunkResult:
    configure_worker_runtime()
    worker_pid = os.getpid()
    wall_started = perf_counter()
    busy_ms = 0.0
    metadata: list[BatchImageMetadata] = []

    process = _current_process()
    cpu_before = _cpu_times(process)
    pipeline = PreprocessingPipeline.from_dict(request.pipeline_config)
    contour_settings = ContourExtractionSettings.from_dict(request.contour_settings)
    save_options = SaveOptions.from_dict(request.save_options)
    display_settings = DisplaySettings.from_dict(request.display_settings)

    for image_path in request.image_paths:
        if cancel_event is not None and cancel_event.is_set():
            break
        frame_started = perf_counter()
        try:
            result, timing = process_image_path_timed(
                image_path=image_path,
                pipeline_config=request.pipeline_config,
                contour_settings=contour_settings,
                output_directory=request.output_directory,
                save_options=save_options,
                display_settings=display_settings,
                pipeline=pipeline,
            )
            metadata.append(
                BatchImageMetadata(
                    image_path=image_path,
                    polygon_count=len(result.polygons),
                    saved_files=dict(result.saved_files),
                    timing=timing,
                    worker_pid=worker_pid,
                )
            )
        except Exception as exc:
            elapsed = (perf_counter() - frame_started) * 1000.0
            metadata.append(
                BatchImageMetadata(
                    image_path=image_path,
                    polygon_count=0,
                    timing=BatchFrameTiming(total_frame_ms=elapsed),
                    worker_pid=worker_pid,
                    error=str(exc),
                )
            )
        finally:
            busy_ms += (perf_counter() - frame_started) * 1000.0

    gc.collect()
    wall_ms = (perf_counter() - wall_started) * 1000.0
    cpu_after = _cpu_times(process)
    diagnostics = BatchChunkDiagnostics(
        chunk_id=request.chunk_id,
        worker_pid=worker_pid,
        frame_count=len(metadata),
        wall_ms=wall_ms,
        busy_ms=busy_ms,
        utilization=busy_ms / max(1.0, wall_ms),
        rss_mb=_rss_mb(process),
        cpu_user_seconds=_cpu_delta(cpu_before, cpu_after, 0),
        cpu_system_seconds=_cpu_delta(cpu_before, cpu_after, 1),
    )
    return BatchChunkResult(chunk_id=request.chunk_id, metadata=metadata, diagnostics=diagnostics)


def _current_process() -> Any | None:
    try:
        import psutil

        return psutil.Process(os.getpid())
    except Exception:
        return None


def _cpu_times(process: Any | None) -> tuple[float, float] | None:
    if process is None:
        return None
    try:
        times = process.cpu_times()
        return float(times.user), float(times.system)
    except Exception:
        return None


def _cpu_delta(before: tuple[float, float] | None, after: tuple[float, float] | None, index: int) -> float | None:
    if before is None or after is None:
        return None
    return max(0.0, float(after[index]) - float(before[index]))


def _rss_mb(process: Any | None) -> float | None:
    if process is None:
        return None
    try:
        return float(process.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None
