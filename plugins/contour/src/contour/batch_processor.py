from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from queue import Empty
from time import perf_counter
from typing import Any

import cv2
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from .application.processing import (
    BatchImageMetadata,
    BatchImageResult,
    ContourExtractionSettings,
    DisplaySettings,
    SaveOptions,
)
from .application.use_cases.processing import process_image_path as run_image_processing
from .batch_worker import BatchChunkRequest, BatchChunkResult, configure_worker_runtime, process_batch_chunk
from .i18n import active_language, tr
from .pipeline import PreprocessingPipeline


def configure_batch_runtime() -> None:
    cv2.setNumThreads(1)
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def process_image_path(
    image_path: str,
    pipeline_config: dict,
    contour_settings: ContourExtractionSettings,
    output_directory: str | None = None,
    save_options: SaveOptions | None = None,
    display_settings: DisplaySettings | None = None,
    pipeline: PreprocessingPipeline | None = None,
) -> BatchImageResult:
    return run_image_processing(
        image_path=image_path,
        pipeline_config=pipeline_config,
        contour_settings=contour_settings,
        output_directory=output_directory,
        save_options=save_options,
        display_settings=display_settings,
        pipeline=pipeline,
        include_images_in_result=False,
    )


class BatchQueueSignals(QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(str, str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    log = pyqtSignal(str)
    diagnostics = pyqtSignal(object)


def _batch_process_worker(
    file_queue,
    result_queue,
    cancel_event,
    pipeline_config: dict,
    contour_settings: ContourExtractionSettings,
    output_directory: str | None,
    save_options: SaveOptions,
    display_settings: DisplaySettings,
    ui_language: str,
) -> None:
    """Compatibility worker used by older tests; production batch uses chunks."""
    configure_worker_runtime()
    pipeline = PreprocessingPipeline.from_dict(pipeline_config)
    while not cancel_event.is_set():
        try:
            image_path = file_queue.get_nowait()
        except Empty:
            break

        result_queue.put(("log", tr("processing_log", language=ui_language, image_path=image_path)))
        try:
            result = process_image_path(
                image_path=image_path,
                pipeline_config=pipeline_config,
                contour_settings=contour_settings,
                output_directory=output_directory,
                save_options=save_options,
                display_settings=display_settings,
                pipeline=pipeline,
            )
            result_queue.put(("result", result))
        except Exception as exc:
            result_queue.put(("error", image_path, str(exc)))
        finally:
            result_queue.put(("progress", image_path))

    result_queue.put(("worker_done", mp.current_process().pid))


class BatchQueueRunnable(QRunnable):
    def __init__(
        self,
        image_paths: list[str],
        pipeline_config: dict,
        contour_settings: ContourExtractionSettings,
        output_directory: str | None,
        save_options: SaveOptions,
        display_settings: DisplaySettings,
        max_workers: int,
        ui_language: str,
        chunk_size: int = 16,
        progress_interval_seconds: float = 0.5,
    ) -> None:
        super().__init__()
        self.image_paths = list(image_paths)
        self.pipeline_config = dict(pipeline_config)
        self.contour_settings = replace(contour_settings)
        self.output_directory = output_directory
        self.save_options = replace(save_options)
        self.display_settings = replace(display_settings)
        self.max_workers = max(1, int(max_workers))
        self.chunk_size = max(1, int(chunk_size))
        self.ui_language = active_language(ui_language)
        self.progress_interval_seconds = max(0.1, float(progress_interval_seconds))
        self.signals = BatchQueueSignals()
        self._cancel_event: Any | None = None
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True
        if self._cancel_event is not None:
            self._cancel_event.set()

    def run(self) -> None:
        configure_batch_runtime()
        completed = 0
        total = len(self.image_paths)
        started_at = perf_counter()
        last_progress_emit = 0.0
        chunk_results: list[BatchChunkResult] = []
        futures: dict[Any, int] = {}
        chunks = self._build_chunk_requests()

        manager = None
        try:
            context = mp.get_context("spawn")
            manager = context.Manager()
            self._cancel_event = manager.Event()
            if self._stop_requested:
                self._cancel_event.set()
            worker_count = min(self.max_workers, max(1, len(chunks)))
            self.signals.log.emit(
                _diagnostic_line(
                    "batch_start",
                    process_count=worker_count,
                    chunk_size=self.chunk_size,
                    chunks=len(chunks),
                    total=total,
                )
            )
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
                initializer=configure_worker_runtime,
            ) as executor:
                pending_chunks = list(chunks)
                while pending_chunks and len(futures) < worker_count:
                    request = pending_chunks.pop(0)
                    future = executor.submit(process_batch_chunk, request, self._cancel_event)
                    futures[future] = request.chunk_id

                while futures:
                    done, _pending = wait(tuple(futures), timeout=0.2, return_when=FIRST_COMPLETED)
                    if not done:
                        now = perf_counter()
                        if now - last_progress_emit >= self.progress_interval_seconds:
                            self.signals.progress.emit(completed, total)
                            self.signals.diagnostics.emit(
                                {
                                    "type": "runtime",
                                    "completed": completed,
                                    "total": total,
                                    "active_chunks": len(futures),
                                    "queued_chunks": len(pending_chunks),
                                    "throughput_fps": completed / max(1e-6, now - started_at),
                                }
                            )
                            last_progress_emit = now
                        if self._cancel_event.is_set():
                            for future in futures:
                                future.cancel()
                            pending_chunks.clear()
                        continue

                    for future in done:
                        chunk_id = futures.pop(future)
                        try:
                            chunk_result = future.result()
                        except Exception as exc:
                            self.signals.error.emit("", f"Chunk {chunk_id} failed: {exc}")
                            chunk_result = BatchChunkResult(chunk_id=chunk_id)
                        chunk_results.append(chunk_result)
                        for item in chunk_result.metadata:
                            completed += 1
                            if item.error:
                                self.signals.error.emit(item.image_path, item.error)
                            else:
                                self.signals.result.emit(item)
                        if chunk_result.diagnostics is not None:
                            self.signals.diagnostics.emit(chunk_result.diagnostics)
                            self.signals.log.emit(_format_chunk_diagnostics(chunk_result.diagnostics))
                        now = perf_counter()
                        if now - last_progress_emit >= self.progress_interval_seconds or completed >= total:
                            self.signals.progress.emit(completed, total)
                            last_progress_emit = now
                        if not self._cancel_event.is_set() and pending_chunks:
                            request = pending_chunks.pop(0)
                            next_future = executor.submit(process_batch_chunk, request, self._cancel_event)
                            futures[next_future] = request.chunk_id

                executor.shutdown(wait=True, cancel_futures=True)
            self._emit_summary(chunk_results, completed, total, started_at)
        except Exception as exc:
            self.signals.error.emit("", str(exc))
            self.stop()
        finally:
            if manager is not None:
                with contextlib.suppress(Exception):
                    manager.shutdown()
            self.signals.finished.emit()

    def _build_chunk_requests(self) -> list[BatchChunkRequest]:
        requests: list[BatchChunkRequest] = []
        settings_payload = self.contour_settings.to_dict()
        save_payload = self.save_options.to_dict()
        display_payload = self.display_settings.to_dict()
        for chunk_id, start in enumerate(range(0, len(self.image_paths), self.chunk_size), start=1):
            requests.append(
                BatchChunkRequest(
                    chunk_id=chunk_id,
                    image_paths=tuple(self.image_paths[start : start + self.chunk_size]),
                    pipeline_config=dict(self.pipeline_config),
                    contour_settings=dict(settings_payload),
                    output_directory=self.output_directory,
                    save_options=dict(save_payload),
                    display_settings=dict(display_payload),
                )
            )
        return requests

    def _emit_summary(
        self,
        chunk_results: list[BatchChunkResult],
        completed: int,
        total: int,
        started_at: float,
    ) -> None:
        elapsed = max(1e-6, perf_counter() - started_at)
        timings = [item.timing for chunk in chunk_results for item in chunk.metadata if item.error is None]
        avg_total = sum(t.total_frame_ms for t in timings) / max(1, len(timings))
        avg_load = sum(t.image_loading_ms for t in timings) / max(1, len(timings))
        avg_contour = sum(t.contour_extraction_ms for t in timings) / max(1, len(timings))
        avg_save = sum(t.saving_ms for t in timings) / max(1, len(timings))
        utilization = [
            chunk.diagnostics.utilization
            for chunk in chunk_results
            if chunk.diagnostics is not None and chunk.diagnostics.frame_count > 0
        ]
        avg_utilization = sum(utilization) / max(1, len(utilization))
        self.signals.log.emit(
            _diagnostic_line(
                "batch_summary",
                completed=completed,
                total=total,
                throughput_fps=f"{completed / elapsed:.2f}",
                avg_frame_ms=f"{avg_total:.1f}",
                avg_load_ms=f"{avg_load:.1f}",
                avg_contour_ms=f"{avg_contour:.1f}",
                avg_save_ms=f"{avg_save:.1f}",
                avg_worker_utilization=f"{avg_utilization:.2f}",
            )
        )


class BatchProcessor(QObject):
    resultReady = pyqtSignal(object)
    progressChanged = pyqtSignal(int, int)
    finished = pyqtSignal()
    logMessage = pyqtSignal(str)
    errorOccurred = pyqtSignal(str, str)
    diagnosticsChanged = pyqtSignal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread_pool = QThreadPool(self)
        self._thread_pool.setExpiryTimeout(-1)
        self._results: dict[str, BatchImageMetadata] = {}
        self._completed = 0
        self._total = 0
        self._cancel_requested = False
        self._max_workers = 1
        self._chunk_size = _default_chunk_size()
        self._pipeline_config: dict = {}
        self._contour_settings = ContourExtractionSettings()
        self._output_directory: str | None = None
        self._save_options = SaveOptions()
        self._display_settings = DisplaySettings()
        self._ui_language = active_language()
        self._running = False
        self._runner: BatchQueueRunnable | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def set_ui_language(self, language: str | None) -> None:
        self._ui_language = active_language(language)

    def start(
        self,
        image_paths: Iterable[str],
        pipeline_config: dict,
        contour_settings: ContourExtractionSettings,
        output_directory: str | None,
        save_options: SaveOptions,
        display_settings: DisplaySettings,
        max_workers: int = 4,
        chunk_size: int | None = None,
    ) -> None:
        if self._running:
            raise RuntimeError(tr("batch_already_running_log", language=self._ui_language))

        image_list = [path for path in image_paths]
        self._results = {}
        self._completed = 0
        self._total = len(image_list)
        self._cancel_requested = False
        self._max_workers = max(1, int(max_workers))
        self._chunk_size = max(1, int(chunk_size or _default_chunk_size()))
        self._pipeline_config = dict(pipeline_config)
        self._contour_settings = replace(contour_settings)
        self._output_directory = output_directory
        self._save_options = replace(save_options)
        self._display_settings = replace(display_settings)
        self._running = True
        self._thread_pool.setMaxThreadCount(1)

        if self._total == 0:
            self.logMessage.emit(tr("batch_skipped_no_images_log", language=self._ui_language))
            self._running = False
            self.finished.emit()
            return

        self.logMessage.emit(tr("batch_started_log", language=self._ui_language, count=self._total))
        runner = BatchQueueRunnable(
            image_paths=image_list,
            pipeline_config=self._pipeline_config,
            contour_settings=self._contour_settings,
            output_directory=self._output_directory,
            save_options=self._save_options,
            display_settings=self._display_settings,
            max_workers=self._max_workers,
            chunk_size=self._chunk_size,
            ui_language=self._ui_language,
        )
        runner.signals.result.connect(self._on_worker_result)
        runner.signals.error.connect(self._on_worker_error)
        runner.signals.progress.connect(self._on_worker_progress)
        runner.signals.log.connect(self.logMessage.emit)
        runner.signals.diagnostics.connect(self.diagnosticsChanged.emit)
        runner.signals.finished.connect(self._finish)
        self._runner = runner
        self._thread_pool.start(runner)

    def stop(self) -> None:
        if not self._running:
            return
        self._cancel_requested = True
        if self._runner is not None:
            self._runner.stop()
        self.logMessage.emit(tr("batch_stop_requested_log", language=self._ui_language))

    def results(self) -> dict[str, BatchImageMetadata]:
        return dict(self._results)

    def _on_worker_result(self, result: BatchImageMetadata) -> None:
        self._results[result.image_path] = result
        self.resultReady.emit(result)

    def _on_worker_error(self, image_path: str, message: str) -> None:
        self.errorOccurred.emit(image_path, message)

    def _on_worker_progress(self, completed: int, total: int) -> None:
        self._completed = completed
        self._total = total
        self.progressChanged.emit(self._completed, self._total)

    def _finish(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._cancel_requested:
            self.logMessage.emit(tr("batch_stopped_log", language=self._ui_language))
        else:
            self.logMessage.emit(tr("batch_finished_log", language=self._ui_language))
        self._runner = None
        self.finished.emit()


def run_batch_benchmark(
    image_path: str,
    *,
    repeats: int,
    pipeline_config: dict,
    contour_settings: ContourExtractionSettings,
    output_directory: str | None,
    save_options: SaveOptions,
    display_settings: DisplaySettings,
    max_workers: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Headless benchmark helper for repeated-frame long-run stability checks."""
    configure_batch_runtime()
    repeated = [image_path for _ in range(max(1, int(repeats)))]
    context = mp.get_context("spawn")
    manager = context.Manager()
    cancel_event = manager.Event()
    chunks = [
        BatchChunkRequest(
            chunk_id=index,
            image_paths=tuple(repeated[start : start + chunk_size]),
            pipeline_config=dict(pipeline_config),
            contour_settings=contour_settings.to_dict(),
            output_directory=output_directory,
            save_options=save_options.to_dict(),
            display_settings=display_settings.to_dict(),
        )
        for index, start in enumerate(range(0, len(repeated), max(1, int(chunk_size))), start=1)
    ]
    started = perf_counter()
    try:
        with ProcessPoolExecutor(
            max_workers=max(1, int(max_workers)),
            mp_context=context,
            initializer=configure_worker_runtime,
        ) as executor:
            futures = [executor.submit(process_batch_chunk, request, cancel_event) for request in chunks]
            results = [future.result() for future in futures]
    finally:
        manager.shutdown()
    elapsed = perf_counter() - started
    frame_times = [m.timing.total_frame_ms for r in results for m in r.metadata if m.error is None]
    midpoint = max(1, len(frame_times) // 2)
    first_half = frame_times[:midpoint]
    second_half = frame_times[midpoint:]
    return {
        "frames": len(frame_times),
        "elapsed_seconds": elapsed,
        "throughput_fps": len(frame_times) / max(1e-6, elapsed),
        "avg_frame_ms": sum(frame_times) / max(1, len(frame_times)),
        "first_half_avg_ms": sum(first_half) / max(1, len(first_half)),
        "second_half_avg_ms": sum(second_half) / max(1, len(second_half)),
        "degradation_ratio": (sum(second_half) / max(1, len(second_half)))
        / max(1e-6, (sum(first_half) / max(1, len(first_half)))),
    }


def run_sequential_benchmark(
    image_path: str,
    *,
    repeats: int,
    pipeline_config: dict,
    contour_settings: ContourExtractionSettings,
    output_directory: str | None,
    save_options: SaveOptions,
    display_settings: DisplaySettings,
) -> dict[str, Any]:
    from .application.use_cases.processing import process_image_path_timed

    configure_batch_runtime()
    pipeline = PreprocessingPipeline.from_dict(pipeline_config)
    frame_times: list[float] = []
    started = perf_counter()
    for _index in range(max(1, int(repeats))):
        _result, timing = process_image_path_timed(
            image_path=image_path,
            pipeline_config=pipeline_config,
            contour_settings=contour_settings,
            output_directory=output_directory,
            save_options=save_options,
            display_settings=display_settings,
            pipeline=pipeline,
        )
        frame_times.append(float(timing.total_frame_ms))
    elapsed = perf_counter() - started
    midpoint = max(1, len(frame_times) // 2)
    first_half = frame_times[:midpoint]
    second_half = frame_times[midpoint:]
    return {
        "frames": len(frame_times),
        "elapsed_seconds": elapsed,
        "throughput_fps": len(frame_times) / max(1e-6, elapsed),
        "avg_frame_ms": sum(frame_times) / max(1, len(frame_times)),
        "first_half_avg_ms": sum(first_half) / max(1, len(first_half)),
        "second_half_avg_ms": sum(second_half) / max(1, len(second_half)),
        "degradation_ratio": (sum(second_half) / max(1, len(second_half)))
        / max(1e-6, (sum(first_half) / max(1, len(first_half)))),
    }


def _default_chunk_size() -> int:
    with contextlib.suppress(ValueError, TypeError):
        return max(1, int(os.environ.get("CONTOUR_BATCH_CHUNK_SIZE", "16")))
    return 16


def _diagnostic_line(kind: str, **fields: Any) -> str:
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    return f"[contour batch] {kind} {payload}".strip()


def _format_chunk_diagnostics(diagnostics: Any) -> str:
    rss = "n/a" if diagnostics.rss_mb is None else f"{diagnostics.rss_mb:.1f}MB"
    return _diagnostic_line(
        "chunk_done",
        chunk=diagnostics.chunk_id,
        pid=diagnostics.worker_pid,
        frames=diagnostics.frame_count,
        wall_ms=f"{diagnostics.wall_ms:.1f}",
        utilization=f"{diagnostics.utilization:.2f}",
        rss=rss,
    )
