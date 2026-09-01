"""Background Qt workers for the extended validation gradient widget."""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from time import monotonic
from threading import Event
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from PyQt6.QtCore import QObject, pyqtSignal

from .domain import BuildOptions, BuildResult, FolderSpec, FrameRecord, ModelSpec
from .grid_anomaly import (
    GridDamageAnalysisConfig,
    GridCellReferenceProfile,
    GridFrameAnalysisResult,
    analyze_grid_frame_chunk,
    analyze_grid_frame_single_source_path,
    analyze_grid_frame_sources_chunk,
    configure_grid_worker_process,
    load_cached_grid_frame_result,
)
from .performance import PerformanceConfig, load_performance_config
from .profiling import ProfileSnapshot, ProfilerRun, WorkerProfilePacket, activate_profiler, export_profile
from .analytics import collect_frame_records, compute_build_result_analytics
from .frame_details import load_frame_detail_base, load_frame_detail_model_confidence


def _grid_inspection_frame_timeout_seconds() -> float:
    try:
        return max(5.0, float(os.environ.get("KARAKAL_GRID_INSPECTION_FRAME_TIMEOUT", "20") or "20"))
    except Exception:
        return 20.0


def _grid_inspection_execution_settings(config: PerformanceConfig | None = None) -> tuple[str, int, int, int]:
    performance = config or load_performance_config()
    mode = "sequential" if performance.sequential_debug_mode or not performance.parallel_enabled else "process"
    return mode, performance.cpu_workers, performance.batch_size, performance.opencv_threads


_LOGGER = logging.getLogger(__name__)


def _profiled_grid_chunk(
    chunk: tuple[tuple[str, str], ...],
    config: GridDamageAnalysisConfig,
    use_cache: bool,
    reference_profile: GridCellReferenceProfile | None,
    read_cache: bool,
    write_cache: bool,
    performance: PerformanceConfig,
) -> tuple[tuple[dict[str, GridFrameAnalysisResult], dict[str, str]], WorkerProfilePacket]:
    profiler = ProfilerRun("validation_grid_worker", performance)
    with (
        activate_profiler(profiler),
        profiler.stage(
            "validation.grid.worker_batch",
            frame_count=len(chunk),
            batch_count=1,
        ),
    ):
        result = analyze_grid_frame_chunk(
            chunk,
            config,
            use_cache,
            reference_profile=reference_profile,
            read_cache=read_cache,
            write_cache=write_cache,
        )
    return result, profiler.worker_packet(
        f"pid-{os.getpid()}",
        processed_frames=len(chunk),
        processed_batches=1,
    )


def _profiled_grid_sources_chunk(
    chunk: tuple[tuple[str, str, str], ...],
    config: GridDamageAnalysisConfig,
    use_cache: bool,
    reference_profile: GridCellReferenceProfile | None,
    single_source_layer: str | None,
    performance: PerformanceConfig,
) -> tuple[tuple[dict[str, dict[str, GridFrameAnalysisResult]], dict[str, str]], WorkerProfilePacket]:
    profiler = ProfilerRun("validation_grid_worker", performance)
    with (
        activate_profiler(profiler),
        profiler.stage(
            "validation.grid.worker_batch",
            frame_count=len(chunk),
            batch_count=1,
        ),
    ):
        result = analyze_grid_frame_sources_chunk(
            chunk,
            config,
            use_cache,
            reference_profile=reference_profile,
            single_source_layer=single_source_layer,
        )
    return result, profiler.worker_packet(
        f"pid-{os.getpid()}",
        processed_frames=len(chunk),
        processed_batches=1,
    )


class WorkerBase(QObject):
    """Provide cancellation and signal plumbing for background workers."""

    PROGRESS_MIN_INTERVAL_SECONDS = 0.10
    FRAME_STATE_MIN_INTERVAL_SECONDS = 0.05

    progress = pyqtSignal(int, int, str)
    frameStateChanged = pyqtSignal(str, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    frameStatesChanged = pyqtSignal(object, str)
    profilingUpdated = pyqtSignal(object)
    profilingFinished = pyqtSignal(object)

    def __init__(
        self,
        *,
        performance_config: PerformanceConfig | None = None,
        analysis_type: str = "validation",
    ) -> None:
        super().__init__()
        self._performance_config = performance_config or load_performance_config()
        self._profiler = ProfilerRun(analysis_type, self._performance_config)
        self._cancel_requested = Event()
        self._last_progress_emit_at = 0.0
        self._last_profile_emit_at = 0.0
        self._last_frame_state_emit_at = 0.0
        self._emitted_running_keys: set[str] = set()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def _is_cancelled(self) -> bool:
        return self._cancel_requested.is_set()

    def _emit_progress(self, current: int, total: int, key: str, *, force: bool = False) -> None:
        current_i = int(current)
        total_i = int(total)
        now = monotonic()
        if (
            force
            or current_i <= 0
            or (total_i > 0 and current_i >= total_i)
            or now - self._last_progress_emit_at >= self._performance_config.progress_update_interval_ms / 1000.0
        ):
            self._last_progress_emit_at = now
            self.progress.emit(current_i, total_i, str(key or ""))
        if self._profiler.enabled:
            self._profiler.set_counter("frames.processed", max(0, current_i))
            self._profiler.set_counter("frames.total", max(0, total_i))
            if now - self._last_profile_emit_at >= self._performance_config.profiling_ui_refresh_interval_ms / 1000.0:
                self._last_profile_emit_at = now
                self.profilingUpdated.emit(self._profiler.snapshot())

    def _emit_frame_state(self, key: str, status: str) -> None:
        normalized_key = str(key or "")
        if not normalized_key:
            return
        normalized_status = str(status or "running")
        if normalized_status == "running":
            if normalized_key in self._emitted_running_keys:
                return
            self._emitted_running_keys.add(normalized_key)
            self.frameStateChanged.emit(normalized_key, normalized_status)
            return
        if normalized_key in self._emitted_running_keys:
            self._emitted_running_keys.discard(normalized_key)
            self.frameStateChanged.emit(normalized_key, normalized_status)

    def _emit_frame_states(self, keys, status: str) -> None:
        normalized_status = str(status or "running")
        normalized_keys = tuple(dict.fromkeys(str(key or "") for key in keys if str(key or "")))
        if not normalized_keys:
            return
        if normalized_status == "running":
            changed = tuple(key for key in normalized_keys if key not in self._emitted_running_keys)
            self._emitted_running_keys.update(changed)
        else:
            changed = tuple(key for key in normalized_keys if key in self._emitted_running_keys)
            self._emitted_running_keys.difference_update(changed)
        if changed:
            self.frameStatesChanged.emit(changed, normalized_status)

    @property
    def profiler(self) -> ProfilerRun:
        return self._profiler

    def _finish_profiling(self) -> ProfileSnapshot:
        snapshot = self._profiler.snapshot()
        self.profilingUpdated.emit(snapshot)
        if self._profiler.enabled:
            try:
                export_profile(snapshot, self._performance_config.profiling_directory, self._performance_config)
            except OSError as error:
                _LOGGER.warning("Could not export Karakal profile %s: %s", snapshot.run_id, error)
        self.profilingFinished.emit(snapshot)
        return snapshot


class FrameIndexWorker(WorkerBase):
    """Index all shared frames across the selected model folders."""

    def __init__(
        self,
        model_specs: tuple[ModelSpec, ...],
        options: BuildOptions,
        original_folder: FolderSpec | None,
        *,
        performance_config: PerformanceConfig | None = None,
    ) -> None:
        super().__init__(performance_config=performance_config, analysis_type="validation_index")
        self._model_specs = model_specs
        self._options = options
        self._original_folder = original_folder

    def run(self) -> None:
        with activate_profiler(self._profiler):
            try:
                self._emit_progress(0, 0, "", force=True)
                with self._profiler.stage("validation.prepare.index", frame_count=len(self._model_specs)):
                    result = collect_frame_records(
                        self._model_specs,
                        self._options,
                        original_folder=self._original_folder,
                        cancel_check=self._is_cancelled,
                        progress_callback=self._emit_progress,
                    )
            except Exception as error:
                _LOGGER.exception("Frame indexing failed")
                self.failed.emit(str(error))
                return
            finally:
                self._finish_profiling()
        self.finished.emit(result)


class AnalyticsWorker(WorkerBase):
    """Compute frame-level analytics for an indexed build result."""

    def __init__(
        self,
        build_result: BuildResult,
        metric_key: str,
        excluded_record_keys: set[str] | None = None,
        *,
        performance_config: PerformanceConfig | None = None,
    ) -> None:
        super().__init__(performance_config=performance_config, analysis_type="validation_metrics")
        self._build_result = build_result
        self._metric_key = metric_key
        self._excluded_record_keys = {str(key) for key in (excluded_record_keys or set()) if str(key)}

    def run(self) -> None:
        with activate_profiler(self._profiler):
            try:
                self._emit_progress(0, 0, "", force=True)
                with self._profiler.stage("validation.metrics", frame_count=len(self._build_result.records)):
                    result = compute_build_result_analytics(
                        self._build_result,
                        metric_key=self._metric_key,
                        excluded_record_keys=self._excluded_record_keys,
                        progress_callback=self._emit_progress,
                        state_callback=self._emit_frame_state,
                        cancel_check=self._is_cancelled,
                    )
            except Exception as error:
                _LOGGER.exception("Validation analytics failed")
                self.failed.emit(str(error))
                return
            finally:
                self._finish_profiling()
        self.finished.emit(result)


class GridInspectionWorker(WorkerBase):
    """Compute grid-damage results for all indexed frames in the background."""

    FRAME_TIMEOUT_SECONDS = _grid_inspection_frame_timeout_seconds()
    PARTIAL_RESULT_BATCH_SIZE = 256
    partialResultsReady = pyqtSignal(object)
    analysisErrors = pyqtSignal(object)

    def __init__(
        self,
        records: tuple[FrameRecord, ...] | list[FrameRecord],
        config: GridDamageAnalysisConfig | None = None,
        *,
        model_id: str | None = None,
        reference_profile: GridCellReferenceProfile | None = None,
        use_cache: bool = True,
        performance_config: PerformanceConfig | None = None,
    ) -> None:
        super().__init__(performance_config=performance_config, analysis_type="validation_grid")
        self._records = tuple(records)
        self._config = (config or GridDamageAnalysisConfig()).normalized()
        self._model_id = str(model_id or "")
        self._reference_profile = reference_profile
        self._use_cache = bool(use_cache)
        self._partial_result_buffer: dict[str, GridFrameAnalysisResult] = {}
        self._analysis_errors: dict[str, str] = {}

    def _source_path(self, record: FrameRecord) -> str:
        model_masks = getattr(record, "model_mask_paths", {}) or {}
        if self._model_id and model_masks.get(self._model_id):
            return str(model_masks[self._model_id] or "")
        if model_masks:
            return str(next(iter(model_masks.values())) or "")
        return str(
            getattr(record, "first_path", "")
            or getattr(record, "base_path", "")
            or getattr(record, "original_path", "")
            or ""
        )

    def run(self) -> None:
        with activate_profiler(self._profiler):
            try:
                self._partial_result_buffer.clear()
                self._analysis_errors.clear()
                self._emit_progress(0, 0, "", force=True)
                payloads: dict[str, GridFrameAnalysisResult] = {}
                cfg = replace(self._config, include_debug_payload=False, debug=False)
                records = [
                    (str(getattr(record, "key", "") or ""), self._source_path(record))
                    for record in self._records
                    if str(getattr(record, "key", "") or "") and self._source_path(record)
                ]
                total = len(records)
                completed = 0
                with self._profiler.stage("validation.grid.cache_lookup", frame_count=total):
                    if self._use_cache:
                        records, completed = self._partition_cached_records(records, cfg, payloads, total)
                mode, max_workers, chunk_size, opencv_threads = _grid_inspection_execution_settings(
                    self._performance_config
                )
                chunks = tuple(
                    tuple(records[offset : offset + chunk_size]) for offset in range(0, len(records), chunk_size)
                )
                with self._profiler.stage(
                    "validation.grid.worker_pool", frame_count=len(records), batch_count=len(chunks)
                ):
                    if mode == "sequential" or len(records) <= chunk_size:
                        self._run_sequential_chunks(chunks, cfg, payloads, total, completed=completed)
                    else:
                        try:
                            self._run_process_chunks(
                                chunks, cfg, payloads, total, max_workers, opencv_threads, completed=completed
                            )
                        except (OSError, RuntimeError) as error:
                            _LOGGER.warning("Grid process pool failed; continuing sequentially: %s", error)
                            remaining = tuple(
                                tuple((key, path_text) for key, path_text in chunk if key not in payloads)
                                for chunk in chunks
                            )
                            self._run_sequential_chunks(
                                tuple(chunk for chunk in remaining if chunk),
                                cfg,
                                payloads,
                                total,
                                completed=len(payloads),
                            )
                self._flush_partial_results()
                if self._is_cancelled():
                    self._profiler.increment("runs.cancelled")
                    self.cancelled.emit()
                    return
            except Exception as error:
                _LOGGER.exception("Grid inspection failed")
                self.failed.emit(str(error))
                return
            finally:
                self._finish_profiling()
        self.finished.emit(payloads)

    def _queue_partial_results(self, payloads: dict[str, GridFrameAnalysisResult], *, force: bool = False) -> None:
        if payloads:
            self._partial_result_buffer.update(payloads)
        if force or len(self._partial_result_buffer) >= self.PARTIAL_RESULT_BATCH_SIZE:
            self._flush_partial_results()

    def _flush_partial_results(self) -> None:
        if not self._partial_result_buffer:
            return
        self.partialResultsReady.emit(dict(self._partial_result_buffer))
        self._partial_result_buffer.clear()

    def _partition_cached_records(
        self,
        records: list[tuple[str, str]],
        config: GridDamageAnalysisConfig,
        payloads: dict[str, GridFrameAnalysisResult],
        total: int,
    ) -> tuple[list[tuple[str, str]], int]:
        misses: list[tuple[str, str]] = []
        completed = 0
        for key, path_text in records:
            if self._is_cancelled():
                break
            result = load_cached_grid_frame_result(
                path_text,
                frame_id=key,
                config=config,
                reference_profile=self._reference_profile,
            )
            if isinstance(result, GridFrameAnalysisResult):
                payloads[key] = result
                completed += 1
                self._queue_partial_results({key: result})
                if completed % self.PARTIAL_RESULT_BATCH_SIZE == 0:
                    self._emit_progress(completed, total, key)
            else:
                misses.append((key, path_text))
        if completed:
            last_key = next(reversed(payloads))
            self._emit_progress(completed, total, last_key, force=completed >= total)
        return misses, completed

    def _apply_chunk_result(
        self,
        chunk: tuple[tuple[str, str], ...],
        result: tuple[dict[str, GridFrameAnalysisResult], dict[str, str]],
        payloads: dict[str, GridFrameAnalysisResult],
        *,
        completed: int,
        total: int,
    ) -> int:
        chunk_payloads, errors = result
        if errors:
            self._analysis_errors.update(errors)
            self._profiler.increment("frames.errors", len(errors))
            grouped: dict[str, int] = {}
            for message in errors.values():
                grouped[message] = grouped.get(message, 0) + 1
            _LOGGER.warning("Grid batch completed with errors: %s", grouped)
            self.analysisErrors.emit(dict(errors))
        if chunk_payloads:
            payloads.update(chunk_payloads)
            self._queue_partial_results(chunk_payloads)
        next_completed = min(total, completed + len(chunk))
        last_key = chunk[-1][0] if chunk else ""
        self._emit_progress(next_completed, total, last_key, force=next_completed >= total)
        return next_completed

    def _run_sequential_chunks(
        self,
        chunks: tuple[tuple[tuple[str, str], ...], ...],
        config: GridDamageAnalysisConfig,
        payloads: dict[str, GridFrameAnalysisResult],
        total: int,
        *,
        completed: int,
    ) -> int:
        for chunk in chunks:
            if self._is_cancelled():
                break
            result = analyze_grid_frame_chunk(
                chunk,
                config,
                self._use_cache,
                reference_profile=self._reference_profile,
                read_cache=False,
                write_cache=self._use_cache,
            )
            completed = self._apply_chunk_result(chunk, result, payloads, completed=completed, total=total)
        return completed

    def _run_process_chunks(
        self,
        chunks: tuple[tuple[tuple[str, str], ...], ...],
        config: GridDamageAnalysisConfig,
        payloads: dict[str, GridFrameAnalysisResult],
        total: int,
        max_workers: int,
        opencv_threads: int,
        *,
        completed: int,
    ) -> None:
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=configure_grid_worker_process,
            initargs=(opencv_threads,),
        )
        pending: dict[object, tuple[tuple[tuple[str, str], ...], float]] = {}
        submitted = 0
        fallback_chunks: list[tuple[tuple[str, str], ...]] = []
        try:
            while not self._is_cancelled() and (submitted < len(chunks) or pending):
                while submitted < len(chunks) and len(pending) < max_workers * 2:
                    chunk = chunks[submitted]
                    submitted += 1
                    try:
                        if self._profiler.enabled and self._performance_config.profiling_collect_worker_stats:
                            future = executor.submit(
                                _profiled_grid_chunk,
                                chunk,
                                config,
                                self._use_cache,
                                self._reference_profile,
                                False,
                                self._use_cache,
                                self._performance_config,
                            )
                        else:
                            future = executor.submit(
                                analyze_grid_frame_chunk,
                                chunk,
                                config,
                                self._use_cache,
                                reference_profile=self._reference_profile,
                                read_cache=False,
                                write_cache=self._use_cache,
                            )
                    except (OSError, RuntimeError) as error:
                        _LOGGER.warning("Could not submit grid batch; scheduling sequential fallback: %s", error)
                        fallback_chunks.append(chunk)
                        fallback_chunks.extend(chunks[submitted:])
                        submitted = len(chunks)
                        break
                    pending[future] = (chunk, monotonic())
                    self._profiler.set_counter("worker.queue.depth", len(pending))
                if not pending:
                    break
                done, _not_done = wait(tuple(pending), timeout=0.05, return_when=FIRST_COMPLETED)
                now = monotonic()
                timed_out = {
                    future
                    for future, (chunk, started_at) in pending.items()
                    if now - started_at >= self.FRAME_TIMEOUT_SECONDS * max(1, len(chunk))
                }
                for future in set(done) | timed_out:
                    entry = pending.pop(future, None)
                    self._profiler.set_counter("worker.queue.depth", len(pending))
                    if entry is None:
                        continue
                    chunk, _started_at = entry
                    if future in timed_out and not future.done():
                        future.cancel()
                        errors = {key: "timeout" for key, _path in chunk}
                        result = ({}, errors)
                    else:
                        try:
                            result = future.result()
                            if len(result) == 2 and isinstance(result[1], WorkerProfilePacket):
                                result, packet = result
                                self._profiler.merge_worker(packet)
                        except Exception as error:
                            _LOGGER.warning("Grid worker batch failed; scheduling sequential fallback: %s", error)
                            fallback_chunks.append(chunk)
                            continue
                    completed = self._apply_chunk_result(chunk, result, payloads, completed=completed, total=total)
            if fallback_chunks and not self._is_cancelled():
                completed = self._run_sequential_chunks(
                    tuple(fallback_chunks),
                    config,
                    payloads,
                    total,
                    completed=completed,
                )
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)


class PairedGridInspectionWorker(GridInspectionWorker):
    """Compute all available grid matrices for one model layer."""

    def __init__(
        self,
        records: tuple[FrameRecord, ...] | list[FrameRecord],
        model_id: str,
        config: GridDamageAnalysisConfig | None = None,
        *,
        reference_profile: GridCellReferenceProfile | None = None,
        use_cache: bool = True,
        performance_config: PerformanceConfig | None = None,
    ) -> None:
        super().__init__(
            records,
            config,
            reference_profile=reference_profile,
            use_cache=use_cache,
            performance_config=performance_config,
        )
        self._model_id = str(model_id)

    def _paired_records(self) -> list[tuple[str, str, str]]:
        records: list[tuple[str, str, str]] = []
        for record in self._records:
            key = str(getattr(record, "key", "") or "")
            binary_path = str((getattr(record, "model_mask_paths", {}) or {}).get(self._model_id) or "")
            confidence_path = str((getattr(record, "model_prob_paths", {}) or {}).get(self._model_id) or "")
            if key and (binary_path or confidence_path):
                records.append((key, confidence_path, binary_path))
        return records

    def _resolve_single_source_layer(
        self,
        records: list[tuple[str, str, str]],
        config: GridDamageAnalysisConfig,
    ) -> str | None:
        has_confidence = any(bool(confidence_path) for _key, confidence_path, _binary_path in records)
        has_binary = any(bool(binary_path) for _key, _confidence_path, binary_path in records)
        if has_confidence and has_binary:
            return "binary"
        if has_confidence:
            return "confidence"
        mask_only = [entry for entry in records if entry[2]]
        if not mask_only:
            return None
        sample_count = min(7, len(mask_only))
        if sample_count == 1:
            sample_indexes = (0,)
        else:
            sample_indexes = tuple(
                sorted({int(round(index * (len(mask_only) - 1) / (sample_count - 1))) for index in range(sample_count)})
            )
        votes = {"confidence": 0, "binary": 0}
        for sample_index in sample_indexes:
            key, _confidence_path, binary_path = mask_only[sample_index]
            selected = analyze_grid_frame_single_source_path(
                binary_path,
                frame_id=key,
                config=config,
                reference_profile=self._reference_profile,
                use_cache=self._use_cache,
            )
            if selected is not None:
                votes[selected[0]] = int(votes.get(selected[0], 0) + 1)
        return "binary" if votes["binary"] > votes["confidence"] else "confidence"

    def _apply_pair_chunk_result(
        self,
        chunk: tuple[tuple[str, str, str], ...],
        result: tuple[dict[str, dict[str, GridFrameAnalysisResult]], dict[str, str]],
        payloads: dict[str, dict[str, GridFrameAnalysisResult]],
        *,
        completed: int,
        total: int,
    ) -> int:
        chunk_payloads, errors = result
        if errors:
            self._analysis_errors.update(errors)
            self._profiler.increment("frames.errors", len(errors))
            grouped: dict[str, int] = {}
            for message in errors.values():
                grouped[message] = grouped.get(message, 0) + 1
            _LOGGER.warning("Paired grid batch completed with errors: %s", grouped)
            self.analysisErrors.emit(dict(errors))
        if chunk_payloads:
            payloads.update(chunk_payloads)
            self._queue_partial_results(chunk_payloads)
        next_completed = min(total, completed + len(chunk))
        last_key = chunk[-1][0] if chunk else ""
        self._emit_progress(next_completed, total, last_key, force=next_completed >= total)
        return next_completed

    def _run_pair_sequential(
        self,
        chunks: tuple[tuple[tuple[str, str, str], ...], ...],
        config: GridDamageAnalysisConfig,
        payloads: dict[str, dict[str, GridFrameAnalysisResult]],
        total: int,
        *,
        completed: int = 0,
        single_source_layer: str | None = None,
    ) -> int:
        for chunk in chunks:
            if self._is_cancelled():
                break
            result = analyze_grid_frame_sources_chunk(
                chunk,
                config,
                self._use_cache,
                reference_profile=self._reference_profile,
                single_source_layer=single_source_layer,
            )
            completed = self._apply_pair_chunk_result(chunk, result, payloads, completed=completed, total=total)
        return completed

    def run(self) -> None:
        with activate_profiler(self._profiler):
            try:
                self._partial_result_buffer.clear()
                self._analysis_errors.clear()
                self._emit_progress(0, 0, "", force=True)
                payloads: dict[str, dict[str, GridFrameAnalysisResult]] = {}
                config = replace(self._config, include_debug_payload=False, debug=False)
                records = self._paired_records()
                with self._profiler.stage("validation.grid.source_detection", frame_count=len(records)):
                    single_source_layer = self._resolve_single_source_layer(records, config)
                total = len(records)
                mode, max_workers, chunk_size, opencv_threads = _grid_inspection_execution_settings(
                    self._performance_config
                )
                chunks = tuple(tuple(records[offset : offset + chunk_size]) for offset in range(0, total, chunk_size))
                with self._profiler.stage("validation.grid.worker_pool", frame_count=total, batch_count=len(chunks)):
                    if mode == "sequential" or total <= chunk_size:
                        self._run_pair_sequential(
                            chunks,
                            config,
                            payloads,
                            total,
                            single_source_layer=single_source_layer,
                        )
                    else:
                        self._run_pair_process(
                            chunks,
                            config,
                            payloads,
                            total,
                            max_workers,
                            opencv_threads,
                            single_source_layer=single_source_layer,
                        )
                self._flush_partial_results()
                if self._is_cancelled():
                    self._profiler.increment("runs.cancelled")
                    self.cancelled.emit()
                    return
            except Exception as error:
                _LOGGER.exception("Paired grid inspection failed")
                self.failed.emit(str(error))
                return
            finally:
                self._finish_profiling()
        self.finished.emit(payloads)

    def _run_pair_process(
        self,
        chunks: tuple[tuple[tuple[str, str, str], ...], ...],
        config: GridDamageAnalysisConfig,
        payloads: dict[str, dict[str, GridFrameAnalysisResult]],
        total: int,
        max_workers: int,
        opencv_threads: int,
        *,
        single_source_layer: str | None,
    ) -> None:
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=configure_grid_worker_process,
            initargs=(opencv_threads,),
        )
        pending: dict[object, tuple[tuple[tuple[str, str, str], ...], float]] = {}
        submitted = 0
        completed = 0
        fallback_chunks: list[tuple[tuple[str, str, str], ...]] = []
        try:
            while not self._is_cancelled() and (submitted < len(chunks) or pending):
                while submitted < len(chunks) and len(pending) < max_workers * 2:
                    chunk = chunks[submitted]
                    submitted += 1
                    try:
                        if self._profiler.enabled and self._performance_config.profiling_collect_worker_stats:
                            future = executor.submit(
                                _profiled_grid_sources_chunk,
                                chunk,
                                config,
                                self._use_cache,
                                self._reference_profile,
                                single_source_layer,
                                self._performance_config,
                            )
                        else:
                            future = executor.submit(
                                analyze_grid_frame_sources_chunk,
                                chunk,
                                config,
                                self._use_cache,
                                reference_profile=self._reference_profile,
                                single_source_layer=single_source_layer,
                            )
                    except (OSError, RuntimeError) as error:
                        _LOGGER.warning("Could not submit paired grid batch; scheduling sequential fallback: %s", error)
                        fallback_chunks.append(chunk)
                        continue
                    pending[future] = (chunk, monotonic())
                    self._profiler.set_counter("worker.queue.depth", len(pending))
                if not pending:
                    break
                done, _not_done = wait(tuple(pending), timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    chunk, _started_at = pending.pop(future)
                    self._profiler.set_counter("worker.queue.depth", len(pending))
                    try:
                        result = future.result()
                        if len(result) == 2 and isinstance(result[1], WorkerProfilePacket):
                            result, packet = result
                            self._profiler.merge_worker(packet)
                    except Exception as error:
                        _LOGGER.warning("Paired grid worker batch failed; scheduling sequential fallback: %s", error)
                        fallback_chunks.append(chunk)
                        continue
                    completed = self._apply_pair_chunk_result(chunk, result, payloads, completed=completed, total=total)
            if fallback_chunks and not self._is_cancelled():
                self._run_pair_sequential(
                    tuple(fallback_chunks),
                    config,
                    payloads,
                    total,
                    completed=completed,
                    single_source_layer=single_source_layer,
                )
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            self._profiler.set_counter("worker.queue.depth", 0)
            self._profiler.set_counter("worker.queue.depth", 0)


class DetailPayloadWorker(WorkerBase):
    """Load the base detail payload without blocking the UI thread."""

    def __init__(
        self,
        record: FrameRecord,
        build_result: BuildResult,
        model_id: str | None,
        max_side: int | None,
        *,
        performance_config: PerformanceConfig | None = None,
    ) -> None:
        super().__init__(performance_config=performance_config, analysis_type="validation_detail")
        self._record = record
        self._build_result = build_result
        self._model_id = model_id
        self._max_side = max_side

    def run(self) -> None:
        with activate_profiler(self._profiler):
            try:
                self._emit_progress(0, 0, "", force=True)
                with self._profiler.stage("validation.ui.detail.prepare", frame_id=self._record.key, frame_count=1):
                    payload = load_frame_detail_base(
                        self._record,
                        self._build_result,
                        model_id=self._model_id,
                        max_side=self._max_side,
                    )
            except Exception as error:
                _LOGGER.exception("Detail payload loading failed for %s", self._record.key)
                self.failed.emit(str(error))
                return
            finally:
                self._finish_profiling()
        self.finished.emit(payload)


class DetailConfidenceWorker(WorkerBase):
    """Compute heavy confidence/debug payload for one selected model in background."""

    def __init__(
        self,
        record: FrameRecord,
        build_result: BuildResult,
        model_id: str | None,
        max_side: int | None,
        detail_payload: dict[str, object],
        *,
        performance_config: PerformanceConfig | None = None,
    ) -> None:
        super().__init__(performance_config=performance_config, analysis_type="validation_detail_confidence")
        self._record = record
        self._build_result = build_result
        self._model_id = model_id
        self._max_side = max_side
        self._detail_payload = dict(detail_payload)
        self._detail_payload["model_confidence"] = dict((detail_payload.get("model_confidence") or {}))

    def run(self) -> None:
        with activate_profiler(self._profiler):
            try:
                self._emit_progress(0, 0, "", force=True)
                with self._profiler.stage("validation.confidence.detail", frame_id=self._record.key, frame_count=1):
                    payload = load_frame_detail_model_confidence(
                        self._record,
                        self._build_result,
                        model_id=self._model_id,
                        max_side=self._max_side,
                        detail_payload=self._detail_payload,
                    )
            except Exception as error:
                _LOGGER.exception("Detail confidence loading failed for %s", self._record.key)
                self.failed.emit(str(error))
                return
            finally:
                self._finish_profiling()
        self.finished.emit(payload)


# Preferred alias for the analytics worker used by the widget.
MetricsWorker = AnalyticsWorker

# Backward-compatible alias for legacy lite imports.
MismatchWorker = AnalyticsWorker
