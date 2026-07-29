"""Background Qt workers for the extended validation gradient widget."""
from __future__ import annotations

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
    configure_grid_worker_process,
    load_cached_grid_frame_result,
)
from .repository import collect_frame_records, compute_build_result_analytics, load_frame_detail_base, load_frame_detail_model_confidence


def _grid_inspection_frame_timeout_seconds() -> float:
    try:
        return max(5.0, float(os.environ.get("KARAKAL_GRID_INSPECTION_FRAME_TIMEOUT", "20") or "20"))
    except Exception:
        return 20.0


def _positive_env_int(name: str, default: int, *, maximum: int) -> int:
    try:
        return max(1, min(int(maximum), int(os.environ.get(name, str(default)) or default)))
    except Exception:
        return max(1, min(int(maximum), int(default)))


def _grid_inspection_execution_settings() -> tuple[str, int, int, int]:
    logical_cpus = max(1, int(os.cpu_count() or 1))
    default_workers = max(1, min(4, (logical_cpus + 1) // 2))
    mode = str(os.environ.get("KARAKAL_GRID_INSPECTION_EXECUTION", "process") or "process").strip().lower()
    if mode not in {"process", "sequential"}:
        mode = "process"
    workers = _positive_env_int("KARAKAL_GRID_INSPECTION_WORKERS", default_workers, maximum=logical_cpus)
    chunk_size = _positive_env_int("KARAKAL_GRID_INSPECTION_CHUNK_SIZE", 8, maximum=256)
    opencv_threads = _positive_env_int("KARAKAL_GRID_INSPECTION_OPENCV_THREADS", 1, maximum=logical_cpus)
    return mode, workers, chunk_size, opencv_threads


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

    def __init__(self) -> None:
        super().__init__()
        self._cancel_requested = Event()
        self._last_progress_emit_at = 0.0
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
            or now - self._last_progress_emit_at >= self.PROGRESS_MIN_INTERVAL_SECONDS
        ):
            self._last_progress_emit_at = now
            self.progress.emit(current_i, total_i, str(key or ""))

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


class FrameIndexWorker(WorkerBase):
    """Index all shared frames across the selected model folders."""

    def __init__(self, model_specs: tuple[ModelSpec, ...], options: BuildOptions, original_folder: FolderSpec | None, gt_folder: FolderSpec | None) -> None:
        super().__init__()
        self._model_specs = model_specs
        self._options = options
        self._original_folder = original_folder
        self._gt_folder = gt_folder

    def run(self) -> None:
        try:
            self._emit_progress(0, 0, "", force=True)
            result = collect_frame_records(
                self._model_specs,
                self._options,
                original_folder=self._original_folder,
                gt_folder=self._gt_folder,
                cancel_check=self._is_cancelled,
                progress_callback=self._emit_progress,
            )
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished.emit(result)


class AnalyticsWorker(WorkerBase):
    """Compute frame-level analytics for an indexed build result."""

    def __init__(self, build_result: BuildResult, metric_key: str, excluded_record_keys: set[str] | None = None) -> None:
        super().__init__()
        self._build_result = build_result
        self._metric_key = metric_key
        self._excluded_record_keys = {str(key) for key in (excluded_record_keys or set()) if str(key)}

    def run(self) -> None:
        try:
            self._emit_progress(0, 0, "", force=True)
            result = compute_build_result_analytics(
                self._build_result,
                metric_key=self._metric_key,
                excluded_record_keys=self._excluded_record_keys,
                progress_callback=self._emit_progress,
                state_callback=self._emit_frame_state,
                cancel_check=self._is_cancelled,
            )
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished.emit(result)


class GridInspectionWorker(WorkerBase):
    """Compute grid-damage results for all indexed frames in the background."""

    FRAME_TIMEOUT_SECONDS = _grid_inspection_frame_timeout_seconds()
    PARTIAL_RESULT_BATCH_SIZE = 256
    partialResultsReady = pyqtSignal(object)

    def __init__(
        self,
        records: tuple[FrameRecord, ...] | list[FrameRecord],
        config: GridDamageAnalysisConfig | None = None,
        *,
        reference_profile: GridCellReferenceProfile | None = None,
        use_cache: bool = True,
    ) -> None:
        super().__init__()
        self._records = tuple(records)
        self._config = (config or GridDamageAnalysisConfig()).normalized()
        self._reference_profile = reference_profile
        self._use_cache = bool(use_cache)
        self._partial_result_buffer: dict[str, GridFrameAnalysisResult] = {}

    @staticmethod
    def _source_path(record: FrameRecord) -> str:
        model_masks = getattr(record, "model_mask_paths", {}) or {}
        if model_masks:
            return str(next(iter(model_masks.values())) or "")
        return str(getattr(record, "first_path", "") or getattr(record, "base_path", "") or getattr(record, "original_path", "") or "")

    def run(self) -> None:
        try:
            self._partial_result_buffer.clear()
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
            if self._use_cache:
                records, completed = self._partition_cached_records(records, cfg, payloads, total)
            mode, max_workers, chunk_size, opencv_threads = _grid_inspection_execution_settings()
            chunks = tuple(tuple(records[offset : offset + chunk_size]) for offset in range(0, len(records), chunk_size))
            if mode == "sequential" or len(records) <= chunk_size:
                self._run_sequential_chunks(chunks, cfg, payloads, total, completed=completed)
            else:
                try:
                    self._run_process_chunks(chunks, cfg, payloads, total, max_workers, opencv_threads, completed=completed)
                except Exception:
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
                self.cancelled.emit()
                return
        except Exception as error:
            self.failed.emit(str(error))
            return
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
        chunk_payloads, _errors = result
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
                        future = executor.submit(
                            analyze_grid_frame_chunk,
                            chunk,
                            config,
                            self._use_cache,
                            reference_profile=self._reference_profile,
                            read_cache=False,
                            write_cache=self._use_cache,
                        )
                    except Exception:
                        fallback_chunks.append(chunk)
                        fallback_chunks.extend(chunks[submitted:])
                        submitted = len(chunks)
                        break
                    pending[future] = (chunk, monotonic())
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
                        except Exception:
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


class DetailPayloadWorker(WorkerBase):
    """Load the base detail payload without blocking the UI thread."""

    def __init__(self, record: FrameRecord, build_result: BuildResult, model_id: str | None, max_side: int | None) -> None:
        super().__init__()
        self._record = record
        self._build_result = build_result
        self._model_id = model_id
        self._max_side = max_side

    def run(self) -> None:
        try:
            self._emit_progress(0, 0, "", force=True)
            payload = load_frame_detail_base(
                self._record,
                self._build_result,
                model_id=self._model_id,
                max_side=self._max_side,
            )
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished.emit(payload)


class DetailConfidenceWorker(WorkerBase):
    """Compute heavy confidence/debug payload for one selected model in background."""

    def __init__(self, record: FrameRecord, build_result: BuildResult, model_id: str | None, max_side: int | None, detail_payload: dict[str, object]) -> None:
        super().__init__()
        self._record = record
        self._build_result = build_result
        self._model_id = model_id
        self._max_side = max_side
        self._detail_payload = dict(detail_payload)
        self._detail_payload["model_confidence"] = dict((detail_payload.get("model_confidence") or {}))

    def run(self) -> None:
        try:
            self._emit_progress(0, 0, "", force=True)
            payload = load_frame_detail_model_confidence(
                self._record,
                self._build_result,
                model_id=self._model_id,
                max_side=self._max_side,
                detail_payload=self._detail_payload,
            )
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished.emit(payload)


# Preferred alias for the analytics worker used by the widget.
MetricsWorker = AnalyticsWorker

# Backward-compatible alias for legacy lite imports.
MismatchWorker = AnalyticsWorker
