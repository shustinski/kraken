"""repository: iter record payloads."""
from __future__ import annotations
import math
from time import perf_counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    FrameRecord,
    GeometryMode,
    ModelSpec,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import (
    ANALYTICS_STALL_PROGRESS_SECONDS,
    ANALYTICS_STALL_TIMEOUT_SECONDS,
    ANALYTICS_WAIT_TIMEOUT_SECONDS,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.analytics_worker_count import (
    _analytics_worker_count,
    _remember_analytics_worker_plan,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.analyze_record_payload import (
    _analyze_record_payload,
    _analyze_record_payload_batch_for_executor,
    _analyze_record_payload_for_executor,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import (
    BuildCancelledError,
    _analytics_batch_size,
    _use_thread_pool_for_analytics,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.record_payload_cache_key import _clear_runtime_image_caches

def _iter_record_payloads(
    records: list[FrameRecord],
    model_specs: tuple[ModelSpec, ...],
    analysis_max_side: int | None,
    max_workers: int,
    *,
    geometry_mode: GeometryMode,
    point_match_radius: float,
    boundary_radius: int,
    confidence_uncertainty_delta: float,
    point_confidence_radius: int,
    polygon_confidence_summary: str,
    cache_enabled: bool,
    include_model_confidence: bool = True,
    include_pairwise_metrics: bool = True,
    include_model_metrics: bool = True,
    progress_callback=None,
    state_callback=None,
    cancel_check=None,
):
    """Yield per-record analytics payloads without retaining the whole dataset in RAM."""

    use_thread_pool = _use_thread_pool_for_analytics()
    worker_count = _analytics_worker_count(
        max_workers,
        geometry_mode=geometry_mode,
        analysis_max_side=analysis_max_side,
        include_model_confidence=include_model_confidence,
        include_pairwise_metrics=include_pairwise_metrics,
        include_model_metrics=include_model_metrics,
        model_count=len(model_specs),
        use_thread_pool=use_thread_pool,
    )
    _remember_analytics_worker_plan(
        requested=max_workers,
        selected=worker_count,
        use_thread_pool=use_thread_pool,
        geometry_mode=geometry_mode,
        analysis_max_side=analysis_max_side,
        model_count=len(model_specs),
        include_model_confidence=include_model_confidence,
        include_pairwise_metrics=include_pairwise_metrics,
        include_model_metrics=include_model_metrics,
    )

    def run_sequential():
        for index, record in enumerate(records, start=1):
            if cancel_check is not None and cancel_check():
                raise BuildCancelledError("Build cancelled")
            if state_callback is not None:
                state_callback(record.key, "running")
            payload = _analyze_record_payload(
                record,
                model_specs,
                analysis_max_side,
                geometry_mode,
                point_match_radius,
                boundary_radius,
                confidence_uncertainty_delta,
                point_confidence_radius,
                polygon_confidence_summary,
                cache_enabled,
                include_model_confidence,
                include_pairwise_metrics,
                include_model_metrics,
            )
            if state_callback is not None:
                state_callback(record.key, "done")
            if progress_callback is not None:
                progress_callback(index, len(records), record.key)
            yield record.key, payload

    if worker_count <= 1 or len(records) <= 1:
        yield from run_sequential()
        return

    try:
        executor_cls = ThreadPoolExecutor if use_thread_pool else ProcessPoolExecutor
        total_records = len(records)
        batch_size = 1 if use_thread_pool else _analytics_batch_size(total_records, worker_count)
        batch_count = int(math.ceil(float(total_records) / float(batch_size)))

        def make_work_item(record: FrameRecord):
            return (
                record,
                model_specs,
                analysis_max_side,
                geometry_mode,
                point_match_radius,
                boundary_radius,
                confidence_uncertainty_delta,
                point_confidence_radius,
                polygon_confidence_summary,
                cache_enabled,
                include_model_confidence,
                include_pairwise_metrics,
                include_model_metrics,
            )

        def make_batch(order_index: int):
            start = int(order_index) * int(batch_size)
            stop = min(total_records, start + int(batch_size))
            return tuple(make_work_item(records[index]) for index in range(start, stop))

        executor = executor_cls(max_workers=worker_count)
        shutdown_wait = True
        executor_shutdown = False
        try:
            completed = 0
            max_in_flight = max(1, worker_count)
            next_index = 0
            future_to_batch: dict[object, tuple[tuple[object, ...], ...]] = {}
            last_result_at = perf_counter()
            last_stall_progress_at = last_result_at

            def submit_work_item(order_index: int) -> None:
                batch = make_batch(order_index)
                if use_thread_pool:
                    future = executor.submit(_analyze_record_payload_for_executor, batch[0])
                else:
                    future = executor.submit(_analyze_record_payload_batch_for_executor, batch)
                future_to_batch[future] = batch
                if state_callback is not None:
                    for item in batch:
                        state_callback(item[0].key, "running")

            while next_index < batch_count and len(future_to_batch) < max_in_flight:
                submit_work_item(next_index)
                next_index += 1

            while future_to_batch:
                if cancel_check is not None and cancel_check():
                    if state_callback is not None:
                        for batch in future_to_batch.values():
                            for item in batch:
                                state_callback(item[0].key, "stale")
                    shutdown_wait = False
                    executor.shutdown(wait=False, cancel_futures=True)
                    executor_shutdown = True
                    raise BuildCancelledError("Build cancelled")
                done, _pending = wait(
                    tuple(future_to_batch.keys()),
                    timeout=ANALYTICS_WAIT_TIMEOUT_SECONDS,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    now = perf_counter()
                    if progress_callback is not None and now - last_stall_progress_at >= ANALYTICS_STALL_PROGRESS_SECONDS:
                        last_stall_progress_at = now
                        progress_callback(completed, len(records), "Waiting for analytics workers")
                    if now - last_result_at >= ANALYTICS_STALL_TIMEOUT_SECONDS:
                        if state_callback is not None:
                            for batch in future_to_batch.values():
                                for item in batch:
                                    state_callback(item[0].key, "stale")
                        shutdown_wait = False
                        executor.shutdown(wait=False, cancel_futures=True)
                        executor_shutdown = True
                        raise TimeoutError(
                            "Analytics workers stalled: no completed frame payloads within "
                            f"{ANALYTICS_STALL_TIMEOUT_SECONDS:.0f}s."
                        )
                    continue
                last_result_at = perf_counter()
                for future in done:
                    future_to_batch.pop(future)
                    result = future.result()
                    batch_result = (result,) if use_thread_pool else tuple(result)
                    if state_callback is not None:
                        for record_key, _payload in batch_result:
                            state_callback(record_key, "done")
                    for record_key, payload in batch_result:
                        completed += 1
                        if progress_callback is not None:
                            progress_callback(completed, len(records), record_key)
                        yield record_key, payload
                    if completed % 32 == 0:
                        _clear_runtime_image_caches()
                while next_index < batch_count and len(future_to_batch) < max_in_flight:
                    submit_work_item(next_index)
                    next_index += 1
        except Exception:
            shutdown_wait = False
            if state_callback is not None:
                for batch in future_to_batch.values():
                    for item in batch:
                        state_callback(item[0].key, "stale")
            if not executor_shutdown:
                executor.shutdown(wait=False, cancel_futures=True)
                executor_shutdown = True
            raise
        finally:
            if not executor_shutdown:
                executor.shutdown(wait=shutdown_wait, cancel_futures=not shutdown_wait)
    except (PermissionError, OSError):
        yield from run_sequential()
