"""model_train_and_recognition: per-batch cProfile context."""
from __future__ import annotations

import cProfile
from contextlib import contextmanager, nullcontext
from time import perf_counter
from typing import ContextManager, Iterator

from neuralimage.infrastructure import profiling


def train_step_profile_context(*, epoch: int, batch_index: int, global_batch: int) -> ContextManager:
    if profiling.training_profiling_enabled() and (global_batch % max(1, profiling.training_profile_stride()) == 0):
        return _train_step_region(
            extra=f"epoch={epoch + 1} batch={batch_index + 1} global={global_batch}",
        )
    return nullcontext()


@contextmanager
def _train_step_region(*, extra: str) -> Iterator[None]:
    # Mirrors ``profiling.cprofile_region``, but the enabled decision is made
    # by ``train_step_profile_context`` (``training_profiling_enabled`` +
    # stride) instead of re-checking ``profiling_enabled`` inside the region,
    # so the gate lives at one observable, patchable call site.
    profiler = profiling.cProfile.Profile()
    active = profiling.try_enable_profiler(profiler)
    started = perf_counter()
    try:
        yield
    finally:
        wall_ms = (perf_counter() - started) * 1000.0
        if active:
            profiling.try_disable_profiler(profiler)
        report_profiler = profiler if active and isinstance(profiler, cProfile.Profile) else None
        profiling.write_profile_report(
            *profiling._format_cprofile_report(
                "training",
                label="train step",
                extra=extra,
                wall_ms=wall_ms,
                profiler=report_profiler,
            )
        )


def train_profiling_notice_line() -> str | None:
    if not profiling.training_profiling_enabled():
        return None
    return f'cProfile profiling enabled, log={profiling.profiling_log_path()}'
