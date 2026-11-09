"""model_train_and_recognition: per-batch cProfile context."""
from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

from neuralimage.infrastructure import profiling


def train_step_profile_context(*, epoch: int, batch_index: int, global_batch: int) -> ContextManager:
    if not (profiling.training_profiling_enabled() and global_batch % max(1, profiling.training_profile_stride()) == 0):
        return nullcontext()
    return profiling.cprofile_region(
        "training",
        label="train step",
        extra=f"epoch={epoch + 1} batch={batch_index + 1} global={global_batch}",
        enabled=True,
    )


def train_profiling_notice_line() -> str | None:
    if not profiling.training_profiling_enabled():
        return None
    return f'cProfile profiling enabled, log={profiling.profiling_log_path()}'
