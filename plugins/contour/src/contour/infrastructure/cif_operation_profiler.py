"""Thread-local CIF operation timings for frame-switch profiling."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from collections.abc import Iterator
from time import perf_counter

_cif_timings: contextvars.ContextVar[dict[str, float] | None] = contextvars.ContextVar(
    "cif_operation_timings",
    default=None,
)


@contextmanager
def cif_operation_profiling() -> Iterator[dict[str, float]]:
    """Collect nested CIF phase timings for the current thread."""

    timings: dict[str, float] = {}
    token = _cif_timings.set(timings)
    try:
        yield timings
    finally:
        _cif_timings.reset(token)


def note_cif_operation_timing(name: str, elapsed_ms: float) -> None:
    timings = _cif_timings.get()
    if timings is None:
        return
    timings[name] = float(timings.get(name, 0.0)) + float(elapsed_ms)


def note_cif_operation_count(name: str, increment: float = 1.0) -> None:
    """Increment a counter stored alongside CIF phase timings."""

    note_cif_operation_timing(name, increment)


def profile_cif_operation(name: str):
    """Context manager that records wall time for one CIF phase."""

    @contextmanager
    def _scope() -> Iterator[None]:
        started_at = perf_counter()
        try:
            yield
        finally:
            note_cif_operation_timing(name, (perf_counter() - started_at) * 1000.0)

    return _scope()
