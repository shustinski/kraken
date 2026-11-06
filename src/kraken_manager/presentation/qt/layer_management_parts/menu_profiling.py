"""Temporary diagnostics for synchronous layer action menu preparation."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import cProfile
import io
import pstats
import sys
from time import perf_counter


# Set to False to disable diagnostics without changing launch arguments.
PROFILE_LAYER_ACTION_MENU = True


@contextmanager
def profile_menu_open(target: str) -> Iterator[Callable[[str], None]]:
    """Report preparation only; leave QMenu.exec and action execution outside."""
    if not PROFILE_LAYER_ACTION_MENU:
        yield lambda _stage: None
        return

    profiler = cProfile.Profile()
    stages: list[tuple[str, float]] = []
    started = previous = perf_counter()

    def checkpoint(stage: str) -> None:
        nonlocal previous
        now = perf_counter()
        stages.append((stage, (now - previous) * 1000))
        previous = now

    profiler.enable()
    try:
        yield checkpoint
    finally:
        profiler.disable()
        elapsed_ms = (perf_counter() - started) * 1000
        checkpoint("finalize")
        output = io.StringIO()
        output.write(f"[layer-menu-profile] {target} total_ms={elapsed_ms:.3f}\n")
        for stage, milliseconds in stages:
            output.write(f"  {stage}: {milliseconds:.3f} ms\n")
        pstats.Stats(profiler, stream=output).strip_dirs().sort_stats("cumulative").print_stats(40)
        if sys.stderr is not None:
            print(output.getvalue(), file=sys.stderr, flush=True)
