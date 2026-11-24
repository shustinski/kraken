from __future__ import annotations

import io
import pstats
import threading

from contour.infrastructure.profiling import SharedProfile, try_disable_profiler, try_enable_profiler


def _busy_main() -> int:
    return sum(index * index for index in range(20_000))


def _busy_worker() -> int:
    return sum(index * 3 for index in range(20_000))


def _function_names(profile: SharedProfile) -> set[str]:
    stats = pstats.Stats(profile, stream=io.StringIO())
    return {name for (_file, _line, name) in stats.stats}


def test_overlapping_sessions_both_collect_stats() -> None:
    outer = SharedProfile()
    inner = SharedProfile()
    assert try_enable_profiler(outer)
    _busy_main()
    assert try_enable_profiler(inner)
    _busy_worker()
    try_disable_profiler(inner)
    try_disable_profiler(outer)

    assert {"_busy_main", "_busy_worker"} <= _function_names(outer)
    inner_names = _function_names(inner)
    assert "_busy_worker" in inner_names
    assert "_busy_main" not in inner_names


def test_window_includes_work_from_other_threads() -> None:
    profile = SharedProfile()
    assert try_enable_profiler(profile)
    thread = threading.Thread(target=_busy_worker)
    thread.start()
    thread.join()
    try_disable_profiler(profile)

    assert "_busy_worker" in _function_names(profile)


def test_windows_accumulate_and_clear_resets() -> None:
    profile = SharedProfile()
    for _ in range(2):
        profile.enable()
        _busy_main()
        profile.disable()
    counts = {
        entry.code.co_name: entry.callcount
        for entry in profile.getstats()
        if hasattr(entry.code, "co_name")
    }
    assert counts["_busy_main"] == 2

    profile.clear()
    assert profile.getstats() == []


def test_host_slot_is_released_after_last_session() -> None:
    import cProfile

    profile = SharedProfile()
    profile.enable()
    profile.disable()
    plain = cProfile.Profile()
    plain.enable()  # would raise ValueError if the shared host were still enabled
    plain.disable()
