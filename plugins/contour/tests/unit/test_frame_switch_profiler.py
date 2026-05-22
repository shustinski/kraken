from __future__ import annotations

import cProfile
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from contour.infrastructure.frame_switch_profiler import (
    FrameSwitchProfile,
    frame_switch_profiling_enabled,
    profile_callable,
)
from contour.infrastructure.profiling import processing_profiling_enabled, vertex_move_profiling_enabled
from contour.infrastructure.profiling import try_enable_profiler


class FrameSwitchProfilerTests(unittest.TestCase):
    def test_profile_callable_attaches_worker_profile(self) -> None:
        session = FrameSwitchProfile.begin("frame.png", generation=1)

        def work() -> int:
            total = 0
            for index in range(2000):
                total += index
            return total

        value = profile_callable("worker_test", session, work)
        self.assertEqual(value, 1999000)
        self.assertEqual(len(session.worker_profilers), 1)
        self.assertEqual(session.worker_profilers[0][0], "worker_test")

    def test_pending_phase_timing(self) -> None:
        session = FrameSwitchProfile.begin("frame.png", generation=1)
        session.mark_pending("editor_display")
        session.complete_pending("editor_display", suffix="_ready")
        self.assertIn("editor_display_ready", session.timings_ms)
        self.assertGreater(session.timings_ms["editor_display_ready"], 0.0)

    def test_profile_callable_uses_wall_time_when_main_profiler_active(self) -> None:
        session = FrameSwitchProfile.begin("frame.png", generation=1)
        session.enable_main_profiler()
        try:

            def work() -> int:
                return sum(range(100))

            self.assertEqual(profile_callable("worker_nested", session, work), 4950)
            self.assertEqual(len(session.worker_profilers), 0)
            self.assertIn("worker_worker_nested_wall", session.timings_ms)
        finally:
            session.disable_main_profiler()

    def test_profile_callable_collects_cprofile_when_main_slot_free(self) -> None:
        session = FrameSwitchProfile.begin("frame.png", generation=1)

        def work() -> int:
            return sum(range(200))

        self.assertEqual(profile_callable("worker_free", session, work), 19900)
        self.assertEqual(len(session.worker_profilers), 1)

    def test_enable_main_profiler_skips_when_slot_taken(self) -> None:
        outer = cProfile.Profile()
        outer.enable()
        try:
            session = FrameSwitchProfile.begin("frame.png", generation=1)
            self.assertFalse(session.enable_main_profiler())
            self.assertTrue(session.main_stats_skipped)
        finally:
            outer.disable()

    def test_shared_profiler_enable_helper_skips_when_slot_taken(self) -> None:
        outer = cProfile.Profile()
        inner = cProfile.Profile()
        outer.enable()
        try:
            self.assertFalse(try_enable_profiler(inner))
        finally:
            outer.disable()

    def test_profiling_enabled_by_default(self) -> None:
        env = os.environ
        for key in (
            "CONTOUR_PROFILE",
            "CONTOUR_PROFILING",
            "CONTOUR_PROFILE_ALL",
            "CONTOUR_PROFILE_FRAME_SWITCH",
            "CONTOUR_PROFILE_FRAME_OPEN",
            "CONTOUR_PROFILE_CIF_OPEN",
        ):
            env.pop(key, None)
        self.assertTrue(frame_switch_profiling_enabled())
        env["CONTOUR_PROFILE_FRAME_SWITCH"] = "0"
        try:
            self.assertFalse(frame_switch_profiling_enabled())
        finally:
            env.pop("CONTOUR_PROFILE_FRAME_SWITCH", None)

    def test_master_profile_switch_controls_profilers(self) -> None:
        env = os.environ
        keys = (
            "CONTOUR_PROFILE",
            "CONTOUR_PROFILE_FRAME_SWITCH",
            "CONTOUR_PROFILE_PROCESSING",
            "CONTOUR_PROFILE_VERTEX_MOVE",
            "CONTOUR_PROFILE_FRAME_OPEN",
            "CONTOUR_PROFILE_CIF_OPEN",
        )
        previous = {key: env.get(key) for key in keys}
        try:
            for key in keys:
                env.pop(key, None)
            env["CONTOUR_PROFILE"] = "0"
            self.assertFalse(frame_switch_profiling_enabled())
            self.assertFalse(processing_profiling_enabled())
            self.assertFalse(vertex_move_profiling_enabled())

            env["CONTOUR_PROFILE"] = "1"
            self.assertTrue(frame_switch_profiling_enabled())
            self.assertTrue(processing_profiling_enabled())
            self.assertTrue(vertex_move_profiling_enabled())

            env["CONTOUR_PROFILE_VERTEX_MOVE"] = "0"
            self.assertFalse(vertex_move_profiling_enabled())
        finally:
            for key, value in previous.items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
