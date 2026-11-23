from __future__ import annotations

import multiprocessing as mp
import os
import unittest
from unittest import mock

from contour.application.processing import BatchFrameTiming, BatchImageMetadata
from contour.batch_processor import (
    BatchRunSummary,
    limit_worker_native_threads,
    memory_limited_worker_count,
)
from contour.batch_worker import BatchChunkDiagnostics, BatchChunkResult

_MB = 1024 * 1024


class MemoryLimitedWorkerCountTests(unittest.TestCase):
    def test_caps_workers_by_available_memory(self) -> None:
        # 16 GB free, 2 GB reserved, 1.5 GB per worker -> 9 workers fit.
        count = memory_limited_worker_count(
            32,
            available_bytes=16 * 1024 * _MB,
            per_worker_bytes=1536 * _MB,
            reserve_bytes=2048 * _MB,
        )
        self.assertEqual(count, 9)

    def test_keeps_requested_count_when_memory_is_enough(self) -> None:
        count = memory_limited_worker_count(
            4,
            available_bytes=64 * 1024 * _MB,
            per_worker_bytes=1536 * _MB,
            reserve_bytes=2048 * _MB,
        )
        self.assertEqual(count, 4)

    def test_always_allows_one_worker(self) -> None:
        count = memory_limited_worker_count(
            8,
            available_bytes=512 * _MB,
            per_worker_bytes=1536 * _MB,
            reserve_bytes=2048 * _MB,
        )
        self.assertEqual(count, 1)

    def test_unknown_memory_keeps_requested_count(self) -> None:
        count = memory_limited_worker_count(6, available_bytes=None)
        self.assertEqual(count, 6)


def _child_blas_threads(queue) -> None:  # type: ignore[no-untyped-def]
    queue.put(os.environ.get("OPENBLAS_NUM_THREADS"))


class WorkerNativeThreadTests(unittest.TestCase):
    def test_spawned_workers_inherit_single_blas_thread(self) -> None:
        # numpy's OpenBLAS commits a buffer per thread at import time; spawned
        # workers import numpy before any initializer runs, so the limit must
        # already be in the environment they inherit.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENBLAS_NUM_THREADS", None)
            limit_worker_native_threads()
            context = mp.get_context("spawn")
            queue = context.Queue()
            process = context.Process(target=_child_blas_threads, args=(queue,))
            process.start()
            value = queue.get(timeout=30)
            process.join(timeout=30)
        self.assertEqual(value, "1")

    def test_explicit_user_setting_is_kept(self) -> None:
        with mock.patch.dict(os.environ, {"OPENBLAS_NUM_THREADS": "3"}, clear=False):
            limit_worker_native_threads()
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "3")


class BatchRunSummaryTests(unittest.TestCase):
    def test_summary_keeps_timings_but_not_polygons(self) -> None:
        summary = BatchRunSummary()
        polygons = [object(), object()]
        chunk = BatchChunkResult(
            chunk_id=1,
            metadata=[
                BatchImageMetadata(
                    image_path="a.png",
                    polygon_count=2,
                    polygons=polygons,  # type: ignore[arg-type]
                    timing=BatchFrameTiming(total_frame_ms=10.0, saving_ms=2.0),
                ),
                BatchImageMetadata(image_path="b.png", error="boom"),
            ],
            diagnostics=BatchChunkDiagnostics(
                chunk_id=1,
                worker_pid=1,
                frame_count=2,
                wall_ms=20.0,
                busy_ms=10.0,
                utilization=0.5,
            ),
        )

        summary.add(chunk)

        self.assertEqual([t.total_frame_ms for t in summary.timings], [10.0])
        self.assertEqual(summary.utilizations, [0.5])
        self.assertEqual(set(BatchRunSummary.__slots__), {"timings", "utilizations"})


if __name__ == "__main__":
    unittest.main()


class _ClosedManagerEvent:
    """A manager Event proxy whose manager process has already shut down."""

    def set(self) -> None:
        raise BrokenPipeError(232, "pipe is being closed")

    def is_set(self) -> bool:
        raise BrokenPipeError(232, "pipe is being closed")


def _runner(image_paths, **kwargs):
    from contour.application.processing import ContourExtractionSettings, DisplaySettings, SaveOptions
    from contour.batch_processor import BatchQueueRunnable

    return BatchQueueRunnable(
        image_paths=list(image_paths),
        pipeline_config={"steps": []},
        contour_settings=ContourExtractionSettings(min_area=1.0),
        output_directory=None,
        save_options=SaveOptions(),
        display_settings=DisplaySettings(),
        max_workers=2,
        ui_language="en",
        chunk_size=1,
        **kwargs,
    )


class BatchStopTests(unittest.TestCase):
    def test_stop_from_the_ui_never_calls_the_manager(self) -> None:
        # Closing the window right after a stop raced the runner shutting the
        # manager down; stop() then raised BrokenPipeError out of closeEvent.
        runner = _runner(["a.png"])
        runner._cancel_event = _ClosedManagerEvent()

        runner.stop()

        self.assertTrue(runner._stop_requested)

    def test_stop_before_start_reaches_workers_and_finishes(self) -> None:
        import tempfile
        from pathlib import Path

        import cv2
        import numpy as np
        from PyQt6.QtCore import QCoreApplication, Qt

        QCoreApplication.instance() or QCoreApplication([])
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(6):
                image = np.zeros((32, 32), dtype=np.uint8)
                image[8:24, 8:24] = 255
                path = str(Path(directory) / f"frame-{index}.png")
                cv2.imwrite(path, image)
                paths.append(path)
            runner = _runner(paths)
            results, finished = [], []
            direct = Qt.ConnectionType.DirectConnection
            runner.signals.result.connect(results.append, direct)
            runner.signals.finished.connect(lambda: finished.append(True), direct)

            runner.stop()
            runner.run()

        self.assertEqual(results, [])
        self.assertEqual(finished, [True])
