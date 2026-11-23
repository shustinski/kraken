from __future__ import annotations

import unittest

from contour.application.processing import BatchFrameTiming, BatchImageMetadata
from contour.batch_processor import BatchRunSummary, memory_limited_worker_count
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
