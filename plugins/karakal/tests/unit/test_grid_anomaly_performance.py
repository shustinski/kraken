from __future__ import annotations

import dataclasses
import hashlib
import json

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

import karakal.core.grid_anomaly as grid_anomaly
import karakal.core.workers as workers_module
from karakal.core.grid_anomaly import _outline_side_coverages, detect_grid_cell_anomalies
from karakal.core.domain import FrameRecord
from karakal.core.workers import GridInspectionWorker
from karakal.ui.matrix_view import MatrixListWidget


GOLDEN_DIGESTS = (
    "1459a05912216239747feae38365d4d8f8402d878915f08515469d8bdb329e83",
    "f0e6cb5c973f719747009213a44668ea9da0902915c470db76cb61c666ca9360",
    "61f0338b714799f3eec395e617aa6267efdd435080d9ecb6717bf3a80f3693f8",
    "a0f243616f7c2d38a8519d5d90735e519550e5c023e6bcb3e3bc1468e19128c7",
    "44621366b8f0fd75f9df0bdd93e1276ff919d874ba2cec41e47bba49df6eebed",
    "666cd197ce437cfbce316b1c7966835b9de7d236a0e47b79bbc66cb0ed120a42",
    "1459a05912216239747feae38365d4d8f8402d878915f08515469d8bdb329e83",
    "1459a05912216239747feae38365d4d8f8402d878915f08515469d8bdb329e83",
)


def _synthetic_grid_frame(variant: int, shape: tuple[int, int] = (768, 1024)) -> np.ndarray:
    height, width = shape
    rows, cols = 12, 16
    cell_w = max(8, width // (cols + 2))
    cell_h = max(8, height // (rows + 2))
    margin_x = max(4, (width - cols * cell_w) // 2)
    margin_y = max(4, (height - rows * cell_h) // 2)
    image = np.zeros((height, width), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            x = margin_x + col * cell_w
            y = margin_y + row * cell_h
            cv2.rectangle(image, (x + 2, y + 2), (x + cell_w - 3, y + cell_h - 3), 255, 2)
    x = margin_x + (variant * 5 % cols) * cell_w
    y = margin_y + (variant * 3 % rows) * cell_h
    defect = variant % 8
    if defect == 1:
        cv2.rectangle(image, (x + 4, y + 4), (x + cell_w - 5, y + cell_h - 5), 255, -1)
    elif defect == 2:
        cv2.rectangle(image, (x + 4, y + cell_h // 2), (x + cell_w - 5, y + cell_h - 5), 255, -1)
    elif defect == 3:
        cv2.circle(image, (min(width - 2, x + cell_w + 5), min(height - 2, y + 5)), 3, 255, -1)
    elif defect == 4:
        cv2.line(image, (x + 2, y + 2), (x + cell_w - 3, y + cell_h - 3), 0, 3)
    elif defect == 5:
        cv2.rectangle(image, (x + cell_w - 5, y + 4), (min(width - 1, x + cell_w + 8), y + cell_h - 5), 255, -1)
    elif defect == 6:
        image[:, :] = cv2.add(image, np.full_like(image, 20))
    elif defect == 7:
        noise = np.random.default_rng(variant).normal(0.0, 7.0, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    return image


def _result_digest(result) -> str:
    payload = dataclasses.asdict(result)
    payload.pop("frame_id")
    payload.pop("frame_path")
    payload.pop("debug")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(("variant", "expected_digest"), enumerate(GOLDEN_DIGESTS))
def test_grid_analysis_matches_pre_optimization_golden_result(variant: int, expected_digest: str) -> None:
    result = detect_grid_cell_anomalies(_synthetic_grid_frame(variant))

    assert _result_digest(result) == expected_digest


def test_outline_side_coverages_matches_numpy_reference() -> None:
    rng = np.random.default_rng(20260720)
    for height, width in ((4, 4), (17, 31), (64, 53)):
        roi = (rng.random((height, width)) > 0.72).astype(np.uint8) * 255
        band = max(1, int(round(min(width, height) * 0.18)))
        band = min(band, max(1, width), max(1, height))
        expected = (
            float(np.mean(np.any(roi[:band, :] > 0, axis=0))),
            float(np.mean(np.any(roi[height - band :, :] > 0, axis=0))),
            float(np.mean(np.any(roi[:, :band] > 0, axis=1))),
            float(np.mean(np.any(roi[:, width - band :] > 0, axis=1))),
        )

        assert _outline_side_coverages(roi) == expected


def test_grid_analysis_ignores_thin_cell_edge_fragments() -> None:
    image = np.zeros((360, 520), dtype=np.uint8)
    rows, cols = 6, 8
    cell_w, cell_h = 42, 38
    margin_x, margin_y = 70, 55
    for row in range(rows):
        for col in range(cols):
            x = margin_x + col * cell_w
            y = margin_y + row * cell_h
            cv2.rectangle(image, (x + 4, y + 4), (x + cell_w - 5, y + cell_h - 5), 255, 2)

    cv2.rectangle(image, (margin_x + cell_w - 1, margin_y + 5), (margin_x + cell_w + 3, margin_y + cell_h - 6), 255, -1)
    cv2.line(
        image,
        (margin_x + 3 * cell_w + 2, margin_y + 3 * cell_h + 2),
        (margin_x + 3 * cell_w + 2, margin_y + 4 * cell_h - 8),
        255,
        2,
    )

    result = detect_grid_cell_anomalies(
        image,
        config=grid_anomaly.GridDamageAnalysisConfig(min_contour_area=4.0, min_cell_size=2),
    )

    assert result.grid_detected
    assert not any("small_artifact" in cell.reasons for cell in result.cells)


def test_process_and_sequential_workers_produce_identical_results(tmp_path, monkeypatch) -> None:
    process_records = []
    sequential_records = []
    for index in range(12):
        image = _synthetic_grid_frame(index % 8, shape=(384, 512))
        process_path = tmp_path / f"process_{index:02d}.png"
        sequential_path = tmp_path / f"sequential_{index:02d}.png"
        assert cv2.imwrite(str(process_path), image)
        assert cv2.imwrite(str(sequential_path), image)
        process_records.append(FrameRecord(f"process_{index:02d}", process_path.name, first_path=str(process_path)))
        sequential_records.append(FrameRecord(f"sequential_{index:02d}", sequential_path.name, first_path=str(sequential_path)))

    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_WORKERS", "2")
    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_CHUNK_SIZE", "4")
    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_OPENCV_THREADS", "1")

    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_EXECUTION", "process")
    process_finished = []
    process_failed = []
    process_batches = []
    process_worker = GridInspectionWorker(process_records, use_cache=False)
    process_worker.finished.connect(process_finished.append)
    process_worker.failed.connect(process_failed.append)
    process_worker.partialResultsReady.connect(process_batches.append)
    process_worker.run()

    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_EXECUTION", "sequential")
    sequential_finished = []
    sequential_worker = GridInspectionWorker(sequential_records, use_cache=False)
    sequential_worker.finished.connect(sequential_finished.append)
    sequential_worker.run()

    assert process_failed == []
    assert len(process_finished) == 1
    assert len(sequential_finished) == 1
    assert 1 <= len(process_batches) <= 3
    process_digests = sorted(_result_digest(result) for result in process_finished[0].values())
    sequential_digests = sorted(_result_digest(result) for result in sequential_finished[0].values())
    assert process_digests == sequential_digests


def test_grid_worker_cancelled_before_start_emits_cancelled(monkeypatch) -> None:
    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_EXECUTION", "sequential")
    worker = GridInspectionWorker([], use_cache=False)
    cancelled = []
    finished = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.finished.connect(finished.append)

    worker.request_cancel()
    worker.run()

    assert cancelled == [True]
    assert finished == []


def test_grid_worker_coalesces_partial_results_for_ui() -> None:
    worker = GridInspectionWorker([], use_cache=False)
    batches = []
    worker.partialResultsReady.connect(batches.append)

    for index in range(600):
        worker._queue_partial_results({str(index): index})
    worker._flush_partial_results()

    assert [len(batch) for batch in batches] == [256, 256, 88]
    assert set().union(*(set(batch) for batch in batches)) == {str(index) for index in range(600)}


def test_identical_grid_payload_does_not_rebuild_scene(monkeypatch) -> None:
    _app = QApplication.instance() or QApplication([])
    view = MatrixListWidget()
    view.set_grid_inspection_payloads({}, enabled=True)

    def fail_refresh() -> None:
        raise AssertionError("unchanged grid payload rebuilt the scene")

    monkeypatch.setattr(view, "refresh_scene", fail_refresh)
    view.set_grid_inspection_payloads({}, enabled=True)
    view.close()


def test_grid_worker_falls_back_when_process_pool_cannot_start(tmp_path, monkeypatch) -> None:
    records = []
    for index in range(4):
        path = tmp_path / f"fallback_{index}.png"
        assert cv2.imwrite(str(path), _synthetic_grid_frame(index, shape=(384, 512)))
        records.append(FrameRecord(f"fallback_{index}", path.name, first_path=str(path)))

    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_EXECUTION", "process")
    monkeypatch.setenv("KARAKAL_GRID_INSPECTION_CHUNK_SIZE", "2")

    class BrokenExecutor:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("process pool unavailable")

    monkeypatch.setattr(workers_module, "ProcessPoolExecutor", BrokenExecutor)
    finished = []
    failed = []
    worker = GridInspectionWorker(records, use_cache=False)
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)

    worker.run()

    assert failed == []
    assert len(finished) == 1
    assert set(finished[0]) == {record.key for record in records}


def test_grid_cache_hit_corruption_and_file_change(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    path = tmp_path / "frame.png"
    assert cv2.imwrite(str(path), _synthetic_grid_frame(1, shape=(384, 512)))
    monkeypatch.setattr(grid_anomaly, "GRID_DAMAGE_CACHE_DIR", cache_dir)

    first = grid_anomaly.analyze_grid_frame_path(path, frame_id="frame", use_cache=True)
    assert first is not None
    cache_files = list(cache_dir.glob("*.pickle"))
    assert len(cache_files) == 1

    original_analyze = grid_anomaly.detect_grid_cell_anomalies

    def fail_if_reanalyzed(*_args, **_kwargs):
        raise AssertionError("cache hit unexpectedly re-ran analysis")

    monkeypatch.setattr(grid_anomaly, "detect_grid_cell_anomalies", fail_if_reanalyzed)
    cached = grid_anomaly.analyze_grid_frame_path(path, frame_id="frame", use_cache=True)
    assert cached == first

    cache_files[0].write_bytes(b"not a pickle")
    monkeypatch.setattr(grid_anomaly, "detect_grid_cell_anomalies", original_analyze)
    recovered = grid_anomaly.analyze_grid_frame_path(path, frame_id="frame", use_cache=True)
    assert recovered == first

    assert cv2.imwrite(str(path), _synthetic_grid_frame(3, shape=(384, 512)))
    changed = grid_anomaly.analyze_grid_frame_path(path, frame_id="frame", use_cache=True)
    assert changed is not None
    assert _result_digest(changed) != _result_digest(first)
    assert len(list(cache_dir.glob("*.pickle"))) == 2


def test_grid_chunk_keeps_valid_results_when_one_image_is_corrupt(tmp_path) -> None:
    valid_path = tmp_path / "valid.png"
    corrupt_path = tmp_path / "corrupt.png"
    assert cv2.imwrite(str(valid_path), _synthetic_grid_frame(2, shape=(384, 512)))
    corrupt_path.write_bytes(b"not an image")

    payloads, errors = grid_anomaly.analyze_grid_frame_chunk(
        (("valid", str(valid_path)), ("corrupt", str(corrupt_path))),
        grid_anomaly.GridDamageAnalysisConfig(),
        use_cache=False,
    )

    assert set(payloads) == {"valid"}
    assert errors == {"corrupt": "decode_error"}


@pytest.mark.parametrize("chunk_size", (1, 2, 5, 16))
def test_chunk_partition_does_not_change_results(chunk_size: int, tmp_path) -> None:
    entries = []
    for index in range(5):
        path = tmp_path / f"frame_{index}.png"
        assert cv2.imwrite(str(path), _synthetic_grid_frame(index, shape=(384, 512)))
        entries.append((f"frame_{index}", str(path)))
    entries_tuple = tuple(entries)
    expected, expected_errors = grid_anomaly.analyze_grid_frame_chunk(
        entries_tuple,
        grid_anomaly.GridDamageAnalysisConfig(),
        use_cache=False,
    )

    actual = {}
    actual_errors = {}
    for offset in range(0, len(entries_tuple), chunk_size):
        payloads, errors = grid_anomaly.analyze_grid_frame_chunk(
            entries_tuple[offset : offset + chunk_size],
            grid_anomaly.GridDamageAnalysisConfig(),
            use_cache=False,
        )
        actual.update(payloads)
        actual_errors.update(errors)

    assert actual_errors == expected_errors
    assert {key: _result_digest(value) for key, value in actual.items()} == {
        key: _result_digest(value) for key, value in expected.items()
    }
