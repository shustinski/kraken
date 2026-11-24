from __future__ import annotations

import threading
import time

import pytest

from contour import batch_processor
from contour.application.processing import ContourExtractionSettings, DisplaySettings, SaveOptions
from contour.batch_worker import BatchChunkResult


def stubborn_chunk(request, _cancel_event, _counter=None, _lock=None):
    """A chunk that never looks at the cancel event (e.g. stuck in native code)."""
    from pathlib import Path

    (Path(request.output_directory) / "worker-started").write_text("1", encoding="utf-8")
    time.sleep(120)
    return BatchChunkResult(chunk_id=request.chunk_id)


def _runner(tmp_path, grace):
    return batch_processor.BatchQueueRunnable(
        image_paths=[str(tmp_path / "a.png")],
        pipeline_config={"steps": []},
        contour_settings=ContourExtractionSettings(),
        output_directory=str(tmp_path),
        save_options=SaveOptions(),
        display_settings=DisplaySettings(),
        max_workers=1,
        ui_language="en",
        stop_grace_seconds=grace,
    )


def test_pause_terminates_workers_that_ignore_cancellation(tmp_path, monkeypatch):
    from PyQt6.QtCore import QCoreApplication, Qt

    QCoreApplication.instance() or QCoreApplication([])
    monkeypatch.setattr(batch_processor, "process_batch_chunk", stubborn_chunk)
    runner = _runner(tmp_path, grace=1.0)
    logs, finished = [], []
    direct = Qt.ConnectionType.DirectConnection
    runner.signals.log.connect(logs.append, direct)
    runner.signals.finished.connect(lambda: finished.append(True), direct)
    thread = threading.Thread(target=runner.run, daemon=True)
    thread.start()
    marker = tmp_path / "worker-started"
    deadline = time.monotonic() + 60
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.exists()
    paused_at = time.monotonic()
    runner.stop()
    runner.stop()  # a second Pause must not reset the deadline or block
    thread.join(timeout=60)
    assert not thread.is_alive()
    assert time.monotonic() - paused_at < 30  # the worker would sleep for 120 s
    assert finished == [True]
    assert any("batch_workers_terminated" in line for line in logs)


def test_interrupted_cif_write_leaves_previous_file_and_no_partial(tmp_path, monkeypatch):
    from contour.serializers_parts import save_polygons_cif as module

    target = tmp_path / "a.cif"
    target.write_text("previous", encoding="utf-8")

    def interrupted(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(module.os, "replace", interrupted)
    with pytest.raises(OSError):
        module.save_polygons_cif(target, "a.png", [], (2, 2))
    assert target.read_text(encoding="utf-8") == "previous"
    assert [path.name for path in tmp_path.iterdir()] == ["a.cif"]


def test_cif_write_publishes_complete_file(tmp_path):
    from contour.local_batch_resume import is_complete_cif
    from contour.serializers_parts.save_polygons_cif import save_polygons_cif

    target = save_polygons_cif(tmp_path / "a.cif", "a.png", [], (2, 2))
    assert is_complete_cif(target)
    assert [path.name for path in tmp_path.iterdir()] == ["a.cif"]
