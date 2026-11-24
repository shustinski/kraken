from __future__ import annotations

import cProfile
import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QBuffer, QEvent, QIODevice, QPointF, Qt, QTimer
from PyQt6.QtGui import QImage, QMouseEvent
from PyQt6.QtWidgets import QApplication, QWidget

from kraken_core.frame_matrix import MatrixAssetRef, MatrixBounds, MatrixViewportRequest
from kraken_core.frame_matrix import profiling
from kraken_core.frame_matrix.qt import FrameMatrixWidget
from kraken_hub.presentation.qt.matrix_source import KrakenMatrixAssetSource, KrakenMatrixDataSource
from kraken_hub.presentation.qt.parts.run_manager_gui import _MainWindowReadyFilter


def test_matrix_zoom_profile_reports_frames_per_second(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text(
        "[profiling]\nenabled = false\nmatrix_zoom = true\ntop_lines = 5\n",
        encoding="utf-8",
    )
    profiling.configure(tmp_path)
    session = profiling.begin_matrix_zoom(1.0, 1.28)
    assert session is not None
    session.record_frame(session.started_at)
    session.record_frame(session.started_at)
    session.finish(status="displayed", final_zoom=1.28)

    report = (tmp_path / "profiling.log").read_text(encoding="utf-8")
    assert "[matrix_zoom] started" in report
    assert "status=displayed" in report
    assert "frames=2" in report
    assert "fps=" in report
    assert not session.active


def test_profiling_ini_lives_at_project_root():
    project_root = Path(__file__).resolve().parents[1]

    assert profiling.project_directory() == project_root
    assert (project_root / "kraken.ini").is_file()


def test_kraken_profiling_ini_is_created_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_config", None)
    profiling.configure(tmp_path)

    ini = tmp_path / "kraken.ini"
    assert ini.is_file()
    assert "matrix_thumbnail =" in ini.read_text(encoding="utf-8")
    assert not profiling.enabled("matrix_thumbnail")
    assert not (tmp_path / "profiling.log").exists()


def test_ini_profiles_viewport_thumbnail_and_gui_regions(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text(
        "[profiling]\nenabled = false\nmatrix_viewport = true\n"
        "matrix_thumbnail = true\nmatrix_ui = true\ntop_lines = 5\n",
        encoding="utf-8",
    )
    profiling.configure(tmp_path)
    content = io.BytesIO()
    Image.new("RGB", (20, 16), "red").save(content, "PNG")

    class Service:
        def matrix_viewport(self, *_args, **_kwargs):
            return {"cells": (), "aggregates": ()}

        def is_remote_project(self, _project_id):
            return True

        def read_project_blob(self, *_args):
            return content.getvalue()

    service = Service()
    result = KrakenMatrixDataSource(service, project_id="project").load_viewport(
        MatrixViewportRequest(MatrixBounds(1, 1, 1, 1))
    )
    assert result.items == ()
    assert KrakenMatrixAssetSource(service, project_id="project").load_asset(
        MatrixAssetRef("image", "1"), width=8, height=8
    )
    qapp = QApplication.instance() or QApplication([])
    view = FrameMatrixWidget(1, 1)
    view._poll_background_results()
    view.close()
    assert qapp is not None

    report = (tmp_path / "profiling.log").read_text(encoding="utf-8")
    assert "matrix_viewport" in report
    assert "matrix_thumbnail" in report
    assert "matrix_ui" in report


def test_invalid_profiling_setting_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text("[profiling]\nmatrix_thumbnail = maybe\n", encoding="utf-8")

    with pytest.raises(ValueError, match="matrix_thumbnail"):
        profiling.configure(tmp_path)


def test_individual_and_total_thumbnail_profiling_are_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text(
        "[profiling]\nmatrix_thumbnail = true\nmatrix_thumbnails_total = true\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="cannot both be true"):
        profiling.configure(tmp_path)


def test_active_profiler_does_not_interrupt_matrix_ui(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text("[profiling]\nmatrix_ui = true\n", encoding="utf-8")
    profiling.configure(tmp_path)
    outer = cProfile.Profile()
    outer.enable()
    try:
        with profiling.region("matrix_ui"):
            sum(range(10))
        sum(range(20))
    finally:
        outer.disable()

    report = (tmp_path / "profiling.log").read_text(encoding="utf-8")
    assert "[matrix_ui]" in report
    assert "cprofile_skipped=yes" in report
    assert outer.getstats()


def test_profiler_slot_conflict_uses_wall_time_without_hiding_other_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text("[profiling]\nmatrix_ui = true\n", encoding="utf-8")
    profiling.configure(tmp_path)

    class ConflictingProfile:
        def enable(self):
            raise ValueError("tool 2 is already in use")

    monkeypatch.setattr(profiling.cProfile, "Profile", ConflictingProfile)
    with profiling.region("matrix_ui"):
        sum(range(10))
    assert "cprofile_skipped=yes" in (tmp_path / "profiling.log").read_text(encoding="utf-8")

    class BrokenProfile:
        def enable(self):
            raise ValueError("unexpected profiler state")

    monkeypatch.setattr(profiling.cProfile, "Profile", BrokenProfile)
    with pytest.raises(ValueError, match="unexpected profiler state"):
        with profiling.region("matrix_ui"):
            pass


def test_profile_session_finishes_only_when_marked_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text("[profiling]\nmain_window = true\n", encoding="utf-8")
    profiling.configure(tmp_path)

    session = profiling.begin_region("main_window")
    assert session is not None
    assert not (tmp_path / "profiling.log").exists()

    session.finish(ready=True)
    session.finish(ready=False)
    report = (tmp_path / "profiling.log").read_text(encoding="utf-8")
    assert "[main_window]" in report
    assert "ready=yes" in report
    assert report.count("[main_window]") == 1


def test_profile_session_reports_named_startup_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text("[profiling]\nplugin_launch = true\nlayer_manager = true\n", encoding="utf-8")
    profiling.configure(tmp_path)
    for kind in ("plugin_launch", "layer_manager"):
        session = profiling.begin_region(kind)
        assert session is not None
        session.checkpoint("prepared")
        session.finish(ready=True)

    report = (tmp_path / "profiling.log").read_text(encoding="utf-8")
    assert "[plugin_launch]" in report
    assert "[layer_manager]" in report
    assert report.count("started_at=") == 2
    assert report.count("prepared:") == 2


def test_main_window_profile_waits_for_paint_and_event_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text("[profiling]\nmain_window = true\n", encoding="utf-8")
    profiling.configure(tmp_path)
    app = QApplication.instance() or QApplication([])
    profile = profiling.begin_region("main_window")
    assert profile is not None
    window = QWidget()
    ready_filter = _MainWindowReadyFilter(window, profile)
    window.show()
    assert not (tmp_path / "profiling.log").exists()

    QTimer.singleShot(100, app.quit)
    app.exec()
    window.close()

    assert ready_filter is not None
    report = (tmp_path / "profiling.log").read_text(encoding="utf-8")
    assert "ready=yes" in report


def _mouse(viewport, kind: QEvent.Type, pos, buttons) -> None:
    local = QPointF(pos)
    QApplication.sendEvent(
        viewport,
        QMouseEvent(
            kind,
            local,
            QPointF(viewport.mapToGlobal(pos)),
            Qt.MouseButton.LeftButton,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        ),
    )


def test_layer_selection_profile_runs_from_press_until_redraw(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text("[profiling]\nlayer_selection = true\n", encoding="utf-8")
    profiling.configure(tmp_path)
    app = QApplication.instance() or QApplication([])
    view = FrameMatrixWidget(4, 4)
    view.resize(400, 400)
    view.show()
    app.processEvents()
    viewport = view.viewport()
    start = view.mapFromScene(view.scene_rect_for_frame(1, 1).center())
    end = view.mapFromScene(view.scene_rect_for_frame(3, 3).center())

    _mouse(viewport, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    assert not (tmp_path / "profiling.log").exists()
    _mouse(viewport, QEvent.Type.MouseMove, end, Qt.MouseButton.LeftButton)
    _mouse(viewport, QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton)
    QTimer.singleShot(200, app.quit)
    app.exec()
    view.close()

    report = (tmp_path / "profiling.log").read_text(encoding="utf-8")
    assert report.count("[layer_selection]") == 1
    assert "ready=yes" in report
    assert view.selection().rectangles


def test_concurrent_thumbnail_profiles_keep_the_loaded_bytes(tmp_path, monkeypatch):
    image = QImage(40, 30, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.red)
    encoded = QBuffer()
    encoded.open(QIODevice.OpenModeFlag.WriteOnly)
    assert image.save(encoded, "PNG")
    payload = bytes(encoded.data())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/blob"
    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text("[profiling]\nmatrix_thumbnails_total = true\n", encoding="utf-8")
    profiling.configure(tmp_path)
    batch = profiling.begin_thumbnail_batch()
    assert batch is not None
    failures: list[BaseException] = []
    loaded: list[int] = []
    lock = threading.Lock()

    def load_one():
        from urllib.request import urlopen

        with urlopen(url, timeout=5) as response:
            downloaded = response.read()
        decoded = QImage.fromData(downloaded)
        scaled = decoded.scaled(8, 8, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        return scaled.width()

    def worker():
        for _ in range(8):
            try:
                width = batch.profile_call(load_one)
            except Exception as exc:
                with lock:
                    failures.append(exc)
            else:
                with lock:
                    loaded.append(width)

    workers = [threading.Thread(target=worker) for _ in range(3)]
    try:
        for worker_thread in workers:
            worker_thread.start()
        for worker_thread in workers:
            worker_thread.join()
    finally:
        server.shutdown()
        thread.join(timeout=3)

    assert failures == []
    assert loaded == [8] * 24
