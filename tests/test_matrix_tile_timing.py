from __future__ import annotations

import io

import pytest
from PIL import Image

from kraken_core.frame_matrix import profiling
from kraken_core.frame_matrix.prepared_tiles import MatrixTileNotReady, PreparedTilePyramid


def _enable(tmp_path, monkeypatch, *, log_file: str | None = None) -> None:
    monkeypatch.setattr(profiling, "_config", None)
    monkeypatch.setattr(profiling, "_handler", None)
    (tmp_path / "kraken.ini").write_text(
        "[profiling]\nenabled = false\nmatrix_tile_timing = true\n", encoding="utf-8"
    )
    profiling.configure(tmp_path, log_file=log_file)


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 0, 0)).save(output, "PNG")
    return output.getvalue()


def test_timing_writes_one_line_per_event(tmp_path, monkeypatch):
    _enable(tmp_path, monkeypatch)

    profiling.timing("matrix_tile_timing", "tile_call", side="client", result="ready", ms=12.3456)

    lines = (tmp_path / "profiling.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("[matrix_tile_timing] at=")
    assert "event=tile_call side=client result=ready ms=12.346" in lines[0]


def test_timing_is_silent_when_the_kind_is_off(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_config", None)
    (tmp_path / "kraken.ini").write_text("[profiling]\nenabled = false\n", encoding="utf-8")
    profiling.configure(tmp_path)

    profiling.timing("matrix_tile_timing", "tile_call", ms=1.0)

    assert not (tmp_path / "profiling.log").exists()


def test_server_log_file_override(tmp_path, monkeypatch):
    _enable(tmp_path, monkeypatch, log_file="profiling-server.log")

    profiling.timing("matrix_tile_timing", "tile_read", side="server")

    assert (tmp_path / "profiling-server.log").is_file()
    assert not (tmp_path / "profiling.log").exists()


def test_urgent_tile_reports_queue_wait_and_source_reads(tmp_path, monkeypatch):
    _enable(tmp_path, monkeypatch)
    cells = ({"x": 1, "y": 1, "asset_sha256": "aa11"}, {"x": 2, "y": 1, "asset_sha256": "bb22"})
    pyramid = PreparedTilePyramid(tmp_path / "pyramid", lambda: ("rev", 16, 16, cells), lambda _cell: _png())
    pyramid.build_now()

    with pytest.raises(MatrixTileNotReady):
        pyramid.read(x1=1, y1=1, x2=2, y2=1, orientation="y_down")
    pyramid.build_now()

    lines = [
        line
        for line in (tmp_path / "profiling.log").read_text(encoding="utf-8").splitlines()
        if "event=urgent_tile" in line
    ]
    assert len(lines) == 1
    line = lines[0]
    assert "rect=1:1:2:1" in line
    assert "queue_wait_ms=" in line
    assert "frames=2" in line
    # Previews were stored by the full build, so the urgent tile reads none from the source.
    assert "previews_cached=2" in line
    assert "previews_read=0" in line
    assert "already_on_disk=False" in line


def test_server_times_tile_requests_end_to_end(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kraken_server.interface.http.tile_timing import register_tile_timing

    _enable(tmp_path, monkeypatch, log_file="profiling-server.log")
    app = FastAPI()
    register_tile_timing(app)

    @app.get("/api/v1/projects/p/viewport-mosaic")
    def tile() -> dict[str, str]:
        return {}

    @app.get("/api/v1/other")
    def other() -> dict[str, str]:
        return {}

    client = TestClient(app)
    client.get("/api/v1/projects/p/viewport-mosaic?x1=1&y1=2&x2=3&y2=4")
    client.get("/api/v1/other")

    lines = (tmp_path / "profiling-server.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "event=http_request side=server rect=1:2:3:4 status=200 ms=" in lines[0]
