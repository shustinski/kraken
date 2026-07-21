from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from contour.adapters.qt import thumbnails as thumbnails_module
from contour.adapters.qt.thumbnails import ThumbnailLoadRunnable
import contour.widget as widget_module
from contour.ui.large_dataset import clamp_thumbnail_source_size
from contour.widget import PolygonExtractionWidget


def _app() -> QApplication:
    instance = QApplication.instance()
    return instance if instance is not None else QApplication([])


def test_clamp_thumbnail_source_size_fits_inside_512_box() -> None:
    assert clamp_thumbnail_source_size(2048, 1536) == (512, 384)
    assert clamp_thumbnail_source_size(256, 192) == (256, 192)


def test_thumbnail_worker_reuses_disk_cache_without_decoding_source() -> None:
    with tempfile.TemporaryDirectory() as directory:
        image_path = Path(directory) / "frame_001.png"
        cache_dir = Path(directory) / "thumb-cache"
        cv2.imwrite(str(image_path), np.full((16, 16, 3), 255, dtype=np.uint8))

        first_results: list[object] = []
        first = ThumbnailLoadRunnable(1, str(image_path), 64, 48, str(cache_dir))
        first.signals.result.connect(lambda *_args: first_results.append(_args[-1]))
        first.run()

        assert first_results
        assert not first_results[0].isNull()
        assert list(cache_dir.glob("*.jpg"))

        second_results: list[object] = []
        second = ThumbnailLoadRunnable(2, str(image_path), 64, 48, str(cache_dir))
        second.signals.result.connect(lambda *_args: second_results.append(_args[-1]))
        with patch.object(thumbnails_module, "load_image_color_thumbnail", side_effect=AssertionError("cache miss")):
            second.run()

        assert second_results
        assert not second_results[0].isNull()


def test_thumbnail_worker_preserves_aspect_ratio_and_caches_each_lod_size() -> None:
    with tempfile.TemporaryDirectory() as directory:
        image_path = Path(directory) / "square.png"
        cache_dir = Path(directory) / "thumb-cache"
        cv2.imwrite(str(image_path), np.full((400, 400, 3), 255, dtype=np.uint8))

        first_results: list[object] = []
        first = ThumbnailLoadRunnable(1, str(image_path), 64, 48, str(cache_dir))
        first.signals.result.connect(lambda *_args: first_results.append(_args[-1]))
        first.run()

        second_results: list[object] = []
        second = ThumbnailLoadRunnable(1, str(image_path), 128, 96, str(cache_dir))
        second.signals.result.connect(lambda *_args: second_results.append(_args[-1]))
        second.run()

        assert first_results[0].width() == 64
        assert first_results[0].height() == 64
        assert second_results[0].width() == 128
        assert second_results[0].height() == 128
        assert len(list(cache_dir.glob("*.jpg"))) == 2


def test_thumbnail_worker_supports_high_lod_from_large_source() -> None:
    with tempfile.TemporaryDirectory() as directory:
        image_path = Path(directory) / "large_square.png"
        cache_dir = Path(directory) / "thumb-cache"
        cv2.imwrite(str(image_path), np.full((2000, 2000, 3), 255, dtype=np.uint8))

        results: list[object] = []
        worker = ThumbnailLoadRunnable(1, str(image_path), 512, 384, str(cache_dir))
        worker.signals.result.connect(lambda *_args: results.append(_args[-1]))
        worker.run()

        assert results[0].width() == 512
        assert results[0].height() == 512


def test_thumbnail_worker_profiles_gap_between_load_starts(capsys) -> None:
    with tempfile.TemporaryDirectory() as directory:
        image_path = Path(directory) / "frame_001.png"
        cache_dir = Path(directory) / "thumb-cache"
        cv2.imwrite(str(image_path), np.full((16, 16, 3), 255, dtype=np.uint8))
        ThumbnailLoadRunnable._previous_start_at = None

        with patch.dict(os.environ, {"CONTOUR_PROFILE_THUMBNAIL": "1", "CONTOUR_PROFILE_THUMBNAIL_FULL": "1"}):
            ThumbnailLoadRunnable(1, str(image_path), 64, 48, str(cache_dir)).run()
            ThumbnailLoadRunnable(1, str(image_path), 64, 48, str(cache_dir)).run()

        output = capsys.readouterr().out
        assert "[contour thumbnail profiling]" in output
        assert "[contour thumbnail profiling stats]" in output
        assert "full_function_usage" in output
        assert "since_previous_start=<first>" in output
        assert "since_previous_start=" in output
        assert "cache=hit" in output


def test_widget_clears_thumbnail_cache_when_base_folder_changes() -> None:
    _app()
    widget = PolygonExtractionWidget()
    try:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "thumb-cache"
            cache_dir.mkdir()
            stale_file = cache_dir / "stale.png"
            stale_file.write_bytes(b"stale")
            (cache_dir / "cache.key").write_text(str(Path(directory) / "old"), encoding="utf-8")
            widget._thumbnail_disk_cache_dir = cache_dir

            new_image = str(Path(directory) / "new" / "frame_001.png")
            widget._reset_thumbnail_disk_cache_for_base_paths([new_image])

            assert not stale_file.exists()
            assert (cache_dir / "cache.key").read_text(encoding="utf-8") == str(Path(new_image).parent)
    finally:
        widget.close()
        widget.deleteLater()
        _app().processEvents()


def test_widget_creates_thumbnail_cache_directory_on_startup() -> None:
    _app()
    with tempfile.TemporaryDirectory() as directory:
        expected_cache_dir = Path(directory) / "contour-frame-thumbnails"
        with patch.object(widget_module.tempfile, "gettempdir", return_value=directory):
            widget = PolygonExtractionWidget()
            try:
                assert widget._thumbnail_disk_cache_dir == expected_cache_dir
                assert expected_cache_dir.is_dir()
            finally:
                widget.close()
                widget.deleteLater()
                _app().processEvents()



def test_thumbnail_worker_creates_cache_directory_before_source_decode() -> None:
    with tempfile.TemporaryDirectory() as directory:
        cache_dir = Path(directory) / "thumb-cache"
        missing_image = Path(directory) / "missing.png"

        worker = ThumbnailLoadRunnable(1, str(missing_image), 64, 48, str(cache_dir))
        worker.run()

        assert cache_dir.is_dir()
