from __future__ import annotations

import cProfile
import hashlib
import io
import os
import pstats
from pathlib import Path
from threading import Lock
from time import perf_counter

from PyQt6.QtCore import QObject, Qt, QRunnable, pyqtSignal
from PyQt6.QtGui import QImage

from ...infrastructure.profiling import (
    thumbnail_full_function_usage_enabled,
    thumbnail_profiling_enabled,
    thumbnail_top_lines,
    try_disable_profiler,
    try_enable_profiler,
)
from ...utils import load_image_color_thumbnail
from .image_conversion import cv_to_qimage


class ThumbnailLoadSignals(QObject):
    result = pyqtSignal(int, str, int, int, object)
    finished = pyqtSignal(int, str)


class ThumbnailLoadRunnable(QRunnable):
    _profile_lock = Lock()
    _previous_start_at: float | None = None
    _cache_directory_lock = Lock()
    _prepared_cache_directories: set[str] = set()

    def __init__(self, generation: int, path: str, width: int, height: int, cache_directory: str | None = None) -> None:
        super().__init__()
        self.generation = int(generation)
        self.path = str(path)
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.cache_directory = "" if cache_directory is None else str(cache_directory)
        self.signals = ThumbnailLoadSignals()

    def _cache_path(self) -> Path | None:
        if not self.cache_directory:
            return None
        cache_root = Path(self.cache_directory)
        cache_directory_key = str(cache_root)
        if cache_directory_key not in self.__class__._prepared_cache_directories:
            with self.__class__._cache_directory_lock:
                if cache_directory_key not in self.__class__._prepared_cache_directories:
                    try:
                        cache_root.mkdir(parents=True, exist_ok=True)
                    except OSError:
                        return None
                    self.__class__._prepared_cache_directories.add(cache_directory_key)
        try:
            stat = os.stat(self.path)
        except OSError:
            return None
        digest = hashlib.sha256(
            f"{self.path}|{stat.st_size}|{stat.st_mtime_ns}|{self.width}x{self.height}".encode(
                "utf-8",
                errors="surrogatepass",
            )
        ).hexdigest()
        return cache_root / f"{digest}.jpg"

    def _load_cached_qimage(self, cache_path: Path | None) -> QImage | None:
        if cache_path is None or not cache_path.is_file():
            return None
        qimage = QImage(str(cache_path))
        if qimage.isNull():
            return None
        if qimage.width() < self.width or qimage.height() < self.height:
            return None
        return qimage if not qimage.isNull() else None

    def _save_cached_qimage(self, cache_path: Path | None, qimage: QImage) -> None:
        if cache_path is None or qimage.isNull():
            return
        try:
            tmp_path = cache_path.with_suffix(".tmp.jpg")
            if qimage.save(str(tmp_path), "JPG", 82):
                os.replace(tmp_path, cache_path)
        except Exception:
            return

    def _profile_start(self, started_at: float) -> float | None:
        with self._profile_lock:
            previous_start_at = self.__class__._previous_start_at
            self.__class__._previous_start_at = started_at
        if previous_start_at is None:
            return None
        return (started_at - previous_start_at) * 1000.0

    def _print_profile(
        self,
        *,
        started_at: float,
        since_previous_start_ms: float | None,
        cache_status: str,
        qimage: QImage | None,
        profiler: cProfile.Profile | None,
    ) -> None:
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        previous = "<first>" if since_previous_start_ms is None else f"{since_previous_start_ms:.3f}ms"
        output_width = 0 if qimage is None or qimage.isNull() else qimage.width()
        output_height = 0 if qimage is None or qimage.isNull() else qimage.height()
        print(
            "[contour thumbnail profiling] "
            f"image={Path(self.path).name} generation={self.generation} "
            f"request={self.width}x{self.height} output={output_width}x{output_height} "
            f"cache={cache_status} load={elapsed_ms:.3f}ms since_previous_start={previous}",
            flush=True,
        )
        if profiler is None or not profiler.getstats():
            print(
                f"[contour thumbnail profiling stats] image={Path(self.path).name} cprofile_skipped=yes",
                flush=True,
            )
            return
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
        if thumbnail_full_function_usage_enabled():
            stats.print_stats()
            stats_detail = "full_function_usage"
        else:
            top_lines = thumbnail_top_lines()
            stats.print_stats(top_lines)
            stats_detail = f"top={top_lines}"
        print(
            f"[contour thumbnail profiling stats] image={Path(self.path).name} {stats_detail}\n"
            f"{stream.getvalue()}",
            flush=True,
        )

    def _load_thumbnail(self) -> tuple[QImage | None, str]:
        cache_path = self._cache_path()
        qimage = self._load_cached_qimage(cache_path)
        if qimage is None:
            cache_status = "miss" if cache_path is not None else "none"
            image = load_image_color_thumbnail(self.path, self.width, self.height, cover=True)
            qimage = cv_to_qimage(image)
            if not qimage.isNull():
                if qimage.width() > self.width and qimage.height() > self.height:
                    qimage = qimage.scaled(
                        self.width,
                        self.height,
                        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        Qt.TransformationMode.FastTransformation,
                    )
                self._save_cached_qimage(cache_path, qimage)
            return qimage, cache_status
        return qimage, "hit"

    def run(self) -> None:
        qimage: QImage | None = None
        profiling_enabled = thumbnail_profiling_enabled()
        started_at = perf_counter()
        since_previous_start_ms = self._profile_start(started_at) if profiling_enabled else None
        cache_status = "disabled"
        profiler: cProfile.Profile | None = None
        try:
            if profiling_enabled:
                candidate = cProfile.Profile()
                if try_enable_profiler(candidate):
                    profiler = candidate
                    try:
                        qimage, cache_status = self._load_thumbnail()
                    finally:
                        try_disable_profiler(candidate)
                else:
                    qimage, cache_status = self._load_thumbnail()
            else:
                qimage, cache_status = self._load_thumbnail()
        except Exception:
            qimage = None
            cache_status = "error"
        if profiling_enabled:
            self._print_profile(
                started_at=started_at,
                since_previous_start_ms=since_previous_start_ms,
                cache_status=cache_status,
                qimage=qimage,
                profiler=profiler,
            )
        try:
            self.signals.result.emit(self.generation, self.path, self.width, self.height, qimage)
        except RuntimeError:
            return
        try:
            self.signals.finished.emit(self.generation, self.path)
        except RuntimeError:
            return
