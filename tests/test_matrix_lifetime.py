"""Run native Qt lifetime regressions in children with Windows crash UI disabled."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def run_qt_child(script: str) -> None:
    bootstrap = """
import os
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
import faulthandler
faulthandler.enable()
"""
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    root = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = str(root / "src")
    child = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(bootstrap) + textwrap.dedent(script)],
        cwd=root, env=environment, capture_output=True, text=True, timeout=35,
    )
    assert child.returncode == 0, child.stdout + child.stderr


def test_parent_deletion_stops_matrix_background_resources():
    run_qt_child("""
        import time
        from PyQt6 import sip
        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtWidgets import QApplication, QWidget
        from kraken_core.frame_matrix.qt import FrameMatrixWidget
        from kraken_core.frame_matrix.models import MatrixSession, MatrixViewportResult
        from kraken_core.frame_matrix.adapters.memory import MemoryThumbnailStore

        app = QApplication([])
        class Source:
            def load_viewport(self, request, cancellation=None):
                return MatrixViewportResult(request)
        parent = QWidget()
        matrix = FrameMatrixWidget(parent=parent, data_source=Source(), thumbnail_store=MemoryThumbnailStore())
        matrix.set_session(MatrixSession("lifetime", 10, 10))
        matrix._request_visible_viewport()
        cache_threads = list(matrix._thumbnail_cache._threads)
        executor = matrix._viewport_executor
        time.sleep(0.05)
        parent.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert sip.isdeleted(matrix)
        assert all(not thread.is_alive() for thread in cache_threads), "Cache survived parent destruction"
        assert executor._shutdown, "Viewport executor survived parent destruction"
        app.processEvents()
    """)


def test_unreferenced_matrix_wrappers_are_released_without_gc_cycles():
    run_qt_child("""
        import gc, weakref
        from PyQt6.QtWidgets import QApplication
        from kraken_core.frame_matrix.qt import FrameMatrixView, FrameMatrixWidget, FrameMatrixMinimap
        app = QApplication([])
        for kind in (FrameMatrixView, FrameMatrixWidget):
            for iteration in range(30):
                matrix = kind(10, 10)
                minimap = FrameMatrixMinimap(matrix)
                matrix.show()
                app.processEvents()
                reference = weakref.ref(matrix)
                del minimap
                del matrix
                # Ownership must not rely on cyclic GC destroying Python graphics
                # items in an arbitrary order while their C++ scene is still live.
                assert reference() is None, "Graphics items retained their matrix owner"
                gc.collect()
                app.processEvents()
    """)


def test_parent_deletion_with_pending_thumbnail_does_not_retain_qt_wrapper():
    run_qt_child("""
        import threading, weakref
        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtWidgets import QApplication, QWidget
        from kraken_core.frame_matrix.qt import FrameMatrixWidget
        from kraken_core.frame_matrix.models import MatrixSession, MatrixViewportResult, MatrixItem, MatrixAssetRef
        from kraken_core.frame_matrix.adapters.memory import MemoryThumbnailStore
        app = QApplication([])
        started, release = threading.Event(), threading.Event()
        class Assets:
            def load_asset(self, reference, *, width, height, cancellation=None):
                started.set()
                assert release.wait(5)
                return b"encoded"
        parent = QWidget()
        matrix = FrameMatrixWidget(parent=parent, asset_source=Assets(), thumbnail_store=MemoryThumbnailStore())
        matrix.set_session(MatrixSession("pending-asset", 10, 10))
        matrix._apply_viewport_result(MatrixViewportResult(matrix._current_viewport_request(), items=(
            MatrixItem("frame-1", 1, 1, asset=MatrixAssetRef("frame-1", "1")),
        )))
        assert started.wait(5)
        threads = list(matrix._thumbnail_cache._threads)
        reference = weakref.ref(matrix)
        release_timer = threading.Timer(0.05, release.set)
        release_timer.start()
        parent.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        release_timer.join()
        del matrix, parent
        assert reference() is None, "Thumbnail worker retained its deleted QWidget"
        assert all(not thread.is_alive() for thread in threads)
        app.processEvents()
    """)


def test_thumbnail_storage_errors_are_delivered_on_gui_poll_only():
    run_qt_child("""
        import threading
        from PyQt6.QtWidgets import QApplication
        from kraken_core.frame_matrix.qt import FrameMatrixWidget
        from kraken_core.frame_matrix.models import MatrixSession
        from kraken_core.frame_matrix.storage import ThumbnailKey, ThumbnailStoreError
        from kraken_core.frame_matrix.adapters.memory import MemoryThumbnailStore
        app = QApplication([])
        class BrokenStore(MemoryThumbnailStore):
            def get(self, key):
                raise ThumbnailStoreError("read failed")
        matrix = FrameMatrixWidget(thumbnail_store=BrokenStore())
        matrix.set_session(MatrixSession("errors", 10, 10))
        received = []
        matrix.errorOccurred.connect(lambda message: received.append((message, threading.get_ident())))
        key = ThumbnailKey("frame-1", "1", 0, 10, 10, "png")
        future = matrix._thumbnail_cache.request(key, lambda: b"encoded")
        assert future.result(5).payload == b"encoded"
        assert received == [], "Worker directly invoked a QWidget signal"
        matrix._poll_background_results()
        assert received == [("read failed", threading.get_ident())]
        matrix.close()
        del matrix
        app.processEvents()
    """)
