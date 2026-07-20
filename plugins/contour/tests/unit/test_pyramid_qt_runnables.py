from __future__ import annotations

import numpy as np
from PyQt6 import sip

from contour.adapters.qt.pyramid import PyramidFrameLoadRunnable, PyramidThumbnailLoadRunnable


class _PyramidStoreStub:
    def get_frame(self, _frame_id: int, _lod: int) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def get_thumbnail(self, _frame_id: int, _lod: int, *, max_size: int) -> np.ndarray:
        del max_size
        return np.zeros((8, 8, 3), dtype=np.uint8)


class _FailingPyramidStoreStub(_PyramidStoreStub):
    def get_thumbnail(self, _frame_id: int, _lod: int, *, max_size: int) -> np.ndarray:
        del max_size
        raise OSError("cancelled load")


def test_frame_runnable_ignores_deleted_signal_owner() -> None:
    runnable = PyramidFrameLoadRunnable(1, 0, 0, _PyramidStoreStub())  # type: ignore[arg-type]
    sip.delete(runnable.signals)

    runnable.run()


def test_thumbnail_runnable_ignores_deleted_signal_owner_after_success() -> None:
    runnable = PyramidThumbnailLoadRunnable(1, 0, 0, _PyramidStoreStub(), 8, 8)  # type: ignore[arg-type]
    sip.delete(runnable.signals)

    runnable.run()


def test_thumbnail_runnable_ignores_deleted_signal_owner_after_error() -> None:
    runnable = PyramidThumbnailLoadRunnable(1, 0, 0, _FailingPyramidStoreStub(), 8, 8)  # type: ignore[arg-type]
    sip.delete(runnable.signals)

    runnable.run()
