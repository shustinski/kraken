"""Background ownership independent of the lifetime of a Qt wrapper."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from queue import SimpleQueue

from kraken_core.frame_matrix.cache import ThumbnailCacheCoordinator


class MatrixBackgroundResources:
    def __init__(self, executor: ThreadPoolExecutor) -> None:
        self.executor = executor
        self.cache: ThumbnailCacheCoordinator | None = None
        self.errors: SimpleQueue[Exception] = SimpleQueue()
        self.closed = False

    def close(self, *_args: object) -> None:
        # QObject.destroyed is emitted after its QWidget wrapper becomes invalid.
        # This receiver must never access the widget or any other Qt object.
        if self.closed:
            return
        self.closed = True
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.cache is not None:
            self.cache.close()
            self.cache = None
