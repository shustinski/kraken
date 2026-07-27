"""Storage-agnostic RAM/disk thumbnail cache coordinator."""

from __future__ import annotations

import itertools
import queue
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future

from .storage import (
    StoreCapability,
    ThumbnailKey,
    ThumbnailRecord,
    ThumbnailStore,
    ThumbnailStoreError,
)


ThumbnailProducer = Callable[[], ThumbnailRecord | bytes]


class ThumbnailCacheCoordinator:
    """Combine RAM LRU, a pluggable store and de-duplicated background work."""

    def __init__(
        self,
        store: ThumbnailStore,
        *,
        ram_items: int = 512,
        workers: int = 3,
        error_handler: Callable[[Exception], None] | None = None,
    ) -> None:
        self.store = store
        self.ram_items = max(1, int(ram_items))
        self.error_handler = error_handler
        self._ram: OrderedDict[ThumbnailKey, ThumbnailRecord] = OrderedDict()
        self._inflight: dict[ThumbnailKey, Future[ThumbnailRecord]] = {}
        self._lock = threading.RLock()
        self._queue: queue.PriorityQueue[tuple[int, int, object]] = queue.PriorityQueue()
        self._sequence = itertools.count()
        self._stop_marker = object()
        self._generation = 0
        self._threads = [
            threading.Thread(target=self._worker, name=f"kraken-thumbnail-cache-{index}", daemon=True)
            for index in range(max(1, int(workers)))
        ]
        for thread in self._threads:
            thread.start()

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def change_generation(self, generation: int | None = None) -> int:
        with self._lock:
            self._generation = self._generation + 1 if generation is None else int(generation)
            for future in self._inflight.values():
                future.cancel()
            self._inflight.clear()
            return self._generation

    def get(self, key: ThumbnailKey) -> ThumbnailRecord | None:
        with self._lock:
            cached = self._ram.get(key)
            if cached is not None:
                self._ram.move_to_end(key)
                return cached
        try:
            record = self.store.get(key)
        except ThumbnailStoreError as exc:
            self._report_error(exc)
            return None
        if record is not None:
            self._remember(record)
        return record

    def get_many(self, keys: Iterable[ThumbnailKey]) -> Mapping[ThumbnailKey, ThumbnailRecord]:
        pending: list[ThumbnailKey] = []
        found: dict[ThumbnailKey, ThumbnailRecord] = {}
        with self._lock:
            for key in keys:
                cached = self._ram.get(key)
                if cached is None:
                    pending.append(key)
                else:
                    self._ram.move_to_end(key)
                    found[key] = cached
        if not pending:
            return found
        try:
            if self.store.capabilities & StoreCapability.BATCH_READ:
                loaded = self.store.get_many(pending)
            else:
                loaded = {
                    key: record
                    for key in pending
                    if (record := self.store.get(key)) is not None
                }
        except ThumbnailStoreError as exc:
            self._report_error(exc)
            return found
        for record in loaded.values():
            self._remember(record)
        found.update(loaded)
        return found

    def request(
        self,
        key: ThumbnailKey,
        producer: ThumbnailProducer,
        *,
        priority: int = 100,
        generation: int | None = None,
    ) -> Future[ThumbnailRecord]:
        requested_generation = self.generation if generation is None else int(generation)
        with self._lock:
            cached = self._ram.get(key)
            if cached is not None:
                self._ram.move_to_end(key)
                ready: Future[ThumbnailRecord] = Future()
                ready.set_result(cached)
                return ready
            existing = self._inflight.get(key)
            if existing is not None and not existing.done():
                return existing
            future: Future[ThumbnailRecord] = Future()
            self._inflight[key] = future
            task = (key, producer, future, requested_generation)
            self._queue.put((int(priority), next(self._sequence), task))
            return future

    def _worker(self) -> None:
        while True:
            _priority, _sequence, task = self._queue.get()
            if task is self._stop_marker:
                return
            key, producer, future, generation = task
            if future.cancelled() or generation != self.generation:
                future.cancel()
                self._forget_inflight(key, future)
                continue
            try:
                try:
                    cached = self.store.get(key)
                except ThumbnailStoreError as exc:
                    self._report_error(exc)
                    cached = None
                if cached is not None:
                    self._remember(cached)
                    if not future.cancelled():
                        future.set_result(cached)
                    continue
                produced = producer()
                record = (
                    produced
                    if isinstance(produced, ThumbnailRecord)
                    else ThumbnailRecord(key=key, payload=bytes(produced), codec=key.codec)
                )
                if record.key != key:
                    raise ValueError("thumbnail producer returned a record for a different key")
                if generation != self.generation:
                    future.cancel()
                    continue
                self._remember(record)
                try:
                    self.store.put(record)
                except ThumbnailStoreError as exc:
                    self._report_error(exc)
                if not future.cancelled():
                    future.set_result(record)
            except BaseException as exc:
                if not future.cancelled():
                    future.set_exception(exc)
            finally:
                self._forget_inflight(key, future)

    def _remember(self, record: ThumbnailRecord) -> None:
        with self._lock:
            self._ram[record.key] = record
            self._ram.move_to_end(record.key)
            while len(self._ram) > self.ram_items:
                self._ram.popitem(last=False)

    def _forget_inflight(self, key: ThumbnailKey, future: Future[ThumbnailRecord]) -> None:
        with self._lock:
            if self._inflight.get(key) is future:
                self._inflight.pop(key, None)

    def _report_error(self, error: Exception) -> None:
        if self.error_handler is not None:
            self.error_handler(error)

    def clear_ram(self) -> None:
        with self._lock:
            self._ram.clear()

    def close(self) -> None:
        self.change_generation()
        for _thread in self._threads:
            self._queue.put((0, next(self._sequence), self._stop_marker))
        for thread in self._threads:
            thread.join(timeout=3.0)
        self._threads.clear()
        self.store.close()


__all__ = ["ThumbnailCacheCoordinator", "ThumbnailProducer"]
