from __future__ import annotations

import threading

import pytest

from kraken_core.frame_matrix import (
    InvalidationSelector,
    StoreCapability,
    StoreFull,
    StoreNamespace,
    StorePolicy,
    ThumbnailCacheCoordinator,
    ThumbnailKey,
    ThumbnailRecord,
    ThumbnailStoreFactory,
)
from kraken_core.frame_matrix.adapters.filesystem import FilesystemThumbnailStore
from kraken_core.frame_matrix.adapters.memory import MemoryThumbnailStore
from kraken_core.frame_matrix.adapters.sqlite import ShardedSQLiteThumbnailStore


def _key(index: int, *, revision: str = "r1") -> ThumbnailKey:
    return ThumbnailKey(
        source_key=f"frame-{index}",
        source_revision=revision,
        lod=index % 3,
        width=64,
        height=48,
        codec="png",
        renderer_fingerprint="base-v1",
    )


def _record(index: int, *, revision: str = "r1") -> ThumbnailRecord:
    return ThumbnailRecord(_key(index, revision=revision), f"thumbnail-{index}".encode(), metadata={"index": index})


@pytest.fixture(params=["memory", "files", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        value = MemoryThumbnailStore()
    elif request.param == "files":
        value = FilesystemThumbnailStore(tmp_path / "files")
    else:
        value = ShardedSQLiteThumbnailStore(tmp_path / "sqlite")
    value.open(StoreNamespace("tests", project="p1"), StorePolicy(batch_delay_ms=1))
    try:
        yield value
    finally:
        value.close()


def test_thumbnail_key_is_stable_and_tracks_render_inputs() -> None:
    first = _key(1)
    same = _key(1)
    changed = _key(1, revision="r2")

    assert first.digest() == same.digest()
    assert first.digest() != changed.digest()
    assert len(first.digest()) == 32


def test_store_contract_put_get_batch_invalidate_and_clear(store) -> None:
    records = tuple(_record(index) for index in range(8))
    store.put(records[0])
    store.put_many(records[1:])

    assert store.get(records[0].key).payload == records[0].payload
    assert set(store.get_many(record.key for record in records[1:])) == {record.key for record in records[1:]}
    assert store.stats().entries == 8
    assert store.invalidate(InvalidationSelector(source_revision="r1")) == 8
    assert store.stats().entries == 0

    store.put(_record(20))
    store.clear_namespace()
    assert store.get(_key(20)) is None


def test_store_namespaces_are_isolated(store) -> None:
    store.put(_record(1))
    store.open(StoreNamespace("tests", project="p2"), StorePolicy(batch_delay_ms=1))
    assert store.get(_key(1)) is None
    store.put(_record(2))
    store.open(StoreNamespace("tests", project="p1"), StorePolicy(batch_delay_ms=1))
    assert store.get(_key(1)) is not None
    assert store.get(_key(2)) is None


def test_store_rejects_oversized_entries(store) -> None:
    store.open(StoreNamespace("tests", project="small"), StorePolicy(max_entry_bytes=3, batch_delay_ms=1))
    with pytest.raises(StoreFull):
        store.put(ThumbnailRecord(_key(1), b"four"))


def test_filesystem_adapter_uses_hash_directories_and_one_journal(tmp_path) -> None:
    root = tmp_path / "files"
    store = FilesystemThumbnailStore(root)
    namespace = StoreNamespace("tests", project="files")
    store.open(namespace)
    store.put_many(_record(index) for index in range(20))

    namespace_root = root / namespace.digest()
    payloads = [path for path in namespace_root.rglob("*") if path.is_file() and path.suffix == ".png"]
    store.close()

    assert len(payloads) == 20
    assert (namespace_root / "namespace.journal").is_file()
    assert not list(namespace_root.rglob("*.meta.json"))


def test_sqlite_adapter_keeps_payloads_in_bounded_shards(tmp_path) -> None:
    root = tmp_path / "sqlite"
    store = ShardedSQLiteThumbnailStore(root)
    namespace = StoreNamespace("tests", project="sqlite")
    store.open(namespace, StorePolicy(batch_delay_ms=1))
    store.put_many(_record(index) for index in range(300))
    store.compact()
    store.close()

    namespace_root = root / namespace.digest()
    databases = list(namespace_root.glob("*.sqlite3"))

    assert (namespace_root / "catalog.sqlite3").is_file()
    assert len(databases) <= 65
    assert not list(namespace_root.rglob("*.png"))


def test_factory_supports_builtin_schemes_and_lazy_paths(tmp_path) -> None:
    factory = ThumbnailStoreFactory()

    assert {"memory", "files", "sqlite"} <= set(factory.schemes())
    memory = factory.create("memory://")
    files = factory.create(f"files:///{(tmp_path / 'files').as_posix()}")
    sqlite = factory.create(f"sqlite:///{(tmp_path / 'sqlite').as_posix()}")

    assert memory.capabilities & StoreCapability.BATCH_READ
    assert files.capabilities & StoreCapability.ATOMIC_REPLACE
    assert sqlite.capabilities & StoreCapability.INTEGRITY_CHECK


def test_cache_coordinator_deduplicates_inflight_requests() -> None:
    store = MemoryThumbnailStore()
    store.open(StoreNamespace("tests", project="cache"))
    coordinator = ThumbnailCacheCoordinator(store, workers=2)
    calls = 0
    release = threading.Event()

    def produce() -> bytes:
        nonlocal calls
        calls += 1
        release.wait(timeout=2)
        return b"encoded"

    first = coordinator.request(_key(1), produce)
    second = coordinator.request(_key(1), produce)
    release.set()

    assert first is second
    assert first.result(timeout=3).payload == b"encoded"
    assert calls == 1
    coordinator.close()
