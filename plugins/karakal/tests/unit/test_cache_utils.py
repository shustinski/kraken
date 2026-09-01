from __future__ import annotations

import os

import numpy as np

from karakal.core.cache_utils import ByteLruCache, atomic_pickle_dump, estimate_size_bytes, trim_directory_by_bytes


def test_atomic_pickle_dump_replaces_complete_payload(tmp_path) -> None:
    target = tmp_path / "entry.pickle"

    atomic_pickle_dump(target, {"values": np.arange(8, dtype=np.float32)})

    assert target.is_file()
    assert target.stat().st_size > 0
    assert not tuple(tmp_path.glob("*.tmp"))


def test_trim_directory_enforces_byte_limit_oldest_first(tmp_path) -> None:
    older = tmp_path / "older.pickle"
    newer = tmp_path / "newer.pickle"
    older.write_bytes(b"a" * 20)
    newer.write_bytes(b"b" * 20)
    os.utime(older, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    os.utime(newer, ns=(1_700_000_001_000_000_000, 1_700_000_001_000_000_000))

    removed_files, removed_bytes = trim_directory_by_bytes(tmp_path, max_bytes=20)

    assert (removed_files, removed_bytes) == (1, 20)
    assert not older.exists()
    assert newer.exists()


def test_estimate_size_counts_numpy_buffers() -> None:
    values = np.zeros((8, 8), dtype=np.float32)

    assert estimate_size_bytes(values) == values.nbytes


def test_byte_lru_evicts_oldest_entry_by_retained_size() -> None:
    cache: ByteLruCache[str, bytes] = ByteLruCache(6, size_of=len)
    cache.put("old", b"1234")
    cache.put("new", b"5678")

    assert cache.get("old") is None
    assert cache.get("new") == b"5678"
    assert cache.total_bytes == 4
