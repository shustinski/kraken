from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from kraken_manager.infrastructure.filesystem.event_store import (
    CorruptEventLogError,
    EventStreamConflict,
    FilesystemEventStore,
)


def test_filtered_read_keeps_positions_isolation_and_canonical_validation(tmp_path):
    store = FilesystemEventStore(tmp_path, "filtered-project")
    store.append("a", expected_revision=0, events=[
        {"event_type": "Other", "payload": {"values": [0]}},
        {"event_type": "Selected", "payload": {"values": [1]}},
        {"type": "Selected", "payload": {"values": [2]}},
    ])
    selected = tuple(store.iter_project(event_types=("Selected",)))
    assert [e.global_position for e in selected] == [2, 3]
    assert tuple(store.iter_project(after_global_position=2, event_types=("Selected",))) == selected[1:]
    assert tuple(store.iter_project(event_types=())) == ()
    selected[0].data["payload"]["values"].append(99)
    assert tuple(store.iter_project(event_types=("Selected",)))[0].data["payload"]["values"] == [1]
    # Filtering must still verify segments containing no selected events.
    segment = next(store.layout.events_dir.glob("*.jsonl"))
    segment.write_bytes(segment.read_bytes().replace(b'"revision":1', b'"revision":9', 1))
    with pytest.raises(CorruptEventLogError):
        tuple(store.iter_project(event_types=("Missing",)))


def test_new_store_instances_parse_only_new_segments_and_keep_cas(tmp_path):
    store = FilesystemEventStore(tmp_path, "cache-project")
    store.append("a", expected_revision=0, events=[{"idempotency_key": "first"}])
    real_loads = json.loads
    with patch("kraken_manager.infrastructure.filesystem.event_store.json.loads", wraps=real_loads) as parse:
        assert store.current_revision("a") == 1
        before = parse.call_count
        reopened = FilesystemEventStore(tmp_path, "cache-project")
        assert len(reopened.load_stream("a")) == 1
        assert len(reopened.find_by_idempotency_key(None, "first")) == 1
        assert reopened.current_revision("missing") == 0
        assert parse.call_count == before
        reopened.append("b", expected_revision=0, events=[{}])
        # The cooperative lock's JSON is also decoded during release.
        after_append = parse.call_count
        assert store.current_revision("b") == 1
        assert parse.call_count == after_append + 1
        with pytest.raises(EventStreamConflict):
            store.append("b", expected_revision=0, events=[{}])
        assert len(list(store.iter_project(after_global_position=1))) == 1


def test_cache_never_exposes_mutable_payload_and_invalidates_changed_old_file(tmp_path):
    store = FilesystemEventStore(tmp_path, "cache-project")
    store.append("a", expected_revision=0, events=[{"payload": {"value": [1]}}])
    record = store.load_stream("a")[0]
    record.data["payload"]["value"].append(2)
    assert store.load_stream("a")[0].data["payload"]["value"] == [1]
    segment = next(store.layout.events_dir.glob("*.jsonl"))
    segment.write_bytes(segment.read_bytes().rstrip(b"\n"))
    for _ in range(2):
        with pytest.raises(CorruptEventLogError):
            store.current_revision("a")
    with pytest.raises(CorruptEventLogError):
        store.append("a", expected_revision=1, events=[{}])


def test_cache_revalidates_deleted_segment_and_partial_tail_after_failure(tmp_path):
    store = FilesystemEventStore(tmp_path, "cache-project")
    store.append("a", expected_revision=0, events=[{}])
    assert store.current_revision("a") == 1
    store.append("a", expected_revision=1, events=[{}, {}])
    segment = sorted(store.layout.events_dir.glob("*.jsonl"))[-1]
    original = segment.read_bytes()
    segment.write_bytes(original.rstrip(b"\n"))
    for _ in range(2):
        with pytest.raises(CorruptEventLogError):
            store.current_revision("a")
    segment.write_bytes(original)
    assert store.current_revision("a") == 3
    sorted(store.layout.events_dir.glob("*.jsonl"))[0].unlink()
    with pytest.raises(CorruptEventLogError):
        list(store.iter_project())


def test_cache_rejects_empty_reverse_range(tmp_path):
    store = FilesystemEventStore(tmp_path, "cache-project")
    (store.layout.events_dir / "00000000000000000001-00000000000000000000.jsonl").write_bytes(b"")
    with pytest.raises(CorruptEventLogError):
        store.current_revision("a")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="Requires actual POSIX fork")
def test_forked_reader_revalidates_after_parent_thread_owned_cache_lock(tmp_path):
    import subprocess
    import sys

    script = r"""
import os, signal, sys, threading
import kraken_manager.infrastructure.filesystem.event_store as module
store = module.FilesystemEventStore(sys.argv[1], 'fork-cache')
store.append('a', expected_revision=0, events=[{'payload': 'canonical'}])
ready, done = threading.Event(), threading.Event()
def owner():
    with module._READ_CACHE_LOCK:
        ready.set()
        done.wait(10)
thread = threading.Thread(target=owner)
thread.start()
assert ready.wait(1)
child = os.fork()
if child == 0:
    signal.alarm(2)
    assert module._READ_CACHE == {}
    assert store.current_revision('a') == 1
    assert store.load_stream('a')[0]['payload'] == 'canonical'
    os._exit(0)
try:
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
finally:
    done.set()
    thread.join()
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
