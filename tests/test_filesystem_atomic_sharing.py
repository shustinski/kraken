import os
import threading

import pytest

from kraken_manager.infrastructure.filesystem._atomic import atomic_write_bytes
from kraken_manager.infrastructure.stitch_worker import _write_json


@pytest.mark.skipif(os.name != "nt", reason="Windows delete-sharing semantics")
@pytest.mark.parametrize("kind", ["catalog", "stitch_progress"])
def test_atomic_replace_retries_temporary_reader_without_exposing_partial_bytes(tmp_path, kind):
    target = tmp_path / "checkpoint.json"
    target.write_bytes(b"old checkpoint")
    reader = target.open("rb")
    started = threading.Event()
    done = threading.Event()
    failures = []

    def writer():
        started.set()
        try:
            if kind == "catalog":
                atomic_write_bytes(target, b"complete replacement")
            else:
                _write_json(target, {"completed": 1})
        except BaseException as exc:
            failures.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert started.wait(2)
        assert not done.wait(0.1)
        assert reader.read() == b"old checkpoint"
    finally:
        reader.close()
        thread.join(4)
    assert done.is_set() and not failures
    assert target.read_bytes() == (b"complete replacement" if kind == "catalog" else b'{"completed":1}')
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.part-*"))
