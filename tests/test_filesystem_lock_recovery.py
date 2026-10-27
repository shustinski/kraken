"""Real process death, legacy clients and competing writers, without stale unlink races."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from kraken_manager.infrastructure.filesystem.event_store import FilesystemEventStore
from kraken_manager.infrastructure.filesystem.locking import ProjectFileLock, ProjectLockTimeout


CHILD = r"""
import json, os, socket, sys, time
from pathlib import Path
from kraken_manager.infrastructure.filesystem.locking import ProjectFileLock
from kraken_manager.infrastructure.filesystem.event_store import FilesystemEventStore
path, mode = Path(sys.argv[1]), sys.argv[2]
if mode == 'legacy':
    fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(dict(schema_version=1, pid=os.getpid(), host=socket.gethostname(), token='legacy'), stream)
        stream.flush(); os.fsync(stream.fileno())
    print('READY', flush=True)
    sys.stdin.read(1)
    path.unlink()
elif mode == 'crash':
    store = FilesystemEventStore(path, 'recovery')
    store.lock.acquire()
    store.append('a', expected_revision=0, events=[{'payload': 'committed before death'}])
    print('READY', flush=True)
    sys.stdin.read(1)
elif mode == 'counter':
    counter = path.with_name('counter')
    for _ in range(15):
        with ProjectFileLock(path, poll_interval=.002):
            value = int(counter.read_text())
            time.sleep(.002)
            counter.write_text(str(value + 1))
else:
    raise ValueError(mode)
"""


def child(path, mode):
    return subprocess.Popen(
        [sys.executable, "-u", "-c", CHILD, str(path), mode],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def finish(process):
    if process.poll() is None:
        process.kill()
    process.communicate(timeout=10)


def test_process_death_releases_kernel_gate_and_preserves_committed_events(tmp_path):
    process = child(tmp_path, "crash")
    try:
        assert process.stdout.readline().strip() == "READY"
        store = FilesystemEventStore(tmp_path, "recovery")
        with pytest.raises(ProjectLockTimeout):
            store.lock.acquire(timeout=0.05)
        process.kill()
        process.wait(timeout=10)
        with store.lock.hold(timeout=2):
            assert store.current_revision("a") == 1
            store.append("a", expected_revision=1, events=[{"payload": "after death"}])
        assert store.verify() == (2, 1)
        assert not store.layout.lock_path.exists()
        assert store.layout.lock_path.with_name("lock.gate").is_file()
    finally:
        finish(process)


def test_live_legacy_owner_is_respected_then_dead_owner_reclaimed(tmp_path):
    path = tmp_path / "lock"
    process = child(path, "legacy")
    try:
        assert process.stdout.readline().strip() == "READY"
        original = path.read_bytes()
        with pytest.raises(ProjectLockTimeout):
            ProjectFileLock(path).acquire(timeout=0.05)
        assert path.read_bytes() == original
        process.kill()
        process.wait(timeout=10)
        with ProjectFileLock(path).hold(timeout=2):
            assert json.loads(path.read_text())["pid"] == os.getpid()
        assert not path.exists()
    finally:
        finish(process)


def test_competing_processes_and_instances_never_share_critical_section(tmp_path):
    (tmp_path / "counter").write_text("0")
    processes = [child(tmp_path / "lock", "counter") for _ in range(4)]
    try:
        for process in processes:
            _, error = process.communicate(timeout=30)
            assert process.returncode == 0, error
        assert (tmp_path / "counter").read_text() == "60"
    finally:
        for process in processes:
            finish(process)


@pytest.mark.parametrize(
    "data",
    [
        b"partial legacy record",
        json.dumps(dict(schema_version=1, pid=os.getpid(), host="another-host", token="foreign")).encode(),
    ],
)
def test_unknown_legacy_owner_is_never_stolen(tmp_path, data):
    path = tmp_path / "lock"
    path.write_bytes(data)
    with pytest.raises(ProjectLockTimeout):
        ProjectFileLock(path).acquire(timeout=0.05)
    assert path.read_bytes() == data


def test_reentrant_token_and_temporary_cleanup_failure_do_not_strand_live_process(tmp_path, monkeypatch):
    path = tmp_path / "lock"
    lock = ProjectFileLock(path)
    with lock:
        token = lock.held_token
        with lock:
            assert lock.held_token == token
        assert lock.held_token == token
        original_unlink = Path.unlink

        def deny_owned_record(target, *args, **kwargs):
            if target == path:
                raise PermissionError("Temporary Windows sharing denial")
            return original_unlink(target, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", deny_owned_record)
    assert lock.held_token is None and path.exists()
    monkeypatch.setattr(Path, "unlink", original_unlink)
    with ProjectFileLock(path).hold(timeout=2):
        assert json.loads(path.read_text())["token"] != token
    assert not path.exists()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="Requires actual POSIX fork")
def test_forked_child_cannot_reenter_or_release_parent_lock(tmp_path):
    # Run fork in a dedicated interpreter so pytest/plugin threads are not
    # themselves forked. Both inherited ownership and inherited guard must reset.
    script = r"""
import os, threading, sys
from pathlib import Path
from kraken_manager.infrastructure.filesystem.locking import ProjectFileLock, ProjectLockTimeout
lock = ProjectFileLock(Path(sys.argv[1]) / 'lock')
with lock:
    parent_token = lock.held_token
    lock._guard.acquire()
    child = os.fork()
    if child == 0:
        try:
            assert lock.held_token is None
            try:
                lock.release()
            except RuntimeError:
                pass
            else:
                raise AssertionError('child released parent lock')
            for contender in (lock, ProjectFileLock(lock.path)):
                try:
                    contender.acquire(timeout=0.05)
                except ProjectLockTimeout:
                    pass
                else:
                    raise AssertionError('child entered parent critical section')
        except BaseException:
            import traceback
            traceback.print_exc()
            os._exit(1)
        os._exit(0)
    lock._guard.release()
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert lock.held_token == parent_token
with ProjectFileLock(lock.path).hold(timeout=0.1):
    pass
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not hasattr(os, "fork"), reason="Requires actual POSIX fork")
def test_live_forked_child_does_not_keep_dead_parent_gate_locked(tmp_path):
    script = r'''
import os, sys
from pathlib import Path
from kraken_manager.infrastructure.filesystem.locking import ProjectFileLock
lock = ProjectFileLock(Path(sys.argv[1]) / 'lock')
lock.acquire()
child = os.fork()
if child:
    os._exit(0)
print('CHILD_ALIVE', flush=True)
sys.stdin.read(1)
os._exit(0)
'''
    process = subprocess.Popen([sys.executable, "-u", "-c", script, str(tmp_path)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "CHILD_ALIVE"
        assert process.wait(timeout=5) == 0
        with ProjectFileLock(tmp_path / "lock").hold(timeout=0.5):
            pass
    finally:
        process.communicate(input="x", timeout=5)
