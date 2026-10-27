from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ._atomic import atomic_write_bytes


_LOCK_INSTANCES: weakref.WeakSet = weakref.WeakSet()
_OPEN_GATE_FDS: set[int] = set()
_FORK_GUARD = threading.Lock()


def _open_gate(path: Path) -> int:
    with _FORK_GUARD:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        _OPEN_GATE_FDS.add(fd)
        return fd


def _close_gate(fd: int) -> None:
    with _FORK_GUARD:
        _OPEN_GATE_FDS.discard(fd)
        os.close(fd)


def _reset_inherited_locks() -> None:
    # flock belongs to the open file description shared across fork. Never
    # unlock that description here: it would release the parent's lock too.
    # Closing only the child's duplicate also prevents it keeping a dead
    # parent's kernel lock alive indefinitely.
    for fd in _OPEN_GATE_FDS:
        os.close(fd)
    _OPEN_GATE_FDS.clear()
    for lock in _LOCK_INSTANCES:
        lock._guard = threading.Lock()
        lock._owner_thread = None
        lock._depth = 0
        lock._token = None
        lock._gate_fd = None
    _FORK_GUARD.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_FORK_GUARD.acquire, after_in_parent=_FORK_GUARD.release, after_in_child=_reset_inherited_locks
    )


def _kernel_lock(fd: int) -> bool:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:
            if exc.errno not in (11, 13, 36):
                raise
            return False
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _kernel_unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def _known_dead_owner(value: object) -> bool:
    if not isinstance(value, dict) or value.get("host") != socket.gethostname():
        return False
    pid = value.get("pid")
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
        if not handle:
            # Access denied is not evidence that a process has exited.
            return ctypes.get_last_error() == 87  # ERROR_INVALID_PARAMETER
        try:
            return kernel.WaitForSingleObject(handle, 0) == 0  # signalled/exited
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


class ProjectLockTimeout(TimeoutError):
    pass


class ProjectFileLock:
    """Reentrant project lock with OS-released gate and legacy-lock compatibility.

    The persistent .gate inode must never be unlinked: all new clients lock the
    same kernel object. The JSON owner record remains compatible with old O_EXCL
    clients. Dead same-host records are reclaimed only while holding the gate;
    live, unknown and foreign-host owners are never stolen.
    """

    def __init__(self, path: str | Path, *, poll_interval: float = 0.05) -> None:
        self.path = Path(path)
        self.poll_interval = poll_interval
        self._guard = threading.Lock()
        self._owner_thread: int | None = None
        self._depth = 0
        self._token: str | None = None
        self._gate_fd: int | None = None
        _LOCK_INSTANCES.add(self)

    @property
    def held_token(self) -> str | None:
        """Identity of this thread's current outermost cooperative lock hold."""
        with self._guard:
            return self._token if self._owner_thread == threading.get_ident() else None

    def acquire(self, timeout: float | None = 10.0) -> None:
        thread_id = threading.get_ident()
        with self._guard:
            if self._owner_thread == thread_id:
                self._depth += 1
                return

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        token = uuid.uuid4().hex
        payload = json.dumps(
            {
                "schema_version": 2,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "thread": thread_id,
                "token": token,
                "created_unix": time.time(),
                "backend": "kernel-gate",
            },
            sort_keys=True,
        ).encode("utf-8")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        gate_path = self.path.with_name(self.path.name + ".gate")
        fd = _open_gate(gate_path)
        acquired = False
        try:
            while True:
                if _kernel_lock(fd):
                    keep_gate = False
                    try:
                        try:
                            old_bytes = self.path.read_bytes()
                        except FileNotFoundError:
                            old_bytes = None
                        if old_bytes is not None:
                            try:
                                owner = json.loads(old_bytes)
                            except ValueError, UnicodeError:
                                owner = None
                            inactive_kernel_owner = (
                                isinstance(owner, dict)
                                and owner.get("schema_version") == 2
                                and owner.get("backend") == "kernel-gate"
                                and owner.get("host") == socket.gethostname()
                            )
                            if inactive_kernel_owner or _known_dead_owner(owner):
                                # New contenders are excluded by the kernel gate.
                                # Old clients cannot create while this record exists;
                                # after unlink an old-client winner is respected by
                                # the atomic no-overwrite publication below.
                                try:
                                    if self.path.read_bytes() == old_bytes:
                                        self.path.unlink()
                                except FileNotFoundError:
                                    pass
                        try:
                            atomic_write_bytes(self.path, payload, overwrite=False)
                        except FileExistsError:
                            pass
                        else:
                            with self._guard:
                                self._owner_thread = thread_id
                                self._depth = 1
                                self._token = token
                                self._gate_fd = fd
                            keep_gate = acquired = True
                            return
                    finally:
                        if not keep_gate:
                            _kernel_unlock(fd)
                if deadline is not None and time.monotonic() >= deadline:
                    raise ProjectLockTimeout(f"timed out waiting for project lock {self.path}")
                time.sleep(self.poll_interval)
        finally:
            if not acquired:
                _close_gate(fd)

    def release(self) -> None:
        thread_id = threading.get_ident()
        with self._guard:
            if self._owner_thread != thread_id or self._depth == 0:
                raise RuntimeError("project lock may only be released by its owner")
            self._depth -= 1
            if self._depth:
                return
            token = self._token
            gate_fd = self._gate_fd
            self._owner_thread = None
            self._token = None
            self._gate_fd = None

        # Never unlink a replacement lock if ownership was externally disturbed.
        try:
            try:
                current = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError, OSError, ValueError:
                return
            if isinstance(current, dict) and current.get("token") == token:
                try:
                    self.path.unlink(missing_ok=True)
                except PermissionError:
                    # The OS gate still releases. The next gate owner can safely
                    # reclaim this v2 diagnostic record even if our PID is alive.
                    pass
        finally:
            if gate_fd is not None:
                try:
                    _kernel_unlock(gate_fd)
                finally:
                    _close_gate(gate_fd)

    @contextmanager
    def hold(self, timeout: float | None = 10.0) -> Iterator[None]:
        self.acquire(timeout)
        try:
            yield
        finally:
            self.release()

    def __enter__(self) -> ProjectFileLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()
