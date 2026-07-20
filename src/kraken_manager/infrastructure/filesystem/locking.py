from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class ProjectLockTimeout(TimeoutError):
    pass


class ProjectFileLock:
    """Portable cooperative inter-process lock based on O_EXCL creation."""

    def __init__(self, path: str | Path, *, poll_interval: float = 0.05) -> None:
        self.path = Path(path)
        self.poll_interval = poll_interval
        self._guard = threading.Lock()
        self._owner_thread: int | None = None
        self._depth = 0
        self._token: str | None = None

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
                "schema_version": 1,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "thread": thread_id,
                "token": token,
                "created_unix": time.time(),
            },
            sort_keys=True,
        ).encode("utf-8")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise ProjectLockTimeout(f"timed out waiting for project lock {self.path}")
                time.sleep(self.poll_interval)
                continue

            try:
                os.write(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
            with self._guard:
                self._owner_thread = thread_id
                self._depth = 1
                self._token = token
            return

    def release(self) -> None:
        thread_id = threading.get_ident()
        with self._guard:
            if self._owner_thread != thread_id or self._depth == 0:
                raise RuntimeError("project lock may only be released by its owner")
            self._depth -= 1
            if self._depth:
                return
            token = self._token
            self._owner_thread = None
            self._token = None

        # Never unlink a replacement lock if ownership was externally disturbed.
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return
        if current.get("token") == token:
            self.path.unlink(missing_ok=True)

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
