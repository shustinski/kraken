from __future__ import annotations

import ctypes
import os
import shutil
import tempfile
import threading
import time
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from kraken_manager.infrastructure.auth.local import LocalAccountStore, ScryptPasswordHasher

kraken_defect = "D20"

# Transient Windows sharing errors an external on-access scanner (antivirus) raises
# against a freshly created SQLite file; the lock clears within milliseconds.
_RETRYABLE_CLEANUP_WINERRORS = frozenset({5, 32, 145})
_CLEANUP_RETRY_SECONDS = 10.0
_CLEANUP_RETRY_PAUSE_SECONDS = 0.05


@contextmanager
def _robust_tempdir() -> Iterator[str]:
    """A temp directory whose cleanup retries past transient external file locks.

    On Windows machines with on-access antivirus the scanner may open the just
    created SQLite file without FILE_SHARE_DELETE (deletion is then refused —
    WinError 32) and may stage a transient ``<db>-SHM.tmp`` file in the same
    directory while it is being removed (final rmdir — WinError 145). The lock
    clears within milliseconds, so a bounded retry keeps cleanup deterministic
    without weakening what the tests prove.
    """
    directory = tempfile.mkdtemp()
    try:
        yield directory
    finally:
        deadline = time.monotonic() + _CLEANUP_RETRY_SECONDS
        while True:
            try:
                shutil.rmtree(directory)
                return
            except FileNotFoundError:
                return
            except OSError as exc:
                if (
                    getattr(exc, "winerror", None) not in _RETRYABLE_CLEANUP_WINERRORS
                    or time.monotonic() >= deadline
                ):
                    raise
                time.sleep(_CLEANUP_RETRY_PAUSE_SECONDS)


def _kernel32() -> ctypes.WinDLL:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    return kernel32


def _open_without_share_delete(path: str) -> int:
    """Open a file the way an external scanner does: a handle that blocks deletion."""
    kernel32 = _kernel32()
    handle = kernel32.CreateFileW(path, 0x80000000, 0x1, None, 3, 0, None)  # GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING
    if handle in (None, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF):
        raise OSError(ctypes.get_last_error(), "CreateFileW", path)
    return int(handle)


class LocalAccountTests(unittest.TestCase):
    def test_empty_password_is_allowed(self) -> None:
        with _robust_tempdir() as temporary:
            store = LocalAccountStore(Path(temporary) / "accounts.sqlite3", ScryptPasswordHasher())
            store.create_account("operator", "Operator", "")

            self.assertIsNotNone(store.authenticate("operator", ""))

    def test_account_sessions_are_opaque_revocable_and_persistent(self) -> None:
        with _robust_tempdir() as temporary:
            database = Path(temporary) / "accounts.sqlite3"
            store = LocalAccountStore(database, ScryptPasswordHasher())
            account = store.create_account("operator", "Оператор", "correct horse battery staple")
            self.assertIsNone(store.authenticate("operator", "incorrect password"))
            session = store.authenticate("OPERATOR", "correct horse battery staple")
            self.assertIsNotNone(session)
            assert session is not None
            self.assertEqual(account.account_id, LocalAccountStore(database, ScryptPasswordHasher()).resolve_session(session.token).account_id)
            store.revoke_session(session.token)
            self.assertIsNone(store.resolve_session(session.token))

    def test_password_reset_revokes_existing_sessions(self) -> None:
        with _robust_tempdir() as temporary:
            store = LocalAccountStore(Path(temporary) / "accounts.sqlite3", ScryptPasswordHasher())
            account = store.create_account("admin", "Admin", "original password 123")
            session = store.authenticate("admin", "original password 123")
            assert session is not None
            store.reset_password(account.account_id, "replacement password 456")
            self.assertIsNone(store.resolve_session(session.token))
            self.assertIsNotNone(store.authenticate("admin", "replacement password 456"))

    @unittest.skipUnless(os.name == "nt", "Windows file-sharing semantics")
    def test_cleanup_waits_for_a_foreign_handle_on_the_database(self) -> None:
        opened = threading.Event()
        holder_error: list[BaseException] = []

        def hold(database_path: str) -> None:
            kernel32 = _kernel32()
            try:
                handle = _open_without_share_delete(database_path)
            except BaseException as exc:
                holder_error.append(exc)
                opened.set()
                return
            opened.set()
            time.sleep(0.6)  # outlast the first failed cleanup attempt
            kernel32.CloseHandle(handle)

        with _robust_tempdir() as temporary:
            database = Path(temporary) / "accounts.sqlite3"
            store = LocalAccountStore(database, ScryptPasswordHasher())
            store.create_account("operator", "Оператор", "correct horse battery staple")
            holder = threading.Thread(target=hold, args=(str(database),))
            holder.start()
            self.assertTrue(opened.wait(10), f"foreign handle was not acquired: {holder_error!r}")
            self.assertFalse(holder_error, f"foreign handle acquisition failed: {holder_error!r}")

        holder.join(timeout=10)
        self.assertFalse(holder.is_alive())
        self.assertFalse(Path(temporary).exists())

    def test_cleanup_retries_when_the_final_rmdir_reports_not_empty(self) -> None:
        real_rmtree = shutil.rmtree
        attempts = {"count": 0}

        def flaky_rmtree(path, *args, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                # The exact error the unit gate hit: a transient <db>-SHM.tmp file
                # recreated between the rmtree walk and the final rmdir.
                raise OSError(145, "Папка не пуста", str(path), 145, None)
            return real_rmtree(path, *args, **kwargs)

        shutil.rmtree = flaky_rmtree
        try:
            with _robust_tempdir() as temporary:
                database = Path(temporary) / "accounts.sqlite3"
                store = LocalAccountStore(database, ScryptPasswordHasher())
                store.create_account("operator", "Оператор", "correct horse battery staple")
        finally:
            shutil.rmtree = real_rmtree

        self.assertGreaterEqual(attempts["count"], 2)
        self.assertFalse(Path(temporary).exists())


if __name__ == "__main__":
    unittest.main()
