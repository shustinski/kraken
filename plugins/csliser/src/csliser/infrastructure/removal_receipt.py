"""Write-ahead receipts and process locks for quarantined source files.

Receipts authorize recovery of one exact sibling, never a directory tree or a
new destructive operation. The receipt is immutable while the owner is alive.
"""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

from kraken_core.safe_files import ensure_regular_directory, open_exclusive_write, open_regular_read
from csliser.domain.models import SourceFileSnapshot

_HEX = re.compile(r"[0-9a-f]{32}")
MAX_RECEIPT_BYTES = 65536


def sync_directory(path: Path) -> None:
    """POSIX directory durability; Windows file fsync alone is not a power-cut SLA."""
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@contextmanager
def receipt_lock(stream):
    stream.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def verified_digest(path: Path, snapshot: SourceFileSnapshot, *, cancelled=None) -> str:
    with open_regular_read(path, root=path.parent) as stream:
        if SourceFileSnapshot.from_stat(os.fstat(stream.fileno())) != snapshot:
            raise OSError(f"Recovery file changed: {path}")
        digest = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            if cancelled is not None and cancelled():
                raise InterruptedError("Recovery cancelled; files preserved")
            digest.update(chunk)
        if SourceFileSnapshot.from_stat(os.fstat(stream.fileno())) != snapshot:
            raise OSError(f"Recovery file changed during verification: {path}")
        return digest.hexdigest()


class RemovalReceipt:
    def __init__(self, item, quarantine: Path, *, cancelled=None):
        self.item = item
        self.quarantine = quarantine
        self.path = quarantine.with_name(quarantine.name + ".json")
        self._stream_context = None
        self._lock_context = None
        self._identity = None
        self.cancelled = cancelled

    def __enter__(self):
        # Hash before any source rename and recheck the opened object's metadata.
        if self.item.source_snapshot is None or self.item.source_snapshot.inode <= 0:
            raise OSError("Source has no recoverable filesystem identity")
        digest = verified_digest(self.item.source, self.item.source_snapshot, cancelled=self.cancelled)
        payload = dict(version=1, source=self.item.source.name,
                       snapshot=asdict(self.item.source_snapshot), sha256=digest)
        data = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        if len(data) > MAX_RECEIPT_BYTES:
            raise ValueError("Source removal receipt exceeds its size budget")
        self._stream_context = open_exclusive_write(self.path)
        stream = self._stream_context.__enter__()
        self._identity = SourceFileSnapshot.from_stat(os.fstat(stream.fileno()))
        try:
            # One immutable record, flushed before the first filesystem mutation.
            lock = receipt_lock(stream)
            lock.__enter__()
            self._lock_context = lock
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            sync_directory(self.path.parent)
        except BaseException:
            self.__exit__(*sys.exc_info())
            # __enter__ has not handed control to a mutating caller yet.
            # An incomplete owned receipt can be removed without touching data.
            try:
                self.discard()
            except OSError:
                pass
            raise
        self.sha256 = digest
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self._lock_context is not None:
                self._lock_context.__exit__(exc_type, exc, traceback)
        finally:
            if self._stream_context is not None:
                self._stream_context.__exit__(exc_type, exc, traceback)
        if exc_type is None:
            self.discard()

    def discard(self):
        value = os.lstat(self.path)
        if (value.st_dev, value.st_ino) != (self._identity.device, self._identity.inode):
            raise OSError(f"Removal receipt changed; preserved: {self.path}")
        self.path.unlink()
        sync_directory(self.path.parent)


def read_receipt(path: Path, stream) -> tuple[Path, Path, SourceFileSnapshot, str]:
    ensure_regular_directory(path.parent)
    value = os.fstat(stream.fileno())
    if not stat.S_ISREG(value.st_mode) or value.st_size > MAX_RECEIPT_BYTES:
        raise ValueError("Invalid or oversized removal receipt")
    stream.seek(0)
    payload = json.loads(stream.read(MAX_RECEIPT_BYTES + 1))
    if not isinstance(payload, dict) or set(payload) != {"version", "source", "snapshot", "sha256"} or type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("Unsupported removal receipt")
    name = payload["source"]
    if not isinstance(name, str) or not name or name in (".", "..") or any(c in name for c in ("/", "\\", ":", "\0")):
        raise ValueError("Receipt source must be a sibling filename")
    prefix = f".{name}.csliser-remove-"
    if not path.name.startswith(prefix) or not path.name.endswith(".json") or not _HEX.fullmatch(path.name[len(prefix):-5]):
        raise ValueError("Receipt filename does not match its source")
    fields = payload["snapshot"]
    if not isinstance(fields, dict) or set(fields) != {"device", "inode", "size_bytes", "modified_ns"}:
        raise ValueError("Invalid source snapshot")
    if any(type(v) is not int for v in fields.values()) or any(fields[k] < 0 for k in ("device", "inode", "size_bytes")) or fields["inode"] == 0:
        raise ValueError("Invalid source identity")
    digest = payload["sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid source SHA-256")
    return path.parent / name, path.with_name(path.name[:-5]), SourceFileSnapshot(**fields), digest
