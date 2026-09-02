from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

from kraken_manager.application.dto import StoredContent
from kraken_manager.domain.artifacts import BlobRef
from kraken_manager.infrastructure.filesystem._atomic import fsync_directory
from kraken_manager.infrastructure.filesystem.layout import FileProjectLayout
from kraken_manager.infrastructure.filesystem.locking import ProjectFileLock


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BlobIntegrityError(RuntimeError):
    pass


class BlobNotFoundError(FileNotFoundError):
    pass


class UnsafeBlobPath(RuntimeError):
    pass


def _digest_string(reference: str | BlobRef | Any) -> str:
    if isinstance(reference, BlobRef):
        digest = reference.sha256
    elif isinstance(reference, str):
        digest = reference.removeprefix("sha256:")
    else:
        digest = getattr(reference, "digest", None) or getattr(reference, "sha256", None)
    digest = str(digest).lower()
    if not _SHA256.fullmatch(digest):
        raise ValueError("invalid SHA-256 blob reference")
    return digest


class FilesystemBlobStore:
    """Immutable, streaming, content-addressed SHA-256 blob store."""

    chunk_size = 1024 * 1024

    def __init__(self, object_root: str | Path, *, staging_root: str | Path | None = None) -> None:
        self.root = Path(object_root).resolve()
        self.staging_root = (
            Path(staging_root).resolve() if staging_root is not None else (self.root.parent / ".blob-staging").resolve()
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_project(cls, catalog_root: str | Path, project_id: str) -> FilesystemBlobStore:
        layout = FileProjectLayout(Path(catalog_root), project_id)
        layout.ensure_directories()
        return cls(layout.objects_dir, staging_root=layout.staging_dir / "blobs")

    def _path(self, digest: str) -> Path:
        digest = _digest_string(digest)
        candidate = self.root / digest[:2] / digest[2:4] / digest
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(self.root):
            raise UnsafeBlobPath(f"blob path escapes object root: {candidate}")
        return candidate

    @staticmethod
    def _chunks(source: bytes | bytearray | memoryview | BinaryIO | Iterable[bytes]) -> Iterator[bytes]:
        if isinstance(source, (bytes, bytearray, memoryview)):
            yield bytes(source)
            return
        read = getattr(source, "read", None)
        if callable(read):
            while True:
                chunk = read(FilesystemBlobStore.chunk_size)
                if not chunk:
                    return
                if not isinstance(chunk, bytes):
                    raise TypeError("binary blob stream returned non-bytes data")
                yield chunk
            return
        for chunk in source:
            if not isinstance(chunk, bytes):
                raise TypeError("blob iterable must yield bytes")
            if chunk:
                yield chunk

    def put(
        self,
        source: bytes | bytearray | memoryview | BinaryIO | Iterable[bytes],
        *,
        expected_sha256: str | None = None,
        expected_digest: str | None = None,
    ) -> StoredContent:
        if expected_sha256 is not None and expected_digest is not None and expected_sha256 != expected_digest:
            raise ValueError("expected_sha256 and expected_digest disagree")
        expected_value = expected_sha256 if expected_sha256 is not None else expected_digest
        expected = None if expected_value is None else _digest_string(expected_value)
        fd, temporary_name = tempfile.mkstemp(prefix="blob-", suffix=".tmp", dir=self.staging_root)
        temporary = Path(temporary_name)
        hasher = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(fd, "wb") as target:
                for chunk in self._chunks(source):
                    target.write(chunk)
                    hasher.update(chunk)
                    size += len(chunk)
                target.flush()
                os.fsync(target.fileno())

            digest = hasher.hexdigest()
            if expected is not None and digest != expected:
                raise BlobIntegrityError(f"blob digest mismatch: expected {expected}, received {digest}")

            final = self._path(digest)
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                self._verify_path(final, digest, size)
                return StoredContent(blob=BlobRef(sha256=digest, size_bytes=size), already_existed=True)

            already_existed = False
            try:
                # Hard-link creation is both atomic and create-only.
                os.link(temporary, final)
            except FileExistsError:
                self._verify_path(final, digest, size)
                already_existed = True
            except OSError:
                # Some filesystems disable hard links.  Cooperating writers use a
                # digest-specific lock, then rename only while the target is absent.
                lock = ProjectFileLock(final.with_name(f".{digest}.lock"))
                with lock.hold(30.0):
                    if final.exists():
                        self._verify_path(final, digest, size)
                        already_existed = True
                    else:
                        os.rename(temporary, final)
            fsync_directory(final.parent)
            return StoredContent(
                blob=BlobRef(sha256=digest, size_bytes=size),
                already_existed=already_existed,
            )
        finally:
            temporary.unlink(missing_ok=True)

    write = put

    def put_file(self, source: str | Path, *, expected_sha256: str | None = None) -> StoredContent:
        source_path = Path(source)
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError(f"blob source must be a regular file: {source_path}")
        with source_path.open("rb") as stream:
            return self.put(stream, expected_sha256=expected_sha256)

    def open(self, reference: str | BlobRef | Any) -> BinaryIO:
        digest = _digest_string(reference)
        path = self._path(digest)
        if path.is_symlink() or not path.is_file():
            raise BlobNotFoundError(digest)
        return path.open("rb")

    def path_for_read(self, reference: str | BlobRef | Any) -> Path:
        """Return a verified, immutable managed-blob path for local staging."""

        digest = _digest_string(reference)
        path = self._path(digest)
        if path.is_symlink() or not path.is_file():
            raise BlobNotFoundError(digest)
        return path

    open_read = open

    def read(self, reference: str | BlobRef | Any) -> bytes:
        with self.open(reference) as stream:
            return stream.read()

    def iter_bytes(self, reference: str | BlobRef | Any, *, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        size = chunk_size
        if size <= 0:
            raise ValueError("chunk_size must be positive")
        with self.open(reference) as stream:
            while chunk := stream.read(size):
                yield chunk

    def exists(self, reference: str | BlobRef | Any) -> bool:
        path = self._path(_digest_string(reference))
        return path.is_file() and not path.is_symlink()

    def stat(self, reference: str | BlobRef | Any) -> BlobRef:
        digest = _digest_string(reference)
        path = self._path(digest)
        if path.is_symlink() or not path.is_file():
            raise BlobNotFoundError(digest)
        return BlobRef(sha256=digest, size_bytes=path.stat().st_size)

    def verify(self, reference: str | BlobRef | Any) -> BlobRef:
        expected_size = reference.size_bytes if isinstance(reference, BlobRef) else None
        digest = _digest_string(reference)
        path = self._path(digest)
        if path.is_symlink() or not path.is_file():
            raise BlobNotFoundError(digest)
        return self._verify_path(path, digest, expected_size)

    @staticmethod
    def _verify_path(path: Path, digest: str, expected_size: int | None = None) -> BlobRef:
        hasher = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(FilesystemBlobStore.chunk_size):
                    hasher.update(chunk)
                    size += len(chunk)
        except FileNotFoundError as exc:
            raise BlobNotFoundError(digest) from exc
        if hasher.hexdigest() != digest:
            raise BlobIntegrityError(f"managed blob {digest} has been modified")
        if expected_size is not None and size != expected_size:
            raise BlobIntegrityError(f"managed blob {digest} size mismatch: expected {expected_size}, found {size}")
        return BlobRef(sha256=digest, size_bytes=size)

    def iter_refs(self) -> Iterator[BlobRef]:
        for directory, names, files in os.walk(self.root, followlinks=False):
            names[:] = [name for name in names if not (Path(directory) / name).is_symlink()]
            for name in sorted(files):
                if not _SHA256.fullmatch(name):
                    continue
                path = Path(directory) / name
                if path.is_symlink() or path != self._path(name):
                    continue
                yield BlobRef(sha256=name, size_bytes=path.stat().st_size)

    def verify_all(self) -> list[BlobRef]:
        return [self.verify(reference) for reference in self.iter_refs()]
