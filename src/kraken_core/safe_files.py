"""Small cross-platform primitives for handling untrusted filesystem paths.

These helpers are intentionally dependency free.  On POSIX a final symbolic
link is rejected atomically with ``O_NOFOLLOW``.  On Windows we additionally
reject every reparse point (including junctions) and compare the file identity
before and after opening it.  Callers must still keep untrusted plugin code in
its own staging directory; these checks prevent Kraken itself from following a
path out of that directory.
"""

from __future__ import annotations

import os
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


class UnsafeFilesystemPath(ValueError):
    """Raised when a path contains a link, reparse point or special file."""


def replace_atomic_with_retry(source: Path | str, destination: Path | str) -> None:
    """Replace caller-owned staged bytes; tolerate brief Windows reader sharing.

    Retry the rename only. Never expose a partial in-place write or remove the
    old destination first. Persistent permissions and other I/O errors surface.
    """
    deadline = time.monotonic() + 2.0
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if os.name != "nt" or exc.winerror not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            time.sleep(0.025)


_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _identity(value: os.stat_result) -> tuple[int, int]:
    return (value.st_dev, value.st_ino)


def _validate_parts(parts: tuple[str, ...]) -> None:
    for part in parts:
        if not part or part in {".", ".."} or any(char in part for char in "/\\:\0"):
            raise UnsafeFilesystemPath("Paths must contain only regular relative components")


def is_link_or_reparse(path: Path | str) -> bool:
    """Inspect *path* without following its final component."""

    value = os.lstat(path)
    return stat.S_ISLNK(value.st_mode) or bool(
        getattr(value, "st_file_attributes", 0) & _REPARSE_POINT
    )


def contained_path(root: Path | str, relative_parts: tuple[str, ...]) -> Path:
    """Build a contained path and reject links in every existing component."""

    _validate_parts(relative_parts)
    base = ensure_regular_directory(root)
    candidate = base.joinpath(*relative_parts)
    try:
        candidate.relative_to(base)
    except ValueError as exc:  # defensive: parts are normally pre-validated
        raise UnsafeFilesystemPath("Path escapes its trusted root") from exc

    current = base
    for part in relative_parts:
        current = current / part
        try:
            value = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(value.st_mode) or bool(
            getattr(value, "st_file_attributes", 0) & _REPARSE_POINT
        ):
            raise UnsafeFilesystemPath(f"Link or reparse point is forbidden: {current}")
    return candidate


def ensure_regular_directory(path: Path | str) -> Path:
    """Require an existing, non-reparse directory and return its canonical path."""

    original = Path(path)
    original_value = os.lstat(original)
    if stat.S_ISLNK(original_value.st_mode) or bool(
        getattr(original_value, "st_file_attributes", 0) & _REPARSE_POINT
    ):
        raise UnsafeFilesystemPath(f"Directory link or reparse point is forbidden: {path}")
    candidate = original.resolve(strict=True)
    value = os.lstat(candidate)
    if (
        not stat.S_ISDIR(value.st_mode)
        or stat.S_ISLNK(value.st_mode)
        or bool(getattr(value, "st_file_attributes", 0) & _REPARSE_POINT)
    ):
        raise UnsafeFilesystemPath(f"Not a safe directory: {path}")
    return candidate


def make_contained_directories(root: Path | str, relative_parts: tuple[str, ...]) -> Path:
    """Create directories one component at a time, rejecting link substitution."""

    _validate_parts(relative_parts)
    base = ensure_regular_directory(root)
    current = base
    for part in relative_parts:
        current = current / part
        try:
            os.mkdir(current)
        except FileExistsError:
            pass
        value = os.lstat(current)
        if (
            not stat.S_ISDIR(value.st_mode)
            or stat.S_ISLNK(value.st_mode)
            or bool(getattr(value, "st_file_attributes", 0) & _REPARSE_POINT)
        ):
            raise UnsafeFilesystemPath(f"Unsafe directory component: {current}")
    return current


@contextmanager
def open_regular_read(path: Path | str, *, root: Path | str | None = None) -> Iterator[BinaryIO]:
    """Open a regular file without following its final link.

    When ``root`` is supplied, all existing components below that root are
    checked for symlinks/reparse points before and after opening.  The returned
    handle is the same handle callers must hash and consume, avoiding a
    check-then-reopen race.
    """

    candidate = Path(path)
    base: Path | None = None
    relative_parts: tuple[str, ...] = ()
    if root is not None:
        base = ensure_regular_directory(root)
        try:
            relative_parts = candidate.relative_to(base).parts
        except ValueError as exc:
            raise UnsafeFilesystemPath("File is outside its trusted root") from exc
        candidate = contained_path(base, relative_parts)

    before = os.lstat(candidate)
    if stat.S_ISLNK(before.st_mode) or bool(
        getattr(before, "st_file_attributes", 0) & _REPARSE_POINT
    ):
        raise UnsafeFilesystemPath(f"Link or reparse point is forbidden: {candidate}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise UnsafeFilesystemPath(f"Cannot safely open file: {candidate}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _identity(before) != _identity(opened):
            raise UnsafeFilesystemPath(f"Path changed while opening: {candidate}")
        if base is not None:
            # Detect an intermediate junction/symlink replacement on platforms
            # without an openat2-style API.  POSIX also has O_NOFOLLOW above for
            # the final component.
            contained_path(base, relative_parts)
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            yield stream
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def open_exclusive_write(path: Path | str) -> BinaryIO:
    """Create a new regular file without following an existing final link."""

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode):
        os.close(descriptor)
        raise UnsafeFilesystemPath(f"Refusing to create a special file: {path}")
    return os.fdopen(descriptor, "wb", closefd=True)


def open_regular_append(path: Path | str, *, root: Path | str | None = None) -> BinaryIO:
    """Open/create a regular append-only file without following a final link."""

    candidate = Path(path)
    base: Path | None = None
    parts: tuple[str, ...] = ()
    if root is not None:
        base = ensure_regular_directory(root)
        try:
            parts = candidate.relative_to(base).parts
        except ValueError as exc:
            raise UnsafeFilesystemPath("Append target is outside its trusted root") from exc
        candidate = contained_path(base, parts)
    try:
        before = os.lstat(candidate)
    except FileNotFoundError:
        before = None
    if before is not None and (
        stat.S_ISLNK(before.st_mode)
        or bool(getattr(before, "st_file_attributes", 0) & _REPARSE_POINT)
    ):
        raise UnsafeFilesystemPath(f"Link or reparse point is forbidden: {candidate}")
    flags = (
        os.O_WRONLY
        | os.O_APPEND
        | os.O_CREAT
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(candidate, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            before is not None and _identity(before) != _identity(opened)
        ):
            raise UnsafeFilesystemPath(f"Path changed while opening for append: {candidate}")
        if base is not None:
            contained_path(base, parts)
    except BaseException:
        os.close(descriptor)
        raise
    return os.fdopen(descriptor, "ab", closefd=True)


__all__ = [
    "UnsafeFilesystemPath",
    "contained_path",
    "ensure_regular_directory",
    "is_link_or_reparse",
    "make_contained_directories",
    "open_exclusive_write",
    "open_regular_append",
    "open_regular_read",
]
