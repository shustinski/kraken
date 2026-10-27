"""Exclusive, verified transfers and metadata-fenced source removal.

The application already owns the selected file operation. These helpers never
replace an existing destination or silently remove a changed source. A failed
source removal restores its name or reports the preserved quarantine path.
"""
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import shutil
import stat
import sys
from uuid import uuid4

from kraken_core.safe_files import contained_path, ensure_regular_directory, make_contained_directories, open_exclusive_write, open_regular_read

from csliser.domain.models import SourceFileSnapshot


class TransferCancelled(RuntimeError):
    pass


def _rename_new(source, destination):
    if os.name == "nt":
        os.rename(source, destination)
        return
    if sys.platform != "linux":
        raise OSError(errno.ENOSYS, "Exclusive source rename is unsupported on this platform")
    # renameat2(RENAME_NOREPLACE), not POSIX rename which replaces destination.
    library = ctypes.CDLL(None, use_errno=True)
    function = getattr(library, "renameat2", None)
    if function is None:
        raise OSError(errno.ENOSYS, "renameat2 is required for safe source removal")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def source_matches(item, path=None):
    root = ensure_regular_directory(item.source_folder)
    if root != item.source_folder:
        raise OSError(f"Source root changed since preview: {item.source_folder}")
    if path is None:
        contained_path(root, item.source.relative_to(root).parts)
    path = item.source if path is None else path
    value = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(value.st_mode) or getattr(value, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            or item.source_snapshot is None or SourceFileSnapshot.from_stat(value) != item.source_snapshot):
        raise OSError(f"Source changed since preview; rebuild the plan: {item.source}")
    return value


def safe_parent(destination):
    parent = Path(destination).absolute().parent
    root = Path(parent.anchor)
    return make_contained_directories(root, parent.relative_to(root).parts)


def _cancelled(predicate):
    if predicate is not None and predicate():
        raise TransferCancelled("Transfer cancelled before file publication")


def _identity(path):
    value = os.stat(path, follow_symlinks=False)
    return value.st_dev, value.st_ino


def copy_verified(item, *, copy_metadata, cancelled=None):
    assert item.destination is not None
    source_matches(item)
    parent = safe_parent(item.destination)
    parent_identity = _identity(parent)
    if os.path.lexists(item.destination):
        raise FileExistsError(f"Destination already exists; no file was replaced: {item.destination}")
    temporary = parent / ("."+item.destination.name+".part-"+uuid4().hex)
    owned = None
    try:
        digest = hashlib.sha256()
        size = 0
        with open_regular_read(item.source, root=item.source_folder) as source:
            if SourceFileSnapshot.from_stat(os.fstat(source.fileno())) != item.source_snapshot:
                raise OSError(f"Source changed while opening: {item.source}")
            with open_exclusive_write(temporary) as sink:
                value = os.fstat(sink.fileno())
                owned = value.st_dev, value.st_ino
                while chunk := source.read(1024*1024):
                    _cancelled(cancelled)
                    digest.update(chunk)
                    size += len(chunk)
                    sink.write(chunk)
                sink.flush()
                os.fsync(sink.fileno())
            if SourceFileSnapshot.from_stat(os.fstat(source.fileno())) != item.source_snapshot:
                raise OSError(f"Source changed during transfer: {item.source}")
        if size != item.source_snapshot.size_bytes:
            raise OSError(f"Source length changed during transfer: {item.source}")
        copied = hashlib.sha256()
        with open_regular_read(temporary, root=parent) as stream:
            while chunk := stream.read(1024*1024):
                _cancelled(cancelled)
                copied.update(chunk)
        if copied.digest() != digest.digest():
            raise OSError(f"Copied bytes failed SHA-256 validation: {item.destination}")
        source_matches(item)
        if copy_metadata:
            shutil.copystat(item.source, temporary, follow_symlinks=False)
            source_matches(item)
        _cancelled(cancelled)
        if _identity(safe_parent(item.destination)) != parent_identity or _identity(temporary) != owned:
            raise OSError("Destination directory or staged transfer changed before publication")
        try:
            _rename_new(temporary, item.destination)
        except OSError as exc:
            if os.name == "nt" or exc.errno not in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
                raise
            # A private verified staging copy can also be published by a new
            # hard link. Never use this fallback to remove a user's source.
            os.link(temporary, item.destination)
    finally:
        if owned is not None:
            try:
                if _identity(temporary) == owned:
                    try:
                        temporary.unlink()
                    except PermissionError:
                        if os.name != "nt":
                            raise
                        # copystat may have made our unpublished staging copy
                        # read-only. The destination/source are never chmodded.
                        os.chmod(temporary, stat.S_IREAD | stat.S_IWRITE)
                        temporary.unlink()
            except FileNotFoundError:
                pass


def remove_verified_source(item):
    source_matches(item)
    quarantine = item.source.with_name("."+item.source.name+".csliser-remove-"+uuid4().hex)
    try:
        _rename_new(item.source, quarantine)
        # Verify the object actually renamed. A replacement arriving after the
        # initial stat is preserved/restored, never removed as the planned file.
        source_matches(item, quarantine)
        quarantine.unlink()
    except BaseException as exc:
        if os.path.lexists(quarantine):
            try:
                _rename_new(quarantine, item.source)
            except OSError as restore_error:
                raise OSError(f"Source was not deleted; preserved at {quarantine}. Restore failed: {restore_error}") from exc
        raise


def move_verified(item, *, cancelled=None):
    assert item.destination is not None
    source_matches(item)
    safe_parent(item.destination)
    _cancelled(cancelled)
    try:
        _rename_new(item.source, item.destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV and getattr(exc, "winerror", None) != 17:
            raise
        copy_verified(item, copy_metadata=True, cancelled=cancelled)
        try:
            remove_verified_source(item)
        except OSError as remove_error:
            raise OSError(f"Verified destination copy is preserved at {item.destination}; source removal failed: {remove_error}") from remove_error
        return
    try:
        source_matches(item, item.destination)
    except BaseException as exc:
        try:
            _rename_new(item.destination, item.source)
        except OSError as restore_error:
            raise OSError(f"Changed source was not deleted; preserved at {item.destination}. Restore failed: {restore_error}") from exc
        raise
