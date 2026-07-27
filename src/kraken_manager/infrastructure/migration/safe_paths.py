from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath


class UnsafeBundlePath(ValueError):
    pass


_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_DRIVE = re.compile(r"^[A-Za-z]:")


def validate_bundle_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise UnsafeBundlePath("bundle path must be a non-empty string")
    if "\\" in value or _DRIVE.match(value):
        raise UnsafeBundlePath(f"bundle path must use relative POSIX syntax: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeBundlePath(f"unsafe bundle path: {value!r}")
    for part in path.parts:
        if ":" in part or part.endswith((" ", ".")):
            raise UnsafeBundlePath(f"path is not portable to Windows: {value!r}")
        stem = part.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED:
            raise UnsafeBundlePath(f"reserved Windows path component: {part!r}")
    return path


def safe_join(root: str | Path, relative: str, *, reject_symlinks: bool = True) -> Path:
    root_path = Path(root).resolve()
    relative_path = validate_bundle_path(relative)
    candidate = root_path.joinpath(*relative_path.parts)

    current = root_path
    for part in relative_path.parts:
        current = current / part
        if reject_symlinks and current.exists() and current.is_symlink():
            raise UnsafeBundlePath(f"symbolic links are forbidden in bundle paths: {relative!r}")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root_path):
        raise UnsafeBundlePath(f"bundle path escapes root: {relative!r}")
    return candidate


def portable_relative(path: Path, root: Path) -> str:
    path = path.resolve()
    root = root.resolve()
    if not path.is_relative_to(root):
        raise UnsafeBundlePath(f"path {path} is outside {root}")
    relative = path.relative_to(root).as_posix()
    validate_bundle_path(relative)
    return relative


def iter_regular_files(root: str | Path):
    root_path = Path(root).resolve()
    if not root_path.exists():
        return
    for directory, names, files in os.walk(root_path, followlinks=False):
        directory_path = Path(directory)
        names[:] = sorted(name for name in names if not (directory_path / name).is_symlink())
        for name in sorted(files):
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                continue
            yield path
