"""Atomic package writing and safe return classification."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from collections.abc import Callable, Iterator
from typing import Mapping

from kraken_core.plugin_protocol import safe_relative_path
from kraken_core.safe_files import (
    UnsafeFilesystemPath,
    contained_path,
    ensure_regular_directory,
    make_contained_directories,
    open_exclusive_write,
    open_regular_read,
)

from .crypto import decrypt_archive, encrypt_archive, sign, verify
from kraken_manager.domain.workflows import ReviewPackageManifestV1

from .manifest import canonical_manifest_json, manifest_from_json


MANIFEST_NAME = "kraken-review.json"
SIGNATURE_NAME = "kraken-review.sig"


class UnsafeReviewPackage(ValueError):
    """The package exceeds resource limits or contains unsafe filesystem data."""


@dataclass(frozen=True, slots=True)
class ReviewPackageLimits:
    """Resource limits applied before and during archive extraction.

    Limits are configurable because installations have different frame sizes,
    but an archive is never extracted without finite bounds.
    """

    max_entries: int = 250_000
    max_file_bytes: int = 2 * 1024**3
    max_total_bytes: int = 16 * 1024**3
    max_encrypted_bytes: int = 4 * 1024**3
    max_compression_ratio: float = 200.0
    ratio_check_min_bytes: int = 1024 * 1024
    max_manifest_bytes: int = 16 * 1024**2
    max_signature_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        integer_values = (
            self.max_entries,
            self.max_file_bytes,
            self.max_total_bytes,
            self.max_encrypted_bytes,
            self.ratio_check_min_bytes,
            self.max_manifest_bytes,
            self.max_signature_bytes,
        )
        if any(isinstance(value, bool) or value <= 0 for value in integer_values):
            raise ValueError("Review package limits must be positive")
        if self.max_compression_ratio < 1:
            raise ValueError("Compression ratio limit must be at least 1")


class ReturnCategory(StrEnum):
    UNCHANGED = "unchanged"
    CHANGED = "changed"
    MISSING = "missing"
    EXTRA = "extra"
    DUPLICATE = "duplicate"
    INVALID = "invalid"
    STALE_BASE_CONFLICT = "stale_base_conflict"


@dataclass(frozen=True, slots=True)
class ReturnInspection:
    frame_id: str | None
    relative_path: str
    category: ReturnCategory
    original_sha256: str | None
    returned_sha256: str | None
    detail: str = ""


def _sha256(
    path: Path,
    *,
    root: Path | None = None,
    maximum: int | None = None,
) -> str:
    digest = hashlib.sha256()
    observed = 0
    with open_regular_read(path, root=root) as stream:
        while chunk := stream.read(1024 * 1024):
            observed += len(chunk)
            if maximum is not None and observed > maximum:
                raise UnsafeReviewPackage(f"File exceeds the configured size limit: {path}")
            digest.update(chunk)
    return digest.hexdigest()


def _read_limited(path: Path, *, root: Path, maximum: int, label: str) -> bytes:
    chunks: list[bytes] = []
    observed = 0
    with open_regular_read(path, root=root) as stream:
        while chunk := stream.read(min(1024 * 1024, maximum + 1 - observed)):
            observed += len(chunk)
            if observed > maximum:
                raise UnsafeReviewPackage(f"{label} exceeds its size limit")
            chunks.append(chunk)
    return b"".join(chunks)


def _copy_source_verified(source: Path | str, sink: object, expected_sha256: str) -> int:
    """Copy and hash through one source handle to avoid check/reopen races."""

    digest = hashlib.sha256()
    written = 0
    with open_regular_read(Path(source)) as reader:
        while chunk := reader.read(1024 * 1024):
            digest.update(chunk)
            written += len(chunk)
            sink.write(chunk)  # type: ignore[attr-defined]
    if digest.hexdigest() != expected_sha256:
        raise ValueError("Source hash differs from manifest")
    return written


def _expected_hashes(manifest: ReviewPackageManifestV1) -> dict[str, str]:
    expected: dict[str, str] = {}
    casefolded: set[str] = set()
    for item in manifest.files:
        relative = safe_relative_path(item.relative_path)
        folded = relative.casefold()
        if folded in casefolded:
            raise UnsafeReviewPackage(f"Duplicate package path: {relative}")
        casefolded.add(folded)
        expected[relative] = item.sha256
    return expected


class ReviewPackageWriter:
    """Write a signed folder or encrypted ``.kraken-review`` archive."""

    def __init__(self, private_key: bytes) -> None:
        self.private_key = private_key

    @staticmethod
    def validate_sources(manifest: ReviewPackageManifestV1, sources: Mapping[str, Path | str]) -> None:
        expected = _expected_hashes(manifest)
        if set(sources) != set(expected):
            missing = sorted(set(expected) - set(sources))
            extra = sorted(set(sources) - set(expected))
            raise ValueError(f"Source mapping differs from manifest; missing={missing}, extra={extra}")
        for relative, source in sources.items():
            safe_relative_path(relative)
            try:
                digest = _sha256(Path(source))
            except (OSError, UnsafeFilesystemPath) as exc:
                raise ValueError(f"Not a regular source file: {source}") from exc
            if digest != expected[relative]:
                raise ValueError(f"Source hash differs from manifest: {relative}")

    def write_folder(
        self,
        target: Path | str,
        manifest: ReviewPackageManifestV1,
        sources: Mapping[str, Path | str],
    ) -> Path:
        self.validate_sources(manifest, sources)
        destination = Path(target).resolve()
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        try:
            raw = canonical_manifest_json(manifest).encode("utf-8")
            with open_exclusive_write(staging / MANIFEST_NAME) as stream:
                stream.write(raw)
            with open_exclusive_write(staging / SIGNATURE_NAME) as stream:
                stream.write(sign(raw, self.private_key).encode("ascii"))
            expected = _expected_hashes(manifest)
            for relative, source in sources.items():
                parts = safe_relative_path(relative).split("/")
                make_contained_directories(staging, tuple(parts[:-1]))
                output = contained_path(staging, tuple(parts))
                with open_exclusive_write(output) as sink:
                    try:
                        _copy_source_verified(source, sink, expected[relative])
                    except ValueError as exc:
                        raise ValueError(f"Source hash differs from manifest: {relative}") from exc
                    sink.flush()
                    os.fsync(sink.fileno())
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return destination

    def write_encrypted(
        self,
        target: Path | str,
        manifest: ReviewPackageManifestV1,
        sources: Mapping[str, Path | str],
        *,
        password: str,
    ) -> Path:
        self.validate_sources(manifest, sources)
        destination = Path(target).resolve()
        if destination.suffix != ".kraken-review":
            destination = destination.with_suffix(".kraken-review")
        if destination.exists():
            raise FileExistsError(destination)
        buffer = io.BytesIO()
        raw = canonical_manifest_json(manifest).encode("utf-8")
        expected = _expected_hashes(manifest)
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr(MANIFEST_NAME, raw)
            archive.writestr(SIGNATURE_NAME, sign(raw, self.private_key))
            for relative, source in sources.items():
                normalized = safe_relative_path(relative)
                with archive.open(normalized, "w", force_zip64=True) as sink:
                    try:
                        _copy_source_verified(source, sink, expected[normalized])
                    except ValueError as exc:
                        raise ValueError(f"Source hash differs from manifest: {normalized}") from exc
        encrypted = encrypt_archive(buffer.getvalue(), password)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        with temporary.open("xb") as stream:
            stream.write(encrypted)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return destination

    def write(
        self,
        destination: str,
        manifest: ReviewPackageManifestV1,
        files: Mapping[str, Callable[[], Iterator[bytes]]],
    ) -> None:
        """Implement the application ReviewPackageWriter port with streaming inputs."""

        expected = _expected_hashes(manifest)
        if set(files) != set(expected):
            raise ValueError("Review package streams do not match manifest paths")
        target = Path(destination).resolve()
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        try:
            raw = canonical_manifest_json(manifest).encode("utf-8")
            with open_exclusive_write(staging / MANIFEST_NAME) as stream:
                stream.write(raw)
            with open_exclusive_write(staging / SIGNATURE_NAME) as stream:
                stream.write(sign(raw, self.private_key).encode("ascii"))
            for relative, open_chunks in files.items():
                parts = tuple(safe_relative_path(relative).split("/"))
                make_contained_directories(staging, parts[:-1])
                output = contained_path(staging, parts)
                digest = hashlib.sha256()
                with open_exclusive_write(output) as stream:
                    for chunk in open_chunks():
                        if not isinstance(chunk, bytes):
                            raise TypeError("Review package stream must yield bytes")
                        digest.update(chunk)
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                if digest.hexdigest() != expected[relative]:
                    raise ValueError(f"Review package stream hash mismatch: {relative}")
            os.replace(staging, target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def _zip_members(
    archive: zipfile.ZipFile,
    limits: ReviewPackageLimits,
) -> tuple[tuple[zipfile.ZipInfo, str, bool], ...]:
    infos = archive.infolist()
    if len(infos) > limits.max_entries:
        raise UnsafeReviewPackage("Archive contains too many entries")
    seen: set[str] = set()
    total = 0
    result: list[tuple[zipfile.ZipInfo, str, bool]] = []
    for info in infos:
        if info.flag_bits & 0x1:
            raise UnsafeReviewPackage("Nested ZIP encryption is not supported")
        directory = info.is_dir()
        raw_name = info.filename[:-1] if directory and info.filename.endswith("/") else info.filename
        relative = safe_relative_path(raw_name)
        folded = relative.casefold()
        if folded in seen:
            raise UnsafeReviewPackage(f"Archive contains a duplicate path: {relative}")
        seen.add(folded)

        unix_mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(unix_mode)
        if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise UnsafeReviewPackage(f"Archive contains a link or special file: {relative}")
        if directory and file_type == stat.S_IFREG:
            raise UnsafeReviewPackage(f"Archive directory has a regular-file mode: {relative}")
        if not directory and file_type == stat.S_IFDIR:
            raise UnsafeReviewPackage(f"Archive file has a directory mode: {relative}")
        if info.file_size < 0 or info.compress_size < 0:
            raise UnsafeReviewPackage("Archive contains an invalid size")
        if info.file_size > limits.max_file_bytes:
            raise UnsafeReviewPackage(f"Archive member exceeds the file limit: {relative}")
        total += info.file_size
        if total > limits.max_total_bytes:
            raise UnsafeReviewPackage("Archive exceeds the total uncompressed size limit")
        if info.file_size >= limits.ratio_check_min_bytes:
            if info.compress_size == 0 or info.file_size / info.compress_size > limits.max_compression_ratio:
                raise UnsafeReviewPackage(f"Suspicious compression ratio: {relative}")
        result.append((info, relative, directory))
    return tuple(result)


def _walk_regular_files(
    root: Path,
    limits: ReviewPackageLimits,
    *,
    reject_unsafe: bool = True,
) -> tuple[tuple[str, Path], ...]:
    """Walk a folder without following symlinks/junctions and with finite bounds."""

    base = ensure_regular_directory(root)
    pending = [base]
    result: list[tuple[str, Path]] = []
    total = 0
    entries = 0
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as iterator:
            for entry in iterator:
                entries += 1
                if entries > limits.max_entries:
                    raise UnsafeReviewPackage("Package folder contains too many entries")
                value = entry.stat(follow_symlinks=False)
                reparse = bool(
                    getattr(value, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                )
                relative = Path(entry.path).relative_to(base).as_posix()
                if entry.is_symlink() or reparse:
                    if reject_unsafe:
                        raise UnsafeReviewPackage(f"Package folder contains a link/reparse point: {relative}")
                    continue
                if stat.S_ISDIR(value.st_mode):
                    pending.append(Path(entry.path))
                    continue
                if not stat.S_ISREG(value.st_mode):
                    if reject_unsafe:
                        raise UnsafeReviewPackage(f"Package folder contains a special file: {relative}")
                    continue
                if value.st_size > limits.max_file_bytes:
                    raise UnsafeReviewPackage(f"Package file exceeds the size limit: {relative}")
                total += value.st_size
                if total > limits.max_total_bytes:
                    raise UnsafeReviewPackage("Package folder exceeds the total size limit")
                result.append((safe_relative_path(relative), Path(entry.path)))
    return tuple(result)


class ReviewPackageReader:
    def __init__(self, public_key: bytes, *, limits: ReviewPackageLimits | None = None) -> None:
        self.public_key = public_key
        self.limits = limits or ReviewPackageLimits()

    def read_manifest(self, folder: Path | str) -> ReviewPackageManifestV1:
        root = ensure_regular_directory(folder)
        raw = _read_limited(
            contained_path(root, (MANIFEST_NAME,)),
            root=root,
            maximum=self.limits.max_manifest_bytes,
            label="Review manifest",
        )
        signature_raw = _read_limited(
            contained_path(root, (SIGNATURE_NAME,)),
            root=root,
            maximum=self.limits.max_signature_bytes,
            label="Review signature",
        )
        try:
            signature = signature_raw.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise UnsafeReviewPackage("Review signature is not ASCII") from exc
        if not verify(raw, signature, self.public_key):
            raise ValueError("Review package signature is invalid")
        try:
            manifest = manifest_from_json(raw.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise UnsafeReviewPackage("Review manifest is not UTF-8") from exc
        _expected_hashes(manifest)
        return manifest

    def iter_file(self, source: str, relative_path: str) -> Iterator[bytes]:
        root = ensure_regular_directory(source)
        parts = tuple(safe_relative_path(relative_path).split("/"))
        candidate = contained_path(root, parts)
        observed = 0
        with open_regular_read(candidate, root=root) as stream:
            while chunk := stream.read(1024 * 1024):
                observed += len(chunk)
                if observed > self.limits.max_file_bytes:
                    raise UnsafeReviewPackage("Review package file exceeds the size limit")
                yield chunk

    def list_relative_paths(self, source: str) -> tuple[str, ...]:
        root = ensure_regular_directory(source)
        return tuple(
            sorted(
                relative
                for relative, _ in _walk_regular_files(root, self.limits)
                if relative not in {MANIFEST_NAME, SIGNATURE_NAME}
            )
        )

    def decrypt_to(self, package: Path | str, target: Path | str, *, password: str) -> Path:
        package_path = Path(package)
        envelope_chunks: list[bytes] = []
        envelope_size = 0
        with open_regular_read(package_path) as stream:
            while chunk := stream.read(1024 * 1024):
                envelope_size += len(chunk)
                if envelope_size > self.limits.max_encrypted_bytes:
                    raise UnsafeReviewPackage("Encrypted review package exceeds its size limit")
                envelope_chunks.append(chunk)
        payload = decrypt_archive(b"".join(envelope_chunks), password)
        if len(payload) > self.limits.max_encrypted_bytes:
            raise UnsafeReviewPackage("Decrypted ZIP exceeds its compressed-size limit")
        destination = Path(target).resolve()
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                members = _zip_members(archive, self.limits)
                if MANIFEST_NAME.casefold() not in {relative.casefold() for _, relative, _ in members}:
                    raise UnsafeReviewPackage("Archive has no review manifest")
                if SIGNATURE_NAME.casefold() not in {relative.casefold() for _, relative, _ in members}:
                    raise UnsafeReviewPackage("Archive has no review signature")
                actual_total = 0
                for info, relative, directory in members:
                    parts = tuple(relative.split("/"))
                    if directory:
                        make_contained_directories(staging, parts)
                        continue
                    make_contained_directories(staging, parts[:-1])
                    output = contained_path(staging, parts)
                    member_size = 0
                    with archive.open(info) as source, open_exclusive_write(output) as sink:
                        while chunk := source.read(1024 * 1024):
                            member_size += len(chunk)
                            actual_total += len(chunk)
                            if member_size > self.limits.max_file_bytes or member_size > info.file_size:
                                raise UnsafeReviewPackage(f"Archive member expanded beyond its declared size: {relative}")
                            if actual_total > self.limits.max_total_bytes:
                                raise UnsafeReviewPackage("Archive expanded beyond the total size limit")
                            sink.write(chunk)
                        if member_size != info.file_size:
                            raise UnsafeReviewPackage(f"Archive member size mismatch: {relative}")
                        sink.flush()
                        os.fsync(sink.fileno())
            manifest = self.read_manifest(staging)
            expected = _expected_hashes(manifest)
            actual = {
                relative
                for _, relative, directory in members
                if not directory and relative not in {MANIFEST_NAME, SIGNATURE_NAME}
            }
            if actual != set(expected):
                missing = sorted(set(expected) - actual)
                extra = sorted(actual - set(expected))
                raise UnsafeReviewPackage(
                    f"Archive content differs from its signed manifest; missing={missing}, extra={extra}"
                )
            for relative, expected_digest in expected.items():
                candidate = contained_path(staging, tuple(relative.split("/")))
                if _sha256(candidate, root=staging, maximum=self.limits.max_file_bytes) != expected_digest:
                    raise UnsafeReviewPackage(f"Archive member hash differs from its signed manifest: {relative}")
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return destination

    def inspect_return(
        self,
        folder: Path | str,
        manifest: ReviewPackageManifestV1,
        *,
        active_vector_versions: Mapping[str, str] | None = None,
    ) -> tuple[ReturnInspection, ...]:
        root = ensure_regular_directory(folder)
        expected_by_path = {
            safe_relative_path(item.relative_path): item for item in manifest.files if item.role == "vector"
        }
        package_data_paths = {safe_relative_path(item.relative_path) for item in manifest.files}
        results: list[ReturnInspection] = []
        for relative, item in expected_by_path.items():
            try:
                candidate = contained_path(root, tuple(relative.split("/")))
                os.lstat(candidate)
            except FileNotFoundError:
                results.append(ReturnInspection(str(item.frame_id), relative, ReturnCategory.MISSING, item.sha256, None))
                continue
            except (OSError, UnsafeFilesystemPath) as exc:
                results.append(
                    ReturnInspection(
                        str(item.frame_id),
                        relative,
                        ReturnCategory.INVALID,
                        item.sha256,
                        None,
                        f"Unsafe file: {exc}",
                    )
                )
                continue
            try:
                digest = _sha256(candidate, root=root, maximum=self.limits.max_file_bytes)
            except (OSError, UnsafeFilesystemPath, UnsafeReviewPackage) as exc:
                results.append(
                    ReturnInspection(
                        str(item.frame_id),
                        relative,
                        ReturnCategory.INVALID,
                        item.sha256,
                        None,
                        f"Unsafe file: {exc}",
                    )
                )
                continue
            if active_vector_versions is not None and active_vector_versions.get(str(item.frame_id)) != str(item.artifact_version_id):
                category = ReturnCategory.STALE_BASE_CONFLICT
            elif digest == item.sha256:
                category = ReturnCategory.UNCHANGED
            else:
                category = ReturnCategory.CHANGED
            results.append(ReturnInspection(str(item.frame_id), relative, category, item.sha256, digest))
        reserved = {MANIFEST_NAME, SIGNATURE_NAME}
        for relative, candidate in _walk_regular_files(root, self.limits, reject_unsafe=False):
            if relative not in expected_by_path and relative not in package_data_paths and relative not in reserved:
                results.append(
                    ReturnInspection(
                        None,
                        relative,
                        ReturnCategory.EXTRA,
                        None,
                        _sha256(candidate, root=root, maximum=self.limits.max_file_bytes),
                    )
                )
        return tuple(results)


__all__ = [
    "MANIFEST_NAME",
    "SIGNATURE_NAME",
    "ReviewPackageLimits",
    "ReturnCategory",
    "ReturnInspection",
    "UnsafeReviewPackage",
    "ReviewPackageReader",
    "ReviewPackageWriter",
]
