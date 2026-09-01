"""Safe external neural-network model links and per-run staging."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .safe_files import open_regular_read


def _digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open_regular_read(path) as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


@dataclass(frozen=True, slots=True)
class ExternalModelLink:
    """Metadata-only reference; model bytes are deliberately not imported."""

    path: str
    size: int
    observed_sha256: str

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError("External model size cannot be negative")
        digest = self.observed_sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("External model SHA-256 is invalid")
        object.__setattr__(self, "observed_sha256", digest)

    @classmethod
    def observe(cls, path: str | Path) -> "ExternalModelLink":
        resolved = Path(path).expanduser().resolve(strict=True)
        digest, size = _digest_file(resolved)
        return cls(str(resolved), size, digest)

    def stage(self, staging_directory: str | Path) -> "StagedExternalModel":
        """Re-hash and atomically copy the exact bytes used by one Agent run."""

        source = Path(self.path).resolve(strict=True)
        used_sha256, used_size = _digest_file(source)
        destination_root = Path(staging_directory).resolve()
        destination_root.mkdir(parents=True, exist_ok=True)
        if destination_root.is_symlink() or not destination_root.is_dir():
            raise ValueError("Model staging target must be a regular directory")
        suffix = source.suffix if len(source.suffix) <= 20 else ""
        destination = destination_root / f"external-model-{used_sha256[:16]}{suffix}"
        temporary = destination_root / f".{destination.name}.{uuid4().hex}.tmp"
        try:
            shutil.copyfile(source, temporary)
            copied_sha256, copied_size = _digest_file(temporary)
            if (copied_sha256, copied_size) != (used_sha256, used_size):
                raise OSError("External model changed while it was staged")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return StagedExternalModel(
            source_path=str(source),
            relative_path=destination.name,
            size=used_size,
            observed_sha256=self.observed_sha256,
            used_sha256=used_sha256,
            changed_since_observation=(
                used_sha256 != self.observed_sha256 or used_size != self.size
            ),
        )


@dataclass(frozen=True, slots=True)
class StagedExternalModel:
    source_path: str
    relative_path: str
    size: int
    observed_sha256: str
    used_sha256: str
    changed_since_observation: bool


__all__ = ["ExternalModelLink", "StagedExternalModel"]
