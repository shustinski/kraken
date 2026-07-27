from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ._atomic import atomic_write_json


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class InvalidStorageIdentifier(ValueError):
    pass


def validate_storage_identifier(value: str, *, field: str = "identifier") -> str:
    if not isinstance(value, str) or not _SAFE_IDENTIFIER.fullmatch(value):
        raise InvalidStorageIdentifier(
            f"{field} must contain 1-128 ASCII letters, digits, '.', '_' or '-' and may not start with punctuation"
        )
    if value in {".", ".."}:
        raise InvalidStorageIdentifier(f"unsafe {field}: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class FileProjectLayout:
    catalog_root: Path
    project_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "catalog_root", Path(self.catalog_root).resolve())
        validate_storage_identifier(self.project_id, field="project_id")

    @property
    def project_dir(self) -> Path:
        return self.catalog_root / "projects" / self.project_id

    @property
    def descriptor_path(self) -> Path:
        return self.project_dir / "project.json"

    @property
    def events_dir(self) -> Path:
        return self.project_dir / "events"

    @property
    def objects_dir(self) -> Path:
        return self.project_dir / "objects" / "sha256"

    @property
    def snapshots_dir(self) -> Path:
        return self.project_dir / "snapshots"

    @property
    def staging_dir(self) -> Path:
        return self.project_dir / "staging"

    @property
    def lock_path(self) -> Path:
        return self.project_dir / "lock"

    @property
    def index_dir(self) -> Path:
        return self.project_dir / ".index"

    @property
    def index_path(self) -> Path:
        return self.index_dir / "read.sqlite3"

    def ensure_directories(self) -> None:
        for path in (
            self.events_dir,
            self.objects_dir,
            self.snapshots_dir,
            self.staging_dir,
            self.index_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def initialize(self, descriptor: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        """Create the canonical project tree and immutable identity descriptor."""

        self.ensure_directories()
        value: dict[str, Any] = {
            "schema_version": 1,
            "project_id": self.project_id,
        }
        if descriptor:
            value.update(descriptor)
            if value.get("project_id") != self.project_id:
                raise ValueError("descriptor project_id does not match the layout")

        if self.descriptor_path.exists():
            existing = self.read_descriptor()
            if existing.get("project_id") != self.project_id:
                raise ValueError("existing descriptor belongs to another project")
            return existing

        try:
            atomic_write_json(self.descriptor_path, value, overwrite=False)
        except FileExistsError:
            return self.read_descriptor()
        return value

    def read_descriptor(self) -> Mapping[str, Any]:
        try:
            value = json.loads(self.descriptor_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"project {self.project_id!r} is not initialized") from exc
        if not isinstance(value, dict) or value.get("project_id") != self.project_id:
            raise ValueError(f"invalid project descriptor at {self.descriptor_path}")
        return value
