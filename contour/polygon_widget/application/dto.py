from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PersistedPaths:
    input_directory: str = ""
    cif_directory: str = ""
    output_directory: str = ""


@dataclass(frozen=True, slots=True)
class InputDirectoryState:
    directory: str
    image_paths: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class CifDirectoryState:
    directory: str
    indexed_paths: dict[str, str] = field(default_factory=dict)
    available: bool = False
