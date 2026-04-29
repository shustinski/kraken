from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class SelectionMode(StrEnum):
    FULL_RANGE = "full_range"
    RECTANGLE = "rectangle"


class FileOperation(StrEnum):
    COPY = "copy"
    MOVE = "move"
    DELETE = "delete"


@dataclass(frozen=True, slots=True)
class FrameRange:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SourceFolder:
    path: Path
    extensions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    sources: tuple[SourceFolder, ...]
    frame_expression: str
    selection_mode: SelectionMode = SelectionMode.RECTANGLE
    frames_per_row: int = 135
    operation: FileOperation = FileOperation.COPY
    destination: Path | None = None
    add_extension_prefix: bool = True


@dataclass(frozen=True, slots=True)
class PlannedOperation:
    source: Path
    destination: Path | None
    frame: int
    extension: str
    source_folder: Path


@dataclass(frozen=True, slots=True)
class MissingFrameSet:
    source_folder: Path
    extension: str
    frames: tuple[int, ...]


@dataclass(slots=True)
class OperationPlan:
    config: ProcessingConfig
    operations: list[PlannedOperation] = field(default_factory=list)
    skipped_sources: list[tuple[Path, str]] = field(default_factory=list)
    missing_frames: list[MissingFrameSet] = field(default_factory=list)
    total_bytes: int = 0

    @property
    def total_gib(self) -> float:
        return self.total_bytes / 1024 / 1024 / 1024


@dataclass(frozen=True, slots=True)
class OperationError:
    source: Path
    destination: Path | None
    message: str


@dataclass(frozen=True, slots=True)
class OperationResult:
    requested: int
    completed: int
    skipped: int
    errors: tuple[OperationError, ...] = ()
    cancelled: bool = False

    @property
    def ok(self) -> bool:
        return not self.cancelled and not self.errors
