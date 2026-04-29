from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

FrameSelectionMode = Literal["all", "area"]


@dataclass(frozen=True, slots=True)
class FrameRange:
    first: int
    last: int

    def __post_init__(self) -> None:
        if self.first <= 0:
            raise ValueError("First frame must be positive.")
        if self.last < self.first:
            raise ValueError("Last frame must be greater than or equal to first frame.")


@dataclass(frozen=True, slots=True)
class LayerInfo:
    name: str
    author_frames: dict[str, tuple[int, ...]]
    frames_in_layer: int
    frames_in_row: int
    cif_folder: Path | None = None
    jpg_folder: Path | None = None
    done: bool = False
    mismatched_cells: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class CrystalInfo:
    layers: dict[str, LayerInfo]

    def layer_names(self) -> tuple[str, ...]:
        return tuple(self.layers)


@dataclass(frozen=True, slots=True)
class SampleGenerationConfig:
    layer_name: str
    authors: tuple[str, ...]
    percent_per_author: int = 100
    frame_range: FrameRange | None = None
    selection_mode: FrameSelectionMode = "all"
    date_filtered_frames: frozenset[int] | None = None
    random_seed: int | None = None


@dataclass(frozen=True, slots=True)
class VerificationManifest:
    vector_folder: str
    layer_name: str
    frame_range: str
    check_name: str
    selection_mode: FrameSelectionMode
    frames: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IncorrectVector:
    number: int
    is_correct: bool
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AuthorVerificationResult:
    author: str
    checked_frames: tuple[int, ...]
    incorrect_frames: tuple[int, ...]

    @property
    def incorrect_count(self) -> int:
        return len(self.incorrect_frames)

    @property
    def incorrect_percent(self) -> float:
        if not self.checked_frames:
            return 0.0
        return round(len(self.incorrect_frames) / len(self.checked_frames) * 100, 2)


@dataclass(frozen=True, slots=True)
class CopySource:
    folder: Path
    role: str


@dataclass(frozen=True, slots=True)
class CopyPlan:
    sources: tuple[CopySource, ...]
    frames: tuple[int, ...]
    destination: Path
    check_name: str
    rewrite_cif_references: bool = True
