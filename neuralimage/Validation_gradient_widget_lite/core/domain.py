"""Define minimal immutable domain models used by the mismatch-only lite widget."""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ComparisonMode(str, Enum):
    """Define supported mismatch operations."""

    OVERLAY_ONLY = "overlay_only"
    XOR = "xor"
    FIRST_MINUS_SECOND = "first_minus_second"
    SECOND_MINUS_FIRST = "second_minus_first"
    DISAGREEMENT = "disagreement"
    GRAYSCALE_DIFF = "grayscale_diff"

    @property
    def label(self) -> str:
        labels = {
            self.OVERLAY_ONLY: "Overlay only",
            self.XOR: "XOR",
            self.FIRST_MINUS_SECOND: "First - second",
            self.SECOND_MINUS_FIRST: "Second - first",
            self.DISAGREEMENT: "Disagreement",
            self.GRAYSCALE_DIFF: "Grayscale difference",
        }
        return labels[self]


@dataclass(frozen=True, slots=True)
class FolderSpec:
    """Describe one folder used as a comparison or base-layer source."""

    path: Path
    label: str


@dataclass(frozen=True, slots=True)
class BuildOptions:
    """Store backend options used to index frames and compute mismatches."""

    comparison_mode: ComparisonMode = ComparisonMode.XOR
    thumbnail_size: int = 64
    recursive: bool = True
    file_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    max_workers: int = max(1, os.cpu_count() or 4)
    progress_update_interval: int = 64
    cache_enabled: bool = True


@dataclass(frozen=True, slots=True)
class FrameIdentity:
    """Store stable frame identifiers and matrix coordinates for one tile."""

    frame_id: int
    base_id: int | None = None
    tile_x: int | None = None
    tile_y: int | None = None
    source_key: str | None = None


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """Store one pair of matched frames and their computed mismatch values."""

    key: str
    display_name: str
    identity: FrameIdentity | None = None
    score: float = 0.0
    first_path: str = ""
    second_path: str = ""
    base_path: str | None = None
    absolute_score: float | None = None
    relative_score: float | None = None
    score_ready: bool = False


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Store the result of one matrix build and its aggregated mismatch range."""

    records: tuple[FrameRecord, ...]
    first_folder: FolderSpec
    second_folder: FolderSpec
    base_folder: FolderSpec | None
    options: BuildOptions
    min_score: float
    max_score: float
    eligible_key_count: int
    scores_computed: bool = False
    best_match_key: str | None = None
    min_absolute_score: float | None = None
    max_absolute_score: float | None = None
