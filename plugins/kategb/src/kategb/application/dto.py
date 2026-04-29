from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kategb.domain.models import FrameSelectionMode


@dataclass(frozen=True, slots=True)
class GenerateManifestRequest:
    vector_folder: str
    layer_name: str
    check_name: str
    frame_range_text: str
    selection_mode: FrameSelectionMode
    frames: tuple[int, ...]
    encryption_key: str
    output_folder: Path


@dataclass(frozen=True, slots=True)
class AnalyzeVerificationRequest:
    manifest_path: Path
    encryption_key: str
    incorrect_xml_path: Path
    markup_path: Path | None = None
    layer_name: str | None = None


@dataclass(frozen=True, slots=True)
class SampleRequest:
    layer_name: str
    authors: tuple[str, ...]
    percent_per_author: int
    frame_range_text: str = ""
    selection_mode: FrameSelectionMode = "all"
    random_seed: int | None = None
