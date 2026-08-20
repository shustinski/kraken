"""Classify OS file drops into additive project image and vector paths."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .frame_asset_sync import VECTOR_FILE_SUFFIXES
from ..utils import is_visible_image_path, scan_image_files


@dataclass(frozen=True, slots=True)
class DroppedProjectPayload:
    image_paths: tuple[str, ...]
    vector_paths: tuple[str, ...]

    def is_empty(self) -> bool:
        return not self.image_paths and not self.vector_paths


def classify_dropped_paths(paths: Iterable[str | Path]) -> DroppedProjectPayload:
    images: list[str] = []
    vectors: list[str] = []
    seen_images: set[str] = set()
    seen_vectors: set[str] = set()

    def add_image(raw: str | Path) -> None:
        normalized = str(Path(raw))
        if normalized in seen_images or not is_visible_image_path(normalized):
            return
        seen_images.add(normalized)
        images.append(normalized)

    def add_vector(raw: str | Path) -> None:
        candidate = Path(raw)
        if candidate.suffix.lower() not in VECTOR_FILE_SUFFIXES:
            return
        normalized = str(candidate)
        if normalized in seen_vectors:
            return
        seen_vectors.add(normalized)
        vectors.append(normalized)

    for raw in paths:
        candidate = Path(raw)
        if candidate.is_dir():
            for image_path in scan_image_files(candidate):
                add_image(image_path)
            children = sorted(candidate.iterdir(), key=lambda item: item.name.lower())
            for child in children:
                if child.is_file():
                    add_vector(child)
            continue
        if is_visible_image_path(candidate):
            add_image(candidate)
            continue
        add_vector(candidate)

    return DroppedProjectPayload(tuple(images), tuple(vectors))
