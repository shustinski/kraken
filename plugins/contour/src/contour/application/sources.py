"""Loaded frame and vector sources.

Every load (a folder or a set of files) becomes one :class:`FrameSource`.
The :class:`SourceRegistry` turns the sources into the flat state the
workspace needs (ordered image paths and a stem -> CIF index), and explains
how image and vector sources pair up so the UI can show it and remove any
single load again.

Vector priority follows registry order: a vector source later in the list
wins over an earlier one for the same frame stem.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

SOURCE_IMAGES = "images"
SOURCE_VECTORS = "vectors"
ORIGIN_FOLDER = "folder"
ORIGIN_FILES = "files"
_KINDS = (SOURCE_IMAGES, SOURCE_VECTORS)
_ORIGINS = (ORIGIN_FOLDER, ORIGIN_FILES)


def normalize_source_path(path: str | Path) -> str:
    return str(Path(path))


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def frame_stem(path: str | Path) -> str:
    """Stem used to pair frames with vectors (matches the existing CIF index)."""

    return Path(path).stem.lower()


def common_root(paths: Iterable[str]) -> str:
    parents = [str(Path(path).parent) for path in paths]
    if not parents:
        return ""
    try:
        return normalize_source_path(os.path.commonpath(parents))
    except ValueError:  # different drives
        return parents[0]


@dataclass(frozen=True, slots=True)
class FrameSource:
    id: str
    kind: str
    origin: str
    root: str
    paths: tuple[str, ...] = ()
    excluded: frozenset[str] = frozenset()
    locked: bool = False
    scanned: bool = True
    color_index: int = -1  # -1: the registry assigns the next free color

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"Unknown source kind: {self.kind}")
        if self.origin not in _ORIGINS:
            raise ValueError(f"Unknown source origin: {self.origin}")

    @property
    def is_folder(self) -> bool:
        return self.origin == ORIGIN_FOLDER

    @property
    def label(self) -> str:
        root = Path(self.root) if self.root else None
        if self.is_folder and root is not None:
            return root.name or str(root)
        return ""

    def effective_paths(self) -> tuple[str, ...]:
        if not self.excluded:
            return self.paths
        excluded = {_path_key(path) for path in self.excluded}
        return tuple(path for path in self.paths if _path_key(path) not in excluded)

    def stems(self) -> list[str]:
        return [frame_stem(path) for path in self.effective_paths()]

    def contains(self, path: str | Path) -> bool:
        key = _path_key(path)
        return any(_path_key(candidate) == key for candidate in self.effective_paths())

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "origin": self.origin,
            "root": self.root,
            "color_index": self.color_index,
        }
        if self.is_folder:
            if self.excluded:
                payload["excluded"] = sorted(self.excluded)
        else:
            payload["paths"] = list(self.paths)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FrameSource:
        origin = str(payload.get("origin") or "")
        paths = tuple(normalize_source_path(path) for path in payload.get("paths") or () if str(path).strip())
        return cls(
            id=str(payload.get("id") or new_source_id()),
            kind=str(payload.get("kind") or ""),
            origin=origin,
            root=normalize_source_path(payload["root"]) if payload.get("root") else common_root(paths),
            paths=paths,
            excluded=frozenset(str(path) for path in payload.get("excluded") or ()),
            # Folders are rescanned after restore; file lists are known already.
            scanned=origin != ORIGIN_FOLDER,
            color_index=int(payload.get("color_index", -1)),
        )


def new_source_id() -> str:
    return uuid.uuid4().hex[:10]


def folder_source(kind: str, folder: str | Path, *, locked: bool = False) -> FrameSource:
    return FrameSource(
        id=new_source_id(),
        kind=kind,
        origin=ORIGIN_FOLDER,
        root=normalize_source_path(folder),
        locked=locked,
        scanned=False,
    )


def files_source(kind: str, paths: Iterable[str | Path], *, locked: bool = False) -> FrameSource:
    normalized: list[str] = []
    seen: set[str] = set()
    for path in paths:
        text = normalize_source_path(path)
        key = _path_key(text)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return FrameSource(
        id=new_source_id(),
        kind=kind,
        origin=ORIGIN_FILES,
        root=common_root(normalized),
        paths=tuple(normalized),
        locked=locked,
    )


@dataclass(frozen=True, slots=True)
class SourcePair:
    image_source_id: str
    vector_source_id: str
    matched: int
    total: int


@dataclass(frozen=True, slots=True)
class PairingReport:
    pairs: tuple[SourcePair, ...] = ()
    unpaired_vector_counts: Mapping[str, int] = field(default_factory=dict)
    overridden_counts: Mapping[str, int] = field(default_factory=dict)

    def pairs_for_image_source(self, source_id: str) -> list[SourcePair]:
        return [pair for pair in self.pairs if pair.image_source_id == source_id]

    def pairs_for_vector_source(self, source_id: str) -> list[SourcePair]:
        return [pair for pair in self.pairs if pair.vector_source_id == source_id]


class SourceRegistry:
    def __init__(self, sources: Iterable[FrameSource] = ()) -> None:
        self._sources: list[FrameSource] = []
        self._next_color = 0
        for source in sources:
            self.add(source)

    # ---------------------------------------------------------------- access
    def sources(self, kind: str | None = None) -> tuple[FrameSource, ...]:
        return tuple(source for source in self._sources if kind is None or source.kind == kind)

    def get(self, source_id: str) -> FrameSource | None:
        return next((source for source in self._sources if source.id == source_id), None)

    def is_empty(self) -> bool:
        return not self._sources

    def __len__(self) -> int:
        return len(self._sources)

    # ---------------------------------------------------------------- change
    def add(self, source: FrameSource, *, low_priority: bool = False) -> FrameSource:
        """Add a source; ``low_priority`` puts a vector source below the others."""

        if self.get(source.id) is not None:
            raise ValueError(f"Duplicate source id: {source.id}")
        used_colors = {item.color_index for item in self._sources}
        if source.color_index < 0 or source.color_index in used_colors:
            source = replace(source, color_index=self._next_color)
        self._next_color = max(self._next_color, source.color_index + 1)
        if low_priority and source.kind == SOURCE_VECTORS:
            first_vector = next(
                (index for index, item in enumerate(self._sources) if item.kind == SOURCE_VECTORS),
                len(self._sources),
            )
            self._sources.insert(first_vector, source)
        else:
            self._sources.append(source)
        return source

    def update(self, source: FrameSource) -> None:
        for index, existing in enumerate(self._sources):
            if existing.id == source.id:
                self._sources[index] = source
                return
        raise KeyError(source.id)

    def set_scanned_paths(self, source_id: str, paths: Iterable[str | Path]) -> FrameSource | None:
        source = self.get(source_id)
        if source is None:
            return None
        updated = replace(source, paths=tuple(normalize_source_path(path) for path in paths), scanned=True)
        self.update(updated)
        return updated

    def set_vector_priority(self, source_id: str, *, low: bool) -> None:
        """Move a vector source below (``low``) or above all other vector sources."""

        source = self.get(source_id)
        if source is None or source.kind != SOURCE_VECTORS:
            return
        self._sources.remove(source)
        if low:
            first_vector = next(
                (index for index, item in enumerate(self._sources) if item.kind == SOURCE_VECTORS),
                len(self._sources),
            )
            self._sources.insert(first_vector, source)
        else:
            self._sources.append(source)

    def remove(self, source_id: str) -> FrameSource | None:
        source = self.get(source_id)
        if source is not None:
            self._sources.remove(source)
        return source

    def clear(self) -> None:
        self._sources.clear()

    def exclude_images(self, paths: Iterable[str | Path]) -> None:
        """Drop frames: removed from file sources, excluded from folder sources."""

        keys = {_path_key(path) for path in paths}
        for source in list(self.sources(SOURCE_IMAGES)):
            hit = [path for path in source.effective_paths() if _path_key(path) in keys]
            if not hit:
                continue
            if source.is_folder:
                self.update(replace(source, excluded=source.excluded | frozenset(hit)))
            else:
                remaining = tuple(path for path in source.paths if _path_key(path) not in keys)
                if remaining:
                    self.update(replace(source, paths=remaining, root=common_root(remaining)))
                else:
                    self.remove(source.id)

    # ---------------------------------------------------------- derived state
    def image_paths(self) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for source in self.sources(SOURCE_IMAGES):
            for path in source.effective_paths():
                key = _path_key(path)
                if key in seen:
                    continue
                seen.add(key)
                result.append(path)
        return result

    def image_source_for(self, path: str | Path) -> FrameSource | None:
        key = _path_key(path)
        for source in self.sources(SOURCE_IMAGES):
            if any(_path_key(candidate) == key for candidate in source.effective_paths()):
                return source
        return None

    def cif_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for source in self.sources(SOURCE_VECTORS):
            for path in source.effective_paths():
                index[frame_stem(path)] = path
        return index

    def vector_source_for_stem(self, stem: str) -> FrameSource | None:
        winner: FrameSource | None = None
        for source in self.sources(SOURCE_VECTORS):
            if stem in set(source.stems()):
                winner = source
        return winner

    def vector_conflicts(self, candidate: FrameSource) -> set[str]:
        """Stems of ``candidate`` that an existing vector source already provides."""

        if candidate.kind != SOURCE_VECTORS:
            return set()
        existing = set(self.cif_index())
        return existing & set(candidate.stems())

    def image_duplicates(self, candidate: FrameSource) -> list[str]:
        """Paths of ``candidate`` whose frame stem is already loaded."""

        if candidate.kind != SOURCE_IMAGES:
            return []
        loaded = {frame_stem(path) for path in self.image_paths()}
        return [path for path in candidate.effective_paths() if frame_stem(path) in loaded]

    def pairing(self) -> PairingReport:
        image_sources = self.sources(SOURCE_IMAGES)
        vector_sources = self.sources(SOURCE_VECTORS)
        all_image_stems: set[str] = set()
        image_stems_by_source: dict[str, set[str]] = {}
        for source in image_sources:
            stems = set(source.stems())
            image_stems_by_source[source.id] = stems
            all_image_stems |= stems
        winners: dict[str, str] = {}
        for source in vector_sources:
            for stem in source.stems():
                winners[stem] = source.id
        pairs: list[SourcePair] = []
        unpaired: dict[str, int] = {}
        overridden: dict[str, int] = {}
        for vector in vector_sources:
            vector_stems = set(vector.stems())
            overridden[vector.id] = sum(1 for stem in vector_stems if winners.get(stem) != vector.id)
            unpaired[vector.id] = len(vector_stems - all_image_stems)
            for image in image_sources:
                image_stems = image_stems_by_source[image.id]
                matched = len(image_stems & vector_stems)
                if matched:
                    pairs.append(SourcePair(image.id, vector.id, matched, len(image_stems)))
        return PairingReport(tuple(pairs), unpaired, overridden)

    # ------------------------------------------------------------ persistence
    def to_payload(self) -> list[dict[str, Any]]:
        return [source.to_dict() for source in self._sources if not source.locked]

    @classmethod
    def from_payload(cls, payload: Iterable[Mapping[str, Any]]) -> SourceRegistry:
        registry = cls()
        for item in payload:
            try:
                registry.add(FrameSource.from_dict(item))
            except (KeyError, TypeError, ValueError):
                continue
        return registry


def migrate_legacy_sources(
    *,
    input_directory: str,
    cif_directory: str,
    explicit_image_list: bool,
    image_paths: Iterable[str],
    vector_paths: Iterable[str],
) -> SourceRegistry:
    """Build sources from the settings saved before sources existed."""

    registry = SourceRegistry()
    images = [path for path in image_paths if str(path).strip()]
    if not explicit_image_list and input_directory.strip():
        registry.add(folder_source(SOURCE_IMAGES, input_directory))
    elif images:
        registry.add(files_source(SOURCE_IMAGES, images))
    if cif_directory.strip():
        registry.add(folder_source(SOURCE_VECTORS, cif_directory))
    vectors = [path for path in vector_paths if str(path).strip()]
    if vectors:
        folder_root = _path_key(cif_directory) if cif_directory.strip() else None
        extra = [path for path in vectors if folder_root is None or _path_key(Path(path).parent) != folder_root]
        if extra:
            registry.add(files_source(SOURCE_VECTORS, extra))
    return registry


__all__ = [
    "ORIGIN_FILES",
    "ORIGIN_FOLDER",
    "SOURCE_IMAGES",
    "SOURCE_VECTORS",
    "FrameSource",
    "PairingReport",
    "SourcePair",
    "SourceRegistry",
    "common_root",
    "files_source",
    "folder_source",
    "frame_stem",
    "migrate_legacy_sources",
    "new_source_id",
    "normalize_source_path",
]
