"""Framework-neutral import planning and preflight diagnostics."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


class ImportMappingMode(StrEnum):
    XY_FILENAME = "xy_filename"
    ROW_MAJOR_SUFFIX = "row_major_suffix"
    REGEX = "regex"
    EXPLICIT = "explicit"


@dataclass(frozen=True, slots=True)
class ImportSource:
    source_key: str
    display_name: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.source_key.strip() or not self.display_name.strip():
            raise ValueError("import source key and display name are required")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("import source size cannot be negative")


@dataclass(frozen=True, slots=True)
class ImportPlanItem:
    source: ImportSource
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class ImportIssue:
    code: str
    message: str
    blocking: bool
    source_keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ImportPlan:
    width: int
    height: int
    items: tuple[ImportPlanItem, ...]
    issues: tuple[ImportIssue, ...]
    total_bytes: int
    missing_coordinates: int

    @property
    def ready(self) -> bool:
        return not any(issue.blocking for issue in self.issues)


class ImportPlanner:
    _XY = re.compile(r"(?<!\d)(?P<x>\d+)_(?P<y>\d+)(?!\d)")
    _SUFFIX = re.compile(r"(?P<index>\d+)$")

    def plan(
        self,
        *,
        width: int,
        height: int,
        sources: tuple[ImportSource, ...],
        mode: ImportMappingMode,
        regex: str | None = None,
        explicit: Mapping[str, tuple[int, int]] | None = None,
    ) -> ImportPlan:
        if isinstance(width, bool) or isinstance(height, bool) or width < 1 or height < 1:
            raise ValueError("import grid dimensions must be positive")
        mode = ImportMappingMode(mode)
        pattern: re.Pattern[str] | None = None
        if mode is ImportMappingMode.REGEX:
            if not regex:
                raise ValueError("regex mapping requires a pattern")
            try:
                pattern = re.compile(regex)
            except re.error as exc:
                raise ValueError(f"invalid import regex: {exc}") from exc
            if not {"x", "y"}.issubset(pattern.groupindex):
                raise ValueError("import regex requires named groups 'x' and 'y'")
        explicit = explicit or {}
        row_major_base = 1
        if mode is ImportMappingMode.ROW_MAJOR_SUFFIX:
            suffixes = (
                match
                for source in sources
                if (match := self._SUFFIX.search(source.display_name.rsplit(".", 1)[0]))
            )
            if any(int(match["index"]) == 0 for match in suffixes):
                row_major_base = 0

        issues: list[ImportIssue] = []
        items: list[ImportPlanItem] = []
        source_keys = [source.source_key.casefold() for source in sources]
        duplicate_keys = sorted(key for key, count in Counter(source_keys).items() if count > 1)
        if duplicate_keys:
            issues.append(
                ImportIssue(
                    "duplicate_source",
                    "The same source was included more than once",
                    True,
                    tuple(duplicate_keys),
                )
            )

        basenames: dict[str, list[str]] = {}
        for source in sources:
            basename = source.display_name.replace("\\", "/").rsplit("/", 1)[-1].casefold()
            basenames.setdefault(basename, []).append(source.source_key)
        for basename, keys in basenames.items():
            if len(keys) > 1:
                issues.append(
                    ImportIssue(
                        "duplicate_basename",
                        f"Several sources have basename {basename!r}",
                        True,
                        tuple(keys),
                    )
                )

        for source in sources:
            coordinate = self._coordinate(
                source,
                width=width,
                mode=mode,
                pattern=pattern,
                explicit=explicit,
                row_major_base=row_major_base,
            )
            if coordinate is None:
                issues.append(
                    ImportIssue(
                        "unmapped_source",
                        f"Could not determine coordinates for {source.display_name}",
                        True,
                        (source.source_key,),
                    )
                )
                continue
            x, y = coordinate
            if not (1 <= x <= width and 1 <= y <= height):
                issues.append(
                    ImportIssue(
                        "outside_grid",
                        f"{source.display_name} maps outside the grid to ({x}, {y})",
                        True,
                        (source.source_key,),
                    )
                )
                continue
            items.append(ImportPlanItem(source, x, y))

        coordinates: dict[tuple[int, int], list[str]] = {}
        for item in items:
            coordinates.setdefault((item.x, item.y), []).append(item.source.source_key)
        for coordinate, keys in coordinates.items():
            if len(keys) > 1:
                issues.append(
                    ImportIssue(
                        "duplicate_coordinate",
                        f"Several sources map to frame {coordinate}",
                        True,
                        tuple(keys),
                    )
                )
        occupied = len(coordinates)
        missing = width * height - occupied
        if missing:
            issues.append(
                ImportIssue(
                    "sparse_coverage",
                    f"Representation will be sparse: {missing} frame(s) have no source",
                    False,
                )
            )
        return ImportPlan(
            width=width,
            height=height,
            items=tuple(sorted(items, key=lambda item: (item.y, item.x, item.source.source_key.casefold()))),
            issues=tuple(issues),
            total_bytes=sum(source.size_bytes for source in sources),
            missing_coordinates=missing,
        )

    def _coordinate(
        self,
        source: ImportSource,
        *,
        width: int,
        mode: ImportMappingMode,
        pattern: re.Pattern[str] | None,
        explicit: Mapping[str, tuple[int, int]],
        row_major_base: int,
    ) -> tuple[int, int] | None:
        if mode is ImportMappingMode.EXPLICIT:
            value = explicit.get(source.source_key)
            if value is None or len(value) != 2:
                return None
            return int(value[0]), int(value[1])
        normalized = source.display_name.replace("\\", "/").rsplit("/", 1)[-1]
        stem = normalized.rsplit(".", 1)[0]
        if mode is ImportMappingMode.XY_FILENAME:
            match = self._XY.search(stem)
            return None if match is None else (int(match["x"]), int(match["y"]))
        if mode is ImportMappingMode.ROW_MAJOR_SUFFIX:
            match = self._SUFFIX.search(stem)
            if match is None:
                return None
            index = int(match["index"]) - row_major_base
            if index < 0:
                return None
            return (index % width + 1, index // width + 1)
        assert pattern is not None
        match = pattern.search(normalized)
        return None if match is None else (int(match["x"]), int(match["y"]))


__all__ = [
    "ImportIssue",
    "ImportMappingMode",
    "ImportPlan",
    "ImportPlanItem",
    "ImportPlanner",
    "ImportSource",
]
