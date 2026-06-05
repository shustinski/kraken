from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_FRAME_SUFFIX_RE = re.compile(r"_(\d+)$")


@dataclass(frozen=True, slots=True)
class BaseFrameRecord:
    path: str
    stem: str
    frame_number: int | None


def extract_frame_number(file_name: str | Path) -> int | None:
    stem = Path(file_name).stem
    match = _FRAME_SUFFIX_RE.search(stem)
    if match is None:
        return None
    return int(match.group(1))


def sort_base_frame_records(paths: Iterable[str | Path]) -> list[BaseFrameRecord]:
    records = [
        BaseFrameRecord(path=str(Path(path)), stem=Path(path).stem, frame_number=extract_frame_number(path))
        for path in paths
    ]
    return sorted(
        records,
        key=lambda record: (
            record.frame_number is None,
            record.frame_number if record.frame_number is not None else 0,
            record.stem.lower(),
            record.path.lower(),
        ),
    )


def build_base_frame_records(paths: Iterable[str | Path]) -> tuple[list[BaseFrameRecord], list[str]]:
    warnings: list[str] = []
    unique_by_number: dict[int, BaseFrameRecord] = {}
    unnamed_frames: list[BaseFrameRecord] = []

    for record in sort_base_frame_records(paths):
        if record.frame_number is None:
            unnamed_frames.append(record)
            warnings.append(f"Base frame has no numeric suffix and will not be matched to layers: {record.path}")
            continue
        existing = unique_by_number.get(record.frame_number)
        if existing is not None:
            warnings.append(
                f"Duplicate base frame number {record.frame_number}: keep {existing.path}, ignore {record.path}"
            )
            continue
        unique_by_number[record.frame_number] = record

    numbered = [unique_by_number[key] for key in sorted(unique_by_number)]
    return numbered + unnamed_frames, warnings


def build_base_frame_number_map(records: Iterable[BaseFrameRecord]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for record in records:
        if record.frame_number is not None:
            mapping[record.path] = record.frame_number
    return mapping


def build_additional_layer_frame_map(
    paths: Iterable[str | Path],
    *,
    base_frame_numbers: set[int],
) -> tuple[dict[int, str], list[str]]:
    warnings: list[str] = []
    frame_map: dict[int, str] = {}
    ordered_paths = sorted((str(Path(path)) for path in paths), key=lambda value: value.lower())
    for path in ordered_paths:
        frame_number = extract_frame_number(path)
        if frame_number is None:
            warnings.append(f"Ignore layer frame without numeric suffix: {path}")
            continue
        if frame_number not in base_frame_numbers:
            warnings.append(f"Ignore layer frame {frame_number} not present in base layer: {path}")
            continue
        if frame_number in frame_map:
            warnings.append(
                f"Duplicate layer frame number {frame_number}: keep {frame_map[frame_number]}, ignore {path}"
            )
            continue
        frame_map[frame_number] = path
    return frame_map, warnings
