from __future__ import annotations

import math
import random

from .models import FrameRange, FrameSelectionMode, LayerInfo, SampleGenerationConfig


class FrameSelectionError(ValueError):
    pass


def parse_frame_range(text: str) -> FrameRange:
    value = text.strip()
    if not value:
        raise FrameSelectionError("Frame range is empty.")
    if "-" not in value:
        frame = int(value)
        return FrameRange(frame, frame)
    first_text, last_text = value.split("-", 1)
    return FrameRange(int(first_text.strip()), int(last_text.strip()))


def frames_from_range(frame_range: FrameRange, frames_in_row: int, mode: FrameSelectionMode) -> tuple[int, ...]:
    if frames_in_row <= 0:
        raise FrameSelectionError("Frames per row must be positive.")
    if mode == "all":
        return tuple(range(frame_range.first, frame_range.last + 1))
    rows = (frame_range.last - frame_range.first) // frames_in_row
    row_end = frame_range.last - rows * frames_in_row
    frames: list[int] = []
    row_start = frame_range.first
    for _ in range(rows + 1):
        frames.extend(range(row_start, row_end + 1))
        row_start += frames_in_row
        row_end += frames_in_row
    return tuple(frames)


def generate_sample_frames(layer: LayerInfo, config: SampleGenerationConfig) -> tuple[int, ...]:
    if config.layer_name != layer.name:
        raise FrameSelectionError(f"Config targets layer {config.layer_name!r}, got {layer.name!r}.")
    if not config.authors:
        raise FrameSelectionError("At least one author must be selected.")
    if not 0 < config.percent_per_author <= 100:
        raise FrameSelectionError("Percent per author must be in range 1..100.")
    if config.frame_range and config.frame_range.last > layer.frames_in_layer:
        raise FrameSelectionError("Frame range exceeds the number of frames in the layer.")

    allowed_frames: set[int] | None = None
    if config.frame_range:
        allowed_frames = set(frames_from_range(config.frame_range, layer.frames_in_row, config.selection_mode))
    if config.date_filtered_frames is not None:
        allowed_frames = set(config.date_filtered_frames) if allowed_frames is None else allowed_frames & set(config.date_filtered_frames)

    rng = random.Random(config.random_seed)
    selected: list[int] = []
    for author in config.authors:
        if author not in layer.author_frames:
            raise FrameSelectionError(f"Unknown author for layer {layer.name!r}: {author!r}.")
        candidates = set(layer.author_frames[author])
        if allowed_frames is not None:
            candidates &= allowed_frames
        ordered_candidates = sorted(candidates)
        if config.percent_per_author == 100:
            selected.extend(ordered_candidates)
            continue
        sample_size = math.ceil(len(ordered_candidates) / 100 * config.percent_per_author)
        selected.extend(rng.sample(ordered_candidates, sample_size))
    return tuple(sorted(set(selected)))
