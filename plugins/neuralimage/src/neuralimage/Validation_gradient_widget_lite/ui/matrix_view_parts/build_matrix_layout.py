"""matrix_view: build matrix layout."""
from __future__ import annotations
from typing import TYPE_CHECKING
import math
from ...core.domain import FrameRecord
from ...core.repository import natural_sort_key

from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.compute_auto_color_window import extract_frame_number

def build_matrix_layout(records: list[FrameRecord], layout_config: MatrixLayoutConfig) -> tuple[list[tuple[FrameRecord, int, int]], int, int]:
    """Place frame records into one indexed or custom matrix layout."""
    if not records:
        return [], 0, 0
    mode = str(getattr(layout_config, "mode", "indexed_grid") or "indexed_grid")
    fixed_positions = all(
        record.identity is not None and record.identity.tile_x is not None and record.identity.tile_y is not None
        for record in records
    )
    if mode == "manual_grid":
        rows = int(layout_config.rows)
        columns = int(layout_config.columns)
        if rows <= 0 or columns <= 0:
            raise ValueError("Invalid custom matrix layout")
        capacity = rows * columns
        if len(records) > capacity:
            raise ValueError(f"Custom matrix layout capacity {capacity} is smaller than the frame count {len(records)}")
        if fixed_positions:
            placements: list[tuple[FrameRecord, int, int]] = []
            for record in records:
                row = int(record.identity.tile_y)
                column = int(record.identity.tile_x)
                if row < 0 or row >= rows or column < 0 or column >= columns:
                    raise ValueError("Stored matrix coordinates are outside the configured custom layout")
                placements.append((record, row, column))
            return sorted(placements, key=lambda item: (item[1], item[2], natural_sort_key(item[0].key))), columns, rows
        placements: list[tuple[FrameRecord, int, int]] = []
        for index, record in enumerate(records):
            row = index // columns
            column = index % columns
            placements.append((record, row, column))
        return placements, columns, rows

    columns = int(layout_config.frames_per_row)
    total_frames = int(layout_config.total_frames)
    if columns <= 0 or total_frames <= 0:
        raise ValueError("Invalid indexed matrix layout")
    if fixed_positions:
        rows = max(1, math.ceil(total_frames / columns))
        placements = []
        for record in records:
            row = int(record.identity.tile_y)
            column = int(record.identity.tile_x)
            if row < 0 or row >= rows or column < 0 or column >= columns:
                raise ValueError("Stored matrix coordinates are outside the configured indexed layout")
            placements.append((record, row, column))
        return sorted(placements, key=lambda item: (item[1], item[2], natural_sort_key(item[0].key))), columns, rows
    indexed_records: list[tuple[FrameRecord, int]] = []
    for record in sorted(records, key=lambda item: natural_sort_key(item.display_name or item.key)):
        frame_index = extract_frame_number(record.display_name or record.key)
        indexed_records.append((record, frame_index))
    if not indexed_records:
        return [], columns, 0
    raw_indices = [frame_index for _record, frame_index in indexed_records]
    min_index = min(raw_indices)
    max_index = max(raw_indices)
    one_based_sequential = min_index >= 1 and max_index <= total_frames and 0 not in raw_indices
    index_offset = 1 if one_based_sequential else 0
    normalized_max_index = max_index - index_offset
    if normalized_max_index < 0:
        raise ValueError("Invalid indexed matrix layout")
    effective_total_frames = max(total_frames, normalized_max_index + 1)
    rows = max(1, math.ceil(effective_total_frames / columns))
    placements: list[tuple[FrameRecord, int, int]] = []
    for record, raw_frame_index in indexed_records:
        frame_index = raw_frame_index - index_offset
        if frame_index < 0:
            raise ValueError(f"Frame index {raw_frame_index} is outside the configured matrix range")
        row = frame_index // columns
        column = frame_index % columns
        placements.append((record, row, column))
    return placements, columns, rows

if TYPE_CHECKING:
    from neuralimage.Validation_gradient_widget_lite.ui.matrix_view_parts.score_fill_color import MatrixLayoutConfig
