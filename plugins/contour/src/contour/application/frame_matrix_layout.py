"""Project frames-per-row forwarded into the Contour frame matrix."""

from __future__ import annotations

FRAMES_PER_ROW_OPTION = "--frames-per-row"


def coerce_frames_per_row(value: object) -> int | None:
    """Return a project grid width, or None when the project did not provide one."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        width = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if width < 1:
        return None
    return width


def matrix_layout_columns(frames_per_row: int) -> int:
    columns = coerce_frames_per_row(frames_per_row)
    if columns is None:
        raise ValueError("frames per row must be a positive integer")
    return columns


def frame_matrix_cell(frame_index: int, frames_per_row: int) -> tuple[int, int]:
    """Row and column of a frame in the matrix that uses the project row length."""

    columns = matrix_layout_columns(frames_per_row)
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("frame index must be a non-negative integer")
    return divmod(frame_index, columns)


def frames_per_row_arguments(frames_per_row: int) -> tuple[str, str]:
    return FRAMES_PER_ROW_OPTION, str(matrix_layout_columns(frames_per_row))


def with_frames_per_row_argument(arguments: list[str], frames_per_row: object) -> list[str]:
    """Insert the project value once. Do not invent a width when it is absent."""

    if FRAMES_PER_ROW_OPTION in arguments:
        return list(arguments)
    width = coerce_frames_per_row(frames_per_row)
    if width is None:
        return list(arguments)
    option, text = frames_per_row_arguments(width)
    return [option, text, *arguments]
