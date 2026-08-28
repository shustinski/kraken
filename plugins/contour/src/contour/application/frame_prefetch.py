"""Neighborhood of frames that should stay decoded for instant switching."""

from __future__ import annotations


def neighborhood_indices(
    current_index: int,
    frame_count: int,
    *,
    list_radius: int = 2,
    matrix_enabled: bool = False,
    columns: int = 1,
    matrix_row_radius: int = 2,
) -> tuple[int, ...]:
    """Return current index plus list ±radius and, if the matrix is on, same-column ±rows."""

    if frame_count <= 0 or current_index < 0 or current_index >= frame_count:
        return ()
    ordered: list[int] = [current_index]
    seen = {current_index}

    def add(index: int) -> None:
        if 0 <= index < frame_count and index not in seen:
            seen.add(index)
            ordered.append(index)

    for offset in range(1, max(0, list_radius) + 1):
        add(current_index - offset)
        add(current_index + offset)
    if matrix_enabled and columns > 0:
        for row_offset in range(1, max(0, matrix_row_radius) + 1):
            add(current_index - row_offset * columns)
            add(current_index + row_offset * columns)
    return tuple(ordered)
