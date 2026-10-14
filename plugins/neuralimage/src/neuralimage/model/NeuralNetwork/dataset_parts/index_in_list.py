"""dataset: index in list."""
from __future__ import annotations
from bisect import bisect_right

def summarise_list(datalist: list[int]):
    for i in range(len(datalist)):
        if i == 0:
            continue
        datalist[i] += datalist[i - 1]
    return datalist

def index_in_list(index: int, datalist: list[int]):
    """Map a global sample index to (frame_index, local_part_index).

    ``datalist`` is expected to be a cumulative-length lookup list.
    """
    if not datalist:
        raise ValueError('datalist must not be empty')
    if index < 0:
        raise IndexError('dataset index out of range')

    frame = bisect_right(datalist, index)
    if frame == 0:
        return 0, index

    if frame < len(datalist):
        return frame, index - datalist[frame - 1]

    raise IndexError('dataset index out of range')
