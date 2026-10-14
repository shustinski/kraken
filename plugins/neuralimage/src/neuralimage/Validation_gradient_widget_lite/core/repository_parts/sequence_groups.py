"""repository: sequence groups."""
from __future__ import annotations
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    FrameRecord,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.analytics_worker_count import natural_sort_key

def _sequence_groups(records: tuple[FrameRecord, ...]) -> dict[str, list[FrameRecord]]:
    groups: dict[str, list[FrameRecord]] = {}
    for record in records:
        sequence_id = record.identity.sequence_id if record.identity is not None and record.identity.sequence_id else "__root__"
        groups.setdefault(sequence_id, []).append(record)
    for items in groups.values():
        items.sort(key=lambda item: natural_sort_key(item.key))
    return groups
