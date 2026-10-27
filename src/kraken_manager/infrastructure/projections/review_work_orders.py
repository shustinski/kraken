"""Project terminal review decisions onto their linked performer work."""
from __future__ import annotations

from kraken_manager.domain.work_orders import WorkKind, WorkOrderState
from kraken_manager.domain.workflows import ReviewBatchState


def synchronize_review_work_orders(store, batch, *, save=None):
    if batch.state not in {ReviewBatchState.COMPLETED, ReviewBatchState.CANCELLED}:
        return
    listing = getattr(store, 'list_work_orders', None)
    if listing is None:  # Compatibility with adapters predating performer work.
        return
    for order in listing(batch.project_id):
        if (
            order.related_batch_id != str(batch.id) or order.kind is not WorkKind.VECTOR_REVIEW
            or order.state in {WorkOrderState.COMPLETED, WorkOrderState.CANCELLED}
        ):
            continue
        at = max(batch.updated_at, order.updated_at)
        changed = order.complete(at=at) if batch.state is ReviewBatchState.COMPLETED else order.cancel(at=at)
        (save or store.save_work_order)(changed)
