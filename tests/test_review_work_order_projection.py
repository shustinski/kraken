from __future__ import annotations

from pathlib import Path

import pytest

from kraken_manager.infrastructure.projections import rebuild_filesystem_index
from tests.test_work_orders import _project_with_frames, _service


@pytest.mark.parametrize('decision', ['unchanged_return', 'cancel'])
def test_review_decision_updates_linked_work_and_survives_replay(tmp_path, decision):
    service = _service(tmp_path)
    owner, performer, project, layer, images, vectors = _project_with_frames(service, tmp_path / 'source')
    parent = tmp_path / 'usb'
    parent.mkdir()
    order = service.issue_work(
        principal=owner.principal, project_id=project.id, layer_id=layer.id,
        kind='vector_review', placement='external', assignee_id=performer.id,
        coordinates=((1, 1), (2, 1)), image_representation_id=str(images.id),
        vector_representation_id=str(vectors.id), destination_parent=parent, idempotency_key='issue',
    )
    batch = service._projection(project.id).get_review_batch(order.related_batch_id)
    if decision == 'unchanged_return':
        result = service.commit_review_return(
            principal=owner.principal, batch=batch, source=Path(order.destination), idempotency_key='return',
        )
        assert result.batch.state.value == 'completed'
        expected = 'completed'
    else:
        service.cancel_review_batch(principal=owner.principal, batch=batch, idempotency_key='cancel')
        expected = 'cancelled'
    stored = service._projection(project.id).get_work_order(order.id)
    assert stored.state.value == expected and stored.revision == 1
    with service._uow(str(project.id))() as unit:
        rebuild_filesystem_index(unit._backing_event_store, unit._backing_projections)
    assert service._projection(project.id).get_work_order(order.id) == stored
