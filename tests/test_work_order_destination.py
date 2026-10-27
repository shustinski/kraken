from datetime import UTC, datetime
from uuid import uuid4

import pytest

from kraken_manager.domain.common import DomainValidationError
from kraken_manager.domain.work_orders import WorkKind, WorkOrder, WorkPlacement
from kraken_manager.infrastructure.work_package import selection_from_coordinates
from kraken_manager.infrastructure.filesystem._codec import encode_model, decode_model


def order(destination):
    return WorkOrder.create(
        project_id=str(uuid4()),
        layer_id=str(uuid4()),
        kind=WorkKind.STITCH,
        placement=WorkPlacement.EXTERNAL,
        assignee_id=str(uuid4()),
        created_by=str(uuid4()),
        destination=destination,
        frame_count=1,
        selection=selection_from_coordinates(((1, 1),)),
        created_at=datetime.now(UTC),
    )


def test_long_destination_is_preserved_exactly_across_history_codec():
    destination = "D:/" + "/".join(["Reference project data"] * 18) + "/signed-work-package"
    assert len(destination) > 255
    value = order(destination)
    assert value.destination == destination
    assert decode_model(WorkOrder, encode_model(value)) == value


@pytest.mark.parametrize("destination", ["  ", "D:/bad\x00/path", "x" * 32768], ids=["empty", "nul", "unbounded"])
def test_empty_nul_or_unbounded_destination_is_rejected(destination):
    with pytest.raises(DomainValidationError):
        order(destination)
