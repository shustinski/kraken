"""Remote work packaging keeps version IDs and signed review lifecycle intact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest

from kraken_hub.remote_client_parts.issue_work import IssueWorkMixin
from kraken_manager.domain.work_orders import WorkOrder
from tests.test_work_orders import _project_with_frames, _service


@pytest.fixture
def remote(tmp_path):
    service = _service(tmp_path)
    owner, performer, project, layer, images, vectors = _project_with_frames(service, tmp_path / 'source')
    client = IssueWorkMixin()
    client.get_project = lambda _id: service.get_project(project.id)
    client.list_layers = lambda _id: service.list_layers(project.id)
    client.list_performers = lambda: service.list_performers()
    client.active_artifact_version = lambda _project, series: service._projection(project.id).get_active_artifact_version(series)
    client.artifact_version = lambda _project, version: service._projection(project.id).get_artifact_version(version)
    requested = []

    def read(_project, version_id):
        requested.append(version_id)
        version = service._projection(project.id).get_artifact_version(version_id)
        assert version is not None, 'Remote blob API expects a version UUID, not its SHA-256'
        return service.read_project_blob(project.id, version.sha256)

    client.read_project_blob = read
    client.create_review_batch = service.create_review_batch
    client.export_review_batch = service.export_review_batch
    client._call = lambda _method, _path, *, payload, **_kwargs: payload['record']
    parent = tmp_path / 'usb'
    parent.mkdir()
    arguments = dict(principal=owner.principal, project_id=project.id, layer_id=layer.id,
                     placement='external', assignee_id=performer.id, coordinates=((1, 1),),
                     image_representation_id=str(images.id), vector_representation_id=str(vectors.id),
                     destination_parent=parent, idempotency_key=str(uuid4()))
    return client, service, project, arguments, requested


def test_external_remote_vectorization_downloads_by_version_uuid(remote):
    client, _service, _project, arguments, requested = remote
    order = client.issue_work(kind='vectorize', **arguments)
    assert isinstance(order, WorkOrder)
    assert len(requested) == 1 and len(requested[0]) == 36
    assert (Path(order.destination) / 'images/1_1.png').read_bytes() == b'image-a'


def test_external_remote_review_is_signed_and_links_issued_review_batch(remote):
    client, service, project, arguments, requested = remote
    order = client.issue_work(kind='vector_review', **arguments)
    assert order.related_batch_id
    batch = service._projection(project.id).get_review_batch(order.related_batch_id)
    assert batch is not None and batch.state.value == 'issued'
    assert batch.selection == order.selection and len(batch.items) == order.frame_count
    assert (Path(order.destination) / 'kraken-review.sig').is_file()
    vectors = list((Path(order.destination) / 'vectors').glob('*.cif'))
    assert len(vectors) == 1
    assert hashlib.sha256(vectors[0].read_bytes()).hexdigest() == batch.items[0].vector_sha256
    assert requested == []
    package = Path(order.destination)
    manifest = json.loads((package / 'kraken-review.json').read_text(encoding='utf-8'))
    expected = {item['relative_path'] for item in manifest['files']}
    actual = {path.relative_to(package).as_posix() for path in package.rglob('*') if path.is_file()}
    assert actual == expected | {'kraken-review.json', 'kraken-review.sig'}
