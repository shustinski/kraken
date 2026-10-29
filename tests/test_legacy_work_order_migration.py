"""Preserve projection-only work before rebuilding its disposable cache."""
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
from pathlib import Path

import pytest

from kraken_manager.domain.identity import Principal
from kraken_manager.domain.work_orders import WorkKind, WorkOrder, WorkPlacement
from kraken_manager.infrastructure.projections import rebuild_filesystem_index
from kraken_manager.infrastructure.work_package import selection_from_coordinates
from tests.test_work_orders import _project_with_frames, _service


@pytest.fixture
def legacy(tmp_path):
    service = _service(tmp_path)
    owner, performer, project, layer, _images, _vectors = _project_with_frames(service, tmp_path/'source', with_vectors=False)
    order = WorkOrder.create(project_id=project.id, layer_id=layer.id, kind=WorkKind.STITCH,
        placement=WorkPlacement.EXTERNAL, assignee_id=performer.id, created_by=owner.principal.id,
        destination=str(tmp_path/'unavailable-external-drive'), frame_count=1,
        selection=selection_from_coordinates(((1, 1),)), created_at=datetime(2020, 1, 1, tzinfo=UTC),
        instructions='Preserve original legacy instructions').mark_running(at=datetime(2020, 1, 2, tzinfo=UTC))
    order = replace(order, revision=7)
    projection = service._projection(project.id)
    projection.save_work_order(order)
    return service, owner, project, order, projection


def test_rebuild_refuses_to_discard_projection_only_work_and_keeps_original_index(legacy):
    service, _owner, project, order, projection = legacy
    before = hashlib.sha256(projection.path.read_bytes()).hexdigest()
    with service._uow(str(project.id))() as unit:
        with pytest.raises(RuntimeError, match='legacy work order'):
            rebuild_filesystem_index(unit._backing_event_store, projection)
    assert hashlib.sha256(projection.path.read_bytes()).hexdigest() == before
    assert projection.get_work_order(order.id) == order


def test_reviewed_migration_preserves_snapshot_then_survives_rebuild(legacy):
    service, owner, project, order, projection = legacy
    plan = service.plan_legacy_work_order_migration(principal=owner.principal, project_id=project.id)
    assert [row['id'] for row in plan['legacy']] == [order.id]
    assert plan['legacy'][0]['revision'] == 7 and not Path(order.destination).exists()
    result = service.migrate_legacy_work_orders(principal=owner.principal, project_id=project.id,
        expected_plan_digest=plan['digest'], idempotency_key='legacy-import')
    assert result['migrated'] == 1
    assert service._projection(project.id).get_work_order(order.id) == order
    with service._uow(str(project.id))() as unit:
        events = unit.event_store.load_stream(f'work-order:{order.id}')
        rebuild_filesystem_index(unit._backing_event_store, projection)
    assert len(events) == 1 and events[0].event_type == 'WorkOrderImportedFromProjectionV1'
    assert events[0].recorded_at > order.updated_at
    assert events[0].actor.principal_id == owner.principal.id
    assert service._projection(project.id).get_work_order(order.id) == order


def test_unauthorized_migration_does_not_capture_or_disclose_legacy_work(legacy):
    service, _owner, project, _order, _projection = legacy
    outsider = Principal.local(subject='migration-outsider', display_name='Outsider')
    service.identities.save(outsider)
    from kraken_manager.application.errors import AuthorizationError
    with pytest.raises(AuthorizationError):
        service.plan_legacy_work_order_migration(principal=outsider, project_id=project.id)


def apply(legacy, *, key='legacy-import', plan=None, principal=None):
    service, owner, project, _order, _projection = legacy
    actor = principal or owner.principal
    plan = plan or service.plan_legacy_work_order_migration(principal=actor, project_id=project.id)
    return service.migrate_legacy_work_orders(principal=actor, project_id=project.id,
        expected_plan_digest=plan['digest'], idempotency_key=key)


def test_current_snapshot_and_cached_observations_are_both_preserved(legacy):
    service, owner, project, order, projection = legacy
    changed = replace(order, revision=8, instructions='Second known legacy observation')
    projection.save_work_order(changed)
    plan = service.plan_legacy_work_order_migration(principal=owner.principal, project_id=project.id)
    observed = plan['legacy'][0]['observations']
    assert [row['snapshot']['instructions'] for row in observed] == [order.instructions, changed.instructions]
    apply(legacy, plan=plan)
    projection.destroy_cache()
    assert service._projection(project.id).get_work_order(order.id) == changed
    with service._uow(str(project.id))() as unit:
        event, = unit.event_store.load_stream(f'work-order:{order.id}')
    assert len(event.payload['legacy_observations']) == 2
    assert event.payload['prior_history'] == 'unknown'
    assert event.effective_at == changed.updated_at


@pytest.mark.parametrize('change', ['snapshot', 'history-only'])
def test_changed_reviewed_plan_cannot_capture_unreviewed_data(legacy, change):
    service, owner, project, order, projection = legacy
    plan = service.plan_legacy_work_order_migration(principal=owner.principal, project_id=project.id)
    projection.save_work_order(replace(order, instructions='changed') if change == 'snapshot' else order)
    from kraken_manager.application.errors import ConcurrencyError
    with pytest.raises(ConcurrencyError, match='fresh migration plan'):
        apply(legacy, plan=plan)
    with service._uow(str(project.id))() as unit:
        assert unit.event_store.load_stream(f'work-order:{order.id}') == ()
    assert projection.get_work_order(order.id).instructions == ('changed' if change == 'snapshot' else order.instructions)


def test_idempotent_retry_returns_original_receipt_and_normal_work_continues(legacy):
    service, owner, project, order, _projection = legacy
    plan = service.plan_legacy_work_order_migration(principal=owner.principal, project_id=project.id)
    first = apply(legacy, plan=plan)
    assert apply(legacy, plan=plan) == first
    cancelled = service.cancel_work_order(order, principal=owner.principal, idempotency_key='cancel-after-import')
    assert cancelled.revision == 8
    service._projection(project.id).destroy_cache()
    assert service._projection(project.id).get_work_order(order.id) == cancelled
    with service._uow(str(project.id))() as unit:
        assert len(unit.event_store.load_stream(f'work-order:{order.id}')) == 2


def test_export_and_cache_deletion_refuse_unrecorded_work(legacy, tmp_path):
    from kraken_manager.infrastructure.blob import FilesystemBlobStore
    from kraken_manager.infrastructure.filesystem import FilesystemEventStore
    from kraken_manager.infrastructure.migration import CanonicalBundleExporter
    service, _owner, project, order, projection = legacy
    exporter = CanonicalBundleExporter(FilesystemEventStore(service.catalog_root, str(project.id)),
        FilesystemBlobStore.for_project(service.catalog_root, str(project.id)))
    for operation in (lambda: exporter.plan(tmp_path/'backup'), lambda: exporter.export(tmp_path/'backup'), projection.destroy_cache):
        with pytest.raises(RuntimeError, match='legacy work order'):
            operation()
    assert projection.get_work_order(order.id) == order and not (tmp_path/'backup').exists()


def test_rebuild_failure_after_durable_import_can_be_retried_without_duplicate_events(legacy, monkeypatch):
    import kraken_manager.infrastructure.projections as projections
    service, owner, project, order, projection = legacy
    plan = service.plan_legacy_work_order_migration(principal=owner.principal, project_id=project.id)
    original = projections.rebuild_filesystem_index
    def fail(*_args, **_kwargs):
        raise OSError('controlled replay failure')
    monkeypatch.setattr(projections, 'rebuild_filesystem_index', fail)
    with pytest.raises(OSError, match='replay'):
        apply(legacy, plan=plan)
    with service._uow(str(project.id))() as unit:
        assert len(unit.event_store.load_stream(f'work-order:{order.id}')) == 1
        assert projection.checkpoint < unit._backing_event_store.last_global_position()
    assert projection.get_work_order(order.id) == order
    monkeypatch.setattr(projections, 'rebuild_filesystem_index', original)
    assert apply(legacy, plan=plan)['migrated'] == 1
    assert service.get_project(project.id) is not None


def test_durable_role_revocation_overrides_stale_acl_during_migration(legacy):
    from kraken_manager.application.errors import AuthorizationError
    from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
    service, owner, project, _order, _projection = legacy
    with service._uow(str(project.id))() as unit:
        # Model the crash boundary: canonical revocation durable, ACL checkpoint
        # not updated. The legacy guard prevents ordinary full cache replay.
        store = unit._backing_event_store
        stream = f'acl:{project.id}:{owner.principal.id}'
        revision = store.current_revision(stream)
        store.append(stream, expected_revision=revision, events=(EventEnvelope.create(
            stream_id=stream, project_id=project.id, revision=revision+1, event_type='ProjectRoleRevoked',
            payload={'principal_id': str(owner.principal.id), 'role': 'maintainer'},
            actor=ActorSnapshot.from_principal(owner.principal)),))
    with pytest.raises(AuthorizationError):
        service.plan_legacy_work_order_migration(principal=owner.principal, project_id=project.id)


def test_backup_restore_contains_imported_snapshot_and_legacy_observations(legacy, tmp_path):
    service, owner, project, order, _projection = legacy
    apply(legacy)
    service.export_backup(project.id, tmp_path/'backup', principal=owner.principal)
    target = _service(tmp_path/'restored')
    account = target.create_initial_account('restore-owner', 'Restore owner', '')
    restored = target.import_backup(tmp_path/'backup', principal=account.principal, take_ownership=True)
    assert target._projection(restored.id).get_work_order(order.id) == order
    with target._uow(str(restored.id))() as unit:
        event, = unit.event_store.load_stream(f'work-order:{order.id}')
    assert event.payload['legacy_observations'][0]['snapshot']['instructions'] == order.instructions


def test_catalog_stays_visible_and_migration_recovers_other_lagging_state(legacy):
    from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
    from kraken_manager.infrastructure.filesystem._codec import encode_model
    service, owner, project, order, projection = legacy
    with service._uow(str(project.id))() as unit:
        store = unit._backing_event_store
        stream = f'project:{project.id}'
        revision = store.current_revision(stream)
        store.append(stream, expected_revision=revision, events=(EventEnvelope.create(
            stream_id=stream, project_id=project.id, revision=revision+1, event_type='ProjectRenamed',
            payload={'project': encode_model(replace(project, name='Recovered canonical name', revision=revision+1))},
            actor=ActorSnapshot.from_principal(owner.principal)),))
    assert str(project.id) in {str(p.id) for p in service.list_projects()}
    with pytest.raises(RuntimeError, match='legacy work order'):
        service.get_project(project.id)
    apply(legacy)
    assert service.get_project(project.id).name == 'Recovered canonical name'
    assert service._projection(project.id).get_work_order(order.id) == order
    assert projection.checkpoint == unit._backing_event_store.last_global_position()


def test_import_actor_is_migrator_and_original_creator_is_preserved(legacy):
    from kraken_manager.domain.identity import ProjectRole
    service, owner, project, order, _projection = legacy
    migrator = Principal.local(subject='migration-maintainer', display_name='Migration maintainer')
    service.identities.save(migrator)
    service.assign_project_role(principal=owner.principal, project=project,
        target_principal_id=migrator.id, role=ProjectRole.MAINTAINER, expected_revision=0,
        idempotency_key='migration-maintainer-role')
    plan = service.plan_legacy_work_order_migration(principal=migrator, project_id=project.id)
    apply(legacy, plan=plan, principal=migrator)
    with service._uow(str(project.id))() as unit:
        event, = unit.event_store.load_stream(f'work-order:{order.id}')
    assert event.actor.principal_id == migrator.id
    assert event.payload['work_order']['created_by'] == str(owner.principal.id)
    from kraken_manager.application.errors import ConcurrencyError
    with pytest.raises(ConcurrencyError, match='another command or actor'):
        apply(legacy, plan=plan, principal=owner.principal)


def test_resigned_cross_project_observation_is_not_accepted_by_rebuilder(legacy, tmp_path):
    from uuid import uuid4
    from kraken_manager.application.legacy_work_orders import snapshot_digest
    from kraken_manager.infrastructure.filesystem._codec import encode_model
    from kraken_manager.infrastructure.filesystem import SQLiteProjectionStore
    from kraken_manager.infrastructure.projections import ProjectionRebuilder
    service, _owner, project, order, _projection = legacy
    apply(legacy)
    with service._uow(str(project.id))() as unit:
        event, = unit.event_store.load_stream(f'work-order:{order.id}')
    payload = encode_model(event.payload)
    payload['legacy_observations'][0]['snapshot']['project_id'] = str(uuid4())
    payload['observations_sha256'] = snapshot_digest(payload['legacy_observations'], encode_model)
    target = SQLiteProjectionStore(tmp_path/'wrong-observation', str(project.id))
    with pytest.raises(ValueError, match='another work order/project'):
        ProjectionRebuilder(target).apply(replace(event, payload=payload))
    assert target.get_work_order(order.id) is None


def test_future_acl_schema_cannot_grant_migration_authority(legacy):
    from kraken_manager.application.errors import ConcurrencyError
    from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
    service, owner, project, _order, _projection = legacy
    outsider = Principal.local(subject='future-schema-user', display_name='Future schema user')
    service.identities.save(outsider)
    with service._uow(str(project.id))() as unit:
        stream = f'acl:{project.id}:{outsider.id}'
        unit._backing_event_store.append(stream, expected_revision=0, events=(EventEnvelope.create(
            stream_id=stream, project_id=project.id, revision=1, event_type='ProjectRoleAssigned', schema_version=2,
            payload={'principal_id': str(outsider.id), 'role': 'maintainer'},
            actor=ActorSnapshot.from_principal(owner.principal)),))
    with pytest.raises(ConcurrencyError, match='Unsupported ACL'):
        service.plan_legacy_work_order_migration(principal=outsider, project_id=project.id)
