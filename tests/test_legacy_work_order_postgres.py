"""Actual PostgreSQL and HTTP/Qt acceptance of the legacy snapshot import."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import socket
import threading
import time
from uuid import uuid4

import pytest
import uvicorn
from PyQt6.QtCore import Qt

from kraken_hub.presentation.qt.legacy_work_order_dialog import LegacyWorkOrderMigrationDialog
from kraken_hub.infrastructure.remote.client import RemoteHttpClient, RemoteServerError, RemoteServerProjectService
from kraken_manager.domain.identity import Performer, Principal, SystemRole
from kraken_manager.domain.selection import FrameSelectionV1
from kraken_manager.domain.work_orders import WorkKind, WorkOrder, WorkPlacement
from kraken_manager.infrastructure.filesystem import FilesystemEventStore, SQLiteProjectionStore
from kraken_manager.infrastructure.postgres.event_store import PostgresEventStore
from kraken_manager.infrastructure.postgres.performer_store import PostgresPerformerStore
from kraken_manager.infrastructure.projections import rebuild_filesystem_index
from kraken_server.app import create_app
from kraken_server.services import CommandContext, ConflictError, ForbiddenError
from tests.test_admin_project_workflows import project

pytest_plugins = ["tests.test_admin_project_workflows"]


def old_work(database, tmp_path):
    services, _accounts, owner, _config = database
    pid = project(services, owner)['project_id']
    layer = services.create_layer(pid, {'name': 'M1', 'type': 'metal', 'order': 0},
                                 CommandContext(owner.id, str(uuid4()), 1))
    performer = PostgresPerformerStore(services.engine).create(Performer.create(name='Legacy engineer', color='#446688'))
    order = WorkOrder.create(project_id=pid, layer_id=layer['layer_id'], kind=WorkKind.STITCH,
        placement=WorkPlacement.EXTERNAL, assignee_id=performer.id, created_by=owner.id,
        destination=str(tmp_path/'unavailable-original-package'), frame_count=1,
        selection=FrameSelectionV1.rectangle(1, 1, 1, 1), instructions='Legacy PostgreSQL instructions')
    services.projections.save_work_order(order)
    order = replace(order, revision=5, instructions='Latest known legacy snapshot')
    services.projections.save_work_order(order)
    return services, owner, order


def test_postgres_import_preserves_observations_and_replays_into_filesystem(database, tmp_path):
    services, owner, order = old_work(database, tmp_path)
    plan = services.plan_legacy_work_order_migration(str(order.project_id), str(owner.id))
    assert len(plan['legacy'][0]['observations']) == 2
    result = services.migrate_legacy_work_orders(str(order.project_id), {'expected_plan_digest': plan['digest']},
                                                CommandContext(owner.id, 'legacy-import', None))
    assert result['migrated'] == 1 and services.projections.get_work_order(order.id) == order
    target_events = FilesystemEventStore(tmp_path/'replayed', str(order.project_id))
    target_events.append_preserved(event for _position, event in services.events.list_project_events(order.project_id))
    target = SQLiteProjectionStore(tmp_path/'replayed', str(order.project_id))
    rebuild_filesystem_index(target_events, target)
    assert target.get_work_order(order.id) == order
    imported, = target_events.load_stream(f'work-order:{order.id}')
    assert len(imported.payload['legacy_observations']) == 2


def test_postgres_failure_rolls_back_all_import_events_and_receipt(database, tmp_path, monkeypatch):
    services, owner, order = old_work(database, tmp_path)
    plan = services.plan_legacy_work_order_migration(str(order.project_id), str(owner.id))
    original = PostgresEventStore.append
    def fail_receipt(self, stream_id, **kwargs):
        result = original(self, stream_id, **kwargs)
        if stream_id.startswith('work-order-migrations:'):
            raise OSError('controlled receipt persistence failure')
        return result
    monkeypatch.setattr(PostgresEventStore, 'append', fail_receipt)
    with pytest.raises(OSError, match='receipt'):
        services.migrate_legacy_work_orders(str(order.project_id), {'expected_plan_digest': plan['digest']},
                                            CommandContext(owner.id, 'legacy-import', None))
    assert services.events.load_stream(f'work-order:{order.id}') == ()
    assert services.events.find_by_idempotency_key(order.project_id, 'legacy-import') == ()
    assert services.projections.get_work_order(order.id) == order


def test_concurrent_postgres_migrations_have_one_winner(database, tmp_path):
    services, owner, order = old_work(database, tmp_path)
    plan = services.plan_legacy_work_order_migration(str(order.project_id), str(owner.id))
    barrier = threading.Barrier(2)
    def migrate(index):
        barrier.wait(timeout=5)
        try:
            return services.migrate_legacy_work_orders(str(order.project_id), {'expected_plan_digest': plan['digest']},
                                                       CommandContext(owner.id, f'import-{index}', None))
        except ConflictError:
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(migrate, (1, 2)))
    assert sum(r is not None for r in results) == 1
    assert len(services.events.load_stream(f'work-order:{order.id}')) == 1


def test_network_migration_does_not_reintroduce_stripped_console_admin_role(database, tmp_path):
    services, _owner, order = old_work(database, tmp_path)
    console_admin = replace(Principal.local(subject='console-only', display_name='Console admin'),
                            system_roles=frozenset({SystemRole.SERVER_ADMIN}))
    services.identities.save(console_admin)
    with pytest.raises(ForbiddenError):
        services.plan_legacy_work_order_migration(str(order.project_id), str(console_admin.id))


def test_actual_authenticated_http_qt_retry_after_discarded_reply(database, tmp_path, qtbot):
    services, _owner, order = old_work(database, tmp_path)
    accounts = database[1]
    app = create_app(services=services, account_store=accounts)
    server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='off'))
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    url = f'http://127.0.0.1:{sock.getsockname()[1]}'
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    class RecordingHttp(RemoteHttpClient):
        discard_reply = False
        def request(self, method, path, **kwargs):
            result = super().request(method, path, **kwargs)
            if self.discard_reply and method == 'POST' and path.endswith('/legacy-migration'):
                self.discard_reply = False
                raise OSError('controlled client-side loss of committed reply')
            return result
    dialog = None
    try:
        deadline = time.monotonic()+5
        while not server.started and time.monotonic() < deadline:
            time.sleep(.01)
        assert server.started
        http = RecordingHttp(url)
        remote = RemoteServerProjectService.authenticate_local(url, username='admin', password='password',
            data_dir=tmp_path/'remote', http=http)
        remote.acting_role = 'viewer'
        with pytest.raises(RemoteServerError) as denied:
            remote.plan_legacy_work_order_migration(principal=remote.auth.principal, project_id=order.project_id)
        assert denied.value.status == 403
        del remote.acting_role
        dialog = LegacyWorkOrderMigrationDialog(remote, remote.auth.principal, order.project_id)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        qtbot.addWidget(dialog)
        dialog.show()
        qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)
        assert dialog._plan and dialog.apply_button.isEnabled()
        http.discard_reply = True
        dialog.apply_button.click()
        qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)
        assert 'client-side loss' in dialog.status.text() and dialog.apply_button.isEnabled()
        assert len(services.events.load_stream(f'work-order:{order.id}')) == 1
        dialog.apply_button.click()
        qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)
        assert dialog._plan is None and '1' in dialog.status.text()
        assert len(services.events.load_stream(f'work-order:{order.id}')) == 1
        assert services.projections.get_work_order(order.id) == order
    finally:
        if dialog is not None:
            dialog.close()
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        assert not thread.is_alive()
