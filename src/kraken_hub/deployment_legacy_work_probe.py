"""Own legacy metadata fixture and real Qt migration in the installed Desktop."""
import time

from kraken_manager.domain.selection import FrameSelectionV1
from kraken_manager.domain.work_orders import WorkKind, WorkOrder, WorkPlacement


def probe_legacy_work_orders(service, session, project, run, app):
    from kraken_hub.legacy_work_order_dialog import LegacyWorkOrderMigrationDialog
    from kraken_manager.infrastructure.migration import BundleVerifier, MigrationBundleError
    from kraken_manager.infrastructure.filesystem import FilesystemEventStore

    performer = service.create_manual_performer(name="Deployment legacy-work diagnostic", color="#446688")
    layer = service.list_layers(project.id)[0]
    order = WorkOrder.create(project_id=project.id, layer_id=layer.id, kind=WorkKind.STITCH,
        placement=WorkPlacement.EXTERNAL, assignee_id=performer.id, created_by=session.principal.id,
        destination=str(run / "absent-legacy-work-package"), frame_count=1,
        selection=FrameSelectionV1.rectangle(1, 1, 1, 1), instructions="Own diagnostic projection-only work")
    projection = service._projection(project.id)
    projection.save_work_order(order)
    order = order.mark_running()
    projection.save_work_order(order)
    dialog = LegacyWorkOrderMigrationDialog(service, session.principal, project.id)

    def stopped():
        deadline = time.monotonic()+20
        while dialog._worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        if dialog._worker is not None:
            raise TimeoutError("Legacy work migration did not reach a stopped-thread state")
    try:
        dialog.show()
        stopped()
        if not dialog._plan or len(dialog._plan["legacy"]) != 1 or len(dialog._plan["legacy"][0]["observations"]) != 2:
            raise RuntimeError("Installed legacy migration did not read its own current/history fixture")
        if not dialog.grab().save(str(run / "legacy-work-order-plan.png")):
            raise RuntimeError("Legacy work migration preview did not render")
        dialog.apply_button.click()
        stopped()
        if dialog._plan is not None or service._projection(project.id).get_work_order(order.id) != order:
            raise RuntimeError("Installed legacy migration did not preserve the approved snapshot")
        projection.destroy_cache()
        if service._projection(project.id).get_work_order(order.id) != order:
            raise RuntimeError("Imported legacy work did not survive cache destruction/replay")
        events = [e for e in service.history(project.id) if e.event_type == "WorkOrderImportedFromProjectionV1"]
        if len(events) != 1 or len(events[0].payload["legacy_observations"]) != 2:
            raise RuntimeError("Imported legacy work lost its cached observations")
        if (run / "absent-legacy-work-package").exists():
            raise RuntimeError("Metadata migration unexpectedly created a work package")
        target = run / "legacy-work-backup"
        manifest = service.export_backup(project.id, target, principal=session.principal)
        BundleVerifier().verify(target).raise_for_errors()
        cancelled = service.cancel_work_order(order, principal=session.principal, idempotency_key="backup-freshness-cancel")
        try:
            service.export_backup(project.id, target, principal=session.principal)
        except MigrationBundleError:
            pass
        else:
            raise RuntimeError("Backup export silently returned an obsolete completed archive")
        latest_target = run / "latest-work-backup"
        latest = service.export_backup(project.id, latest_target, principal=session.principal)
        BundleVerifier().verify(target).raise_for_errors()
        BundleVerifier().verify(latest_target).raise_for_errors()
        if latest.bundle_id == manifest.bundle_id or latest.event_count != manifest.event_count + 1:
            raise RuntimeError("Latest backup did not capture the additional metadata event")
        # Restore into an independent catalogue and replay its current work state.
        from kraken_manager.infrastructure.blob import FilesystemBlobStore
        from kraken_manager.infrastructure.filesystem import SQLiteProjectionStore
        from kraken_manager.infrastructure.migration import CanonicalBundleImporter
        from kraken_manager.infrastructure.projections import rebuild_filesystem_index

        restored_root = run / "backup-freshness-restored"
        restored_events = FilesystemEventStore(restored_root, str(project.id))
        restored_blobs = FilesystemBlobStore.for_project(restored_root, str(project.id))
        CanonicalBundleImporter(restored_events, restored_blobs).import_bundle(latest_target)
        restored_projection = SQLiteProjectionStore(restored_root, str(project.id))
        rebuild_filesystem_index(restored_events, restored_projection)
        if restored_projection.get_work_order(order.id) != cancelled:
            raise RuntimeError("Restored latest backup did not replay the changed work state")
        return dict(status="passed", imported=1, observations=2, snapshot_preserved=True,
                    cache_replayed=True, backup_verified=True, bundle_id=manifest.bundle_id,
                    stale_backup_refused=True, latest_backup_restored=True, latest_bundle_id=latest.bundle_id)
    finally:
        if dialog._worker is not None:
            dialog._worker.wait()
            app.processEvents()
        dialog.close()
        app.processEvents()
