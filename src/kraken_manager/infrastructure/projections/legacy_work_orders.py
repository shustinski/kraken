"""Refuse cache replacement/export which would silently discard legacy work."""
from contextlib import closing
import sqlite3

from kraken_manager.application.legacy_work_orders import legacy_work_orders


class LegacyWorkOrderMigrationRequired(RuntimeError):
    """Keep the only remaining copy of unrecorded performer work."""


def assert_work_orders_replayable(events, projections=None):
    if projections is None:
        path = events.layout.index_path
        if not path.exists():
            return
        with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True)) as connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='typed_current_models'"
            ).fetchone()
        if table is None:
            return
        from kraken_manager.infrastructure.filesystem.projection_store import SQLiteProjectionStore

        projections = object.__new__(SQLiteProjectionStore)
        projections.path, projections.layout = path, events.layout
    legacy, _ = legacy_work_orders(projections, events, events.project_id)
    if legacy:
        raise LegacyWorkOrderMigrationRequired("Cannot rebuild/export: legacy work orders lack canonical events; old index preserved. "
                           "Review and apply a legacy work order migration first: "
                           + ", ".join(str(order.id) for order in legacy[:20]))
