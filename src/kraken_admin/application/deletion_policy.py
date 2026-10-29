"""Shared administrator choice to skip interactive confirmation."""

from __future__ import annotations

from kraken_admin.infrastructure.deletion_settings import (
    read_auto_confirm,
    settings_table,
    write_auto_confirm,
)


class DeletionPolicy:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.table = settings_table()

    def enabled(self, connection=None) -> bool:
        if connection is None:
            with self.engine.connect() as owned:
                return read_auto_confirm(owned, self.table)
        return read_auto_confirm(connection, self.table)

    def set_enabled(self, enabled: bool, accounts=None) -> None:
        with self.engine.begin() as connection:
            write_auto_confirm(connection, self.table, enabled)
            if accounts is not None:
                action = "deletion.auto_confirm_enabled" if enabled else "deletion.auto_confirm_disabled"
                accounts._record_audit(connection, None, action, None, {"enabled": enabled})
