"""Unit-of-work construction must not rebuild SQLAlchemy table definitions per command."""
from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from kraken_manager.infrastructure.postgres.event_store import PostgresEventStore
from kraken_manager.infrastructure.postgres.identity_store import PostgresIdentityAclStore, _identity_tables
from kraken_manager.infrastructure.postgres.projection_store import PostgresProjectionStore


def test_stores_share_table_definitions_between_units_of_work() -> None:
    engine = object()
    first, second = PostgresProjectionStore(engine), PostgresProjectionStore(engine)
    assert first.current is second.current
    assert first.catalog.tables["images"] is second.catalog.tables["images"]
    assert PostgresEventStore(engine).events is PostgresEventStore(engine).events
    assert PostgresIdentityAclStore(engine).acl is PostgresIdentityAclStore(engine).acl


def test_identity_table_factory_still_returns_fresh_metadata_for_extensions() -> None:
    # performer/session stores add their own tables to this metadata.
    assert _identity_tables()[0] is not _identity_tables()[0]
    assert "performers" not in PostgresIdentityAclStore(object()).metadata.tables
