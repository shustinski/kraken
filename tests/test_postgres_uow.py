from __future__ import annotations

import unittest
from unittest.mock import patch

from kraken_manager.infrastructure.postgres.unit_of_work import PostgresUnitOfWork


class _Transaction:
    def __init__(self) -> None:
        self.is_active = True
        self.committed = False
        self.rolled_back = False

    def commit(self) -> None:
        self.committed = True
        self.is_active = False

    def rollback(self) -> None:
        self.rolled_back = True
        self.is_active = False


class _Connection:
    def __init__(self) -> None:
        self.transaction = _Transaction()
        self.closed = False

    def begin(self):
        return self.transaction

    def close(self) -> None:
        self.closed = True


class _Engine:
    def __init__(self) -> None:
        self.connection = _Connection()

    def connect(self):
        return self.connection


class _Adapter:
    def __init__(self, engine, *, connection=None):
        self.engine = engine
        self.connection = connection


class PostgresUnitOfWorkTests(unittest.TestCase):
    def _patches(self):
        return (
            patch("kraken_manager.infrastructure.postgres.unit_of_work.PostgresEventStore", _Adapter),
            patch("kraken_manager.infrastructure.postgres.unit_of_work.PostgresProjectionStore", _Adapter),
            patch("kraken_manager.infrastructure.postgres.unit_of_work.PostgresIdentityAclStore", _Adapter),
        )

    def test_every_metadata_adapter_uses_one_committed_connection(self) -> None:
        engine = _Engine()
        first, second, third = self._patches()
        with first, second, third:
            with PostgresUnitOfWork(engine, object()) as uow:
                self.assertIs(engine.connection, uow.event_store.connection)
                self.assertIs(engine.connection, uow.projections.connection)
                self.assertIs(engine.connection, uow.identities.connection)
                self.assertIs(uow.identities, uow.acl)
                uow.commit()
        self.assertTrue(engine.connection.transaction.committed)
        self.assertFalse(engine.connection.transaction.rolled_back)
        self.assertTrue(engine.connection.closed)

    def test_uncommitted_work_is_rolled_back(self) -> None:
        engine = _Engine()
        first, second, third = self._patches()
        with first, second, third:
            with PostgresUnitOfWork(engine, object()):
                pass
        self.assertTrue(engine.connection.transaction.rolled_back)
        self.assertTrue(engine.connection.closed)


if __name__ == "__main__":
    unittest.main()
