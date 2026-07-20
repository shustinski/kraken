from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from kraken_manager.application import PerformerStore, ensure_gitlab_performer
from kraken_manager.application.errors import ConflictError
from kraken_manager.domain.common import PrincipalId
from kraken_manager.domain.identity import Performer, Principal
from kraken_manager.infrastructure.auth import LocalIdentityAclStore, LocalSQLitePerformerStore
from storage_contract_suite import (
    LINKED_PERFORMER_ID,
    PERFORMER_ID,
    PRINCIPAL_ID,
    PerformerStoreContract,
)


def _gitlab_principal(*, display_name: str = "GitLab Worker") -> Principal:
    return Principal.gitlab(
        principal_id=PRINCIPAL_ID,
        issuer="https://gitlab.example",
        subject="42",
        display_name=display_name,
        email="worker@example.test",
    )


class LocalSQLitePerformerStoreTests(unittest.TestCase, PerformerStoreContract):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        database = Path(self.temporary.name) / "identity.sqlite3"
        identities = LocalIdentityAclStore(database)
        identities.save(_gitlab_principal())
        self.performer_store = LocalSQLitePerformerStore(database)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_semantic_contract(self) -> None:
        self.assertIsInstance(self.performer_store, PerformerStore)
        self.assert_performer_store_contract()

    def test_link_to_unknown_principal_is_rejected(self) -> None:
        with self.assertRaises(ConflictError):
            self.performer_store.create(
                Performer.create(
                    name="Unknown",
                    color="#123456",
                    principal_id=PrincipalId("20000000-0000-0000-0000-000000000099"),
                )
            )

    def test_gitlab_first_login_helper_is_idempotent_and_updates_name(self) -> None:
        first = ensure_gitlab_performer(_gitlab_principal(), self.performer_store)
        again = ensure_gitlab_performer(_gitlab_principal(), self.performer_store)
        self.assertEqual(first, again)

        renamed = ensure_gitlab_performer(
            _gitlab_principal(display_name="New GitLab Name"), self.performer_store
        )
        self.assertEqual(first.id, renamed.id)
        self.assertEqual(first.color, renamed.color)
        self.assertEqual("New GitLab Name", renamed.name)

        archived = self.performer_store.archive(first.id)
        after_login = ensure_gitlab_performer(
            _gitlab_principal(display_name="Newest Name"), self.performer_store
        )
        self.assertFalse(archived.active)
        self.assertFalse(after_login.active)

    def test_link_cannot_be_rebound_or_removed(self) -> None:
        linked = ensure_gitlab_performer(_gitlab_principal(), self.performer_store)
        with self.assertRaises(ConflictError):
            self.performer_store.update(replace(linked, principal_id=None))

    def test_local_principal_is_not_automatically_a_performer(self) -> None:
        principal = Principal.local(subject="local", display_name="Local User")
        with self.assertRaises(ValueError):
            ensure_gitlab_performer(principal, self.performer_store)


@unittest.skipUnless(
    os.environ.get("KRAKEN_TEST_POSTGRES_URL"),
    "KRAKEN_TEST_POSTGRES_URL is not configured",
)
class PostgresPerformerStoreContractTests(unittest.TestCase, PerformerStoreContract):
    """Run the same public contract against an explicitly configured test DB."""

    def setUp(self) -> None:
        from sqlalchemy import create_engine, delete

        from kraken_manager.infrastructure.postgres import (
            PostgresIdentityAclStore,
            PostgresPerformerStore,
        )

        self._delete = delete
        self.engine = create_engine(os.environ["KRAKEN_TEST_POSTGRES_URL"])
        identities = PostgresIdentityAclStore(self.engine, create_schema_for_tests=True)
        self.performer_store = PostgresPerformerStore(self.engine, create_schema_for_tests=True)
        with self.engine.begin() as connection:
            connection.execute(
                delete(self.performer_store.performers).where(
                    self.performer_store.performers.c.performer_id.in_(
                        (PERFORMER_ID, LINKED_PERFORMER_ID)
                    )
                )
            )
            principals = self.performer_store.metadata.tables["principals"]
            connection.execute(delete(principals).where(principals.c.principal_id == PRINCIPAL_ID))
        identities.save(_gitlab_principal())

    def tearDown(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                self._delete(self.performer_store.performers).where(
                    self.performer_store.performers.c.performer_id.in_(
                        (PERFORMER_ID, LINKED_PERFORMER_ID)
                    )
                )
            )
            identities = self.performer_store.metadata.tables["principals"]
            connection.execute(
                self._delete(identities).where(identities.c.principal_id == PRINCIPAL_ID)
            )
        self.engine.dispose()

    def test_semantic_contract(self) -> None:
        self.assertIsInstance(self.performer_store, PerformerStore)
        self.assert_performer_store_contract()


if __name__ == "__main__":
    unittest.main()
