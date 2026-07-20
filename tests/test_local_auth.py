from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kraken_manager.infrastructure.auth.local import LocalAccountStore, ScryptPasswordHasher


class LocalAccountTests(unittest.TestCase):
    def test_account_sessions_are_opaque_revocable_and_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "accounts.sqlite3"
            store = LocalAccountStore(database, ScryptPasswordHasher())
            account = store.create_account("operator", "Оператор", "correct horse battery staple")
            self.assertIsNone(store.authenticate("operator", "incorrect password"))
            session = store.authenticate("OPERATOR", "correct horse battery staple")
            self.assertIsNotNone(session)
            assert session is not None
            self.assertEqual(account.account_id, LocalAccountStore(database, ScryptPasswordHasher()).resolve_session(session.token).account_id)
            store.revoke_session(session.token)
            self.assertIsNone(store.resolve_session(session.token))

    def test_password_reset_revokes_existing_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalAccountStore(Path(temporary) / "accounts.sqlite3", ScryptPasswordHasher())
            account = store.create_account("admin", "Admin", "original password 123")
            session = store.authenticate("admin", "original password 123")
            assert session is not None
            store.reset_password(account.account_id, "replacement password 456")
            self.assertIsNone(store.resolve_session(session.token))
            self.assertIsNotNone(store.authenticate("admin", "replacement password 456"))


if __name__ == "__main__":
    unittest.main()

