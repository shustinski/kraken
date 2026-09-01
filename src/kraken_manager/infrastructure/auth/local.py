"""No-self-registration local accounts and revocable opaque sessions."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from uuid import uuid4


class PasswordHasher(Protocol):
    def hash(self, password: str) -> str: ...

    def verify(self, encoded: str, password: str) -> bool: ...


class ScryptPasswordHasher:
    """Dependency-free offline desktop hasher; server profiles use Argon2id."""

    algorithm = "scrypt"

    def hash(self, password: str) -> str:
        salt = os.urandom(16)
        derived = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
        return "$".join(
            (
                self.algorithm,
                "n=16384,r=8,p=1",
                base64.urlsafe_b64encode(salt).decode().rstrip("="),
                base64.urlsafe_b64encode(derived).decode().rstrip("="),
            )
        )

    def verify(self, encoded: str, password: str) -> bool:
        try:
            algorithm, parameters, salt_text, digest_text = encoded.split("$", 3)
            if algorithm != self.algorithm or parameters != "n=16384,r=8,p=1":
                return False
            padding = "=" * (-len(salt_text) % 4)
            salt = base64.urlsafe_b64decode(salt_text + padding)
            padding = "=" * (-len(digest_text) % 4)
            expected = base64.urlsafe_b64decode(digest_text + padding)
            actual = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=len(expected))
            return hmac.compare_digest(actual, expected)
        except (ValueError, TypeError):
            return False


class Argon2PasswordHasher:
    """Argon2id adapter loaded only by the server extra."""

    def __init__(self) -> None:
        try:
            from argon2 import PasswordHasher as Backend
            from argon2.low_level import Type
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install Kraken with the 'server' extra to use Argon2id") from exc
        self._backend = Backend(type=Type.ID)

    def hash(self, password: str) -> str:
        return str(self._backend.hash(password))

    def verify(self, encoded: str, password: str) -> bool:
        try:
            return bool(self._backend.verify(encoded, password))
        except Exception:  # argon2 uses multiple verification exception classes
            return False


@dataclass(frozen=True, slots=True)
class LocalAccount:
    account_id: str
    username: str
    display_name: str
    enabled: bool
    created_at: str


@dataclass(frozen=True, slots=True)
class LocalSession:
    token: str
    account: LocalAccount
    expires_at: str


class LocalAccountStore:
    """SQLite account store for a workstation or Kraken Server instance."""

    def __init__(self, database: Path | str, hasher: PasswordHasher) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.hasher = hasher
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    blocked_until TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL REFERENCES accounts(account_id) ON DELETE CASCADE,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_sessions_account ON sessions(account_id);
                CREATE TABLE IF NOT EXISTS global_roles (
                    account_id TEXT NOT NULL REFERENCES accounts(account_id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    granted_at TEXT NOT NULL,
                    PRIMARY KEY(account_id, role)
                );
                CREATE TABLE IF NOT EXISTS administration_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    actor_id TEXT,
                    action TEXT NOT NULL,
                    target_account_id TEXT,
                    details_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _normalize_username(username: str) -> str:
        value = username.strip()
        if not value or len(value) > 128 or any(char.isspace() for char in value):
            raise ValueError("Invalid username")
        return value

    @staticmethod
    def _account(row: sqlite3.Row) -> LocalAccount:
        return LocalAccount(
            row["account_id"], row["username"], row["display_name"], bool(row["enabled"]), row["created_at"]
        )

    def create_account(
        self,
        username: str,
        display_name: str,
        password: str,
        *,
        actor_id: str | None = None,
    ) -> LocalAccount:
        username = self._normalize_username(username)
        display_name = display_name.strip()
        if not display_name:
            raise ValueError("Display name is required")
        account_id = str(uuid4())
        created_at = datetime.now(UTC).isoformat()
        password_hash = self.hasher.hash(password)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO accounts(account_id, username, display_name, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                    (account_id, username, display_name, password_hash, created_at),
                )
                if actor_id is not None:
                    self._audit(connection, actor_id, "account.created", account_id)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return LocalAccount(account_id, username, display_name, True, created_at)

    def account_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM accounts").fetchone()[0])

    def get_account(self, account_id: str) -> LocalAccount | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM accounts WHERE account_id=?", (account_id,)).fetchone()
        return None if row is None else self._account(row)

    def get_by_username(self, username: str) -> LocalAccount | None:
        normalized = self._normalize_username(username)
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM accounts WHERE username=? COLLATE NOCASE", (normalized,)).fetchone()
        return None if row is None else self._account(row)

    def list_accounts(self, *, include_disabled: bool = True) -> tuple[LocalAccount, ...]:
        statement = "SELECT * FROM accounts"
        if not include_disabled:
            statement += " WHERE enabled=1"
        statement += " ORDER BY display_name COLLATE NOCASE, username COLLATE NOCASE"
        with self._connect() as connection:
            rows = connection.execute(statement).fetchall()
        return tuple(self._account(row) for row in rows)

    def reset_password(self, account_id: str, new_password: str, *, actor_id: str | None = None) -> None:
        password_hash = self.hasher.hash(new_password)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    "UPDATE accounts SET password_hash=?, failed_attempts=0, blocked_until=NULL WHERE account_id=?",
                    (password_hash, account_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(account_id)
                connection.execute("DELETE FROM sessions WHERE account_id=?", (account_id,))
                if actor_id is not None:
                    self._audit(connection, actor_id, "account.password_reset", account_id)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def authenticate(
        self, username: str, password: str, *, lifetime: timedelta = timedelta(hours=12)
    ) -> LocalSession | None:
        now = datetime.now(UTC)
        normalized = self._normalize_username(username)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM accounts WHERE username=? COLLATE NOCASE", (normalized,)).fetchone()
            if row is None or not bool(row["enabled"]):
                connection.rollback()
                return None
            blocked = datetime.fromisoformat(row["blocked_until"]) if row["blocked_until"] else None
            if blocked is not None and blocked > now:
                connection.rollback()
                return None
            if not self.hasher.verify(row["password_hash"], password):
                attempts = int(row["failed_attempts"]) + 1
                blocked_until = (
                    (now + timedelta(minutes=min(60, 2 ** max(0, attempts - 5)))).isoformat() if attempts >= 5 else None
                )
                connection.execute(
                    "UPDATE accounts SET failed_attempts=?, blocked_until=? WHERE account_id=?",
                    (attempts, blocked_until, row["account_id"]),
                )
                connection.commit()
                return None
            token = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            expires = now + lifetime
            connection.execute(
                "UPDATE accounts SET failed_attempts=0, blocked_until=NULL WHERE account_id=?",
                (row["account_id"],),
            )
            connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?)",
                (token_hash, row["account_id"], expires.isoformat(), now.isoformat()),
            )
            connection.commit()
        return LocalSession(token, self._account(row), expires.isoformat())

    def resolve_session(self, token: str) -> LocalAccount | None:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT a.* FROM sessions s JOIN accounts a ON a.account_id=s.account_id "
                "WHERE s.token_hash=? AND s.expires_at>? AND a.enabled=1",
                (token_hash, now),
            ).fetchone()
        return None if row is None else self._account(row)

    def revoke_session(self, token: str) -> None:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))

    def revoke_all_sessions(self, account_id: str, *, actor_id: str | None = None) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if connection.execute("SELECT 1 FROM accounts WHERE account_id=?", (account_id,)).fetchone() is None:
                    raise KeyError(account_id)
                connection.execute("DELETE FROM sessions WHERE account_id=?", (account_id,))
                self._audit(connection, actor_id, "sessions.revoked", account_id)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def grant_global_role(self, account_id: str, role: str, *, actor_id: str | None = None) -> None:
        value = role.strip()
        if value not in {"server_admin"}:
            raise ValueError("Unsupported global role")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT OR IGNORE INTO global_roles VALUES (?, ?, ?)",
                    (account_id, value, datetime.now(UTC).isoformat()),
                )
                if connection.execute("SELECT 1 FROM accounts WHERE account_id=?", (account_id,)).fetchone() is None:
                    raise KeyError(account_id)
                self._audit(connection, actor_id, "global_role.granted", account_id, {"role": value})
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise KeyError(account_id) from exc
            except BaseException:
                connection.rollback()
                raise

    def revoke_global_role(
        self,
        account_id: str,
        role: str,
        *,
        actor_id: str | None = None,
        preserve_last_enabled: bool = False,
    ) -> None:
        value = role.strip()
        if value != "server_admin":
            raise ValueError("Unsupported global role")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if connection.execute("SELECT 1 FROM accounts WHERE account_id=?", (account_id,)).fetchone() is None:
                    raise KeyError(account_id)
                if preserve_last_enabled:
                    target_is_enabled_admin = connection.execute(
                        "SELECT 1 FROM accounts a JOIN global_roles r "
                        "ON r.account_id=a.account_id "
                        "WHERE a.account_id=? AND a.enabled=1 AND r.role=?",
                        (account_id, value),
                    ).fetchone()
                    enabled_admin_count = connection.execute(
                        "SELECT COUNT(*) FROM accounts a JOIN global_roles r "
                        "ON r.account_id=a.account_id WHERE a.enabled=1 AND r.role=?",
                        (value,),
                    ).fetchone()[0]
                    if target_is_enabled_admin is not None and enabled_admin_count <= 1:
                        raise ValueError("The last active Server Administrator cannot be removed")
                connection.execute(
                    "DELETE FROM global_roles WHERE account_id=? AND role=?",
                    (account_id, value),
                )
                self._audit(connection, actor_id, "global_role.revoked", account_id, {"role": value})
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def global_roles_for(self, account_id: str) -> frozenset[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT role FROM global_roles WHERE account_id=?", (account_id,)).fetchall()
        return frozenset(str(row["role"]) for row in rows)

    def set_enabled(
        self,
        account_id: str,
        enabled: bool,
        *,
        actor_id: str | None = None,
        preserve_last_admin: bool = False,
    ) -> LocalAccount:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if not enabled and preserve_last_admin:
                    target_is_admin = connection.execute(
                        "SELECT 1 FROM global_roles WHERE account_id=? AND role='server_admin'",
                        (account_id,),
                    ).fetchone()
                    enabled_admin_count = connection.execute(
                        "SELECT COUNT(*) FROM accounts a JOIN global_roles r "
                        "ON r.account_id=a.account_id "
                        "WHERE a.enabled=1 AND r.role='server_admin'"
                    ).fetchone()[0]
                    if target_is_admin is not None and enabled_admin_count <= 1:
                        raise ValueError("The last active Server Administrator cannot be disabled")
                cursor = connection.execute(
                    "UPDATE accounts SET enabled=? WHERE account_id=?",
                    (int(bool(enabled)), account_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(account_id)
                if not enabled:
                    connection.execute("DELETE FROM sessions WHERE account_id=?", (account_id,))
                self._audit(
                    connection,
                    actor_id,
                    "account.enabled" if enabled else "account.disabled",
                    account_id,
                )
                row = connection.execute("SELECT * FROM accounts WHERE account_id=?", (account_id,)).fetchone()
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        assert row is not None
        return self._account(row)

    @staticmethod
    def _audit(
        connection: sqlite3.Connection,
        actor_id: str | None,
        action: str,
        target_account_id: str | None,
        details: dict[str, object] | None = None,
    ) -> None:
        import json

        connection.execute(
            "INSERT INTO administration_audit(actor_id, action, target_account_id, details_json, recorded_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                actor_id,
                action,
                target_account_id,
                json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
                datetime.now(UTC).isoformat(),
            ),
        )

    def administration_audit(self, *, limit: int = 500) -> tuple[dict[str, object], ...]:
        import json

        safe_limit = max(1, min(int(limit), 2_000))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM administration_audit ORDER BY audit_id DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return tuple(
            {
                "audit_id": int(row["audit_id"]),
                "actor_id": row["actor_id"],
                "action": str(row["action"]),
                "target_account_id": row["target_account_id"],
                "details": json.loads(str(row["details_json"])),
                "recorded_at": str(row["recorded_at"]),
            }
            for row in rows
        )

    def accounts_with_global_role(self, role: str) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT account_id FROM global_roles WHERE role=? ORDER BY account_id", (role,)
            ).fetchall()
        return tuple(str(row["account_id"]) for row in rows)


__all__ = [
    "Argon2PasswordHasher",
    "LocalAccount",
    "LocalAccountStore",
    "LocalSession",
    "PasswordHasher",
    "ScryptPasswordHasher",
]
