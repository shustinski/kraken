"""No-self-registration local accounts and revocable opaque sessions."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator, Protocol
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
        return LocalAccount(row["account_id"], row["username"], row["display_name"], bool(row["enabled"]), row["created_at"])

    def create_account(self, username: str, display_name: str, password: str) -> LocalAccount:
        username = self._normalize_username(username)
        display_name = display_name.strip()
        if not display_name:
            raise ValueError("Display name is required")
        account_id = str(uuid4())
        created_at = datetime.now(UTC).isoformat()
        password_hash = self.hasher.hash(password)
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO accounts(account_id, username, display_name, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (account_id, username, display_name, password_hash, created_at),
            )
        return LocalAccount(account_id, username, display_name, True, created_at)

    def account_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM accounts").fetchone()[0])

    def get_account(self, account_id: str) -> LocalAccount | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM accounts WHERE account_id=?", (account_id,)).fetchone()
        return None if row is None else self._account(row)

    def reset_password(self, account_id: str, new_password: str) -> None:
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
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def authenticate(self, username: str, password: str, *, lifetime: timedelta = timedelta(hours=12)) -> LocalSession | None:
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
                blocked_until = (now + timedelta(minutes=min(60, 2 ** max(0, attempts - 5)))).isoformat() if attempts >= 5 else None
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

    def grant_global_role(self, account_id: str, role: str) -> None:
        value = role.strip()
        if value not in {"server_admin"}:
            raise ValueError("Unsupported global role")
        with self._connect() as connection:
            try:
                connection.execute(
                    "INSERT INTO global_roles VALUES (?, ?, ?)",
                    (account_id, value, datetime.now(UTC).isoformat()),
                )
            except sqlite3.IntegrityError as exc:
                account = connection.execute("SELECT 1 FROM accounts WHERE account_id=?", (account_id,)).fetchone()
                if account is None:
                    raise KeyError(account_id) from exc

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
