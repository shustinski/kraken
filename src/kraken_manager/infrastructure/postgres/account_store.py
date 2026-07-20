"""PostgreSQL server-local accounts and revocable opaque sessions."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from kraken_manager.infrastructure.auth.local import LocalAccount, LocalSession, PasswordHasher

from .event_store import _sqlalchemy


def _account_tables() -> tuple[Any, Any, Any, Any]:
    sa, _ = _sqlalchemy()
    metadata = sa.MetaData()
    accounts = sa.Table(
        "server_accounts",
        metadata,
        sa.Column("account_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("username", sa.Text, nullable=False),
        sa.Column("username_key", sa.Text, nullable=False, unique=True),
        sa.Column("display_name", sa.Text, nullable=False),
        sa.Column("password_hash", sa.Text, nullable=False),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("failed_attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("blocked_until", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    sessions = sa.Table(
        "server_sessions",
        metadata,
        sa.Column("token_hash", sa.Text, primary_key=True),
        sa.Column(
            "account_id",
            sa.Uuid(as_uuid=False),
            sa.ForeignKey("server_accounts.account_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    sa.Index("ix_server_sessions_account", sessions.c.account_id)
    sa.Index("ix_server_sessions_expiry", sessions.c.expires_at)
    roles = sa.Table(
        "server_global_roles",
        metadata,
        sa.Column(
            "account_id",
            sa.Uuid(as_uuid=False),
            sa.ForeignKey("server_accounts.account_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("role", sa.Text, primary_key=True),
        sa.Column("granted_at", sa.DateTime(timezone=True), nullable=False),
    )
    return metadata, accounts, sessions, roles


class PostgresAccountStore:
    def __init__(self, engine: Any, hasher: PasswordHasher, *, create_schema_for_tests: bool = False) -> None:
        self.engine = engine
        self.hasher = hasher
        self.metadata, self.accounts, self.sessions, self.roles = _account_tables()
        if create_schema_for_tests:
            self.metadata.create_all(engine)

    @staticmethod
    def _normalize_username(username: str) -> tuple[str, str]:
        value = username.strip()
        if not value or len(value) > 128 or any(character.isspace() for character in value):
            raise ValueError("Invalid username")
        return value, value.casefold()

    @staticmethod
    def _account(row: Any) -> LocalAccount:
        created_at = row["created_at"]
        return LocalAccount(
            str(row["account_id"]),
            str(row["username"]),
            str(row["display_name"]),
            bool(row["enabled"]),
            created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
        )

    def create_account(self, username: str, display_name: str, password: str) -> LocalAccount:
        username, username_key = self._normalize_username(username)
        display_name = display_name.strip()
        if not display_name:
            raise ValueError("Display name is required")
        now = datetime.now(UTC)
        values = {
            "account_id": str(uuid4()),
            "username": username,
            "username_key": username_key,
            "display_name": display_name,
            "password_hash": self.hasher.hash(password),
            "enabled": True,
            "failed_attempts": 0,
            "blocked_until": None,
            "created_at": now,
        }
        with self.engine.begin() as connection:
            connection.execute(self.accounts.insert(), values)
        return LocalAccount(values["account_id"], username, display_name, True, now.isoformat())

    def account_count(self) -> int:
        sa, _ = _sqlalchemy()
        with self.engine.connect() as connection:
            return int(connection.execute(sa.select(sa.func.count()).select_from(self.accounts)).scalar_one())

    def get_account(self, account_id: str) -> LocalAccount | None:
        sa, _ = _sqlalchemy()
        with self.engine.connect() as connection:
            row = connection.execute(
                sa.select(self.accounts).where(self.accounts.c.account_id == account_id)
            ).mappings().first()
        return None if row is None else self._account(row)

    def reset_password(self, account_id: str, new_password: str) -> None:
        sa, _ = _sqlalchemy()
        password_hash = self.hasher.hash(new_password)
        with self.engine.begin() as connection:
            result = connection.execute(
                sa.update(self.accounts)
                .where(self.accounts.c.account_id == account_id)
                .values(password_hash=password_hash, failed_attempts=0, blocked_until=None)
            )
            if result.rowcount != 1:
                raise KeyError(account_id)
            connection.execute(sa.delete(self.sessions).where(self.sessions.c.account_id == account_id))

    def authenticate(
        self,
        username: str,
        password: str,
        *,
        lifetime: timedelta = timedelta(hours=12),
    ) -> LocalSession | None:
        _, username_key = self._normalize_username(username)
        now = datetime.now(UTC)
        sa, _ = _sqlalchemy()
        with self.engine.begin() as connection:
            row = connection.execute(
                sa.select(self.accounts)
                .where(self.accounts.c.username_key == username_key)
                .with_for_update()
            ).mappings().first()
            if row is None or not bool(row["enabled"]):
                return None
            blocked = row["blocked_until"]
            if blocked is not None and blocked > now:
                return None
            if not self.hasher.verify(str(row["password_hash"]), password):
                attempts = int(row["failed_attempts"]) + 1
                blocked_until = now + timedelta(minutes=min(60, 2 ** max(0, attempts - 5))) if attempts >= 5 else None
                connection.execute(
                    sa.update(self.accounts)
                    .where(self.accounts.c.account_id == row["account_id"])
                    .values(failed_attempts=attempts, blocked_until=blocked_until)
                )
                return None
            token = secrets.token_urlsafe(32)
            expires_at = now + lifetime
            connection.execute(
                sa.update(self.accounts)
                .where(self.accounts.c.account_id == row["account_id"])
                .values(failed_attempts=0, blocked_until=None)
            )
            connection.execute(
                self.sessions.insert(),
                {
                    "token_hash": hashlib.sha256(token.encode()).hexdigest(),
                    "account_id": str(row["account_id"]),
                    "expires_at": expires_at,
                    "created_at": now,
                },
            )
        return LocalSession(token, self._account(row), expires_at.isoformat())

    def resolve_session(self, token: str) -> LocalAccount | None:
        sa, _ = _sqlalchemy()
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        now = datetime.now(UTC)
        statement = (
            sa.select(self.accounts)
            .select_from(self.sessions.join(self.accounts, self.accounts.c.account_id == self.sessions.c.account_id))
            .where(
                self.sessions.c.token_hash == token_hash,
                self.sessions.c.expires_at > now,
                self.accounts.c.enabled.is_(True),
            )
        )
        with self.engine.connect() as connection:
            row = connection.execute(statement).mappings().first()
        return None if row is None else self._account(row)

    def revoke_session(self, token: str) -> None:
        sa, _ = _sqlalchemy()
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with self.engine.begin() as connection:
            connection.execute(sa.delete(self.sessions).where(self.sessions.c.token_hash == token_hash))

    def grant_global_role(self, account_id: str, role: str) -> None:
        if role != "server_admin":
            raise ValueError("Unsupported global role")
        _, pg_insert = _sqlalchemy()
        with self.engine.begin() as connection:
            account_exists = connection.execute(
                self.accounts.select().with_only_columns(self.accounts.c.account_id).where(
                    self.accounts.c.account_id == account_id
                )
            ).first()
            if account_exists is None:
                raise KeyError(account_id)
            connection.execute(
                pg_insert(self.roles)
                .values(account_id=account_id, role=role, granted_at=datetime.now(UTC))
                .on_conflict_do_nothing(index_elements=[self.roles.c.account_id, self.roles.c.role])
            )

    def accounts_with_global_role(self, role: str) -> tuple[str, ...]:
        sa, _ = _sqlalchemy()
        with self.engine.connect() as connection:
            values = connection.execute(
                sa.select(self.roles.c.account_id).where(self.roles.c.role == role).order_by(self.roles.c.account_id)
            ).scalars().all()
        return tuple(str(value) for value in values)


__all__ = ["PostgresAccountStore"]
