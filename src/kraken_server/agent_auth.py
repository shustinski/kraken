"""Revocable machine credentials for server-connected Kraken agents."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    token_id: str
    name: str
    capabilities: frozenset[str]


class PostgresAgentTokenStore:
    def __init__(self, engine: Any) -> None:
        import sqlalchemy as sa

        self.engine = engine
        self.metadata = sa.MetaData()
        self.tokens = sa.Table(
            "server_agent_tokens",
            self.metadata,
            sa.Column("token_id", sa.Uuid(as_uuid=False), primary_key=True),
            sa.Column("name", sa.Text, nullable=False),
            sa.Column("token_hash", sa.Text, nullable=False),
            sa.Column("capabilities", sa.JSON, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("revoked_at", sa.DateTime(timezone=True)),
        )

    @staticmethod
    def _digest(secret: str) -> str:
        return hashlib.sha256(secret.encode("utf-8")).hexdigest()

    def create(self, name: str, capabilities: frozenset[str]) -> tuple[AgentIdentity, str]:
        clean_name = str(name).strip()
        clean_capabilities = frozenset(str(item).strip() for item in capabilities if str(item).strip())
        if not clean_name or not clean_capabilities:
            raise ValueError("Agent name and at least one capability are required")
        token_id = str(uuid4())
        secret = secrets.token_urlsafe(32)
        token = f"kat_{token_id}_{secret}"
        import sqlalchemy as sa

        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(self.tokens).values(
                    token_id=token_id,
                    name=clean_name,
                    token_hash=self._digest(secret),
                    capabilities=sorted(clean_capabilities),
                    created_at=datetime.now(UTC),
                    revoked_at=None,
                )
            )
        return AgentIdentity(token_id, clean_name, clean_capabilities), token

    def resolve(self, token: str) -> AgentIdentity | None:
        if not token.startswith("kat_"):
            return None
        try:
            token_id, secret = token.removeprefix("kat_").split("_", 1)
        except ValueError:
            return None
        import sqlalchemy as sa

        with self.engine.connect() as connection:
            row = connection.execute(
                sa.select(self.tokens).where(
                    self.tokens.c.token_id == token_id,
                    self.tokens.c.revoked_at.is_(None),
                )
            ).mappings().one_or_none()
        if row is None or not secrets.compare_digest(str(row["token_hash"]), self._digest(secret)):
            return None
        return AgentIdentity(
            str(row["token_id"]),
            str(row["name"]),
            frozenset(str(item) for item in row["capabilities"]),
        )

    def revoke(self, token_id: str) -> bool:
        import sqlalchemy as sa

        with self.engine.begin() as connection:
            result = connection.execute(
                sa.update(self.tokens)
                .where(self.tokens.c.token_id == token_id, self.tokens.c.revoked_at.is_(None))
                .values(revoked_at=datetime.now(UTC))
            )
        return result.rowcount == 1

    def has_capability(self, capability: str) -> bool:
        import sqlalchemy as sa

        with self.engine.connect() as connection:
            rows = connection.execute(
                sa.select(self.tokens.c.capabilities).where(self.tokens.c.revoked_at.is_(None))
            ).scalars().all()
        return any(capability in {str(item) for item in values} for values in rows)


__all__ = ["AgentIdentity", "PostgresAgentTokenStore"]
