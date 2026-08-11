"""PostgreSQL-backed gateway and lease protocol for remote Kraken agents."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from kraken_manager.domain.workflows import PluginJobManifestV1
from kraken_manager.domain.artifacts import BlobRef
from kraken_manager.infrastructure.filesystem._codec import decode_model, encode_model

from .agent_auth import AgentIdentity, PostgresAgentTokenStore


class PostgresAgentGateway:
    def __init__(self, engine: Any, tokens: PostgresAgentTokenStore, blobs: Any) -> None:
        import sqlalchemy as sa

        self.engine = engine
        self.tokens = tokens
        self.blobs = blobs
        self.metadata = sa.MetaData()
        self.jobs = sa.Table(
            "background_jobs",
            self.metadata,
            sa.Column("job_id", sa.Uuid(as_uuid=False), primary_key=True),
            sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
            sa.Column("state", sa.Text, nullable=False),
            sa.Column("payload", sa.JSON, nullable=False),
            sa.Column("lease_owner", sa.Text),
            sa.Column("lease_until", sa.DateTime(timezone=True)),
            sa.Column("lease_attempts", sa.Integer, nullable=False),
            sa.Column("last_error", sa.Text),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        self.publications = sa.Table(
            "agent_job_publications",
            self.metadata,
            sa.Column("publication_id", sa.Text, primary_key=True),
            sa.Column("job_id", sa.Uuid(as_uuid=False), primary_key=True),
            sa.Column("payload", sa.JSON, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        self.outputs = sa.Table(
            "agent_job_outputs",
            self.metadata,
            sa.Column("job_id", sa.Uuid(as_uuid=False), primary_key=True),
            sa.Column("output_id", sa.Text, primary_key=True),
            sa.Column("sha256", sa.Text, nullable=False),
            sa.Column("size_bytes", sa.BigInteger, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )

    def is_available(self, capability: str, protocol_version: str) -> bool:
        return protocol_version == "1.0" and self.tokens.has_capability(capability)

    def submit(self, manifest: PluginJobManifestV1) -> None:
        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        now = datetime.now(UTC)
        with self.engine.begin() as connection:
            connection.execute(
                pg_insert(self.jobs)
                .values(
                    job_id=str(manifest.job_id),
                    project_id=str(manifest.project_id),
                    state="queued",
                    payload=encode_model(manifest),
                    lease_owner=None,
                    lease_until=None,
                    lease_attempts=0,
                    last_error=None,
                    created_at=now,
                    updated_at=now,
                )
                .on_conflict_do_nothing(index_elements=[self.jobs.c.job_id])
            )

    def cancel(self, job_id: object) -> None:
        import sqlalchemy as sa

        with self.engine.begin() as connection:
            result = connection.execute(
                sa.update(self.jobs)
                .where(
                    self.jobs.c.job_id == str(job_id),
                    self.jobs.c.state.in_(("queued", "leased")),
                )
                .values(
                    state="cancelled",
                    lease_owner=None,
                    lease_until=None,
                    updated_at=datetime.now(UTC),
                )
            )
        if result.rowcount != 1:
            raise RuntimeError("Agent job is not cancellable")

    def lease(self, agent: AgentIdentity, *, seconds: int = 60) -> dict[str, Any] | None:
        import sqlalchemy as sa

        now = datetime.now(UTC)
        lease_until = now + timedelta(seconds=max(15, min(int(seconds), 300)))
        with self.engine.begin() as connection:
            rows = connection.execute(
                sa.select(self.jobs)
                .where(
                    self.jobs.c.state == "queued",
                    sa.or_(self.jobs.c.lease_until.is_(None), self.jobs.c.lease_until < now),
                )
                .order_by(self.jobs.c.created_at)
                .with_for_update(skip_locked=True)
                .limit(100)
            ).mappings().all()
            row = next(
                (
                    item
                    for item in rows
                    if str(item["payload"].get("capability", "")) in agent.capabilities
                ),
                None,
            )
            if row is None:
                return None
            connection.execute(
                sa.update(self.jobs)
                .where(self.jobs.c.job_id == row["job_id"])
                .values(
                    lease_owner=agent.token_id,
                    lease_until=lease_until,
                    lease_attempts=self.jobs.c.lease_attempts + 1,
                    updated_at=now,
                )
            )
        return {
            "job_id": str(row["job_id"]),
            "lease_until": lease_until.isoformat(),
            "manifest": dict(row["payload"]),
        }

    def heartbeat(self, job_id: str, agent: AgentIdentity, *, seconds: int = 60) -> str:
        import sqlalchemy as sa

        now = datetime.now(UTC)
        lease_until = now + timedelta(seconds=max(15, min(int(seconds), 300)))
        with self.engine.begin() as connection:
            result = connection.execute(
                sa.update(self.jobs)
                .where(
                    self.jobs.c.job_id == job_id,
                    self.jobs.c.lease_owner == agent.token_id,
                    self.jobs.c.state == "queued",
                )
                .values(lease_until=lease_until, updated_at=now)
            )
        if result.rowcount != 1:
            raise PermissionError("Agent does not own an active lease for this job")
        return lease_until.isoformat()

    def manifest(self, job_id: str, agent: AgentIdentity) -> PluginJobManifestV1:
        import sqlalchemy as sa

        with self.engine.connect() as connection:
            payload = connection.execute(
                sa.select(self.jobs.c.payload).where(
                    self.jobs.c.job_id == job_id,
                    self.jobs.c.lease_owner == agent.token_id,
                    self.jobs.c.lease_until > datetime.now(UTC),
                )
            ).scalar_one_or_none()
        if payload is None:
            raise PermissionError("Agent does not own an active lease for this job")
        return decode_model(PluginJobManifestV1, payload)

    def publish(self, job_id: str, agent: AgentIdentity, payload: Mapping[str, Any]) -> bool:
        manifest = self.manifest(job_id, agent)
        del manifest
        canonical = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)
        publication_id = str(payload.get("publication_id") or hashlib.sha256(canonical.encode()).hexdigest())
        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        with self.engine.begin() as connection:
            result = connection.execute(
                pg_insert(self.publications)
                .values(
                    publication_id=publication_id,
                    job_id=job_id,
                    payload=dict(payload),
                    created_at=datetime.now(UTC),
                )
                .on_conflict_do_nothing(
                    index_elements=[
                        self.publications.c.job_id,
                        self.publications.c.publication_id,
                    ]
                )
            )
            stored_payload = connection.execute(
                sa.select(self.publications.c.payload).where(
                    self.publications.c.job_id == job_id,
                    self.publications.c.publication_id == publication_id,
                )
            ).scalar_one()
            if json.dumps(
                stored_payload, sort_keys=True, separators=(",", ":"), default=str
            ) != canonical:
                raise RuntimeError(
                    "A publication ID was already used with a different payload"
                )
        return result.rowcount == 1

    def upload_output(
        self,
        job_id: str,
        output_id: str,
        agent: AgentIdentity,
        chunks: Any,
        *,
        expected_sha256: str,
    ) -> dict[str, Any]:
        self.manifest(job_id, agent)
        stored = self.blobs.put(chunks, expected_sha256=expected_sha256)
        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        with self.engine.begin() as connection:
            existing = connection.execute(
                sa.select(self.outputs.c.sha256, self.outputs.c.size_bytes).where(
                    self.outputs.c.job_id == job_id,
                    self.outputs.c.output_id == output_id,
                )
            ).mappings().one_or_none()
            if existing is not None:
                if (
                    str(existing["sha256"]) != stored.blob.sha256
                    or int(existing["size_bytes"]) != stored.blob.size_bytes
                ):
                    raise RuntimeError(
                        "An output ID was already published with different content"
                    )
                return {
                    "sha256": str(existing["sha256"]),
                    "size_bytes": int(existing["size_bytes"]),
                }
            connection.execute(
                pg_insert(self.outputs)
                .values(
                    job_id=job_id,
                    output_id=output_id,
                    sha256=stored.blob.sha256,
                    size_bytes=stored.blob.size_bytes,
                    created_at=datetime.now(UTC),
                )
                .on_conflict_do_nothing(
                    index_elements=[self.outputs.c.job_id, self.outputs.c.output_id]
                )
            )
            persisted = connection.execute(
                sa.select(self.outputs.c.sha256, self.outputs.c.size_bytes).where(
                    self.outputs.c.job_id == job_id,
                    self.outputs.c.output_id == output_id,
                )
            ).mappings().one()
            if (
                str(persisted["sha256"]) != stored.blob.sha256
                or int(persisted["size_bytes"]) != stored.blob.size_bytes
            ):
                raise RuntimeError(
                    "An output ID was already published with different content"
                )
        return {"sha256": stored.blob.sha256, "size_bytes": stored.blob.size_bytes}

    def finish(self, job_id: str, agent: AgentIdentity, *, failed: bool, error: str = "") -> None:
        import sqlalchemy as sa

        self.manifest(job_id, agent)
        now = datetime.now(UTC)
        with self.engine.begin() as connection:
            result = connection.execute(
                sa.update(self.jobs)
                .where(self.jobs.c.job_id == job_id, self.jobs.c.lease_owner == agent.token_id)
                .values(
                    state="failed" if failed else "completed",
                    lease_owner=None,
                    lease_until=None,
                    last_error=error or None,
                    updated_at=now,
                )
            )
        if result.rowcount != 1:
            raise PermissionError("Agent does not own this job")

    def publications_for(self, job_id: str, agent: AgentIdentity) -> tuple[dict[str, Any], ...]:
        self.manifest(job_id, agent)
        import sqlalchemy as sa

        with self.engine.connect() as connection:
            rows = connection.execute(
                sa.select(self.publications.c.payload)
                .where(self.publications.c.job_id == job_id)
                .order_by(self.publications.c.created_at)
            ).scalars().all()
        return tuple(dict(item) for item in rows)

    def output_blob(self, job_id: str, output_id: str) -> BlobRef:
        import sqlalchemy as sa

        with self.engine.connect() as connection:
            row = connection.execute(
                sa.select(self.outputs.c.sha256, self.outputs.c.size_bytes).where(
                    self.outputs.c.job_id == job_id,
                    self.outputs.c.output_id == output_id,
                )
            ).mappings().one_or_none()
        if row is None:
            raise FileNotFoundError(f"Agent output {output_id} was not uploaded")
        return BlobRef(str(row["sha256"]), int(row["size_bytes"]))


__all__ = ["PostgresAgentGateway"]
