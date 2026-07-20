"""Lease-based PostgreSQL queue using ``FOR UPDATE SKIP LOCKED``."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Mapping

from kraken_manager.domain.common import PluginJobId
from kraken_manager.domain.workflows import PluginJob

from .event_store import _sqlalchemy


class PostgresLeaseJobQueue:
    def __init__(
        self,
        engine: Any,
        *,
        encode: Callable[[PluginJob], Mapping[str, Any]],
        decode: Callable[[Mapping[str, Any]], PluginJob],
        create_schema_for_tests: bool = False,
    ) -> None:
        sa, _ = _sqlalchemy()
        self.engine = engine
        self.encode = encode
        self.decode = decode
        self.metadata = sa.MetaData()
        self.jobs = sa.Table(
            "background_jobs",
            self.metadata,
            sa.Column("job_id", sa.Uuid(as_uuid=False), primary_key=True),
            sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False, index=True),
            sa.Column("state", sa.Text, nullable=False, index=True),
            sa.Column("payload", sa.JSON, nullable=False),
            sa.Column("lease_owner", sa.Text),
            sa.Column("lease_until", sa.DateTime(timezone=True)),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        if create_schema_for_tests:
            self.metadata.create_all(engine)

    def enqueue(self, job: PluginJob) -> None:
        sa, pg_insert = _sqlalchemy()
        payload = dict(self.encode(job))
        with self.engine.begin() as connection:
            result = connection.execute(
                pg_insert(self.jobs)
                .values(
                    job_id=str(job.id),
                    project_id=str(job.project_id),
                    state=job.state.value,
                    payload=payload,
                    lease_owner=None,
                    lease_until=None,
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                )
                .on_conflict_do_nothing(index_elements=[self.jobs.c.job_id])
            )
            if result.rowcount != 1:
                raise ValueError(f"Background job {job.id} already exists")

    def get(self, job_id: PluginJobId) -> PluginJob | None:
        sa, _ = _sqlalchemy()
        with self.engine.connect() as connection:
            row = connection.execute(
                sa.select(self.jobs.c.payload).where(self.jobs.c.job_id == str(job_id))
            ).scalar_one_or_none()
        return None if row is None else self.decode(row)

    def lease_next(self, *, worker_id: str, lease_until: datetime) -> PluginJob | None:
        sa, _ = _sqlalchemy()
        now = datetime.now(lease_until.tzinfo)
        with self.engine.begin() as connection:
            row = connection.execute(
                sa.select(self.jobs)
                .where(
                    self.jobs.c.state == "queued",
                    sa.or_(self.jobs.c.lease_until.is_(None), self.jobs.c.lease_until < now),
                )
                .order_by(self.jobs.c.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            ).mappings().first()
            if row is None:
                return None
            connection.execute(
                sa.update(self.jobs)
                .where(self.jobs.c.job_id == row["job_id"])
                .values(lease_owner=worker_id, lease_until=lease_until)
            )
        return self.decode(row["payload"])

    def acknowledge(self, job_id: PluginJobId, *, worker_id: str) -> None:
        sa, _ = _sqlalchemy()
        with self.engine.begin() as connection:
            result = connection.execute(
                sa.delete(self.jobs)
                .where(self.jobs.c.job_id == str(job_id), self.jobs.c.lease_owner == worker_id)
            )
            if result.rowcount != 1:
                raise PermissionError("Job lease is not owned by this worker")


__all__ = ["PostgresLeaseJobQueue"]
