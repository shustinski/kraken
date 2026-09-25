"""Durable deletion requests and project operation barriers."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from uuid import uuid4

import sqlalchemy as sa

from .services import ConflictError


def request_table(metadata: sa.MetaData | None = None) -> sa.Table:
    return sa.Table(
        "project_deletion_requests", metadata if metadata is not None else sa.MetaData(),
        sa.Column("request_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("project_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("project_name", sa.Text, nullable=False),
        sa.Column("requested_by", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("state", sa.Text, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("plan", sa.JSON, nullable=False),
        sa.Column("completed", sa.JSON, nullable=False),
        sa.Column("error", sa.Text, nullable=False),
    )


def lock(connection, key: str, *, shared: bool = False, attempt: bool = False) -> bool:
    if connection.dialect.name != "postgresql":
        return True
    function = "pg_" + ("try_" if attempt else "") + "advisory_xact_lock" + ("_shared" if shared else "")
    result = connection.execute(sa.text(f"SELECT {function}(hashtextextended(:key, 0))"), {"key": key}).scalar()
    return bool(result) if attempt else True


def ensure_available(connection, project_id: str) -> None:
    table = request_table()
    blocked = connection.execute(sa.select(table.c.request_id).where(
        table.c.project_id == project_id, table.c.state.in_(("deleting", "failed", "deleted")),
    )).first()
    if blocked:
        raise ConflictError("Проект удаляется или уже удалён администратором")


@contextmanager
def operation(engine, project_id: str | None):
    with engine.begin() as connection:
        lock(connection, "kraken:storage", shared=True)
        if project_id is not None:
            lock(connection, f"project:{project_id}", shared=True)
            ensure_available(connection, project_id)
        yield


class DeletionRequests:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.table = request_table()

    def record_transfer(self, digest: str, context) -> None:
        from sqlalchemy.dialects.postgresql import insert

        if not context or not context.get("project"):
            raise ValueError("Project context is required for a transfer")
        table = sa.table("project_transfer_blobs", sa.column("project_id", sa.Uuid(as_uuid=False)),
                         sa.column("sha256", sa.Text))
        with self.engine.begin() as connection:
            connection.execute(insert(table).values(project_id=str(context["project"]), sha256=digest)
                               .on_conflict_do_nothing())

    def list(self, project_id: str | None = None) -> list[dict]:
        query = sa.select(self.table).order_by(self.table.c.requested_at.desc())
        if project_id is not None:
            query = query.where(self.table.c.project_id == project_id)
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(query).mappings()]

    def request(self, project_id: str, project_name: str, actor_id: str) -> dict:
        with self.engine.begin() as connection:
            lock(connection, f"deletion-request:{project_id}")
            ensure_available(connection, project_id)
            existing = connection.execute(sa.select(self.table).where(
                self.table.c.project_id == project_id, self.table.c.state == "pending",
            )).mappings().first()
            if existing:
                return dict(existing)
            now = datetime.now(UTC)
            row = dict(request_id=str(uuid4()), project_id=project_id, project_name=project_name,
                       requested_by=actor_id, requested_at=now, updated_at=now, state="pending",
                       plan={}, completed=[], error="")
            connection.execute(self.table.insert().values(**row))
            return row

    def reject(self, request_id: str) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(self.table.update().where(
                self.table.c.request_id == request_id, self.table.c.state == "pending",
            ).values(state="rejected", updated_at=datetime.now(UTC)))
            if result.rowcount != 1:
                raise ConflictError("Заявка уже обработана")


class ProjectOperationMiddleware:
    """Hold the barrier until the entire response/stream has completed."""

    def __init__(self, app, engine) -> None:
        self.app = app
        # Barriers live for the duration of streamed responses. They must not
        # consume the application pool needed by the operation they protect.
        self.engine = sa.create_engine(engine.url, poolclass=sa.pool.NullPool)

    async def __call__(self, scope, receive, send) -> None:
        import anyio
        from starlette.responses import JSONResponse

        path = scope.get("path", "").split("/")
        if scope["type"] != "http" or len(path) < 4 or path[1:3] != ["api", "v1"]:
            await self.app(scope, receive, send)
            return
        project_id = None
        creating = path[3:] == ["projects"] and scope.get("method") == "POST"
        if len(path) >= 5 and path[3] == "projects":
            try:
                from uuid import UUID

                project_id = str(UUID(path[4]))
            except ValueError:
                await self.app(scope, receive, send)
                return
        elif len(path) > 5 and path[3:5] == ["agent", "jobs"]:
            from uuid import UUID

            try:
                job_id = str(UUID(path[5]))
            except ValueError:
                await JSONResponse({"detail": "Invalid job ID"}, status_code=422)(scope, receive, send)
                return
            def lookup():
                with self.engine.connect() as connection:
                    return connection.execute(sa.text(
                        "SELECT project_id FROM background_jobs WHERE job_id = :job"
                    ), {"job": job_id}).scalar()
            project_id = await anyio.to_thread.run_sync(lookup)
        if project_id is None and not creating:
            await self.app(scope, receive, send)
            return
        guard = operation(self.engine, None if project_id is None else str(project_id))
        try:
            await anyio.to_thread.run_sync(guard.__enter__)
        except ConflictError as exc:
            await JSONResponse({"detail": str(exc)}, status_code=409)(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(lambda: guard.__exit__(None, None, None))
