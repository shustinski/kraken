"""Viewport, history, statistics and legacy analysis publication."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import Request
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from kraken_server.interface.http.context import HttpContext
from kraken_server.interface.http.prefix import API_PREFIX
from kraken_server.interface.session import SessionPrincipal
from kraken_server.services import ValidationError


def register(ctx: HttpContext) -> None:
    app = ctx.app
    backend = ctx.backend
    Depends = ctx.Depends
    Header = ctx.Header
    Query = ctx.Query
    run_in_threadpool = ctx.run_in_threadpool
    shared_mutation_actor = ctx.shared_mutation_actor
    project_reader = ctx.project_reader
    command_context = ctx.command_context

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/viewport")
    def viewport(
        project_id: str,
        layer_id: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        lod: int = 0,
        include_missing: bool = True,
        representation_id: list[str] = Query(default=[]),
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return backend.matrix_viewport(
            project_id,
            layer_id=layer_id,
            representation_ids=representation_id,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            lod=lod,
            include_missing=include_missing,
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/frames")
    def frame_states(
        project_id: str,
        layer_id: str,
        representation_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        project = backend.get_project(project_id)
        viewport = backend.matrix_viewport(
            project_id,
            layer_id=layer_id,
            representation_ids=(representation_id,),
            x1=1,
            y1=1,
            x2=int(project["width"]),
            y2=int(project["height"]),
            lod=0,
            include_missing=False,
        )
        return {"items": viewport["cells"], "revision": viewport["revision"]}

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/history")
    def history(
        project_id: str,
        cursor: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return backend.history(project_id, cursor=cursor, limit=limit)

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/statistics")
    def statistics(
        project_id: str,
        start: datetime,
        end: datetime,
        timezone: str = "UTC",
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValidationError("Statistics timestamps must include a timezone")
        try:
            report_timezone = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValidationError(f"Unknown statistics timezone: {timezone}") from exc
        return backend.statistics(project_id, start=start, end=end, timezone=report_timezone)

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/analyses/karakal", status_code=201)
    async def publish_karakal_analysis(
        project_id: str,
        request: Request,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        payload = dict(await request.json())
        payload.setdefault("run_id", str(uuid4()))
        return await run_in_threadpool(
            backend.publish_karakal_analysis,
            project_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/pipeline-actions", status_code=201)
    async def append_pipeline_event(
        project_id: str,
        request: Request,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        payload = dict(await request.json())
        return await run_in_threadpool(
            backend.append_pipeline_event,
            project_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )
