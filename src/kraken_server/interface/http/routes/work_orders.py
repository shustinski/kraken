"""Issued work orders."""

from __future__ import annotations

from typing import Any

from kraken_server.interface.http.context import HttpContext
from kraken_server.interface.http.prefix import API_PREFIX
from kraken_server.interface.session import SessionPrincipal
from kraken_server.services import ValidationError


def register(ctx: HttpContext) -> None:
    app = ctx.app
    backend = ctx.backend
    heavy_operations = ctx.heavy_operations
    Depends = ctx.Depends
    Header = ctx.Header
    run_in_threadpool = ctx.run_in_threadpool
    shared_mutation_actor = ctx.shared_mutation_actor
    project_reader = ctx.project_reader
    command_context = ctx.command_context

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/work-orders")
    def list_work_orders(project_id: str, _: SessionPrincipal = Depends(project_reader)) -> dict[str, Any]:
        return {"items": backend.list_work_orders(project_id)}

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/work-orders", status_code=201)
    def issue_work_order(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return backend.issue_work_order(
            project_id,
            payload,
            command_context(actor, idempotency_key, None, revision_required=False),
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/work-orders/legacy-migration")
    async def plan_legacy_work_order_migration(
        project_id: str, actor: SessionPrincipal = Depends(shared_mutation_actor),
    ) -> dict[str, Any]:
        method = getattr(backend, "plan_legacy_work_order_migration", None)
        if not callable(method):
            raise ValidationError("Legacy migration requires a persistent metadata backend")
        async with heavy_operations:
            return await run_in_threadpool(method, project_id, actor.principal_id)

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/work-orders/legacy-migration")
    async def migrate_legacy_work_orders(
        project_id: str, payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        method = getattr(backend, "migrate_legacy_work_orders", None)
        if not callable(method):
            raise ValidationError("Legacy migration requires a persistent metadata backend")
        context = command_context(actor, idempotency_key, None, revision_required=False)
        async with heavy_operations:
            return await run_in_threadpool(method, project_id, payload, context)

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/work-orders/{{order_id}}/cancel")
    def cancel_work_order(
        project_id: str,
        order_id: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.cancel_work_order(
            project_id,
            order_id,
            command_context(actor, idempotency_key, if_match, revision_required=False),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/work-orders/{{order_id}}/complete")
    def complete_work_order(
        project_id: str,
        order_id: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.complete_work_order(
            project_id,
            order_id,
            command_context(actor, idempotency_key, if_match, revision_required=False),
        )
