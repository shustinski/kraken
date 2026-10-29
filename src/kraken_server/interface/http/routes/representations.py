"""Representations on a layer."""

from __future__ import annotations

from typing import Any

from kraken_server.interface.http.context import HttpContext
from kraken_server.interface.http.prefix import API_PREFIX
from kraken_server.interface.session import SessionPrincipal


def register(ctx: HttpContext) -> None:
    app = ctx.app
    backend = ctx.backend
    heavy_operations = ctx.heavy_operations
    Depends = ctx.Depends
    HTTPException = ctx.HTTPException
    Header = ctx.Header
    run_in_threadpool = ctx.run_in_threadpool
    shared_mutation_actor = ctx.shared_mutation_actor
    project_reader = ctx.project_reader
    command_context = ctx.command_context

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/layers/{{layer_id}}/representations")
    def list_representations(
        project_id: str,
        layer_id: str,
        include_archived: bool = False,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"items": backend.list_representations(project_id, layer_id, include_archived=include_archived)}

    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/layers/{{layer_id}}/representations",
        status_code=201,
    )
    def create_representation(
        project_id: str,
        layer_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.create_representation(
            project_id,
            layer_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/layers/{{layer_id}}/representations/{{representation_id}}/prepare-inputs"
    )
    async def prepare_representation_inputs(
        project_id: str,
        layer_id: str,
        representation_id: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        operation = getattr(backend, "materialize_representation_inputs", None)
        if operation is None:
            raise HTTPException(status_code=503, detail="Сервер не поддерживает подготовку изображений")
        async with heavy_operations:
            return await run_in_threadpool(
                operation,
                project_id,
                layer_id,
                representation_id,
                command_context(actor, idempotency_key, None, revision_required=False),
            )

    @app.patch(f"{API_PREFIX}/projects/{{project_id}}/layers/{{layer_id}}/representations/{{representation_id}}")
    def update_representation(
        project_id: str,
        layer_id: str,
        representation_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.update_representation(
            project_id,
            layer_id,
            representation_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )
