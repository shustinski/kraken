"""Layer catalog and layer commands."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from kraken_server.interface.http.context import HttpContext
from kraken_server.interface.http.prefix import API_PREFIX
from kraken_server.interface.session import SessionPrincipal
from kraken_server.services import CommandContext, NotFoundError, ValidationError


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

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/layers")
    def list_layers(
        project_id: str,
        include_archived: bool = False,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"items": backend.list_layers(project_id, include_archived=include_archived)}

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers", status_code=201)
    def create_layer(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.create_layer(
            project_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/layers/{{layer_id}}/files")
    def get_layer_files(
        project_id: str,
        layer_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        binding = getattr(backend, "layer_file_binding", lambda *_args: None)(project_id, layer_id)
        if binding is None:
            raise NotFoundError(layer_id)
        return binding

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers/scan")
    def scan_layer_source(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
    ) -> dict[str, Any]:
        directory = payload.get("directory")
        if not isinstance(directory, str) or not directory.strip():
            raise ValidationError("A source directory is required")
        operation = getattr(backend, "scan_layer_source", None)
        if operation is None:
            raise HTTPException(status_code=503, detail="Server source scanning is unavailable")
        return operation(project_id, directory, CommandContext(actor.principal_id, str(uuid4()), None))

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers/external", status_code=201)
    def create_external_layer(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        operation = getattr(backend, "create_external_layer", None)
        if operation is None:
            raise HTTPException(status_code=503, detail="Сервер не принимает файловые слои")
        return operation(
            project_id,
            payload,
            command_context(actor, idempotency_key, None, revision_required=False),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers/import", status_code=201)
    async def import_workspace_layer(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        operation = getattr(backend, "import_workspace_layer", None)
        if operation is None:
            raise HTTPException(status_code=503, detail="Сервер не принимает импорт слоя")
        async with heavy_operations:
            return await run_in_threadpool(
                operation,
                project_id,
                payload,
                command_context(actor, idempotency_key, None, revision_required=False),
            )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers/{{layer_id}}/rename")
    def rename_layer(
        project_id: str,
        layer_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.rename_layer(
            project_id,
            layer_id,
            str(payload.get("name", "")),
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers/{{layer_id}}/reorder")
    def reorder_layer(
        project_id: str,
        layer_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.reorder_layer(
            project_id,
            layer_id,
            int(payload.get("order", -1)),
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers/reorder")
    def reorder_layers(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return {
            "items": backend.reorder_layers(
                project_id,
                payload,
                command_context(actor, idempotency_key, None, revision_required=False),
            )
        }

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/layers/{{layer_id}}/archive")
    def archive_layer(
        project_id: str,
        layer_id: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.archive_layer(
            project_id,
            layer_id,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )
