"""One authenticated transport for the application's declared public command catalog."""
from typing import Any

from kraken_server.interface.http.context import HttpContext
from kraken_server.interface.http.prefix import API_PREFIX
from kraken_server.interface.session import SessionPrincipal


def register(ctx: HttpContext) -> None:
    app, backend = ctx.app, ctx.backend
    Depends, Header = ctx.Depends, ctx.Header
    shared_mutation_actor = ctx.shared_mutation_actor

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/commands")
    async def execute_project_command(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        context = ctx.command_context(actor, idempotency_key, None, revision_required=False)
        operation = getattr(backend, "execute_project_command", None)
        if operation is None:
            raise ctx.HTTPException(status_code=503, detail="The project command gateway is unavailable")
        async with ctx.heavy_operations:
            return await ctx.run_in_threadpool(operation, project_id, payload, context)
