"""Project notes."""

from __future__ import annotations

from typing import Any

from kraken_server.interface.http.context import HttpContext
from kraken_server.interface.http.prefix import API_PREFIX
from kraken_server.interface.session import SessionPrincipal


def register(ctx: HttpContext) -> None:
    app = ctx.app
    backend = ctx.backend
    Depends = ctx.Depends
    Header = ctx.Header
    shared_mutation_actor = ctx.shared_mutation_actor
    project_reader = ctx.project_reader
    command_context = ctx.command_context

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/notes")
    def list_notes(
        project_id: str,
        layer_id: str | None = None,
        frame_id: str | None = None,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"items": backend.list_notes(project_id, layer_id=layer_id, frame_id=frame_id)}

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/notes", status_code=201)
    def create_note(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return backend.create_note(
            project_id,
            payload,
            command_context(actor, idempotency_key, None, revision_required=False),
        )

    @app.patch(f"{API_PREFIX}/projects/{{project_id}}/notes/{{note_id}}")
    def revise_note(
        project_id: str,
        note_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.revise_note(
            project_id,
            note_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )
