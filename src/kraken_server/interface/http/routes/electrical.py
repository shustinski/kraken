"""Project-authorized electrical evidence and partial EDIF netlists."""
from __future__ import annotations

from typing import Any


def register_electrical_routes(app, backend, project_reader, shared_mutation_actor, command_context):
    from fastapi import Depends, Header
    from kraken_server.services import ConflictError

    def supported():
        if not hasattr(backend, "transport_mutate_electrical_model"):
            raise ConflictError("electrical evidence requires a persistent backend")

    @app.get("/api/v1/projects/{project_id}/electrical")
    def get_model(project_id: str, _=Depends(project_reader)) -> dict[str, Any]:
        supported()
        return backend.get_electrical_model(project_id)

    @app.get("/api/v1/projects/{project_id}/electrical/export")
    def export(project_id: str, _=Depends(project_reader)) -> dict[str, Any]:
        supported()
        return backend.export_electrical_model(project_id)

    @app.post("/api/v1/projects/{project_id}/electrical/commands")
    def command(project_id: str, payload: dict[str, Any], actor=Depends(shared_mutation_actor),
                idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
                if_match: str | None = Header(default=None, alias="If-Match")) -> dict[str, Any]:
        supported()
        return backend.transport_mutate_electrical_model(project_id, payload,
            command_context(actor, idempotency_key, if_match, revision_required=True))
