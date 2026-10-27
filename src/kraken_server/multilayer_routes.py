"""Project-authorized physical model reads and compare-and-swap mutations."""
from __future__ import annotations

from typing import Any


def register_multilayer_routes(app, backend, project_reader, shared_mutation_actor, command_context):
    from fastapi import Depends, Header
    from kraken_server.services import ConflictError

    def supported():
        if not hasattr(backend, "transport_mutate_multilayer_model"):
            raise ConflictError("Physical registration requires a persistent backend")

    @app.get("/api/v1/projects/{project_id}/multilayer")
    def get_model(project_id: str, _=Depends(project_reader)) -> dict[str, Any]:
        supported()
        return backend.get_multilayer_model(project_id)

    @app.post("/api/v1/projects/{project_id}/multilayer/commands")
    def command(
        project_id: str, payload: dict[str, Any], actor=Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        supported()
        return backend.transport_mutate_multilayer_model(
            project_id, payload, command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.get("/api/v1/projects/{project_id}/stitch-renders/{job_id}/blocks/{column}/{row}")
    def stitch_block(project_id: str, job_id: str, column: int, row: int,
                     coverage: bool = False, _=Depends(project_reader)) -> dict[str, Any]:
        supported()
        return backend.stitch_block_preview(project_id, job_id, column, row, coverage=coverage)

    @app.get("/api/v1/projects/{project_id}/stitch-renders/{job_id}/vectors")
    def vector_header(project_id: str, job_id: str, _=Depends(project_reader)) -> dict[str, Any]:
        supported()
        return backend.stitch_vector_header(project_id, job_id)

    @app.get("/api/v1/projects/{project_id}/stitch-renders/{job_id}/vector-blocks/{column}/{row}")
    def vector_block(project_id: str, job_id: str, column: int, row: int, _=Depends(project_reader)) -> dict[str, Any]:
        supported()
        return backend.stitch_vector_block(project_id, job_id, column, row)
