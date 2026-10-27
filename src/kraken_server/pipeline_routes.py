"""The executable definition and run state of the existing layer graph."""

from kraken_server.services import ConflictError


def register_pipeline_routes(app, backend, project_reader, shared_mutation_actor, command_context):
    from fastapi import Depends, Header

    def capability():
        if not hasattr(backend, "transport_layer_pipeline"):
            raise ConflictError("Executable pipelines require a persistent backend")

    @app.get("/api/v1/projects/{project_id}/layers/{layer_id}/pipeline/workflow")
    def get_pipeline(project_id: str, layer_id: str, _=Depends(project_reader)):
        capability()
        return backend.get_layer_pipeline(project_id, layer_id)

    @app.put("/api/v1/projects/{project_id}/layers/{layer_id}/pipeline/workflow")
    def define_pipeline(
        project_id: str,
        layer_id: str,
        payload: dict,
        actor=Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ):
        capability()
        return backend.transport_layer_pipeline(
            project_id,
            layer_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
            define=True,
        )

    @app.post("/api/v1/projects/{project_id}/layers/{layer_id}/pipeline/workflow/commands")
    def command_pipeline(
        project_id: str,
        layer_id: str,
        payload: dict,
        actor=Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ):
        capability()
        return backend.transport_layer_pipeline(
            project_id, layer_id, payload, command_context(actor, idempotency_key, if_match, revision_required=True)
        )
