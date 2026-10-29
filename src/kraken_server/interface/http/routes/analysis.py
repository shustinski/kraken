"""Project-authorized AnalysisRun commands and immutable measurements."""


def register_analysis_routes(app, backend, project_reader, shared_mutation_actor, command_context):
    from fastapi import Depends, Header, Response
    from kraken_server.services import ConflictError

    def capability():
        if not hasattr(backend, "transport_create_analysis_run"):
            raise ConflictError("Analysis execution requires a persistent server backend")

    base = "/api/v1/projects/{project_id}/analysis-runs"

    @app.get(base)
    def list_runs(project_id: str, _=Depends(project_reader)):
        capability()
        return {"items": backend.list_analysis_runs(project_id)}

    @app.post(base, status_code=201)
    def create_run(
        project_id: str,
        payload: dict,
        actor=Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        capability()
        return backend.transport_create_analysis_run(
            project_id, payload, command_context(actor, idempotency_key, None, revision_required=False)
        )

    @app.get(base + "/{run_id}")
    def get_run(project_id: str, run_id: str, _=Depends(project_reader)):
        capability()
        return backend.get_analysis_run(project_id, run_id)

    @app.get(base + "/{run_id}/frames")
    def frames(project_id: str, run_id: str, metric_key: str, _=Depends(project_reader)):
        capability()
        return {"items": backend.analysis_frame_results(project_id, run_id, metric_key)}

    @app.post(base + "/{run_id}/frames/{frame_id}/map", status_code=202)
    def request_map(project_id: str, run_id: str, frame_id: str,
                    actor=Depends(shared_mutation_actor),
                    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
                    if_match: str | None = Header(default=None, alias="If-Match")):
        capability()
        return backend.transport_request_analysis_map(
            project_id, run_id, frame_id, command_context(actor, idempotency_key, if_match, revision_required=True))

    @app.get(base + "/{run_id}/frames/{frame_id}/map")
    def get_map(project_id: str, run_id: str, frame_id: str, _=Depends(project_reader)):
        capability()
        data = backend.analysis_map_bytes(project_id, run_id, frame_id)
        import hashlib
        return Response(data, media_type="image/png", headers={"ETag": '"' + hashlib.sha256(data).hexdigest() + '"',
                                                              "Cache-Control": "private, no-cache"})

    @app.get(base + "/{run_id}/compare/{other_run_id}")
    def compare(project_id: str, run_id: str, other_run_id: str, metric_key: str, _=Depends(project_reader)):
        capability()
        return backend.compare_analysis_runs(project_id, run_id, other_run_id, metric_key)

    @app.post(base + "/{run_id}/commands")
    def command(
        project_id: str,
        run_id: str,
        payload: dict,
        actor=Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ):
        capability()
        return backend.transport_analysis_command(
            project_id, run_id, payload, command_context(actor, idempotency_key, if_match, revision_required=True)
        )
