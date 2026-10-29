"""Quality issue commands, review history and reverse artifact lookup."""


def register_quality_routes(app, backend, project_reader, shared_mutation_actor, command_context):
    from fastapi import Depends, Header
    from kraken_server.services import ConflictError

    def capability():
        if not hasattr(backend, "transport_create_quality_issue"):
            raise ConflictError("Quality workflow requires a persistent project backend")

    @app.get("/api/v1/projects/{project_id}/quality-analysis-runs")
    def analysis_runs(project_id: str, _=Depends(project_reader)):
        capability()
        return {"items": backend.quality_analysis_runs(project_id)}

    @app.get("/api/v1/projects/{project_id}/quality-analysis-runs/{run_id}/readiness")
    def analysis_readiness(project_id: str, run_id: str, _=Depends(project_reader)):
        capability()
        return backend.quality_analysis_readiness(project_id, run_id)

    @app.post("/api/v1/projects/{project_id}/quality-issues/from-analysis", status_code=201)
    def analysis_findings(project_id: str, payload: dict, actor=Depends(shared_mutation_actor),
                          idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        capability()
        return backend.transport_quality_analysis(project_id, payload,
            command_context(actor, idempotency_key, None, revision_required=False))

    @app.post("/api/v1/projects/{project_id}/quality-issues/{issue_id}/analysis-recheck")
    def analysis_recheck(project_id: str, issue_id: str, payload: dict, actor=Depends(shared_mutation_actor),
                         idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
                         if_match: str | None = Header(default=None, alias="If-Match")):
        capability()
        return backend.transport_quality_analysis(project_id, payload,
            command_context(actor, idempotency_key, if_match, revision_required=True), issue_id=issue_id)

    @app.get("/api/v1/projects/{project_id}/quality-artifacts")
    def frame_versions(project_id: str, layer_id: str, x: int, y: int, _=Depends(project_reader)):
        capability()
        return {"items": backend.quality_frame_versions(project_id, layer_id, x, y)}

    @app.get("/api/v1/projects/{project_id}/quality-issues")
    def list_issues(project_id: str, artifact_version_id: str | None = None,
                    frame_id: str | None = None, _=Depends(project_reader)):
        capability()
        return {"items": backend.list_quality_issues(project_id, artifact_version_id=artifact_version_id,
                                                      frame_id=frame_id)}

    @app.get("/api/v1/projects/{project_id}/quality-issues/{issue_id}")
    def get_issue(project_id: str, issue_id: str, _=Depends(project_reader)):
        capability()
        return backend.get_quality_issue(project_id, issue_id)

    @app.get("/api/v1/projects/{project_id}/quality-issues/{issue_id}/history")
    def history(project_id: str, issue_id: str, _=Depends(project_reader)):
        capability()
        return {"items": backend.quality_issue_history(project_id, issue_id)}

    @app.post("/api/v1/projects/{project_id}/quality-issues", status_code=201)
    def create_issue(project_id: str, payload: dict, actor=Depends(shared_mutation_actor),
                     idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        capability()
        return backend.transport_create_quality_issue(
            project_id, payload, command_context(actor, idempotency_key, None, revision_required=False),
        )

    @app.post("/api/v1/projects/{project_id}/quality-issues/{issue_id}/commands")
    def mutate_issue(project_id: str, issue_id: str, payload: dict, actor=Depends(shared_mutation_actor),
                     idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
                     if_match: str | None = Header(default=None, alias="If-Match")):
        capability()
        return backend.transport_mutate_quality_issue(
            project_id, issue_id, payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )
