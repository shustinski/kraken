"""HTTP operations for the existing durable engineering-plan event streams."""

from __future__ import annotations

from typing import Any

from kraken_server.services import ConflictError


def register_work_planning_routes(app, backend, project_reader, shared_mutation_actor, command_context):
    from fastapi import Depends, Header

    def capability():
        if not hasattr(backend, "transport_create_work_plan"):
            raise ConflictError("Work planning requires a persistent project backend")

    @app.get("/api/v1/projects/{project_id}/work-plans")
    def list_work_plans(project_id: str, _=Depends(project_reader)) -> dict[str, Any]:
        capability()
        return {"items": backend.list_work_plans(project_id)}

    @app.get("/api/v1/projects/{project_id}/work-plans/{plan_id}")
    def get_work_plan(project_id: str, plan_id: str, _=Depends(project_reader)) -> dict[str, Any]:
        capability()
        return backend.get_work_plan(project_id, plan_id)

    @app.post("/api/v1/projects/{project_id}/work-plans", status_code=201)
    def create_work_plan(
        project_id: str,
        payload: dict[str, Any],
        actor=Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        capability()
        return backend.transport_create_work_plan(
            project_id,
            payload,
            command_context(actor, idempotency_key, None, revision_required=False),
        )

    @app.post("/api/v1/projects/{project_id}/work-plans/{plan_id}/commands")
    def mutate_work_plan(
        project_id: str,
        plan_id: str,
        payload: dict[str, Any],
        actor=Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        capability()
        return backend.transport_mutate_work_plan(
            project_id,
            plan_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.get("/api/v1/projects/{project_id}/work-plan-templates")
    def list_work_templates(project_id: str, _=Depends(project_reader)) -> dict[str, Any]:
        capability()
        return {"items": backend.list_work_templates(project_id)}

    @app.post("/api/v1/projects/{project_id}/work-planning/analytics")
    def work_plan_analytics(project_id: str, payload: dict[str, Any], actor=Depends(project_reader)) -> dict[str, Any]:
        capability()
        return backend.transport_work_plan_analytics(project_id, payload, actor.principal_id)

    @app.get("/api/v1/projects/{project_id}/work-plan-templates/{template_id}")
    def get_work_template(
        project_id: str, template_id: str, version: int | None = None, _=Depends(project_reader)
    ) -> dict[str, Any]:
        capability()
        return backend.get_work_template(project_id, template_id, version=version)

    @app.post("/api/v1/projects/{project_id}/work-plan-templates", status_code=201)
    def create_work_template(
        project_id: str,
        payload: dict[str, Any],
        actor=Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default="0", alias="If-Match"),
    ) -> dict[str, Any]:
        capability()
        return backend.transport_save_work_template(
            project_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.put("/api/v1/projects/{project_id}/work-plan-templates/{template_id}")
    def update_work_template(
        project_id: str,
        template_id: str,
        payload: dict[str, Any],
        actor=Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        capability()
        return backend.transport_save_work_template(
            project_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
            template_id=template_id,
        )
