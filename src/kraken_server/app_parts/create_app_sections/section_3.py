"""create_app section 3."""
from __future__ import annotations
import asyncio
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import anyio
from kraken_server.services import (
    CommandContext,
    ConflictError,
    ForbiddenError,
    InMemoryServerServices,
    NotFoundError,
    ServerServices,
    ValidationError,
)
from kraken_server.app_parts.module_state import API_PREFIX
from kraken_server.app_parts.session_principal import SessionPrincipal

def _create_app_section_3(blob_gateway, Depends, HTTPException, Header, Query, Request, app, backend, command_context, heavy_operations, project_reader, run_in_threadpool, shared_mutation_actor, transfer_identity, transfer_payload, transfer_ticket_context):
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

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/viewport")
    def viewport(
        project_id: str,
        layer_id: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        lod: int = 0,
        include_missing: bool = True,
        representation_id: list[str] = Query(default=[]),
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return backend.matrix_viewport(
            project_id,
            layer_id=layer_id,
            representation_ids=representation_id,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            lod=lod,
            include_missing=include_missing,
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/frames")
    def frame_states(
        project_id: str,
        layer_id: str,
        representation_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        project = backend.get_project(project_id)
        viewport = backend.matrix_viewport(
            project_id,
            layer_id=layer_id,
            representation_ids=(representation_id,),
            x1=1,
            y1=1,
            x2=int(project["width"]),
            y2=int(project["height"]),
            lod=0,
            include_missing=False,
        )
        return {"items": viewport["cells"], "revision": viewport["revision"]}

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/history")
    def history(
        project_id: str,
        cursor: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return backend.history(project_id, cursor=cursor, limit=limit)

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/statistics")
    def statistics(
        project_id: str,
        start: datetime,
        end: datetime,
        timezone: str = "UTC",
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValidationError("Statistics timestamps must include a timezone")
        try:
            report_timezone = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValidationError(f"Unknown statistics timezone: {timezone}") from exc
        return backend.statistics(project_id, start=start, end=end, timezone=report_timezone)

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/analyses/karakal", status_code=201)
    async def publish_karakal_analysis(
        project_id: str,
        request: Request,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        payload = dict(await request.json())
        payload.setdefault("run_id", str(uuid4()))
        return await run_in_threadpool(
            backend.publish_karakal_analysis,
            project_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/pipeline-actions", status_code=201)
    async def append_pipeline_event(
        project_id: str,
        request: Request,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        payload = dict(await request.json())
        return await run_in_threadpool(
            backend.append_pipeline_event,
            project_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/artifacts")
    def list_artifacts(
        project_id: str,
        layer_id: str | None = None,
        representation_id: str | None = None,
        frame_id: str | None = None,
        include_archived: bool = False,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {
            "items": backend.list_artifact_series(
                project_id,
                layer_id=layer_id,
                representation_id=representation_id,
                frame_id=frame_id,
                include_archived=include_archived,
            )
        }

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/artifacts", status_code=201)
    def create_artifact_series(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        return backend.create_artifact_series(
            project_id,
            payload,
            command_context(actor, idempotency_key, None, revision_required=False),
        )

    @app.patch(f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}")
    def mutate_artifact_series(
        project_id: str,
        series_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.mutate_artifact_series(
            project_id,
            series_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}/versions")
    def list_artifact_versions(
        project_id: str,
        series_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"items": backend.list_artifact_versions(project_id, series_id)}

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}/revision")
    def artifact_stream_revision(
        project_id: str,
        series_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, int]:
        return {"revision": backend.artifact_stream_revision(project_id, series_id)}

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}/active")
    def active_artifact_version(
        project_id: str,
        series_id: str,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"item": backend.get_active_artifact_version(project_id, series_id)}

    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}/versions",
        status_code=201,
    )
    async def upload_artifact_version(
        project_id: str,
        series_id: str,
        request: Request,
        filename: str,
        media_type: str = "application/octet-stream",
        sha256: str = "",
        parent_version_id: str | None = None,
        content_length: int | None = Header(default=None, alias="Content-Length"),
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        context = command_context(actor, idempotency_key, if_match, revision_required=True)
        if re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
            raise ValidationError("A valid SHA-256 digest is required")
        sha256 = sha256.lower()
        received = 0
        with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as staged:
            async for chunk in request.stream():
                received += len(chunk)
                staged.write(chunk)
            if content_length is None or received != content_length:
                raise ValidationError("Content-Length is required and must match the uploaded blob size")
            staged.seek(0)

            def chunks() -> Any:
                while True:
                    chunk = staged.read(1024 * 1024)
                    if not chunk:
                        return
                    yield chunk

            return await run_in_threadpool(
                backend.add_managed_artifact_version,
                project_id,
                series_id,
                {
                    "filename": filename,
                    "media_type": media_type,
                    "sha256": sha256,
                    "parent_version_id": parent_version_id,
                },
                chunks(),
                context,
            )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/artifacts/{{series_id}}/uploads")
    def initiate_artifact_upload(
        project_id: str,
        series_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        if blob_gateway is None:
            return {"mode": "proxy"}
        context = command_context(actor, idempotency_key, if_match, revision_required=True)
        digest, size_bytes = transfer_identity(payload)
        command_payload = transfer_payload(payload, digest=digest, size_bytes=size_bytes)
        existing = backend.prepare_managed_artifact_upload(project_id, series_id, command_payload, context)
        if existing is not None:
            return {"mode": "completed", "artifact": existing}
        return blob_gateway.ticket(
            "upload",
            digest,
            size_bytes,
            context=transfer_ticket_context(actor, project_id, series_id, command_payload, context.idempotency_key),
        )
