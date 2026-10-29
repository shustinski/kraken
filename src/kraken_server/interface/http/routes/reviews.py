"""Review batches, packages and returns."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from fastapi import Request

from kraken_server.interface.http.context import HttpContext
from kraken_server.interface.http.prefix import API_PREFIX
from kraken_server.interface.session import SessionPrincipal
from kraken_server.services import ValidationError


def register(ctx: HttpContext) -> None:
    app = ctx.app
    backend = ctx.backend
    heavy_operations = ctx.heavy_operations
    Depends = ctx.Depends
    Header = ctx.Header
    StreamingResponse = ctx.StreamingResponse
    run_in_threadpool = ctx.run_in_threadpool
    shared_mutation_actor = ctx.shared_mutation_actor
    project_reader = ctx.project_reader
    command_context = ctx.command_context
    extract_review_archive = ctx.extract_review_archive

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/reviews")
    def list_reviews(
        project_id: str,
        active_only: bool = False,
        _: SessionPrincipal = Depends(project_reader),
    ) -> dict[str, Any]:
        return {"items": backend.list_review_batches(project_id, active_only=active_only)}

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/reviews", status_code=201)
    def create_review(
        project_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.create_review_batch(
            project_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.patch(f"{API_PREFIX}/projects/{{project_id}}/reviews/{{batch_id}}")
    def mutate_review(
        project_id: str,
        batch_id: str,
        payload: dict[str, Any],
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return backend.mutate_review_batch(
            project_id,
            batch_id,
            payload,
            command_context(actor, idempotency_key, if_match, revision_required=True),
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/reviews/{{batch_id}}/package")
    async def export_review_package(
        project_id: str,
        batch_id: str,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> Any:
        def build_archive() -> tuple[Path, Path, dict[str, Any]]:
            staging = Path(tempfile.mkdtemp(prefix="kraken-review-export-"))
            package_dir = staging / "package"
            archive = staging / "review.zip"
            try:
                issued = backend.export_review_package(
                    project_id,
                    batch_id,
                    str(package_dir),
                    command_context(actor, idempotency_key, if_match, revision_required=True),
                )
                with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as output:
                    for source in sorted(package_dir.rglob("*")):
                        if source.is_file():
                            output.write(source, source.relative_to(package_dir).as_posix())
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            return staging, archive, issued

        async with heavy_operations:
            staging, archive, issued = await run_in_threadpool(build_archive)

        def chunks() -> Any:
            try:
                with archive.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        yield chunk
            finally:
                shutil.rmtree(staging, ignore_errors=True)

        return StreamingResponse(
            chunks(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="review-{batch_id}.zip"',
                "X-Kraken-Review-Revision": str(issued.get("revision", "")),
            },
        )

    async def receive_review_return(
        *,
        project_id: str,
        batch_id: str,
        request: Request,
        actor: SessionPrincipal,
        idempotency_key: str | None,
        if_match: str | None,
        commit: bool,
    ) -> dict[str, Any]:
        staging = Path(tempfile.mkdtemp(prefix="kraken-review-return-"))
        archive = staging / "return.zip"
        source = staging / "package"
        try:
            size = 0
            with archive.open("xb") as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 16 * 1024**3:
                        raise ValidationError("Review return archive is too large")
                    output.write(chunk)
            context = command_context(actor, idempotency_key, if_match, revision_required=True)

            def apply_return() -> dict[str, Any]:
                source.mkdir()
                extract_review_archive(archive, source)
                if commit:
                    return backend.commit_review_return(project_id, batch_id, str(source), context)
                return backend.inspect_review_return(project_id, batch_id, str(source), context)

            async with heavy_operations:
                return await run_in_threadpool(apply_return)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/reviews/{{batch_id}}/return/preflight")
    async def review_return_preflight(
        project_id: str,
        batch_id: str,
        request: Request,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return await receive_review_return(
            project_id=project_id,
            batch_id=batch_id,
            request=request,
            actor=actor,
            idempotency_key=idempotency_key,
            if_match=if_match,
            commit=False,
        )

    @app.post(f"{API_PREFIX}/projects/{{project_id}}/reviews/{{batch_id}}/return/commit")
    async def commit_review_return(
        project_id: str,
        batch_id: str,
        request: Request,
        actor: SessionPrincipal = Depends(shared_mutation_actor),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> dict[str, Any]:
        return await receive_review_return(
            project_id=project_id,
            batch_id=batch_id,
            request=request,
            actor=actor,
            idempotency_key=idempotency_key,
            if_match=if_match,
            commit=True,
        )
