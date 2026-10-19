"""services: assign project role."""
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime
from typing import Any, Protocol


class AssignProjectRoleMixin(Protocol):
    def request_project_deletion(self, project_id: str, actor_id: str, reason: str = "") -> dict[str, Any]: ...

    def project_deletion_requests(self, project_id: str) -> list[dict[str, Any]]: ...

    def health(self) -> dict[str, Any]: ...

    def list_projects(self, *, include_archived: bool = False) -> list[dict[str, Any]]: ...

    def list_principals(self, *, include_inactive: bool = False) -> list[dict[str, Any]]: ...

    def list_performers(self, *, include_archived: bool = False) -> list[dict[str, Any]]: ...

    def create_project(self, payload: Mapping[str, Any], context: CommandContext) -> dict[str, Any]: ...

    def get_project(self, project_id: str) -> dict[str, Any]: ...

    def rename_project(
        self, project_id: str, name: str, context: CommandContext
    ) -> dict[str, Any]: ...

    def archive_project(self, project_id: str, context: CommandContext) -> dict[str, Any]: ...

    def restore_project(self, project_id: str, context: CommandContext) -> dict[str, Any]: ...

    def list_layers(
        self, project_id: str, *, include_archived: bool = False
    ) -> list[dict[str, Any]]: ...

    def create_layer(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def rename_layer(
        self, project_id: str, layer_id: str, name: str, context: CommandContext
    ) -> dict[str, Any]: ...

    def reorder_layer(
        self, project_id: str, layer_id: str, order: int, context: CommandContext
    ) -> dict[str, Any]: ...

    def reorder_layers(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> list[dict[str, Any]]: ...

    def archive_layer(
        self, project_id: str, layer_id: str, context: CommandContext
    ) -> dict[str, Any]: ...

    def project_roles(self, project_id: str, principal_id: str) -> dict[str, Any]: ...

    def assign_project_role(
        self,
        project_id: str,
        principal_id: str,
        role: str,
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def revoke_project_role(
        self,
        project_id: str,
        principal_id: str,
        role: str,
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def list_representations(
        self, project_id: str, layer_id: str, *, include_archived: bool = False
    ) -> list[dict[str, Any]]: ...

    def create_representation(
        self,
        project_id: str,
        layer_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def update_representation(
        self, project_id: str, layer_id: str, representation_id: str,
        payload: Mapping[str, Any], context: CommandContext,
    ) -> dict[str, Any]: ...

    def matrix_viewport(
        self, project_id: str, *, layer_id: str, representation_ids: Iterable[str],
        x1: int, y1: int, x2: int, y2: int, lod: int, include_missing: bool = True
    ) -> dict[str, Any]: ...

    def history(self, project_id: str, *, cursor: str | None, limit: int) -> dict[str, Any]: ...

    def statistics(
        self, project_id: str, *, start: datetime, end: datetime, timezone: Any
    ) -> dict[str, Any]: ...

    def publish_karakal_analysis(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def append_pipeline_event(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def list_artifact_series(
        self, project_id: str, *, layer_id: str | None = None,
        representation_id: str | None = None, frame_id: str | None = None,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]: ...

    def list_artifact_versions(self, project_id: str, series_id: str) -> list[dict[str, Any]]: ...

    def artifact_stream_revision(self, project_id: str, series_id: str) -> int: ...

    def get_active_artifact_version(self, project_id: str, series_id: str) -> dict[str, Any] | None: ...

    def get_artifact_version(self, project_id: str, version_id: str) -> dict[str, Any]: ...

    def create_artifact_series(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def add_managed_artifact_version(
        self, project_id: str, series_id: str, payload: Mapping[str, Any],
        chunks: Iterable[bytes], context: CommandContext,
    ) -> dict[str, Any]: ...

    def prepare_managed_artifact_upload(
        self, project_id: str, series_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any] | None: ...

    def register_managed_artifact_upload(
        self, project_id: str, series_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def add_external_artifact_version(
        self, project_id: str, series_id: str, payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def mutate_artifact_series(
        self, project_id: str, series_id: str, payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def list_notes(
        self, project_id: str, *, layer_id: str | None = None, frame_id: str | None = None
    ) -> list[dict[str, Any]]: ...

    def create_note(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def revise_note(
        self, project_id: str, note_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def iter_artifact_bytes(
        self, project_id: str, version_id: str, *, chunk_size: int = 1024 * 1024
    ) -> Iterator[bytes]: ...

    def list_review_batches(
        self, project_id: str, *, active_only: bool = False
    ) -> list[dict[str, Any]]: ...

    def create_review_batch(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def mutate_review_batch(
        self, project_id: str, batch_id: str, payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def export_review_package(
        self, project_id: str, batch_id: str, destination: str,
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def inspect_review_return(
        self, project_id: str, batch_id: str, source: str,
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def commit_review_return(
        self, project_id: str, batch_id: str, source: str,
        context: CommandContext,
    ) -> dict[str, Any]: ...

    def list_plugin_jobs(self, project_id: str | None = None) -> list[dict[str, Any]]: ...

    def start_desktop_job(self, project_id: str, job_id: str, context: CommandContext) -> None: ...

    def authorize_desktop_job(self, project_id: str, principal_id: str) -> None: ...

    def submit_plugin_job(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]: ...

    def cancel_plugin_job(
        self, project_id: str, job_id: str, context: CommandContext
    ) -> dict[str, Any]: ...

    def fail_agent_job(self, job_id: str, error: str) -> dict[str, Any]: ...

    def import_agent_result(self, job_id: str, agent: Any) -> dict[str, Any]: ...

if TYPE_CHECKING:
    from kraken_server.services_parts.command_context import CommandContext
