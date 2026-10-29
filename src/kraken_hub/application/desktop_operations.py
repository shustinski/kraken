"""Desktop host ports: sessions, administration, execution, and subscriptions.

These concerns are separate from the backend-independent ProjectOperations port.
Qt names this interface; implementations are supplied by the composition root.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from os import PathLike
from typing import Any, Protocol

from kraken_core.plugin_protocol import WorkspacePluginResultV1
from kraken_hub.application.notifications import ProjectEventWake
from kraken_manager.application.reporting import ReportMetrics, ReportSeries
from kraken_manager.application.results import IntegrityScanResult, ProjectDeletionResult, UserSession
from kraken_manager.domain.common import LayerId, PerformerId, ProjectId
from kraken_manager.domain.identity import Performer, Principal
from kraken_manager.domain.project import Layer, Project, Representation
from kraken_manager.workspace import DerivedRun, DerivedRunKind, ProjectWorkspaceBinding


class DesktopOperations(Protocol):
    def add_catalog_handler(self, handler: Callable[[], None]) -> None: ...

    def add_status_handler(self, handler: Callable[[str], None]) -> None: ...

    def add_wake_handler(self, handler: Callable[[ProjectEventWake], None]) -> None: ...

    def archive_performer(self, performer_id: PerformerId | str) -> Performer: ...

    def attach_project(
        self, project_directory: PathLike[str] | str, *, principal: Principal | None = ..., take_ownership: bool = ...
    ) -> Project: ...

    def begin_derived_run(
        self,
        *,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        layer_name: str,
        kind: DerivedRunKind,
        plugin_id: str,
        operation: str,
        principal: Principal | None = ...,
    ) -> DerivedRun: ...

    def bind_project_workspace(
        self,
        *,
        principal: Principal,
        project: Project,
        source_root: PathLike[str] | str,
        derived_root: PathLike[str] | str,
        idempotency_key: str,
    ) -> ProjectWorkspaceBinding: ...

    def command_server_analysis(
        self,
        project_id: object,
        run_id: object,
        operation: object,
        *,
        expected_revision: object,
        idempotency_key: object,
    ) -> object: ...

    def configure_remote(self, *, base_url: str, access_token: str, principal: Principal) -> object: ...

    def configure_remote_local(self, *, base_url: str, username: str, password: str) -> object: ...

    def create_initial_account(self, username: str, display_name: str, password: str) -> UserSession: ...

    def create_manual_performer(self, *, name: str, color: str) -> Performer: ...

    def create_server_analysis(self, project_id: object, payload: object, *, idempotency_key: object) -> object: ...

    def delete_project(
        self, *, principal: Principal, project: Project, confirmation_name: str
    ) -> ProjectDeletionResult: ...

    def export_backup(
        self, project_id: ProjectId | str, destination: PathLike[str] | str, *, principal: Principal | None = ...
    ) -> object: ...

    def fail_derived_run(
        self, *, principal: Principal, project_id: ProjectId | str, run_id: str, error: str
    ) -> DerivedRun: ...

    def import_agent_publications(
        self, *, principal: Principal, publications: Iterable[Mapping[str, object]], staging_root: PathLike[str] | str
    ) -> object: ...

    def import_agent_result(
        self,
        *,
        principal: Principal,
        result_payload: Mapping[str, object],
        staging_root: PathLike[str] | str,
        confirm_partial: bool = ...,
    ) -> object: ...

    def import_backup(
        self, bundle_root: PathLike[str] | str, *, principal: Principal | None = ..., take_ownership: bool = ...
    ) -> Project: ...

    def list_derived_runs(
        self, project_id: ProjectId | str, layer_id: LayerId | str = ...
    ) -> tuple[DerivedRun, ...]: ...

    def list_project_performers(self, project_id: object, *, include_archived: bool = ...) -> object: ...

    def login(self, username: str, password: str) -> UserSession | None: ...

    def publish_workspace_plugin_result(
        self, *, principal: Principal, project: Project, layer: Layer, result: WorkspacePluginResultV1
    ) -> tuple[DerivedRun, Representation | None]: ...

    def refresh_stitch(self, *, project_id: object, job_id: object, principal: object, gateway: object) -> object: ...

    def request_project_deletion(self, *, project: Project, reason: str = ..., **_kwargs) -> object: ...

    def request_server_analysis_map(
        self,
        project_id: object,
        run_id: object,
        frame_id: object,
        *,
        expected_revision: object,
        idempotency_key: object,
    ) -> object: ...

    def run_interactive_job(self, project_id: object, job_id: object, *, registry: object) -> dict[str, Any]: ...

    def scan_integrity(self) -> IntegrityScanResult: ...

    def server_analysis_frames(self, project_id: object, run_id: object, metric_key: object) -> object: ...

    def server_analysis_map_bytes(self, project_id: object, run_id: object, frame_id: object) -> object: ...

    def server_analysis_run(self, project_id: object, run_id: object) -> object: ...

    def server_analysis_runs(self, project_id: object) -> object: ...

    def start_sync(self) -> None: ...

    def statistics(
        self, project_id: object, *, start: datetime, end: datetime, timezone: object
    ) -> tuple[ReportMetrics, dict[str, ReportSeries]]: ...

    def stop_sync(self) -> None: ...

    def subscribe_project(self, project_id: object) -> None: ...

    def tick_layer_pipeline(
        self,
        *,
        project_id: object,
        layer_id: object,
        principal: object,
        gateway: object = ...,
        stitch_gateway: object = ...,
    ) -> object: ...

    def unsubscribe_project(self, project_id: object) -> None: ...

    def update_performer(self, *, performer_id: PerformerId | str, name: str, color: str) -> Performer: ...
