"""Compose local filesystem and remote server catalogs behind one façade."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from kraken_manager.application.ports import StorageProfile
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.project import GridOrientation, Project
from kraken_manager.infrastructure.review.manifest import manifest_from_json

from .composition import EmbeddedProjectService
from .remote_client import RemoteServerError, RemoteServerProjectService
from .workspace_service import REMOTE_STORAGE_PROFILE, ProjectEventWake

LOGGER = logging.getLogger(__name__)


class DualCatalogService:
    """Route catalog/project operations to local or remote backends by ownership."""

    def __init__(
        self,
        local: EmbeddedProjectService,
        remote: RemoteServerProjectService | None = None,
    ) -> None:
        self.local = local
        self.remote = remote
        self.data_dir = local.data_dir
        self.profile = local.profile
        self._remote_ids: set[str] = set()
        if remote is not None:
            try:
                for project in remote.list_projects(include_archived=True):
                    self._remote_ids.add(str(project.id))
            except (OSError, RemoteServerError, ValueError):
                # Server may be temporarily unreachable; catalog refresh will retry.
                LOGGER.exception("Could not load the initial remote project catalog")
                self._remote_ids = set()

    @property
    def supports_workspace_roots(self) -> bool:
        return True

    @property
    def supports_live_sync(self) -> bool:
        return self.remote is not None and self.remote.supports_live_sync

    @property
    def has_accounts(self) -> bool:
        return self.local.has_accounts

    @property
    def accounts(self) -> Any:
        return self.local.accounts

    @property
    def identities(self) -> Any:
        return self.local.identities

    @property
    def performers(self) -> Any:
        return self.local.performers

    @property
    def default_source_root(self) -> Path:
        return self.local.default_source_root

    @property
    def default_derived_root(self) -> Path:
        return self.local.default_derived_root

    @property
    def workspace_files(self) -> Any:
        return self.local.workspace_files

    @property
    def workspace_registry(self) -> Any:
        return self.local.workspace_registry

    def list_storage_profiles(self) -> tuple[StorageProfile, ...]:
        return (self.local.profile, REMOTE_STORAGE_PROFILE)

    def is_remote_project(self, project_id: object) -> bool:
        return str(project_id) in self._remote_ids

    def project_storage_label(self, project_id: object) -> str:
        if self.is_remote_project(project_id):
            return "Сервер PostgreSQL"
        return "Локальный файл"

    def _backend(self, project_id: object) -> Any:
        if self.is_remote_project(project_id) and self.remote is not None:
            return self.remote
        return self.local

    def list_projects(self, *, include_archived: bool = False) -> tuple[Project, ...]:
        local_projects = list(self.local.list_projects(include_archived=include_archived))
        remote_projects: list[Project] = []
        if self.remote is not None:
            try:
                remote_projects = list(self.remote.list_projects(include_archived=include_archived))
                self._remote_ids = {
                    str(project.id)
                    for project in self.remote.list_projects(include_archived=True)
                }
            except (OSError, RemoteServerError, ValueError):
                LOGGER.exception("Could not refresh the remote project catalog")
                remote_projects = []
        merged = {str(project.id): project for project in local_projects}
        for project in remote_projects:
            merged[str(project.id)] = project
        return tuple(sorted(merged.values(), key=lambda item: item.name.casefold()))

    def get_project(
        self,
        project_id: object,
        *,
        as_of: datetime | None = None,
    ) -> Project | None:
        if self.is_remote_project(project_id) and self.remote is not None:
            return self.remote.get_project(project_id, as_of=as_of)
        project = self.local.get_project(project_id, as_of=as_of)
        if project is not None:
            return project
        if self.remote is not None:
            remote = self.remote.get_project(project_id, as_of=as_of)
            if remote is not None:
                self._remote_ids.add(str(remote.id))
            return remote
        return None

    def create_project(
        self,
        *,
        principal: Principal,
        name: str,
        width: int,
        height: int,
        orientation: GridOrientation,
        idempotency_key: str,
        layer_template: bool = False,
        source_root: Path | str | None = None,
        derived_root: Path | str | None = None,
        storage_profile_id: str | None = None,
    ) -> Project:
        profile_id = storage_profile_id or self.local.profile.id
        if profile_id == REMOTE_STORAGE_PROFILE.id:
            if self.remote is None:
                raise RuntimeError(
                    "Shared PostgreSQL catalog is not configured. "
                    "Set KRAKEN_SERVER_URL and KRAKEN_GITLAB_TOKEN."
                )
            actor = self.remote.auth.principal
            project = self.remote.create_project(
                principal=actor,
                name=name,
                width=width,
                height=height,
                orientation=orientation,
                idempotency_key=idempotency_key,
                storage_profile_id=profile_id,
            )
            self._remote_ids.add(str(project.id))
            return project
        return self.local.create_project(
            principal=principal,
            name=name,
            width=width,
            height=height,
            orientation=orientation,
            idempotency_key=idempotency_key,
            layer_template=layer_template,
            source_root=source_root,
            derived_root=derived_root,
        )

    def project_workspace(self, project_id: object) -> object | None:
        if self.is_remote_project(project_id):
            return None
        return self.local.project_workspace(project_id)

    def start_sync(self) -> None:
        if self.remote is not None:
            self.remote.start_sync()

    def stop_sync(self) -> None:
        if self.remote is not None:
            self.remote.stop_sync()

    def subscribe_project(self, project_id: object) -> None:
        if self.is_remote_project(project_id) and self.remote is not None:
            self.remote.subscribe_project(project_id)

    def unsubscribe_project(self, project_id: object) -> None:
        if self.remote is not None:
            self.remote.unsubscribe_project(project_id)

    def add_wake_handler(self, handler: Callable[[ProjectEventWake], None]) -> None:
        if self.remote is not None:
            self.remote.add_wake_handler(handler)

    def add_catalog_handler(self, handler: Callable[[], None]) -> None:
        if self.remote is not None:
            self.remote.add_catalog_handler(handler)

    def add_status_handler(self, handler: Callable[[str], None]) -> None:
        if self.remote is not None:
            self.remote.add_status_handler(handler)

    def rename_project(self, **kwargs):
        project = kwargs.get("project")
        if project is not None and self.is_remote_project(project.id) and self.remote is not None:
            return self.remote.rename_project(**kwargs)
        return self.local.rename_project(**kwargs)

    def archive_project(self, **kwargs):
        project = kwargs.get("project")
        if project is not None and self.is_remote_project(project.id) and self.remote is not None:
            return self.remote.archive_project(**kwargs)
        return self.local.archive_project(**kwargs)

    def restore_project(self, **kwargs):
        project = kwargs.get("project")
        if project is not None and self.is_remote_project(project.id) and self.remote is not None:
            return self.remote.restore_project(**kwargs)
        return self.local.restore_project(**kwargs)

    def list_layers(self, project_id, *, include_archived: bool = False):
        return self._backend(project_id).list_layers(project_id, include_archived=include_archived)

    def create_layer(self, **kwargs):
        project = kwargs.get("project")
        return self._backend(project.id).create_layer(**kwargs)

    def rename_layer(self, **kwargs):
        project = kwargs.get("project")
        return self._backend(project.id).rename_layer(**kwargs)

    def reorder_layers(self, **kwargs):
        project = kwargs.get("project")
        return self._backend(project.id).reorder_layers(**kwargs)

    def archive_layer(self, **kwargs):
        project = kwargs.get("project")
        return self._backend(project.id).archive_layer(**kwargs)

    def list_representations(
        self,
        project_id,
        layer_id,
        *,
        include_archived: bool = False,
    ):
        return self._backend(project_id).list_representations(
            project_id,
            layer_id,
            include_archived=include_archived,
        )

    def create_representation(self, **kwargs):
        project = kwargs.get("project")
        return self._backend(project.id).create_representation(**kwargs)

    def rename_representation(self, **kwargs):
        project = kwargs.get("project")
        return self._backend(project.id).rename_representation(**kwargs)

    def update_representation_note(self, **kwargs):
        project = kwargs.get("project")
        return self._backend(project.id).update_representation_note(**kwargs)

    def activate_representation(self, **kwargs):
        project = kwargs.get("project")
        return self._backend(project.id).activate_representation(**kwargs)

    def deactivate_representation(self, **kwargs):
        project = kwargs.get("project")
        return self._backend(project.id).deactivate_representation(**kwargs)

    def archive_representation(self, **kwargs):
        project = kwargs.get("project")
        return self._backend(project.id).archive_representation(**kwargs)

    def project_permissions(self, project_id, principal):
        backend = self._backend(project_id)
        if backend is self.remote and self.remote is not None:
            return self.remote.project_permissions(project_id, self.remote.auth.principal)
        return self.local.project_permissions(project_id, principal)

    def project_roles(self, project_id, principal_id):
        return self._backend(project_id).project_roles(project_id, principal_id)

    def list_project_principals(
        self,
        project_id,
        *,
        include_inactive: bool = False,
    ):
        backend = self._backend(project_id)
        reader = getattr(backend, "list_project_principals", None)
        if callable(reader):
            return reader(
                project_id,
                include_inactive=include_inactive,
            )
        return backend.list_principals(include_inactive=include_inactive)

    def list_performers(self, *, include_archived: bool = False):
        local = self.local.list_performers(include_archived=include_archived)
        remote = (
            ()
            if self.remote is None
            else self.remote.list_performers(include_archived=include_archived)
        )
        merged = {str(item.id): item for item in (*local, *remote)}
        return tuple(
            sorted(merged.values(), key=lambda item: (item.name.casefold(), str(item.id)))
        )

    def list_project_performers(
        self, project_id, *, include_archived: bool = False
    ):
        return self._backend(project_id).list_performers(
            include_archived=include_archived
        )

    def project_role_revision(self, project_id, principal_id):
        return self._backend(project_id).project_role_revision(
            project_id,
            principal_id,
        )

    def assign_project_role(self, **kwargs):
        project = kwargs.get("project")
        return self._backend(project.id).assign_project_role(**kwargs)

    def revoke_project_role(self, **kwargs):
        project = kwargs.get("project")
        return self._backend(project.id).revoke_project_role(**kwargs)

    def history(self, project_id, *, as_of=None):
        return self._backend(project_id).history(project_id, as_of=as_of)

    def activity_records(self):
        local_records = self.local.activity_records()
        remote_records = () if self.remote is None else self.remote.activity_records()
        return tuple(
            sorted(
                (*local_records, *remote_records),
                key=lambda item: (item.recorded_at, item.event_id),
            )
        )

    def statistics(self, project_id, **kwargs):
        if self.is_remote_project(project_id) and self.remote is not None:
            return self.remote.statistics(project_id, **kwargs)
        from kraken_manager.infrastructure.reports import (
            ReportFilters,
            ReportGranularity,
            ReportService,
        )

        records = self.local.activity_records()
        filters = ReportFilters(
            kwargs["start"],
            kwargs["end"],
            project_ids=frozenset((str(project_id),)),
        )
        reports = ReportService()
        return reports.aggregate(records, filters), {
            granularity.value: reports.aggregate_series(
                records, filters, granularity, timezone=kwargs["timezone"]
            )
            for granularity in ReportGranularity
        }

    def latest_karakal_analysis(self, project_id, layer_id, **kwargs):
        return self._backend(project_id).latest_karakal_analysis(
            project_id, layer_id, **kwargs
        )

    def publish_karakal_analysis(self, **kwargs):
        return self._backend(kwargs.get("project_id")).publish_karakal_analysis(**kwargs)

    def record_layer_pipeline_action(self, **kwargs):
        return self._backend(kwargs.get("project_id")).record_layer_pipeline_action(**kwargs)

    def remove_layer_pipeline_action(self, **kwargs):
        return self._backend(kwargs.get("project_id")).remove_layer_pipeline_action(**kwargs)

    def list_derived_runs(self, project_id, layer_id=""):
        if self.is_remote_project(project_id):
            # Server execution is represented by durable plugin jobs; legacy
            # workstation derived-run folders are deliberately local-only.
            return ()
        return self.local.list_derived_runs(project_id, layer_id)

    def list_artifact_series(self, project_id, **kwargs):
        return self._backend(project_id).list_artifact_series(project_id, **kwargs)

    def create_artifact_series(self, **kwargs):
        project_id = kwargs.get("project_id")
        return self._backend(project_id).create_artifact_series(**kwargs)

    def artifact_stream_revision(self, project_id, series_id):
        return self._backend(project_id).artifact_stream_revision(project_id, series_id)

    def artifact_versions(self, project_id, series_id):
        return self._backend(project_id).artifact_versions(project_id, series_id)

    def active_artifact_version(self, project_id, series_id):
        return self._backend(project_id).active_artifact_version(project_id, series_id)

    def artifact_version(self, project_id, version_id):
        return self._backend(project_id).artifact_version(project_id, version_id)

    def add_managed_artifact_version(self, **kwargs):
        return self._backend(kwargs.get("project_id")).add_managed_artifact_version(**kwargs)

    def add_external_artifact_version(self, **kwargs):
        return self._backend(kwargs.get("project_id")).add_external_artifact_version(**kwargs)

    def rename_artifact_series(self, **kwargs):
        series = kwargs.get("series")
        return self._backend(series.project_id).rename_artifact_series(**kwargs)

    def archive_artifact_series(self, **kwargs):
        series = kwargs.get("series")
        return self._backend(series.project_id).archive_artifact_series(**kwargs)

    def activate_artifact_version(self, **kwargs):
        return self._backend(kwargs.get("project_id")).activate_artifact_version(**kwargs)

    def export_managed_artifact(self, project_id, version, destination):
        return self._backend(project_id).export_managed_artifact(project_id, version, destination)

    def export_artifact_version(self, project_id, version, destination):
        return self._backend(project_id).export_artifact_version(project_id, version, destination)

    def managed_artifact_path(self, project_id, version_id):
        return self._backend(project_id).managed_artifact_path(project_id, version_id)

    def read_project_blob(self, project_id, source_key):
        return self._backend(project_id).read_project_blob(project_id, source_key)

    def external_artifact_changed(self, version):
        series = None
        for project in self.list_projects(include_archived=True):
            candidate = self.artifact_version(project.id, version.id)
            if candidate is not None:
                series = project.id
                break
        if series is None:
            raise ValueError("Artifact version was not found")
        return self._backend(series).external_artifact_changed(version)

    def list_notes(self, project_id, **kwargs):
        return self._backend(project_id).list_notes(project_id, **kwargs)

    def create_note(self, **kwargs):
        return self._backend(kwargs.get("project_id")).create_note(**kwargs)

    def revise_note(self, **kwargs):
        note = kwargs.get("note")
        return self._backend(note.project_id).revise_note(**kwargs)

    def review_batches(self):
        local_batches = list(self.local.review_batches())
        remote_batches = [] if self.remote is None else list(self.remote.review_batches())
        return tuple(
            sorted(
                (*local_batches, *remote_batches),
                key=lambda item: (item.updated_at, str(item.id)),
                reverse=True,
            )
        )

    def active_review_batches(self):
        local_batches = list(self.local.active_review_batches())
        remote_batches = (
            [] if self.remote is None else list(self.remote.active_review_batches())
        )
        return tuple(
            sorted(
                (*local_batches, *remote_batches),
                key=lambda item: (item.updated_at, str(item.id)),
                reverse=True,
            )
        )

    def create_review_batch(self, **kwargs):
        return self._backend(kwargs.get("project_id")).create_review_batch(**kwargs)

    def accept_review(self, **kwargs):
        batch = kwargs.get("batch")
        return self._backend(batch.project_id).accept_review(**kwargs)

    def request_review_changes(self, **kwargs):
        batch = kwargs.get("batch")
        return self._backend(batch.project_id).request_review_changes(**kwargs)

    def cancel_review_batch(self, **kwargs):
        batch = kwargs.get("batch")
        return self._backend(batch.project_id).cancel_review_batch(**kwargs)

    def review_candidate_version_ids(self, batch):
        return self._backend(batch.project_id).review_candidate_version_ids(batch)

    def export_review_batch(self, **kwargs):
        batch = kwargs.get("batch")
        return self._backend(batch.project_id).export_review_batch(**kwargs)

    def review_return_preflight(self, **kwargs):
        source = Path(kwargs["source"]).resolve(strict=True)
        raw = (source / "kraken-review.json").read_bytes()
        if len(raw) > 16 * 1024**2:
            raise ValueError("Review manifest exceeds the size limit")
        manifest = manifest_from_json(raw.decode("utf-8"))
        if self.is_remote_project(manifest.project_id):
            if self.remote is None:
                raise RuntimeError("Kraken Server is not configured")
            return self.remote.review_return_preflight(**kwargs)
        return self.local.review_return_preflight(**kwargs)

    def commit_review_return(self, **kwargs):
        batch = kwargs.get("batch")
        return self._backend(batch.project_id).commit_review_return(**kwargs)

    def plugin_jobs(self):
        local_jobs = list(self.local.plugin_jobs())
        remote_jobs = [] if self.remote is None else list(self.remote.plugin_jobs())
        return tuple(
            sorted(
                (*local_jobs, *remote_jobs),
                key=lambda item: (getattr(item, "updated_at", None), str(item.id)),
                reverse=True,
            )
        )

    def submit_plugin_job(self, **kwargs):
        return self._backend(kwargs.get("project_id")).submit_plugin_job(**kwargs)

    def cancel_plugin_job(self, **kwargs):
        job = kwargs.get("job")
        return self._backend(job.project_id).cancel_plugin_job(**kwargs)

    def synchronize_plugin_jobs(self, **kwargs):
        local_jobs = self.local.synchronize_plugin_jobs(**kwargs)
        remote_jobs = () if self.remote is None else self.remote.synchronize_plugin_jobs(**kwargs)
        return (*local_jobs, *remote_jobs)

    def matrix_viewport(self, project_id, **kwargs):
        backend = self._backend(project_id)
        if backend is self.remote and self.remote is not None:
            return self.remote.matrix_viewport(
                project_id,
                layer_id=kwargs["layer_id"],
                x1=kwargs["x1"],
                y1=kwargs["y1"],
                x2=kwargs["x2"],
                y2=kwargs["y2"],
                lod=kwargs.get("lod", 0),
                representation_ids=kwargs.get("representation_ids", ()),
            )
        return self.local.matrix_viewport(project_id, **kwargs)

    def frame_cells(self, project_id, layer_id, representation_id, **kwargs):
        return self._backend(project_id).frame_cells(
            project_id, layer_id, representation_id, **kwargs
        )

    def frame_management_states(
        self, project_id, layer_id, representation_id, **kwargs
    ):
        return self._backend(project_id).frame_management_states(
            project_id, layer_id, representation_id, **kwargs
        )

    def create_initial_account(self, username: str, display_name: str, password: str):
        return self.local.create_initial_account(username, display_name, password)

    def login(self, username: str, password: str):
        return self.local.login(username, password)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.local, name)


def build_workspace_service(
    *,
    data_dir: Path | str | None = None,
) -> DualCatalogService:
    local = EmbeddedProjectService(data_dir=data_dir)
    from .remote_client import build_remote_service

    try:
        remote = build_remote_service(data_dir=local.data_dir)
    except (OSError, RemoteServerError, ValueError):
        LOGGER.exception("Could not configure the remote Kraken service")
        remote = None
    try:
        return DualCatalogService(local, remote)
    except RemoteServerError:
        return DualCatalogService(local)


__all__ = ["DualCatalogService", "build_workspace_service"]
