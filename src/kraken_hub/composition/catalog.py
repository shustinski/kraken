"""Compose local filesystem and remote server catalogs behind one façade."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from kraken_manager.application.ports import StorageProfile
from kraken_manager.application.backend_contract import validate_project_operations
from kraken_manager.application.project_application import ProjectApplication
from kraken_manager.application.project_operations import ProjectOperations
from kraken_manager.domain.common import ProjectId
from kraken_manager.domain.identity import Principal
from kraken_manager.domain.project import GridOrientation, Project
from kraken_manager.infrastructure.review.manifest import manifest_from_json

from kraken_hub.domain.catalog_route import storage_label

from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.infrastructure.remote.client import RemoteAuth, RemoteServerError, RemoteServerProjectService
from kraken_hub.application.workspace import REMOTE_STORAGE_PROFILE, ProjectEventWake

LOGGER = logging.getLogger(__name__)


class DualCatalogService(ProjectApplication):
    """Route catalog/project operations to local or remote backends by ownership."""

    @property
    def local_administration(self):
        return self.local

    @property
    def supports_server_connection(self) -> bool:
        return True

    @property
    def server_connection(self):
        from kraken_hub.application.workspace import ServerConnectionSummary

        if self.remote is None:
            return None
        return ServerConnectionSummary(self.remote.base_url, self.remote.auth.principal.subject)

    def __init__(
        self,
        local: EmbeddedProjectService,
        remote: RemoteServerProjectService | None = None,
    ) -> None:
        validate_project_operations(local)
        if remote is not None:
            validate_project_operations(remote)
        self.local = local
        self.remote = remote
        self.data_dir = local.data_dir
        self.profile = local.profile
        self._remote_ids: set[str] = set()
        self._wake_handlers: list[Callable[[ProjectEventWake], None]] = []
        self._catalog_handlers: list[Callable[[], None]] = []
        self._status_handlers: list[Callable[[str], None]] = []
        self._sync_requested = False
        if remote is not None:
            try:
                for project in remote.list_projects(include_archived=True):
                    self._remote_ids.add(str(project.id))
            except OSError, RemoteServerError, ValueError:
                # Server may be temporarily unreachable; catalog refresh will retry.
                LOGGER.exception("Could not load the initial remote project catalog")
                self._remote_ids = set()

    def request_project_deletion(self, *, project, reason: str = ""):
        if self.remote is None or not self.is_remote_project(project.id):
            raise ValueError("Заявка доступна только для серверного проекта")
        return self.remote.request_project_deletion(project=project, reason=reason)

    def project_deletion_requests(self, project_id):
        if self.remote is None or not self.is_remote_project(project_id):
            return []
        return self.remote.project_deletion_requests(project_id)

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
        return storage_label(remote=self.is_remote_project(project_id))

    def _backend(self, project_id: object) -> ProjectOperations:
        if self.is_remote_project(project_id):
            if self.remote is None:
                raise RemoteServerError("Kraken Server is not configured")
            return self.remote
        return self.local

    def _project_operations(self, project_id: object) -> ProjectOperations:
        return self._backend(project_id)

    def _local_operations(self, project_id: object, operation: str):
        if self.is_remote_project(project_id):
            raise RemoteServerError(
                f"Local catalog maintenance operation {operation!r} is unavailable for a server project"
            )
        return self.local

    @property
    def catalog_root(self):
        return self.local.catalog_root

    @property
    def clock(self):
        return self.local.clock

    @property
    def profiles(self):
        return self.local.profiles

    @property
    def secrets(self):
        return self.local.secrets

    @property
    def _natural_path_key(self):
        return self.local._natural_path_key

    def list_projects(self, *, include_archived: bool = False) -> tuple[Project, ...]:
        local_projects = list(self.local.list_projects(include_archived=include_archived))
        remote_projects: list[Project] = []
        if self.remote is not None:
            try:
                remote_projects = list(self.remote.list_projects(include_archived=include_archived))
                self._remote_ids = {str(project.id) for project in self.remote.list_projects(include_archived=True)}
            except OSError, RemoteServerError, ValueError:
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
        if self.is_remote_project(project_id):
            return self._project_operations(project_id).get_project(project_id, as_of=as_of)
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
                    "Kraken Server не настроен. Укажите адрес сервера и учётную запись "
                    "по умолчанию в меню Вид → Настройки."
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
                layer_template=layer_template,
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

    def start_sync(self) -> None:
        self._sync_requested = True
        if self.remote is not None:
            self.remote.start_sync()

    def stop_sync(self) -> None:
        self._sync_requested = False
        if self.remote is not None:
            self.remote.stop_sync()

    def subscribe_project(self, project_id: object) -> None:
        if self.is_remote_project(project_id) and self.remote is not None:
            self.remote.subscribe_project(project_id)

    def unsubscribe_project(self, project_id: object) -> None:
        if self.remote is not None:
            self.remote.unsubscribe_project(project_id)

    def add_wake_handler(self, handler: Callable[[ProjectEventWake], None]) -> None:
        self._wake_handlers.append(handler)
        if self.remote is not None:
            self.remote.add_wake_handler(handler)

    def add_catalog_handler(self, handler: Callable[[], None]) -> None:
        self._catalog_handlers.append(handler)
        if self.remote is not None:
            self.remote.add_catalog_handler(handler)

    def add_status_handler(self, handler: Callable[[str], None]) -> None:
        self._status_handlers.append(handler)
        if self.remote is not None:
            self.remote.add_status_handler(handler)

    def configure_remote(
        self,
        *,
        base_url: str,
        access_token: str,
        principal: Principal,
    ) -> RemoteServerProjectService:
        url = str(base_url).strip().rstrip("/")
        token = str(access_token).strip()
        if not url or not token:
            raise ValueError("Укажите адрес Kraken Server и GitLab-токен доступа")
        if not url.startswith(("http://", "https://")):
            raise ValueError("Адрес Kraken Server должен начинаться с http:// или https://")
        parsed = urlparse(url)
        if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("Network connections to Kraken Server require HTTPS")
        candidate = RemoteServerProjectService(
            url,
            auth=RemoteAuth(access_token=token, principal=principal),
            data_dir=self.local.data_dir,
        )
        return self._install_remote(candidate)

    def configure_remote_local(
        self,
        *,
        base_url: str,
        username: str,
        password: str,
    ) -> RemoteServerProjectService:
        url = str(base_url).strip().rstrip("/")
        if not url or not username.strip() or not password:
            raise ValueError("Укажите адрес Kraken Server, логин и пароль")
        if not url.startswith(("http://", "https://")):
            raise ValueError("Адрес Kraken Server должен начинаться с http:// или https://")
        parsed = urlparse(url)
        if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("Network sign-in to Kraken Server requires HTTPS")
        try:
            candidate = RemoteServerProjectService.authenticate_local(
                url,
                username=username.strip(),
                password=password,
                data_dir=self.local.data_dir,
            )
        except RemoteServerError as exc:
            if exc.status != 401:
                raise
            try:
                candidate = RemoteServerProjectService.register_local(
                    url,
                    username=username.strip(),
                    password=password,
                    display_name=username.strip(),
                    data_dir=self.local.data_dir,
                )
            except RemoteServerError as register_error:
                if register_error.status == 401:
                    raise ValueError("Пароль не подходит к уже существующей учётной записи") from register_error
                raise
        return self._install_remote(candidate)

    def _install_remote(self, candidate: RemoteServerProjectService) -> RemoteServerProjectService:
        validate_project_operations(candidate)
        authenticated = candidate.current_principal()
        if not authenticated.active:
            raise RemoteServerError("Серверная учётная запись отключена")
        candidate.auth = RemoteAuth(access_token=candidate.auth.access_token, principal=authenticated)
        projects = candidate.list_projects(include_archived=True)

        previous = self.remote
        if previous is not None:
            previous.stop_sync()
        self.remote = candidate
        self._remote_ids = {str(project.id) for project in projects}
        for handler in self._wake_handlers:
            candidate.add_wake_handler(handler)
        for handler in self._catalog_handlers:
            candidate.add_catalog_handler(handler)
        for handler in self._status_handlers:
            candidate.add_status_handler(handler)
        if self._sync_requested:
            candidate.start_sync()
        return candidate

    def project_permissions(self, project_id, principal):
        backend = self._backend(project_id)
        if backend is self.remote and self.remote is not None:
            return self.remote.project_permissions(project_id, self.remote.auth.principal)
        return self.local.project_permissions(project_id, principal)

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

    def list_principals(self, *, include_inactive: bool = False):
        people = self.local.list_principals(include_inactive=include_inactive)
        if self.remote is not None:
            people = (*people, *self.remote.list_principals(include_inactive=include_inactive))
        merged = {str(person.id): person for person in people}
        return tuple(sorted(merged.values(), key=lambda person: (person.display_name.casefold(), str(person.id))))

    def list_role_candidates(self, project_id: ProjectId | str) -> tuple[Principal, ...]:
        """Role assignment includes users who do not yet belong to the project."""
        return self._backend(project_id).list_principals()

    def list_performers(self, *, include_archived: bool = False):
        local = self.local.list_performers(include_archived=include_archived)
        remote = () if self.remote is None else self.remote.list_performers(include_archived=include_archived)
        merged = {str(item.id): item for item in (*local, *remote)}
        return tuple(sorted(merged.values(), key=lambda item: (item.name.casefold(), str(item.id))))

    def list_project_performers(self, project_id, *, include_archived: bool = False):
        return self._backend(project_id).list_performers(include_archived=include_archived)

    def run_interactive_job(self, project_id, job_id, *, registry):
        return self._backend(project_id).run_interactive_job(project_id, job_id, registry=registry)

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
        from kraken_manager.application.reporting import (
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
            granularity.value: reports.aggregate_series(records, filters, granularity, timezone=kwargs["timezone"])
            for granularity in ReportGranularity
        }

    def list_derived_runs(self, project_id, layer_id=""):
        if self.is_remote_project(project_id):
            # Server execution is represented by durable plugin jobs; legacy
            # workstation derived-run folders are deliberately local-only.
            return ()
        return self.local.list_derived_runs(project_id, layer_id)

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

    def work_orders(self):
        local_orders = list(self.local.work_orders())
        remote_orders = [] if self.remote is None else list(self.remote.work_orders())
        return tuple(sorted((*local_orders, *remote_orders), key=lambda item: (item.updated_at, item.id), reverse=True))

    def dispatch_stitch(self, **kwargs):
        return self._backend(kwargs["project_id"]).dispatch_stitch(**kwargs)

    def refresh_stitch(self, **kwargs):
        return self._backend(kwargs["project_id"]).refresh_stitch(**kwargs)

    def tick_layer_pipeline(self, **kwargs):
        return self._backend(kwargs["project_id"]).tick_layer_pipeline(**kwargs)

    def server_analysis_runs(self, project_id):
        return self._backend(project_id).server_analysis_runs(project_id)

    def request_server_analysis_map(self, project_id, run_id, frame_id, **kwargs):
        return self._backend(project_id).request_server_analysis_map(project_id, run_id, frame_id, **kwargs)

    def server_analysis_map_bytes(self, project_id, run_id, frame_id):
        return self._backend(project_id).server_analysis_map_bytes(project_id, run_id, frame_id)

    def server_analysis_run(self, project_id, run_id):
        return self._backend(project_id).server_analysis_run(project_id, run_id)

    def server_analysis_frames(self, project_id, run_id, metric_key):
        return self._backend(project_id).server_analysis_frames(project_id, run_id, metric_key)

    def create_server_analysis(self, project_id, payload, **kwargs):
        return self._backend(project_id).create_server_analysis(project_id, payload, **kwargs)

    def command_server_analysis(self, project_id, run_id, operation, **kwargs):
        return self._backend(project_id).command_server_analysis(project_id, run_id, operation, **kwargs)

    def create_quality_from_analysis(self, **kwargs):
        backend = self._backend(kwargs["project_id"])
        if backend is self.remote:
            kwargs["principal"] = self.remote.auth.principal
        return backend.create_quality_from_analysis(**kwargs)

    def recheck_quality_from_analysis(self, **kwargs):
        backend = self._backend(kwargs["project_id"])
        if backend is self.remote:
            kwargs["principal"] = self.remote.auth.principal
        return backend.recheck_quality_from_analysis(**kwargs)

    def create_quality_issue(self, **kwargs):
        backend = self._backend(kwargs["project_id"])
        if backend is self.remote:
            kwargs["principal"] = self.remote.auth.principal
        return backend.create_quality_issue(**kwargs)

    def mutate_quality_issue(self, **kwargs):
        backend = self._backend(kwargs["project_id"])
        if backend is self.remote:
            kwargs["principal"] = self.remote.auth.principal
        return backend.mutate_quality_issue(**kwargs)

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
        remote_batches = [] if self.remote is None else list(self.remote.active_review_batches())
        return tuple(
            sorted(
                (*local_batches, *remote_batches),
                key=lambda item: (item.updated_at, str(item.id)),
                reverse=True,
            )
        )

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

    def synchronize_plugin_jobs(self, **kwargs):
        local_jobs = self.local.synchronize_plugin_jobs(**kwargs)
        remote_jobs = () if self.remote is None else self.remote.synchronize_plugin_jobs(**kwargs)
        return (*local_jobs, *remote_jobs)

    def create_initial_account(self, username: str, display_name: str, password: str):
        return self.local.create_initial_account(username, display_name, password)

    def login(self, username: str, password: str):
        return self.local.login(username, password)

    def archive_performer(self, performer_id):
        return self.local.archive_performer(performer_id)

    def attach_project(self, project_directory, *, principal=None, take_ownership=False):
        return self.local.attach_project(project_directory, principal=principal, take_ownership=take_ownership)

    def begin_derived_run(self, *, project_id, layer_id, layer_name, kind, plugin_id, operation, principal=None):
        return self._local_operations(project_id, "begin_derived_run").begin_derived_run(
            project_id=project_id,
            layer_id=layer_id,
            layer_name=layer_name,
            kind=kind,
            plugin_id=plugin_id,
            operation=operation,
            principal=principal,
        )

    def bind_project_workspace(self, *, principal, project, source_root, derived_root, idempotency_key):
        return self._local_operations(project.id, "bind_project_workspace").bind_project_workspace(
            principal=principal,
            project=project,
            source_root=source_root,
            derived_root=derived_root,
            idempotency_key=idempotency_key,
        )

    def create_manual_performer(self, *, name, color):
        return self.local.create_manual_performer(name=name, color=color)

    def delete_project(self, *, principal, project, confirmation_name):
        return self._local_operations(project.id, "delete_project").delete_project(
            principal=principal, project=project, confirmation_name=confirmation_name
        )

    def export_backup(self, project_id, destination, *, principal=None):
        return self._local_operations(project_id, "export_backup").export_backup(
            project_id, destination, principal=principal
        )

    def fail_derived_run(self, *, principal, project_id, run_id, error):
        return self._local_operations(project_id, "fail_derived_run").fail_derived_run(
            principal=principal, project_id=project_id, run_id=run_id, error=error
        )

    def import_agent_publications(self, *, principal, publications, staging_root):
        return self.local.import_agent_publications(
            principal=principal, publications=publications, staging_root=staging_root
        )

    def import_agent_result(self, *, principal, result_payload, staging_root, confirm_partial=False):
        return self.local.import_agent_result(
            principal=principal,
            result_payload=result_payload,
            staging_root=staging_root,
            confirm_partial=confirm_partial,
        )

    def import_backup(self, bundle_root, *, principal=None, take_ownership=False):
        return self.local.import_backup(bundle_root, principal=principal, take_ownership=take_ownership)

    def import_stitch_publication(self, *, project_id, principal, publication, source_path):
        return self._local_operations(project_id, "import_stitch_publication").import_stitch_publication(
            project_id=project_id, principal=principal, publication=publication, source_path=source_path
        )

    def publish_workspace_plugin_result(self, *, principal, project, layer, result):
        return self._local_operations(project.id, "publish_workspace_plugin_result").publish_workspace_plugin_result(
            principal=principal, project=project, layer=layer, result=result
        )

    def resolve_session(self, token):
        return self.local.resolve_session(token)

    def scan_integrity(self):
        return self.local.scan_integrity()

    def update_performer(self, *, performer_id, name, color):
        return self.local.update_performer(performer_id=performer_id, name=name, color=color)


def build_workspace_service(
    *,
    data_dir: Path | str | None = None,
) -> DualCatalogService:
    local = EmbeddedProjectService(data_dir=data_dir)
    from kraken_hub.infrastructure.remote.client import build_remote_service

    try:
        remote = build_remote_service(data_dir=local.data_dir)
    except OSError, RemoteServerError, ValueError:
        LOGGER.exception("Could not configure the remote Kraken service")
        remote = None
    try:
        return DualCatalogService(local, remote)
    except RemoteServerError:
        return DualCatalogService(local)


__all__ = ["DualCatalogService", "build_workspace_service"]
