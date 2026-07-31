"""Desktop composition root for authenticated local file projects."""

from __future__ import annotations

import json
import hashlib
import mimetypes
import os
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from threading import Thread
from uuid import NAMESPACE_URL, uuid5

from kraken_core.safe_files import ensure_regular_directory, open_regular_read
from kraken_core.plugin_protocol import (
    PluginResultManifest,
    PluginResultPublicationV2,
    WorkspacePluginResultV1,
    parse_plugin_result_json,
)
from kraken_core.frame_matrix import StoreNamespace

from kraken_manager.application.dto import (
    AddArtifactVersionCommand,
    ActivateArtifactVersionCommand,
    AddExternalArtifactVersionCommand,
    ActivateRepresentationCommand,
    AssignProjectRoleCommand,
    ArchiveLayerCommand,
    ArchiveProjectCommand,
    ArchiveArtifactSeriesCommand,
    ArchiveRepresentationCommand,
    AcceptReviewCommand,
    CancelReviewBatchCommand,
    CommitReviewReturnCommand,
    CommandContext,
    CreateLayerCommand,
    CreateArtifactSeriesCommand,
    CreateNoteCommand,
    CreateProjectCommand,
    CreateRepresentationCommand,
    CreateReviewBatchCommand,
    DeactivateRepresentationCommand,
    DryRunReviewReturnCommand,
    ExportReviewPackageCommand,
    RenameLayerCommand,
    RenameArtifactSeriesCommand,
    RenameProjectCommand,
    RenameRepresentationCommand,
    ReorderLayerCommand,
    ReorderLayersCommand,
    RestoreProjectCommand,
    ReviseNoteCommand,
    RetryPluginJobCommand,
    SubmitPluginJobCommand,
    CancelPluginJobCommand,
    SynchronizePluginJobCommand,
    ImportPluginResultCommand,
    RequestReviewChangesCommand,
    RevokeProjectRoleCommand,
    UpdateRepresentationNoteCommand,
)
from kraken_manager.application.imports import ImportMappingMode, ImportPlan, ImportPlanner, ImportSource
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.workspace import (
    DerivedRun,
    DerivedRunKind,
    ImageConversionSettings,
    LayerFileBinding,
    LayerSourceMode,
    LayerSourceScan,
    ProjectWorkspaceBinding,
    WorkspaceValidationError,
    layer_binding_to_dict,
    map_frame_positions,
    project_workspace_to_dict,
    validate_workspace_name,
)
from kraken_manager.application.ports import StorageProfile
from kraken_manager.application.use_cases import (
    AddArtifactVersionHandler,
    CreateArtifactSeriesHandler,
    CreateLayerHandler,
    CreateProjectHandler,
    CreateRepresentationHandler,
)
from kraken_manager.application.lifecycle import (
    ArchiveLayerHandler,
    ArchiveProjectHandler,
    RenameLayerHandler,
    RenameProjectHandler,
    ReorderLayerHandler,
    ReorderLayersHandler,
    RestoreProjectHandler,
)
from kraken_manager.application.acl import AssignProjectRoleHandler, RevokeProjectRoleHandler
from kraken_manager.application.artifact_lifecycle import (
    ActivateArtifactVersionHandler,
    AddExternalArtifactVersionHandler,
    ArchiveArtifactSeriesHandler,
    CreateNoteHandler,
    RenameArtifactSeriesHandler,
    ReviseNoteHandler,
)
from kraken_manager.application.representation_lifecycle import (
    ActivateRepresentationHandler,
    ArchiveRepresentationHandler,
    DeactivateRepresentationHandler,
    RenameRepresentationHandler,
    UpdateRepresentationNoteHandler,
)
from kraken_manager.application.review_workflow import (
    AcceptReviewHandler,
    CancelReviewBatchHandler,
    CommitReviewReturnHandler,
    CreateReviewBatchHandler,
    DryRunReviewReturnHandler,
    ExportReviewPackageHandler,
    RequestReviewChangesHandler,
)
from kraken_manager.application.plugin_jobs import (
    CancelPluginJobHandler,
    ImportPluginResultHandler,
    RetryPluginJobHandler,
    SubmitPluginJobHandler,
    SynchronizePluginJobHandler,
)
from kraken_manager.domain.artifacts import (
    ArtifactScope,
    ArtifactSeries,
    ArtifactVersion,
    NoteRevision,
    deterministic_frame_series_id,
)
from kraken_manager.domain.common import (
    ArtifactSeriesId,
    ArtifactVersionId,
    FrameId,
    LayerId,
    PerformerId,
    PluginJobId,
    PrincipalId,
    ProjectId,
    RepresentationId,
    ReviewBatchId,
    new_uuid,
)
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
from kraken_manager.domain.identity import (
    Permission,
    Performer,
    Principal,
    ProjectRole,
    ProjectRoleAssignment,
    ROLE_PERMISSIONS,
    SystemRole,
)
from kraken_manager.domain.project import (
    GridOrientation,
    Layer,
    LayerType,
    Project,
    Representation,
    RepresentationKind,
    RepresentationPurpose,
)
from kraken_manager.domain.selection import FrameRowRange, FrameSelectionV1
from kraken_manager.domain.workflows import (
    PluginFrameOutcome,
    PluginFrameResultV1,
    PluginInputV1,
    PluginResultManifestV1,
    PluginResultOutcome,
    ReviewBatch,
    ReviewItem,
)
from kraken_manager.infrastructure.auth.identity_store import LocalIdentityAclStore
from kraken_manager.infrastructure.auth.local import LocalAccountStore, ScryptPasswordHasher
from kraken_manager.infrastructure.auth.performer_store import LocalSQLitePerformerStore
from kraken_manager.infrastructure.filesystem import (
    FileProjectLayout,
    FilesystemEventStore,
    LocalProjectUnitOfWorkFactory,
    SQLiteProjectionStore,
    filesystem_storage_profile,
)
from kraken_manager.infrastructure.blob.filesystem import FilesystemBlobStore
from kraken_manager.infrastructure.migration import (
    BundleVerifier,
    CanonicalBundleExporter,
    CanonicalBundleImporter,
    KrakenMigrationBundleV1,
    load_bundle_manifest,
)
from kraken_manager.infrastructure.projections import rebuild_filesystem_index
from kraken_manager.infrastructure.reports import ActivityRecord
from kraken_manager.infrastructure.review import ReviewPackageReader, ReviewPackageWriter
from kraken_manager.infrastructure.review.crypto import Ed25519KeyPair
from kraken_manager.infrastructure.plugin import (
    AgentStagingResultContentReader,
    domain_result_from_transport,
)
from kraken_manager.infrastructure.workspace_files import WorkspaceFileService, WorkspaceRegistry

from .secret_store import KeyringSecretStore


def default_data_dir() -> Path:
    override = os.environ.get("KRAKEN_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".kraken").resolve()


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class DesktopStorageProfiles:
    def __init__(self, *profiles: StorageProfile) -> None:
        self._profiles = {profile.id: profile for profile in profiles}

    def get(self, profile_id: str) -> StorageProfile | None:
        return self._profiles.get(profile_id)

    def list(self) -> tuple[StorageProfile, ...]:
        return tuple(self._profiles.values())


@dataclass(frozen=True, slots=True)
class DesktopSession:
    token: str
    principal: Principal
    expires_at: str


@dataclass(frozen=True, slots=True)
class ManagedImportResult:
    plan: ImportPlan
    versions: tuple[ArtifactVersion, ...]


@dataclass(frozen=True, slots=True)
class IntegrityScanResult:
    projects: int
    events: int
    blobs: int
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class FrameCellSnapshot:
    x: int
    y: int
    status: str
    frame_id: str
    artifact_version_id: str
    sha256: str
    modified_at: str = ""
    performer_color: str = ""
    performer_initials: str = ""
    review_status: str = "not_checked"
    quality: float | None = None


@dataclass(frozen=True, slots=True)
class FrameManagementState:
    frame_id: str
    artifact_version_id: str
    modified_at: str
    performer_color: str
    performer_initials: str
    review_status: str


@dataclass(frozen=True, slots=True)
class KarakalAnalysisRun:
    run_id: str
    project_id: str
    layer_id: str
    publication_sequence: int
    created_at: str
    frame_confidence: dict[str, float]
    report: dict[str, object]
    parameters: dict[str, object]
    plugin_version: str


@dataclass(frozen=True, slots=True)
class ProjectDeletionResult:
    project_id: str
    catalog_cache_removed: bool
    thumbnail_cache_removed: bool
    staging_directories_removed: int


class EmbeddedProjectService:
    """Autonomous application service for machine-local file projects."""

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        secret_store: object | None = None,
    ) -> None:
        self.data_dir = Path(data_dir or default_data_dir()).resolve()
        self.catalog_root = self.data_dir / "catalog"
        self.catalog_root.mkdir(parents=True, exist_ok=True)
        self.accounts = LocalAccountStore(self.data_dir / "accounts.sqlite3", ScryptPasswordHasher())
        self.identities = LocalIdentityAclStore(self.data_dir / "identity.sqlite3")
        self.performers = LocalSQLitePerformerStore(self.data_dir / "identity.sqlite3")
        self.profile = filesystem_storage_profile(str(self.catalog_root))
        self.workspace_registry = WorkspaceRegistry(self.catalog_root)
        self.workspace_files = WorkspaceFileService(self.workspace_registry)
        self.default_source_root = self.data_dir / "workspace-source"
        self.default_derived_root = self.data_dir / "workspace-derived"
        self.default_source_root.mkdir(parents=True, exist_ok=True)
        self.default_derived_root.mkdir(parents=True, exist_ok=True)
        self._external_source_indexes: dict[
            tuple[str, str, int], tuple[Path | None, ...]
        ] = {}
        self.profiles = DesktopStorageProfiles(self.profile)
        self.clock = SystemClock()
        self.secrets = secret_store or KeyringSecretStore()

    @property
    def has_accounts(self) -> bool:
        return self.accounts.account_count() > 0

    def create_initial_account(self, username: str, display_name: str, password: str) -> DesktopSession:
        """Create and sign in the first workstation-local account."""
        if self.has_accounts:
            raise ValueError("A local account already exists")
        account = self.accounts.create_account(username, display_name, password)
        principal = Principal.local(
            subject=account.username,
            display_name=account.display_name,
            principal_id=account.account_id,
        )
        self.identities.save(principal)
        session = self.accounts.authenticate(account.username, password)
        if session is None:  # pragma: no cover - the freshly created account is enabled
            raise RuntimeError("The new local account could not be signed in")
        return DesktopSession(session.token, principal, session.expires_at)

    def list_performers(self, *, include_archived: bool = False) -> tuple[Performer, ...]:
        return self.performers.list(include_archived=include_archived)

    def create_manual_performer(self, *, name: str, color: str) -> Performer:
        return self.performers.create(Performer.create(name=name, color=color))

    def update_performer(
        self,
        *,
        performer_id: PerformerId | str,
        name: str,
        color: str,
    ) -> Performer:
        performer = self.performers.get(PerformerId(str(performer_id)))
        if performer is None:
            raise ValueError(f"Performer {performer_id} was not found")
        return self.performers.update(replace(performer, name=name, color=color))

    def archive_performer(self, performer_id: PerformerId | str) -> Performer:
        return self.performers.archive(PerformerId(str(performer_id)))

    def project_roles(
        self, project_id: ProjectId | str, principal_id: PrincipalId | str
    ) -> frozenset[ProjectRole]:
        return self.identities.roles_for(ProjectId(str(project_id)), PrincipalId(str(principal_id)))

    def list_principals(self, *, include_inactive: bool = False) -> tuple[Principal, ...]:
        return self.identities.list(include_inactive=include_inactive)

    def project_role_revision(
        self,
        project_id: ProjectId | str,
        principal_id: PrincipalId | str,
    ) -> int:
        return FilesystemEventStore(
            self.catalog_root,
            str(project_id),
        ).current_revision(f"acl:{project_id}:{principal_id}")

    def project_permissions(
        self,
        project_id: ProjectId | str,
        principal: Principal,
    ) -> frozenset[Permission]:
        if SystemRole.SERVER_ADMIN in principal.system_roles:
            return frozenset(Permission)
        permissions: set[Permission] = set()
        for role in self.project_roles(project_id, principal.id):
            permissions.update(ROLE_PERMISSIONS[role])
        return frozenset(permissions)

    def assign_project_role(
        self,
        *,
        principal: Principal,
        project: Project,
        target_principal_id: PrincipalId,
        role: ProjectRole,
        expected_revision: int,
        idempotency_key: str,
    ) -> frozenset[ProjectRole]:
        return AssignProjectRoleHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            AssignProjectRoleCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id,
                principal_id=target_principal_id,
                role=role,
                expected_revision=expected_revision,
            )
        )

    def revoke_project_role(
        self,
        *,
        principal: Principal,
        project: Project,
        target_principal_id: PrincipalId,
        role: ProjectRole,
        expected_revision: int,
        idempotency_key: str,
    ) -> frozenset[ProjectRole]:
        if role is ProjectRole.OWNER:
            owners = tuple(
                candidate
                for candidate in self.identities.list()
                if ProjectRole.OWNER
                in self.identities.roles_for(project.id, candidate.id)
            )
            if len(owners) <= 1 and any(
                candidate.id == target_principal_id for candidate in owners
            ):
                raise ValueError("Нельзя отозвать роль последнего владельца проекта.")
        return RevokeProjectRoleHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            RevokeProjectRoleCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id,
                principal_id=target_principal_id,
                role=role,
                expected_revision=expected_revision,
            )
        )

    def login(self, username: str, password: str) -> DesktopSession | None:
        session = self.accounts.authenticate(username, password)
        if session is None:
            return None
        principal = self.identities.get_by_external_key(f"local:{session.account.username}")
        if principal is None:
            principal = Principal.local(
                subject=session.account.username,
                display_name=session.account.display_name,
                principal_id=session.account.account_id,
            )
            self.identities.save(principal)
        return DesktopSession(session.token, principal, session.expires_at)

    def resolve_session(self, token: str) -> Principal | None:
        account = self.accounts.resolve_session(token)
        if account is None:
            return None
        return self.identities.get(PrincipalId(account.account_id))

    def _uow(self, project_id: str) -> LocalProjectUnitOfWorkFactory:
        return LocalProjectUnitOfWorkFactory(
            self.catalog_root,
            project_id,
            identities=self.identities,
            acl=self.identities,
        )

    def _record_workspace_event(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        stream_id: str,
        event_type: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> EventEnvelope:
        store = FilesystemEventStore(self.catalog_root, str(project_id))
        revision = store.current_revision(stream_id)
        event = EventEnvelope.create(
            stream_id=stream_id,
            project_id=ProjectId(str(project_id)),
            revision=revision + 1,
            event_type=event_type,
            payload=payload,
            actor=ActorSnapshot.from_principal(principal),
            program=ProgramSnapshot("Kraken workspace", "1"),
            idempotency_key=idempotency_key,
        )
        store.append(stream_id, expected_revision=revision, events=(event,))
        return event

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
    ) -> Project:
        safe_name = validate_workspace_name(name, field_name="Название проекта")
        command = CreateProjectCommand(
            context=CommandContext(actor=principal, idempotency_key=idempotency_key),
            name=safe_name,
            width=width,
            height=height,
            orientation=orientation,
            storage_profile_id=self.profile.id,
        )
        assert command.project_id is not None
        project_id = str(command.project_id)
        binding = self.workspace_files.create_project(
            project_id=project_id,
            project_name=safe_name,
            source_root=source_root or self.default_source_root,
            derived_root=derived_root or self.default_derived_root,
        )

        try:
            project = CreateProjectHandler(self._uow(project_id), self.profiles, self.clock)(command)
        except Exception:
            self.workspace_files.remove_project_layout(binding)
            self.workspace_registry.remove_project(project_id)
            raise
        self._record_workspace_event(
            principal=principal,
            project_id=project.id,
            stream_id=f"project-workspace:{project.id}",
            event_type="ProjectWorkspaceBoundV1",
            payload=project_workspace_to_dict(binding),
            idempotency_key=f"{idempotency_key}:workspace",
        )
        if layer_template:
            for order, layer_type in enumerate(LayerType, start=1):
                self.create_layer(
                    principal=principal,
                    project=project,
                    name=layer_type.value.capitalize(),
                    layer_type=layer_type,
                    order=order,
                    idempotency_key=f"{idempotency_key}:layer:{layer_type.value}",
                )
                project = self.get_project(project.id) or project
        return project

    def project_workspace(self, project_id: ProjectId | str) -> ProjectWorkspaceBinding | None:
        return self.workspace_registry.get_project(str(project_id))

    def layer_file_binding(
        self, project_id: ProjectId | str, layer_id: LayerId | str
    ) -> LayerFileBinding | None:
        return self.workspace_registry.get_layer(str(project_id), str(layer_id))

    def scan_layer_source(
        self, directory: Path | str, *, maximum_frames: int
    ) -> LayerSourceScan:
        return self.workspace_files.scan(directory, maximum_frames=maximum_frames)

    def create_layer_from_disk(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        layer_type: LayerType,
        order: int,
        scan: LayerSourceScan,
        conversion: ImageConversionSettings,
        idempotency_key: str,
        progress=None,
        cancelled=None,
    ) -> tuple[Layer, LayerFileBinding, Representation]:
        workspace = self.project_workspace(project.id)
        if workspace is None:
            raise WorkspaceValidationError("project has no two-root workspace")
        safe_name = validate_workspace_name(name, field_name="Название слоя")
        reserved_layer_id = LayerId(new_uuid())
        binding = self.workspace_files.import_layer(
            project=workspace,
            layer_id=str(reserved_layer_id),
            layer_name=safe_name,
            scan=scan,
            conversion=conversion,
            progress=progress,
            cancelled=cancelled,
        )
        try:
            current_project = self.get_project(project.id) or project
            layer = self.create_layer(
                principal=principal,
                project=current_project,
                name=safe_name,
                layer_type=layer_type,
                order=order,
                idempotency_key=idempotency_key,
                layer_id=reserved_layer_id,
            )
            latest_project = self.get_project(project.id) or current_project
            representation = self.create_representation(
                principal=principal,
                project=latest_project,
                layer=layer,
                name="Исходные изображения",
                kind=RepresentationKind.IMAGE,
                idempotency_key=f"{idempotency_key}:representation",
                source=binding.image_directory,
                active=True,
                purpose=RepresentationPurpose.SOURCE,
            )
            self._record_workspace_event(
                principal=principal,
                project_id=project.id,
                stream_id=f"layer-files:{layer.id}",
                event_type="LayerFileBoundV1",
                payload=layer_binding_to_dict(binding),
                idempotency_key=f"{idempotency_key}:file-binding",
            )
            return layer, binding, representation
        except Exception:
            stored = self.layer_file_binding(project.id, reserved_layer_id)
            if stored is not None:
                self.workspace_files.remove_managed_layer_layout(stored, workspace)
                self.workspace_registry.remove_layer(str(project.id), str(reserved_layer_id))
            current_project = self.get_project(project.id) or project
            current_layer = next(
                (item for item in self.list_layers(project.id) if item.id == reserved_layer_id),
                None,
            )
            if current_layer is not None:
                try:
                    self.archive_layer(
                        principal=principal,
                        project=current_project,
                        layer=current_layer,
                        idempotency_key=f"{idempotency_key}:compensate",
                    )
                except Exception:
                    pass
            raise

    def create_external_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        layer_type: LayerType,
        order: int,
        image_directory: Path | str,
        ssc_directory: Path | str | None,
        prv_directory: Path | str | None,
        idempotency_key: str,
    ) -> tuple[Layer, LayerFileBinding, Representation]:
        safe_name = validate_workspace_name(name, field_name="Название слоя")
        reserved_layer_id = LayerId(new_uuid())
        binding = self.workspace_files.bind_external_layer(
            project_id=str(project.id),
            layer_id=str(reserved_layer_id),
            layer_name=safe_name,
            image_directory=image_directory,
            ssc_directory=ssc_directory,
            prv_directory=prv_directory,
            maximum_frames=project.frame_count,
        )
        try:
            current_project = self.get_project(project.id) or project
            layer = self.create_layer(
                principal=principal,
                project=current_project,
                name=safe_name,
                layer_type=layer_type,
                order=order,
                idempotency_key=idempotency_key,
                layer_id=reserved_layer_id,
            )
            latest_project = self.get_project(project.id) or current_project
            representation = self.create_representation(
                principal=principal,
                project=latest_project,
                layer=layer,
                name="Исходные изображения",
                kind=RepresentationKind.IMAGE,
                idempotency_key=f"{idempotency_key}:representation",
                source=binding.image_directory,
                active=True,
                purpose=RepresentationPurpose.SOURCE,
            )
            self._record_workspace_event(
                principal=principal,
                project_id=project.id,
                stream_id=f"layer-files:{layer.id}",
                event_type="LayerFileBoundV1",
                payload=layer_binding_to_dict(binding),
                idempotency_key=f"{idempotency_key}:file-binding",
            )
            return layer, binding, representation
        except Exception:
            self.workspace_registry.remove_layer(str(project.id), str(reserved_layer_id))
            current_project = self.get_project(project.id) or project
            current_layer = next(
                (item for item in self.list_layers(project.id) if item.id == reserved_layer_id),
                None,
            )
            if current_layer is not None:
                try:
                    self.archive_layer(
                        principal=principal,
                        project=current_project,
                        layer=current_layer,
                        idempotency_key=f"{idempotency_key}:compensate",
                    )
                except Exception:
                    pass
            raise

    def begin_derived_run(
        self,
        *,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        layer_name: str,
        kind: DerivedRunKind,
        plugin_id: str,
        operation: str,
        principal: Principal | None = None,
    ) -> DerivedRun:
        run = self.workspace_files.begin_run(
            project_id=str(project_id),
            layer_id=str(layer_id),
            layer_name=layer_name,
            kind=kind,
            plugin_id=plugin_id,
            operation=operation,
        )
        if principal is not None:
            self._record_workspace_event(
                principal=principal,
                project_id=project_id,
                stream_id=f"derived-run:{run.run_id}",
                event_type="DerivedRunStartedV1",
                payload={
                    "run_id": run.run_id,
                    "layer_id": run.layer_id,
                    "kind": run.kind.value,
                    "state": run.state.value,
                    "path": run.path,
                    "plugin_id": run.plugin_id,
                    "operation": run.operation,
                    "created_at": run.created_at,
                },
                idempotency_key=f"derived-run:{run.run_id}:start",
            )
        return run

    def list_derived_runs(
        self,
        project_id: ProjectId | str,
        layer_id: LayerId | str = "",
    ) -> tuple[DerivedRun, ...]:
        return self.workspace_registry.list_runs(
            str(project_id),
            str(layer_id) if layer_id else "",
        )

    def delete_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        confirmation_name: str,
        idempotency_key: str,
    ) -> Layer:
        if confirmation_name != layer.name:
            raise WorkspaceValidationError("layer name confirmation does not match")
        workspace = self.project_workspace(project.id)
        binding = self.layer_file_binding(project.id, layer.id)
        if workspace is None or binding is None:
            raise WorkspaceValidationError("layer has no two-root file binding")
        active_runs = [
            run
            for run in self.workspace_registry.list_runs(str(project.id), str(layer.id))
            if run.state.value == "running"
        ]
        if active_runs:
            raise WorkspaceValidationError("layer has an active plugin run")
        stage = self.workspace_files.stage_layer_deletion(
            project=workspace,
            binding=binding,
            delete_id=idempotency_key,
        )
        try:
            current_project = self.get_project(project.id) or project
            current_layer = next(
                (item for item in self.list_layers(project.id) if item.id == layer.id),
                None,
            )
            if current_layer is None:
                raise WorkspaceValidationError("layer is no longer available")
            archived = self.archive_layer(
                principal=principal,
                project=current_project,
                layer=current_layer,
                idempotency_key=idempotency_key,
            )
            self._record_workspace_event(
                principal=principal,
                project_id=project.id,
                stream_id=f"layer-files:{layer.id}",
                event_type="LayerDeletedV1",
                payload={
                    "layer_id": str(layer.id),
                    "layer_name": layer.name,
                    "delete_id": idempotency_key,
                    "managed_files": binding.mode.value == "managed_copy",
                },
                idempotency_key=f"{idempotency_key}:files",
            )
            self.workspace_registry.remove_layer(str(project.id), str(layer.id))
        except Exception:
            self.workspace_files.rollback_layer_deletion(stage)
            raise
        Thread(
            target=self.workspace_files.purge_layer_deletion,
            args=(stage,),
            name=f"kraken-layer-trash-{idempotency_key}",
            daemon=True,
        ).start()
        return archived

    def publish_workspace_plugin_result(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        result: WorkspacePluginResultV1,
    ) -> tuple[DerivedRun, Representation | None]:
        run = self.workspace_registry.get_run(str(project.id), result.run_id)
        if run is None or run.layer_id != str(layer.id):
            raise WorkspaceValidationError("workspace result does not belong to this layer")
        if run.plugin_id != result.plugin_id or run.operation != result.operation:
            raise WorkspaceValidationError("workspace result provenance does not match the run")
        if result.outcome != "succeeded":
            raise WorkspaceValidationError(
                f"plugin result is not publishable: {result.outcome}"
            )
        published = self.workspace_files.publish_run(
            project_id=str(project.id),
            run_id=run.run_id,
            output_directory=result.output_directory,
            provenance=dict(result.provenance),
        )
        self._record_workspace_event(
            principal=principal,
            project_id=project.id,
            stream_id=f"derived-run:{published.run_id}",
            event_type="DerivedRunPublishedV1",
            payload={
                "run_id": published.run_id,
                "layer_id": published.layer_id,
                "kind": published.kind.value,
                "state": published.state.value,
                "path": published.path,
                "plugin_id": published.plugin_id,
                "operation": published.operation,
                "provenance": dict(published.provenance),
            },
            idempotency_key=f"derived-run:{published.run_id}:publish",
        )
        representation: Representation | None = None
        if published.kind in {DerivedRunKind.VECTOR, DerivedRunKind.RESULT}:
            current_project = self.get_project(project.id) or project
            current_layer = next(
                (item for item in self.list_layers(project.id) if item.id == layer.id),
                layer,
            )
            kind = (
                RepresentationKind.VECTOR
                if published.kind is DerivedRunKind.VECTOR
                else RepresentationKind.IMAGE
            )
            representation = self.create_representation(
                principal=principal,
                project=current_project,
                layer=current_layer,
                name=f"{layer.name} · {published.kind.value} · {published.run_id[:8]}",
                kind=kind,
                idempotency_key=f"workspace-run:{published.run_id}",
                source=published.path,
                active=True,
                purpose=(
                    RepresentationPurpose.SOURCE
                    if kind is RepresentationKind.VECTOR
                    else RepresentationPurpose.BINARY
                ),
            )
        return published, representation

    def fail_derived_run(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        run_id: str,
        error: str,
    ) -> DerivedRun:
        failed = self.workspace_files.fail_run(
            project_id=str(project_id),
            run_id=run_id,
            error=error,
        )
        self._record_workspace_event(
            principal=principal,
            project_id=project_id,
            stream_id=f"derived-run:{failed.run_id}",
            event_type="DerivedRunFailedV1",
            payload={
                "run_id": failed.run_id,
                "layer_id": failed.layer_id,
                "kind": failed.kind.value,
                "state": failed.state.value,
                "plugin_id": failed.plugin_id,
                "operation": failed.operation,
                "error": str(error)[:10_000],
            },
            idempotency_key=f"derived-run:{failed.run_id}:failed",
        )
        return failed

    def create_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        layer_type: LayerType,
        order: int,
        idempotency_key: str,
        layer_id: LayerId | str | None = None,
    ) -> Layer:
        command = CreateLayerCommand(
            context=CommandContext(actor=principal, idempotency_key=idempotency_key),
            project_id=project.id,
            name=name,
            type=layer_type,
            order=order,
            expected_project_revision=project.revision,
            layer_id=LayerId(str(layer_id)) if layer_id is not None else LayerId(new_uuid()),
        )
        return CreateLayerHandler(self._uow(str(project.id)), self.profiles, self.clock)(command)

    def create_representation(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        name: str,
        kind: RepresentationKind,
        idempotency_key: str,
        note: str = "",
        source: str | None = None,
        source_image_representation_id=None,
        active: bool = False,
        purpose: RepresentationPurpose = RepresentationPurpose.SOURCE,
    ) -> Representation:
        current_layer = next((item for item in self.list_layers(project.id) if item.id == layer.id), None)
        if current_layer is None:
            raise ValueError("Layer is no longer available")
        command = CreateRepresentationCommand(
            context=CommandContext(actor=principal, idempotency_key=idempotency_key),
            project_id=project.id,
            layer_id=current_layer.id,
            name=name,
            kind=kind,
            expected_layer_revision=current_layer.revision,
            note=note,
            source=source,
            source_image_representation_id=source_image_representation_id,
            active=active,
            purpose=purpose,
        )
        return CreateRepresentationHandler(self._uow(str(project.id)), self.profiles, self.clock)(command)

    def rename_representation(
        self, *, principal: Principal, project: Project, layer: Layer,
        representation: Representation, name: str, idempotency_key: str
    ) -> Representation:
        return RenameRepresentationHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            RenameRepresentationCommand(
                CommandContext(actor=principal, idempotency_key=idempotency_key),
                project.id, layer.id, representation.id, name, layer.revision, representation.revision,
            )
        )

    def update_representation_note(
        self, *, principal: Principal, project: Project, layer: Layer,
        representation: Representation, note: str, idempotency_key: str
    ) -> Representation:
        return UpdateRepresentationNoteHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            UpdateRepresentationNoteCommand(
                CommandContext(actor=principal, idempotency_key=idempotency_key),
                project.id, layer.id, representation.id, note, layer.revision, representation.revision,
            )
        )

    def activate_representation(
        self, *, principal: Principal, project: Project, layer: Layer,
        representation: Representation, idempotency_key: str
    ) -> Representation:
        return ActivateRepresentationHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            ActivateRepresentationCommand(
                CommandContext(actor=principal, idempotency_key=idempotency_key),
                project.id, layer.id, representation.id, layer.revision, representation.revision,
            )
        )

    def deactivate_representation(
        self, *, principal: Principal, project: Project, layer: Layer,
        representation: Representation, idempotency_key: str
    ) -> Representation:
        return DeactivateRepresentationHandler(
            self._uow(str(project.id)), self.profiles, self.clock
        )(
            DeactivateRepresentationCommand(
                CommandContext(actor=principal, idempotency_key=idempotency_key),
                project.id,
                layer.id,
                representation.id,
                layer.revision,
                representation.revision,
            )
        )

    def archive_representation(
        self, *, principal: Principal, project: Project, layer: Layer,
        representation: Representation, idempotency_key: str
    ) -> Representation:
        return ArchiveRepresentationHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            ArchiveRepresentationCommand(
                CommandContext(actor=principal, idempotency_key=idempotency_key),
                project.id, layer.id, representation.id, layer.revision, representation.revision,
            )
        )

    def plan_import_directory(
        self,
        *,
        project: Project,
        directory: Path | str,
        mode: ImportMappingMode = ImportMappingMode.XY_FILENAME,
        regex: str | None = None,
    ) -> ImportPlan:
        root = ensure_regular_directory(directory)
        sources = tuple(
            ImportSource(str(path), path.name, path.stat().st_size)
            for path in sorted(root.iterdir(), key=lambda item: item.name.casefold())
            if path.is_file()
        )
        return ImportPlanner().plan(
            width=project.width,
            height=project.height,
            sources=sources,
            mode=mode,
            regex=regex,
        )

    def commit_managed_import(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        representation: Representation,
        plan: ImportPlan,
        idempotency_key: str,
    ) -> ManagedImportResult:
        if not plan.ready:
            raise ValueError("Import plan contains blocking issues")
        if representation.project_id != project.id or representation.layer_id != layer.id:
            raise ValueError("Representation does not belong to the selected project and layer")
        versions: list[ArtifactVersion] = []
        for item in plan.items:
            coordinate = project.coordinate(item.x, item.y)
            frame_id = coordinate.frame_id(project.id)
            series_id = deterministic_frame_series_id(representation.id, frame_id)
            per_frame_key = f"{idempotency_key}:{item.x}:{item.y}"
            series = CreateArtifactSeriesHandler(
                self._uow(str(project.id)), self.profiles, self.clock
            )(
                CreateArtifactSeriesCommand(
                    context=CommandContext(actor=principal, idempotency_key=f"{per_frame_key}:series"),
                    project_id=project.id,
                    scope=ArtifactScope.FRAME_REPRESENTATION,
                    name=item.source.display_name,
                    layer_id=layer.id,
                    representation_id=representation.id,
                    frame_id=frame_id,
                    series_id=series_id,
                )
            )
            source_path = Path(item.source.source_key)
            root = source_path.parent
            media_type = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
            with open_regular_read(source_path, root=root) as stream:
                chunks = iter(lambda: stream.read(1024 * 1024), b"")
                version = AddArtifactVersionHandler(
                    self._uow(str(project.id)), self.profiles, self.clock
                )(
                    AddArtifactVersionCommand(
                        context=CommandContext(actor=principal, idempotency_key=f"{per_frame_key}:version"),
                        project_id=project.id,
                        series_id=series.id,
                        filename=source_path.name,
                        media_type=media_type,
                        expected_series_revision=1,
                        tool_name="Kraken managed import",
                        parameters={"x": item.x, "y": item.y, "mapping": "preflight"},
                    ),
                    chunks,
                )
            versions.append(version)
        return ManagedImportResult(plan, tuple(versions))

    def rename_project(
        self, *, principal: Principal, project: Project, name: str, idempotency_key: str
    ) -> Project:
        workspace = self.project_workspace(project.id)
        if workspace is None:
            return RenameProjectHandler(self._uow(str(project.id)), self.profiles, self.clock)(
                RenameProjectCommand(
                    context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                    project_id=project.id,
                    name=name,
                    expected_revision=project.revision,
                )
            )
        normalized = validate_workspace_name(name, field_name="Название проекта")
        if normalized == workspace.project_name:
            return project
        active_runs = tuple(
            run for run in self.list_derived_runs(project.id) if run.state.value == "running"
        )
        latest_job_states: dict[str, str] = {}
        for event in self.history(project.id):
            payload = getattr(event, "payload", {})
            job = payload.get("job", {}) if isinstance(payload, Mapping) else {}
            if not isinstance(job, Mapping):
                continue
            job_id = str(payload.get("plugin_job_id", job.get("id", "")))
            if job_id:
                latest_job_states[job_id] = str(job.get("state", ""))
        if active_runs or any(
            state and state not in {"succeeded", "failed", "cancelled"}
            for state in latest_job_states.values()
        ):
            raise WorkspaceValidationError(
                "Нельзя переименовать проект, пока выполняются задания."
            )

        source_old = Path(workspace.source_project_dir).resolve()
        derived_old = Path(workspace.derived_project_dir).resolve()
        source_new = Path(workspace.source_root).resolve() / normalized
        derived_new = Path(workspace.derived_root).resolve() / normalized
        if source_new.exists() or derived_new.exists():
            raise FileExistsError(
                f"Целевая папка проекта «{normalized}» уже существует."
            )
        layer_bindings = tuple(
            binding
            for layer in self.list_layers(project.id, include_archived=True)
            if (binding := self.layer_file_binding(project.id, layer.id)) is not None
        )
        runs = self.list_derived_runs(project.id)

        def remap(value: str, old_root: Path, new_root: Path) -> str:
            if not value:
                return value
            candidate = Path(value).resolve(strict=False)
            try:
                relative = candidate.relative_to(old_root)
            except ValueError:
                return value
            return str(new_root / relative)

        updated_workspace = replace(
            workspace,
            project_name=normalized,
            source_project_dir=str(source_new),
            derived_project_dir=str(derived_new),
        )
        updated_layers = tuple(
            replace(
                binding,
                image_directory=remap(binding.image_directory, source_old, source_new),
                ssc_directory=remap(binding.ssc_directory, source_old, source_new),
                prv_directory=remap(binding.prv_directory, source_old, source_new),
                aux_directory=remap(binding.aux_directory, source_old, source_new),
                import_root=remap(binding.import_root, source_old, source_new),
            )
            for binding in layer_bindings
        )
        updated_runs = tuple(
            replace(run, path=remap(run.path, derived_old, derived_new))
            for run in runs
        )
        moved: list[tuple[Path, Path]] = []
        try:
            os.replace(source_old, source_new)
            moved.append((source_new, source_old))
            os.replace(derived_old, derived_new)
            moved.append((derived_new, derived_old))
            self.workspace_registry.save_project(updated_workspace)
            for binding in updated_layers:
                self.workspace_registry.save_layer(str(project.id), binding)
            for run in updated_runs:
                self.workspace_registry.save_run(str(project.id), run)
            renamed = RenameProjectHandler(
                self._uow(str(project.id)), self.profiles, self.clock
            )(
                RenameProjectCommand(
                    context=CommandContext(
                        actor=principal,
                        idempotency_key=idempotency_key,
                    ),
                    project_id=project.id,
                    name=normalized,
                    expected_revision=project.revision,
                )
            )
        except Exception:
            for current, original in reversed(moved):
                if current.exists() and not original.exists():
                    os.replace(current, original)
            self.workspace_registry.save_project(workspace)
            for binding in layer_bindings:
                self.workspace_registry.save_layer(str(project.id), binding)
            for run in runs:
                self.workspace_registry.save_run(str(project.id), run)
            raise
        self._external_source_indexes.clear()
        return renamed

    def archive_project(
        self, *, principal: Principal, project: Project, idempotency_key: str
    ) -> Project:
        return ArchiveProjectHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            ArchiveProjectCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id,
                expected_revision=project.revision,
            )
        )

    def restore_project(
        self, *, principal: Principal, project: Project, idempotency_key: str
    ) -> Project:
        return RestoreProjectHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            RestoreProjectCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id,
                expected_revision=project.revision,
            )
        )

    @staticmethod
    def _remove_cache_tree(root: Path, target: Path) -> bool:
        safe_root = root.resolve()
        safe_target = target.resolve()
        if safe_target.parent != safe_root:
            raise WorkspaceValidationError(
                f"Путь кэша проекта выходит за допустимые границы: {safe_target}"
            )
        if not safe_target.exists():
            return False
        if safe_target.is_symlink():
            raise WorkspaceValidationError(
                f"Кэш проекта не должен быть символической ссылкой: {safe_target}"
            )
        shutil.rmtree(safe_target)
        return True

    @staticmethod
    def _staging_entry_belongs_to_project(entry: Path, project_id: str) -> bool:
        if entry.name.startswith(f"{project_id}-"):
            return True
        inspected = 0
        try:
            manifests = entry.rglob("*.json")
            for manifest in manifests:
                inspected += 1
                if inspected > 128:
                    break
                try:
                    if not manifest.is_file() or manifest.stat().st_size > 2 * 1024 * 1024:
                        continue
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict) and str(payload.get("project_id", "")) == project_id:
                    return True
        except OSError:
            return False
        return False

    def delete_project(
        self,
        *,
        principal: Principal,
        project: Project,
        confirmation_name: str,
    ) -> ProjectDeletionResult:
        """Forget a local project while preserving both workspace data trees."""

        current = self.get_project(project.id)
        if current is None:
            raise WorkspaceValidationError("project is no longer available")
        if confirmation_name != current.name:
            raise WorkspaceValidationError("project name confirmation does not match")
        AuthorizationPolicy().require(
            principal=principal,
            storage=self.profile,
            permission=Permission.ARCHIVE_PROJECT,
            roles=self.project_roles(current.id, principal.id),
        )

        active_runs = [
            run
            for run in self.list_derived_runs(current.id)
            if run.state.value == "running"
        ]
        latest_jobs: dict[str, str] = {}
        for event in self.history(current.id):
            payload = event.payload
            job = payload.get("job", {})
            if not isinstance(job, Mapping):
                continue
            job_id = str(payload.get("plugin_job_id", job.get("id", "")))
            if job_id:
                latest_jobs[job_id] = str(job.get("state", ""))
        active_job_states = {
            state
            for state in latest_jobs.values()
            if state and state not in {"succeeded", "failed", "cancelled"}
        }
        if active_runs or active_job_states:
            raise WorkspaceValidationError(
                "project has an active plugin run; finish or cancel it before deletion"
            )

        identifier = str(current.id)
        projects_root = (self.catalog_root / "projects").resolve()
        project_cache = (projects_root / identifier).resolve()
        binding = self.project_workspace(current.id)
        if binding is not None:
            for workspace_path in (
                binding.source_project_dir,
                binding.derived_project_dir,
            ):
                resolved_workspace = Path(workspace_path).resolve()
                if resolved_workspace == project_cache or resolved_workspace.is_relative_to(project_cache):
                    raise WorkspaceValidationError(
                        "Папка данных проекта находится внутри внутреннего кэша Kraken; "
                        "безопасное удаление невозможно."
                    )

        thumbnail_root = (self.data_dir / "cache" / "frame-thumbnails").resolve()
        namespace = StoreNamespace(
            plugin="matrix",
            project=identifier,
            generation="v1",
        )
        thumbnail_cache = (thumbnail_root / namespace.digest()).resolve()
        thumbnail_removed = self._remove_cache_tree(thumbnail_root, thumbnail_cache)

        staging_removed = 0
        staging_root = (self.data_dir / "agent-staging").resolve()
        if staging_root.is_dir():
            for entry in tuple(staging_root.iterdir()):
                if (
                    not entry.is_dir()
                    or entry.is_symlink()
                    or entry.resolve().parent != staging_root
                    or not self._staging_entry_belongs_to_project(entry, identifier)
                ):
                    continue
                shutil.rmtree(entry)
                staging_removed += 1

        catalog_removed = self._remove_cache_tree(projects_root, project_cache)
        self.identities.remove_project(current.id)
        self._external_source_indexes.clear()
        return ProjectDeletionResult(
            project_id=identifier,
            catalog_cache_removed=catalog_removed,
            thumbnail_cache_removed=thumbnail_removed,
            staging_directories_removed=staging_removed,
        )

    def rename_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        name: str,
        idempotency_key: str,
    ) -> Layer:
        binding = self.layer_file_binding(project.id, layer.id)
        workspace = self.project_workspace(project.id)
        normalized = validate_workspace_name(name, field_name="Название слоя")
        if binding is None or workspace is None:
            return RenameLayerHandler(self._uow(str(project.id)), self.profiles, self.clock)(
                RenameLayerCommand(
                    context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                    project_id=project.id,
                    layer_id=layer.id,
                    name=normalized,
                    expected_revision=layer.revision,
                )
            )
        if normalized == binding.layer_name:
            return layer
        active_runs = tuple(
            run
            for run in self.list_derived_runs(project.id, layer.id)
            if run.state.value == "running"
        )
        if active_runs:
            raise WorkspaceValidationError(
                "Нельзя переименовать слой, пока выполняются задания."
            )
        if binding.mode is LayerSourceMode.EXTERNAL:
            try:
                self.workspace_registry.save_layer(
                    str(project.id),
                    replace(binding, layer_name=normalized),
                )
                renamed = RenameLayerHandler(
                    self._uow(str(project.id)), self.profiles, self.clock
                )(
                    RenameLayerCommand(
                        context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                        project_id=project.id,
                        layer_id=layer.id,
                        name=normalized,
                        expected_revision=layer.revision,
                    )
                )
            except (OSError, RuntimeError, ValueError):
                self.workspace_registry.save_layer(str(project.id), binding)
                raise
            return renamed

        source_root = Path(workspace.source_project_dir).resolve()
        derived_root = Path(workspace.derived_project_dir).resolve()
        source_paths = tuple(
            Path(value).resolve()
            for value in (
                binding.image_directory,
                binding.ssc_directory,
                binding.prv_directory,
                binding.aux_directory,
            )
            if value
        )
        moves: list[tuple[Path, Path]] = [
            (source, source.parent / normalized)
            for source in source_paths
        ]
        moves.extend(
            (
                derived_root / kind.value / binding.layer_name,
                derived_root / kind.value / normalized,
            )
            for kind in DerivedRunKind
        )
        for source, target in moves:
            if source.exists() and target.exists():
                raise FileExistsError(f"Целевая папка слоя уже существует: {target}")
            try:
                source.relative_to(source_root if source in source_paths else derived_root)
                target.relative_to(source_root if source in source_paths else derived_root)
            except ValueError as exc:
                raise WorkspaceValidationError(
                    f"Путь слоя выходит за границы рабочего каталога: {source}"
                ) from exc
        updated_binding = replace(
            binding,
            layer_name=normalized,
            image_directory=str(Path(binding.image_directory).parent / normalized),
            ssc_directory=str(Path(binding.ssc_directory).parent / normalized),
            prv_directory=str(Path(binding.prv_directory).parent / normalized),
            aux_directory=str(Path(binding.aux_directory).parent / normalized),
        )
        runs = self.list_derived_runs(project.id, layer.id)
        updated_runs = tuple(
            replace(
                run,
                path=str(
                    Path(run.path).parent / normalized
                    if Path(run.path).name == binding.layer_name
                    else Path(run.path)
                ),
            )
            for run in runs
        )
        completed: list[tuple[Path, Path]] = []
        try:
            for source, target in moves:
                if not source.exists():
                    continue
                os.replace(source, target)
                completed.append((target, source))
            self.workspace_registry.save_layer(str(project.id), updated_binding)
            for run in updated_runs:
                self.workspace_registry.save_run(str(project.id), run)
            renamed = RenameLayerHandler(
                self._uow(str(project.id)), self.profiles, self.clock
            )(
                RenameLayerCommand(
                    context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                    project_id=project.id,
                    layer_id=layer.id,
                    name=normalized,
                    expected_revision=layer.revision,
                )
            )
        except Exception:
            for current, original in reversed(completed):
                if current.exists() and not original.exists():
                    os.replace(current, original)
            self.workspace_registry.save_layer(str(project.id), binding)
            for run in runs:
                self.workspace_registry.save_run(str(project.id), run)
            raise
        self._external_source_indexes.clear()
        return renamed

    def reorder_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        order: int,
        idempotency_key: str,
    ) -> Layer:
        return ReorderLayerHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            ReorderLayerCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id,
                layer_id=layer.id,
                order=order,
                expected_revision=layer.revision,
            )
        )

    def archive_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        idempotency_key: str,
    ) -> Layer:
        return ArchiveLayerHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            ArchiveLayerCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id,
                layer_id=layer.id,
                expected_revision=layer.revision,
            )
        )

    def get_project(self, project_id: ProjectId | str, *, as_of: datetime | None = None) -> Project | None:
        projections = self._projection(project_id)
        return projections.get_project(ProjectId(str(project_id)), as_of=as_of)

    def _projection(self, project_id: ProjectId | str) -> SQLiteProjectionStore:
        identifier = str(project_id)
        events = FilesystemEventStore(self.catalog_root, identifier)
        with events.lock.hold(events.lock_timeout):
            projections = SQLiteProjectionStore(self.catalog_root, identifier)
            if projections.checkpoint != events.last_global_position():
                rebuild_filesystem_index(events, projections, acl=self.identities)
                projections = SQLiteProjectionStore(self.catalog_root, identifier)
        return projections

    def list_projects(self, *, include_archived: bool = False) -> tuple[Project, ...]:
        projects_root = self.catalog_root / "projects"
        if not projects_root.is_dir():
            return ()
        result: list[Project] = []
        for directory in sorted(path for path in projects_root.iterdir() if path.is_dir()):
            descriptor = directory / "project.json"
            try:
                project_id = ProjectId(
                    str(json.loads(descriptor.read_text(encoding="utf-8"))["project_id"])
                    if descriptor.is_file()
                    else directory.name
                )
                project = self._projection(project_id).get_project(project_id)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if project is not None and not descriptor.is_file():
                FileProjectLayout(self.catalog_root, str(project.id)).initialize(
                    {
                        "project_id": str(project.id),
                        "name": project.name,
                        "width": project.width,
                        "height": project.height,
                        "orientation": project.orientation.value,
                        "storage_profile": project.storage_profile,
                    }
                )
            if project is not None and (include_archived or project.state.value != "archived"):
                result.append(project)
        return tuple(sorted(result, key=lambda project: project.name.casefold()))

    def list_layers(
        self,
        project_id: ProjectId | str,
        *,
        as_of: datetime | None = None,
        include_archived: bool = False,
    ) -> tuple[Layer, ...]:
        return self._projection(project_id).list_layers(
            ProjectId(str(project_id)), as_of=as_of, include_archived=include_archived
        )

    def list_representations(
        self, project_id: ProjectId | str, layer_id: LayerId | str, *, as_of: datetime | None = None
    ) -> tuple[Representation, ...]:
        return self._projection(project_id).list_representations(LayerId(str(layer_id)), as_of=as_of)

    def list_artifact_series(
        self,
        project_id: ProjectId | str,
        *,
        layer_id: LayerId | str | None = None,
        representation_id: RepresentationId | str | None = None,
        include_archived: bool = False,
    ) -> tuple[ArtifactSeries, ...]:
        return self._projection(project_id).list_artifact_series(
            ProjectId(str(project_id)),
            layer_id=None if layer_id is None else LayerId(str(layer_id)),
            representation_id=(
                None
                if representation_id is None
                else RepresentationId(str(representation_id))
            ),
            include_archived=include_archived,
        )

    def create_artifact_series(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        scope: ArtifactScope,
        name: str,
        layer_id: LayerId | str | None = None,
        representation_id: RepresentationId | str | None = None,
        frame_id: str | None = None,
        idempotency_key: str,
    ) -> ArtifactSeries:
        return CreateArtifactSeriesHandler(
            self._uow(str(project_id)),
            self.profiles,
            self.clock,
        )(
            CreateArtifactSeriesCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=ProjectId(str(project_id)),
                scope=scope,
                name=name,
                layer_id=None if layer_id is None else LayerId(str(layer_id)),
                representation_id=(
                    None
                    if representation_id is None
                    else RepresentationId(str(representation_id))
                ),
                frame_id=None if frame_id is None else FrameId(str(frame_id)),
            )
        )

    def artifact_stream_revision(
        self,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
    ) -> int:
        return FilesystemEventStore(
            self.catalog_root,
            str(project_id),
        ).current_revision(f"artifact-series:{series_id}")

    def add_managed_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
        source: Path | str,
        parent_version_id: ArtifactVersionId | str | None = None,
        idempotency_key: str,
    ) -> ArtifactVersion:
        path = Path(source).resolve(strict=True)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with open_regular_read(path, root=path.parent) as stream:
            return AddArtifactVersionHandler(
                self._uow(str(project_id)),
                self.profiles,
                self.clock,
            )(
                AddArtifactVersionCommand(
                    context=CommandContext(
                        actor=principal,
                        idempotency_key=idempotency_key,
                    ),
                    project_id=ProjectId(str(project_id)),
                    series_id=ArtifactSeriesId(str(series_id)),
                    filename=path.name,
                    media_type=media_type,
                    expected_series_revision=self.artifact_stream_revision(
                        project_id,
                        series_id,
                    ),
                    parent_version_id=(
                        None
                        if parent_version_id is None
                        else ArtifactVersionId(str(parent_version_id))
                    ),
                    tool_name="Kraken Desktop",
                ),
                iter(lambda: stream.read(1024 * 1024), b""),
            )

    def add_external_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
        source: Path | str,
        parent_version_id: ArtifactVersionId | str | None = None,
        idempotency_key: str,
    ) -> ArtifactVersion:
        path = Path(source).resolve(strict=True)
        digest = hashlib.sha256()
        size = 0
        with open_regular_read(path, root=path.parent) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        return AddExternalArtifactVersionHandler(
            self._uow(str(project_id)),
            self.profiles,
            self.clock,
        )(
            AddExternalArtifactVersionCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=ProjectId(str(project_id)),
                series_id=ArtifactSeriesId(str(series_id)),
                filename=path.name,
                media_type=mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
                uri=path.as_uri(),
                fingerprint_sha256=digest.hexdigest(),
                observed_size_bytes=size,
                expected_series_revision=self.artifact_stream_revision(
                    project_id,
                    series_id,
                ),
                parent_version_id=(
                    None
                    if parent_version_id is None
                    else ArtifactVersionId(str(parent_version_id))
                ),
                parameters={"local_path": str(path)},
            )
        )

    def rename_artifact_series(
        self,
        *,
        principal: Principal,
        series: ArtifactSeries,
        name: str,
        idempotency_key: str,
    ) -> ArtifactSeries:
        return RenameArtifactSeriesHandler(
            self._uow(str(series.project_id)),
            self.profiles,
            self.clock,
        )(
            RenameArtifactSeriesCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=series.project_id,
                series_id=series.id,
                name=name,
                expected_series_revision=series.revision,
            )
        )

    def archive_artifact_series(
        self,
        *,
        principal: Principal,
        series: ArtifactSeries,
        idempotency_key: str,
    ) -> ArtifactSeries:
        return ArchiveArtifactSeriesHandler(
            self._uow(str(series.project_id)),
            self.profiles,
            self.clock,
        )(
            ArchiveArtifactSeriesCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=series.project_id,
                series_id=series.id,
                expected_series_revision=series.revision,
            )
        )

    def activate_artifact_version(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
        version_id: ArtifactVersionId | str,
        idempotency_key: str,
    ) -> ArtifactVersion:
        return ActivateArtifactVersionHandler(
            self._uow(str(project_id)),
            self.profiles,
            self.clock,
        )(
            ActivateArtifactVersionCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=ProjectId(str(project_id)),
                series_id=ArtifactSeriesId(str(series_id)),
                version_id=ArtifactVersionId(str(version_id)),
            )
        )

    def external_artifact_changed(self, version: ArtifactVersion) -> bool:
        if version.external is None:
            return False
        from urllib.parse import unquote, urlparse

        parsed = urlparse(version.external.uri)
        if parsed.scheme != "file":
            raise ValueError("Desktop supports only local external file links")
        path = Path(unquote(parsed.path.lstrip("/") if os.name == "nt" else parsed.path))
        if os.name == "nt" and parsed.netloc:
            path = Path(f"//{parsed.netloc}/{unquote(parsed.path.lstrip('/'))}")
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            return True
        digest = hashlib.sha256()
        size = 0
        with open_regular_read(resolved, root=resolved.parent) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        return (
            size != version.external.observed_size_bytes
            or digest.hexdigest() != version.external.fingerprint_sha256
        )

    def export_managed_artifact(
        self,
        project_id: ProjectId | str,
        version: ArtifactVersion,
        destination: Path | str,
    ) -> Path:
        if version.blob is None:
            raise ValueError("The selected artifact version is external")
        target = Path(destination).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as stream:
                for chunk in FilesystemBlobStore.for_project(
                    self.catalog_root,
                    str(project_id),
                ).iter_bytes(version.blob):
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return target

    def list_notes(
        self,
        project_id: ProjectId | str,
        *,
        layer_id: LayerId | str | None = None,
        frame_id: str | None = None,
    ) -> tuple[NoteRevision, ...]:
        return self._projection(project_id).list_notes(
            ProjectId(str(project_id)),
            layer_id=None if layer_id is None else LayerId(str(layer_id)),
            frame_id=frame_id,
        )

    def create_note(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        body: str,
        layer_id: LayerId | str | None = None,
        frame_id: str | None = None,
        idempotency_key: str,
    ) -> NoteRevision:
        return CreateNoteHandler(
            self._uow(str(project_id)),
            self.profiles,
            self.clock,
        )(
            CreateNoteCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=ProjectId(str(project_id)),
                body=body,
                layer_id=None if layer_id is None else LayerId(str(layer_id)),
                frame_id=None if frame_id is None else FrameId(str(frame_id)),
            )
        )

    def revise_note(
        self,
        *,
        principal: Principal,
        note: NoteRevision,
        body: str,
        idempotency_key: str,
    ) -> NoteRevision:
        return ReviseNoteHandler(
            self._uow(str(note.project_id)),
            self.profiles,
            self.clock,
        )(
            ReviseNoteCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=note.project_id,
                note_id=note.note_id,
                body=body,
                expected_revision=note.revision,
            )
        )

    def frame_cells(
        self,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        representation_id: RepresentationId | str,
        *,
        as_of: datetime | None = None,
    ) -> tuple[FrameCellSnapshot, ...]:
        project = self.get_project(project_id, as_of=as_of)
        if project is None:
            return ()
        projection = self._projection(project.id)
        representation = projection.get_representation(
            RepresentationId(str(representation_id)), as_of=as_of
        )
        if representation is None or str(representation.layer_id) != str(layer_id):
            return ()
        cells: dict[str, FrameCellSnapshot] = {}
        performers = {
            str(item.principal_id): item
            for item in self.list_performers()
            if item.principal_id is not None
        }
        accepted_versions = {
            str(identifier)
            for event in self.history(project.id, as_of=as_of)
            if event.event_type == "ReviewBatchAccepted"
            for identifier in event.payload.get("candidate_version_ids", ())
        }
        analysis = self.latest_karakal_analysis(project.id, layer_id, as_of=as_of)
        confidence = {} if analysis is None else analysis.frame_confidence
        for series in projection.list_artifact_series(
            project.id,
            layer_id=LayerId(str(layer_id)),
            representation_id=representation.id,
            as_of=as_of,
        ):
            version = projection.get_active_artifact_version(series.id, as_of=as_of)
            cursor = version
            coordinate: tuple[int, int] | None = None
            visited: set[str] = set()
            while cursor is not None and str(cursor.id) not in visited:
                visited.add(str(cursor.id))
                x, y = cursor.parameters.get("x"), cursor.parameters.get("y")
                if (
                    isinstance(x, int)
                    and not isinstance(x, bool)
                    and isinstance(y, int)
                    and not isinstance(y, bool)
                    and 1 <= x <= project.width
                    and 1 <= y <= project.height
                ):
                    coordinate = (x, y)
                    break
                cursor = (
                    None
                    if cursor.parent_version_id is None
                    else projection.get_artifact_version(cursor.parent_version_id, as_of=as_of)
                )
            if version is None or coordinate is None or series.frame_id is None:
                continue
            status = "image_ready" if representation.kind is RepresentationKind.IMAGE else "vectorized"
            performer = performers.get(str(version.author_principal_id))
            cells[str(series.frame_id)] = FrameCellSnapshot(
                coordinate[0],
                coordinate[1],
                status,
                str(series.frame_id),
                str(version.id),
                version.sha256,
                version.created_at.isoformat(),
                "" if performer is None else performer.color,
                "" if performer is None else "".join(part[:1] for part in performer.name.split()[:2]).upper(),
                "checked" if str(version.id) in accepted_versions else "not_checked",
                confidence.get(str(series.frame_id)),
            )
        review_status = {
            "issued": "in_review",
            "partially_returned": "in_review",
            "awaiting_acceptance": "returned_changed",
            "changes_requested": "changes_requested",
        }
        for batch in projection.list_active_review_batches(
            project.id, LayerId(str(layer_id)), as_of=as_of
        ):
            status = review_status.get(batch.state.value)
            if status is None:
                continue
            for item in batch.items:
                previous = cells.get(str(item.frame_id))
                if previous is not None:
                    cells[str(item.frame_id)] = FrameCellSnapshot(
                        previous.x,
                        previous.y,
                        status,
                        previous.frame_id,
                        previous.artifact_version_id,
                        previous.sha256,
                        previous.modified_at,
                        previous.performer_color,
                        previous.performer_initials,
                        "in_review",
                        previous.quality,
                    )
        return tuple(sorted(cells.values(), key=lambda item: (item.y, item.x)))

    def frame_management_states(
        self,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        representation_id: RepresentationId | str,
        *,
        as_of: datetime | None = None,
    ) -> tuple[FrameManagementState, ...]:
        return tuple(
            FrameManagementState(
                frame_id=item.frame_id,
                artifact_version_id=item.artifact_version_id,
                modified_at=item.modified_at,
                performer_color=item.performer_color,
                performer_initials=item.performer_initials,
                review_status=item.review_status,
            )
            for item in self.frame_cells(
                project_id,
                layer_id,
                representation_id,
                as_of=as_of,
            )
        )

    def publish_karakal_analysis(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        frame_confidence: dict[str, float],
        report: dict[str, object],
        parameters: dict[str, object],
        plugin_version: str,
        idempotency_key: str,
    ) -> KarakalAnalysisRun:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project does not exist")
        layer = next(
            (value for value in self.list_layers(project.id) if str(value.id) == str(layer_id)),
            None,
        )
        if layer is None:
            raise ValueError("Layer does not exist")
        normalized: dict[str, float] = {}
        for frame_id, raw_value in frame_confidence.items():
            value = float(raw_value)
            if not 0.0 <= value <= 1.0:
                raise ValueError("Karakal confidence must be in range 0..1")
            normalized[str(frame_id)] = value
        prior = next(
            (
                event
                for event in reversed(self.history(project.id))
                if event.event_type == "KarakalAnalysisPublished"
                and event.idempotency_key == idempotency_key
            ),
            None,
        )
        if prior is None:
            stream_id = f"karakal:{layer.id}"
            store = FilesystemEventStore(self.catalog_root, str(project.id))
            revision = store.current_revision(stream_id)
            run_id = new_uuid()
            event = EventEnvelope.create(
                stream_id=stream_id,
                project_id=project.id,
                revision=revision + 1,
                event_type="KarakalAnalysisPublished",
                payload={
                    "run_id": run_id,
                    "layer_id": str(layer.id),
                    "publication_sequence": revision + 1,
                    "frame_confidence": normalized,
                    "report": dict(report),
                    "parameters": dict(parameters),
                    "plugin_version": str(plugin_version),
                },
                actor=ActorSnapshot.from_principal(principal),
                program=ProgramSnapshot("Karakal", str(plugin_version)),
                idempotency_key=idempotency_key,
            )
            store.append(stream_id, expected_revision=revision, events=(event,))
            prior = event
        return self._karakal_run_from_event(prior)

    def record_layer_pipeline_action(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        action: str,
        node_id: str,
        plugin_id: str,
        capability: str,
        mode: str,
        parameters: dict[str, object] | None = None,
    ) -> EventEnvelope:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project does not exist")
        if not any(str(value.id) == str(layer_id) for value in self.list_layers(project.id)):
            raise ValueError("Layer does not exist")
        stream_id = f"layer-pipeline:{layer_id}"
        store = FilesystemEventStore(self.catalog_root, str(project.id))
        revision = store.current_revision(stream_id)
        event = EventEnvelope.create(
            stream_id=stream_id,
            project_id=project.id,
            revision=revision + 1,
            event_type="LayerPipelineActionRequested",
            payload={
                "layer_id": str(layer_id),
                "action": str(action),
                "node_id": str(node_id),
                "plugin_id": str(plugin_id),
                "capability": str(capability),
                "mode": str(mode),
                "parameters": dict(parameters or {}),
                "state": "launched",
            },
            actor=ActorSnapshot.from_principal(principal),
            program=ProgramSnapshot(str(plugin_id)),
        )
        store.append(stream_id, expected_revision=revision, events=(event,))
        return event

    def remove_layer_pipeline_action(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        action_event_id: str,
    ) -> EventEnvelope:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project does not exist")
        target = next(
            (
                event
                for event in self.history(project.id)
                if event.event_id == str(action_event_id)
                and event.event_type == "LayerPipelineActionRequested"
                and str(event.payload.get("layer_id", "")) == str(layer_id)
            ),
            None,
        )
        if target is None:
            raise ValueError("Pipeline step does not exist in this layer")
        prior = next(
            (
                event
                for event in self.history(project.id)
                if event.event_type == "LayerPipelineActionRemoved"
                and str(event.payload.get("action_event_id", "")) == target.event_id
            ),
            None,
        )
        if prior is not None:
            return prior
        stream_id = f"layer-pipeline:{layer_id}"
        store = FilesystemEventStore(self.catalog_root, str(project.id))
        revision = store.current_revision(stream_id)
        event = EventEnvelope.create(
            stream_id=stream_id,
            project_id=project.id,
            revision=revision + 1,
            event_type="LayerPipelineActionRemoved",
            payload={
                "layer_id": str(layer_id),
                "action_event_id": target.event_id,
                "action": str(target.payload.get("action", "")),
            },
            actor=ActorSnapshot.from_principal(principal),
            program=ProgramSnapshot("Kraken"),
        )
        store.append(stream_id, expected_revision=revision, events=(event,))
        return event

    def latest_karakal_analysis(
        self,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        *,
        as_of: datetime | None = None,
    ) -> KarakalAnalysisRun | None:
        event = next(
            (
                item
                for item in reversed(self.history(project_id, as_of=as_of))
                if item.event_type == "KarakalAnalysisPublished"
                and str(item.payload.get("layer_id", "")) == str(layer_id)
            ),
            None,
        )
        return None if event is None else self._karakal_run_from_event(event)

    @staticmethod
    def _karakal_run_from_event(event: EventEnvelope) -> KarakalAnalysisRun:
        payload = event.payload
        return KarakalAnalysisRun(
            run_id=str(payload["run_id"]),
            project_id=str(event.project_id),
            layer_id=str(payload["layer_id"]),
            publication_sequence=int(payload["publication_sequence"]),
            created_at=event.recorded_at.isoformat(),
            frame_confidence={
                str(identifier): float(value)
                for identifier, value in dict(payload.get("frame_confidence", {})).items()
            },
            report=dict(payload.get("report", {})),
            parameters=dict(payload.get("parameters", {})),
            plugin_version=str(payload.get("plugin_version", "")),
        )

    def matrix_viewport(
        self,
        project_id: ProjectId | str,
        *,
        layer_id: LayerId | str,
        representation_ids: Iterable[RepresentationId | str],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        lod: int = 0,
    ) -> dict[str, Any]:
        """Return sparse cells or aggregate buckets for a visible matrix region."""

        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project does not exist")
        if not (1 <= int(x1) <= int(x2) <= project.width and 1 <= int(y1) <= int(y2) <= project.height):
            raise ValueError("Viewport is outside the project grid")
        if int(lod) < 0 or int(lod) > 24:
            raise ValueError("LOD must be between 0 and 24")

        identifiers = tuple(str(value) for value in representation_ids if str(value))
        representations = {
            str(value.id): value
            for value in self.list_representations(project.id, LayerId(str(layer_id)))
            if str(value.id) in identifiers
        }
        priority = {
            "empty": 0,
            "image_ready": 1,
            "processing": 2,
            "vectorized": 3,
            "in_review": 4,
            "returned_unchanged": 5,
            "returned_changed": 6,
            "approved": 7,
            "changes_requested": 8,
            "conflict": 9,
            "error": 10,
        }
        merged: dict[tuple[int, int], dict[str, Any]] = {}
        coverage: dict[str, set[tuple[int, int]]] = {}
        for identifier in identifiers:
            current_coverage = coverage.setdefault(identifier, set())
            representation = representations.get(identifier)
            frame_cells = self.frame_cells(project.id, layer_id, identifier)
            for item in frame_cells:
                if not (x1 <= item.x <= x2 and y1 <= item.y <= y2):
                    continue
                coordinate = (item.x, item.y)
                current_coverage.add(coordinate)
                current = merged.get(coordinate)
                image_asset = (
                    {
                        "asset_sha256": item.sha256,
                        "asset_revision": item.artifact_version_id,
                    }
                    if representation is not None
                    and representation.kind is RepresentationKind.IMAGE
                    and item.sha256
                    else {}
                )
                if current is not None and priority.get(str(current["status"]), 0) > priority.get(item.status, 0):
                    current.update(image_asset)
                    continue
                replacement = {
                    "artifact_version_id": item.artifact_version_id,
                    "frame_id": item.frame_id,
                    "sha256": item.sha256,
                    "status": item.status,
                    "x": item.x,
                    "y": item.y,
                    "modified_at": item.modified_at,
                    "performer_color": item.performer_color,
                    "performer_initials": item.performer_initials,
                    "review_status": item.review_status,
                    "quality": item.quality,
                }
                if current is not None:
                    for field in ("asset_sha256", "asset_revision"):
                        if field in current:
                            replacement[field] = current[field]
                replacement.update(image_asset)
                merged[coordinate] = replacement
            if (
                not frame_cells
                and representation is not None
                and representation.kind is RepresentationKind.IMAGE
                and representation.source
                and representation.source != "managed-import"
            ):
                for cell in self._external_directory_viewport(
                    project,
                    representation,
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                ):
                    coordinate = (int(cell["x"]), int(cell["y"]))
                    current_coverage.add(coordinate)
                    current = merged.get(coordinate)
                    if current is None:
                        merged[coordinate] = cell
                    else:
                        current.update(
                            {
                                field: cell[field]
                                for field in (
                                    "asset_source_key",
                                    "asset_revision",
                                    "asset_path",
                                    "asset_media_type",
                                )
                            }
                        )

        managed = {
            identifier
            for identifier, representation in representations.items()
            if representation.source == "managed-import"
        }
        if int(lod) == 0 and managed:
            for y in range(int(y1), int(y2) + 1):
                for x in range(int(x1), int(x2) + 1):
                    coordinate = (x, y)
                    missing_identifiers = tuple(
                        identifier
                        for identifier in managed
                        if coordinate not in coverage.get(identifier, set())
                    )
                    if not missing_identifiers:
                        continue
                    current = merged.get(coordinate)
                    if current is None:
                        current = {
                            "artifact_version_id": "",
                            "frame_id": str(project.frame_id_at(x, y)),
                            "sha256": "",
                            "x": x,
                            "y": y,
                        }
                        merged[coordinate] = current
                    # A sparse managed vector representation must not erase the
                    # raster asset already merged into this coordinate.  The
                    # missing state is orthogonal to the image used for the
                    # thumbnail.
                    current.update(
                        {
                            "status": "error",
                            "missing": True,
                            "missing_representation_ids": missing_identifiers,
                        }
                    )

        revision = str(self._projection(project.id).checkpoint)
        if int(lod) == 0:
            return {
                "project_id": str(project.id),
                "layer_id": str(layer_id),
                "bounds": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "lod": 0,
                "revision": revision,
                "cells": tuple(sorted(merged.values(), key=lambda value: (value["y"], value["x"]))),
                "aggregates": (),
            }

        span = 1 << int(lod)
        buckets: dict[tuple[int, int], dict[str, Any]] = {}
        for cell in merged.values():
            bucket_x = (int(cell["x"]) - 1) // span
            bucket_y = (int(cell["y"]) - 1) // span
            bucket = buckets.setdefault(
                (bucket_x, bucket_y),
                {
                    "bounds": {
                        "x1": bucket_x * span + 1,
                        "y1": bucket_y * span + 1,
                        "x2": min(project.width, (bucket_x + 1) * span),
                        "y2": min(project.height, (bucket_y + 1) * span),
                    },
                    "materialized_count": 0,
                    "status_counts": {},
                },
            )
            bucket["materialized_count"] += 1
            status = str(cell["status"])
            bucket["status_counts"][status] = int(bucket["status_counts"].get(status, 0)) + 1
        return {
            "project_id": str(project.id),
            "layer_id": str(layer_id),
            "bounds": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "lod": int(lod),
            "revision": revision,
            "cells": (),
            "aggregates": tuple(buckets[key] for key in sorted(buckets, key=lambda value: (value[1], value[0]))),
        }

    @staticmethod
    def _natural_path_key(path: Path) -> tuple[object, ...]:
        return tuple(
            int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", path.name)
        )

    def reorder_layers(
        self,
        *,
        principal: Principal,
        project: Project,
        layers: Iterable[Layer],
        layer_ids: Iterable[LayerId | str],
        idempotency_key: str,
    ) -> tuple[Layer, ...]:
        current = tuple(layers)
        return ReorderLayersHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            ReorderLayersCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id,
                layer_ids=tuple(LayerId(str(value)) for value in layer_ids),
                expected_revisions=tuple((layer.id, layer.revision) for layer in current),
            )
        )

    def _external_directory_viewport(
        self,
        project: Project,
        representation: Representation,
        *,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> tuple[dict[str, Any], ...]:
        """Read legacy explicit-directory representations without importing them."""

        try:
            root = ensure_regular_directory(representation.source)
            directory_revision = root.stat().st_mtime_ns
        except (OSError, ValueError):
            return ()
        cache_key = (str(representation.id), str(root), directory_revision)
        paths = self._external_source_indexes.get(cache_key)
        if paths is None:
            discovered = tuple(
                sorted(
                    (
                        path
                        for path in root.iterdir()
                        if path.is_file()
                        and path.suffix.casefold() in {".jpg", ".jpeg", ".bmp", ".png"}
                    ),
                    key=self._natural_path_key,
                )
            )
            binding = self.layer_file_binding(project.id, representation.layer_id)
            if binding is not None and binding.frame_positions:
                slots: list[Path | None] = [None] * project.frame_count
                matched = 0
                for path in discovered:
                    position = binding.frame_positions.get(path.name)
                    if position is not None and 1 <= position <= project.frame_count:
                        slots[position - 1] = path
                        matched += 1
                if matched:
                    paths = tuple(slots)
                else:
                    try:
                        derived_positions = map_frame_positions(
                            discovered,
                            maximum=project.frame_count,
                        )
                    except WorkspaceValidationError:
                        paths = discovered
                    else:
                        for path in discovered:
                            slots[derived_positions[path.name] - 1] = path
                        paths = tuple(slots)
            else:
                paths = discovered
            self._external_source_indexes = {
                key: value
                for key, value in self._external_source_indexes.items()
                if key[0] != str(representation.id)
            }
            self._external_source_indexes[cache_key] = paths

        cells: list[dict[str, Any]] = []
        analysis = self.latest_karakal_analysis(project.id, representation.layer_id)
        confidence = {} if analysis is None else analysis.frame_confidence
        for index, path in enumerate(paths[: project.frame_count]):
            x = index % project.width + 1
            visual_row = index // project.width
            y = (
                visual_row + 1
                if project.orientation is GridOrientation.Y_DOWN
                else project.height - visual_row
            )
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                continue
            if path is None:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            normalized_path = str(path.resolve())
            frame_id = str(project.frame_id_at(x, y))
            cells.append(
                {
                    "artifact_version_id": "",
                    "frame_id": frame_id,
                    "sha256": "",
                    "status": "image_ready",
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                    "performer_color": "",
                    "performer_initials": "",
                    "review_status": "not_checked",
                    "quality": confidence.get(frame_id),
                    "x": x,
                    "y": y,
                    "asset_source_key": hashlib.sha256(
                        normalized_path.encode("utf-8")
                    ).hexdigest(),
                    "asset_revision": f"{stat.st_mtime_ns}:{stat.st_size}",
                    "asset_path": normalized_path,
                    "asset_media_type": mimetypes.guess_type(path.name)[0] or "image/*",
                }
            )
        return tuple(cells)

    def read_project_blob(self, project_id: ProjectId | str, sha256: str) -> bytes:
        return FilesystemBlobStore.for_project(self.catalog_root, str(project_id)).read(sha256)

    def history(self, project_id: ProjectId | str, *, as_of: datetime | None = None) -> tuple[object, ...]:
        store = FilesystemEventStore(self.catalog_root, str(project_id))
        events = []
        for stored in store.iter_project():
            event = store._decode(stored)
            if as_of is None or getattr(event, "recorded_at", self.clock.now()) <= as_of:
                events.append(event)
        return tuple(events)

    def activity_records(self) -> tuple[ActivityRecord, ...]:
        records = [
            ActivityRecord.from_event(event)
            for project in self.list_projects(include_archived=True)
            for event in self.history(project.id)
        ]
        return tuple(sorted(records, key=lambda item: (item.recorded_at, item.event_id)))

    def active_review_batches(self) -> tuple[ReviewBatch, ...]:
        batches: dict[str, ReviewBatch] = {}
        for project in self.list_projects(include_archived=True):
            projection = self._projection(project.id)
            for layer in projection.list_layers(project.id, include_archived=True):
                for batch in projection.list_active_review_batches(project.id, layer.id):
                    batches[str(batch.id)] = batch
        return tuple(sorted(batches.values(), key=lambda item: (item.due_at is None, item.due_at, str(item.id))))

    def review_batches(self) -> tuple[ReviewBatch, ...]:
        batches: dict[str, ReviewBatch] = {}
        for project in self.list_projects(include_archived=True):
            projection = self._projection(project.id)
            for batch in projection.list_review_batches(project.id):
                batches[str(batch.id)] = batch
        return tuple(
            sorted(
                batches.values(),
                key=lambda item: (item.updated_at, str(item.id)),
                reverse=True,
            )
        )

    def plugin_jobs(self) -> tuple[object, ...]:
        jobs: dict[str, object] = {}
        for project in self.list_projects(include_archived=True):
            for job in self._projection(project.id).list_plugin_jobs(project.id):
                jobs[str(job.id)] = job
        return tuple(
            sorted(
                jobs.values(),
                key=lambda item: (getattr(item, "updated_at", None), str(item.id)),
                reverse=True,
            )
        )

    def submit_plugin_job(
        self,
        *,
        principal: Principal,
        gateway: object,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        source_representation_id: RepresentationId | str,
        target_representation_id: RepresentationId | str,
        coordinates: Iterable[tuple[int, int]],
        capability: str,
        parameters: Mapping[str, object],
        idempotency_key: str,
    ):
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project was not found")
        projection = self._projection(project.id)
        source_id = RepresentationId(str(source_representation_id))
        target_id = RepresentationId(str(target_representation_id))
        source = projection.get_representation(source_id)
        target = projection.get_representation(target_id)
        if source is None or str(source.layer_id) != str(layer_id):
            raise ValueError("Source representation was not found in the layer")
        if target is None or str(target.layer_id) != str(layer_id):
            raise ValueError("Target representation was not found in the layer")
        selection = self._domain_selection(coordinates)
        inputs: list[PluginInputV1] = []
        missing: list[str] = []
        for coordinate in selection.iter_coordinates():
            frame_id = coordinate.frame_id(project.id)
            series_id = deterministic_frame_series_id(source.id, frame_id)
            version = projection.get_active_artifact_version(series_id)
            if version is None:
                missing.append(f"({coordinate.x}, {coordinate.y})")
                continue
            suffix = Path(version.filename).suffix or ".bin"
            inputs.append(
                PluginInputV1(
                    frame_id=frame_id,
                    artifact_version_id=version.id,
                    sha256=version.sha256,
                    relative_path=f"inputs/{coordinate.x}_{coordinate.y}{suffix}",
                )
            )
        if missing:
            raise ValueError(
                "Для запуска отсутствуют входные версии кадров: "
                + ", ".join(missing)
            )
        return SubmitPluginJobHandler(
            self._uow(str(project.id)),
            self.profiles,
            self.clock,
            gateway,
        )(
            SubmitPluginJobCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=project.id,
                layer_id=LayerId(str(layer_id)),
                target_representation_id=target.id,
                selection=selection,
                capability=capability,
                inputs=tuple(inputs),
                parameters=dict(parameters),
            )
        )

    def cancel_plugin_job(
        self,
        *,
        principal: Principal,
        gateway: object,
        job: object,
        idempotency_key: str,
    ):
        return CancelPluginJobHandler(
            self._uow(str(job.project_id)),
            self.profiles,
            self.clock,
            gateway,
        )(
            CancelPluginJobCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=job.project_id,
                job_id=job.id,
                expected_revision=job.revision,
            )
        )

    def retry_plugin_job(
        self,
        *,
        principal: Principal,
        gateway: object,
        job: object,
        idempotency_key: str,
    ):
        return RetryPluginJobHandler(
            self._uow(str(job.project_id)),
            self.profiles,
            self.clock,
            gateway,
        )(
            RetryPluginJobCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=job.project_id,
                job_id=job.id,
                expected_revision=job.revision,
            )
        )

    def synchronize_plugin_jobs(
        self,
        *,
        principal: Principal,
        gateway: object,
    ) -> tuple[object, ...]:
        synchronized: list[object] = []
        for job in self.plugin_jobs():
            if job.state.value in {"succeeded", "failed", "cancelled"}:
                synchronized.append(job)
                continue
            try:
                agent = gateway.get_job(job.id)
            except Exception:
                target = "recovery_required"
                error = "Kraken Agent недоступен или потерял состояние задания"
                progress = job.progress
            else:
                target = str(agent.get("state", job.state.value))
                error = (
                    None
                    if agent.get("error") in {None, ""}
                    else str(agent.get("error"))
                )
                progress = job.progress
            if target == job.state.value:
                synchronized.append(job)
                continue
            updated = SynchronizePluginJobHandler(
                self._uow(str(job.project_id)),
                self.profiles,
                self.clock,
            )(
                SynchronizePluginJobCommand(
                    context=CommandContext(
                        actor=principal,
                        idempotency_key=f"agent-sync:{job.id}:{job.revision}:{target}",
                    ),
                    project_id=job.project_id,
                    job_id=job.id,
                    expected_revision=job.revision,
                    state=target,
                    progress=progress,
                    error=error,
                )
            )
            synchronized.append(updated)
        return tuple(synchronized)

    def import_agent_result(
        self,
        *,
        principal: Principal,
        result_payload: Mapping[str, object],
        staging_root: Path | str,
        confirm_partial: bool = False,
    ):
        parsed = parse_plugin_result_json(
            json.dumps(dict(result_payload), ensure_ascii=False)
        )
        publication: PluginResultPublicationV2 | None = None
        if isinstance(parsed, PluginResultManifest):
            manifest = domain_result_from_transport(parsed)
        elif isinstance(parsed, PluginResultPublicationV2):
            publication = parsed
            frame_results = tuple(
                PluginFrameResultV1(
                    output_id=str(
                        uuid5(
                            NAMESPACE_URL,
                            f"kraken:v2-output:{parsed.job_id}:{asset.asset_id}",
                        )
                    ),
                    frame_id=FrameId(str(asset.frame_id)),
                    outcome=PluginFrameOutcome.SUCCEEDED,
                    relative_path=asset.relative_path,
                    sha256=asset.sha256,
                    media_type=asset.media_type,
                    role=asset.role,
                )
                for asset in parsed.outputs
                if asset.scope.value == "frame" and asset.frame_id is not None
            )
            manifest = PluginResultManifestV1(
                job_id=PluginJobId(parsed.job_id),
                plugin_name=parsed.plugin_id,
                plugin_version=parsed.plugin_version,
                results=frame_results,
                parameters_applied={
                    **dict(parsed.applied_parameters),
                    "v2_publication_id": parsed.publication_id,
                    "v2_sequence": parsed.sequence,
                },
                outcome=PluginResultOutcome(parsed.outcome),
                # The authoritative domain job keeps the backwards-compatible
                # V1 command contract while the transport may be V2.
                protocol_version="1.0",
            )
        else:
            raise TypeError("Unsupported Agent result")
        source_job = next(
            (
                job
                for job in self.plugin_jobs()
                if str(job.id) == str(manifest.job_id)
            ),
            None,
        )
        if source_job is None:
            raise ValueError("Agent result belongs to an unknown plugin job")
        result = ImportPluginResultHandler(
            self._uow(str(source_job.project_id)),
            self.profiles,
            self.clock,
            AgentStagingResultContentReader(staging_root),
        )(
            ImportPluginResultCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=f"agent-import:{manifest.job_id}",
                ),
                manifest=manifest,
                confirm_partial=confirm_partial or publication is not None,
            )
        )
        if publication is None:
            return result
        job = result.job
        for asset in publication.outputs:
            if asset.scope.value != "layer":
                continue
            series = self.create_artifact_series(
                principal=principal,
                project_id=job.project_id,
                layer_id=job.layer_id,
                scope=ArtifactScope.LAYER_ATTACHMENT,
                name=asset.role,
                idempotency_key=(
                    f"agent-v2-series:{publication.publication_id}:{asset.asset_id}"
                ),
            )
            source = (
                Path(staging_root)
                / str(publication.job_id)
                / Path(asset.relative_path)
            )
            self.add_managed_artifact_version(
                principal=principal,
                project_id=job.project_id,
                series_id=series.id,
                source=source,
                idempotency_key=(
                    f"agent-v2-version:{publication.publication_id}:{asset.asset_id}"
                ),
            )
        if publication.frame_values:
            self._record_workspace_event(
                principal=principal,
                project_id=job.project_id,
                stream_id=f"plugin-job:{job.id}",
                event_type="PluginFrameValuesPublishedV2",
                payload={
                    "plugin_job_id": str(job.id),
                    "publication_id": publication.publication_id,
                    "frame_values": {
                        frame_id: dict(values)
                        for frame_id, values in publication.frame_values.items()
                    },
                },
                idempotency_key=f"agent-v2-values:{publication.publication_id}",
            )
        return result

    def import_agent_publications(
        self,
        *,
        principal: Principal,
        publications: Iterable[Mapping[str, object]],
        staging_root: Path | str,
    ):
        parsed = tuple(
            PluginResultPublicationV2.from_dict(publication)
            for publication in publications
        )
        if not parsed:
            raise ValueError("Agent did not publish any V2 results")
        ordered = tuple(sorted(parsed, key=lambda item: item.sequence))
        first = ordered[0]
        if any(
            item.job_id != first.job_id
            or item.plugin_id != first.plugin_id
            or item.plugin_version != first.plugin_version
            for item in ordered
        ):
            raise ValueError("V2 publications do not belong to one plugin job")
        sequences = tuple(item.sequence for item in ordered)
        if len(sequences) != len(set(sequences)):
            raise ValueError("V2 publication sequence contains duplicates")
        if not ordered[-1].final:
            raise ValueError("The final V2 publication has not arrived yet")
        outputs = tuple(
            asset
            for publication in ordered
            for asset in publication.outputs
        )
        asset_ids = tuple(asset.asset_id for asset in outputs)
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("V2 publications contain duplicate output asset IDs")
        frame_values: dict[str, dict[str, float]] = {}
        parameters: dict[str, object] = {}
        report: dict[str, object] = {}
        for publication in ordered:
            parameters.update(publication.applied_parameters)
            report.update(publication.report)
            for frame_id, measurements in publication.frame_values.items():
                frame_values.setdefault(frame_id, {}).update(measurements)
        parameters["v2_publication_ids"] = [
            publication.publication_id for publication in ordered
        ]
        final = ordered[-1]
        combined = PluginResultPublicationV2(
            job_id=final.job_id,
            publication_id=final.publication_id,
            sequence=final.sequence,
            plugin_id=final.plugin_id,
            plugin_version=final.plugin_version,
            outputs=outputs,
            outcome=final.outcome,
            applied_parameters=parameters,
            frame_values=frame_values,
            report=report,
            final=True,
        )
        return self.import_agent_result(
            principal=principal,
            result_payload=combined.to_dict(),
            staging_root=staging_root,
        )

    def artifact_versions(
        self,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
    ) -> tuple[ArtifactVersion, ...]:
        return self._projection(project_id).list_artifact_versions(
            ArtifactSeriesId(str(series_id))
        )

    def active_artifact_version(
        self,
        project_id: ProjectId | str,
        series_id: ArtifactSeriesId | str,
    ) -> ArtifactVersion | None:
        return self._projection(project_id).get_active_artifact_version(
            ArtifactSeriesId(str(series_id))
        )

    def artifact_version(
        self,
        project_id: ProjectId | str,
        version_id: ArtifactVersionId | str,
    ) -> ArtifactVersion | None:
        return self._projection(project_id).get_artifact_version(
            ArtifactVersionId(str(version_id))
        )

    def managed_artifact_path(
        self,
        project_id: ProjectId | str,
        version_id: ArtifactVersionId | str,
    ) -> Path:
        version = self.artifact_version(project_id, version_id)
        if version is None or version.blob is None:
            raise ValueError("Plugin inputs must be managed immutable versions")
        store = FilesystemBlobStore.for_project(self.catalog_root, str(project_id))
        path = store._path(version.blob.sha256)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _review_keys(self) -> Ed25519KeyPair:
        private_name = f"review-signing:{self.profile.id}:private"
        public_name = f"review-signing:{self.profile.id}:public"
        get_secret = getattr(self.secrets, "get")
        set_secret = getattr(self.secrets, "set")
        private_key = get_secret(private_name)
        public_key = get_secret(public_name)
        if private_key is None or public_key is None:
            pair = Ed25519KeyPair.generate()
            set_secret(private_name, pair.private_key)
            set_secret(public_name, pair.public_key)
            return pair
        return Ed25519KeyPair(private_key=private_key, public_key=public_key)

    @staticmethod
    def _domain_selection(coordinates: Iterable[tuple[int, int]]) -> FrameSelectionV1:
        unique = sorted(
            {(int(x), int(y)) for x, y in coordinates},
            key=lambda item: (item[1], item[0]),
        )
        if not unique:
            raise ValueError("Select at least one frame")
        return FrameSelectionV1(
            row_ranges=tuple(
                FrameRowRange(y=y, x_start=x, x_end=x)
                for x, y in unique
            )
        )

    def create_review_batch(
        self,
        *,
        principal: Principal,
        project_id: ProjectId | str,
        layer_id: LayerId | str,
        image_representation_id: RepresentationId | str,
        vector_representation_id: RepresentationId | str,
        coordinates: Iterable[tuple[int, int]],
        assignee_id: PerformerId | str,
        instructions: str = "",
        due_at: datetime | None = None,
        idempotency_key: str,
    ) -> ReviewBatch:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project was not found")
        layer = next(
            (
                item
                for item in self.list_layers(project.id)
                if str(item.id) == str(layer_id)
            ),
            None,
        )
        if layer is None:
            raise ValueError("Layer was not found")
        representations = {
            str(item.id): item
            for item in self.list_representations(project.id, layer.id)
        }
        image_representation = representations.get(str(image_representation_id))
        vector_representation = representations.get(str(vector_representation_id))
        if (
            image_representation is None
            or image_representation.kind is not RepresentationKind.IMAGE
        ):
            raise ValueError("Select an image representation")
        if (
            vector_representation is None
            or vector_representation.kind is not RepresentationKind.VECTOR
            or vector_representation.source_image_representation_id
            != image_representation.id
        ):
            raise ValueError("The CIF representation is not linked to the selected images")

        selection = self._domain_selection(coordinates)
        projection = self._projection(project.id)
        items: list[ReviewItem] = []
        missing: list[str] = []
        for coordinate in selection.iter_coordinates():
            frame_id = coordinate.frame_id(project.id)
            image_series = deterministic_frame_series_id(
                image_representation.id,
                frame_id,
            )
            vector_series = deterministic_frame_series_id(
                vector_representation.id,
                frame_id,
            )
            image_version = projection.get_active_artifact_version(image_series)
            vector_version = projection.get_active_artifact_version(vector_series)
            if image_version is None or vector_version is None:
                absent = []
                if image_version is None:
                    absent.append("изображение")
                if vector_version is None:
                    absent.append("CIF")
                missing.append(
                    f"({coordinate.x}, {coordinate.y}): {', '.join(absent)}"
                )
                continue
            items.append(
                ReviewItem(
                    frame_id=frame_id,
                    vector_version_id=vector_version.id,
                    vector_sha256=vector_version.sha256,
                    image_version_id=image_version.id,
                )
            )
        if missing:
            raise ValueError(
                "Нельзя выдать кадры без обязательных файлов:\n"
                + "\n".join(missing)
            )
        return CreateReviewBatchHandler(
            self._uow(str(project.id)),
            self.profiles,
            self.clock,
            self.performers,
        )(
            CreateReviewBatchCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=project.id,
                layer_id=layer.id,
                selection=selection,
                items=tuple(items),
                assignee_id=PerformerId(str(assignee_id)),
                expected_layer_revision=layer.revision,
                instructions=instructions,
                due_at=due_at,
            )
        )

    def export_review_batch(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        destination: Path | str,
        idempotency_key: str,
    ) -> ReviewBatch:
        writer = ReviewPackageWriter(self._review_keys().private_key)
        return ExportReviewPackageHandler(
            self._uow(str(batch.project_id)),
            self.profiles,
            self.clock,
            writer,
        )(
            ExportReviewPackageCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=batch.project_id,
                batch_id=batch.id,
                expected_batch_revision=batch.revision,
                destination=str(destination),
                include_images=True,
            )
        )

    def review_return_preflight(
        self,
        *,
        principal: Principal,
        source: Path | str,
        idempotency_key: str,
    ):
        reader = ReviewPackageReader(self._review_keys().public_key)
        manifest = reader.read_manifest(source)
        batch_id = manifest.batch_id or manifest.package_id
        projection = self._projection(manifest.project_id)
        batch = projection.get_review_batch(batch_id)
        if batch is None:
            raise ValueError("Manifest does not match a known review batch")
        plan = DryRunReviewReturnHandler(
            self._uow(str(manifest.project_id)),
            self.profiles,
            self.clock,
            reader,
        )(
            DryRunReviewReturnCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=manifest.project_id,
                batch_id=batch.id,
                expected_batch_revision=batch.revision,
                source=str(source),
            )
        )
        return batch, plan

    def commit_review_return(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        source: Path | str,
        idempotency_key: str,
    ):
        reader = ReviewPackageReader(self._review_keys().public_key)
        return CommitReviewReturnHandler(
            self._uow(str(batch.project_id)),
            self.profiles,
            self.clock,
            reader,
        )(
            CommitReviewReturnCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=batch.project_id,
                batch_id=batch.id,
                expected_batch_revision=batch.revision,
                source=str(source),
            )
        )

    def accept_review(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        candidate_version_ids: Iterable[ArtifactVersionId | str],
        idempotency_key: str,
    ) -> ReviewBatch:
        return AcceptReviewHandler(
            self._uow(str(batch.project_id)),
            self.profiles,
            self.clock,
        )(
            AcceptReviewCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=batch.project_id,
                batch_id=batch.id,
                expected_batch_revision=batch.revision,
                candidate_version_ids=tuple(
                    ArtifactVersionId(str(value))
                    for value in candidate_version_ids
                ),
            )
        )

    def request_review_changes(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        reason: str,
        idempotency_key: str,
    ) -> ReviewBatch:
        return RequestReviewChangesHandler(
            self._uow(str(batch.project_id)),
            self.profiles,
            self.clock,
        )(
            RequestReviewChangesCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=batch.project_id,
                batch_id=batch.id,
                expected_batch_revision=batch.revision,
                reason=reason,
            )
        )

    def cancel_review_batch(
        self,
        *,
        principal: Principal,
        batch: ReviewBatch,
        idempotency_key: str,
    ) -> ReviewBatch:
        return CancelReviewBatchHandler(
            self._uow(str(batch.project_id)),
            self.profiles,
            self.clock,
        )(
            CancelReviewBatchCommand(
                context=CommandContext(
                    actor=principal,
                    idempotency_key=idempotency_key,
                ),
                project_id=batch.project_id,
                batch_id=batch.id,
                expected_batch_revision=batch.revision,
            )
        )

    def review_candidate_version_ids(
        self,
        batch: ReviewBatch,
    ) -> tuple[ArtifactVersionId, ...]:
        latest_by_series: dict[str, ArtifactVersionId] = {}
        projection = self._projection(batch.project_id)
        for event in self.history(batch.project_id):
            if (
                event.event_type != "ReviewReturnCommitted"
                or str(event.payload.get("review_batch_id", "")) != str(batch.id)
            ):
                continue
            for value in event.payload.get("candidate_version_ids", ()):
                identifier = ArtifactVersionId(str(value))
                version = projection.get_artifact_version(identifier)
                if version is not None:
                    latest_by_series[str(version.series_id)] = identifier
        return tuple(latest_by_series[key] for key in sorted(latest_by_series))

    def scan_integrity(self) -> IntegrityScanResult:
        event_count = 0
        blob_count = 0
        errors: list[str] = []
        projects = self.list_projects(include_archived=True)
        for project in projects:
            try:
                events = FilesystemEventStore(self.catalog_root, str(project.id))
                verified_events, _ = events.verify()
                event_count += verified_events
                blobs = FilesystemBlobStore.for_project(self.catalog_root, str(project.id))
                blob_count += len(blobs.verify_all())
                self._projection(project.id)
            except Exception as exc:
                errors.append(f"{project.name} ({project.id}): {exc}")
        return IntegrityScanResult(len(projects), event_count, blob_count, tuple(errors))

    def export_backup(
        self,
        project_id: ProjectId | str,
        destination: Path | str,
        *,
        principal: Principal | None = None,
    ) -> KrakenMigrationBundleV1:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project was not found")
        if principal is not None:
            AuthorizationPolicy().require(
                principal=principal,
                storage=self.profile,
                permission=Permission.MIGRATE_PROJECT,
                roles=self.project_roles(project.id, principal.id),
            )
        events = FilesystemEventStore(self.catalog_root, str(project.id))
        blobs = FilesystemBlobStore.for_project(self.catalog_root, str(project.id))
        return CanonicalBundleExporter(
            events, blobs, source_profile=self.profile.id
        ).export(destination)

    def import_backup(
        self,
        bundle_root: Path | str,
        *,
        principal: Principal | None = None,
        take_ownership: bool = False,
    ) -> Project:
        root = Path(bundle_root).resolve(strict=True)
        report = BundleVerifier().verify(root)
        report.raise_for_errors()
        manifest = load_bundle_manifest(root)
        available_owners = tuple(
            candidate
            for candidate in self.identities.list()
            if ProjectRole.OWNER
            in self.project_roles(manifest.project_id, candidate.id)
        )
        if principal is not None:
            if available_owners:
                AuthorizationPolicy().require(
                    principal=principal,
                    storage=self.profile,
                    permission=Permission.MIGRATE_PROJECT,
                    roles=self.project_roles(manifest.project_id, principal.id),
                )
            elif not take_ownership:
                raise ValueError(
                    "В локальном хранилище нет доступного владельца. "
                    "Для восстановления явно подтвердите принятие владения."
                )
        events = FilesystemEventStore(self.catalog_root, manifest.project_id)
        blobs = FilesystemBlobStore.for_project(self.catalog_root, manifest.project_id)
        CanonicalBundleImporter(events, blobs).import_bundle(root, verify_first=False)
        rebuild_filesystem_index(events, SQLiteProjectionStore(self.catalog_root, manifest.project_id), acl=self.identities)
        project = self.get_project(manifest.project_id)
        if project is None:
            raise RuntimeError("Imported backup has no project projection")
        if principal is not None and not available_owners:
            self.identities.assign(
                ProjectRoleAssignment.create(
                    project_id=project.id,
                    principal_id=principal.id,
                    role=ProjectRole.OWNER,
                    assigned_by=principal.id,
                    assigned_at=self.clock.now(),
                )
            )
        return project

    def attach_project(
        self,
        project_directory: Path | str,
        *,
        principal: Principal | None = None,
        take_ownership: bool = False,
    ) -> Project:
        source = Path(project_directory).resolve(strict=True)
        descriptor = json.loads((source / "project.json").read_text(encoding="utf-8"))
        project_id = str(descriptor["project_id"])
        expected = FileProjectLayout(self.catalog_root, project_id).project_dir.resolve()
        if source != expected:
            raise ValueError("v1 attaches only projects already located in the configured local catalog")
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project cannot be reconstructed from its authoritative event log")
        if principal is not None:
            available_owners = tuple(
                candidate
                for candidate in self.identities.list()
                if ProjectRole.OWNER in self.project_roles(project.id, candidate.id)
            )
            if available_owners:
                AuthorizationPolicy().require(
                    principal=principal,
                    storage=self.profile,
                    permission=Permission.MIGRATE_PROJECT,
                    roles=self.project_roles(project.id, principal.id),
                )
            elif not take_ownership:
                raise ValueError(
                    "В локальном хранилище нет доступного владельца. "
                    "Для подключения явно подтвердите принятие владения."
                )
            else:
                self.identities.assign(
                    ProjectRoleAssignment.create(
                        project_id=project.id,
                        principal_id=principal.id,
                        role=ProjectRole.OWNER,
                        assigned_by=principal.id,
                        assigned_at=self.clock.now(),
                    )
                )
        return project


__all__ = [
    "DesktopSession",
    "DesktopStorageProfiles",
    "EmbeddedProjectService",
    "FrameCellSnapshot",
    "IntegrityScanResult",
    "ManagedImportResult",
    "ProjectDeletionResult",
    "SystemClock",
    "default_data_dir",
]
