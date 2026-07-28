"""Desktop composition root for authenticated local file projects."""

from __future__ import annotations

import json
import hashlib
import mimetypes
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kraken_core.safe_files import ensure_regular_directory, open_regular_read

from kraken_manager.application.dto import (
    AddArtifactVersionCommand,
    ActivateRepresentationCommand,
    AssignProjectRoleCommand,
    ArchiveLayerCommand,
    ArchiveProjectCommand,
    ArchiveRepresentationCommand,
    CommandContext,
    CreateLayerCommand,
    CreateArtifactSeriesCommand,
    CreateProjectCommand,
    CreateRepresentationCommand,
    RenameLayerCommand,
    RenameProjectCommand,
    RenameRepresentationCommand,
    ReorderLayerCommand,
    ReorderLayersCommand,
    RestoreProjectCommand,
    RevokeProjectRoleCommand,
    UpdateRepresentationNoteCommand,
)
from kraken_manager.application.imports import ImportMappingMode, ImportPlan, ImportPlanner, ImportSource
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
from kraken_manager.application.representation_lifecycle import (
    ActivateRepresentationHandler,
    ArchiveRepresentationHandler,
    RenameRepresentationHandler,
    UpdateRepresentationNoteHandler,
)
from kraken_manager.domain.artifacts import ArtifactScope, ArtifactVersion, deterministic_frame_series_id
from kraken_manager.domain.common import LayerId, PrincipalId, ProjectId, RepresentationId, new_uuid
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
from kraken_manager.domain.identity import Performer, Principal, ProjectRole
from kraken_manager.domain.project import (
    GridOrientation,
    Layer,
    LayerType,
    Project,
    Representation,
    RepresentationKind,
    RepresentationPurpose,
)
from kraken_manager.domain.workflows import ReviewBatch
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


class EmbeddedProjectService:
    """Autonomous application service for machine-local file projects."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir or default_data_dir()).resolve()
        self.catalog_root = self.data_dir / "catalog"
        self.catalog_root.mkdir(parents=True, exist_ok=True)
        self.accounts = LocalAccountStore(self.data_dir / "accounts.sqlite3", ScryptPasswordHasher())
        self.identities = LocalIdentityAclStore(self.data_dir / "identity.sqlite3")
        self.performers = LocalSQLitePerformerStore(self.data_dir / "identity.sqlite3")
        self.profile = filesystem_storage_profile(str(self.catalog_root))
        self._external_source_indexes: dict[tuple[str, str, int], tuple[Path, ...]] = {}
        self.profiles = DesktopStorageProfiles(self.profile)
        self.clock = SystemClock()

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

    def project_roles(
        self, project_id: ProjectId | str, principal_id: PrincipalId | str
    ) -> frozenset[ProjectRole]:
        return self.identities.roles_for(ProjectId(str(project_id)), PrincipalId(str(principal_id)))

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
    ) -> Project:
        command = CreateProjectCommand(
            context=CommandContext(actor=principal, idempotency_key=idempotency_key),
            name=name,
            width=width,
            height=height,
            orientation=orientation,
            storage_profile_id=self.profile.id,
        )
        assert command.project_id is not None
        project_id = str(command.project_id)
        project = CreateProjectHandler(self._uow(project_id), self.profiles, self.clock)(command)
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

    def create_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        name: str,
        layer_type: LayerType,
        order: int,
        idempotency_key: str,
    ) -> Layer:
        command = CreateLayerCommand(
            context=CommandContext(actor=principal, idempotency_key=idempotency_key),
            project_id=project.id,
            name=name,
            type=layer_type,
            order=order,
            expected_project_revision=project.revision,
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
        return RenameProjectHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            RenameProjectCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id,
                name=name,
                expected_revision=project.revision,
            )
        )

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

    def rename_layer(
        self,
        *,
        principal: Principal,
        project: Project,
        layer: Layer,
        name: str,
        idempotency_key: str,
    ) -> Layer:
        return RenameLayerHandler(self._uow(str(project.id)), self.profiles, self.clock)(
            RenameLayerCommand(
                context=CommandContext(actor=principal, idempotency_key=idempotency_key),
                project_id=project.id,
                layer_id=layer.id,
                name=name,
                expected_revision=layer.revision,
            )
        )

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
                    if all(coordinate in coverage.get(identifier, set()) for identifier in managed):
                        continue
                    merged[coordinate] = {
                        "artifact_version_id": "",
                        "frame_id": "",
                        "sha256": "",
                        "status": "error",
                        "missing": True,
                        "x": x,
                        "y": y,
                    }

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
            paths = tuple(
                sorted(
                    (
                        path
                        for path in root.iterdir()
                        if path.is_file()
                        and (mimetypes.guess_type(path.name)[0] or "").startswith("image/")
                    ),
                    key=self._natural_path_key,
                )
            )
            self._external_source_indexes = {
                key: value
                for key, value in self._external_source_indexes.items()
                if key[0] != str(representation.id)
            }
            self._external_source_indexes[cache_key] = paths

        cells: list[dict[str, Any]] = []
        analysis = self.latest_karakal_analysis(project.id, representation.layer_id)
        confidence = {} if analysis is None else analysis.frame_confidence
        first_index = (y1 - 1) * project.width + (x1 - 1)
        last_index = min(len(paths), (y2 - 1) * project.width + x2)
        for index in range(first_index, last_index):
            x = index % project.width + 1
            y = index // project.width + 1
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                continue
            path = paths[index]
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
        self, project_id: ProjectId | str, destination: Path | str
    ) -> KrakenMigrationBundleV1:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project was not found")
        events = FilesystemEventStore(self.catalog_root, str(project.id))
        blobs = FilesystemBlobStore.for_project(self.catalog_root, str(project.id))
        return CanonicalBundleExporter(
            events, blobs, source_profile=self.profile.id
        ).export(destination)

    def import_backup(self, bundle_root: Path | str) -> Project:
        root = Path(bundle_root).resolve(strict=True)
        report = BundleVerifier().verify(root)
        report.raise_for_errors()
        manifest = load_bundle_manifest(root)
        events = FilesystemEventStore(self.catalog_root, manifest.project_id)
        blobs = FilesystemBlobStore.for_project(self.catalog_root, manifest.project_id)
        CanonicalBundleImporter(events, blobs).import_bundle(root, verify_first=False)
        rebuild_filesystem_index(events, SQLiteProjectionStore(self.catalog_root, manifest.project_id), acl=self.identities)
        project = self.get_project(manifest.project_id)
        if project is None:
            raise RuntimeError("Imported backup has no project projection")
        return project

    def attach_project(self, project_directory: Path | str) -> Project:
        source = Path(project_directory).resolve(strict=True)
        descriptor = json.loads((source / "project.json").read_text(encoding="utf-8"))
        project_id = str(descriptor["project_id"])
        expected = FileProjectLayout(self.catalog_root, project_id).project_dir.resolve()
        if source != expected:
            raise ValueError("v1 attaches only projects already located in the configured local catalog")
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Project cannot be reconstructed from its authoritative event log")
        return project


__all__ = [
    "DesktopSession",
    "DesktopStorageProfiles",
    "EmbeddedProjectService",
    "FrameCellSnapshot",
    "IntegrityScanResult",
    "ManagedImportResult",
    "SystemClock",
    "default_data_dir",
]
