"""Desktop composition root for authenticated local file projects."""

from __future__ import annotations

import json
import mimetypes
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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
from kraken_manager.domain.common import LayerId, PrincipalId, ProjectId, RepresentationId
from kraken_manager.domain.identity import Performer, Principal, ProjectRole
from kraken_manager.domain.project import GridOrientation, Layer, LayerType, Project, Representation, RepresentationKind
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
            cells[str(series.frame_id)] = FrameCellSnapshot(
                coordinate[0],
                coordinate[1],
                status,
                str(series.frame_id),
                str(version.id),
                version.sha256,
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
                    )
        return tuple(sorted(cells.values(), key=lambda item: (item.y, item.x)))

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
