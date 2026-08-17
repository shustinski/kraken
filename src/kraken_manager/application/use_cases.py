"""Command handlers for the first local/shared project-manager vertical slice."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    AddArtifactVersionCommand,
    CreateArtifactSeriesCommand,
    CreateLayerCommand,
    CreateProjectCommand,
    CreateRepresentationCommand,
    ReturnedFileDigest,
    StoredContent,
)
from kraken_manager.application.errors import ConcurrencyError, ConflictError, NotFoundError, StorageCapabilityError
from kraken_manager.application.ports import Clock, StorageProfile, StorageProfileCatalog, UnitOfWork, UnitOfWorkFactory
from kraken_manager.domain.artifacts import ArtifactSeries, ArtifactVersion, validate_sha256
from kraken_manager.domain.common import (
    ArtifactSeriesId,
    ArtifactVersionId,
    FrameId,
    LayerId,
    ProjectId,
    RepresentationId,
)
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope
from kraken_manager.domain.identity import Permission, Principal, ProjectRole, ProjectRoleAssignment
from kraken_manager.domain.project import (
    Layer,
    Project,
    ProjectState,
    Representation,
    RepresentationKind,
    RepresentationPurpose,
    StructureState,
)
from kraken_manager.domain.workflows import (
    ReviewComparisonReport,
    ReviewFileComparison,
    ReviewFileStatus,
    ReviewPackageManifestV1,
    validate_relative_manifest_path,
)


def _project_stream(project_id: ProjectId) -> str:
    return f"project:{project_id}"


def _layer_stream(layer_id: str) -> str:
    return f"layer:{layer_id}"


def _series_stream(series_id: str) -> str:
    return f"artifact-series:{series_id}"


def _prior_entity_id(
    uow: UnitOfWork,
    project_id: ProjectId,
    key: str,
    event_type: str,
    field: str,
) -> str | None:
    events = uow.event_store.find_by_idempotency_key(project_id, key)
    matching = [event for event in events if event.event_type == event_type]
    if not matching:
        return None
    value = matching[-1].payload.get(field)
    return str(value) if value is not None else None


class _ProjectHandler:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        storage_profiles: StorageProfileCatalog,
        clock: Clock,
        authorization: AuthorizationPolicy | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._storage_profiles = storage_profiles
        self._clock = clock
        self._authorization = authorization or AuthorizationPolicy()

    def _profile(self, profile_id: str) -> StorageProfile:
        profile = self._storage_profiles.get(profile_id)
        if profile is None:
            raise NotFoundError(f"Storage profile {profile_id!r} was not found")
        return profile

    def _load_project_and_authorize(
        self,
        uow: UnitOfWork,
        *,
        project_id: ProjectId,
        actor: Principal,
        permission: Permission,
        gitlab_identity_verified: bool,
    ) -> tuple[Project, StorageProfile]:
        project = uow.projections.get_project(project_id)
        if project is None:
            raise NotFoundError(f"Project {project_id} was not found")
        if project.state is ProjectState.ARCHIVED:
            raise ConflictError("Archived project is read-only")
        profile = self._profile(project.storage_profile)
        roles = uow.acl.roles_for(project.id, actor.id)
        self._authorization.decide(
            principal=actor,
            storage=profile,
            permission=permission,
            roles=roles,
            gitlab_identity_verified=gitlab_identity_verified,
        ).require()
        return project, profile


class CreateProjectHandler(_ProjectHandler):
    def __call__(self, command: CreateProjectCommand) -> Project:
        profile = self._profile(command.storage_profile_id)
        self._authorization.decide_create_project(
            principal=command.context.actor,
            storage=profile,
            gitlab_identity_verified=command.context.gitlab_identity_verified,
        ).require()
        requested_frames = command.width * command.height
        if profile.capabilities.max_frames is not None and requested_frames > profile.capabilities.max_frames:
            raise StorageCapabilityError(
                f"Storage profile supports at most {profile.capabilities.max_frames} frames, got {requested_frames}"
            )
        with self._uow_factory() as uow:
            previous_id = _prior_entity_id(
                uow,
                command.project_id,
                command.context.idempotency_key,
                "ProjectCreated",
                "project_id",
            )
            if previous_id is not None:
                existing = uow.projections.get_project(ProjectId(previous_id))
                if existing is not None:
                    return existing
            now = self._clock.now()
            project = Project.create(
                name=command.name,
                width=command.width,
                height=command.height,
                orientation=command.orientation,
                storage_profile=profile.id,
                project_id=command.project_id,
                created_at=now,
            )
            persisted = replace(project, revision=1)
            event = EventEnvelope.create(
                stream_id=_project_stream(project.id),
                project_id=project.id,
                revision=1,
                event_type="ProjectCreated",
                payload={
                    "project_id": project.id,
                    "name": project.name,
                    "width": project.width,
                    "height": project.height,
                    "orientation": project.orientation.value,
                    "storage_profile": project.storage_profile,
                    "state": project.state.value,
                    "created_at": project.created_at.isoformat(),
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=command.context.performer_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(_project_stream(project.id), expected_revision=0, events=(event,))
            uow.projections.save_project(persisted)
            uow.acl.assign(
                ProjectRoleAssignment.create(
                    project_id=project.id,
                    principal_id=command.context.actor.id,
                    role=ProjectRole.OWNER,
                    assigned_by=command.context.actor.id,
                    assigned_at=now,
                )
            )
            uow.commit()
            return persisted


class CreateLayerHandler(_ProjectHandler):
    def __call__(self, command: CreateLayerCommand) -> Layer:
        with self._uow_factory() as uow:
            project, _ = self._load_project_and_authorize(
                uow,
                project_id=command.project_id,
                actor=command.context.actor,
                permission=Permission.MANAGE_STRUCTURE,
                gitlab_identity_verified=command.context.gitlab_identity_verified,
            )
            previous_id = _prior_entity_id(
                uow, project.id, command.context.idempotency_key, "LayerCreated", "layer_id"
            )
            if previous_id is not None:
                existing = uow.projections.get_layer(LayerId(previous_id))
                if existing is not None:
                    return existing
            if project.revision != command.expected_project_revision:
                raise ConcurrencyError(
                    f"Expected project revision {command.expected_project_revision}, found {project.revision}"
                )
            if any(layer.name.casefold() == command.name.strip().casefold() for layer in uow.projections.list_layers(project.id)):
                raise ConflictError(f"Layer name {command.name!r} already exists in the project")
            now = self._clock.now()
            layer = Layer.create(
                project_id=project.id,
                name=command.name,
                type=command.type,
                order=command.order,
                layer_id=command.layer_id,
                created_at=now,
            )
            next_project = replace(project, revision=project.revision + 1)
            event = EventEnvelope.create(
                stream_id=_project_stream(project.id),
                project_id=project.id,
                revision=next_project.revision,
                event_type="LayerCreated",
                payload={
                    "layer_id": layer.id,
                    "name": layer.name,
                    "type": layer.type.value,
                    "order": layer.order,
                    "state": layer.state.value,
                    "created_at": layer.created_at.isoformat(),
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=command.context.performer_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(
                _project_stream(project.id),
                expected_revision=project.revision,
                events=(event,),
            )
            uow.projections.save_layer(layer)
            uow.projections.save_project(next_project)
            uow.commit()
            return layer


class CreateRepresentationHandler(_ProjectHandler):
    def __call__(self, command: CreateRepresentationCommand) -> Representation:
        with self._uow_factory() as uow:
            project, _ = self._load_project_and_authorize(
                uow,
                project_id=command.project_id,
                actor=command.context.actor,
                permission=Permission.MANAGE_STRUCTURE,
                gitlab_identity_verified=command.context.gitlab_identity_verified,
            )
            layer = uow.projections.get_layer(command.layer_id)
            if layer is None or layer.project_id != project.id:
                raise NotFoundError("Layer was not found in the project")
            if layer.state is StructureState.ARCHIVED:
                raise ConflictError("Archived layer is read-only")
            previous_id = _prior_entity_id(
                uow,
                project.id,
                command.context.idempotency_key,
                "RepresentationCreated",
                "representation_id",
            )
            if previous_id is not None:
                existing = uow.projections.get_representation(RepresentationId(previous_id))
                if existing is not None:
                    return existing
            if layer.revision != command.expected_layer_revision:
                raise ConcurrencyError(
                    f"Expected layer revision {command.expected_layer_revision}, found {layer.revision}"
                )
            existing_representations = uow.projections.list_representations(layer.id)
            if any(item.name.casefold() == command.name.strip().casefold() for item in existing_representations):
                raise ConflictError(f"Representation name {command.name!r} already exists in the layer")
            source_image_id = command.source_image_representation_id
            has_image_parent = (
                command.kind is RepresentationKind.VECTOR
                or command.purpose is RepresentationPurpose.BINARY
            )
            if has_image_parent:
                if source_image_id is None:
                    active_images = [
                        item
                        for item in existing_representations
                        if item.kind is RepresentationKind.IMAGE and item.active
                    ]
                    if len(active_images) == 1:
                        source_image_id = active_images[0].id
                source_image = next(
                    (
                        item
                        for item in existing_representations
                        if item.id == source_image_id and item.kind is RepresentationKind.IMAGE
                    ),
                    None,
                )
                if source_image_id is not None and source_image is None:
                    raise ConflictError("A derived representation must belong to an image representation")
            elif source_image_id is not None:
                raise ConflictError("A source image representation cannot have a parent image")
            now = self._clock.now()
            representation = Representation.create(
                project_id=project.id,
                layer_id=layer.id,
                name=command.name,
                kind=command.kind,
                note=command.note,
                source=command.source,
                source_image_representation_id=source_image_id,
                active=command.active,
                purpose=command.purpose,
                created_at=now,
            )
            deactivated_ids: list[str] = []
            if representation.active:
                for previous in existing_representations:
                    if (
                        previous.kind is representation.kind
                        and previous.active
                        and (
                            representation.kind is not RepresentationKind.VECTOR
                            or previous.source_image_representation_id
                            == representation.source_image_representation_id
                        )
                    ):
                        uow.projections.save_representation(previous.deactivate())
                        deactivated_ids.append(str(previous.id))
            next_layer = replace(layer, revision=layer.revision + 1)
            event = EventEnvelope.create(
                stream_id=_layer_stream(str(layer.id)),
                project_id=project.id,
                revision=next_layer.revision,
                event_type="RepresentationCreated",
                payload={
                    "representation_id": representation.id,
                    "layer_id": layer.id,
                    "name": representation.name,
                    "kind": representation.kind.value,
                    "purpose": representation.purpose.value,
                    "note": representation.note,
                    "source": representation.source,
                    "source_image_representation_id": representation.source_image_representation_id,
                    "active": representation.active,
                    "state": representation.state.value,
                    "created_at": representation.created_at.isoformat(),
                    "deactivated_representation_ids": deactivated_ids,
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=command.context.performer_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(
                _layer_stream(str(layer.id)),
                expected_revision=layer.revision,
                events=(event,),
            )
            uow.projections.save_representation(representation)
            uow.projections.save_layer(next_layer)
            uow.commit()
            return representation


class AddArtifactVersionHandler(_ProjectHandler):
    """Store content immutably and create a branch instead of stale overwrite."""

    def preflight(self, command: AddArtifactVersionCommand) -> ArtifactVersion | None:
        """Authorize and validate before a potentially very large direct upload."""
        with self._uow_factory() as uow:
            existing, *_ = self._prepare(uow, command)
            return existing

    def __call__(
        self,
        command: AddArtifactVersionCommand,
        chunks: Iterable[bytes] | None = None,
        *,
        stored: StoredContent | None = None,
    ) -> ArtifactVersion:
        if (chunks is None) == (stored is None):
            raise ValueError("Exactly one of chunks or stored content is required")
        with self._uow_factory() as uow:
            existing, project, series, active, parent_id, stream_id, actual_revision = self._prepare(uow, command)
            if existing is not None:
                return existing
            if stored is None:
                assert chunks is not None
                stored = uow.blobs.put(chunks, expected_sha256=command.expected_sha256)
            elif command.expected_sha256 is not None and stored.blob.sha256 != command.expected_sha256:
                raise ConflictError("Stored blob digest does not match the artifact command")
            now = self._clock.now()
            version = ArtifactVersion.managed(
                series_id=series.id,
                blob=stored.blob,
                media_type=command.media_type,
                filename=command.filename,
                author_principal_id=command.context.actor.id,
                created_at=now,
                parent_version_id=parent_id,
                input_version_ids=command.input_version_ids,
                tool_name=command.tool_name,
                tool_version=command.tool_version,
                parameters=command.parameters,
            )
            activate = active is None or parent_id == active.id
            event = EventEnvelope.create(
                stream_id=stream_id,
                project_id=project.id,
                revision=actual_revision + 1,
                event_type="ArtifactVersionCreated",
                payload={
                    "artifact_version_id": version.id,
                    "series_id": series.id,
                    "layer_id": series.layer_id,
                    "representation_id": series.representation_id,
                    "frame_id": series.frame_id,
                    "sha256": version.sha256,
                    "size_bytes": version.size_bytes,
                    "media_type": version.media_type,
                    "filename": version.filename,
                    "blob": {
                        "sha256": version.blob.sha256,
                        "size_bytes": version.blob.size_bytes,
                    },
                    "parent_version_id": version.parent_version_id,
                    "input_version_ids": version.input_version_ids,
                    "tool_name": version.tool_name,
                    "tool_version": version.tool_version,
                    "parameters": version.parameters,
                    "author_principal_id": version.author_principal_id,
                    "created_at": version.created_at.isoformat(),
                    "activated": activate,
                    "branched": active is not None and parent_id != active.id,
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=command.context.performer_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(stream_id, expected_revision=actual_revision, events=(event,))
            uow.projections.save_artifact_version(version, activate=activate)
            uow.commit()
            return version

    def _prepare(
        self,
        uow: UnitOfWork,
        command: AddArtifactVersionCommand,
    ) -> tuple[ArtifactVersion | None, Project, ArtifactSeries, ArtifactVersion | None, ArtifactVersionId | None, str, int]:
        project, _ = self._load_project_and_authorize(
            uow,
            project_id=command.project_id,
            actor=command.context.actor,
            permission=Permission.IMPORT_ARTIFACT,
            gitlab_identity_verified=command.context.gitlab_identity_verified,
        )
        previous_id = _prior_entity_id(
            uow,
            project.id,
            command.context.idempotency_key,
            "ArtifactVersionCreated",
            "artifact_version_id",
        )
        series = uow.projections.get_artifact_series(command.series_id)
        if series is None or series.project_id != project.id:
            raise NotFoundError("Artifact series was not found in the project")
        stream_id = _series_stream(str(series.id))
        actual_revision = uow.event_store.current_revision(stream_id)
        active = uow.projections.get_active_artifact_version(series.id)
        parent_id = command.parent_version_id or (active.id if active is not None else None)
        if previous_id is not None:
            existing = uow.projections.get_artifact_version(ArtifactVersionId(previous_id))
            if existing is not None:
                return existing, project, series, active, parent_id, stream_id, actual_revision
        if actual_revision != command.expected_series_revision:
            raise ConcurrencyError(
                f"Expected artifact series revision {command.expected_series_revision}, found {actual_revision}"
            )
        if parent_id is not None:
            parent = uow.projections.get_artifact_version(parent_id)
            if parent is None or parent.series_id != series.id:
                raise ConflictError("Parent artifact version does not belong to this series")
        for input_version_id in command.input_version_ids:
            if uow.projections.get_artifact_version(input_version_id) is None:
                raise NotFoundError(f"Input artifact version {input_version_id} was not found")
        return None, project, series, active, parent_id, stream_id, actual_revision


class CreateArtifactSeriesHandler(_ProjectHandler):
    """Create the stable logical identity before immutable versions are imported."""

    def __call__(self, command: CreateArtifactSeriesCommand) -> ArtifactSeries:
        with self._uow_factory() as uow:
            project, _ = self._load_project_and_authorize(
                uow,
                project_id=command.project_id,
                actor=command.context.actor,
                permission=Permission.IMPORT_ARTIFACT,
                gitlab_identity_verified=command.context.gitlab_identity_verified,
            )
            previous_id = _prior_entity_id(
                uow,
                project.id,
                command.context.idempotency_key,
                "ArtifactSeriesCreated",
                "artifact_series_id",
            )
            if previous_id is not None:
                existing = uow.projections.get_artifact_series(ArtifactSeriesId(previous_id))
                if existing is not None:
                    return existing
            series = ArtifactSeries(
                id=command.series_id,
                project_id=project.id,
                scope=command.scope,
                name=command.name,
                layer_id=command.layer_id,
                representation_id=command.representation_id,
                frame_id=command.frame_id,
            )
            if series.layer_id is not None:
                layer = uow.projections.get_layer(series.layer_id)
                if layer is None or layer.project_id != project.id:
                    raise NotFoundError("Artifact series layer was not found in the project")
            if series.representation_id is not None:
                representation = uow.projections.get_representation(series.representation_id)
                if representation is None or representation.project_id != project.id:
                    raise NotFoundError("Artifact series representation was not found in the project")
            now = self._clock.now()
            stream_id = _series_stream(str(series.id))
            event = EventEnvelope.create(
                stream_id=stream_id,
                project_id=project.id,
                revision=1,
                event_type="ArtifactSeriesCreated",
                payload={
                    "artifact_series_id": series.id,
                    "scope": series.scope.value,
                    "name": series.name,
                    "layer_id": series.layer_id,
                    "representation_id": series.representation_id,
                    "frame_id": series.frame_id,
                    "archived": series.archived,
                },
                actor=ActorSnapshot.from_principal(command.context.actor),
                recorded_at=now,
                effective_at=command.context.effective_at,
                performer_id=command.context.performer_id,
                correlation_id=command.context.correlation_id,
                idempotency_key=command.context.idempotency_key,
            )
            uow.event_store.append(stream_id, expected_revision=0, events=(event,))
            uow.projections.save_artifact_series(series)
            uow.commit()
            return series


class ReviewReturnComparator:
    """Dry-run a returned folder using strict byte-content SHA-256 equality."""

    def compare(
        self,
        *,
        manifest: ReviewPackageManifestV1,
        returned_files: Sequence[ReturnedFileDigest],
        active_versions: Mapping[FrameId, ArtifactVersionId] | None = None,
    ) -> ReviewComparisonReport:
        active_versions = active_versions or {}
        by_path: dict[str, list[ReturnedFileDigest]] = defaultdict(list)
        unsafe: list[tuple[int, ReturnedFileDigest]] = []
        for index, returned in enumerate(returned_files):
            try:
                normalized = validate_relative_manifest_path(returned.relative_path).casefold()
            except (ValueError, TypeError):
                unsafe.append((index, returned))
                continue
            by_path[normalized].append(returned)

        comparisons: list[ReviewFileComparison] = []
        expected_keys: set[str] = set()
        for expected in manifest.files:
            key = expected.relative_path.casefold()
            expected_keys.add(key)
            candidates = by_path.get(key, [])
            if not candidates:
                status = ReviewFileStatus.MISSING
                returned_sha = None
            elif len(candidates) > 1:
                status = ReviewFileStatus.DUPLICATE
                returned_sha = None
            else:
                candidate = candidates[0]
                try:
                    returned_sha = validate_sha256(candidate.sha256 or "", field="returned_file.sha256")
                except ValueError:
                    returned_sha = None
                if not candidate.valid or returned_sha is None:
                    status = ReviewFileStatus.INVALID
                elif active_versions.get(expected.frame_id, expected.artifact_version_id) != expected.artifact_version_id:
                    status = ReviewFileStatus.STALE_BASE_CONFLICT
                elif returned_sha == expected.sha256:
                    status = ReviewFileStatus.UNCHANGED
                else:
                    status = ReviewFileStatus.CHANGED
            comparisons.append(
                ReviewFileComparison(
                    status=status,
                    relative_path=expected.relative_path,
                    frame_id=expected.frame_id,
                    expected_version_id=expected.artifact_version_id,
                    expected_sha256=expected.sha256,
                    returned_sha256=returned_sha,
                )
            )

        for key in sorted(set(by_path) - expected_keys):
            candidates = by_path[key]
            status = ReviewFileStatus.DUPLICATE if len(candidates) > 1 else ReviewFileStatus.EXTRA
            returned_sha: str | None
            try:
                returned_sha = validate_sha256(candidates[0].sha256 or "")
            except ValueError:
                returned_sha = None
                status = ReviewFileStatus.INVALID
            comparisons.append(
                ReviewFileComparison(
                    status=status,
                    relative_path=candidates[0].relative_path,
                    returned_sha256=returned_sha,
                )
            )
        for index, returned in unsafe:
            comparisons.append(
                ReviewFileComparison(
                    status=ReviewFileStatus.INVALID,
                    relative_path=returned.relative_path or f"invalid-{index}",
                )
            )
        return ReviewComparisonReport(tuple(comparisons))


__all__ = [
    "AddArtifactVersionHandler",
    "CreateArtifactSeriesHandler",
    "CreateLayerHandler",
    "CreateProjectHandler",
    "CreateRepresentationHandler",
    "ReviewReturnComparator",
]
