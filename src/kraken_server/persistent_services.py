"""Production service facade backed by the clean-architecture application API."""

# This adapter is an exception-translation boundary around application handlers.
# ruff: noqa: BLE001

from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from kraken_manager.application.acl import AssignProjectRoleHandler, RevokeProjectRoleHandler
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    AcceptReviewCommand,
    ActivateRepresentationCommand,
    ActivateArtifactVersionCommand,
    AddArtifactVersionCommand,
    AddExternalArtifactVersionCommand,
    ArchiveArtifactSeriesCommand,
    ArchiveLayerCommand,
    ArchiveProjectCommand,
    ArchiveRepresentationCommand,
    AssignProjectRoleCommand,
    CreateLayerCommand,
    CreateArtifactSeriesCommand,
    CreateNoteCommand,
    CreateProjectCommand,
    CreateRepresentationCommand,
    DeactivateRepresentationCommand,
    RenameLayerCommand,
    RenameArtifactSeriesCommand,
    RenameProjectCommand,
    RenameRepresentationCommand,
    ReorderLayerCommand,
    ReorderLayersCommand,
    RestoreProjectCommand,
    RevokeProjectRoleCommand,
    ReviseNoteCommand,
    SubmitPluginJobCommand,
    CancelPluginJobCommand,
    CancelReviewBatchCommand,
    CreateReviewBatchCommand,
    CommitReviewReturnCommand,
    DryRunReviewReturnCommand,
    ExportReviewPackageCommand,
    ImportPluginResultCommand,
    RequestReviewChangesCommand,
    StorageBackendKind,
    StorageScope,
    UpdateRepresentationNoteCommand,
)
from kraken_manager.application.dto import (
    CommandContext as ApplicationCommandContext,
)
from kraken_manager.application.errors import (
    AuthorizationError as ApplicationAuthorizationError,
)
from kraken_manager.application.errors import (
    ConcurrencyError as ApplicationConcurrencyError,
)
from kraken_manager.application.errors import (
    ConflictError as ApplicationConflictError,
)
from kraken_manager.application.errors import (
    NotFoundError as ApplicationNotFoundError,
)
from kraken_manager.application.errors import (
    StorageCapabilityError,
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
from kraken_manager.application.artifact_lifecycle import (
    ActivateArtifactVersionHandler,
    AddExternalArtifactVersionHandler,
    ArchiveArtifactSeriesHandler,
    CreateNoteHandler,
    RenameArtifactSeriesHandler,
    ReviseNoteHandler,
)
from kraken_manager.application.plugin_jobs import (
    CancelPluginJobHandler,
    ImportPluginResultHandler,
    SubmitPluginJobHandler,
)
from kraken_manager.application.review_workflow import (
    AcceptReviewHandler,
    CancelReviewBatchHandler,
    CreateReviewBatchHandler,
    CommitReviewReturnHandler,
    DryRunReviewReturnHandler,
    ExportReviewPackageHandler,
    RequestReviewChangesHandler,
)
from kraken_manager.application.ports import StorageCapabilities, StorageProfile
from kraken_manager.application.representation_lifecycle import (
    ActivateRepresentationHandler,
    ArchiveRepresentationHandler,
    DeactivateRepresentationHandler,
    RenameRepresentationHandler,
    UpdateRepresentationNoteHandler,
)
from kraken_manager.application.use_cases import (
    AddArtifactVersionHandler,
    CreateArtifactSeriesHandler,
    CreateLayerHandler,
    CreateProjectHandler,
    CreateRepresentationHandler,
)
from kraken_manager.domain.artifacts import ArtifactScope
from kraken_manager.domain.common import (
    ArtifactSeriesId,
    ArtifactVersionId,
    FrameId,
    LayerId,
    PerformerId,
    PrincipalId,
    ProjectId,
    RepresentationId,
    ReviewBatchId,
    PluginJobId,
    validate_uuid,
)
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
from kraken_manager.domain.identity import Permission, ProjectRole
from kraken_manager.domain.project import (
    GridOrientation,
    Layer,
    LayerType,
    Project,
    Representation,
    RepresentationKind,
    RepresentationPurpose,
)
from kraken_manager.domain.artifacts import deterministic_frame_series_id
from kraken_manager.domain.selection import FrameRowRange, FrameSelectionV1
from kraken_manager.domain.workflows import (
    PluginInputV1,
    PluginResultManifestV1,
    PluginResultOutcome,
    ReviewItem,
)
from kraken_core.plugin_protocol import PluginResultManifest, safe_relative_path
from kraken_manager.infrastructure.plugin.result_reader import domain_result_from_transport
from kraken_manager.infrastructure.postgres import PostgresEventStore, PostgresIdentityAclStore, PostgresProjectionStore
from kraken_manager.infrastructure.reports import (
    ActivityRecord,
    ReportFilters,
    ReportGranularity,
    ReportService,
)
from kraken_manager.infrastructure.review import ReviewPackageReader, ReviewPackageWriter
from kraken_manager.infrastructure.filesystem._codec import encode_model

from .services import CommandContext, ConflictError, ForbiddenError, NotFoundError, ValidationError


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class _AgentBlobResultReader:
    def __init__(self, blobs: Any, references: Mapping[str, Any]) -> None:
        self.blobs = blobs
        self.references = dict(references)

    def iter_output(self, manifest: Any, relative_path: str) -> Iterator[bytes]:
        del manifest
        normalized = safe_relative_path(relative_path)
        reference = self.references.get(normalized)
        if reference is None:
            raise FileNotFoundError(f"Agent output was not uploaded: {normalized}")
        return self.blobs.iter_bytes(reference)


class ServerStorageProfiles:
    """Server-owned profile catalog; persistence/configuration may replace it."""

    def __init__(self, *, profile_id: str = "server-postgres", max_frames: int | None = None) -> None:
        self.profile = StorageProfile(
            id=profile_id,
            name="Kraken Server PostgreSQL",
            metadata_backend=StorageBackendKind.POSTGRESQL,
            blob_backend="filesystem",
            scope=StorageScope.SHARED,
            capabilities=StorageCapabilities(
                multi_writer=True,
                transactions=True,
                snapshots=True,
                streaming=True,
                external_references=True,
                max_frames=max_frames,
            ),
        )

    def get(self, profile_id: str) -> StorageProfile | None:
        return self.profile if profile_id == self.profile.id else None

    def list(self) -> tuple[StorageProfile, ...]:
        return (self.profile,)


def _project_dict(project: Project) -> dict[str, Any]:
    return {
        "project_id": str(project.id),
        "name": project.name,
        "width": project.width,
        "height": project.height,
        "orientation": project.orientation.value,
        "storage_profile": project.storage_profile,
        "state": project.state.value,
        "revision": project.revision,
        "created_at": project.created_at.isoformat(),
    }


def _layer_dict(layer: Layer) -> dict[str, Any]:
    return {
        "layer_id": str(layer.id),
        "project_id": str(layer.project_id),
        "name": layer.name,
        "type": layer.type.value,
        "order": layer.order,
        "state": layer.state.value,
        "revision": layer.revision,
        "created_at": layer.created_at.isoformat(),
    }


def _representation_dict(representation: Representation) -> dict[str, Any]:
    return {
        "representation_id": str(representation.id),
        "project_id": str(representation.project_id),
        "layer_id": str(representation.layer_id),
        "name": representation.name,
        "kind": representation.kind.value,
        "purpose": representation.purpose.value,
        "source_image_representation_id": (
            None
            if representation.source_image_representation_id is None
            else str(representation.source_image_representation_id)
        ),
        "note": representation.note,
        "source": representation.source,
        "active": representation.active,
        "state": representation.state.value,
        "revision": representation.revision,
        "created_at": representation.created_at.isoformat(),
    }


def _project_id(value: str) -> ProjectId:
    return ProjectId(validate_uuid(value, field="project_id"))


def _layer_id(value: str) -> LayerId:
    return LayerId(validate_uuid(value, field="layer_id"))


class PostgresServerServices:
    """Thin transport facade; mutations execute application command handlers."""

    def __init__(
        self,
        engine: Any,
        uow_factory: Any,
        *,
        profiles: ServerStorageProfiles | None = None,
        agent_gateway: Any | None = None,
        performer_store: Any | None = None,
        review_key_pair: Any | None = None,
    ) -> None:
        self.engine = engine
        self.uow_factory = uow_factory
        self.profiles = profiles or ServerStorageProfiles()
        self.agent_gateway = agent_gateway
        self.performer_store = performer_store
        self.review_key_pair = review_key_pair
        self.projections = PostgresProjectionStore(engine)
        self.events = PostgresEventStore(engine)
        self.identities = PostgresIdentityAclStore(engine)
        self._create_project = CreateProjectHandler(uow_factory, self.profiles, SystemClock())
        self._create_layer = CreateLayerHandler(uow_factory, self.profiles, SystemClock())
        self._create_representation = CreateRepresentationHandler(uow_factory, self.profiles, SystemClock())
        self._rename_project = RenameProjectHandler(uow_factory, self.profiles, SystemClock())
        self._archive_project = ArchiveProjectHandler(uow_factory, self.profiles, SystemClock())
        self._restore_project = RestoreProjectHandler(uow_factory, self.profiles, SystemClock())
        self._rename_layer = RenameLayerHandler(uow_factory, self.profiles, SystemClock())
        self._reorder_layer = ReorderLayerHandler(uow_factory, self.profiles, SystemClock())
        self._reorder_layers = ReorderLayersHandler(uow_factory, self.profiles, SystemClock())
        self._archive_layer = ArchiveLayerHandler(uow_factory, self.profiles, SystemClock())
        self._assign_project_role = AssignProjectRoleHandler(uow_factory, self.profiles, SystemClock())
        self._revoke_project_role = RevokeProjectRoleHandler(uow_factory, self.profiles, SystemClock())
        self._activate_representation = ActivateRepresentationHandler(uow_factory, self.profiles, SystemClock())
        self._deactivate_representation = DeactivateRepresentationHandler(
            uow_factory,
            self.profiles,
            SystemClock(),
        )
        self._archive_representation = ArchiveRepresentationHandler(uow_factory, self.profiles, SystemClock())
        self._rename_representation = RenameRepresentationHandler(uow_factory, self.profiles, SystemClock())
        self._update_representation_note = UpdateRepresentationNoteHandler(uow_factory, self.profiles, SystemClock())
        self._create_artifact_series = CreateArtifactSeriesHandler(uow_factory, self.profiles, SystemClock())
        self._add_artifact_version = AddArtifactVersionHandler(uow_factory, self.profiles, SystemClock())
        self._rename_artifact_series = RenameArtifactSeriesHandler(uow_factory, self.profiles, SystemClock())
        self._archive_artifact_series = ArchiveArtifactSeriesHandler(uow_factory, self.profiles, SystemClock())
        self._activate_artifact_version = ActivateArtifactVersionHandler(uow_factory, self.profiles, SystemClock())
        self._add_external_artifact_version = AddExternalArtifactVersionHandler(uow_factory, self.profiles, SystemClock())
        self._create_note = CreateNoteHandler(uow_factory, self.profiles, SystemClock())
        self._revise_note = ReviseNoteHandler(uow_factory, self.profiles, SystemClock())
        self._create_review_batch = (
            None
            if performer_store is None
            else CreateReviewBatchHandler(
                uow_factory, self.profiles, SystemClock(), performer_store
            )
        )
        self._accept_review = AcceptReviewHandler(uow_factory, self.profiles, SystemClock())
        self._request_review_changes = RequestReviewChangesHandler(
            uow_factory, self.profiles, SystemClock()
        )
        self._cancel_review_batch = CancelReviewBatchHandler(
            uow_factory, self.profiles, SystemClock()
        )
        self._export_review_package = (
            None
            if review_key_pair is None
            else ExportReviewPackageHandler(
                uow_factory,
                self.profiles,
                SystemClock(),
                ReviewPackageWriter(review_key_pair.private_key),
            )
        )
        self._dry_run_review_return = (
            None
            if review_key_pair is None
            else DryRunReviewReturnHandler(
                uow_factory,
                self.profiles,
                SystemClock(),
                ReviewPackageReader(review_key_pair.public_key),
            )
        )
        self._commit_review_return = (
            None
            if review_key_pair is None
            else CommitReviewReturnHandler(
                uow_factory,
                self.profiles,
                SystemClock(),
                ReviewPackageReader(review_key_pair.public_key),
            )
        )
        self._submit_plugin_job = (
            None
            if agent_gateway is None
            else SubmitPluginJobHandler(
                uow_factory, self.profiles, SystemClock(), agent_gateway
            )
        )
        self._cancel_plugin_job = (
            None
            if agent_gateway is None
            else CancelPluginJobHandler(
                uow_factory, self.profiles, SystemClock(), agent_gateway
            )
        )

    def health(self) -> dict[str, Any]:
        try:
            from sqlalchemy import text

            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1")).scalar_one()
        except Exception as exc:
            return {"status": "degraded", "metadata": "unavailable", "detail": str(exc)[:500]}
        return {"status": "ok", "metadata": "postgresql", "api_version": "v1"}

    def list_projects(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        return [
            _project_dict(project)
            for project in self.projections.list_projects(
                include_archived=include_archived
            )
        ]

    def list_principals(self, *, include_inactive: bool = False) -> list[dict[str, Any]]:
        return [
            {
                "principal_id": str(principal.id),
                "provider": principal.provider.value,
                "subject": principal.subject,
                "issuer": principal.issuer,
                "display_name": principal.display_name,
                "email": principal.email,
                "active": principal.active,
                "system_roles": sorted(
                    role.value for role in principal.system_roles
                ),
            }
            for principal in self.identities.list(
                include_inactive=include_inactive
            )
        ]

    def list_performers(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        if self.performer_store is None:
            raise ConflictError("Server performer store is not configured")
        return [
            encode_model(item)
            for item in self.performer_store.list(include_archived=include_archived)
        ]

    def _actor(self, actor_id: str) -> Any:
        try:
            principal_id = PrincipalId(validate_uuid(actor_id, field="actor_id"))
            actor = self.identities.get(principal_id)
        except (TypeError, ValueError) as exc:
            raise ValidationError("Authenticated principal id is invalid") from exc
        if actor is None or not actor.active:
            raise ValidationError("Authenticated principal is unknown or inactive")
        return actor

    def create_project(self, payload: Mapping[str, Any], context: CommandContext) -> dict[str, Any]:
        actor = self._actor(context.actor_id)
        if context.expected_revision not in {None, 0}:
            raise ConflictError("A new project has expected revision 0")
        stable_id = payload.get("project_id") or str(
            uuid5(NAMESPACE_URL, f"kraken:project:{actor.id}:{context.idempotency_key}")
        )
        try:
            command = CreateProjectCommand(
                context=ApplicationCommandContext(
                    actor=actor,
                    idempotency_key=context.idempotency_key,
                    gitlab_identity_verified=True,
                ),
                name=str(payload.get("name", "")),
                width=int(payload.get("width", 0)),
                height=int(payload.get("height", 0)),
                orientation=GridOrientation(str(payload.get("orientation", GridOrientation.Y_DOWN.value))),
                storage_profile_id=str(payload.get("storage_profile_id", self.profiles.profile.id)),
                project_id=ProjectId(str(stable_id)),
            )
            return _project_dict(self._create_project(command))
        except ApplicationNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc
        except (ApplicationConcurrencyError, ApplicationConflictError) as exc:
            raise ConflictError(str(exc)) from exc
        except ApplicationAuthorizationError as exc:
            raise ForbiddenError(str(exc)) from exc
        except (StorageCapabilityError, ValueError, TypeError) as exc:
            raise ValidationError(str(exc)) from exc

    def get_project(self, project_id: str) -> dict[str, Any]:
        try:
            project = self.projections.get_project(_project_id(project_id))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Invalid project id") from exc
        if project is None:
            raise NotFoundError(project_id)
        return _project_dict(project)

    def _application_context(self, context: CommandContext) -> ApplicationCommandContext:
        return ApplicationCommandContext(
            actor=self._actor(context.actor_id),
            idempotency_key=context.idempotency_key,
            gitlab_identity_verified=True,
        )

    @staticmethod
    def _translate_lifecycle_error(exc: Exception) -> None:
        if isinstance(exc, ApplicationNotFoundError):
            raise NotFoundError(str(exc)) from exc
        if isinstance(exc, (ApplicationConcurrencyError, ApplicationConflictError)):
            raise ConflictError(str(exc)) from exc
        if isinstance(exc, ApplicationAuthorizationError):
            raise ForbiddenError(str(exc)) from exc
        if isinstance(exc, (StorageCapabilityError, ValueError, TypeError)):
            raise ValidationError(str(exc)) from exc
        raise exc

    def rename_project(self, project_id: str, name: str, context: CommandContext) -> dict[str, Any]:
        try:
            return _project_dict(
                self._rename_project(
                    RenameProjectCommand(
                        context=self._application_context(context),
                        project_id=_project_id(project_id),
                        name=name,
                        expected_revision=self._require_revision(context),
                    )
                )
            )
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def archive_project(self, project_id: str, context: CommandContext) -> dict[str, Any]:
        try:
            return _project_dict(
                self._archive_project(
                    ArchiveProjectCommand(
                        context=self._application_context(context),
                        project_id=_project_id(project_id),
                        expected_revision=self._require_revision(context),
                    )
                )
            )
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def restore_project(self, project_id: str, context: CommandContext) -> dict[str, Any]:
        try:
            return _project_dict(
                self._restore_project(
                    RestoreProjectCommand(
                        context=self._application_context(context),
                        project_id=_project_id(project_id),
                        expected_revision=self._require_revision(context),
                    )
                )
            )
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    @staticmethod
    def _require_revision(context: CommandContext) -> int:
        if context.expected_revision is None:
            raise ValidationError("If-Match is required for this command")
        return context.expected_revision

    def list_layers(
        self, project_id: str, *, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        self.get_project(project_id)
        return [
            _layer_dict(layer)
            for layer in self.projections.list_layers(
                _project_id(project_id), include_archived=include_archived
            )
        ]

    def create_layer(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> dict[str, Any]:
        actor = self._actor(context.actor_id)
        try:
            layer = self._create_layer(
                CreateLayerCommand(
                    context=ApplicationCommandContext(
                        actor=actor,
                        idempotency_key=context.idempotency_key,
                        gitlab_identity_verified=True,
                    ),
                    project_id=_project_id(project_id),
                    name=str(payload.get("name", "")),
                    type=LayerType(str(payload.get("type", ""))),
                    order=int(payload.get("order", 0)),
                    expected_project_revision=self._require_revision(context),
                )
            )
            return _layer_dict(layer)
        except ApplicationNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc
        except (ApplicationConcurrencyError, ApplicationConflictError) as exc:
            raise ConflictError(str(exc)) from exc
        except ApplicationAuthorizationError as exc:
            raise ForbiddenError(str(exc)) from exc
        except (StorageCapabilityError, ValueError, TypeError) as exc:
            raise ValidationError(str(exc)) from exc

    def rename_layer(
        self, project_id: str, layer_id: str, name: str, context: CommandContext
    ) -> dict[str, Any]:
        try:
            return _layer_dict(
                self._rename_layer(
                    RenameLayerCommand(
                        context=self._application_context(context),
                        project_id=_project_id(project_id),
                        layer_id=_layer_id(layer_id),
                        name=name,
                        expected_revision=self._require_revision(context),
                    )
                )
            )
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def reorder_layer(
        self, project_id: str, layer_id: str, order: int, context: CommandContext
    ) -> dict[str, Any]:
        try:
            return _layer_dict(
                self._reorder_layer(
                    ReorderLayerCommand(
                        context=self._application_context(context),
                        project_id=_project_id(project_id),
                        layer_id=_layer_id(layer_id),
                        order=order,
                        expected_revision=self._require_revision(context),
                    )
                )
            )
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def reorder_layers(
        self, project_id: str, payload: Mapping[str, Any], context: CommandContext
    ) -> list[dict[str, Any]]:
        raw_revisions = payload.get("expected_revisions", {})
        if not isinstance(raw_revisions, Mapping):
            raise ValidationError("expected_revisions must be an object")
        try:
            layers = self._reorder_layers(
                ReorderLayersCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    layer_ids=tuple(_layer_id(str(value)) for value in payload.get("layer_ids", ())),
                    expected_revisions=tuple(
                        (_layer_id(str(identifier)), int(revision))
                        for identifier, revision in raw_revisions.items()
                    ),
                )
            )
            return [_layer_dict(layer) for layer in layers]
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def archive_layer(
        self, project_id: str, layer_id: str, context: CommandContext
    ) -> dict[str, Any]:
        try:
            return _layer_dict(
                self._archive_layer(
                    ArchiveLayerCommand(
                        context=self._application_context(context),
                        project_id=_project_id(project_id),
                        layer_id=_layer_id(layer_id),
                        expected_revision=self._require_revision(context),
                    )
                )
            )
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def project_roles(self, project_id: str, principal_id: str) -> dict[str, Any]:
        project = self.get_project(project_id)
        del project
        try:
            project_identifier = _project_id(project_id)
            principal_identifier = PrincipalId(validate_uuid(principal_id, field="principal_id"))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Invalid project or principal id") from exc
        if self.identities.get(principal_identifier) is None:
            raise NotFoundError(principal_id)
        stream_id = f"acl:{project_identifier}:{principal_identifier}"
        return {
            "project_id": project_id,
            "principal_id": principal_id,
            "roles": sorted(
                role.value for role in self.identities.roles_for(project_identifier, principal_identifier)
            ),
            "revision": self.events.current_revision(stream_id),
        }

    def assign_project_role(
        self,
        project_id: str,
        principal_id: str,
        role: str,
        context: CommandContext,
    ) -> dict[str, Any]:
        try:
            self._assign_project_role(
                AssignProjectRoleCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    principal_id=PrincipalId(validate_uuid(principal_id, field="principal_id")),
                    role=ProjectRole(role),
                    expected_revision=self._require_revision(context),
                )
            )
            return self.project_roles(project_id, principal_id)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def revoke_project_role(
        self,
        project_id: str,
        principal_id: str,
        role: str,
        context: CommandContext,
    ) -> dict[str, Any]:
        try:
            self._revoke_project_role(
                RevokeProjectRoleCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    principal_id=PrincipalId(validate_uuid(principal_id, field="principal_id")),
                    role=ProjectRole(role),
                    expected_revision=self._require_revision(context),
                )
            )
            return self.project_roles(project_id, principal_id)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def list_representations(
        self, project_id: str, layer_id: str, *, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        self.get_project(project_id)
        try:
            layer = self.projections.get_layer(_layer_id(layer_id))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Invalid layer id") from exc
        if layer is None or str(layer.project_id) != project_id:
            raise NotFoundError(layer_id)
        return [
            _representation_dict(representation)
            for representation in self.projections.list_representations(
                layer.id, include_archived=include_archived
            )
        ]

    def create_representation(
        self,
        project_id: str,
        layer_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        actor = self._actor(context.actor_id)
        try:
            representation = self._create_representation(
                CreateRepresentationCommand(
                    context=ApplicationCommandContext(
                        actor=actor,
                        idempotency_key=context.idempotency_key,
                        gitlab_identity_verified=True,
                    ),
                    project_id=_project_id(project_id),
                    layer_id=_layer_id(layer_id),
                    name=str(payload.get("name", "")),
                    kind=RepresentationKind(str(payload.get("kind", ""))),
                    purpose=RepresentationPurpose(
                        str(
                            payload.get(
                                "purpose",
                                "vector" if str(payload.get("kind", "")) == "vector" else "source",
                            )
                        )
                    ),
                    expected_layer_revision=self._require_revision(context),
                    note=str(payload.get("note", "")),
                    source=None if payload.get("source") is None else str(payload["source"]),
                    source_image_representation_id=(
                        None
                        if payload.get("source_image_representation_id") is None
                        else RepresentationId(str(payload["source_image_representation_id"]))
                    ),
                    active=bool(payload.get("active", False)),
                )
            )
            return _representation_dict(representation)
        except ApplicationNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc
        except (ApplicationConcurrencyError, ApplicationConflictError) as exc:
            raise ConflictError(str(exc)) from exc
        except ApplicationAuthorizationError as exc:
            raise ForbiddenError(str(exc)) from exc
        except (StorageCapabilityError, ValueError, TypeError) as exc:
            raise ValidationError(str(exc)) from exc

    def update_representation(
        self,
        project_id: str,
        layer_id: str,
        representation_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        operations = [key for key in ("name", "note", "active", "archive") if key in payload]
        if len(operations) != 1:
            raise ValidationError("Exactly one representation operation is required")
        try:
            common = {
                "context": self._application_context(context),
                "project_id": _project_id(project_id),
                "layer_id": _layer_id(layer_id),
                "representation_id": RepresentationId(
                    validate_uuid(representation_id, field="representation_id")
                ),
                "expected_layer_revision": self._require_revision(context),
                "expected_representation_revision": int(
                    payload.get("expected_representation_revision", -1)
                ),
            }
            operation = operations[0]
            if operation == "name":
                value = self._rename_representation(
                    RenameRepresentationCommand(name=str(payload["name"]), **common)
                )
            elif operation == "note":
                value = self._update_representation_note(
                    UpdateRepresentationNoteCommand(note=str(payload["note"]), **common)
                )
            elif operation == "active":
                value = (
                    self._activate_representation(
                        ActivateRepresentationCommand(**common)
                    )
                    if bool(payload["active"])
                    else self._deactivate_representation(
                        DeactivateRepresentationCommand(**common)
                    )
                )
            else:
                if not bool(payload["archive"]):
                    raise ValueError("archive must be true")
                value = self._archive_representation(ArchiveRepresentationCommand(**common))
            return _representation_dict(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def matrix_viewport(
        self,
        project_id: str,
        *,
        layer_id: str,
        representation_ids: Iterable[str] = (),
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        lod: int,
        include_missing: bool = True,
    ) -> dict[str, Any]:
        project = self.get_project(project_id)
        if not (1 <= x1 <= x2 <= project["width"] and 1 <= y1 <= y2 <= project["height"]):
            raise ValidationError("Viewport is outside the project grid")
        if lod < 0 or lod > 24:
            raise ValidationError("LOD must be between 0 and 24")
        project_model = self.projections.get_project(_project_id(project_id))
        assert project_model is not None
        layer_identifier = _layer_id(layer_id)
        identifiers = tuple(
            RepresentationId(validate_uuid(str(value), field="representation_id"))
            for value in representation_ids
        )
        if not identifiers:
            identifiers = tuple(
                representation.id
                for representation in self.projections.list_representations(
                    project_model.id, layer_identifier
                )
            )
        representations = {
            representation.id: representation
            for identifier in identifiers
            if (representation := self.projections.get_representation(identifier))
            is not None
            and representation.layer_id == layer_identifier
        }
        priority = {
            "image_ready": 1,
            "vectorized": 3,
            "in_review": 4,
            "returned_changed": 6,
            "approved": 7,
            "changes_requested": 8,
            "error": 10,
        }
        merged: dict[tuple[int, int], dict[str, Any]] = {}
        coverage: dict[str, set[tuple[int, int]]] = {}
        newest = ""
        for representation in representations.values():
            current_coverage = coverage.setdefault(str(representation.id), set())
            for series in self.projections.list_artifact_series(
                project_model.id,
                layer_id=layer_identifier,
                representation_id=representation.id,
            ):
                version = self.projections.get_active_artifact_version(series.id)
                cursor = version
                coordinate: tuple[int, int] | None = None
                visited: set[str] = set()
                while cursor is not None and str(cursor.id) not in visited:
                    visited.add(str(cursor.id))
                    raw_x = cursor.parameters.get("x")
                    raw_y = cursor.parameters.get("y")
                    if (
                        isinstance(raw_x, int)
                        and not isinstance(raw_x, bool)
                        and isinstance(raw_y, int)
                        and not isinstance(raw_y, bool)
                    ):
                        coordinate = (raw_x, raw_y)
                        break
                    cursor = (
                        None
                        if cursor.parent_version_id is None
                        else self.projections.get_artifact_version(
                            cursor.parent_version_id
                        )
                    )
                if version is None or coordinate is None or series.frame_id is None:
                    continue
                x, y = coordinate
                if not (x1 <= x <= x2 and y1 <= y <= y2):
                    continue
                current_coverage.add(coordinate)
                newest = max(newest, version.created_at.isoformat())
                status = (
                    "image_ready"
                    if representation.kind is RepresentationKind.IMAGE
                    else "vectorized"
                )
                cell = {
                    "artifact_version_id": str(version.id),
                    "frame_id": str(series.frame_id),
                    "sha256": version.sha256,
                    "status": status,
                    "x": x,
                    "y": y,
                    "modified_at": version.created_at.isoformat(),
                    "review_status": "not_checked",
                }
                if representation.kind is RepresentationKind.IMAGE and version.blob is not None:
                    cell.update(
                        {
                            "asset_sha256": version.blob.sha256,
                            "asset_source_key": str(version.id),
                            "asset_revision": str(version.id),
                            "asset_media_type": version.media_type,
                        }
                    )
                previous = merged.get(coordinate)
                if previous is None or priority[status] >= priority.get(
                    str(previous.get("status")), 0
                ):
                    if previous is not None:
                        for field in (
                            "asset_sha256",
                            "asset_source_key",
                            "asset_revision",
                            "asset_media_type",
                        ):
                            if field in previous and field not in cell:
                                cell[field] = previous[field]
                    merged[coordinate] = cell
                elif "asset_source_key" in cell:
                    previous.update(
                        {
                            key: value
                            for key, value in cell.items()
                            if key.startswith("asset_")
                        }
                    )
        review_status = {
            "issued": "in_review",
            "partially_returned": "in_review",
            "awaiting_acceptance": "returned_changed",
            "changes_requested": "changes_requested",
        }
        cells_by_frame = {
            str(cell["frame_id"]): cell for cell in merged.values()
        }
        for batch in self.projections.list_active_review_batches(
            project_model.id, layer_identifier
        ):
            status = review_status.get(batch.state.value)
            if status is None:
                continue
            for item in batch.items:
                cell = cells_by_frame.get(str(item.frame_id))
                if cell is not None:
                    cell["status"] = status
                    cell["review_status"] = "in_review"
        managed = {
            str(identifier)
            for identifier, representation in representations.items()
            if representation.source == "managed-import"
        }
        if lod == 0 and include_missing and managed:
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    missing = tuple(
                        identifier
                        for identifier in managed
                        if (x, y) not in coverage.get(identifier, set())
                    )
                    if missing:
                        cell = merged.setdefault(
                            (x, y),
                            {
                                "artifact_version_id": "",
                                "frame_id": str(project_model.frame_id_at(x, y)),
                                "sha256": "",
                                "x": x,
                                "y": y,
                            },
                        )
                        cell.update(
                            {
                                "status": "error",
                                "missing": True,
                                "missing_representation_ids": missing,
                            }
                        )
        cells = tuple(
            sorted(merged.values(), key=lambda value: (value["y"], value["x"]))
        )
        if lod > 0:
            span = 1 << lod
            buckets: dict[tuple[int, int], dict[str, Any]] = {}
            for cell in cells:
                bucket_x = (int(cell["x"]) - 1) // span
                bucket_y = (int(cell["y"]) - 1) // span
                bucket = buckets.setdefault(
                    (bucket_x, bucket_y),
                    {
                        "bounds": {
                            "x1": bucket_x * span + 1,
                            "y1": bucket_y * span + 1,
                            "x2": min(project_model.width, (bucket_x + 1) * span),
                            "y2": min(project_model.height, (bucket_y + 1) * span),
                        },
                        "materialized_count": 0,
                        "status_counts": {},
                    },
                )
                bucket["materialized_count"] += 1
                counts = bucket["status_counts"]
                status = str(cell.get("status", "empty"))
                counts[status] = counts.get(status, 0) + 1
        return {
            "project_id": project_id,
            "layer_id": layer_id,
            "bounds": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "lod": lod,
            "revision": newest or str(project["revision"]),
            "cells": cells if lod == 0 else (),
            "aggregates": (
                tuple(buckets[key] for key in sorted(buckets)) if lod > 0 else ()
            ),
        }

    def list_artifact_series(
        self,
        project_id: str,
        *,
        layer_id: str | None = None,
        representation_id: str | None = None,
        frame_id: str | None = None,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        self.get_project(project_id)
        return [
            encode_model(item)
            for item in self.projections.list_artifact_series(
                _project_id(project_id),
                layer_id=None if layer_id is None else LayerId(validate_uuid(layer_id, field="layer_id")),
                representation_id=(
                    None
                    if representation_id is None
                    else RepresentationId(validate_uuid(representation_id, field="representation_id"))
                ),
                frame_id=None if frame_id is None else FrameId(validate_uuid(frame_id, field="frame_id")),
                include_archived=include_archived,
            )
        ]

    def list_artifact_versions(self, project_id: str, series_id: str) -> list[dict[str, Any]]:
        self.get_project(project_id)
        identifier = ArtifactSeriesId(validate_uuid(series_id, field="series_id"))
        series = self.projections.get_artifact_series(identifier)
        if series is None or str(series.project_id) != project_id:
            raise NotFoundError("Artifact series was not found")
        return [encode_model(item) for item in self.projections.list_artifact_versions(identifier)]

    def artifact_stream_revision(self, project_id: str, series_id: str) -> int:
        self.get_project(project_id)
        identifier = ArtifactSeriesId(validate_uuid(series_id, field="series_id"))
        series = self.projections.get_artifact_series(identifier)
        if series is None or str(series.project_id) != project_id:
            raise NotFoundError("Artifact series was not found")
        return self.events.current_revision(f"artifact-series:{identifier}")

    def get_active_artifact_version(self, project_id: str, series_id: str) -> dict[str, Any] | None:
        self.get_project(project_id)
        identifier = ArtifactSeriesId(validate_uuid(series_id, field="series_id"))
        series = self.projections.get_artifact_series(identifier)
        if series is None or str(series.project_id) != project_id:
            raise NotFoundError("Artifact series was not found")
        value = self.projections.get_active_artifact_version(identifier)
        return None if value is None else encode_model(value)

    def get_artifact_version(self, project_id: str, version_id: str) -> dict[str, Any]:
        self.get_project(project_id)
        version = self.projections.get_artifact_version(
            ArtifactVersionId(validate_uuid(version_id, field="version_id"))
        )
        if version is None:
            raise NotFoundError("Artifact version was not found")
        series = self.projections.get_artifact_series(version.series_id)
        if series is None or str(series.project_id) != project_id:
            raise NotFoundError("Artifact version was not found in this project")
        return encode_model(version)

    def create_artifact_series(
        self,
        project_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        try:
            value = self._create_artifact_series(
                CreateArtifactSeriesCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    scope=ArtifactScope(str(payload.get("scope", "project_attachment"))),
                    name=str(payload.get("name", "")),
                    layer_id=(
                        None
                        if payload.get("layer_id") is None
                        else LayerId(validate_uuid(str(payload["layer_id"]), field="layer_id"))
                    ),
                    representation_id=(
                        None
                        if payload.get("representation_id") is None
                        else RepresentationId(
                            validate_uuid(str(payload["representation_id"]), field="representation_id")
                        )
                    ),
                    frame_id=(
                        None
                        if payload.get("frame_id") is None
                        else FrameId(validate_uuid(str(payload["frame_id"]), field="frame_id"))
                    ),
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def add_managed_artifact_version(
        self,
        project_id: str,
        series_id: str,
        payload: Mapping[str, Any],
        chunks: Iterable[bytes],
        context: CommandContext,
    ) -> dict[str, Any]:
        try:
            value = self._add_artifact_version(
                AddArtifactVersionCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    series_id=ArtifactSeriesId(validate_uuid(series_id, field="series_id")),
                    filename=str(payload.get("filename", "")),
                    media_type=str(payload.get("media_type", "application/octet-stream")),
                    expected_series_revision=self._require_revision(context),
                    parent_version_id=(
                        None
                        if payload.get("parent_version_id") is None
                        else ArtifactVersionId(
                            validate_uuid(str(payload["parent_version_id"]), field="parent_version_id")
                        )
                    ),
                    expected_sha256=(
                        None if payload.get("sha256") is None else str(payload["sha256"])
                    ),
                ),
                chunks,
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def add_external_artifact_version(
        self,
        project_id: str,
        series_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        try:
            value = self._add_external_artifact_version(
                AddExternalArtifactVersionCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    series_id=ArtifactSeriesId(validate_uuid(series_id, field="series_id")),
                    filename=str(payload.get("filename", "")),
                    media_type=str(payload.get("media_type", "application/octet-stream")),
                    uri=str(payload.get("uri", "")),
                    fingerprint_sha256=str(payload.get("sha256", "")),
                    observed_size_bytes=int(payload.get("size_bytes", -1)),
                    expected_series_revision=self._require_revision(context),
                    parent_version_id=(
                        None
                        if payload.get("parent_version_id") is None
                        else ArtifactVersionId(
                            validate_uuid(
                                str(payload["parent_version_id"]),
                                field="parent_version_id",
                            )
                        )
                    ),
                    parameters=(
                        dict(payload.get("parameters", {}))
                        if isinstance(payload.get("parameters", {}), Mapping)
                        else {}
                    ),
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def mutate_artifact_series(
        self,
        project_id: str,
        series_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        common = {
            "context": self._application_context(context),
            "project_id": _project_id(project_id),
            "series_id": ArtifactSeriesId(validate_uuid(series_id, field="series_id")),
        }
        try:
            if "name" in payload:
                value = self._rename_artifact_series(
                    RenameArtifactSeriesCommand(
                        name=str(payload["name"]),
                        expected_series_revision=self._require_revision(context),
                        **common,
                    )
                )
            elif bool(payload.get("archive")):
                value = self._archive_artifact_series(
                    ArchiveArtifactSeriesCommand(
                        expected_series_revision=self._require_revision(context),
                        **common,
                    )
                )
            elif payload.get("active_version_id") is not None:
                value = self._activate_artifact_version(
                    ActivateArtifactVersionCommand(
                        version_id=ArtifactVersionId(
                            validate_uuid(str(payload["active_version_id"]), field="version_id")
                        ),
                        **common,
                    )
                )
            else:
                raise ValidationError("Exactly one artifact-series operation is required")
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def list_notes(
        self,
        project_id: str,
        *,
        layer_id: str | None = None,
        frame_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.get_project(project_id)
        return [
            encode_model(item)
            for item in self.projections.list_notes(
                _project_id(project_id),
                layer_id=layer_id,
                frame_id=frame_id,
            )
        ]

    def create_note(
        self,
        project_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        try:
            value = self._create_note(
                CreateNoteCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    body=str(payload.get("body", "")),
                    layer_id=(
                        None
                        if payload.get("layer_id") is None
                        else LayerId(validate_uuid(str(payload["layer_id"]), field="layer_id"))
                    ),
                    frame_id=(
                        None
                        if payload.get("frame_id") is None
                        else FrameId(validate_uuid(str(payload["frame_id"]), field="frame_id"))
                    ),
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def revise_note(
        self,
        project_id: str,
        note_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        try:
            value = self._revise_note(
                ReviseNoteCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    note_id=validate_uuid(note_id, field="note_id"),
                    body=str(payload.get("body", "")),
                    expected_revision=self._require_revision(context),
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def iter_artifact_bytes(
        self,
        project_id: str,
        version_id: str,
        *,
        chunk_size: int = 1024 * 1024,
    ) -> Iterator[bytes]:
        version_payload = self.get_artifact_version(project_id, version_id)
        blob = version_payload.get("blob")
        if not isinstance(blob, Mapping):
            raise ValidationError("External artifact versions do not have managed bytes")
        reference = self.projections.get_artifact_version(
            ArtifactVersionId(validate_uuid(version_id, field="version_id"))
        )
        assert reference is not None and reference.blob is not None
        return self.uow_factory.blobs.iter_bytes(reference.blob, chunk_size=chunk_size)

    def list_review_batches(
        self, project_id: str, *, active_only: bool = False
    ) -> list[dict[str, Any]]:
        project = self.projections.get_project(_project_id(project_id))
        if project is None:
            raise NotFoundError("Project was not found")
        batches = self.projections.list_review_batches(project.id)
        if active_only:
            active_ids = {
                str(batch.id)
                for layer in self.projections.list_layers(project.id)
                for batch in self.projections.list_active_review_batches(project.id, layer.id)
            }
            batches = tuple(batch for batch in batches if str(batch.id) in active_ids)
        return [encode_model(batch) for batch in batches]

    def create_review_batch(
        self,
        project_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        if self._create_review_batch is None:
            raise ConflictError("Server performer store is not configured")
        try:
            project = self.projections.get_project(_project_id(project_id))
            if project is None:
                raise ApplicationNotFoundError("Project was not found")
            layer_id = _layer_id(str(payload.get("layer_id", "")))
            layer = self.projections.get_layer(layer_id)
            if layer is None or layer.project_id != project.id:
                raise ApplicationNotFoundError("Layer was not found in the project")
            image_id = RepresentationId(
                validate_uuid(
                    str(payload.get("image_representation_id", "")),
                    field="image_representation_id",
                )
            )
            vector_id = RepresentationId(
                validate_uuid(
                    str(payload.get("vector_representation_id", "")),
                    field="vector_representation_id",
                )
            )
            image = self.projections.get_representation(image_id)
            vector = self.projections.get_representation(vector_id)
            if image is None or image.layer_id != layer.id or image.kind is not RepresentationKind.IMAGE:
                raise ApplicationConflictError("Select an image representation from the layer")
            if (
                vector is None
                or vector.layer_id != layer.id
                or vector.kind is not RepresentationKind.VECTOR
                or vector.source_image_representation_id != image.id
            ):
                raise ApplicationConflictError(
                    "Select a vector representation linked to the selected images"
                )
            raw_coordinates = payload.get("coordinates", ())
            if not isinstance(raw_coordinates, list):
                raise TypeError("coordinates must be an array")
            coordinates = sorted(
                {(int(item[0]), int(item[1])) for item in raw_coordinates},
                key=lambda item: (item[1], item[0]),
            )
            selection = FrameSelectionV1(
                row_ranges=tuple(
                    FrameRowRange(y=y, x_start=x, x_end=x) for x, y in coordinates
                )
            )
            items: list[ReviewItem] = []
            missing: list[str] = []
            for coordinate in selection.iter_coordinates():
                frame_id = coordinate.frame_id(project.id)
                image_version = self.projections.get_active_artifact_version(
                    deterministic_frame_series_id(image.id, frame_id)
                )
                vector_version = self.projections.get_active_artifact_version(
                    deterministic_frame_series_id(vector.id, frame_id)
                )
                if image_version is None or vector_version is None:
                    missing.append(f"({coordinate.x}, {coordinate.y})")
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
                raise ApplicationConflictError(
                    "Missing image or vector artifact versions for frames: "
                    + ", ".join(missing)
                )
            due_at = payload.get("due_at")
            value = self._create_review_batch(
                CreateReviewBatchCommand(
                    context=self._application_context(context),
                    project_id=project.id,
                    layer_id=layer.id,
                    selection=selection,
                    items=tuple(items),
                    assignee_id=PerformerId(
                        validate_uuid(str(payload.get("assignee_id", "")), field="assignee_id")
                    ),
                    expected_layer_revision=self._require_revision(context),
                    instructions=str(payload.get("instructions", "")),
                    due_at=(
                        None
                        if due_at in (None, "")
                        else datetime.fromisoformat(str(due_at))
                    ),
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def mutate_review_batch(
        self,
        project_id: str,
        batch_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        common = {
            "context": self._application_context(context),
            "project_id": _project_id(project_id),
            "batch_id": ReviewBatchId(validate_uuid(batch_id, field="batch_id")),
            "expected_batch_revision": self._require_revision(context),
        }
        try:
            action = str(payload.get("action", ""))
            if action == "accept":
                raw_ids = payload.get("candidate_version_ids", ())
                if not isinstance(raw_ids, list):
                    raise ValueError("candidate_version_ids must be an array")
                value = self._accept_review(
                    AcceptReviewCommand(
                        candidate_version_ids=tuple(
                            ArtifactVersionId(validate_uuid(str(item), field="version_id"))
                            for item in raw_ids
                        ),
                        **common,
                    )
                )
            elif action == "request_changes":
                value = self._request_review_changes(
                    RequestReviewChangesCommand(reason=str(payload.get("reason", "")), **common)
                )
            elif action == "cancel":
                value = self._cancel_review_batch(CancelReviewBatchCommand(**common))
            else:
                raise ValidationError("Unknown review operation")
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def export_review_package(
        self,
        project_id: str,
        batch_id: str,
        destination: str,
        context: CommandContext,
    ) -> dict[str, Any]:
        if self._export_review_package is None:
            raise ConflictError("Server review-package signing is not configured")
        try:
            value = self._export_review_package(
                ExportReviewPackageCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    batch_id=ReviewBatchId(
                        validate_uuid(batch_id, field="review_batch_id")
                    ),
                    expected_batch_revision=self._require_revision(context),
                    destination=destination,
                    include_images=True,
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def inspect_review_return(
        self,
        project_id: str,
        batch_id: str,
        source: str,
        context: CommandContext,
    ) -> dict[str, Any]:
        if self._dry_run_review_return is None:
            raise ConflictError("Server review-package verification is not configured")
        try:
            value = self._dry_run_review_return(
                DryRunReviewReturnCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    batch_id=ReviewBatchId(
                        validate_uuid(batch_id, field="review_batch_id")
                    ),
                    expected_batch_revision=self._require_revision(context),
                    source=source,
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def commit_review_return(
        self,
        project_id: str,
        batch_id: str,
        source: str,
        context: CommandContext,
    ) -> dict[str, Any]:
        if self._commit_review_return is None:
            raise ConflictError("Server review-package verification is not configured")
        try:
            value = self._commit_review_return(
                CommitReviewReturnCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    batch_id=ReviewBatchId(
                        validate_uuid(batch_id, field="review_batch_id")
                    ),
                    expected_batch_revision=self._require_revision(context),
                    source=source,
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def list_plugin_jobs(self, project_id: str | None = None) -> list[dict[str, Any]]:
        projects = self.projections.list_projects(include_archived=True)
        if project_id is not None:
            projects = tuple(item for item in projects if str(item.id) == project_id)
            if not projects:
                raise NotFoundError("Project was not found")
        jobs = [
            job
            for project in projects
            for job in self.projections.list_plugin_jobs(project.id)
        ]
        jobs.sort(key=lambda item: (item.updated_at, str(item.id)), reverse=True)
        return [encode_model(item) for item in jobs]

    def submit_plugin_job(
        self,
        project_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        if self._submit_plugin_job is None:
            raise ConflictError("Server agent queue is not configured")
        try:
            project = self.projections.get_project(_project_id(project_id))
            if project is None:
                raise ApplicationNotFoundError("Project was not found")
            layer_id = _layer_id(str(payload.get("layer_id", "")))
            source_id = RepresentationId(
                validate_uuid(str(payload.get("source_representation_id", "")), field="source_representation_id")
            )
            target_id = RepresentationId(
                validate_uuid(str(payload.get("target_representation_id", "")), field="target_representation_id")
            )
            source = self.projections.get_representation(source_id)
            target = self.projections.get_representation(target_id)
            if source is None or source.layer_id != layer_id or target is None or target.layer_id != layer_id:
                raise ApplicationNotFoundError("Source or target representation was not found in the layer")
            raw_coordinates = payload.get("coordinates", ())
            if not isinstance(raw_coordinates, list):
                raise TypeError("coordinates must be an array")
            coordinates = sorted(
                {(int(item[0]), int(item[1])) for item in raw_coordinates},
                key=lambda item: (item[1], item[0]),
            )
            selection = FrameSelectionV1(
                row_ranges=tuple(FrameRowRange(y=y, x_start=x, x_end=x) for x, y in coordinates)
            )
            inputs: list[PluginInputV1] = []
            missing: list[str] = []
            for coordinate in selection.iter_coordinates():
                frame_id = coordinate.frame_id(project.id)
                series_id = deterministic_frame_series_id(source.id, frame_id)
                version = self.projections.get_active_artifact_version(series_id)
                if version is None:
                    missing.append(f"({coordinate.x}, {coordinate.y})")
                    continue
                suffix = "." + version.filename.rsplit(".", 1)[-1] if "." in version.filename else ".bin"
                inputs.append(
                    PluginInputV1(
                        frame_id=frame_id,
                        artifact_version_id=version.id,
                        sha256=version.sha256,
                        relative_path=f"inputs/{coordinate.x}_{coordinate.y}{suffix}",
                        external_uri=(
                            None if version.external is None else version.external.uri
                        ),
                    )
                )
            if missing:
                raise ApplicationConflictError(
                    "Missing input artifact versions for frames: " + ", ".join(missing)
                )
            value = self._submit_plugin_job(
                SubmitPluginJobCommand(
                    context=self._application_context(context),
                    project_id=project.id,
                    layer_id=layer_id,
                    target_representation_id=target_id,
                    selection=selection,
                    capability=str(payload.get("capability", "")),
                    inputs=tuple(inputs),
                    parameters=(
                        dict(payload.get("parameters", {}))
                        if isinstance(payload.get("parameters", {}), Mapping)
                        else {}
                    ),
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def cancel_plugin_job(
        self,
        project_id: str,
        job_id: str,
        context: CommandContext,
    ) -> dict[str, Any]:
        if self._cancel_plugin_job is None:
            raise ConflictError("Server agent queue is not configured")
        try:
            value = self._cancel_plugin_job(
                CancelPluginJobCommand(
                    context=self._application_context(context),
                    project_id=_project_id(project_id),
                    job_id=PluginJobId(validate_uuid(job_id, field="job_id")),
                    expected_revision=self._require_revision(context),
                )
            )
            return encode_model(value)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def import_agent_result(self, job_id: str, agent: Any) -> dict[str, Any]:
        if self.agent_gateway is None:
            raise ConflictError("Server agent queue is not configured")
        publications = self.agent_gateway.publications_for(job_id, agent)
        if len(publications) != 1:
            raise ConflictError("Exactly one final V1 publication is required")
        try:
            transport = PluginResultManifest.from_dict(publications[0])
            result_manifest = domain_result_from_transport(transport)
            job = self.projections.get_plugin_job(
                PluginJobId(validate_uuid(job_id, field="job_id"))
            )
            if job is None:
                raise ApplicationNotFoundError("Plugin job was not found")
            actor = self.identities.get(job.actor_principal_id)
            if actor is None or not actor.active:
                raise ApplicationAuthorizationError("Initiating principal is inactive")
            references = {
                output.relative_path: self.agent_gateway.output_blob(job_id, output.output_id)
                for output in transport.outputs
            }
            imported = ImportPluginResultHandler(
                self.uow_factory,
                self.profiles,
                SystemClock(),
                _AgentBlobResultReader(self.uow_factory.blobs, references),
            )(
                ImportPluginResultCommand(
                    context=ApplicationCommandContext(
                        actor=actor,
                        idempotency_key=f"server-agent-import:{job_id}:{transport.completed_at}",
                        gitlab_identity_verified=True,
                    ),
                    manifest=result_manifest,
                    confirm_partial=False,
                )
            )
            return {
                "job": encode_model(imported.job),
                "versions": [encode_model(item) for item in imported.versions],
                "requires_partial_confirmation": imported.requires_partial_confirmation,
                "already_imported": imported.already_imported,
            }
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def fail_agent_job(self, job_id: str, error: str) -> dict[str, Any]:
        try:
            job = self.projections.get_plugin_job(
                PluginJobId(validate_uuid(job_id, field="job_id"))
            )
            if job is None:
                raise ApplicationNotFoundError("Plugin job was not found")
            actor = self.identities.get(job.actor_principal_id)
            if actor is None or not actor.active:
                raise ApplicationAuthorizationError("Initiating principal is inactive")
            manifest = PluginResultManifestV1(
                job_id=job.id,
                plugin_name="kraken-agent",
                plugin_version="1",
                results=(),
                parameters_applied={"error": error},
                protocol_version=job.protocol_version,
                outcome=PluginResultOutcome.FAILED,
            )
            imported = ImportPluginResultHandler(
                self.uow_factory,
                self.profiles,
                SystemClock(),
                _AgentBlobResultReader(self.uow_factory.blobs, {}),
            )(
                ImportPluginResultCommand(
                    context=ApplicationCommandContext(
                        actor=actor,
                        idempotency_key=f"server-agent-failed:{job_id}:{job.revision}",
                        gitlab_identity_verified=True,
                    ),
                    manifest=manifest,
                    confirm_partial=False,
                )
            )
            return {"job": encode_model(imported.job)}
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    @staticmethod
    def _decode_cursor(cursor: str | None) -> int:
        if not cursor:
            return 0
        try:
            raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
            value = int(json.loads(raw)["position"])
        except Exception as exc:
            raise ValidationError("Invalid history cursor") from exc
        if value < 0:
            raise ValidationError("Invalid history cursor")
        return value

    @staticmethod
    def _encode_cursor(position: int) -> str:
        raw = json.dumps({"position": position}, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    def history(self, project_id: str, *, cursor: str | None, limit: int) -> dict[str, Any]:
        self.get_project(project_id)
        if limit < 1 or limit > 500:
            raise ValidationError("History limit must be between 1 and 500")
        after = self._decode_cursor(cursor)
        rows = self.events.list_project_events(_project_id(project_id), after_position=after, limit=limit + 1)
        has_more = len(rows) > limit
        page = rows[:limit]
        items = [
            {
                "position": position,
                "event_id": event.event_id,
                "stream_id": event.stream_id,
                "revision": event.revision,
                "event_type": event.event_type,
                "schema_version": event.schema_version,
                "recorded_at": event.recorded_at.isoformat(),
                "effective_at": None if event.effective_at is None else event.effective_at.isoformat(),
                "actor": {
                    "principal_id": str(event.actor.principal_id),
                    "provider": event.actor.provider.value,
                    "subject": event.actor.subject,
                    "display_name": event.actor.display_name,
                },
                "performer_id": None if event.performer_id is None else str(event.performer_id),
                "payload": dict(event.payload),
            }
            for position, event in page
        ]
        next_cursor = self._encode_cursor(page[-1][0]) if has_more and page else None
        return {"items": items, "next_cursor": next_cursor}

    def statistics(
        self,
        project_id: str,
        *,
        start: datetime,
        end: datetime,
        timezone: Any = UTC,
    ) -> dict[str, Any]:
        project = self.projections.get_project(_project_id(project_id))
        if project is None:
            raise NotFoundError("Project was not found")
        records: list[ActivityRecord] = []
        after = 0
        while True:
            page = self.events.list_project_events(
                project.id, after_position=after, limit=500
            )
            if not page:
                break
            records.extend(ActivityRecord.from_event(event) for _position, event in page)
            after = page[-1][0]
        filters = ReportFilters(start, end, project_ids=frozenset((str(project.id),)))
        reports = ReportService()
        metrics = reports.aggregate(records, filters)
        return {
            "metrics": encode_model(metrics),
            "series": {
                granularity.value: encode_model(
                    reports.aggregate_series(
                        records, filters, granularity, timezone=timezone
                    )
                )
                for granularity in ReportGranularity
            },
        }

    def _append_auxiliary_event(
        self,
        project_id: str,
        *,
        stream_id: str,
        event_type: str,
        payload: Mapping[str, object],
        program: ProgramSnapshot,
        context: CommandContext,
    ) -> dict[str, Any]:
        actor = self._actor(context.actor_id)
        identifier = _project_id(project_id)
        try:
            with self.uow_factory() as uow:
                project = uow.projections.get_project(identifier)
                if project is None:
                    raise ApplicationNotFoundError("Project was not found")
                profile = self.profiles.get(project.storage_profile)
                if profile is None:
                    raise ApplicationNotFoundError("Storage profile was not found")
                AuthorizationPolicy().require(
                    principal=actor,
                    storage=profile,
                    permission=Permission.RUN_PLUGIN,
                    roles=uow.acl.roles_for(project.id, actor.id),
                    gitlab_identity_verified=True,
                )
                prior = uow.event_store.find_by_idempotency_key(
                    project.id, context.idempotency_key
                )
                if prior:
                    return encode_model(prior[-1])
                revision = uow.event_store.current_revision(stream_id)
                if context.expected_revision != revision:
                    raise ApplicationConcurrencyError(
                        f"Expected stream revision {context.expected_revision}, found {revision}"
                    )
                event = EventEnvelope.create(
                    stream_id=stream_id,
                    project_id=project.id,
                    revision=revision + 1,
                    event_type=event_type,
                    payload=payload,
                    actor=ActorSnapshot.from_principal(actor),
                    program=program,
                    idempotency_key=context.idempotency_key,
                )
                uow.event_store.append(
                    stream_id, expected_revision=revision, events=(event,)
                )
                uow.commit()
                return encode_model(event)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

    def publish_karakal_analysis(
        self,
        project_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        layer_id = str(payload.get("layer_id", ""))
        try:
            layer = self.projections.get_layer(_layer_id(layer_id))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Invalid layer id") from exc
        if layer is None or str(layer.project_id) != project_id:
            raise NotFoundError("Layer was not found")
        plugin_version = str(payload.get("plugin_version", "")).strip()
        if not plugin_version:
            raise ValidationError("Karakal plugin version is required")
        confidence = {
            str(frame_id): float(value)
            for frame_id, value in dict(payload.get("frame_confidence", {})).items()
        }
        if any(value < 0.0 or value > 1.0 for value in confidence.values()):
            raise ValidationError("Karakal confidence must be in range 0..1")
        stream_id = f"karakal:{layer_id}"
        revision = self.events.current_revision(stream_id)
        return self._append_auxiliary_event(
            project_id,
            stream_id=stream_id,
            event_type="KarakalAnalysisPublished",
            payload={
                "run_id": str(payload.get("run_id", "")),
                "layer_id": layer_id,
                "publication_sequence": revision + 1,
                "frame_confidence": confidence,
                "report": dict(payload.get("report", {})),
                "parameters": dict(payload.get("parameters", {})),
                "plugin_version": plugin_version,
            },
            program=ProgramSnapshot("Karakal", plugin_version),
            context=context,
        )

    def append_pipeline_event(
        self,
        project_id: str,
        payload: Mapping[str, Any],
        context: CommandContext,
    ) -> dict[str, Any]:
        layer_id = str(payload.get("layer_id", ""))
        try:
            layer = self.projections.get_layer(_layer_id(layer_id))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Invalid layer id") from exc
        if layer is None or str(layer.project_id) != project_id:
            raise NotFoundError("Layer was not found")
        event_type = str(payload.get("event_type", ""))
        if event_type not in {"LayerPipelineActionRequested", "LayerPipelineActionRemoved"}:
            raise ValidationError("Unsupported pipeline event type")
        return self._append_auxiliary_event(
            project_id,
            stream_id=f"layer-pipeline:{layer_id}",
            event_type=event_type,
            payload={key: value for key, value in payload.items() if key != "event_type"},
            program=ProgramSnapshot(str(payload.get("plugin_id") or "Kraken")),
            context=context,
        )


__all__ = ["PostgresServerServices", "ServerStorageProfiles", "SystemClock"]
