"""Shared command entry point assembled from application handlers and abstract ports.

Local storage and server adapters inject different UnitOfWorkFactory instances;
command selection, authorization and domain rules are defined here once.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from kraken_manager.domain.project import Project, Layer, Representation
from kraken_manager.domain.artifacts import ArtifactVersion, ArtifactSeries, NoteRevision
from kraken_manager.domain.workflows import ReviewBatch, PluginJob
from kraken_manager.domain.identity import ProjectRole
from kraken_manager.application.dto import ReviewPackagePlan

from kraken_manager.application.acl import AssignProjectRoleHandler, RevokeProjectRoleHandler
from kraken_manager.application.artifact_lifecycle import (
    ActivateArtifactVersionHandler,
    AddExternalArtifactVersionHandler,
    ArchiveArtifactSeriesHandler,
    CreateNoteHandler,
    RenameArtifactSeriesHandler,
    ReviseNoteHandler,
)
from kraken_manager.application.dto import (
    AcceptReviewCommand,
    ActivateArtifactVersionCommand,
    ActivateRepresentationCommand,
    AddArtifactVersionCommand,
    AddExternalArtifactVersionCommand,
    ArchiveArtifactSeriesCommand,
    ArchiveLayerCommand,
    ArchiveProjectCommand,
    ArchiveRepresentationCommand,
    AssignProjectRoleCommand,
    CancelReviewBatchCommand,
    CreateArtifactSeriesCommand,
    CreateLayerCommand,
    CreateNoteCommand,
    CreateProjectCommand,
    CreateRepresentationCommand,
    DeactivateRepresentationCommand,
    DeleteLayerCommand,
    PlanReviewPackageCommand,
    RenameArtifactSeriesCommand,
    RenameLayerCommand,
    RenameProjectCommand,
    RenameRepresentationCommand,
    ReorderLayerCommand,
    ReorderLayersCommand,
    RequestReviewChangesCommand,
    RestoreProjectCommand,
    ReviseNoteCommand,
    RevokeProjectRoleCommand,
    SynchronizePluginJobCommand,
    UpdateRepresentationNoteCommand,
)
from kraken_manager.application.errors import StorageCapabilityError
from kraken_manager.application.layer_deletion import DeleteLayerHandler, LayerDeletionStorage
from kraken_manager.application.lifecycle import (
    ArchiveLayerHandler,
    ArchiveProjectHandler,
    RenameLayerHandler,
    RenameProjectHandler,
    ReorderLayerHandler,
    ReorderLayersHandler,
    RestoreProjectHandler,
)
from kraken_manager.application.plugin_jobs_parts.synchronize_plugin_job_handler import SynchronizePluginJobHandler
from kraken_manager.application.ports import Clock, StorageProfileCatalog, UnitOfWorkFactory
from kraken_manager.application.representation_lifecycle import (
    ActivateRepresentationHandler,
    ArchiveRepresentationHandler,
    DeactivateRepresentationHandler,
    RenameRepresentationHandler,
    UpdateRepresentationNoteHandler,
)
from kraken_manager.application.review_workflow_parts.accept_review_handler import AcceptReviewHandler
from kraken_manager.application.review_workflow_parts.plan_review_package_handler import PlanReviewPackageHandler
from kraken_manager.application.review_workflow_parts.request_review_changes_handler import (
    CancelReviewBatchHandler,
    RequestReviewChangesHandler,
)
from kraken_manager.application.use_cases import (
    AddArtifactVersionHandler,
    CreateArtifactSeriesHandler,
    CreateLayerHandler,
    CreateProjectHandler,
    CreateRepresentationHandler,
)
from kraken_manager.application.workspace_rename import RenameWorkspaceHandler, WorkspaceRenameStorage


class CommandAudience(StrEnum):
    USER_PROJECT = "user_project"
    USER_CATALOG = "user_catalog"
    CONTENT_PORT = "content_port"
    EXECUTOR = "executor"


@dataclass(frozen=True, slots=True)
class ProjectCommandDefinition:
    command_type: type
    handler_type: type
    result_type: object
    audience: CommandAudience


_COMMAND_HANDLERS = {
    AcceptReviewCommand: ProjectCommandDefinition(
        AcceptReviewCommand, AcceptReviewHandler, ReviewBatch, CommandAudience.USER_PROJECT
    ),
    ActivateArtifactVersionCommand: ProjectCommandDefinition(
        ActivateArtifactVersionCommand, ActivateArtifactVersionHandler, ArtifactVersion, CommandAudience.USER_PROJECT
    ),
    ActivateRepresentationCommand: ProjectCommandDefinition(
        ActivateRepresentationCommand, ActivateRepresentationHandler, Representation, CommandAudience.USER_PROJECT
    ),
    AddArtifactVersionCommand: ProjectCommandDefinition(
        AddArtifactVersionCommand, AddArtifactVersionHandler, ArtifactVersion, CommandAudience.CONTENT_PORT
    ),
    AddExternalArtifactVersionCommand: ProjectCommandDefinition(
        AddExternalArtifactVersionCommand,
        AddExternalArtifactVersionHandler,
        ArtifactVersion,
        CommandAudience.USER_PROJECT,
    ),
    ArchiveArtifactSeriesCommand: ProjectCommandDefinition(
        ArchiveArtifactSeriesCommand, ArchiveArtifactSeriesHandler, ArtifactSeries, CommandAudience.USER_PROJECT
    ),
    ArchiveLayerCommand: ProjectCommandDefinition(
        ArchiveLayerCommand, ArchiveLayerHandler, Layer, CommandAudience.USER_PROJECT
    ),
    ArchiveProjectCommand: ProjectCommandDefinition(
        ArchiveProjectCommand, ArchiveProjectHandler, Project, CommandAudience.USER_PROJECT
    ),
    ArchiveRepresentationCommand: ProjectCommandDefinition(
        ArchiveRepresentationCommand, ArchiveRepresentationHandler, Representation, CommandAudience.USER_PROJECT
    ),
    AssignProjectRoleCommand: ProjectCommandDefinition(
        AssignProjectRoleCommand, AssignProjectRoleHandler, frozenset[ProjectRole], CommandAudience.USER_PROJECT
    ),
    CancelReviewBatchCommand: ProjectCommandDefinition(
        CancelReviewBatchCommand, CancelReviewBatchHandler, ReviewBatch, CommandAudience.USER_PROJECT
    ),
    CreateArtifactSeriesCommand: ProjectCommandDefinition(
        CreateArtifactSeriesCommand, CreateArtifactSeriesHandler, ArtifactSeries, CommandAudience.USER_PROJECT
    ),
    CreateLayerCommand: ProjectCommandDefinition(
        CreateLayerCommand, CreateLayerHandler, Layer, CommandAudience.USER_PROJECT
    ),
    CreateNoteCommand: ProjectCommandDefinition(
        CreateNoteCommand, CreateNoteHandler, NoteRevision, CommandAudience.USER_PROJECT
    ),
    CreateProjectCommand: ProjectCommandDefinition(
        CreateProjectCommand, CreateProjectHandler, Project, CommandAudience.USER_CATALOG
    ),
    CreateRepresentationCommand: ProjectCommandDefinition(
        CreateRepresentationCommand, CreateRepresentationHandler, Representation, CommandAudience.USER_PROJECT
    ),
    DeactivateRepresentationCommand: ProjectCommandDefinition(
        DeactivateRepresentationCommand, DeactivateRepresentationHandler, Representation, CommandAudience.USER_PROJECT
    ),
    DeleteLayerCommand: ProjectCommandDefinition(
        DeleteLayerCommand, DeleteLayerHandler, Layer, CommandAudience.USER_PROJECT
    ),
    PlanReviewPackageCommand: ProjectCommandDefinition(
        PlanReviewPackageCommand, PlanReviewPackageHandler, ReviewPackagePlan, CommandAudience.USER_PROJECT
    ),
    RenameArtifactSeriesCommand: ProjectCommandDefinition(
        RenameArtifactSeriesCommand, RenameArtifactSeriesHandler, ArtifactSeries, CommandAudience.USER_PROJECT
    ),
    RenameLayerCommand: ProjectCommandDefinition(
        RenameLayerCommand, RenameLayerHandler, Layer, CommandAudience.USER_PROJECT
    ),
    RenameProjectCommand: ProjectCommandDefinition(
        RenameProjectCommand, RenameProjectHandler, Project, CommandAudience.USER_PROJECT
    ),
    RenameRepresentationCommand: ProjectCommandDefinition(
        RenameRepresentationCommand, RenameRepresentationHandler, Representation, CommandAudience.USER_PROJECT
    ),
    ReorderLayerCommand: ProjectCommandDefinition(
        ReorderLayerCommand, ReorderLayerHandler, Layer, CommandAudience.USER_PROJECT
    ),
    ReorderLayersCommand: ProjectCommandDefinition(
        ReorderLayersCommand, ReorderLayersHandler, tuple[Layer, ...], CommandAudience.USER_PROJECT
    ),
    RequestReviewChangesCommand: ProjectCommandDefinition(
        RequestReviewChangesCommand, RequestReviewChangesHandler, ReviewBatch, CommandAudience.USER_PROJECT
    ),
    RestoreProjectCommand: ProjectCommandDefinition(
        RestoreProjectCommand, RestoreProjectHandler, Project, CommandAudience.USER_PROJECT
    ),
    ReviseNoteCommand: ProjectCommandDefinition(
        ReviseNoteCommand, ReviseNoteHandler, NoteRevision, CommandAudience.USER_PROJECT
    ),
    RevokeProjectRoleCommand: ProjectCommandDefinition(
        RevokeProjectRoleCommand, RevokeProjectRoleHandler, frozenset[ProjectRole], CommandAudience.USER_PROJECT
    ),
    SynchronizePluginJobCommand: ProjectCommandDefinition(
        SynchronizePluginJobCommand, SynchronizePluginJobHandler, PluginJob, CommandAudience.EXECUTOR
    ),
    UpdateRepresentationNoteCommand: ProjectCommandDefinition(
        UpdateRepresentationNoteCommand, UpdateRepresentationNoteHandler, Representation, CommandAudience.USER_PROJECT
    ),
}


def user_project_command_definition(command_type: type) -> ProjectCommandDefinition:
    definition = _COMMAND_HANDLERS.get(command_type)
    if definition is None or definition.audience is not CommandAudience.USER_PROJECT:
        raise ValueError(f"Not a user project command: {command_type.__name__}")
    return definition


def user_project_command_definitions() -> tuple[ProjectCommandDefinition, ...]:
    return tuple(item for item in _COMMAND_HANDLERS.values() if item.audience is CommandAudience.USER_PROJECT)


class ProjectCommandService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        profiles: StorageProfileCatalog,
        clock: Clock,
        *,
        rename_storage: WorkspaceRenameStorage | None = None,
        deletion_storage: LayerDeletionStorage | None = None,
    ):
        self._handlers = {
            command: definition.handler_type(uow_factory, profiles, clock)
            for command, definition in _COMMAND_HANDLERS.items()
            if command is not DeleteLayerCommand
        }
        if rename_storage is not None:
            for command, is_layer in ((RenameProjectCommand, False), (RenameLayerCommand, True)):
                self._handlers[command] = RenameWorkspaceHandler(
                    uow_factory,
                    profiles,
                    clock,
                    rename_storage,
                    self._handlers[command],
                    layer=is_layer,
                )
        if deletion_storage is not None:
            self._handlers[DeleteLayerCommand] = DeleteLayerHandler(
                uow_factory,
                profiles,
                clock,
                deletion_storage,
                archive=self._handlers[ArchiveLayerCommand],
            )

    @property
    def command_types(self) -> frozenset[type]:
        return frozenset(self._handlers)

    def execute(self, command: object, chunks: Iterable[bytes] | None = None, *, stored=None):
        handler = self._handlers.get(type(command))
        if handler is None:
            if isinstance(command, DeleteLayerCommand):
                raise StorageCapabilityError("Layer-deletion storage must be supplied by the application host")
            raise TypeError(f"Unknown project command: {type(command).__name__}")
        if isinstance(command, AddArtifactVersionCommand):
            return handler(command, chunks, stored=stored)
        if chunks is not None or stored is not None:
            raise TypeError("Content is only accepted for AddArtifactVersionCommand")
        return handler(command)
