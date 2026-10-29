"""persistent_services: construction."""
from __future__ import annotations
from typing import Any
from kraken_manager.application.acl import AssignProjectRoleHandler, RevokeProjectRoleHandler
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
from kraken_manager.infrastructure.postgres import PostgresEventStore, PostgresIdentityAclStore, PostgresProjectionStore
from kraken_manager.infrastructure.review import ReviewPackageReader, ReviewPackageWriter

from kraken_server.persistent_services_parts.layer_source_scan import (
    ServerStorageProfiles,
    SystemClock,
    resolve_storage_path,
)


class ConstructionMixin:
    def resolve_storage_path(self, value):
        return resolve_storage_path(self, value)

    def __init__(
        self,
        engine: Any,
        uow_factory: Any,
        *,
        profiles: ServerStorageProfiles | None = None,
        agent_gateway: Any | None = None,
        performer_store: Any | None = None,
        review_key_pair: Any | None = None,
        workspace_files: Any | None = None,
        source_root: str | None = None,
        derived_root: str | None = None,
        recovery_roots_required: bool = False,
    ) -> None:
        self.engine = engine
        self.uow_factory = uow_factory
        self.profiles = profiles or ServerStorageProfiles()
        self.agent_gateway = agent_gateway
        self.performer_store = performer_store
        self.review_key_pair = review_key_pair
        self.workspace_files = workspace_files
        self.source_root = source_root
        self.derived_root = derived_root
        self.recovery_roots_required = recovery_roots_required
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
