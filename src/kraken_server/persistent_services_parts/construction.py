"""persistent_services: construction."""
from __future__ import annotations
from typing import Any
from kraken_manager.application.project_commands import ProjectCommandService
from kraken_manager.infrastructure.workspace_rename import FilesystemWorkspaceRenameStorage
from kraken_manager.infrastructure.layer_deletion import WorkspaceLayerDeletionStorage
from kraken_manager.application.plugin_jobs import (
    CancelPluginJobHandler,
    SubmitPluginJobHandler,
)
from kraken_manager.application.review_workflow import (
    CreateReviewBatchHandler,
    CommitReviewReturnHandler,
    DryRunReviewReturnHandler,
    ExportReviewPackageHandler,
)
from kraken_manager.infrastructure.postgres import PostgresEventStore, PostgresIdentityAclStore, PostgresProjectionStore
from kraken_manager.infrastructure.review import ReviewPackageReader, ReviewPackageWriter

from kraken_server.persistent_services_parts.layer_source_scan import (
    ServerStorageProfiles,
    SystemClock,
    resolve_storage_path,
)


class ConstructionMixin:
    def execute_project_command(self, project_id, payload, context):
        from kraken_manager.infrastructure.project_command_codec import (
            decode_user_project_command, encode_project_command_result,
        )

        try:
            command = decode_user_project_command(project_id, payload, self._application_context(context))
            result = self.commands.execute(command)
            return encode_project_command_result(command, result)
        except Exception as exc:
            self._translate_lifecycle_error(exc)
            raise AssertionError("unreachable")

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
        self.commands = ProjectCommandService(
            uow_factory, self.profiles, SystemClock(),
            rename_storage=FilesystemWorkspaceRenameStorage(
                workspace_files,
                project_changed=lambda project_id, binding: self.projections.record_project_directories(
                    project_id, source_directory=binding.source_project_dir,
                    derived_directory=binding.derived_project_dir,
                ),
            ),
            deletion_storage=WorkspaceLayerDeletionStorage(workspace_files),
        )
        self._create_project = self.commands.execute
        self._create_layer = self.commands.execute
        self._create_representation = self.commands.execute
        self._rename_project = self.commands.execute
        self._archive_project = self.commands.execute
        self._restore_project = self.commands.execute
        self._rename_layer = self.commands.execute
        self._reorder_layer = self.commands.execute
        self._reorder_layers = self.commands.execute
        self._archive_layer = self.commands.execute
        self._delete_layer = self.commands.execute
        self._assign_project_role = self.commands.execute
        self._revoke_project_role = self.commands.execute
        self._activate_representation = self.commands.execute
        self._deactivate_representation = self.commands.execute
        self._archive_representation = self.commands.execute
        self._rename_representation = self.commands.execute
        self._update_representation_note = self.commands.execute
        self._create_artifact_series = self.commands.execute
        self._add_artifact_version = self.commands.execute
        self._rename_artifact_series = self.commands.execute
        self._archive_artifact_series = self.commands.execute
        self._activate_artifact_version = self.commands.execute
        self._add_external_artifact_version = self.commands.execute
        self._create_note = self.commands.execute
        self._revise_note = self.commands.execute
        self._create_review_batch = (
            None
            if performer_store is None
            else CreateReviewBatchHandler(
                uow_factory, self.profiles, SystemClock(), performer_store
            )
        )
        self._accept_review = self.commands.execute
        self._request_review_changes = self.commands.execute
        self._cancel_review_batch = self.commands.execute
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
