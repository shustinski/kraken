from __future__ import annotations

import inspect

from kraken_hub import manager_app
from kraken_hub.composition import EmbeddedProjectService
from kraken_manager.application import dto

# Every application command that represents a user-visible workflow must have
# both a Desktop service route and a discoverable Qt action/controller route.
DESKTOP_COMMAND_ROUTES = {
    "CreateProjectCommand": ("create_project", "def create_project"),
    "CreateLayerCommand": ("create_layer", "def _add_layer"),
    "CreateRepresentationCommand": ("create_representation", "def _add_representation"),
    "RenameProjectCommand": ("rename_project", "def rename_project"),
    "ArchiveProjectCommand": ("archive_project", "def archive_project"),
    "RestoreProjectCommand": ("restore_project", "def restore_project"),
    "RenameLayerCommand": ("rename_layer", '"rename_layer"'),
    "ReorderLayerCommand": ("reorder_layer", "def _layer_manager_reorder"),
    "ReorderLayersCommand": ("reorder_layers", "def _layer_manager_reorder"),
    "ArchiveLayerCommand": ("archive_layer", '"archive_layer"'),
    "AssignProjectRoleCommand": ("assign_project_role", "def manage_project_participants"),
    "RevokeProjectRoleCommand": ("revoke_project_role", "def manage_project_participants"),
    "RenameRepresentationCommand": ("rename_representation", '"rename_representation"'),
    "UpdateRepresentationNoteCommand": (
        "update_representation_note",
        '"edit_representation_note"',
    ),
    "ActivateRepresentationCommand": ("activate_representation", '"activate_representation"'),
    "DeactivateRepresentationCommand": (
        "deactivate_representation",
        '"deactivate_representation"',
    ),
    "ArchiveRepresentationCommand": ("archive_representation", '"archive_representation"'),
    "AddArtifactVersionCommand": ("add_managed_artifact_version", "def add_version"),
    "CreateArtifactSeriesCommand": ("create_artifact_series", "def create_file_series"),
    "RenameArtifactSeriesCommand": ("rename_artifact_series", "def rename_series"),
    "ArchiveArtifactSeriesCommand": ("archive_artifact_series", "def archive_series"),
    "ActivateArtifactVersionCommand": ("activate_artifact_version", "def activate_version"),
    "AddExternalArtifactVersionCommand": ("add_external_artifact_version", "external=True"),
    "CreateNoteCommand": ("create_note", "def add_note"),
    "ReviseNoteCommand": ("revise_note", "def revise_note"),
    "SubmitPluginJobCommand": ("submit_plugin_job", "def _submit_agent_action"),
    "CancelPluginJobCommand": ("cancel_plugin_job", "def cancel_agent_job"),
    "RetryPluginJobCommand": ("retry_plugin_job", "def retry_agent_job"),
    "SynchronizePluginJobCommand": ("synchronize_plugin_jobs", "def _poll_agent_jobs"),
    "ImportPluginResultCommand": ("import_agent_result", "def _poll_agent_jobs"),
    "CreateReviewBatchCommand": ("create_review_batch", "def send_selection_for_review"),
    "PlanReviewPackageCommand": ("export_review_batch", "def send_selection_for_review"),
    "ExportReviewPackageCommand": ("export_review_batch", "def repeat_export"),
    "DryRunReviewReturnCommand": ("review_return_preflight", "def load_review_return"),
    "CommitReviewReturnCommand": ("commit_review_return", "def load_review_return"),
    "AcceptReviewCommand": ("accept_review", "def accept"),
    "RequestReviewChangesCommand": ("request_review_changes", "def request_changes"),
    "CancelReviewBatchCommand": ("cancel_review_batch", "def cancel"),
}

# These are domain infrastructure/value types rather than executable commands.
TECHNICAL_TYPE_ALLOWLIST = frozenset(
    {
        "BlobRef",
        "EventEnvelope",
        "FrameSelectionV1",
        "FrameCoordinate",
        "SpatialBounds",
    }
)


def test_every_application_command_is_routed_to_desktop_service_and_qt() -> None:
    commands = {
        name
        for name, value in vars(dto).items()
        if name.endswith("Command") and inspect.isclass(value)
    }
    assert commands == set(DESKTOP_COMMAND_ROUTES)

    ui_source = inspect.getsource(manager_app)
    for command, (service_method, ui_marker) in DESKTOP_COMMAND_ROUTES.items():
        assert hasattr(EmbeddedProjectService, service_method), (
            f"{command} has no EmbeddedProjectService.{service_method}"
        )
        assert ui_marker in ui_source, f"{command} has no Qt route marker {ui_marker!r}"

    assert not commands.intersection(TECHNICAL_TYPE_ALLOWLIST)
