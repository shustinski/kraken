"""Production service facade backed by the clean-architecture application API."""

from __future__ import annotations
import base64
import json
import mimetypes
from dataclasses import replace
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from typing import Any
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5
from kraken_core.safe_files import contained_path, ensure_regular_directory, open_regular_read
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
    StoredContent,
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
from kraken_manager.domain.artifacts import ArtifactScope, BlobRef
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
from kraken_manager.domain.identity import Permission, SystemRole, parse_project_role
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

from .persistent_services_parts.request_project_deletion import RequestProjectDeletionMixin
from .persistent_services_parts.construction import ConstructionMixin
from .persistent_services_parts.create_project import CreateProjectMixin
from .persistent_services_parts.create_external_layer import CreateExternalLayerMixin
from .persistent_services_parts.materialize_representation_inputs import MaterializeRepresentationInputsMixin
from .persistent_services_parts.attach_workspace_layer import AttachWorkspaceLayerMixin
from .persistent_services_parts.discard_workspace_layer import DiscardWorkspaceLayerMixin
from .persistent_services_parts.create_representation import CreateRepresentationMixin
from .persistent_services_parts.update_representation import UpdateRepresentationMixin
from .persistent_services_parts.active_artifact_versions import ActiveArtifactVersionsMixin
from .persistent_services_parts.matrix_viewport import MatrixViewportMixin
from .persistent_services_parts.create_artifact_series import CreateArtifactSeriesMixin
from .persistent_services_parts.mutate_artifact_series import MutateArtifactSeriesMixin
from .persistent_services_parts.create_review_batch import CreateReviewBatchMixin
from .persistent_services_parts.mutate_review_batch import MutateReviewBatchMixin
from .persistent_services_parts.start_desktop_job import StartDesktopJobMixin
from .persistent_services_parts.submit_plugin_job import SubmitPluginJobMixin
from .persistent_services_parts.import_agent_result import ImportAgentResultMixin
from .persistent_services_parts.statistics import StatisticsMixin
from .persistent_services_parts.append_auxiliary_event import AppendAuxiliaryEventMixin
from .persistent_services_parts.append_pipeline_event import AppendPipelineEventMixin
from .persistent_services_parts.layer_source_scan import (
    SystemClock,
    _AgentBlobResultReader,
    ServerStorageProfiles,
    _project_dict,
    _layer_dict,
    _representation_dict,
    _project_id,
    _layer_source_scan,
    _layer_id,
)

class PostgresServerServices(
    RequestProjectDeletionMixin,
    ConstructionMixin,
    CreateProjectMixin,
    CreateExternalLayerMixin,
    MaterializeRepresentationInputsMixin,
    AttachWorkspaceLayerMixin,
    DiscardWorkspaceLayerMixin,
    CreateRepresentationMixin,
    UpdateRepresentationMixin,
    ActiveArtifactVersionsMixin,
    MatrixViewportMixin,
    CreateArtifactSeriesMixin,
    MutateArtifactSeriesMixin,
    CreateReviewBatchMixin,
    MutateReviewBatchMixin,
    StartDesktopJobMixin,
    SubmitPluginJobMixin,
    ImportAgentResultMixin,
    StatisticsMixin,
    AppendAuxiliaryEventMixin,
    AppendPipelineEventMixin,
):
    """Thin transport facade; mutations execute application command handlers."""

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    SystemClock,
    _AgentBlobResultReader,
    ServerStorageProfiles,
    _project_dict,
    _layer_dict,
    _representation_dict,
    _project_id,
    _layer_source_scan,
    _layer_id,
)

__all__ = ["PostgresServerServices", "ServerStorageProfiles", "SystemClock"]
