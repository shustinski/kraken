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

from .composition_parts.revoke_project_role import RevokeProjectRoleMixin
from .composition_parts.record_workspace_event import RecordWorkspaceEventMixin
from .composition_parts.create_project import CreateProjectMixin
from .composition_parts.bind_project_workspace import BindProjectWorkspaceMixin
from .composition_parts.create_layer_disk import CreateLayerDiskMixin
from .composition_parts.create_external_layer import CreateExternalLayerMixin
from .composition_parts.begin_derived_run import BeginDerivedRunMixin
from .composition_parts.delete_layer import DeleteLayerMixin
from .composition_parts.publish_workspace_plugin_result import PublishWorkspacePluginResultMixin
from .composition_parts.create_representation import CreateRepresentationMixin
from .composition_parts.commit_managed_import import CommitManagedImportMixin
from .composition_parts.materialize_representation_inputs import MaterializeRepresentationInputsMixin
from .composition_parts.rename_project import RenameProjectMixin
from .composition_parts.staging_entry_belongs_project import StagingEntryBelongsProjectMixin
from .composition_parts.delete_project import DeleteProjectMixin
from .composition_parts.rename_layer import RenameLayerMixin
from .composition_parts.create_artifact_series import CreateArtifactSeriesMixin
from .composition_parts.add_external_artifact_version import AddExternalArtifactVersionMixin
from .composition_parts.export_artifact_version import ExportArtifactVersionMixin
from .composition_parts.frame_cells import FrameCellsMixin
from .composition_parts.frame_management_states import FrameManagementStatesMixin
from .composition_parts.publish_karakal_analysis import PublishKarakalAnalysisMixin
from .composition_parts.record_layer_pipeline_action import RecordLayerPipelineActionMixin
from .composition_parts.remove_layer_pipeline_action import RemoveLayerPipelineActionMixin
from .composition_parts.latest_karakal_analysis import LatestKarakalAnalysisMixin
from .composition_parts.matrix_viewport import MatrixViewportMixin
from .composition_parts.reorder_layers import ReorderLayersMixin
from .composition_parts.external_directory_viewport import ExternalDirectoryViewportMixin
from .composition_parts.plugin_jobs import PluginJobsMixin
from .composition_parts.submit_plugin_job import SubmitPluginJobMixin
from .composition_parts.synchronize_plugin_jobs import SynchronizePluginJobsMixin
from .composition_parts.import_agent_result import ImportAgentResultMixin
from .composition_parts.import_agent_publications import ImportAgentPublicationsMixin
from .composition_parts.domain_selection import DomainSelectionMixin
from .composition_parts.create_review_batch import CreateReviewBatchMixin
from .composition_parts.issue_work import IssueWorkMixin
from .composition_parts.review_return_preflight import ReviewReturnPreflightMixin
from .composition_parts.import_backup import ImportBackupMixin
from .composition_parts.frame_cell_snapshot import (
    default_data_dir,
    SystemClock,
    DesktopStorageProfiles,
    DesktopSession,
    ManagedImportResult,
    IntegrityScanResult,
    FrameCellSnapshot,
    FrameManagementState,
    KarakalAnalysisRun,
    ProjectDeletionResult,
)

class EmbeddedProjectService(
    RevokeProjectRoleMixin,
    RecordWorkspaceEventMixin,
    CreateProjectMixin,
    BindProjectWorkspaceMixin,
    CreateLayerDiskMixin,
    CreateExternalLayerMixin,
    BeginDerivedRunMixin,
    DeleteLayerMixin,
    PublishWorkspacePluginResultMixin,
    CreateRepresentationMixin,
    CommitManagedImportMixin,
    MaterializeRepresentationInputsMixin,
    RenameProjectMixin,
    StagingEntryBelongsProjectMixin,
    DeleteProjectMixin,
    RenameLayerMixin,
    CreateArtifactSeriesMixin,
    AddExternalArtifactVersionMixin,
    ExportArtifactVersionMixin,
    FrameCellsMixin,
    FrameManagementStatesMixin,
    PublishKarakalAnalysisMixin,
    RecordLayerPipelineActionMixin,
    RemoveLayerPipelineActionMixin,
    LatestKarakalAnalysisMixin,
    MatrixViewportMixin,
    ReorderLayersMixin,
    ExternalDirectoryViewportMixin,
    PluginJobsMixin,
    SubmitPluginJobMixin,
    SynchronizePluginJobsMixin,
    ImportAgentResultMixin,
    ImportAgentPublicationsMixin,
    DomainSelectionMixin,
    CreateReviewBatchMixin,
    IssueWorkMixin,
    ReviewReturnPreflightMixin,
    ImportBackupMixin,
):
    """Autonomous application service for machine-local file projects."""

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    default_data_dir,
    SystemClock,
    DesktopStorageProfiles,
    DesktopSession,
    ManagedImportResult,
    IntegrityScanResult,
    FrameCellSnapshot,
    FrameManagementState,
    KarakalAnalysisRun,
    ProjectDeletionResult,
)

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
