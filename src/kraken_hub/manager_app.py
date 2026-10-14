"""PyQt Desktop composition for projects, matrix and legacy plugin launcher."""

from __future__ import annotations
import hashlib
import logging
import os
import shutil
import sys
import platform
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Event
from uuid import uuid4
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal
from kraken_core.analysis_protocol import (
    AnalysisArtifactInput,
    AnalysisFrameInput,
    AnalysisParameter,
    AnalysisSourceRole,
)
from kraken_core.analysis_run_protocol import (
    AnalysisRecipe,
    AnalysisRuntimeIdentity,
    AnalysisSourceBinding,
    canonical_json,
    payload_sha256,
)
from kraken_core.external_model import ExternalModelLink
from kraken_core.frame_matrix import (
    MatrixSession,
    StoreNamespace,
    StorePolicy,
    ThumbnailStoreFactory,
)
from kraken_core.frame_matrix.qt import FrameMatrixWidget
from kraken_core.plugin_protocol import (
    PluginFrameInput,
    PluginJobManifest,
    PluginResultManifest,
    WorkspacePluginContextV1,
    WorkspacePluginResultV1,
)
from kraken_core.plugins import PluginInventoryItem, is_plugin_launchable
from kraken_core.qt import configure_application_identity
from kraken_core.styles import load_shared_stylesheet
from kraken_manager.application.imports import ImportMappingMode
from kraken_manager.application.analysis_runs import AnalysisRunCoordinator
from kraken_manager.domain.artifacts import ArtifactScope, ArtifactSeries, ArtifactVersion
from kraken_manager.domain.identity import Performer, Permission, ProjectRole
from kraken_manager.domain.project import GridOrientation as DomainOrientation
from kraken_manager.domain.project import LayerType, Project, RepresentationKind, RepresentationPurpose
from kraken_manager.infrastructure.analysis import FilesystemAnalysisStore
from kraken_manager.infrastructure.plugin import AgentAnalysisGateway, AgentPluginGateway
from kraken_manager.infrastructure.workspace_files import validate_workspace_roots
from kraken_manager.presentation.qt import (
    LayerCreationDialog,
    LayerManagerDialog,
    LayerPipelineSnapshot,
    ObjectHistoryEntry,
    ObjectPropertiesDialog,
    ObjectPropertiesSnapshot,
    PipelineLane,
    PipelineNode,
    ProjectManagerShell,
    ProjectWorkspacePage,
)
from kraken_manager.presentation.qt.models import LayerListItem, ProjectListItem
from kraken_manager.presentation.qt.widgets import ClickableLabel, GridDimensionsWidget
from kraken_manager.workspace import DerivedRunKind, ProjectWorkspaceBinding
from . import windows_credentials
from .agent_runtime import LocalAgentRuntime
from .composition import DesktopSession, EmbeddedProjectService
from .dual_catalog import DualCatalogService, build_workspace_service
from .matrix_source import KrakenMatrixAssetSource, KrakenMatrixDataSource
from .workspace_service import REMOTE_STORAGE_PROFILE, ProjectEventWake

from .manager_app_parts.construction import ConstructionMixin
from .manager_app_parts.poll_agent_jobs import PollAgentJobsMixin
from .manager_app_parts.gateway_job import GatewayJobMixin
from .manager_app_parts.manage_project_participants import ManageProjectParticipantsMixin
from .manager_app_parts.rename_project import RenameProjectMixin
from .manager_app_parts.delete_project import DeleteProjectMixin
from .manager_app_parts.connect_program_account import ConnectProgramAccountMixin
from .manager_app_parts.create_project import CreateProjectMixin
from .manager_app_parts.open_project import OpenProjectMixin
from .manager_app_parts.configure_analysis import ConfigureAnalysisMixin
from .manager_app_parts.analysis_frames import AnalysisFramesMixin
from .manager_app_parts.start_analysis import StartAnalysisMixin
from .manager_app_parts.send_selection_review import SendSelectionReviewMixin
from .manager_app_parts.load_review_return import LoadReviewReturnMixin
from .manager_app_parts.request_review_changes import RequestReviewChangesMixin
from .manager_app_parts.active_frame_files import ActiveFrameFilesMixin
from .manager_app_parts.show_frame_file_properties import ShowFrameFilePropertiesMixin
from .manager_app_parts.export_selected_frame_files import ExportSelectedFrameFilesMixin
from .manager_app_parts.show_matrix_context_menu import ShowMatrixContextMenuMixin
from .manager_app_parts.show_selected_frame import ShowSelectedFrameMixin
from .manager_app_parts.open_layer_manager import OpenLayerManagerMixin
from .manager_app_parts.layer_manager_activate import LayerManagerActivateMixin
from .manager_app_parts.refresh_layer_pipeline import RefreshLayerPipelineMixin
from .manager_app_parts.pipeline_snapshot import PipelineSnapshotMixin
from .manager_app_parts.open_properties import OpenPropertiesMixin
from .manager_app_parts.scope_content import ScopeContentMixin
from .manager_app_parts.show_project_properties import ShowProjectPropertiesMixin
from .manager_app_parts.show_layer_properties import ShowLayerPropertiesMixin
from .manager_app_parts.representation_file_count import RepresentationFileCountMixin
from .manager_app_parts.show_node_properties import ShowNodePropertiesMixin
from .manager_app_parts.layer_manager_action import LayerManagerActionMixin
from .manager_app_parts.layer_manager_layer_action import LayerManagerLayerActionMixin
from .manager_app_parts.ensure_agent_source_ready import EnsureAgentSourceReadyMixin
from .manager_app_parts.submit_agent_action import SubmitAgentActionMixin
from .manager_app_parts.run_remote_interactive_job import RunRemoteInteractiveJobMixin
from .manager_app_parts.contour_launch_arguments import ContourLaunchArgumentsMixin
from .manager_app_parts.neural_train_launch_arguments import NeuralTrainLaunchArgumentsMixin
from .manager_app_parts.neural_recognition_launch_arguments import NeuralRecognitionLaunchArgumentsMixin
from .manager_app_parts.launch_managed_plugin import LaunchManagedPluginMixin
from .manager_app_parts.monitor_workspace_result import MonitorWorkspaceResultMixin
from .manager_app_parts.monitor_agent_result import MonitorAgentResultMixin
from .manager_app_parts.launch_karakal import LaunchKarakalMixin
from .manager_app_parts.load_representations import LoadRepresentationsMixin
from .manager_app_parts.activate_representation import ActivateRepresentationMixin
from .manager_app_parts.refresh_matrix import RefreshMatrixMixin
from .manager_app_parts.archive_layer_representation import ArchiveLayerRepresentationMixin
from .manager_app_parts.start_representation_import import StartRepresentationImportMixin
from .manager_app_parts.add_representation import AddRepresentationMixin
from .manager_app_parts.add_layer import AddLayerMixin
from .manager_app_parts.error import ErrorMixin
from .manager_app_parts.event_matches_node import (
    _RemoteSignalBridge,
    _plain_value,
    _is_network_path,
    _confirm_server_external_path,
    _event_payload,
    _nested_mapping,
    _event_belongs_to_layer,
    _node_identifiers,
    _event_matches_node,
    _history_entries,
    _creator,
    _count_regular_files,
)
from .manager_app_parts.layer_create_thread import (
    _LayerCreateThread,
    _BackgroundCallThread,
    _volume_key,
)
from .manager_app_parts.configure_workspace_roots import _configure_workspace_roots
from .manager_app_parts.karakal_runtime_identity import (
    _installed_version,
    _karakal_runtime_identity,
)
from .manager_app_parts.representation_dialog import RepresentationDialog
from .manager_app_parts.development_session import (
    _remember_credentials,
    _autofill_credentials,
    _development_session,
)
from .manager_app_parts.login import _login
from .manager_app_parts.deletion_reason_dialog import (
    _deletion_reason_dialog,
    _plugin_panel,
)
from .manager_app_parts.performer_panel import _performer_panel
from .manager_app_parts.my_work_panel import _my_work_panel
from .manager_app_parts.statistics_panel import _statistics_panel
from .manager_app_parts.connect_saved_server import (
    _configure_administration_page,
    _server_project_hint,
    _sign_in_program_account,
    _connect_saved_server,
)
from .manager_app_parts.administration_panel import _administration_panel
from .manager_app_parts.run_manager_gui import run_manager_gui

class DesktopController(
    ConstructionMixin,
    PollAgentJobsMixin,
    GatewayJobMixin,
    ManageProjectParticipantsMixin,
    RenameProjectMixin,
    DeleteProjectMixin,
    ConnectProgramAccountMixin,
    CreateProjectMixin,
    OpenProjectMixin,
    ConfigureAnalysisMixin,
    AnalysisFramesMixin,
    StartAnalysisMixin,
    SendSelectionReviewMixin,
    LoadReviewReturnMixin,
    RequestReviewChangesMixin,
    ActiveFrameFilesMixin,
    ShowFrameFilePropertiesMixin,
    ExportSelectedFrameFilesMixin,
    ShowMatrixContextMenuMixin,
    ShowSelectedFrameMixin,
    OpenLayerManagerMixin,
    LayerManagerActivateMixin,
    RefreshLayerPipelineMixin,
    PipelineSnapshotMixin,
    OpenPropertiesMixin,
    ScopeContentMixin,
    ShowProjectPropertiesMixin,
    ShowLayerPropertiesMixin,
    RepresentationFileCountMixin,
    ShowNodePropertiesMixin,
    LayerManagerActionMixin,
    LayerManagerLayerActionMixin,
    EnsureAgentSourceReadyMixin,
    SubmitAgentActionMixin,
    RunRemoteInteractiveJobMixin,
    ContourLaunchArgumentsMixin,
    NeuralTrainLaunchArgumentsMixin,
    NeuralRecognitionLaunchArgumentsMixin,
    LaunchManagedPluginMixin,
    MonitorWorkspaceResultMixin,
    MonitorAgentResultMixin,
    LaunchKarakalMixin,
    LoadRepresentationsMixin,
    ActivateRepresentationMixin,
    RefreshMatrixMixin,
    ArchiveLayerRepresentationMixin,
    StartRepresentationImportMixin,
    AddRepresentationMixin,
    AddLayerMixin,
    ErrorMixin,
):
    pass

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _RemoteSignalBridge,
    _plain_value,
    _is_network_path,
    _confirm_server_external_path,
    _event_payload,
    _nested_mapping,
    _event_belongs_to_layer,
    _node_identifiers,
    _event_matches_node,
    _history_entries,
    _creator,
    _count_regular_files,
    _LayerCreateThread,
    _BackgroundCallThread,
    _volume_key,
    _configure_workspace_roots,
    _installed_version,
    _karakal_runtime_identity,
    RepresentationDialog,
    _remember_credentials,
    _autofill_credentials,
    _development_session,
    _login,
    _deletion_reason_dialog,
    _plugin_panel,
    _performer_panel,
    _my_work_panel,
    _statistics_panel,
    _configure_administration_page,
    _server_project_hint,
    _sign_in_program_account,
    _connect_saved_server,
    _administration_panel,
    run_manager_gui,
    AgentPluginGateway,
    FilesystemAnalysisStore,
    LocalAgentRuntime,
)

__all__ = ["DesktopController", "run_manager_gui"]
