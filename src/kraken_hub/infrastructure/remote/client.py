"""HTTP/WebSocket client for shared projects hosted by kraken-server."""

from __future__ import annotations
import hashlib
import json
import logging
import mimetypes
import os
import shutil
import stat
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4
from kraken_agent.runner import PluginRegistry
from kraken_manager.application.dto import ReviewReturnCommitResult, ReviewReturnPlan
from kraken_manager.application.ports import StorageProfile
from kraken_manager.domain.artifacts import (
    ArtifactScope,
    ArtifactSeries,
    ArtifactVersion,
    NoteRevision,
)
from kraken_manager.domain.common import ProjectId, validate_uuid
from kraken_manager.domain.identity import (
    Performer,
    Permission,
    Principal,
    PrincipalProvider,
    ProjectRole,
    SystemRole,
    parse_project_role,
    permissions_for_roles,
)
from kraken_manager.domain.project import (
    GridOrientation,
    Layer,
    LayerType,
    Project,
    Representation,
    RepresentationKind,
    RepresentationPurpose,
    StructureState,
)
from kraken_manager.domain.workflows import (
    PluginJob,
    ReviewBatch,
)
from kraken_manager.infrastructure.filesystem._codec import decode_model
from kraken_manager.infrastructure.reports import (
    ReportMetrics,
    ReportSeries,
)
from kraken_manager.infrastructure.review.manifest import manifest_from_json
from kraken_hub.application.workspace import (
    REMOTE_STORAGE_PROFILE,
    ProjectEventWake,
    project_from_server_dict,
)

from .parts.module_state import LOGGER
from .parts.iter_bytes import IterBytesMixin
from .parts.request import RequestMixin
from .parts.download import DownloadMixin
from .parts.register_local import RegisterLocalMixin
from .parts.start_sync import StartSyncMixin
from .parts.call import CallMixin
from .parts.list_layers import ListLayersMixin
from .parts.representation import RepresentationMixin
from .parts.create_representation import CreateRepresentationMixin
from .parts.change_project_role import ChangeProjectRoleMixin
from .parts.create_artifact_series import CreateArtifactSeriesMixin
from .parts.add_managed_artifact_version import AddManagedArtifactVersionMixin
from .parts.add_external_artifact_version import AddExternalArtifactVersionMixin
from .parts.create_review_batch import CreateReviewBatchMixin
from .parts.issue_work import IssueWorkMixin
from .parts.work_planning import WorkPlanningMixin
from .parts.quality import QualityMixin
from .parts.pipeline_execution import PipelineExecutionMixin
from .parts.multilayer import MultilayerMixin
from .parts.electrical import ElectricalMixin
from .parts.export_review_batch import ExportReviewBatchMixin
from .parts.submit_plugin_job import SubmitPluginJobMixin
from .parts.remove_layer_pipeline_action import RemoveLayerPipelineActionMixin
from .parts.frame_cells import FrameCellsMixin
from .parts.extract_review_archive import (
    _layer_file_binding,
    _workspace_layer_result,
    _review_archive,
    _extract_review_archive,
    RemoteServerError,
    RemoteAuth,
)
from .parts.remote_sync_client import RemoteSyncClient
from .parts.principal_server_payload import _principal_from_server_payload
from .parts.remote_history_event import _RemoteHistoryEvent
from .parts.analysis_runs import ServerAnalysisMixin
from .parts.load_remote_auth_env import (
    load_remote_auth_from_env,
    build_remote_service,
)

class RemoteHttpClient(IterBytesMixin, RequestMixin, DownloadMixin):
    """Minimal stdlib HTTP client for `/api/v1`."""

class RemoteServerProjectService(
    ServerAnalysisMixin,
    RegisterLocalMixin,
    StartSyncMixin,
    CallMixin,
    ListLayersMixin,
    RepresentationMixin,
    CreateRepresentationMixin,
    ChangeProjectRoleMixin,
    CreateArtifactSeriesMixin,
    AddManagedArtifactVersionMixin,
    AddExternalArtifactVersionMixin,
    CreateReviewBatchMixin,
    IssueWorkMixin,
    WorkPlanningMixin,
    QualityMixin,
    PipelineExecutionMixin,
    MultilayerMixin,
    ElectricalMixin,
    ExportReviewBatchMixin,
    SubmitPluginJobMixin,
    RemoveLayerPipelineActionMixin,
    FrameCellsMixin,
):
    """Shared/PostgreSQL projects via kraken-server REST (+ optional WS sync)."""

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    LOGGER,
    _layer_file_binding,
    _workspace_layer_result,
    _review_archive,
    _extract_review_archive,
    RemoteServerError,
    RemoteAuth,
    RemoteSyncClient,
    _principal_from_server_payload,
    _RemoteHistoryEvent,
    load_remote_auth_from_env,
    build_remote_service,
)

__all__ = [
    "RemoteAuth",
    "RemoteHttpClient",
    "RemoteServerError",
    "RemoteServerProjectService",
    "RemoteSyncClient",
    "build_remote_service",
    "load_remote_auth_from_env",
]
