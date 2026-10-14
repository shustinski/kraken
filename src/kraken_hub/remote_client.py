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
from .workspace_service import (
    REMOTE_STORAGE_PROFILE,
    ProjectEventWake,
    project_from_server_dict,
)

from .remote_client_parts.module_state import LOGGER
from .remote_client_parts.iter_bytes import IterBytesMixin
from .remote_client_parts.request import RequestMixin
from .remote_client_parts.download import DownloadMixin
from .remote_client_parts.register_local import RegisterLocalMixin
from .remote_client_parts.start_sync import StartSyncMixin
from .remote_client_parts.call import CallMixin
from .remote_client_parts.list_layers import ListLayersMixin
from .remote_client_parts.representation import RepresentationMixin
from .remote_client_parts.create_representation import CreateRepresentationMixin
from .remote_client_parts.change_project_role import ChangeProjectRoleMixin
from .remote_client_parts.create_artifact_series import CreateArtifactSeriesMixin
from .remote_client_parts.add_managed_artifact_version import AddManagedArtifactVersionMixin
from .remote_client_parts.add_external_artifact_version import AddExternalArtifactVersionMixin
from .remote_client_parts.create_review_batch import CreateReviewBatchMixin
from .remote_client_parts.export_review_batch import ExportReviewBatchMixin
from .remote_client_parts.submit_plugin_job import SubmitPluginJobMixin
from .remote_client_parts.remove_layer_pipeline_action import RemoveLayerPipelineActionMixin
from .remote_client_parts.frame_cells import FrameCellsMixin
from .remote_client_parts.extract_review_archive import (
    _layer_file_binding,
    _workspace_layer_result,
    _review_archive,
    _extract_review_archive,
    RemoteServerError,
    RemoteAuth,
)
from .remote_client_parts.remote_sync_client import RemoteSyncClient
from .remote_client_parts.principal_server_payload import _principal_from_server_payload
from .remote_client_parts.remote_history_event import _RemoteHistoryEvent
from .remote_client_parts.load_remote_auth_env import (
    load_remote_auth_from_env,
    build_remote_service,
)

class RemoteHttpClient(IterBytesMixin, RequestMixin, DownloadMixin):
    """Minimal stdlib HTTP client for `/api/v1`."""

class RemoteServerProjectService(
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
