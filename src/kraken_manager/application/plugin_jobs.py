"""Durable plugin-job submission and safe immutable result import."""

from __future__ import annotations
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from uuid import UUID, uuid5
from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import (
    CancelPluginJobCommand,
    ImportPluginResultCommand,
    PluginResultImport,
    RetryPluginJobCommand,
    SubmitPluginJobCommand,
    SynchronizePluginJobCommand,
)
from kraken_manager.application.errors import AuthorizationError, ConflictError, NotFoundError
from kraken_manager.application.ports import (
    Clock,
    PluginJobGateway,
    PluginResultContentReader,
    StorageProfileCatalog,
    UnitOfWork,
    UnitOfWorkFactory,
)
from kraken_manager.domain.artifacts import (
    ArtifactScope,
    ArtifactSeries,
    ArtifactVersion,
    deterministic_frame_series_id,
)
from kraken_manager.domain.common import ArtifactSeriesId, ArtifactVersionId, FrameId, PluginJobId
from kraken_manager.domain.events import ActorSnapshot, EventEnvelope, ProgramSnapshot
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import ProjectState, RepresentationKind, StructureState
from kraken_manager.domain.selection import FrameSelectionV1
from kraken_manager.domain.workflows import (
    PluginFrameOutcome,
    PluginInputV1,
    PluginJob,
    PluginJobManifestV1,
    PluginJobState,
    PluginResultManifestV1,
    PluginResultOutcome,
)

from .plugin_jobs_parts.save_job_event import SaveJobEventMixin
from .plugin_jobs_parts.call import CallMixin
from .plugin_jobs_parts.plain import (
    _job_stream,
    _series_stream,
    _plain,
)
from .plugin_jobs_parts.manifest_payload import (
    _manifest_payload,
    _manifest_from_payload,
)
from .plugin_jobs_parts.result_payload import (
    _job_payload,
    _canonical_fingerprint,
    _submission_fingerprint,
    _result_payload,
    _phase_key,
    _move_job,
)
from .plugin_jobs_parts.submit_plugin_job_handler import SubmitPluginJobHandler
from .plugin_jobs_parts.cancel_plugin_job_handler import CancelPluginJobHandler
from .plugin_jobs_parts.retry_plugin_job_handler import RetryPluginJobHandler
from .plugin_jobs_parts.synchronize_plugin_job_handler import SynchronizePluginJobHandler

class ImportPluginResultHandler(SaveJobEventMixin, CallMixin):
    """Validate an Agent result and commit immutable artifact versions atomically."""

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _job_stream,
    _series_stream,
    _plain,
    _manifest_payload,
    _manifest_from_payload,
    _job_payload,
    _canonical_fingerprint,
    _submission_fingerprint,
    _result_payload,
    _phase_key,
    _move_job,
    SubmitPluginJobHandler,
    CancelPluginJobHandler,
    RetryPluginJobHandler,
    SynchronizePluginJobHandler,
)

__all__ = [
    "CancelPluginJobHandler",
    "ImportPluginResultHandler",
    "RetryPluginJobHandler",
    "SubmitPluginJobHandler",
    "SynchronizePluginJobHandler",
]
