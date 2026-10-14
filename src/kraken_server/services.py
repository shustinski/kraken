"""Transport-facing server service contract.

FastAPI depends on this narrow facade.  Production composition maps it to
application use cases; the in-memory implementation is intentionally only a
development/test profile.
"""

from __future__ import annotations
import base64
import json
import tempfile
import threading
from pathlib import Path
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4
from kraken_manager.domain.common import LayerId, PrincipalId, ProjectId, RepresentationId
from kraken_manager.domain.selection import FrameRowRange, FrameSelectionV1
from kraken_manager.domain.workflows import PluginJob, PluginJobState
from kraken_manager.infrastructure.filesystem._codec import encode_model

from .services_parts.assign_project_role import AssignProjectRoleMixin
from .services_parts.project_lifecycle import ProjectLifecycleMixin
from .services_parts.layer_lifecycle import LayerLifecycleMixin
from .services_parts.change_project_role import ChangeProjectRoleMixin
from .services_parts.list_representations import ListRepresentationsMixin
from .services_parts.create_representation import CreateRepresentationMixin
from .services_parts.update_representation import UpdateRepresentationMixin
from .services_parts.submit_plugin_job import SubmitPluginJobMixin
from .services_parts.command_context import (
    ConflictError,
    NotFoundError,
    ValidationError,
    ForbiddenError,
    CommandContext,
)

class ServerServices(AssignProjectRoleMixin, Protocol):
    pass

class InMemoryServerServices(
    ProjectLifecycleMixin,
    LayerLifecycleMixin,
    ChangeProjectRoleMixin,
    ListRepresentationsMixin,
    CreateRepresentationMixin,
    UpdateRepresentationMixin,
    SubmitPluginJobMixin,
):
    """Sparse, concurrency-safe development backend used by API tests."""

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    ConflictError,
    NotFoundError,
    ValidationError,
    ForbiddenError,
    CommandContext,
)

__all__ = [
    "CommandContext",
    "ConflictError",
    "ForbiddenError",
    "InMemoryServerServices",
    "NotFoundError",
    "ServerServices",
    "ValidationError",
]
