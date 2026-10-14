"""FastAPI application factory; all adapter selection happens here."""

import asyncio
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import anyio
from .services import (
    CommandContext,
    ConflictError,
    ForbiddenError,
    InMemoryServerServices,
    NotFoundError,
    ServerServices,
    ValidationError,
)

from .app_parts.module_state import API_PREFIX
from .app_parts.session_principal import SessionPrincipal
from .app_parts.create_app import create_app

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    API_PREFIX,
    SessionPrincipal,
    create_app,
)

__all__ = ["API_PREFIX", "SessionPrincipal", "create_app"]
