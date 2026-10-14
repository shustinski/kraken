"""app: create app."""
from __future__ import annotations
import asyncio
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import anyio
from kraken_server.services import (
    CommandContext,
    ConflictError,
    ForbiddenError,
    InMemoryServerServices,
    NotFoundError,
    ServerServices,
    ValidationError,
)

from kraken_server.app_parts.module_state import API_PREFIX

from kraken_server.app_parts.session_principal import SessionPrincipal

from .create_app_sections.section_1 import _create_app_section_1
from .create_app_sections.section_2 import _create_app_section_2
from .create_app_sections.section_3 import _create_app_section_3
from .create_app_sections.section_4 import _create_app_section_4
from .create_app_sections.section_5 import _create_app_section_5

def create_app(
    *,
    services: ServerServices | None = None,
    account_store: Any | None = None,
    session_resolver: Callable[[str], SessionPrincipal | None] | None = None,
    live_gitlab_verifier: Callable[[SessionPrincipal], bool] | None = None,
    project_access_mode: str | None = None,
    development: bool = False,
    connection_hub: Any | None = None,
    outbox_publisher: Any | None = None,
    agent_token_store: Any | None = None,
    agent_gateway: Any | None = None,
    blob_gateway: Any | None = None,
) -> Any:
    Body, Depends, HTTPException, Header, Query, Request, StreamingResponse, WebSocket, WebSocketDisconnect, access_mode, agent_identity, app, backend, command_context, desktop_identity, has_project_access, heavy_operations, hub, principal, problem, project_reader, run_in_threadpool, shared_mutation_actor, transfer_identity, transfer_payload = _create_app_section_1(services, account_store, session_resolver, live_gitlab_verifier, project_access_mode, development, connection_hub, outbox_publisher, agent_token_store, agent_gateway, blob_gateway)
    extract_review_archive, transfer_ticket_context = _create_app_section_2(account_store, development, blob_gateway, Body, Depends, HTTPException, Header, access_mode, app, backend, command_context, has_project_access, heavy_operations, principal, problem, project_reader, run_in_threadpool, shared_mutation_actor)
    _create_app_section_3(blob_gateway, Depends, HTTPException, Header, Query, Request, app, backend, command_context, heavy_operations, project_reader, run_in_threadpool, shared_mutation_actor, transfer_identity, transfer_payload, transfer_ticket_context)
    _create_app_section_4(agent_gateway, blob_gateway, Depends, HTTPException, Header, Request, StreamingResponse, agent_identity, app, backend, command_context, desktop_identity, extract_review_archive, heavy_operations, project_reader, run_in_threadpool, shared_mutation_actor, transfer_identity, transfer_payload, transfer_ticket_context)
    _create_app_section_5(account_store, session_resolver, development, agent_gateway, blob_gateway, Depends, HTTPException, Header, Request, StreamingResponse, WebSocket, WebSocketDisconnect, agent_identity, app, backend, has_project_access, hub, run_in_threadpool, transfer_identity)
    return app
