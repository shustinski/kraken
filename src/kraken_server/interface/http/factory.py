"""Compose a FastAPI app. Route bodies live in resource modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kraken_server.interface.http.context import build_http_context
from kraken_server.interface.http.routes.agent import register as register_agent
from kraken_server.interface.http.routes.analysis import register_analysis_routes
from kraken_server.interface.http.routes.artifacts import register as register_artifacts
from kraken_server.interface.http.routes.auth import register as register_auth
from kraken_server.interface.http.routes.electrical import register_electrical_routes
from kraken_server.interface.http.routes.jobs import register as register_jobs
from kraken_server.interface.http.routes.layers import register as register_layers
from kraken_server.interface.http.routes.multilayer import register_multilayer_routes
from kraken_server.interface.http.routes.notes import register as register_notes
from kraken_server.interface.http.routes.pipeline import register_pipeline_routes
from kraken_server.interface.http.routes.project_reads import register as register_project_reads
from kraken_server.interface.http.routes.projects import register as register_projects
from kraken_server.interface.http.routes.quality import register_quality_routes
from kraken_server.interface.http.routes.realtime import register as register_realtime
from kraken_server.interface.http.routes.representations import register as register_representations
from kraken_server.interface.http.routes.reviews import register as register_reviews
from kraken_server.interface.http.routes.session import register as register_session
from kraken_server.interface.http.routes.work_orders import register as register_work_orders
from kraken_server.interface.http.routes.work_planning import register_work_planning_routes
from kraken_server.interface.session import SessionPrincipal
from kraken_server.services import ServerServices


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
    context = build_http_context(
        services,
        account_store,
        session_resolver,
        live_gitlab_verifier,
        project_access_mode,
        development,
        connection_hub,
        outbox_publisher,
        agent_token_store,
        agent_gateway,
        blob_gateway,
    )
    register_session(context)
    register_auth(context)
    register_projects(context)
    register_layers(context)
    register_representations(context)
    register_project_reads(context)
    register_artifacts(context)
    register_notes(context)
    register_reviews(context)
    register_jobs(context)
    register_work_orders(context)
    register_agent(context)
    register_realtime(context)
    register_work_planning_routes(
        context.app,
        context.backend,
        context.project_reader,
        context.shared_mutation_actor,
        context.command_context,
    )
    register_quality_routes(
        context.app,
        context.backend,
        context.project_reader,
        context.shared_mutation_actor,
        context.command_context,
    )
    register_analysis_routes(
        context.app,
        context.backend,
        context.project_reader,
        context.shared_mutation_actor,
        context.command_context,
    )
    register_pipeline_routes(
        context.app,
        context.backend,
        context.project_reader,
        context.shared_mutation_actor,
        context.command_context,
    )
    register_multilayer_routes(
        context.app,
        context.backend,
        context.project_reader,
        context.shared_mutation_actor,
        context.command_context,
    )
    register_electrical_routes(
        context.app,
        context.backend,
        context.project_reader,
        context.shared_mutation_actor,
        context.command_context,
    )
    return context.app
