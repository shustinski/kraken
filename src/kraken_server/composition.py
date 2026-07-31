"""Built-in production composition for PostgreSQL + filesystem blobs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from kraken_manager.application.performers import ensure_gitlab_performer
from kraken_manager.domain.common import PrincipalId
from kraken_manager.domain.identity import Principal
from kraken_manager.infrastructure.auth.gitlab import (
    GitLabAuthenticationError,
    GitLabOidcClient,
    GitLabUnavailable,
)
from kraken_manager.infrastructure.auth.local import Argon2PasswordHasher
from kraken_manager.infrastructure.blob import FilesystemBlobStore
from kraken_manager.infrastructure.postgres import (
    PostgresFederatedSessionCache,
    PostgresAccountStore,
    PostgresIdentityAclStore,
    PostgresPerformerStore,
    PostgresUnitOfWorkFactory,
)

from .app import SessionPrincipal
from .persistent_services import PostgresServerServices, ServerStorageProfiles


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for the PostgreSQL server composition")
    return value


class HybridSessionResolver:
    """Resolve server-local opaque sessions and GitLab access tokens.

    GitLab userinfo is deliberately consulted for every resolution.  This is
    fail-closed and also creates/refreshes the stable ``issuer + sub`` Kraken
    principal on first use.
    """

    def __init__(
        self,
        accounts: Any,
        identities: PostgresIdentityAclStore,
        oidc: GitLabOidcClient | None,
        federated_cache: PostgresFederatedSessionCache | None = None,
        performers: PostgresPerformerStore | None = None,
    ) -> None:
        self.accounts = accounts
        self.identities = identities
        self.oidc = oidc
        self.federated_cache = federated_cache
        self.performers = performers

    @staticmethod
    def _gitlab_principal_id(issuer: str, subject: str) -> PrincipalId:
        return PrincipalId(str(uuid5(NAMESPACE_URL, f"kraken:gitlab:{issuer}:{subject}")))

    def __call__(self, token: str) -> SessionPrincipal | None:
        if not token:
            return None
        local = self.accounts.resolve_session(token)
        if local is not None:
            return SessionPrincipal(local.account_id, "local", token)
        if self.oidc is None:
            return None
        try:
            identity = self.oidc.userinfo(token)
        except GitLabAuthenticationError:
            if self.federated_cache is not None:
                self.federated_cache.revoke(token)
            return None
        except GitLabUnavailable:
            cached = None if self.federated_cache is None else self.federated_cache.resolve(token)
            if cached is None:
                return None
            principal_id, provider = cached
            principal = self.identities.get(principal_id)
            if principal is None or not principal.active:
                return None
            return SessionPrincipal(str(principal_id), provider, token)
        principal = Principal.gitlab(
            principal_id=self._gitlab_principal_id(identity.issuer, identity.subject),
            issuer=identity.issuer,
            subject=identity.subject,
            display_name=identity.name,
            email=identity.email,
        )
        self.identities.save(principal)
        if self.performers is not None:
            ensure_gitlab_performer(principal, self.performers)
        if self.federated_cache is not None:
            self.federated_cache.save(token, principal.id)
        return SessionPrincipal(str(principal.id), "gitlab", token)

    def verify_live(self, session: SessionPrincipal) -> bool:
        if session.provider != "gitlab" or self.oidc is None:
            return False
        try:
            identity = self.oidc.userinfo(session.access_token)
        except (GitLabAuthenticationError, GitLabUnavailable):
            return False
        expected = self._gitlab_principal_id(identity.issuer, identity.subject)
        return str(expected) == session.principal_id


def postgresql_composition() -> dict[str, Any]:
    """Return production keyword arguments for the FastAPI factory."""
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install Kraken with the 'postgres' extra") from exc

    database_url = _required("KRAKEN_DATABASE_URL")
    blob_root = Path(_required("KRAKEN_BLOB_ROOT"))
    max_frames = int(os.environ.get("KRAKEN_SERVER_MAX_FRAMES", "1000000"))
    if max_frames < 1:
        raise RuntimeError("KRAKEN_SERVER_MAX_FRAMES must be positive")

    engine = create_engine(database_url, pool_pre_ping=True)
    blobs = FilesystemBlobStore(blob_root)
    profiles = ServerStorageProfiles(max_frames=max_frames)
    uow_factory = PostgresUnitOfWorkFactory(engine, blobs)
    services = PostgresServerServices(engine, uow_factory, profiles=profiles)

    accounts = PostgresAccountStore(engine, Argon2PasswordHasher())
    identities = PostgresIdentityAclStore(engine)
    performers = PostgresPerformerStore(engine)
    issuer = os.environ.get("KRAKEN_GITLAB_ISSUER", "").strip()
    oidc = None
    if issuer:
        oidc = GitLabOidcClient(
            issuer,
            ca_file=os.environ.get("KRAKEN_GITLAB_CA_FILE") or None,
            timeout=float(os.environ.get("KRAKEN_GITLAB_TIMEOUT", "5")),
        )
    federated_cache = PostgresFederatedSessionCache(engine)
    resolver = HybridSessionResolver(accounts, identities, oidc, federated_cache, performers)
    from .outbox import ConnectionHub, OutboxPublisher

    hub = ConnectionHub()
    publisher = OutboxPublisher(engine, hub)
    return {
        "services": services,
        "account_store": accounts,
        "session_resolver": resolver,
        "live_gitlab_verifier": resolver.verify_live,
        "connection_hub": hub,
        "outbox_publisher": publisher,
    }


__all__ = ["HybridSessionResolver", "postgresql_composition"]
