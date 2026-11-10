"""Built-in production composition for PostgreSQL + filesystem blobs."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from kraken_manager.application.performers import ensure_gitlab_performer, ensure_linked_performer
from kraken_manager.domain.common import PrincipalId
from kraken_manager.domain.identity import Principal, PrincipalProvider, SystemRole
from kraken_manager.infrastructure.auth.gitlab import (
    GitLabAuthenticationError,
    GitLabOidcClient,
    GitLabUnavailable,
)
from kraken_manager.infrastructure.auth.ldap import LdapUnavailable
from kraken_manager.infrastructure.auth.local import Argon2PasswordHasher, LocalSession
from kraken_manager.infrastructure.blob import FilesystemBlobStore
from kraken_manager.infrastructure.postgres import (
    PostgresAccountStore,
    PostgresFederatedSessionCache,
    PostgresIdentityAclStore,
    PostgresPerformerStore,
    PostgresUnitOfWorkFactory,
)
from kraken_manager.infrastructure.review.crypto import Ed25519KeyPair

from .app import SessionPrincipal
from .persistent_services import PostgresServerServices, ServerStorageProfiles

logger = logging.getLogger(__name__)


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for the PostgreSQL server composition")
    return value


def _review_key_pair(blob_root: Path) -> Ed25519KeyPair:
    key_dir = blob_root / ".server"
    key_dir.mkdir(parents=True, exist_ok=True)
    key_path = key_dir / "review-ed25519.key"
    try:
        raw = key_path.read_bytes()
    except FileNotFoundError:
        pair = Ed25519KeyPair.generate()
        raw = pair.private_key + pair.public_key
        try:
            descriptor = os.open(
                key_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            raw = key_path.read_bytes()
        else:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
    if len(raw) != 64:
        raise RuntimeError("Kraken review signing key file is invalid")
    return Ed25519KeyPair(raw[:32], raw[32:])


def _ldap_principal_id(username: str) -> PrincipalId:
    """Stable principal id for a domain login (the same id on every JIT and re-login)."""
    return PrincipalId(str(uuid5(NAMESPACE_URL, f"kraken:ldap:{username.casefold()}")))


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
        ldap_issuer: str | None = None,
    ) -> None:
        self.accounts = accounts
        self.identities = identities
        self.oidc = oidc
        self.federated_cache = federated_cache
        self.performers = performers
        self.ldap_issuer = ldap_issuer

    @staticmethod
    def _gitlab_principal_id(issuer: str, subject: str) -> PrincipalId:
        return PrincipalId(str(uuid5(NAMESPACE_URL, f"kraken:gitlab:{issuer}:{subject}")))

    def __call__(self, token: str) -> SessionPrincipal | None:
        if not token:
            return None
        local = self.accounts.resolve_session(token)
        if local is not None:
            if getattr(local, "auth_provider", "local") == "ldap":
                principal = self._resolve_ldap_account(local)
                if principal is None:
                    return None
                return SessionPrincipal(str(principal.id), "ldap", token)
            self.provision_local_account(local)
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

    def _resolve_ldap_account(self, account: Any) -> Principal | None:
        """Resolve the stable LDAP principal of an LDAP-backed account row.

        JIT and claim transfer always create the row, so a missing one is a
        data anomaly: it is repaired without AD only when the issuer is
        configured, otherwise the session resolves to nothing (401 on the
        account's first API call) and the error is logged.
        """
        principal = self.identities.get_by_external_key(f"ldap:{account.username.casefold()}")
        if principal is not None:
            return principal
        if self.ldap_issuer is None:
            logger.error(
                "LDAP account %s has no principal and no LDAP issuer is configured; "
                "the session resolves without a principal until the row is repaired",
                account.username,
            )
            return None
        principal_id = _ldap_principal_id(account.username)
        principal = Principal.ldap(
            principal_id=principal_id,
            issuer=self.ldap_issuer,
            subject=account.username,
            display_name=account.display_name,
            login=account.username,
            system_roles=frozenset(SystemRole(role) for role in self.accounts.global_roles_for(account.account_id)),
        )
        holder = self.identities.find_by_login(account.username.casefold())
        if holder is not None and str(holder.id) != str(principal_id):
            # Same claim safety net as DomainAuthBroker: no silent rename, no overwrite.
            raise LdapUnavailable("conflicting principals for one login")
        self.identities.save(principal)
        if self.performers is not None:
            ensure_linked_performer(principal, self.performers)
        return principal

    def provision_local_account(self, account: Any) -> None:
        """Mirror a server account into the identity catalog.

        Registration and later session use share this path. Role assignment
        reads the catalog, so the row has to exist before the account's first
        authenticated request.
        """

        login = account.username
        holder = self.identities.find_by_login(account.username.casefold())
        if holder is not None and str(holder.id) != str(account.account_id) and holder.provider is PrincipalProvider.LDAP:
            # The domain user already owns this login: the local principal keeps
            # working but does not claim it (no lockout, no IntegrityError).
            login = None
            self.accounts.record_audit(
                action="login_claim_conflict_local_skipped",
                target_account_id=account.account_id,
                details={"login": account.username, "holder": str(holder.id)},
            )
        principal = Principal(
            id=PrincipalId(account.account_id),
            provider=PrincipalProvider.LOCAL,
            subject=account.username,
            display_name=account.display_name,
            active=account.enabled,
            system_roles=frozenset(SystemRole(role) for role in self.accounts.global_roles_for(account.account_id)),
            login=login,
        )
        self.identities.save(principal)
        if self.performers is not None:
            ensure_linked_performer(principal, self.performers)

    def verify_live(self, session: SessionPrincipal) -> bool:
        if session.provider != "gitlab" or self.oidc is None:
            return False
        try:
            identity = self.oidc.userinfo(session.access_token)
        except (GitLabAuthenticationError, GitLabUnavailable):
            return False
        expected = self._gitlab_principal_id(identity.issuer, identity.subject)
        return str(expected) == session.principal_id


class DomainAuthBroker:
    """Direct domain (LDAP/AD) sign-in at login time only.

    Providers are strictly separate: an LDAP row is verified against AD (the
    stored hash is structurally unusable), and a successful local password
    login never touches AD. A failed local login falls through to AD so a
    domain user holding the same login is never locked out; the claim rules
    (spec section 4) then decide what happens to the row.
    """

    def __init__(self, accounts: Any, identities: Any, ldap: Any | None, *, performers: Any | None = None) -> None:
        self.accounts = accounts
        self.identities = identities
        self.ldap = ldap
        self.performers = performers

    def authenticate(self, username: str, password: str) -> LocalSession | None:
        try:
            account = self.accounts.get_by_username(username)
        except ValueError:
            return None
        if account is None:
            return self._authenticate_new_domain_user(username, password)
        if getattr(account, "auth_provider", "local") == "ldap":
            return self._authenticate_ldap_account(account, username, password)
        return self._authenticate_local_account(account, username, password)

    def _authenticate_ldap_account(self, account: Any, username: str, password: str) -> LocalSession | None:
        identity = self.ldap.verify(username, password)  # Ldap* errors propagate
        ok = identity is not None
        session = self.accounts.authenticate(username, password, verifier=lambda _stored, _pw: ok)
        if session is not None and identity is not None:
            self._provision_ldap(account, identity)  # refresh path, idempotent
        return session

    def _authenticate_local_account(self, account: Any, username: str, password: str) -> LocalSession | None:
        session = self.accounts.authenticate(username, password)
        if session is not None:
            return session  # a successful local login never touches AD
        if self.ldap is None:
            return None
        identity = self.ldap.verify(username, password)  # Ldap* errors propagate
        if identity is None:
            return None
        account = self._resolve_login_claim(account)
        session = self.accounts.issue_session(username)
        self._provision_ldap(account, identity)
        return session

    def _authenticate_new_domain_user(self, username: str, password: str) -> LocalSession | None:
        if self.ldap is None:
            return None
        identity = self.ldap.verify(username, password)  # Ldap* errors propagate
        if identity is None:
            return None
        try:
            from sqlalchemy.exc import IntegrityError
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install Kraken with the 'postgres' extra") from exc
        try:
            account = self.accounts.create_ldap_account(username, identity.display_name)
        except IntegrityError:
            # A concurrent first login won the JIT race; continue with its row.
            account = self.accounts.get_by_username(username)
        session = self.accounts.issue_session(username)
        self._provision_ldap(account, identity)
        return session

    def _resolve_login_claim(self, account: Any) -> Any:
        """Claim rules (spec section 4) when an AD-verified domain user holds a local login."""
        holder = self.identities.find_by_login(account.username.casefold())
        if holder is not None:
            if holder.provider is PrincipalProvider.LDAP:
                if str(holder.id) != str(_ldap_principal_id(account.username)):
                    # A *different* LDAP principal already owns this login: fail closed (503 upstream).
                    raise LdapUnavailable("conflicting LDAP principals for one login")
                return account  # the domain user's own claim: re-login on a shared row, refresh only
            if holder.provider is PrincipalProvider.LOCAL:
                return self._takeover_squatting_local_row(account, holder)
            return account  # any other holder is caught by the _provision_ldap safety net
        if not account.enabled:
            # A disabled local row is unusable by its owner; the domain user takes it.
            return self.accounts.convert_to_ldap_account(account.account_id)
        return account  # the local user never signed in: the row is shared, both keep working

    def _takeover_squatting_local_row(self, account: Any, holder: Any) -> Any:
        """The squatted local identity stops working; the domain user takes the row."""
        self.accounts.set_enabled(account.account_id, False)  # revokes the local user's sessions
        account = self.accounts.convert_to_ldap_account(account.account_id)
        self.accounts.record_audit(
            action="login_conflict_local_disabled",
            target_account_id=account.account_id,
            details={
                "login": account.username,
                "local_principal": str(holder.id),
                "ldap_principal": str(_ldap_principal_id(account.username)),
            },
        )
        self.identities.save(replace(holder, login=None))  # release the claim (login_key=NULL)
        return account

    def _provision_ldap(self, account: Any, identity: Any) -> None:
        """Upsert the stable LDAP principal (uuid5 of the domain login) and its performer."""
        principal_id = _ldap_principal_id(account.username)
        principal = Principal.ldap(
            principal_id=principal_id,
            issuer=self.ldap.issuer,
            subject=account.username,
            display_name=identity.display_name,
            email=identity.email,
            login=account.username,
            avatar_url=identity.avatar_url,
            system_roles=frozenset(SystemRole(role) for role in self.accounts.global_roles_for(account.account_id)),
        )
        holder = self.identities.find_by_login(account.username.casefold())
        if holder is not None and str(holder.id) != str(principal_id):
            # The DB unique index is the final net: no silent rename, no silent overwrite.
            raise LdapUnavailable("conflicting principals for one login")
        self.identities.save(principal)
        if identity.avatar_url is not None:
            self.identities.sync_avatar(principal_id, identity.avatar_url)
        if self.performers is not None:
            ensure_linked_performer(principal, self.performers)


def postgresql_composition() -> dict[str, Any]:
    """Return production keyword arguments for the FastAPI factory."""
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install Kraken with the 'postgres' extra") from exc

    database_url = _required("KRAKEN_DATABASE_URL")
    blob_root = Path(_required("KRAKEN_BLOB_ROOT"))
    source_root = _required("KRAKEN_SOURCE_ROOT")
    derived_root = _required("KRAKEN_DERIVED_ROOT")
    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=int(os.environ.get("KRAKEN_DB_POOL_SIZE", "10")),
        max_overflow=int(os.environ.get("KRAKEN_DB_MAX_OVERFLOW", "10")),
    )
    blobs = FilesystemBlobStore(blob_root)
    profiles = ServerStorageProfiles(max_frames=None)
    uow_factory = PostgresUnitOfWorkFactory(engine, blobs)
    from .agent_auth import PostgresAgentTokenStore
    from .agent_gateway import PostgresAgentGateway

    agent_tokens = PostgresAgentTokenStore(engine)
    agent_gateway = PostgresAgentGateway(engine, agent_tokens, blobs)
    performers = PostgresPerformerStore(engine)
    review_keys = _review_key_pair(blob_root)
    from kraken_manager.infrastructure.workspace_files import WorkspaceFileService, WorkspaceRegistry
    from kraken_manager.infrastructure.root_relocations import RootRelocations

    RootRelocations.load(blob_root, (blob_root, source_root, derived_root),
                         required=os.environ.get("KRAKEN_RECOVERY_ROOTS_REQUIRED") == "1")

    services = PostgresServerServices(
        engine,
        uow_factory,
        profiles=profiles,
        agent_gateway=agent_gateway,
        performer_store=performers,
        review_key_pair=review_keys,
        workspace_files=WorkspaceFileService(WorkspaceRegistry(blob_root / "workspaces")),
        source_root=source_root,
        derived_root=derived_root,
        recovery_roots_required=os.environ.get("KRAKEN_RECOVERY_ROOTS_REQUIRED") == "1",
    )

    accounts = PostgresAccountStore(engine, Argon2PasswordHasher())
    services.accounts = accounts
    identities = PostgresIdentityAclStore(engine)
    ldap_url = os.environ.get("KRAKEN_LDAP_URL", "").strip()
    domain_auth = None
    if ldap_url:
        from kraken_manager.infrastructure.auth.ldap import DEFAULT_USER_FILTER, LdapClient
        from kraken_server.configuration import unprotect_secret

        bind_secret_file = Path(os.environ.get("KRAKEN_LDAP_BIND_SECRET_FILE", ""))
        if not bind_secret_file.is_file():
            raise RuntimeError("KRAKEN_LDAP_BIND_SECRET_FILE is configured but the secret file is missing")
        ldap_client = LdapClient(
            ldap_url,
            base_dn=os.environ["KRAKEN_LDAP_BASE_DN"],
            bind_dn=os.environ["KRAKEN_LDAP_BIND_DN"],
            bind_password=unprotect_secret(bind_secret_file),
            user_filter=os.environ.get("KRAKEN_LDAP_USER_FILTER", "").strip() or DEFAULT_USER_FILTER,
            start_tls=os.environ.get("KRAKEN_LDAP_START_TLS") == "1",
            allow_insecure=os.environ.get("KRAKEN_LDAP_ALLOW_INSECURE") == "1",
            ca_file=os.environ.get("KRAKEN_LDAP_CA_FILE") or None,
            timeout=float(os.environ.get("KRAKEN_LDAP_TIMEOUT", "5")),
        )
        domain_auth = DomainAuthBroker(accounts, identities, ldap_client, performers=performers)
    lan_gate = None
    cidrs = json.loads(os.environ.get("KRAKEN_LAN_CIDRS", "[]"))
    if cidrs:
        from kraken_server.interface.http.ip_gate import is_lan_client

        def lan_gate(ip: str) -> bool:
            return is_lan_client(ip, cidrs)

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
    blob_gateway = None
    if public_url := os.environ.get("KRAKEN_BLOB_GATEWAY_PUBLIC_URL", "").strip():
        from .blob_gateway import BlobGatewayManager

        blob_gateway = BlobGatewayManager(
            public_url=public_url,
            bind=_required("KRAKEN_BLOB_GATEWAY_BIND"),
            blob_root=blob_root,
            executable=Path(_required("KRAKEN_BLOB_GATEWAY_EXECUTABLE")),
            secret=_required("KRAKEN_BLOB_GATEWAY_SECRET"),
            ticket_lifetime_seconds=int(os.environ.get("KRAKEN_BLOB_TICKET_LIFETIME", "900")),
            tls_cert_file=(Path(value) if (value := os.environ.get("KRAKEN_BLOB_GATEWAY_TLS_CERT")) else None),
            tls_key_file=(Path(value) if (value := os.environ.get("KRAKEN_BLOB_GATEWAY_TLS_KEY")) else None),
        )
        from .project_deletion import DeletionRequests

        blob_gateway.record_transfer = DeletionRequests(engine).record_transfer
    return {
        "services": services,
        "account_store": accounts,
        "session_resolver": resolver,
        "live_gitlab_verifier": resolver.verify_live,
        "project_access_mode": os.environ.get("KRAKEN_PROJECT_ACCESS_MODE", "acl"),
        "connection_hub": hub,
        "outbox_publisher": publisher,
        "blob_gateway": blob_gateway,
        "agent_token_store": agent_tokens,
        "agent_gateway": agent_gateway,
        "domain_auth": domain_auth,
        "lan_gate": lan_gate,
        "identities": identities,
    }


__all__ = ["DomainAuthBroker", "HybridSessionResolver", "postgresql_composition"]
