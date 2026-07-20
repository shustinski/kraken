"""Kraken-owned authorization policy, independent from GitLab group roles."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from kraken_manager.application.dto import StorageScope
from kraken_manager.application.errors import AuthorizationError
from kraken_manager.application.ports import StorageProfile
from kraken_manager.domain.identity import (
    Permission,
    Principal,
    PrincipalProvider,
    ProjectRole,
    permissions_for_roles,
)


_IMPLICIT_READ_PERMISSIONS = frozenset(
    {Permission.VIEW_PROJECT, Permission.VIEW_HISTORY, Permission.EXPORT_STATISTICS}
)


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    allowed: bool
    code: str
    reason: str

    def require(self) -> None:
        if not self.allowed:
            error = AuthorizationError(self.reason)
            error.code = self.code
            raise error


class AuthorizationPolicy:
    """Apply global visibility, storage-provider, live-identity, and ACL rules.

    GitLab supplies identity only. Project roles passed here must come from the
    Kraken ``AclStore``. A local principal is hard-denied for every mutation of
    a shared project even if a corrupted ACL grants it ``OWNER``.
    """

    def decide(
        self,
        *,
        principal: Principal,
        storage: StorageProfile,
        permission: Permission,
        roles: Iterable[ProjectRole] = (),
        gitlab_identity_verified: bool = False,
    ) -> AuthorizationDecision:
        if not principal.active:
            return AuthorizationDecision(False, "principal_inactive", "The account is inactive")
        if permission in _IMPLICIT_READ_PERMISSIONS:
            return AuthorizationDecision(True, "allowed", "Every authenticated account may read projects")
        if storage.scope is StorageScope.SHARED:
            if principal.provider is PrincipalProvider.LOCAL:
                return AuthorizationDecision(
                    False,
                    "local_shared_mutation_denied",
                    "A local account cannot modify a shared project",
                )
        if principal.provider is PrincipalProvider.GITLAB and not gitlab_identity_verified:
            return AuthorizationDecision(
                False,
                "gitlab_live_check_required",
                "GitLab identity must be verified immediately before every mutation",
            )
        granted = permissions_for_roles(set(roles))
        if permission not in granted:
            return AuthorizationDecision(
                False,
                "project_permission_missing",
                f"The account does not have the {permission.value!r} project permission",
            )
        return AuthorizationDecision(True, "allowed", "Permission granted by Kraken project ACL")

    def require(
        self,
        *,
        principal: Principal,
        storage: StorageProfile,
        permission: Permission,
        roles: Iterable[ProjectRole] = (),
        gitlab_identity_verified: bool = False,
    ) -> None:
        # A single entry point makes command handlers difficult to accidentally
        # implement as UI-only permission checks.
        self.decide(
            principal=principal,
            storage=storage,
            permission=permission,
            roles=roles,
            gitlab_identity_verified=gitlab_identity_verified,
        ).require()

    def decide_create_project(
        self,
        *,
        principal: Principal,
        storage: StorageProfile,
        gitlab_identity_verified: bool = False,
    ) -> AuthorizationDecision:
        if not principal.active:
            return AuthorizationDecision(False, "principal_inactive", "The account is inactive")
        if storage.scope is StorageScope.SHARED:
            if principal.provider is PrincipalProvider.LOCAL:
                return AuthorizationDecision(
                    False,
                    "local_shared_mutation_denied",
                    "A local account cannot create a shared project",
                )
        if principal.provider is PrincipalProvider.GITLAB and not gitlab_identity_verified:
            return AuthorizationDecision(
                False,
                "gitlab_live_check_required",
                "GitLab identity must be verified before creating a project",
            )
        return AuthorizationDecision(True, "allowed", "Authenticated account may create a project in this scope")


__all__ = ["AuthorizationDecision", "AuthorizationPolicy"]
