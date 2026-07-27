from __future__ import annotations

import unittest
from dataclasses import replace

from kraken_manager.application.authorization import AuthorizationPolicy
from kraken_manager.application.dto import StorageBackendKind, StorageScope
from kraken_manager.application.ports import StorageCapabilities, StorageProfile
from kraken_manager.domain.identity import (
    Permission,
    Principal,
    ProjectRole,
    SystemRole,
)


class AuthorizationMatrixTests(unittest.TestCase):
    SHARED = StorageProfile(
        id="shared",
        name="Shared",
        metadata_backend=StorageBackendKind.POSTGRESQL,
        blob_backend="filesystem",
        scope=StorageScope.SHARED,
        capabilities=StorageCapabilities(True, True, True, True, True, 1_000_000),
    )

    def test_local_principal_is_hard_denied_on_shared_mutation(self) -> None:
        decision = AuthorizationPolicy().decide(
            principal=Principal.local(subject="operator", display_name="Operator"),
            storage=self.SHARED,
            roles={ProjectRole.OWNER},
            permission=Permission.MANAGE_STRUCTURE,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual("local_shared_mutation_denied", decision.code)

    def test_gitlab_mutation_requires_live_check_and_role(self) -> None:
        principal = Principal.gitlab(
            issuer="https://gitlab.example",
            subject="42",
            display_name="Reviewer",
        )
        denied = AuthorizationPolicy().decide(
            principal=principal,
            storage=self.SHARED,
            roles={ProjectRole.OWNER},
            permission=Permission.MANAGE_STRUCTURE,
        )
        self.assertEqual("gitlab_live_check_required", denied.code)
        allowed = AuthorizationPolicy().decide(
            principal=principal,
            storage=self.SHARED,
            roles={ProjectRole.MANAGER},
            permission=Permission.MANAGE_STRUCTURE,
            gitlab_identity_verified=True,
        )
        self.assertTrue(allowed.allowed)

    def test_server_admin_has_no_implicit_content_permission(self) -> None:
        principal = replace(
            Principal.gitlab(issuer="https://gitlab.example", subject="1", display_name="Admin"),
            system_roles=frozenset({SystemRole.SERVER_ADMIN}),
        )
        decision = AuthorizationPolicy().decide(
            principal=principal,
            storage=self.SHARED,
            roles=set(),
            permission=Permission.MANAGE_STRUCTURE,
            gitlab_identity_verified=True,
        )
        self.assertFalse(decision.allowed)


if __name__ == "__main__":
    unittest.main()
