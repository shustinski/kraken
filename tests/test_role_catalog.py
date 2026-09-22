"""Role catalog: inclusion, grants, cascade, and server-side access resolution."""

from __future__ import annotations

import unittest

from kraken_manager.domain.identity import Permission, ProjectRole, permissions_for_roles
from kraken_manager.domain.roles import (
    ActionDefinition,
    CATALOG,
    cascade_lost_assignments,
    default_catalog,
)


class RoleCatalogTests(unittest.TestCase):
    def test_elementer_includes_corrector_but_cannot_appoint_one(self) -> None:
        self.assertIn(Permission.RETURN_REVIEW, permissions_for_roles({ProjectRole.ELEMENTER}))
        self.assertIn(Permission.VIEW_PROJECT, permissions_for_roles({ProjectRole.ELEMENTER}))
        self.assertFalse(CATALOG.can_grant({ProjectRole.ELEMENTER.value}, ProjectRole.CORRECTOR.value))
        self.assertTrue(CATALOG.can_grant({ProjectRole.MAINTAINER.value}, ProjectRole.CORRECTOR.value))
        self.assertTrue(CATALOG.can_grant({ProjectRole.MAINTAINER.value}, ProjectRole.MAINTAINER.value))
        self.assertFalse(CATALOG.can_grant({ProjectRole.MAINTAINER.value}, ProjectRole.ADMIN.value))
        self.assertTrue(CATALOG.can_grant(set(), ProjectRole.VIEWER.value, is_server_admin=True))

    def test_acting_role_is_resolved_against_the_action(self) -> None:
        denied = CATALOG.resolve(
            held_roles={"elementer"},
            acting_role="elementer",
            action="manage_structure",
        )
        self.assertFalse(denied.allowed)
        self.assertIn("Недостаточно прав", denied.message)
        allowed = CATALOG.resolve(
            held_roles={"elementer"},
            acting_role="elementer",
            action="return_review",
        )
        self.assertTrue(allowed.allowed)
        missing = CATALOG.resolve(
            held_roles={"viewer"},
            acting_role="maintainer",
            action="manage_structure",
        )
        self.assertFalse(missing.allowed)
        self.assertIn("не назначена", missing.message)

    def test_action_can_be_opened_to_every_role(self) -> None:
        catalog = default_catalog()
        catalog.register_action(ActionDefinition("export_statistics", "экспорт статистики", roles=None))
        decision = catalog.resolve(
            held_roles={"viewer"},
            acting_role="viewer",
            action="export_statistics",
        )
        self.assertTrue(decision.allowed)

    def test_revoking_maintainer_drops_roles_that_maintainer_assigned(self) -> None:
        assignments = (
            ("maintainer-a", "maintainer", "admin-user"),
            ("elementer-b", "elementer", "maintainer-a"),
            ("maintainer-c", "maintainer", "maintainer-a"),
            ("sewer-d", "sewer", "maintainer-c"),
            ("viewer-e", "viewer", "someone-else"),
        )
        lost = cascade_lost_assignments(
            assignments,
            principal_id="maintainer-a",
            revoked_role="maintainer",
            remaining_roles=set(),
            grantor_is_admin=False,
            catalog=CATALOG,
        )
        self.assertEqual(
            {("elementer-b", "elementer"), ("maintainer-c", "maintainer"), ("sewer-d", "sewer")},
            set(lost),
        )
        kept = cascade_lost_assignments(
            assignments,
            principal_id="maintainer-a",
            revoked_role="maintainer",
            remaining_roles={"admin"},
            grantor_is_admin=False,
            catalog=CATALOG,
        )
        self.assertEqual((), kept)
        no_grant = cascade_lost_assignments(
            assignments,
            principal_id="elementer-b",
            revoked_role="elementer",
            remaining_roles=set(),
            grantor_is_admin=False,
            catalog=CATALOG,
        )
        self.assertEqual((), no_grant)


if __name__ == "__main__":
    unittest.main()
