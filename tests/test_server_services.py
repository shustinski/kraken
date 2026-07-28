from __future__ import annotations

import unittest

from kraken_server.services import CommandContext, InMemoryServerServices, ValidationError


class ServerServiceTests(unittest.TestCase):
    def test_layer_and_representation_commands_are_optimistic_and_idempotent(self) -> None:
        service = InMemoryServerServices()
        project = service.create_project(
            {"name": "Chip", "width": 10, "height": 10}, CommandContext("actor", "project", None)
        )
        layer_context = CommandContext("actor", "layer", project["revision"])
        layer = service.create_layer(
            project["project_id"], {"name": "M1", "type": "metal", "order": 1}, layer_context
        )
        self.assertEqual(
            layer,
            service.create_layer(project["project_id"], {"name": "ignored", "type": "gate"}, layer_context),
        )
        representation = service.create_representation(
            project["project_id"],
            layer["layer_id"],
            {"name": "Raw", "kind": "image", "active": True},
            CommandContext("actor", "representation", layer["revision"]),
        )
        self.assertTrue(representation["active"])
        self.assertEqual([representation], service.list_representations(project["project_id"], layer["layer_id"]))
        second = service.create_representation(
            project["project_id"], layer["layer_id"],
            {"name": "Processed", "kind": "image", "active": False},
            CommandContext("actor", "representation-2", 1),
        )
        representation = service.update_representation(
            project["project_id"], layer["layer_id"], representation["representation_id"],
            {"note": "raw scan", "expected_representation_revision": 0},
            CommandContext("actor", "representation-note", 2),
        )
        second = service.update_representation(
            project["project_id"], layer["layer_id"], second["representation_id"],
            {"active": True, "expected_representation_revision": 0},
            CommandContext("actor", "representation-active", 3),
        )
        values = service.list_representations(project["project_id"], layer["layer_id"])
        self.assertEqual([False, True], [item["active"] for item in values])
        self.assertEqual("raw scan", representation["note"])

    def test_create_is_idempotent_and_grid_is_sparse(self) -> None:
        service = InMemoryServerServices()
        context = CommandContext("actor", "key", None)
        created = service.create_project(
            {"name": "Chip", "width": 1_000_000, "height": 10, "orientation": "y_up"}, context
        )
        repeated = service.create_project({"name": "ignored", "width": 1, "height": 1}, context)
        self.assertEqual(created, repeated)
        viewport = service.matrix_viewport(
            created["project_id"], layer_id="layer", x1=1, y1=1, x2=1000, y2=10, lod=8
        )
        self.assertEqual([], viewport["cells"])

    def test_viewport_validates_project_bounds(self) -> None:
        service = InMemoryServerServices()
        project = service.create_project(
            {"name": "Chip", "width": 10, "height": 10}, CommandContext("actor", "key", None)
        )
        with self.assertRaises(ValidationError):
            service.matrix_viewport(project["project_id"], layer_id="layer", x1=0, y1=1, x2=2, y2=2, lod=0)

    def test_project_and_layer_lifecycle_are_optimistic_idempotent_and_audited(self) -> None:
        service = InMemoryServerServices()
        project = service.create_project(
            {"name": "Chip", "width": 2, "height": 2}, CommandContext("actor", "create", None)
        )
        layer = service.create_layer(
            project["project_id"],
            {"name": "M1", "type": "metal"},
            CommandContext("actor", "layer", project["revision"]),
        )
        project = service.get_project(project["project_id"])
        renamed = service.rename_project(
            project["project_id"],
            "Chip 2",
            CommandContext("actor", "rename", project["revision"]),
        )
        self.assertEqual(
            renamed,
            service.rename_project(
                project["project_id"],
                "ignored",
                CommandContext("actor", "rename", project["revision"]),
            ),
        )
        layer = service.rename_layer(
            project["project_id"], layer["layer_id"], "Metal", CommandContext("actor", "lr", 0)
        )
        layer = service.reorder_layer(
            project["project_id"], layer["layer_id"], 4, CommandContext("actor", "lo", 1)
        )
        layer = service.archive_layer(
            project["project_id"], layer["layer_id"], CommandContext("actor", "la", 2)
        )
        archived = service.archive_project(
            project["project_id"], CommandContext("actor", "archive", renamed["revision"])
        )
        restored = service.restore_project(
            project["project_id"], CommandContext("actor", "restore", archived["revision"])
        )
        self.assertEqual(("Metal", 4, "archived"), (layer["name"], layer["order"], layer["state"]))
        self.assertEqual("active", restored["state"])
        acl = service.assign_project_role(
            project["project_id"],
            "worker",
            "contributor",
            CommandContext("actor", "acl-assign", 0),
        )
        self.assertEqual((["contributor"], 1), (acl["roles"], acl["revision"]))
        acl = service.revoke_project_role(
            project["project_id"],
            "worker",
            "contributor",
            CommandContext("actor", "acl-revoke", 1),
        )
        self.assertEqual(([], 2), (acl["roles"], acl["revision"]))
        event_types = [item["event_type"] for item in service.history(project["project_id"], cursor=None, limit=100)["items"]]
        self.assertIn("ProjectArchive", event_types)
        self.assertIn("LayerReorder", event_types)
        self.assertIn("ProjectRoleAssigned", event_types)

    def test_bulk_layer_order_requires_complete_revision_map(self) -> None:
        service = InMemoryServerServices()
        project = service.create_project(
            {"name": "Chip", "width": 2, "height": 2},
            CommandContext("actor", "project", None),
        )
        first = service.create_layer(
            project["project_id"],
            {"name": "M1", "type": "metal"},
            CommandContext("actor", "first", project["revision"]),
        )
        project = service.get_project(project["project_id"])
        second = service.create_layer(
            project["project_id"],
            {"name": "M2", "type": "metal"},
            CommandContext("actor", "second", project["revision"]),
        )

        result = service.reorder_layers(
            project["project_id"],
            {
                "layer_ids": [second["layer_id"], first["layer_id"]],
                "expected_revisions": {
                    first["layer_id"]: first["revision"],
                    second["layer_id"]: second["revision"],
                },
            },
            CommandContext("actor", "bulk-order", None),
        )

        self.assertEqual([item["layer_id"] for item in result], [second["layer_id"], first["layer_id"]])
        self.assertEqual([item["order"] for item in result], [0, 1])
        self.assertEqual(
            [item["layer_id"] for item in service.list_layers(project["project_id"])],
            [second["layer_id"], first["layer_id"]],
        )


if __name__ == "__main__":
    unittest.main()
