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


if __name__ == "__main__":
    unittest.main()
