from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from kraken_core.plugin_protocol import (
    PluginFrameInput,
    PluginJobManifest,
    safe_relative_path,
)
from kraken_core.plugins import load_plugin_catalog


class PluginProtocolTests(unittest.TestCase):
    def test_manifest_round_trip_is_canonical(self) -> None:
        digest = hashlib.sha256(b"input").hexdigest()
        manifest = PluginJobManifest(
            job_id="job-1",
            operation="frames.vectorize.v1",
            project_id="project-1",
            layer_id="layer-1",
            actor_id="actor-1",
            target_representation_id="representation-1",
            inputs=(PluginFrameInput("frame-1", 1, 2, "version-1", digest, "image/png", "inputs/1_2.png"),),
            parameters={"threshold": 0.5},
        )
        restored = PluginJobManifest.from_json(manifest.to_json())
        self.assertEqual(manifest.to_dict(), restored.to_dict())
        self.assertEqual(manifest.digest(), restored.digest())

    def test_transport_path_rejects_escape_and_drive(self) -> None:
        for invalid in ("../x", "/absolute", "C:/secret", "a/../../b", "a\\..\\b"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                safe_relative_path(invalid)

    def test_catalog_exposes_contour_and_neuralimage_capabilities(self) -> None:
        catalog = load_plugin_catalog("src/kraken_hub/resources/plugins.json")
        by_id = {item.id: item for item in catalog}
        self.assertEqual("frames.vectorize.v1", by_id["contour"].capabilities[0].operation)
        self.assertEqual("frames.binary-segment.v1", by_id["neuralimage"].capabilities[0].operation)
        json.loads(Path("src/kraken_hub/resources/plugins.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
