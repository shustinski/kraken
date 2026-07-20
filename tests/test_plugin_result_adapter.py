from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from kraken_agent.jobs import StagingWorkspace
from kraken_core.plugin_protocol import PluginFrameOutput, PluginResultManifest
from kraken_manager.infrastructure.plugin.result_reader import (
    AgentStagingResultContentReader,
    domain_result_from_transport,
)


class PluginResultAdapterTests(unittest.TestCase):
    def test_non_uuid_plugin_output_identity_is_normalized_and_content_is_streamed_safely(self) -> None:
        job_id = str(uuid4())
        frame_id = str(uuid4())
        payload = b"CIF bytes"
        transport = PluginResultManifest(
            job_id=job_id,
            outcome="succeeded",
            plugin_id="contour",
            plugin_version="1",
            outputs=(
                PluginFrameOutput(
                    output_id=f"{job_id}:contour:1",
                    frame_id=frame_id,
                    relative_path="outputs/1_1.cif",
                    sha256=hashlib.sha256(payload).hexdigest(),
                    media_type="application/x-cif",
                    role="vector",
                ),
            ),
        )
        converted = domain_result_from_transport(transport)
        self.assertEqual(converted, domain_result_from_transport(transport))
        self.assertNotEqual(transport.outputs[0].output_id, converted.results[0].output_id)
        with tempfile.TemporaryDirectory() as directory:
            workspace = StagingWorkspace(Path(directory), job_id)
            workspace.create()
            destination = workspace.resolve("outputs/1_1.cif")
            destination.write_bytes(payload)
            reader = AgentStagingResultContentReader(directory, chunk_size=3)
            self.assertEqual(payload, b"".join(reader.iter_output(converted, "outputs/1_1.cif")))


if __name__ == "__main__":
    unittest.main()
