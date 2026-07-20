from __future__ import annotations

import hashlib
import tempfile
import unittest
import json
import sys
from pathlib import Path

from kraken_agent.jobs import AgentJobState, DurableJobStore, JobStateError, StagingWorkspace
from kraken_agent.runner import PluginProcessSpec, PluginRegistry, SubprocessPluginRunner
from kraken_core.plugin_protocol import (
    PluginFrameInput,
    PluginFrameOutput,
    PluginJobManifest,
    PluginResultManifest,
)


def manifest() -> PluginJobManifest:
    return PluginJobManifest(
        job_id="job-1",
        operation="frames.vectorize.v1",
        project_id="project-1",
        layer_id="layer-1",
        actor_id="actor-1",
        target_representation_id="representation-1",
        inputs=(
            PluginFrameInput(
                "frame-1",
                1,
                1,
                "version-1",
                hashlib.sha256(b"content").hexdigest(),
                "image/png",
                "inputs/1_1.png",
            ),
        ),
    )


class AgentJobTests(unittest.TestCase):
    def test_runner_rejects_outputs_for_unknown_frames(self) -> None:
        result = PluginResultManifest(
            job_id="job-1",
            outcome="succeeded",
            plugin_id="bad",
            plugin_version="1",
            outputs=(
                PluginFrameOutput(
                    output_id="output-1",
                    frame_id="frame-not-requested",
                    relative_path="outputs/bad.cif",
                    sha256=hashlib.sha256(b"bad").hexdigest(),
                    media_type="application/x-cif",
                    role="vector",
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "unknown frame"):
            SubprocessPluginRunner._validate_result_contract(manifest(), result)

    def test_runner_environment_does_not_forward_secrets(self) -> None:
        import os

        previous = os.environ.get("KRAKEN_DATABASE_URL")
        os.environ["KRAKEN_DATABASE_URL"] = "secret"
        try:
            self.assertNotIn("KRAKEN_DATABASE_URL", SubprocessPluginRunner._plugin_environment())
        finally:
            if previous is None:
                os.environ.pop("KRAKEN_DATABASE_URL", None)
            else:
                os.environ["KRAKEN_DATABASE_URL"] = previous

    def test_job_state_is_durable_and_optimistic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "agent.sqlite3"
            store = DurableJobStore(database)
            queued = store.enqueue(manifest())
            staging = store.transition(queued.job_id, AgentJobState.STAGING, expected_revision=0)
            self.assertEqual(AgentJobState.STAGING, DurableJobStore(database).get(queued.job_id).state)
            with self.assertRaises(JobStateError):
                store.transition(queued.job_id, AgentJobState.RUNNING, expected_revision=0)
            store.transition(staging.job_id, AgentJobState.RUNNING, expected_revision=staging.revision)
            self.assertEqual(1, DurableJobStore(database).recover_interrupted())
            self.assertEqual(AgentJobState.RECOVERY_REQUIRED, store.get(queued.job_id).state)

    def test_staging_verifies_hash_and_containment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            source.write_bytes(b"content")
            workspace = StagingWorkspace(root / "staging", "job-1")
            workspace.create()
            staged = workspace.stage_file(
                source,
                "inputs/1_1.png",
                expected_sha256=hashlib.sha256(b"content").hexdigest(),
            )
            self.assertEqual(b"content", staged.read_bytes())
            with self.assertRaises(ValueError):
                workspace.resolve("../outside")

    def test_subprocess_result_waits_for_authorized_project_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = DurableJobStore(root / "jobs.sqlite3")
            job_manifest = manifest()
            workspace = StagingWorkspace(root / "staging", job_manifest.job_id)
            workspace.create()
            source = root / "input.png"
            source.write_bytes(b"content")
            workspace.stage_file(
                source,
                "inputs/1_1.png",
                expected_sha256=hashlib.sha256(b"content").hexdigest(),
            )
            script = (
                "import hashlib,json,os,pathlib;"
                "root=pathlib.Path(os.environ['KRAKEN_STAGING_ROOT']);"
                "out=root/'outputs'/'1_1.cif';out.parent.mkdir(exist_ok=True);out.write_bytes(b'CIF');"
                "result={'schema':'kraken.plugin-result.v1','protocol_version':'1.0','job_id':'job-1',"
                "'outcome':'succeeded','plugin_id':'fake','plugin_version':'1.0','completed_at':'2026-01-01T00:00:00+00:00',"
                "'applied_parameters':{},'errors':[],'outputs':[{'output_id':'output-1','frame_id':'frame-1',"
                "'relative_path':'outputs/1_1.cif','sha256':hashlib.sha256(b'CIF').hexdigest(),"
                "'media_type':'application/x-cif','role':'vector','warnings':[]} ]};"
                "pathlib.Path(os.environ['KRAKEN_RESULT_MANIFEST']).write_text(json.dumps(result),encoding='utf-8')"
            )
            registry = PluginRegistry(
                {
                    "frames.vectorize.v1": PluginProcessSpec(
                        "frames.vectorize.v1",
                        (sys.executable, "-c", script),
                    )
                }
            )
            store.enqueue(job_manifest)
            runner = SubprocessPluginRunner(store, root / "staging", registry)
            self.assertTrue(runner.run_once())
            completed = store.get(job_manifest.job_id)
            self.assertEqual(AgentJobState.IMPORTING, completed.state)
            self.assertIsNotNone(completed.result)


if __name__ == "__main__":
    unittest.main()
