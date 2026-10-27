"""Actual subprocess mode dispatch; this probe does not run an image algorithm."""

from dataclasses import replace
import hashlib
import sys
import threading
import time

import pytest

from kraken_agent.jobs import DurableJobStore, StagingWorkspace, AgentJobState
from kraken_agent.runner import PluginProcessSpec, PluginRegistry, SubprocessPluginRunner
from kraken_core.plugin_protocol import PluginCapability
from tests.test_agent_execution_safety import manifest


PROBE = r"""
import os,json,time,hashlib
from pathlib import Path
from kraken_core.plugin_protocol import PluginFrameOutput,PluginResultManifest
root=Path(os.environ['KRAKEN_STAGING_ROOT'])
job=json.loads(Path(os.environ['KRAKEN_JOB_MANIFEST']).read_text())
(root/'probe-ready').write_text('ready')
while not (root/'probe-release').exists():time.sleep(.01)
payload=b'actual dispatch probe'
(root/'outputs/probe.bin').write_bytes(payload)
result=PluginResultManifest(job_id=job['job_id'],outcome='succeeded',plugin_id='dispatch-probe',plugin_version='1',outputs=(
PluginFrameOutput(output_id='probe',frame_id='frame-1',relative_path='outputs/probe.bin',sha256=hashlib.sha256(payload).hexdigest(),
media_type='application/octet-stream',role='diagnostic'),))
Path(os.environ['KRAKEN_RESULT_MANIFEST']).write_text(result.to_json())
"""


def test_capability_default_mode_round_trip_and_validation():
    value = PluginCapability("dataset.model.train.v1", ("interactive", "headless"), default_mode="interactive")
    assert PluginCapability.from_dict(value.to_dict()) == value
    with pytest.raises(ValueError, match="advertised"):
        PluginCapability("x", ("headless",), default_mode="interactive")
    legacy = PluginCapability("old", ("headless",))
    assert "default_mode" not in legacy.to_dict()


@pytest.mark.parametrize(
    "mode,interactive",
    [(None, True), ("interactive", True), ("headless", False)],
    ids=["legacy-interactive-default", "explicit-interactive", "explicit-headless"],
)
def test_runner_honors_dual_mode_for_actual_subprocess(tmp_path, mode, interactive):
    value = replace(manifest(), operation="mode-probe.v1", parameters={} if mode is None else {"execution_mode": mode})
    store = DurableJobStore(tmp_path / "jobs.sqlite3")
    store.enqueue(value)
    workspace = StagingWorkspace(tmp_path / "staging", value.job_id)
    workspace.create()
    (workspace.path / "inputs/1.png").write_bytes(b"data")
    script = tmp_path / "probe.py"
    script.write_text(PROBE)
    spec = PluginProcessSpec(
        value.operation, (sys.executable, str(script)), interactive=True, modes=frozenset({"interactive", "headless"})
    )
    runner = SubprocessPluginRunner(store, tmp_path / "staging", PluginRegistry({value.operation: spec}))
    thread = threading.Thread(target=runner.run_once)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not (workspace.path / "probe-ready").exists() and time.monotonic() < deadline and thread.is_alive():
            time.sleep(0.01)
        assert (workspace.path / "probe-ready").exists(), store.get(value.job_id).error
        assert store.get(value.job_id).state is (
            AgentJobState.WAITING_FOR_USER if interactive else AgentJobState.RUNNING
        )
    finally:
        (workspace.path / "probe-release").write_text("release")
        thread.join(10)
        if thread.is_alive():
            runner.stop()
            thread.join(5)
    assert not thread.is_alive()
    result = store.get(value.job_id)
    assert result.state is AgentJobState.IMPORTING, result.error
    assert result.result.outputs[0].sha256 == hashlib.sha256(b"actual dispatch probe").hexdigest()


def test_headless_only_registration_cannot_be_switched_to_interactive():
    spec = PluginProcessSpec("x", ("executable",))
    with pytest.raises(ValueError, match="not supported"):
        spec.for_execution({"execution_mode": "interactive"})
