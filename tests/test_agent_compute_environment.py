"""Actual worker subprocess retains explicit compute limits but excludes secrets."""
import json

from tests.test_agent_execution_safety import staged_runner
from kraken_agent.jobs import AgentJobState


def test_actual_worker_preserves_empty_cuda_visibility_and_cpu_thread_limits(tmp_path, monkeypatch):
    expected = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "3"}
    for key, value in expected.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("KRAKEN_DATABASE_URL", "private database must not leak")
    monkeypatch.setenv("UNRELATED_PRIVATE_TOKEN", "private token must not leak")
    script = """
import hashlib, json, os
from pathlib import Path
from kraken_core.plugin_protocol import PluginResultManifest, PluginFrameOutput
root = Path(os.environ['KRAKEN_STAGING_ROOT'])
keys = ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'KRAKEN_DATABASE_URL', 'UNRELATED_PRIVATE_TOKEN']
(root / 'observed.json').write_text(json.dumps({k: os.environ.get(k) for k in keys}))
payload = b'DS 1 1 1;\\nDF;\\nE\\n'
(root / 'outputs/probe.cif').write_bytes(payload)
result = PluginResultManifest(job_id='job-1', outcome='succeeded', plugin_id='environment-probe', plugin_version='1', outputs=(
PluginFrameOutput(output_id='probe', frame_id='frame-1', relative_path='outputs/probe.cif', sha256=hashlib.sha256(payload).hexdigest(), media_type='application/x-cif', role='vector'),))
Path(os.environ['KRAKEN_RESULT_MANIFEST']).write_text(result.to_json())
"""
    store, runner, workspace = staged_runner(tmp_path, script)
    assert runner.run_once()
    assert store.get("job-1").state is AgentJobState.IMPORTING, store.get("job-1").error
    observed = json.loads((workspace.path / "observed.json").read_text())
    assert observed == expected | {"KRAKEN_DATABASE_URL": None, "UNRELATED_PRIVATE_TOKEN": None}
