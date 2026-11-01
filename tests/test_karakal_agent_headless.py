"""D4: source metadata must dispatch the actual worker without user input."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from kraken_agent.jobs import AgentJobState, DurableJobStore, StagingWorkspace
from kraken_agent.runner import PluginRegistry, SubprocessPluginRunner
from kraken_core.plugins import PluginInventoryItem, load_plugin_metadata_from_directory
from kraken_hub.infrastructure.agent.runtime import LocalAgentRuntime
from tests.test_agent_execution_safety import manifest


def test_hub_accepts_installed_headless_karakal_action():
    from types import SimpleNamespace
    from kraken_hub.presentation.qt.manager_app import DesktopController

    plugin = load_plugin_metadata_from_directory(Path(__file__).resolve().parents[1] / "plugins/karakal")
    controller = object.__new__(DesktopController)
    controller._project_id = "project"
    controller.service = SimpleNamespace(is_remote_project=lambda _: False)
    controller.plugin_items = {"karakal": PluginInventoryItem(plugin)}
    controller._has_permission = lambda _: True
    assert controller._action_availability("karakal") == (True, "")


def test_confidence_runs_headless_and_result_survives_reopen(tmp_path):
    plugin = load_plugin_metadata_from_directory(Path(__file__).resolve().parents[1] / "plugins/karakal")
    runtime = LocalAgentRuntime(tmp_path / "agent", (PluginInventoryItem(plugin),))
    registry_path, _ = runtime._registry()
    registry = PluginRegistry.from_json(registry_path)
    spec = registry.get("layer.confidence.analyze.v1")
    # Explicit headless admission also catches the old interactive-only metadata.
    assert not spec.for_execution({"execution_mode": "headless"}).interactive
    assert not spec.interactive
    value = replace(manifest(), operation="layer.confidence.analyze.v1")
    workspace = StagingWorkspace(tmp_path / "staging", value.job_id)
    workspace.create()
    from PIL import Image
    source = tmp_path / "frame.png"
    Image.new("L", (8, 8), 128).save(source)
    value = replace(value, inputs=(replace(value.inputs[0], sha256=hashlib.sha256(source.read_bytes()).hexdigest()),))
    database = tmp_path / "jobs.sqlite3"
    store = DurableJobStore(database)
    store.enqueue(value)
    workspace.stage_file(source, "inputs/1.png", expected_sha256=value.inputs[0].sha256)
    runner = SubprocessPluginRunner(store, tmp_path / "staging", registry)
    assert runner.run_once()
    persisted = DurableJobStore(database).get(value.job_id)
    assert persisted.state is AgentJobState.IMPORTING, persisted.error
    assert persisted.result.outcome == "succeeded"
    output = persisted.result.outputs[0]
    report = workspace.path / output.relative_path
    assert hashlib.sha256(report.read_bytes()).hexdigest() == output.sha256
    assert json.loads(report.read_text(encoding="utf-8")) == {
        "operation": "layer.confidence.analyze.v1", "job_id": "job-1", "input_count": 1,
    }
    assert json.loads((workspace.path / "result.json").read_text(encoding="utf-8"))["outcome"] == "succeeded"
