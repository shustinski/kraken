from types import SimpleNamespace
import subprocess

import pytest

from kraken_agent.telemetry import sample_host
from kraken_server.agent_resources import validate_telemetry


def psutil():
    def percent(*, interval):
        assert interval > 0
        return 23.5
    return SimpleNamespace(cpu_percent=percent, virtual_memory=lambda: SimpleNamespace(total=1024, available=512))


def test_measurements_parse_actual_units_and_unsupported_sensor_unknowns():
    def run(command, **kwargs):
        assert kwargs["timeout"] <= 5
        assert command[0] == "nvidia-smi"
        assert kwargs["capture_output"] is True
        return SimpleNamespace(stdout="GPU-1, Example GPU, 45, 512, 8192, [N/A]\n")
    sample = sample_host(psutil_module=psutil(), run=run)
    assert sample["cpu_percent"] == 23.5
    assert sample["memory_available_bytes"] == 512
    assert sample["gpus"][0]["memory_used_mib"] == 512
    assert sample["gpus"][0]["power_watts"] is None
    assert validate_telemetry(sample) == sample


def test_failed_samplers_are_unknown_instead_of_zero():
    def unavailable(**kwargs):
        raise PermissionError("denied")
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    sample = sample_host(psutil_module=SimpleNamespace(cpu_percent=unavailable), run=timeout)
    assert sample["cpu_percent"] is None
    assert sample["memory_total_bytes"] is None
    assert sample["gpus"] is None
    assert sample["errors"] == ["cpu_memory:PermissionError", "gpu:TimeoutExpired"]
    assert validate_telemetry(sample)["gpus"] is None


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), -1, True, 101])
def test_untrusted_agent_metrics_reject_invalid_percentages(metric):
    sample = sample_host(psutil_module=psutil(), run=lambda *a, **k: SimpleNamespace(stdout="GPU, Model, 1, 1, 2, 1"))
    sample["cpu_percent"] = metric
    with pytest.raises(ValueError):
        validate_telemetry(sample)
