import runpy
import sys

import pytest

torch = pytest.importorskip("torch")


def _run_system_module(*, run_name: str | None = None):
    sys.modules.pop("lib.System", None)
    if run_name is None:
        return runpy.run_module("lib.System")
    return runpy.run_module("lib.System", run_name=run_name)


def test_check_gpu_availability_returns_cuda_device_count(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    module = _run_system_module()

    assert module["check_gpu_availability"]() == 3


def test_system_main_prints_cuda_device_details(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "Fake GPU")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx: "FakeProps")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda idx: 1024**3)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda idx: 2 * 1024**3)

    _run_system_module(run_name="__main__")
    output = capsys.readouterr().out

    assert output.splitlines()[0] == "1"
    assert "Device 0:" in output
    assert "Name: Fake GPU" in output
    assert "Properties: FakeProps" in output
    assert "Allocated: 1.00 GB" in output
    assert "Cached: 2.00 GB" in output
