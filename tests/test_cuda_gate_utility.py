"""Pins for ``tests.helpers.cuda_gate`` — the D15.1 fail-safe CUDA probe.

On D15-family machines the in-process CUDA question kills the interpreter
instead of answering (STATUS_ACCESS_VIOLATION from ``torch.cuda.is_available``
under a non-default ``CUDA_VISIBLE_DEVICES``). ``cuda_precondition_holds()``
therefore asks it in a disposable subprocess, and the contract under test is
fail-safe: **any** abnormal probe outcome — non-zero exit, timeout, interpreter
crash, missing interpreter — must return False (→ skip, tier PARTIAL) and must
never propagate to the pytest process. Only a clean zero exit means
"torch answered and no CUDA device is usable", i.e. the bounded-CPU test may
run. The subprocess is patched here so the unit tier never imports torch.
"""
from __future__ import annotations

import subprocess
import sys

from tests.helpers import cuda_gate

kraken_defect = "D15"


class _Result:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.stdout = b""
        self.stderr = b""


def _patch_run(monkeypatch, *, result=None, error=None):
    """Replace ``cuda_gate.subprocess.run`` and capture its call arguments."""
    calls: list[tuple[tuple, dict]] = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        if error is not None:
            raise error
        assert result is not None
        return result

    monkeypatch.setattr(cuda_gate.subprocess, "run", fake_run)
    return calls


def test_clean_zero_exit_means_probe_answered_and_holds(monkeypatch):
    calls = _patch_run(monkeypatch, result=_Result(0))
    assert cuda_gate.cuda_precondition_holds() is True
    (args, kwargs) = calls[0]
    assert args[0][0] == sys.executable, "the probe must run in a disposable interpreter"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "-1", (
        "the probe must pin CUDA_VISIBLE_DEVICES=-1 (the D15.1 bounded-CPU contract)"
    )
    assert kwargs["timeout"] == cuda_gate.CUDA_PROBE_TIMEOUT_SECONDS


def test_nonzero_exit_reports_not_held(monkeypatch):
    # torch answered "CUDA is available" (exit 1): the bounded-CPU test must skip.
    _patch_run(monkeypatch, result=_Result(1))
    assert cuda_gate.cuda_precondition_holds() is False


def test_timeout_reports_not_held(monkeypatch):
    _patch_run(monkeypatch, error=subprocess.TimeoutExpired(cmd=[sys.executable], timeout=1))
    assert cuda_gate.cuda_precondition_holds() is False


def test_missing_interpreter_reports_not_held(monkeypatch):
    # OSError family: the probe process could not even be spawned.
    _patch_run(monkeypatch, error=OSError("[WinError 2] system cannot find the file specified"))
    assert cuda_gate.cuda_precondition_holds() is False


def test_generic_subprocess_error_reports_not_held(monkeypatch):
    # Crash classes surfacing as SubprocessError (e.g. Popen invalid args) —
    # fail-safe: never propagate, always skip.
    _patch_run(monkeypatch, error=subprocess.SubprocessError("probe process failed"))
    assert cuda_gate.cuda_precondition_holds() is False
