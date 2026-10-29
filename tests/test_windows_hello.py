from __future__ import annotations

import ctypes

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication

from kraken_hub.presentation.qt import windows_hello


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("result", range(7))
def test_only_verified_hello_result_releases_autofill(qapp, monkeypatch, result):
    calls = []

    def method(_pointer, slot, *_types):
        def invoke(_this, output=None):
            calls.append(slot)
            if slot == 7:
                ctypes.cast(output, ctypes.POINTER(ctypes.c_int32))[0] = 1
            elif slot == 8:
                ctypes.cast(output, ctypes.POINTER(ctypes.c_int32))[0] = result
            return 0
        return invoke

    monkeypatch.setattr(windows_hello, "_call", method)
    assert windows_hello._wait_for_verification(object(), object()) is (result == 0)
    assert calls == [7, 8]


@pytest.mark.parametrize("status", [2, 3])
def test_canceled_or_failed_operation_never_reads_result(qapp, monkeypatch, status):
    def method(_pointer, slot, *_types):
        assert slot == 7

        def invoke(_this, output):
            ctypes.cast(output, ctypes.POINTER(ctypes.c_int32))[0] = status
            return 0
        return invoke

    monkeypatch.setattr(windows_hello, "_call", method)
    assert windows_hello._wait_for_verification(object(), object()) is False


def test_timed_out_verification_is_canceled(qapp, monkeypatch):
    canceled = []

    def method(_pointer, slot, *_types):
        if slot == 7:
            return lambda *_args: 0  # Still pending.
        assert slot == 9
        return lambda *_args: canceled.append(True) or 0

    monkeypatch.setattr(windows_hello, "_call", method)
    assert windows_hello._wait_for_verification(object(), object(), timeout_ms=80) is False
    assert canceled == [True]
