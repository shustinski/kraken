from __future__ import annotations

import pytest


def test_disabled_returns_nullcontext(monkeypatch):
    from neuralimage.infrastructure import profiling
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts import train_step_profile as tsp
    monkeypatch.setattr(profiling, "training_profiling_enabled", lambda: False)
    calls = []
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    ctx = tsp.train_step_profile_context(epoch=0, batch_index=0, global_batch=0)
    with ctx:
        sum(range(100))
    assert calls == []  # disabled -> no profiler, no report


def test_enabled_writes_train_step_report(monkeypatch):
    from neuralimage.infrastructure import profiling
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts import train_step_profile as tsp
    calls = []
    monkeypatch.setattr(profiling, "training_profiling_enabled", lambda: True)
    monkeypatch.setattr(profiling, "training_profile_stride", lambda: 1)
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    with tsp.train_step_profile_context(epoch=0, batch_index=0, global_batch=0):
        sum(range(1000))
    output = "\n".join(str(m) for c in calls for m in c)
    assert "[neuralimage train step]" in output
    assert "epoch=1 batch=1 global=0" in output


def test_stride_skips_non_matching_global_batch(monkeypatch):
    from neuralimage.infrastructure import profiling
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts import train_step_profile as tsp
    calls = []
    monkeypatch.setattr(profiling, "training_profiling_enabled", lambda: True)
    monkeypatch.setattr(profiling, "training_profile_stride", lambda: 2)
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    with tsp.train_step_profile_context(epoch=0, batch_index=0, global_batch=1):
        pass
    assert calls == []  # 1 % 2 != 0 -> no report


def test_notice_line(monkeypatch, tmp_path):
    from neuralimage.infrastructure import profiling
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts import train_step_profile as tsp
    profiling.set_profile_base_dir(tmp_path)
    monkeypatch.setattr(profiling, "training_profiling_enabled", lambda: False)
    assert tsp.train_profiling_notice_line() is None
    monkeypatch.setattr(profiling, "training_profiling_enabled", lambda: True)
    line = tsp.train_profiling_notice_line()
    assert line is not None
    assert "cProfile profiling enabled" in line
    assert "log=" in line


def test_region_disables_on_exception(monkeypatch):
    from neuralimage.infrastructure import profiling
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts import train_step_profile as tsp
    monkeypatch.setattr(profiling, "training_profiling_enabled", lambda: True)
    monkeypatch.setattr(profiling, "training_profile_stride", lambda: 1)
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: None)
    state = {"enabled": 0, "disabled": 0}

    class FakeProfiler:
        def enable(self):
            state["enabled"] += 1
        def disable(self):
            state["disabled"] += 1

    class FakeCProfile:
        @staticmethod
        def Profile():
            return FakeProfiler()

    monkeypatch.setattr(profiling, "cProfile", FakeCProfile)
    with pytest.raises(RuntimeError):
        with tsp.train_step_profile_context(epoch=0, batch_index=0, global_batch=0):
            raise RuntimeError("boom")
    assert state["enabled"] == 1
    assert state["disabled"] == 1  # disabled in finally even when the body raises
