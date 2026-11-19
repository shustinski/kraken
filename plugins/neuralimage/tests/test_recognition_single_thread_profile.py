from __future__ import annotations

import torch
from PIL import Image
from multiprocessing import Event

import pytest


def _one_frame(tmp_path) -> str:
    p = tmp_path / "a.png"
    Image.new("RGB", (16, 16), (10, 200, 10)).save(p)
    return str(p)


def _patched(monkeypatch, *, enabled: bool, calls: list):
    from neuralimage.model.NeuralNetwork import recognition_pipeline as rp
    from neuralimage.model.NeuralNetwork.recognition_pipeline_parts import run_single_thread_recognition as rst
    from neuralimage.infrastructure import profiling
    monkeypatch.setattr(profiling, "recognition_profiling_enabled", lambda: enabled)
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    # cut/predict/sew are re-imported from recognition_pipeline inside the function body
    monkeypatch.setattr(rp, "cut_image_prepare", lambda *a, **k: {"cutted_image": "x", "name": "a.png", "source_path": "a.png"})
    monkeypatch.setattr(rp, "gpu_predict", lambda *a, **k: {"predicted_image": "y", "name": "a.png"})
    monkeypatch.setattr(rp, "sew", lambda *a, **k: "out.png")
    # preview/memory helpers are module-level names in run_single_thread_recognition
    monkeypatch.setattr(rst, "_publish_recognition_preview", lambda **k: None)
    monkeypatch.setattr(rst, "_publish_memory_metrics", lambda **k: None)


class _FakeModel:
    def eval(self):
        return self
    def to(self, *a, **k):
        return self


def test_single_thread_writes_frame_report_when_enabled(monkeypatch, tmp_path):
    from neuralimage.model.NeuralNetwork.recognition_pipeline import run_single_thread_recognition
    calls = []
    _patched(monkeypatch, enabled=True, calls=calls)
    run_single_thread_recognition(
        source_files=[tmp_path / "a.png"],
        result_folder=tmp_path,
        part_size=(8, 8),
        overlap=0,
        batch_size=1,
        colors=3,
        model=_FakeModel(),
        device=torch.device("cpu"),
        stop_event=Event(),
        publish=lambda *a, **k: None,
        collect_memory_metrics=lambda: None,
    )
    output = "\n".join(str(m) for c in calls for m in c)
    assert "[neuralimage recognition frame]" in output
    assert "name=a.png" in output


def test_single_thread_no_report_when_disabled(monkeypatch, tmp_path):
    from neuralimage.model.NeuralNetwork.recognition_pipeline import run_single_thread_recognition
    calls = []
    _patched(monkeypatch, enabled=False, calls=calls)
    run_single_thread_recognition(
        source_files=[tmp_path / "a.png"],
        result_folder=tmp_path,
        part_size=(8, 8),
        overlap=0,
        batch_size=1,
        colors=3,
        model=_FakeModel(),
        device=torch.device("cpu"),
        stop_event=Event(),
        publish=lambda *a, **k: None,
        collect_memory_metrics=lambda: None,
    )
    assert calls == []
