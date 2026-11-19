from __future__ import annotations

import queue as queue_module
from pathlib import Path

import pytest


def _run_imgsew(monkeypatch, tmp_path, *, enabled: bool):
    # imgsew re-imports `sew` from recognition_pipeline INSIDE the function body,
    # so patch it there (the re-import source), not on the worker module.
    import multiprocessing
    import neuralimage.model.NeuralNetwork.recognition_pipeline as rp
    import neuralimage.model.NeuralNetwork.recognition_pipeline_parts.imgsew as mod
    from neuralimage.infrastructure import profiling
    calls = []
    monkeypatch.setattr(profiling, "recognition_profiling_enabled", lambda: enabled)
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    monkeypatch.setattr(rp, "sew", lambda *a, **k: str(tmp_path / "o.png"))
    ev = multiprocessing.Event()  # never set
    q = queue_module.Queue()
    q.put({"name": "a.png", "predicted_image": "y", "confidence_image": "c"})
    q.put("__STOP__")
    mod.imgsew(tmp_path, q, queue_module.Queue(), 95, ev, stop_token="__STOP__")
    return calls


def test_imgsew_writes_sew_report_when_enabled(monkeypatch, tmp_path):
    calls = _run_imgsew(monkeypatch, tmp_path, enabled=True)
    output = "\n".join(str(m) for c in calls for m in c)
    assert "[neuralimage recognition sew]" in output
    assert "name=a.png" in output


def test_imgsew_no_report_when_disabled(monkeypatch, tmp_path):
    calls = _run_imgsew(monkeypatch, tmp_path, enabled=False)
    assert calls == []


def _run_imgpredict(monkeypatch, tmp_path, *, enabled: bool):
    # imgpredict re-imports `gpu_predict` from recognition_pipeline INSIDE the
    # function body, so patch it there; `load_model_artifact` is a module-level
    # import, so patch it on the imgpredict module.
    import multiprocessing
    import torch
    import neuralimage.model.NeuralNetwork.recognition_pipeline as rp
    import neuralimage.model.NeuralNetwork.recognition_pipeline_parts.imgpredict as mod
    from neuralimage.infrastructure import profiling

    class _FakeModel:
        def eval(self):
            return self

        def to(self, *a, **k):
            return self

    calls = []
    monkeypatch.setattr(profiling, "recognition_profiling_enabled", lambda: enabled)
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    monkeypatch.setattr(rp, "gpu_predict", lambda *a, **k: {"predicted_image": "y", "name": "a.png"})
    monkeypatch.setattr(mod, "load_model_artifact", lambda *a, **k: _FakeModel())
    ev = multiprocessing.Event()  # never set
    q_in = queue_module.Queue()
    q_in.put({"name": "a.png", "cutted_image": "x"})
    q_in.put("__STOP__")
    mod.imgpredict(q_in, queue_module.Queue(), "m", torch.device("cpu"), 1, ev, stop_token="__STOP__")
    return calls


def test_imgpredict_writes_predict_report_when_enabled(monkeypatch, tmp_path):
    calls = _run_imgpredict(monkeypatch, tmp_path, enabled=True)
    output = "\n".join(str(m) for c in calls for m in c)
    assert "[neuralimage recognition predict]" in output
    assert "name=a.png" in output


def test_imgpredict_no_report_when_disabled(monkeypatch, tmp_path):
    calls = _run_imgpredict(monkeypatch, tmp_path, enabled=False)
    assert calls == []


def _run_cut(monkeypatch, tmp_path, *, enabled: bool):
    # cut_image_process re-imports `cut_image_prepare` from recognition_pipeline
    # INSIDE the function body, so patch it there. The cut queue carries image paths.
    import multiprocessing
    import neuralimage.model.NeuralNetwork.recognition_pipeline as rp
    import neuralimage.model.NeuralNetwork.recognition_pipeline_parts.cut_image_process as mod
    from neuralimage.infrastructure import profiling
    calls = []
    monkeypatch.setattr(profiling, "recognition_profiling_enabled", lambda: enabled)
    monkeypatch.setattr(profiling, "write_profile_report", lambda *m: calls.append(m))
    monkeypatch.setattr(rp, "cut_image_prepare", lambda *a, **k: {"name": "a.png", "cutted_image": "x"})
    ev = multiprocessing.Event()  # never set
    q_in = queue_module.Queue()
    q_in.put(tmp_path / "a.png")
    q_in.put("__STOP__")
    mod.cut_image_process(q_in, queue_module.Queue(), (3, 8, 8), 0, ev, stop_token="__STOP__")
    return calls


def test_cut_writes_cut_report_when_enabled(monkeypatch, tmp_path):
    calls = _run_cut(monkeypatch, tmp_path, enabled=True)
    output = "\n".join(str(m) for c in calls for m in c)
    assert "[neuralimage recognition cut]" in output
    assert "name=a.png" in output


def test_cut_no_report_when_disabled(monkeypatch, tmp_path):
    calls = _run_cut(monkeypatch, tmp_path, enabled=False)
    assert calls == []


def test_run_passes_result_folder_to_workers(monkeypatch, tmp_path):
    import multiprocessing as mp
    from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.run import RunMixin
    from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.recognition_workload import (
        RecognitionWorkload, WorkerCounts, RuntimeWorkerConfig,
    )
    import torch
    captured = {}

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            captured["args"] = kwargs.get("args")
        def start(self):
            return None
    monkeypatch.setattr(mp, "Process", FakeProcess)

    workload = RecognitionWorkload(
        source_files=[], result_folder=tmp_path, part_size=(8, 8), overlap=0,
        batch_size=1, colors=3, jpeg_quality=95, binarize_output=True, threshold=None,
        postprocess_enabled=False, postprocess_kernel_size=0, recognition_tta_enabled=False,
        confidence_tta_enabled=False, confidence_save_mode="off", devices=[], model_source="m",
    )
    runner = RunMixin(
        config=RuntimeWorkerConfig(workload=workload, worker_counts=WorkerCounts(cut=1, predict=1, sew=1), stop_token="__STOP__"),
        stop_event=mp.Event(), publish=lambda *a, **k: None,
    )
    from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.recognition_workload import PipelineQueues
    def run_start(which):
        captured.clear()
        q = PipelineQueues(cut=mp.Queue(), predict=mp.Queue(), sew=mp.Queue(), sewed=mp.Queue(), errors=mp.Queue())
        getattr(runner, which)(q)
        return captured["args"]
    for which in ("_start_cut_workers", "_start_predict_workers", "_start_sew_workers"):
        args = run_start(which)
        assert tmp_path in list(args), f"{which} must receive result_folder"
