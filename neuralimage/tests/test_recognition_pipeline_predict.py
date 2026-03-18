import numpy as np
import pytest
from pathlib import Path

torch = pytest.importorskip('torch')

from model.NeuralNetwork.recognition_pipeline import (
    MultiprocessingRecognitionRunner,
    RecognitionWorkload,
    RuntimeWorkerConfig,
    WorkerCounts,
    WorkerGroups,
    gpu_predict,
)
from model.NeuralNetwork import recognition_pipeline as recognition_pipeline_module


class _NaNModel(torch.nn.Module):
    def forward(self, x):
        batch, _, height, width = x.shape
        return torch.full((batch, 1, height, width), float('nan'), device=x.device)


def test_gpu_predict_sanitizes_non_finite_outputs():
    payload = {
        'cutted_image': np.zeros((2, 1, 8, 8), dtype=np.float32),
    }
    model = _NaNModel()

    predicted = gpu_predict(payload, model, torch.device('cpu'), batch_size=1)

    output = predicted['predicted_image']
    assert np.isfinite(output).all()
    assert output.shape == (2, 1, 8, 8)
    assert np.allclose(output, 0.5, atol=1e-6)

    stats = predicted.get('_prediction_stats')
    assert isinstance(stats, dict)
    assert int(stats.get('non_finite', 0)) > 0


def test_shared_memory_payload_roundtrip_for_cut_batches():
    payload = {
        'cutted_image': np.arange(2 * 1 * 4 * 4, dtype=np.float32).reshape(2, 1, 4, 4),
    }

    recognition_pipeline_module._store_payload_array_for_multiprocessing(payload, 'cutted_image')
    restored_payload = recognition_pipeline_module._restore_payload_array_from_multiprocessing(
        payload,
        'cutted_image',
    )
    restored = restored_payload['cutted_image']
    assert isinstance(restored, np.ndarray)
    assert restored.shape == (2, 1, 4, 4)
    assert restored.dtype == np.float32
    assert np.array_equal(restored, np.arange(2 * 1 * 4 * 4, dtype=np.float32).reshape(2, 1, 4, 4))


def test_store_payload_uses_numpy_transport_when_torch_transport_is_disabled(monkeypatch):
    payload = {
        'cutted_image': np.arange(2 * 1 * 4 * 4, dtype=np.float32).reshape(2, 1, 4, 4),
    }

    monkeypatch.setattr(
        recognition_pipeline_module,
        '_should_use_torch_tensor_transport',
        lambda: False,
    )

    recognition_pipeline_module._store_payload_array_for_multiprocessing(payload, 'cutted_image')

    assert isinstance(payload['cutted_image'], np.ndarray)
    assert payload['cutted_image'].flags['C_CONTIGUOUS']


class _StubQueue:
    def put(self, item):
        return

    def get(self, timeout=None):
        raise recognition_pipeline_module.Empty

    def close(self):
        return


class _FailedProcess:
    def __init__(self, pid=123, exitcode=1):
        self.pid = pid
        self.exitcode = exitcode

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return

    def terminate(self):
        return


def test_multiprocessing_runner_raises_on_failed_child(monkeypatch):
    workload = RecognitionWorkload(
        source_files=[Path('frame_001.png')],
        result_folder=Path('.'),
        part_size=(16, 16),
        overlap=2,
        batch_size=1,
        colors=1,
        jpeg_quality=95,
        binarize_output=True,
        threshold=0.5,
        postprocess_enabled=False,
        postprocess_kernel_size=3,
        devices=[torch.device('cpu')],
        model_source='model.pth',
    )
    runner = MultiprocessingRecognitionRunner(
        config=RuntimeWorkerConfig(
            workload=workload,
            worker_counts=WorkerCounts(cut=1, predict=1, sew=1),
            stop_token='__STOP__',
        ),
        stop_event=torch.multiprocessing.Event(),
        publish=lambda *_args, **_kwargs: None,
    )

    class _StubQueues:
        def __init__(self):
            self.cut = _StubQueue()
            self.predict = _StubQueue()
            self.sew = _StubQueue()
            self.sewed = _StubQueue()

        def close(self):
            return

    monkeypatch.setattr(runner, '_create_queues', lambda: _StubQueues())
    monkeypatch.setattr(runner, '_prime_cut_queue', lambda queues: None)
    monkeypatch.setattr(runner, '_publish_runtime_plan', lambda: None)
    monkeypatch.setattr(
        runner,
        '_start_workers',
        lambda queues: WorkerGroups(cut=[_FailedProcess()], predict=[], sew=[]),
    )
    monkeypatch.setattr(runner, '_shutdown_workers', lambda groups: None)

    with pytest.raises(RuntimeError, match='Child process failed'):
        runner.run()
