import pytest

from neuralimage.lib.logging_policy import should_forward_log_event

torch = pytest.importorskip("torch")

from neuralimage.model.NeuralNetwork.recognition_pipeline import ProgressReporter


def test_should_forward_log_event_filters_redundant_messages():
    assert not should_forward_log_event("training", "Epoch [1/5] Loss: 0.123")
    assert not should_forward_log_event("logging", "Средняя потеря на обучающей выборке: 0.123")
    assert not should_forward_log_event(
        "logging",
        "Epoch [1/5] Validation loss: 0.123 | Validation accuracy: 91.00% | IoU: 80.00%",
    )
    assert not should_forward_log_event(
        "logging",
        "Frame: 3/10. Per-frame time: 0.25 sec. Elapsed: 0:00:03",
    )
    assert should_forward_log_event("logging", "torch.compile disabled by NEURALIMAGE_TORCH_COMPILE=0.")


def test_progress_reporter_publishes_metrics_without_frame_logs():
    events: list[tuple[str, dict[str, int]]] = []
    reporter = ProgressReporter(lambda topic, payload: events.append((topic, payload)), total_frames=5)

    reporter.publish_started()
    reporter.publish_frame(3)

    assert events == [
        ("metrics", {"type": "recognition_progress", "current": 0, "total": 5}),
        ("metrics", {"type": "recognition_progress", "current": 3, "total": 5}),
    ]
