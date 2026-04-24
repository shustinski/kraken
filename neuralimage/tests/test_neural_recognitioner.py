from pathlib import Path

import pytest
torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from lib.data_interfaces import RecognitionParameters
from model.NeuralNetwork.model_train_and_recognition import ModelRecognizer, NeuralRecognizer
from tests.helpers import make_test_dir


class _StubBus:
    def __init__(self):
        self.messages: list[tuple[str, object]] = []

    def publish(self, topic: str, payload):
        self.messages.append((topic, payload))


def _build_params(base_dir: Path, model) -> RecognitionParameters:
    return RecognitionParameters(
        source_files=[Path("frame_001.jpg"), Path("frame_002.jpg")],
        result_folder=base_dir / "result",
        model=model,
        part_size=(16, 16),
        batch_size=4,
        overlap=2,
    )


def test_run_uses_one_thread_on_small_workload(monkeypatch):
    base_dir = make_test_dir("neural_rec_small")
    bus = _StubBus()
    params = _build_params(base_dir, model="dummy_model_path.pth")
    params.source_files = [Path("frame_001.jpg")]
    recognizer = NeuralRecognizer(params, bus)

    calls = {"one": 0, "multi": 0}
    recognizer.prepare_model = lambda: None
    recognizer.run_one_thread = lambda: calls.__setitem__("one", calls["one"] + 1)
    recognizer.run_multiprocessing = lambda _runtime_plan=None: calls.__setitem__("multi", calls["multi"] + 1)
    monkeypatch.setattr("model.NeuralNetwork.model_train_and_recognition.mp.cpu_count", lambda: 64)
    monkeypatch.setattr("model.NeuralNetwork.model_train_and_recognition.torch.cuda.is_available", lambda: False)

    recognizer.run(multithreading=True)

    assert calls["one"] == 1
    assert calls["multi"] == 0


def test_run_uses_multiprocessing_for_two_or_more_source_images(monkeypatch):
    base_dir = make_test_dir("neural_rec_multi")
    bus = _StubBus()
    recognizer = NeuralRecognizer(_build_params(base_dir, model="dummy_model_path.pth"), bus)

    calls = {"one": 0, "multi": 0}
    recognizer.prepare_model = lambda: None
    recognizer.run_one_thread = lambda: calls.__setitem__("one", calls["one"] + 1)
    recognizer.run_multiprocessing = lambda _runtime_plan=None: calls.__setitem__("multi", calls["multi"] + 1)
    monkeypatch.setattr("model.NeuralNetwork.model_train_and_recognition.mp.cpu_count", lambda: 64)
    monkeypatch.setattr("model.NeuralNetwork.model_train_and_recognition.torch.cuda.is_available", lambda: False)

    recognizer.run(multithreading=True)

    assert calls["one"] == 0
    assert calls["multi"] == 1


def test_runtime_plan_uses_four_cut_and_sew_workers_per_gpu(monkeypatch):
    base_dir = make_test_dir("neural_rec_plan_gpu1")
    bus = _StubBus()
    recognizer = NeuralRecognizer(_build_params(base_dir, model="dummy_model_path.pth"), bus)

    monkeypatch.setattr(
        recognizer,
        "_resolve_devices",
        lambda: ([torch.device("cuda:0")], 1),
    )

    runtime_plan = recognizer._build_runtime_plan(multithreading=True)

    assert runtime_plan.predict_workers == 1
    assert runtime_plan.cut_workers == 4
    assert runtime_plan.sew_workers == 4
    assert runtime_plan.threads == 9


def test_runtime_plan_scales_cut_and_sew_workers_with_gpu_count(monkeypatch):
    base_dir = make_test_dir("neural_rec_plan_gpu2")
    bus = _StubBus()
    params = _build_params(base_dir, model="dummy_model_path.pth")
    params.source_files = [Path("frame_001.jpg"), Path("frame_002.jpg"), Path("frame_003.jpg")]
    recognizer = NeuralRecognizer(params, bus)

    monkeypatch.setattr(
        recognizer,
        "_resolve_devices",
        lambda: ([torch.device("cuda:0"), torch.device("cuda:1")], 2),
    )

    runtime_plan = recognizer._build_runtime_plan(multithreading=True)

    assert runtime_plan.predict_workers == 2
    assert runtime_plan.cut_workers == 8
    assert runtime_plan.sew_workers == 8
    assert runtime_plan.threads == 18


def test_stop_sets_both_stop_events():
    base_dir = make_test_dir("neural_rec_stop")
    bus = _StubBus()
    recognizer = NeuralRecognizer(_build_params(base_dir, model="dummy_model_path.pth"), bus)

    recognizer.stop()

    assert recognizer._thread_stop_event.is_set()
    assert recognizer.stop_event.is_set()


def test_run_falls_back_to_one_thread_for_in_memory_model(monkeypatch):
    base_dir = make_test_dir("neural_rec_mem")
    bus = _StubBus()
    recognizer = NeuralRecognizer(_build_params(base_dir, model=nn.Identity()), bus)

    calls = {"one": 0, "multi": 0}
    recognizer.prepare_model = lambda: None
    recognizer.run_one_thread = lambda: calls.__setitem__("one", calls["one"] + 1)
    recognizer.run_multiprocessing = lambda _runtime_plan=None: calls.__setitem__("multi", calls["multi"] + 1)
    monkeypatch.setattr("model.NeuralNetwork.model_train_and_recognition.mp.cpu_count", lambda: 64)
    monkeypatch.setattr("model.NeuralNetwork.model_train_and_recognition.torch.cuda.is_available", lambda: False)

    recognizer.run(multithreading=True)

    assert calls["one"] == 1
    assert calls["multi"] == 0


def test_run_uses_parameter_flag_to_disable_multiprocessing(monkeypatch):
    base_dir = make_test_dir("neural_rec_flag_off")
    bus = _StubBus()
    params = _build_params(base_dir, model="dummy_model_path.pth")
    params.recognition_multiprocessing_enabled = False
    recognizer = NeuralRecognizer(params, bus)

    calls = {"one": 0, "multi": 0}
    recognizer.prepare_model = lambda: None
    recognizer.run_one_thread = lambda: calls.__setitem__("one", calls["one"] + 1)
    recognizer.run_multiprocessing = lambda runtime_plan=None: calls.__setitem__("multi", calls["multi"] + 1)
    monkeypatch.setattr("model.NeuralNetwork.model_train_and_recognition.mp.cpu_count", lambda: 64)
    monkeypatch.setattr("model.NeuralNetwork.model_train_and_recognition.torch.cuda.is_available", lambda: False)

    recognizer.run()

    assert calls["one"] == 1
    assert calls["multi"] == 0


def test_recognizer_indexes_source_files_inside_worker(tmp_path):
    source_dir = tmp_path / "recognition_source"
    source_dir.mkdir()
    (source_dir / "frame_001.png").write_bytes(b"fake")
    (source_dir / "frame_002.bmp").write_bytes(b"fake")

    bus = _StubBus()
    params = RecognitionParameters(
        source_files=[],
        source_folder=source_dir,
        result_folder=tmp_path / "result",
        model="dummy_model_path.pth",
        part_size=(16, 16),
        batch_size=4,
        overlap=2,
    )
    recognizer = NeuralRecognizer(params, bus)

    recognizer._ensure_source_files_indexed()

    assert sorted(params.source_files) == sorted([source_dir / "frame_001.png", source_dir / "frame_002.bmp"])
    assert any(
        "отдельном потоке" in str(payload)
        for topic, payload in bus.messages
        if topic == "logging"
    )


def test_recognizer_indexes_source_files_recursively_when_enabled(tmp_path):
    source_dir = tmp_path / "recognition_source_recursive"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True)
    (source_dir / "frame_root.png").write_bytes(b"fake")
    (nested_dir / "frame_nested.bmp").write_bytes(b"fake")

    bus = _StubBus()
    params = RecognitionParameters(
        source_files=[],
        source_folder=source_dir,
        result_folder=tmp_path / "result",
        model="dummy_model_path.pth",
        part_size=(16, 16),
        batch_size=4,
        overlap=2,
        recursive_file_search=True,
    )
    recognizer = NeuralRecognizer(params, bus)

    recognizer._ensure_source_files_indexed()

    assert sorted(p.relative_to(source_dir).as_posix() for p in params.source_files) == [
        "frame_root.png",
        "nested/frame_nested.bmp",
    ]


def test_prepare_model_resolves_recommended_threshold_from_artifact_metadata(monkeypatch):
    base_dir = make_test_dir("neural_rec_threshold_auto")
    bus = _StubBus()
    model = nn.Conv2d(1, 1, kernel_size=1)
    setattr(model, "_neuralimage_artifact_metadata", {"inference": {"recommended_threshold": 0.73}})
    recognizer = NeuralRecognizer(_build_params(base_dir, model=model), bus)

    captured: dict[str, object] = {}

    monkeypatch.setenv("NEURALIMAGE_TORCH_COMPILE", "0")
    monkeypatch.setattr(
        "model.NeuralNetwork.model_train_and_recognition.run_single_thread_recognition",
        lambda **kwargs: captured.update(kwargs),
    )

    recognizer.prepare_model()
    recognizer.run_one_thread()

    assert recognizer._resolved_output_threshold == pytest.approx(0.73)
    assert captured["threshold"] == pytest.approx(0.73)
    assert captured["binarize_output"] is True
    assert any("recommended model threshold" in str(payload) for topic, payload in bus.messages if topic == "logging")


def test_prepare_model_uses_manual_threshold_and_postprocess_settings(monkeypatch):
    base_dir = make_test_dir("neural_rec_threshold_manual")
    bus = _StubBus()
    model = nn.Conv2d(1, 1, kernel_size=1)
    params = _build_params(base_dir, model=model)
    params.use_auto_threshold = False
    params.threshold = 0.64
    params.postprocess_enabled = True
    params.postprocess_kernel_size = 5
    params.recognition_tta_enabled = True
    params.confidence_tta_enabled = True
    params.confidence_save_mode = 'separate_grayscale'
    recognizer = NeuralRecognizer(params, bus)

    captured: dict[str, object] = {}

    monkeypatch.setenv("NEURALIMAGE_TORCH_COMPILE", "0")
    monkeypatch.setattr(
        "model.NeuralNetwork.model_train_and_recognition.run_single_thread_recognition",
        lambda **kwargs: captured.update(kwargs),
    )

    recognizer.prepare_model()
    recognizer.run_one_thread()

    assert recognizer._resolved_output_threshold == pytest.approx(0.64)
    assert captured["threshold"] == pytest.approx(0.64)
    assert captured["recognition_tta_enabled"] is True
    assert captured["confidence_tta_enabled"] is True
    assert captured["postprocess_enabled"] is True
    assert captured["postprocess_kernel_size"] == 5
    assert captured["confidence_save_mode"] == 'separate_grayscale'


def test_prepare_model_disables_threshold_when_binarization_is_off(monkeypatch):
    base_dir = make_test_dir("neural_rec_threshold_disabled")
    bus = _StubBus()
    model = nn.Conv2d(1, 1, kernel_size=1)
    params = _build_params(base_dir, model=model)
    params.binarize_output = False
    params.use_auto_threshold = True
    params.threshold = 0.81
    recognizer = NeuralRecognizer(params, bus)

    captured: dict[str, object] = {}

    monkeypatch.setenv("NEURALIMAGE_TORCH_COMPILE", "0")
    monkeypatch.setattr(
        "model.NeuralNetwork.model_train_and_recognition.run_single_thread_recognition",
        lambda **kwargs: captured.update(kwargs),
    )

    recognizer.prepare_model()
    recognizer.run_one_thread()

    assert recognizer._resolved_output_threshold is None
    assert captured["binarize_output"] is False
    assert captured["threshold"] is None
    assert any("saving probability maps" in str(payload) for topic, payload in bus.messages if topic == "logging")


def test_run_multiprocessing_propagates_disabled_binarization_as_none_threshold(monkeypatch):
    base_dir = make_test_dir("neural_rec_threshold_disabled_mp")
    bus = _StubBus()
    params = _build_params(base_dir, model="dummy_model_path.pth")
    params.binarize_output = False
    params.recognition_tta_enabled = True
    params.confidence_tta_enabled = False
    params.confidence_save_mode = 'separate_grayscale'
    recognizer = NeuralRecognizer(params, bus)
    recognizer.devices_list = [torch.device("cpu")]
    recognizer._resolved_output_threshold = None

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "model.NeuralNetwork.model_train_and_recognition.run_multiprocessing_recognition",
        lambda **kwargs: captured.update(kwargs),
    )

    recognizer.run_multiprocessing()

    workload = captured["workload"]
    assert workload.binarize_output is False
    assert workload.threshold is None
    assert workload.recognition_tta_enabled is True
    assert workload.confidence_tta_enabled is False
    assert workload.confidence_save_mode == 'separate_grayscale'


def test_run_one_thread_passes_source_root_only_for_recursive_search(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True)
    source_file = nested_dir / "frame.png"
    source_file.write_bytes(b"fake")

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "model.NeuralNetwork.model_train_and_recognition.run_single_thread_recognition",
        lambda **kwargs: captured.update(kwargs),
    )

    bus = _StubBus()
    params = RecognitionParameters(
        source_files=[source_file],
        source_folder=source_dir,
        result_folder=tmp_path / "result",
        model=nn.Identity(),
        part_size=(16, 16),
        batch_size=1,
        overlap=0,
        recursive_file_search=True,
    )
    recognizer = NeuralRecognizer(params, bus)
    recognizer.model = nn.Identity()
    recognizer.colors = 1
    recognizer.devices_list = [torch.device("cpu")]
    recognizer._resolved_output_threshold = 0.5

    recognizer.run_one_thread()

    assert captured["source_root"] == source_dir

    captured.clear()
    params.recursive_file_search = False

    recognizer.run_one_thread()

    assert captured["source_root"] is None


def test_run_multiprocessing_passes_source_root_for_recursive_search(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True)
    source_file = nested_dir / "frame.png"
    source_file.write_bytes(b"fake")

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "model.NeuralNetwork.model_train_and_recognition.run_multiprocessing_recognition",
        lambda **kwargs: captured.update(kwargs),
    )

    bus = _StubBus()
    params = RecognitionParameters(
        source_files=[source_file],
        source_folder=source_dir,
        result_folder=tmp_path / "result",
        model="dummy_model_path.pth",
        part_size=(16, 16),
        batch_size=1,
        overlap=0,
        recursive_file_search=True,
    )
    recognizer = NeuralRecognizer(params, bus)
    recognizer.colors = 1
    recognizer.devices_list = [torch.device("cpu")]
    recognizer._resolved_output_threshold = 0.5

    recognizer.run_multiprocessing()

    assert captured["workload"].source_root == source_dir


def test_model_recognizer_does_not_disable_process_mode_under_debugger(monkeypatch):
    base_dir = make_test_dir("neural_rec_debugger")
    bus = _StubBus()
    params = _build_params(base_dir, model="dummy_model_path.pth")
    started = {"value": False}

    class _FakeProcess:
        exitcode = 0

        def __init__(self, recognition_parameters, message_queue, stop_event, multithreading=False):
            self._alive = False

        def start(self):
            started["value"] = True

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return

    monkeypatch.setattr("model.NeuralNetwork.model_train_and_recognition._is_debugger_attached", lambda: True)
    monkeypatch.setattr("model.NeuralNetwork.model_train_and_recognition.RecognizerProcess", _FakeProcess)

    recognizer = ModelRecognizer(params, bus, multithreading=True)
    recognizer.run()

    assert started["value"] is True
    assert recognizer.succeeded is True
