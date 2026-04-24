from conftest import safe_import_or_skip

safe_import_or_skip('torch')
safe_import_or_skip('numpy')
safe_import_or_skip('PIL')
safe_import_or_skip('sympy')

from neuralimage.model.NeuralNetwork.model_train_and_recognition import _resolve_torch_compile_mode


def test_resolve_torch_compile_mode_honors_env(monkeypatch):
    monkeypatch.setenv('NEURALIMAGE_TORCH_COMPILE_MODE', 'reduce-overhead')
    mode, reason = _resolve_torch_compile_mode('cuda')
    assert mode == 'reduce-overhead'
    assert reason == 'env'


def test_resolve_torch_compile_mode_cpu_defaults(monkeypatch):
    monkeypatch.delenv('NEURALIMAGE_TORCH_COMPILE_MODE', raising=False)
    mode, reason = _resolve_torch_compile_mode('cpu')
    assert mode == 'default'
    assert reason == 'device'


def test_resolve_torch_compile_mode_small_cuda_uses_reduce_overhead(monkeypatch):
    monkeypatch.delenv('NEURALIMAGE_TORCH_COMPILE_MODE', raising=False)

    class _Props:
        multi_processor_count = 16

    monkeypatch.setattr(
        'model.NeuralNetwork.model_train_and_recognition.torch.cuda.is_available',
        lambda: True,
    )
    monkeypatch.setattr(
        'model.NeuralNetwork.model_train_and_recognition.torch.cuda.current_device',
        lambda: 0,
    )
    monkeypatch.setattr(
        'model.NeuralNetwork.model_train_and_recognition.torch.cuda.get_device_properties',
        lambda _idx: _Props(),
    )

    mode, reason = _resolve_torch_compile_mode('cuda')
    assert mode == 'reduce-overhead'
    assert reason == 'sm=16'
