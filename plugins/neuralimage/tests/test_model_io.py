import pytest

torch = pytest.importorskip('torch')

from neuralimage.model.NeuralNetwork import create_model
from neuralimage.model.NeuralNetwork.model_io import load_model_artifact, save_model_artifact
from tests.helpers import make_test_dir


def _first_parameter_tensor(model):
    return next(model.parameters()).detach().cpu().clone()


class _UnknownLegacyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


def test_save_and_load_model_artifact_roundtrip():
    model = create_model('S 660k', 1)
    save_path = make_test_dir('model_io_safe') / 'safe_model.pth'
    metadata = {'inference': {'recommended_threshold': 0.62}}

    save_model_artifact(model, save_path, model_name='S 660k', input_channels=1, metadata=metadata)
    loaded = load_model_artifact(save_path)

    assert loaded is not None
    assert loaded.__class__.__name__ == model.__class__.__name__
    assert getattr(loaded, '_neuralimage_model_name') == 'S 660k'
    assert int(getattr(loaded, '_neuralimage_input_channels')) == 1
    assert getattr(loaded, '_neuralimage_artifact_metadata') == metadata
    assert torch.equal(_first_parameter_tensor(loaded), _first_parameter_tensor(model))


def test_save_and_load_model_artifact_roundtrip_with_model_kwargs():
    transformer_kwargs = {
        'img_size': 64,
        'embed_dim': 48,
        'depth': 1,
        'num_heads': 4,
        'mlp_ratio': 2.0,
        'dropout': 0.0,
    }
    model = create_model('Transformer', 1, **transformer_kwargs)
    save_path = make_test_dir('model_io_safe_transformer') / 'safe_model.pth'

    save_model_artifact(
        model,
        save_path,
        model_name='Transformer',
        input_channels=1,
        model_kwargs=transformer_kwargs,
    )
    loaded = load_model_artifact(save_path)

    assert loaded is not None
    assert getattr(loaded, '_neuralimage_model_name') == 'Transformer'
    assert int(getattr(loaded, '_neuralimage_input_channels')) == 1
    assert getattr(loaded, '_neuralimage_model_kwargs') == transformer_kwargs
    assert int(getattr(loaded, 'img_size')) == 64
    assert torch.equal(_first_parameter_tensor(loaded), _first_parameter_tensor(model))


def test_save_and_load_conv_model_artifact_preserves_deep_supervision_flag():
    model = create_model('S 660k', 1, deep_supervision=True)
    save_path = make_test_dir('model_io_safe_deep_supervision') / 'safe_model.pth'

    save_model_artifact(
        model,
        save_path,
        model_name='S 660k',
        input_channels=1,
        model_kwargs={'deep_supervision': True},
    )
    loaded = load_model_artifact(save_path)

    assert loaded is not None
    assert getattr(loaded, '_neuralimage_model_kwargs') == {'deep_supervision': True}
    assert getattr(loaded, 'deep_supervision') is True


def test_safe_artifact_loader_accepts_legacy_conv_checkpoint_without_confidence_head():
    model = create_model('S 660k', 1)
    legacy_state_dict = {
        key: value
        for key, value in model.state_dict().items()
        if not str(key).startswith('confidence_head.')
    }
    save_path = make_test_dir('model_io_safe_legacy_confidence') / 'safe_model.pth'

    torch.save(
        {
            'format': 'neuralimage_model_artifact',
            'version': 2,
            'model_name': 'S 660k',
            'input_channels': 1,
            'model_kwargs': {},
            'metadata': {},
            'state_dict': legacy_state_dict,
        },
        save_path,
    )

    loaded = load_model_artifact(save_path)

    assert loaded is not None
    assert getattr(loaded, '_neuralimage_model_name') == 'S 660k'


def test_legacy_pickle_registered_model_loads_without_unsafe_opt_in():
    model = create_model('S 660k', 1)
    legacy_path = make_test_dir('model_io_legacy_registered') / 'legacy_model.pth'
    torch.save(model, legacy_path)

    loaded = load_model_artifact(legacy_path, allow_unsafe_legacy_pickle=False)
    assert loaded is not None
    assert loaded.__class__.__name__ == model.__class__.__name__
    assert torch.equal(_first_parameter_tensor(loaded), _first_parameter_tensor(model))


def test_legacy_pickle_torch_module_loads_without_unsafe_opt_in():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 2, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(2, 1, kernel_size=1),
    )
    legacy_path = make_test_dir('model_io_legacy_torch_module') / 'legacy_model.pth'
    torch.save(model, legacy_path)

    loaded = load_model_artifact(legacy_path, allow_unsafe_legacy_pickle=False)
    assert loaded is not None
    assert isinstance(loaded, torch.nn.Module)
    assert torch.equal(_first_parameter_tensor(loaded), _first_parameter_tensor(model))


def test_legacy_pickle_unknown_module_requires_explicit_opt_in():
    model = _UnknownLegacyModule()
    legacy_path = make_test_dir('model_io_legacy_unknown') / 'legacy_model.pth'
    torch.save(model, legacy_path)

    with pytest.raises(RuntimeError):
        load_model_artifact(legacy_path, allow_unsafe_legacy_pickle=False)

    loaded = load_model_artifact(legacy_path, allow_unsafe_legacy_pickle=True)
    assert loaded is not None
