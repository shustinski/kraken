import pytest

torch = pytest.importorskip('torch')

from model.NeuralNetwork import create_model
from model.NeuralNetwork.model_io import load_model_artifact, save_model_artifact
from tests.helpers import make_test_dir


def _first_parameter_tensor(model):
    return next(model.parameters()).detach().cpu().clone()


def test_save_and_load_model_artifact_roundtrip():
    model = create_model('S 660k', 1)
    save_path = make_test_dir('model_io_safe') / 'safe_model.pth'

    save_model_artifact(model, save_path, model_name='S 660k', input_channels=1)
    loaded = load_model_artifact(save_path)

    assert loaded is not None
    assert loaded.__class__.__name__ == model.__class__.__name__
    assert getattr(loaded, '_neuralimage_model_name') == 'S 660k'
    assert int(getattr(loaded, '_neuralimage_input_channels')) == 1
    assert torch.equal(_first_parameter_tensor(loaded), _first_parameter_tensor(model))


def test_legacy_pickle_model_load_requires_explicit_opt_in():
    model = create_model('S 660k', 1)
    legacy_path = make_test_dir('model_io_legacy') / 'legacy_model.pth'
    torch.save(model, legacy_path)

    with pytest.raises(RuntimeError):
        load_model_artifact(legacy_path, allow_unsafe_legacy_pickle=False)

    loaded = load_model_artifact(legacy_path, allow_unsafe_legacy_pickle=True)
    assert loaded is not None
