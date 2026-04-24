from neuralimage.model.config_model import ConfigModel


def test_config_model_constructs():
    obj = ConfigModel()
    assert isinstance(obj, ConfigModel)

