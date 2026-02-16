from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config_model import ConfigModel
    from .neural_worker import NeuralWorker

__all__ = ["ConfigModel", "NeuralWorker"]


def __getattr__(name: str):
    if name == "ConfigModel":
        from .config_model import ConfigModel

        return ConfigModel
    if name == "NeuralWorker":
        from .neural_worker import NeuralWorker

        return NeuralWorker
    raise AttributeError(name)
