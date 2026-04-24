from __future__ import annotations

from importlib import import_module

from .__version__ import __version__

_LAZY_EXPORTS = {
    "BatchImageResult": (".models", "BatchImageResult"),
    "BatchProcessingOptions": (".models", "BatchProcessingOptions"),
    "ContourExtractionSettings": (".models", "ContourExtractionSettings"),
    "DisplaySettings": (".models", "DisplaySettings"),
    "PipelineStepConfig": (".models", "PipelineStepConfig"),
    "PolygonData": (".models", "PolygonData"),
    "SaveOptions": (".models", "SaveOptions"),
    "PolygonExtractionWidget": (".widget", "PolygonExtractionWidget"),
    "ContourApplicationComponents": (".application", "ContourApplicationComponents"),
    "ContourApplicationModel": (".application", "ContourApplicationModel"),
    "ContourMainView": (".application", "ContourMainView"),
    "ContourPresenter": (".application", "ContourPresenter"),
    "ContourStandaloneWindow": (".application", "ContourStandaloneWindow"),
    "StartupConfiguration": (".application", "StartupConfiguration"),
    "assemble_application": (".application", "assemble_application"),
    "build_application": (".application", "build_application"),
    "main": (".application", "main"),
}

__all__ = ["__version__", *_LAZY_EXPORTS]


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
