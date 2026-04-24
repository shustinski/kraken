from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS = {
    "ContourApplicationComponents": (".bootstrap", "ContourApplicationComponents"),
    "ContourApplicationModel": (".model", "ContourApplicationModel"),
    "ContourMainView": (".view", "ContourMainView"),
    "ContourPresenter": (".presenter", "ContourPresenter"),
    "ContourStandaloneWindow": (".view", "ContourStandaloneWindow"),
    "StartupConfiguration": (".model", "StartupConfiguration"),
    "assemble_application": (".bootstrap", "assemble_application"),
    "build_application": (".bootstrap", "build_application"),
    "main": (".cli", "main"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
