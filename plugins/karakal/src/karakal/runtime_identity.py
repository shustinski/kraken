"""Measure the loaded worker code and numerical libraries in the computing process."""
import hashlib
import importlib
import inspect
import json
import marshal
import platform
import sys
from types import CodeType

import cv2
import numpy as np

from kraken_core.analysis_run_protocol import AnalysisRuntimeIdentity
from .version import __version__


def _normalized(code):
    return code.replace(co_filename="", co_consts=tuple(_normalized(item) if isinstance(item, CodeType) else item for item in code.co_consts))


def _value(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, (tuple, list)):
        return {type(value).__name__: [_value(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        items = [_value(item) for item in value]
        return {type(value).__name__: sorted(items, key=lambda item: json.dumps(item, sort_keys=True))}
    if isinstance(value, dict):
        items = [[_value(key), _value(item)] for key, item in value.items()]
        return {"dict": sorted(items, key=lambda item: json.dumps(item[0], sort_keys=True))}
    if inspect.isfunction(value):
        return {"function": value.__qualname__, "code": marshal.dumps(_normalized(value.__code__)).hex()}
    if inspect.isclass(value):
        return {"class": value.__module__ + "." + value.__qualname__}
    # Dataclass factory sentinels and other unsupported objects are identified
    # by type only: no address-bearing repr or claim of hashing mutable state.
    return {"opaque_type": type(value).__module__ + "." + type(value).__qualname__}


def _function_bytes(function):
    return json.dumps({"code": marshal.dumps(_normalized(function.__code__)).hex(),
                       "defaults": _value(function.__defaults__), "kwdefaults": _value(function.__kwdefaults__)},
                      sort_keys=True, allow_nan=False).encode()


def _module_parts(module):
    parts = []
    for name, value in sorted(vars(module).items()):
        if name.startswith("__"):
            continue
        if inspect.isfunction(value) and value.__module__ == module.__name__:
            parts.append(name.encode() + b"\0" + _function_bytes(value))
        elif inspect.isclass(value) and value.__module__ == module.__name__:
            for key, method in sorted(vars(value).items()):
                if isinstance(method, property):
                    for accessor in ("fget", "fset", "fdel"):
                        function = getattr(method, accessor)
                        if function is not None:
                            parts.append(f"{name}.{key}.{accessor}".encode() + b"\0" + _function_bytes(function))
                    continue
                if isinstance(method, (staticmethod, classmethod)):
                    method = method.__func__
                if inspect.isfunction(method):
                    parts.append(f"{name}.{key}".encode() + b"\0" + _function_bytes(method))
        elif isinstance(value, (str, int, float, bool, tuple, bytes, set, frozenset, list, dict)):
            try:
                encoded = json.dumps(_value(value), sort_keys=True, allow_nan=False).encode()
            except (TypeError, ValueError):
                continue
            parts.append(name.encode() + b"\0" + encoded)
    return parts


def worker_runtime_identity(worker_module=None):
    if worker_module is None:
        main = sys.modules.get("__main__")
        worker_module = (main if getattr(getattr(main, "__spec__", None), "name", None) == "karakal.worker"
                         else importlib.import_module("karakal.worker"))
    modules = [("karakal.worker", worker_module)]
    modules.extend((name, importlib.import_module(name)) for name in (
        "karakal.core.headless", "karakal.runtime_identity", "kraken_core.analysis_protocol",
        "kraken_core.analysis_run_protocol", "kraken_core.analysis_bundle", "kraken_core.analysis_runtime"))
    digest = hashlib.sha256()
    for name, module in modules:
        digest.update(name.encode() + b"\0")
        for part in _module_parts(module):
            digest.update(len(part).to_bytes(8, "big") + part)
    return AnalysisRuntimeIdentity(__version__, "sha256:" + digest.hexdigest(), platform.python_version(),
                                   str(np.__version__), str(cv2.__version__), f"{platform.system()} {platform.release()}")
