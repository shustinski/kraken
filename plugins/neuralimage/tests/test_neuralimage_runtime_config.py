from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clear_config_cache():
    from neuralimage.infrastructure import runtime_config
    runtime_config.clear_runtime_config_cache()
    yield
    runtime_config.clear_runtime_config_cache()


def test_bundled_path_uses_resource_path():
    from neuralimage.infrastructure import runtime_config
    from neuralimage.lib.runtime_paths import resolve_resource_path
    assert runtime_config.bundled_config_path() == resolve_resource_path("neuralimage.ini")


def test_env_override_path(monkeypatch, tmp_path):
    from neuralimage.infrastructure import runtime_config
    custom = tmp_path / "custom.ini"
    custom.write_text("[profiling]\nenabled = true\n", encoding="utf-8")
    monkeypatch.setenv("NEURALIMAGE_CONFIG", str(custom))
    assert runtime_config.runtime_config_path() == custom
    assert runtime_config.config_bool("profiling", "enabled", False) is True


def test_config_bool_default_when_missing():
    from neuralimage.infrastructure import runtime_config
    assert runtime_config.config_bool("profiling", "nope", True) is True


def test_config_int_minimum_clamps(monkeypatch, tmp_path):
    from neuralimage.infrastructure import runtime_config
    custom = tmp_path / "c.ini"
    custom.write_text("[profiling]\ntop_lines = 0\n", encoding="utf-8")
    monkeypatch.setenv("NEURALIMAGE_CONFIG", str(custom))
    assert runtime_config.config_int("profiling", "top_lines", 40, minimum=1) == 1
