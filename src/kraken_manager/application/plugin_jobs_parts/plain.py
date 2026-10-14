"""plugin_jobs: plain."""
from __future__ import annotations
from collections.abc import Mapping
from kraken_manager.domain.common import PluginJobId

def _job_stream(job_id: PluginJobId | str) -> str:
    return f"plugin-job:{job_id}"

def _series_stream(series_id: object) -> str:
    return f"artifact-series:{series_id}"

def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    return value
