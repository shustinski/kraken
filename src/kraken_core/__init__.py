"""Shared runtime for Kraken Hub and Kraken plugins."""

from .plugins import PluginMetadata, load_plugin_catalog
from .runtime import current_platform
from .external_model import ExternalModelLink, StagedExternalModel

__all__ = [
    "ExternalModelLink",
    "PluginMetadata",
    "StagedExternalModel",
    "current_platform",
    "load_plugin_catalog",
]
