"""Shared runtime for Kraken Hub and Kraken plugins."""

from .plugins import PluginMetadata, load_plugin_catalog
from .runtime import current_platform

__all__ = ["PluginMetadata", "current_platform", "load_plugin_catalog"]
