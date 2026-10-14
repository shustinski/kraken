"""manager_app: launch managed plugin."""
from __future__ import annotations
from kraken_core.plugins import is_plugin_launchable


class LaunchManagedPluginMixin:
    def _launch_managed_plugin(
        self,
        plugin_id: str,
        *,
        arguments: tuple[str, ...] = (),
    ):
        inventory = self.plugin_items.get(plugin_id)
        if inventory is None or not is_plugin_launchable(inventory) or not inventory.metadata.enabled:
            self._error(f"Плагин {plugin_id} не установлен или отключён")
            return None
        from kraken_hub.app import launch_plugin
        return launch_plugin(inventory.metadata, arguments=arguments)
