from __future__ import annotations

from kraken_core.ipc import ActionRegistry


def create_action_registry() -> ActionRegistry:
    registry = ActionRegistry()
    return registry
