from __future__ import annotations

from pathlib import Path

from kraken_core.ipc import ActionRegistry, ActionResponse


def create_action_registry() -> ActionRegistry:
    registry = ActionRegistry()
    registry.register("open_path", _open_path)
    registry.register("open_folder", _open_folder)
    registry.register("run_action", _run_action)
    return registry


def _open_path(payload: dict) -> ActionResponse:
    path = Path(str(payload.get("path", ""))).expanduser()
    if not path.exists():
        return ActionResponse(False, f"Path does not exist: {path}")
    return ActionResponse(True, data={"path": str(path)})


def _open_folder(payload: dict) -> ActionResponse:
    path = Path(str(payload.get("path", ""))).expanduser()
    if not path.is_dir():
        return ActionResponse(False, f"Folder does not exist: {path}")
    return ActionResponse(True, data={"folder": str(path)})


def _run_action(payload: dict) -> ActionResponse:
    return ActionResponse(False, f"Unsupported Krona action: {payload.get('name', '')}")
