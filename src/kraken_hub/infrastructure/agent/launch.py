"""How the desktop starts one plugin process."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from kraken_core.plugins import PluginMetadata
from kraken_core.runtime import current_platform, workspace_root


def build_launch_command(
    plugin: PluginMetadata,
    *,
    root: Path | None = None,
    arguments: tuple[str, ...] = (),
) -> list[str]:
    platform = current_platform()
    executable = plugin.executable_for(platform)
    if executable.path:
        return [str(Path(executable.path).expanduser()), *arguments]
    plugin_root = (root or workspace_root()) / "plugins" / plugin.id
    if plugin_root.exists() and shutil.which("uv"):
        if (plugin_root / "__main__.py").is_file():
            return ["uv", "run", "python", "__main__.py", *arguments]
        return ["uv", "run", "python", "-m", plugin.id, *arguments]
    if plugin_root.exists() and (plugin_root / "__main__.py").is_file():
        return [sys.executable, str(plugin_root / "__main__.py"), *arguments]
    if executable.command:
        command = [
            part.format(workspace=str(root or workspace_root()), plugin_id=plugin.id)
            for part in executable.command
        ]
        return [*command, *arguments]
    return [sys.executable, "-m", plugin.id, *arguments]
