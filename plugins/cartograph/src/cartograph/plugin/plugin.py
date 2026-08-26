"""Cartograph Kraken plugin.

Cartograph is a standalone Qt plugin (like Karakal/CSliser), not a Kraken Agent
job. Discovery is ``plugins/cartograph/resources/plugin.json`` plus the workspace
member in the root ``pyproject.toml``. Launch: ``python -m cartograph``.

Package layout follows CSliser/KateGB (domain / application / infrastructure /
presentation) rather than the suggested ``kraken_cartograph`` tree: Kraken plugin
ids match the folder and import name.

Public compute API (stable if a native backend is added later):
- ``PairRegistrar.register(fixed, moving, hint) -> RegistrationResult``
- ``RegistrationBackend.register_pairs(...)``
- ``LocalBlockOptimizer.optimize(graph) -> LocalBlockSolution``
- ``RegisterLocalWindow.execute(...)`` / ``RunLocalVerticalSlice.execute(...)``

v1 implements TRANSLATION-only local 3×3 stitching. LOD, interlayer alignment,
and feature fallback are extension points only.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from .api import PluginHost
from ..presentation.qt.window import CartographWindow


class CartographPlugin:
    plugin_id = "cartograph"
    display_name = "Cartograph"

    def __init__(self) -> None:
        self._widget: CartographWindow | None = None
        self._host: PluginHost | None = None

    def create_widget(self, host: PluginHost | None = None, parent: QWidget | None = None) -> CartographWindow:
        _ = parent
        self._host = host
        self._widget = CartographWindow()
        return self._widget

    def shutdown(self) -> None:
        if self._widget is not None:
            self._widget.close()
            self._widget = None
        self._host = None
