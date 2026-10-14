"""settings_panel: settings panel qsettings."""
from __future__ import annotations
import os
from pathlib import Path
from PyQt6.QtCore import QSettings
from neuralimage.view.settings_panel_widgets import (
    SlidingPanel,  # noqa: F401 - compatibility re-export from neuralimage.view
    )

def _settings_panel_qsettings() -> QSettings:
    root = os.getenv('NEURALIMAGE_SETTINGS_DIR')
    if root:
        settings_root = Path(root)
        settings_root.mkdir(parents=True, exist_ok=True)
        return QSettings(
            str(settings_root / 'NeuralImage_SettingsPanel.ini'),
            QSettings.Format.IniFormat,
        )
    return QSettings('NeuralImage', 'SettingsPanel')
