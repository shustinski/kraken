"""builders: build tabs."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QTabWidget,
    QWidget,
)

def build_tabs(self) -> QWidget:
    tabs = QTabWidget()
    tabs.setUsesScrollButtons(True)
    self.paths_tab = self._build_paths_tab()
    self.pipeline_tab = self._build_pipeline_tab()
    self.extraction_tab = self._build_extraction_tab()
    self.display_tab = self._build_display_tab()
    tabs.addTab(self.paths_tab, "Paths")
    tabs.addTab(self.pipeline_tab, "Pipeline")
    tabs.addTab(self.extraction_tab, "Extraction")
    tabs.addTab(self.display_tab, "Display")
    return tabs
