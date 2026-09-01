"""Standalone Karakal persistence adapters."""

from .analysis_history import AnalysisHistoryStore, default_history_database

__all__ = ["AnalysisHistoryStore", "default_history_database"]
