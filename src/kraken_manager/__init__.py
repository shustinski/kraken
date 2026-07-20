"""Kraken project-management application.

The package follows clean-architecture dependency rules.  Business objects
live in :mod:`kraken_manager.domain`, orchestration in
:mod:`kraken_manager.application`, and all framework-specific code in the
outer infrastructure and presentation packages.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
