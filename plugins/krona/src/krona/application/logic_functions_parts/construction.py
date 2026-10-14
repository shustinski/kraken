"""logic_functions: construction."""
from __future__ import annotations
from krona.application.ports import NetlistRepository


class ConstructionMixin:
    def __init__(self, repository: NetlistRepository):
        self._repository = repository
