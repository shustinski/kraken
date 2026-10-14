"""services: rarity roller."""
from __future__ import annotations
from random import Random
from contour.gamification.config import GamificationBalance
from contour.gamification.models import (
    RARITY_ORDER,
    Rarity,
)

class _RarityRoller:
    def __init__(self, balance: GamificationBalance) -> None:
        self._balance = balance

    def roll(self, rng: Random) -> Rarity:
        value = float(rng.random())
        total = 0.0
        for rarity in RARITY_ORDER:
            total += float(self._balance.rarity_distribution.get(rarity, 0.0))
            if value <= total:
                return rarity
        return Rarity.COMMON
