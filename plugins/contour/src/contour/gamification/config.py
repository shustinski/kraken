from __future__ import annotations

from dataclasses import dataclass, field

from .models import CorrectionType, Rarity


@dataclass(slots=True)
class GamificationBalance:
    currency_reward: dict[CorrectionType, int] = field(
        default_factory=lambda: {
            CorrectionType.MINOR: 1,
            CorrectionType.MEDIUM: 2,
            CorrectionType.MAJOR: 4,
            CorrectionType.EXPERT: 8,
            CorrectionType.UNKNOWN: 1,
        }
    )
    pet_drop_chance: dict[CorrectionType, float] = field(
        default_factory=lambda: {
            CorrectionType.MINOR: 0.0005,
            CorrectionType.MEDIUM: 0.0010,
            CorrectionType.MAJOR: 0.0025,
            CorrectionType.EXPERT: 0.0050,
            CorrectionType.UNKNOWN: 0.0005,
        }
    )
    skin_drop_chance: dict[CorrectionType, float] = field(
        default_factory=lambda: {
            CorrectionType.MINOR: 0.0002,
            CorrectionType.MEDIUM: 0.0005,
            CorrectionType.MAJOR: 0.0010,
            CorrectionType.EXPERT: 0.0025,
            CorrectionType.UNKNOWN: 0.0002,
        }
    )
    rarity_distribution: dict[Rarity, float] = field(
        default_factory=lambda: {
            Rarity.COMMON: 0.890,
            Rarity.RARE: 0.100,
            Rarity.EPIC: 0.009,
            Rarity.LEGENDARY: 0.001,
        }
    )
    pet_fragment_pity_threshold: int = 300
    skin_fragment_pity_threshold: int = 500
    pet_common_fragments_required: int = 10
    skin_common_fragments_required: int = 10
    direct_upgrade_fragments_required: int = 10
    fallback_upgrade_fragments_required: int = 100
    exchange_source_fragments_required: int = 10
    exchange_target_fragments_received: int = 1
    random_common_pet_fragment_capsule_price: int = 300
    random_common_skin_fragment_capsule_price: int = 500
    xp_per_currency: int = 1
    xp_per_level: int = 100


DEFAULT_GAMIFICATION_BALANCE = GamificationBalance()
