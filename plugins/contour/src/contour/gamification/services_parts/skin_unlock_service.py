"""services: skin unlock service."""
from __future__ import annotations
from contour.gamification.config import DEFAULT_GAMIFICATION_BALANCE, GamificationBalance
from contour.gamification.models import (
    GamificationProfile,
    Rarity,
    RewardEventType,
    ServiceResult,
    next_rarity,
)
from contour.gamification.registry import (
    SKIN_DEFINITIONS,
)

class SkinUnlockService:
    def __init__(self, balance: GamificationBalance = DEFAULT_GAMIFICATION_BALANCE) -> None:
        self._balance = balance

    def can_unlock_skin(self, profile: GamificationProfile, skin_id: str) -> bool:
        definition = SKIN_DEFINITIONS.get(str(skin_id))
        if definition is None or definition.is_default:
            return False
        return bool(
            profile.pet_progress[definition.pet_type].unlocked
            and not profile.skin_progress[skin_id].unlocked
            and profile.skin_fragments.get(skin_id, Rarity.COMMON) >= self._balance.skin_common_fragments_required
        )

    def unlock_skin(self, profile: GamificationProfile, skin_id: str) -> ServiceResult:
        skin_id = str(skin_id)
        definition = SKIN_DEFINITIONS.get(skin_id)
        if definition is None:
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Неизвестный скин")
        if definition.is_default:
            return ServiceResult.fail(RewardEventType.SKIN_UNLOCKED, "Базовый скин открывается вместе с питомцем")
        if not profile.pet_progress[definition.pet_type].unlocked:
            return ServiceResult.fail(RewardEventType.SKIN_UNLOCKED, "Сначала откройте питомца")
        progress = profile.skin_progress[skin_id]
        if progress.unlocked:
            return ServiceResult.fail(RewardEventType.SKIN_UNLOCKED, "Скин уже открыт")
        if not self.can_unlock_skin(profile, skin_id):
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Недостаточно common-фрагментов скина")
        profile.skin_fragments.spend(skin_id, Rarity.COMMON, self._balance.skin_common_fragments_required)
        progress.unlocked = True
        progress.rarity = Rarity.COMMON
        return ServiceResult.ok(
            RewardEventType.SKIN_UNLOCKED,
            f"Скин открыт: {definition.title}",
            skin_id=skin_id,
        )

    def can_upgrade_skin(self, profile: GamificationProfile, skin_id: str, method: str | None = None) -> bool:
        return self._skin_upgrade_method(profile, skin_id, method) is not None

    def upgrade_skin(self, profile: GamificationProfile, skin_id: str, method: str | None = None) -> ServiceResult:
        skin_id = str(skin_id)
        selected_method = self._skin_upgrade_method(profile, skin_id, method)
        if selected_method is None:
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Недостаточно фрагментов для скина")
        progress = profile.skin_progress[skin_id]
        assert progress.rarity is not None
        target_rarity = next_rarity(progress.rarity)
        if target_rarity is None:
            return ServiceResult.fail(RewardEventType.SKIN_UPGRADED, "Скин уже legendary")
        spend_rarity = target_rarity if selected_method == "direct" else progress.rarity
        spend_amount = (
            self._balance.direct_upgrade_fragments_required
            if selected_method == "direct"
            else self._balance.fallback_upgrade_fragments_required
        )
        profile.skin_fragments.spend(skin_id, spend_rarity, spend_amount)
        progress.rarity = target_rarity
        return ServiceResult.ok(
            RewardEventType.SKIN_UPGRADED,
            f"Скин повышен: {SKIN_DEFINITIONS[skin_id].title} -> {target_rarity.value}",
            skin_id=skin_id,
            rarity=target_rarity,
            method=selected_method,
        )

    def _skin_upgrade_method(self, profile: GamificationProfile, skin_id: str, method: str | None) -> str | None:
        skin_id = str(skin_id)
        definition = SKIN_DEFINITIONS.get(skin_id)
        if definition is None or definition.is_default or not profile.pet_progress[definition.pet_type].unlocked:
            return None
        progress = profile.skin_progress.get(skin_id)
        if progress is None or not progress.unlocked or progress.rarity is None:
            return None
        target_rarity = next_rarity(progress.rarity)
        if target_rarity is None:
            return None
        direct_available = (
            profile.skin_fragments.get(skin_id, target_rarity) >= self._balance.direct_upgrade_fragments_required
        )
        fallback_available = (
            profile.skin_fragments.get(skin_id, progress.rarity)
            >= self._balance.fallback_upgrade_fragments_required
        )
        if method == "direct":
            return "direct" if direct_available else None
        if method == "fallback":
            return "fallback" if fallback_available else None
        if direct_available:
            return "direct"
        if fallback_available:
            return "fallback"
        return None
