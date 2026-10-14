"""services: pet unlock service."""
from __future__ import annotations
from contour.gamification.config import DEFAULT_GAMIFICATION_BALANCE, GamificationBalance
from contour.gamification.models import (
    GamificationProfile,
    PetType,
    Rarity,
    RewardEventType,
    ServiceResult,
    next_rarity,
)
from contour.gamification.registry import (
    DEFAULT_SKIN_BY_PET,
    PET_DEFINITIONS,
)

class PetUnlockService:
    def __init__(self, balance: GamificationBalance = DEFAULT_GAMIFICATION_BALANCE) -> None:
        self._balance = balance

    def can_unlock_pet(self, profile: GamificationProfile, pet_type: PetType) -> bool:
        progress = profile.pet_progress.get(pet_type)
        return bool(
            progress is not None
            and not progress.unlocked
            and profile.pet_fragments.get(pet_type, Rarity.COMMON) >= self._balance.pet_common_fragments_required
        )

    def unlock_pet(self, profile: GamificationProfile, pet_type: PetType) -> ServiceResult:
        if pet_type not in PET_DEFINITIONS:
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Неизвестный питомец")
        if pet_type == PetType.KRAKEN:
            return ServiceResult.fail(RewardEventType.PET_UNLOCKED, "Кракен уже открыт")
        progress = profile.pet_progress[pet_type]
        if progress.unlocked:
            return ServiceResult.fail(RewardEventType.PET_UNLOCKED, "Питомец уже открыт")
        if not self.can_unlock_pet(profile, pet_type):
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Недостаточно common-фрагментов питомца")
        profile.pet_fragments.spend(pet_type, Rarity.COMMON, self._balance.pet_common_fragments_required)
        progress.unlocked = True
        progress.rarity = Rarity.COMMON
        default_skin_id = DEFAULT_SKIN_BY_PET[pet_type]
        default_skin = profile.skin_progress[default_skin_id]
        default_skin.unlocked = True
        default_skin.rarity = Rarity.COMMON
        profile.selected_skin_by_pet[pet_type] = default_skin_id
        return ServiceResult.ok(
            RewardEventType.PET_UNLOCKED,
            f"Питомец открыт: {PET_DEFINITIONS[pet_type].title}",
            pet_type=pet_type,
        )

    def can_upgrade_pet(self, profile: GamificationProfile, pet_type: PetType, method: str | None = None) -> bool:
        return self._pet_upgrade_method(profile, pet_type, method) is not None

    def upgrade_pet(self, profile: GamificationProfile, pet_type: PetType, method: str | None = None) -> ServiceResult:
        selected_method = self._pet_upgrade_method(profile, pet_type, method)
        if selected_method is None:
            return ServiceResult.fail(RewardEventType.NOT_ENOUGH_FRAGMENTS, "Недостаточно фрагментов для повышения")
        progress = profile.pet_progress[pet_type]
        assert progress.rarity is not None
        target_rarity = next_rarity(progress.rarity)
        if target_rarity is None:
            return ServiceResult.fail(RewardEventType.PET_UPGRADED, "Питомец уже legendary")
        spend_rarity = target_rarity if selected_method == "direct" else progress.rarity
        spend_amount = (
            self._balance.direct_upgrade_fragments_required
            if selected_method == "direct"
            else self._balance.fallback_upgrade_fragments_required
        )
        profile.pet_fragments.spend(pet_type, spend_rarity, spend_amount)
        progress.rarity = target_rarity
        return ServiceResult.ok(
            RewardEventType.PET_UPGRADED,
            f"Питомец повышен: {PET_DEFINITIONS[pet_type].title} -> {target_rarity.value}",
            pet_type=pet_type,
            rarity=target_rarity,
            method=selected_method,
        )

    def _pet_upgrade_method(self, profile: GamificationProfile, pet_type: PetType, method: str | None) -> str | None:
        progress = profile.pet_progress.get(pet_type)
        if progress is None or not progress.unlocked or progress.rarity is None:
            return None
        target_rarity = next_rarity(progress.rarity)
        if target_rarity is None:
            return None
        direct_available = (
            profile.pet_fragments.get(pet_type, target_rarity) >= self._balance.direct_upgrade_fragments_required
        )
        fallback_available = (
            profile.pet_fragments.get(pet_type, progress.rarity) >= self._balance.fallback_upgrade_fragments_required
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
