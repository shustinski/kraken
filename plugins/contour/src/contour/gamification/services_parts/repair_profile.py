"""services: repair profile."""
from __future__ import annotations
from contour.gamification.models import (
    RARITY_ORDER,
    GamificationProfile,
    PetProgress,
    PetType,
    Rarity,
    SkinProgress,
)
from contour.gamification.registry import (
    DEFAULT_SKIN_BY_PET,
    PET_DEFINITIONS,
    SKIN_DEFINITIONS,
)

from contour.gamification.services_parts.in_memory_gamification_payload import (
    _clamped_int,
    _profile_level_for_xp,
)


class RepairProfileMixin:
    def repair_profile(self, profile: GamificationProfile) -> GamificationProfile:
        profile.xp = _clamped_int(profile.xp)
        profile.level = max(1, _clamped_int(profile.level, 1))
        profile.wallet_balance = _clamped_int(profile.wallet_balance)
        profile.lifetime_currency_earned = _clamped_int(profile.lifetime_currency_earned)
        profile.lifetime_currency_spent = _clamped_int(profile.lifetime_currency_spent)
        profile.pet_pity_counter = _clamped_int(profile.pet_pity_counter)
        profile.skin_pity_counter = _clamped_int(profile.skin_pity_counter)

        for pet_type in PET_DEFINITIONS:
            progress = profile.pet_progress.get(pet_type)
            if progress is None:
                progress = PetProgress(pet_type=pet_type)
                profile.pet_progress[pet_type] = progress
            progress.pet_type = pet_type
            if pet_type == PetType.KRAKEN:
                progress.unlocked = True
                progress.rarity = progress.rarity or Rarity.COMMON
            if not progress.unlocked:
                progress.rarity = None
            elif progress.rarity not in RARITY_ORDER:
                progress.rarity = Rarity.COMMON

        for skin_id, definition in SKIN_DEFINITIONS.items():
            skin_progress = profile.skin_progress.get(skin_id)
            if skin_progress is None:
                skin_progress = SkinProgress(
                    skin_id=skin_id,
                    pet_type=definition.pet_type,
                    is_default=definition.is_default,
                )
                profile.skin_progress[skin_id] = skin_progress
            skin_progress.skin_id = skin_id
            skin_progress.pet_type = definition.pet_type
            skin_progress.is_default = definition.is_default
            if not profile.pet_progress[definition.pet_type].unlocked:
                skin_progress.unlocked = False
                skin_progress.rarity = None
            if definition.is_default and profile.pet_progress[definition.pet_type].unlocked:
                skin_progress.unlocked = True
                skin_progress.rarity = skin_progress.rarity or Rarity.COMMON
            if not skin_progress.unlocked:
                skin_progress.rarity = None
            elif skin_progress.rarity not in RARITY_ORDER:
                skin_progress.rarity = Rarity.COMMON

        self._repair_pet_inventory(profile.pet_fragments.fragments_by_pet)
        self._repair_skin_inventory(profile.skin_fragments.fragments_by_skin_id)

        if profile.selected_pet not in PET_DEFINITIONS or not profile.pet_progress[profile.selected_pet].unlocked:
            profile.selected_pet = PetType.KRAKEN
        for pet_type in PET_DEFINITIONS:
            selected_skin = profile.selected_skin_by_pet.get(pet_type)
            default_skin = DEFAULT_SKIN_BY_PET[pet_type]
            if selected_skin not in SKIN_DEFINITIONS:
                profile.selected_skin_by_pet[pet_type] = default_skin
                continue
            selected_progress = profile.skin_progress.get(selected_skin)
            if (
                selected_progress is None
                or selected_progress.pet_type != pet_type
                or not selected_progress.unlocked
            ):
                profile.selected_skin_by_pet[pet_type] = default_skin
        selected_skin = profile.selected_skin_by_pet.get(profile.selected_pet, DEFAULT_SKIN_BY_PET[profile.selected_pet])
        if not profile.skin_progress[selected_skin].unlocked:
            profile.selected_skin_by_pet[profile.selected_pet] = DEFAULT_SKIN_BY_PET[profile.selected_pet]
        profile.level = max(profile.level, _profile_level_for_xp(profile.xp, self._balance))
        return profile
