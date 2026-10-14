"""services: create default profile."""
from __future__ import annotations
from contour.gamification.config import DEFAULT_GAMIFICATION_BALANCE, GamificationBalance
from contour.gamification.models import (
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

from contour.gamification.services_parts.in_memory_gamification_payload import InMemoryGamificationPayloadStore


class CreateDefaultProfileMixin:
    def __init__(
        self,
        store: GamificationPayloadStore | None = None,
        balance: GamificationBalance = DEFAULT_GAMIFICATION_BALANCE,
    ) -> None:
        self._store = store or InMemoryGamificationPayloadStore()
        self._balance = balance

    def create_default_profile(self) -> GamificationProfile:
        pet_progress = {
            pet_type: PetProgress(
                pet_type=pet_type,
                unlocked=pet_type == PetType.KRAKEN,
                rarity=Rarity.COMMON if pet_type == PetType.KRAKEN else None,
            )
            for pet_type in PET_DEFINITIONS
        }
        skin_progress: dict[str, SkinProgress] = {}
        for skin_id, skin in SKIN_DEFINITIONS.items():
            unlocked = skin_id == DEFAULT_SKIN_BY_PET[PetType.KRAKEN]
            skin_progress[skin_id] = SkinProgress(
                skin_id=skin_id,
                pet_type=skin.pet_type,
                unlocked=unlocked,
                rarity=Rarity.COMMON if unlocked else None,
                is_default=skin.is_default,
            )
        return GamificationProfile(
            xp=0,
            level=1,
            wallet_balance=0,
            lifetime_currency_earned=0,
            lifetime_currency_spent=0,
            selected_pet=PetType.KRAKEN,
            selected_skin_by_pet={pet_type: DEFAULT_SKIN_BY_PET[pet_type] for pet_type in PET_DEFINITIONS},
            pet_progress=pet_progress,
            skin_progress=skin_progress,
            pet_fragments=self._default_pet_inventory(),
            skin_fragments=self._default_skin_inventory(),
            pet_pity_counter=0,
            skin_pity_counter=0,
        )

    def load_profile(self) -> GamificationProfile:
        payload = self._store.load_payload()
        if not payload:
            profile = self.create_default_profile()
            self.save_profile(profile)
            return profile
        profile = self._profile_from_payload(payload)
        return self.repair_profile(profile)

    def save_profile(self, profile: GamificationProfile) -> None:
        self._store.save_payload(self.profile_to_payload(self.repair_profile(profile)))
